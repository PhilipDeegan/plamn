#ifndef PLAMN_SOLVER_HYBRID_ADVANCE_HPP
#define PLAMN_SOLVER_HYBRID_ADVANCE_HPP

// a level of the hybrid model over a step (PHARE's SolverPPC): predictor 1, average, the ions
//  moved (domain only), predictor 2, average, the ions moved (all), corrector. And the steps the
//  simulator also uses to initialize and regrid a level: the moments, the ion border sums, Ampere
//  and Ohm
//
// level: the simulator's services for one level (Simulator::LevelServices)
//   number()                             the level number
//   data()                               its patch data (LevelData of HybridPatch)
//   per_patch(fn)                        fn(patch) for every local patch
//   copy(how, key)                       damn's copy of a resource key between its patches
//   fill_magnetic_ghosts(key, t),        patch ghosts, then level ghosts from the coarser level
//   fill_electric_ghosts(key, t)          at t
//   alpha(t)                             t within the coarser level's step, in [0, 1]

#include "damn/amr/copy.hpp"
#include "damn/core/logger.hpp"

#include "plamn/core/data/field/field_algorithm.hpp"
#include "plamn/core/numerics/ampere/ampere_transformer.hpp"
#include "plamn/core/numerics/faraday/faraday_transformer.hpp"
#include "plamn/core/numerics/interpolator/interpolating.hpp"
#include "plamn/core/numerics/ohm/ohm_transformer.hpp"
#include "plamn/core/numerics/particles_synchronised_mover/particles_synchronised_mover.hpp"
#include "plamn/core/numerics/particles_synchronised_mover/particles_synchronised_mover_def.hpp"
#include "plamn/simulator/level_accessor.hpp"
#include "plamn/solver/hybrid/model.hpp"

#include "phlop/timing/scope.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace plamn::solver::hybrid
{
// the ions' totals from the populations' (complete) moments
void compute_moments(auto&& level)
{
    DAMN_LOG_SCOPE(1, "Simulator::compute_moments_");
    level.per_patch([](auto& patch) { patch.ions.update(); });
}

void fill_ion_borders(auto&& level)
{
    // the first exchange after the push: waits here are other ranks' longer pushes
    DAMN_LOG_SCOPE(1, "Simulator::fill_ion_borders_");
    level.copy(damn::amr::sum_borders, "IonPopBorderSum");
}

// J from B (Ampere, on the physical box + 1: what Ohm reads, so J is never exchanged), the
// electrons from the ions and J, then E from Ohm's law; electromag: the HybridPatch member
// holding the B used and the E computed, the model's (em) or the predicted one (emPred)
template<auto electromag, auto opts>
void ampere_ohm(HybridModel<opts> const& model, auto&& level)
{
    // one loop each, so each has its own scope (time, counters)
    {
        DAMN_LOG_SCOPE(1, "Simulator::ampere");
        level.per_patch([&](auto& patch) {
            core::AmpereTransformer{}(patch.layout, (patch.*electromag).B, patch.J);
        });
    }
    {
        DAMN_LOG_SCOPE(1, "Simulator::electrons");
        level.per_patch([&](auto& patch) { patch.electrons().update(patch.layout); });
    }
    DAMN_LOG_SCOPE(1, "Simulator::ohm");
    level.per_patch([&](auto& patch) {
        auto& em = patch.*electromag;
        core::OhmTransformer{*model.ohm}(patch.layout, patch.electrons().density(),
                                         patch.electrons().velocity(), patch.electrons().pressure(),
                                         em.B, patch.J, em.E);
    });
}

} // namespace plamn::solver::hybrid

namespace plamn::solver::hybrid::detail
{
// the level border's moments: the level ghost particles split from the coarser level
// at both ends of its step, weighted linearly in time (PHARE's fillIonPopMomentGhosts),
// before the border sums complete every patch's copy: each is held by one patch,
// deposited once (as PHARE: each level ghost deposited once)
template<auto opts>
void deposit_level_ghosts(auto& level, double const t)
{
    if (level.number() == 0)
        return;
    auto alpha = level.alpha(t);
    if (alpha < -1e-10 or alpha > 1 + 1e-10)
        throw std::runtime_error("plamn Simulator: level ghost time interpolation coefficient "
                                 + std::to_string(alpha));
    alpha = std::clamp(alpha, 0., 1.);
    level.per_patch([&](auto& patch) {
        core::Interpolating<opts.dimension, opts.interp_order> interpolate;
        for (auto& pop : patch.ions)
        {
            interpolate(pop.levelGhostParticlesOld(), pop.particleDensity(), pop.flux(),
                        patch.layout, 1. - alpha);
            interpolate(pop.levelGhostParticlesNew(), pop.particleDensity(), pop.flux(),
                        patch.layout, alpha);
        }
    });
}

template<auto opts>
void average(auto& level, double const t, double const dt)
{
    DAMN_LOG_SCOPE(1, "Simulator::average_");
    level.per_patch([](auto& patch) {
        for (std::size_t i = 0; i < 3; ++i)
        {
            core::average(patch.em.B[i], patch.emPred.B[i], patch.emAvg.B[i]);
            core::average(patch.em.E[i], patch.emPred.E[i], patch.emAvg.E[i]);
        }
    });
    level.fill_electric_ghosts("EMAvg_E", t + dt / 2);
}

template<auto opts>
void move_ions(HybridModel<opts>& model, auto& level, double const t, double const dt,
               core::UpdaterMode const mode)
{
    using Patch_t = HybridModel<opts>::Patch_t;
    DAMN_LOG_SCOPE(1, "Simulator::move_ions");
    {
        DAMN_LOG_SCOPE(1, "Simulator::move_ions::push"); // and deposit
        if constexpr (Patch_t::per_patch_mover)
            level.per_patch([&](auto& patch) {
                model.mover.updatePopulations(patch.ions, patch.emAvg, *patch.boxing, dt, mode);
            });
        else
        { // the level's patches at once, each with its boxing by patch id
            auto& data        = level.data();
            using LevelData_t = HybridModel<opts>::LevelData_t;
            simulator::LevelAccessor<LevelData_t, Patch_t, &Patch_t::emAvg> accessor{data};
            using Boxing_t = Patch_t::Boxing_t;
            std::unordered_map<std::string, Boxing_t> boxings;
            for (std::size_t i = 0; i < accessor.size(); ++i)
                boxings.try_emplace(accessor.id(i), *data.data[i]->boxing);
            model.mover.updatePopulations(accessor, boxings, dt, mode);
        }
    }
    if (mode != core::UpdaterMode::domain_only)
    {
        // as PHARE's deposit after the exchange (and the level initialization): the
        //  particles that left their patch go to the patch owning their cell first, then
        //  each patch deposits its domain particles, every particle once (the
        //  domain_only predictor deposits its pushed copies in the updater)
        {
            DAMN_LOG_SCOPE(2, "Simulator::move_ions::gather_leaving");
            level.copy(damn::amr::gather_leaving, "IonGhostParticles");
        }
        DAMN_LOG_SCOPE(2, "Simulator::move_ions::deposit_domain");
        level.per_patch([](auto& patch) {
            for (auto& pop : patch.ions)
                pop.patchGhostParticles().clear();
            core::deposit_domain<opts.dimension, opts.interp_order>(patch.ions, patch.layout);
        });
    }

    // as PHARE: each level ghost particle deposited once, by the patch holding it,
    //  then the border sums: every patch's copy of a node the same
    {
        DAMN_LOG_SCOPE(2, "Simulator::move_ions::deposit_level_ghosts");
        deposit_level_ghosts<opts>(level, t + dt);
    }
    fill_ion_borders(level);
    compute_moments(level);
}

// B predicted (into emPred) or corrected (into em) from E by Faraday, then its ghosts,
//  J, the electrons and E
template<auto B_to, auto opts>
void faraday_ampere_ohm(HybridModel<opts> const& model, auto& level, auto&& E_of,
                        std::string const& key, double const t, double const dt)
{
    {
        DAMN_LOG_SCOPE(1, "Simulator::faraday");
        level.per_patch([&](auto& patch) {
            core::FaradayTransformer{}(patch.layout, patch.em.B, E_of(patch), (patch.*B_to).B, dt);
        });
    }
    level.fill_magnetic_ghosts(key, t + dt);
    ampere_ohm<B_to>(model, level);
}

} // namespace plamn::solver::hybrid::detail

namespace plamn::solver
{
template<auto opts>
void advance(HybridModel<opts>& model, auto&& level, double const t, double const dt)
{
    using Patch_t   = HybridModel<opts>::Patch_t;
    namespace d     = hybrid::detail;
    auto const E    = [](auto& patch) -> auto& { return patch.em.E; };
    auto const Eavg = [](auto& patch) -> auto& { return patch.emAvg.E; };
    {
        DAMN_LOG_SCOPE(1, "Simulator::predictor1_");
        d::faraday_ampere_ohm<&Patch_t::emPred>(model, level, E, "EMPred_B", t, dt);
    }
    d::average<opts>(level, t, dt);
    d::move_ions(model, level, t, dt, core::UpdaterMode::domain_only);
    {
        DAMN_LOG_SCOPE(1, "Simulator::predictor2_");
        d::faraday_ampere_ohm<&Patch_t::emPred>(model, level, Eavg, "EMPred_B", t, dt);
    }
    d::average<opts>(level, t, dt);
    d::move_ions(model, level, t, dt, core::UpdaterMode::all);
    {
        DAMN_LOG_SCOPE(1, "Simulator::corrector_");
        d::faraday_ampere_ohm<&Patch_t::em>(model, level, Eavg, "EM_B", t, dt);
        level.fill_electric_ghosts("EM_E", t + dt);
    }
}

} // namespace plamn::solver

#endif // PLAMN_SOLVER_HYBRID_ADVANCE_HPP
