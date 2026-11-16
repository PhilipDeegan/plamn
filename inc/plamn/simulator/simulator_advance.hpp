#ifndef PLAMN_SIMULATOR_SIMULATOR_ADVANCE_HPP
#define PLAMN_SIMULATOR_SIMULATOR_ADVANCE_HPP

// plamn::Simulator: time refined advance, SolverPPC per level, synchronization and reflux
// IWYU pragma: private, include "plamn/simulator/simulator.hpp"

#include "damn/amr/copy.hpp"
#include "damn/amr/level_map.hpp"
#include "damn/amr/time_refinement.hpp"
#include "damn/core/logger.hpp"

#include "plamn/simulator/simulator.hpp" // IWYU pragma: keep (the class)
#include "plamn/amr/coarsen.hpp"
#include "plamn/amr/level_pairs.hpp"
#include "plamn/amr/refine.hpp"
#include "plamn/amr/tagging.hpp"
#include "plamn/core/data/field/field_algorithm.hpp"
#include "plamn/core/numerics/ampere/ampere_transformer.hpp"
#include "plamn/core/numerics/faraday/faraday_transformer.hpp"
#include "plamn/core/numerics/ohm/ohm.hpp"
#include "plamn/core/numerics/ohm/ohm_transformer.hpp"
#include "plamn/core/numerics/particles_synchronised_mover/particles_synchronised_mover_def.hpp"

#include "phlop/timing/scope.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
// plamn::Simulator: time refined advance, SolverPPC per level, synchronization and reflux
// IWYU pragma: private, include "plamn/simulator/simulator.hpp"

namespace plamn
{
template<auto opts>
double Simulator<opts>::advance()
{
    DAMN_LOG_SCOPE(1, "Simulator::advance");
    if (!isInitialized_)
        throw std::runtime_error("plamn Simulator: initialize() first");

    if (should_rebalance_())
        rebalance_(currentTime_);

    Hooks hooks{*this};
    damn::amr::advance(hierarchy_, hooks, currentTime_, dt_, time_ratio());
    currentTime_ += dt_;
    ++steps_;
    if (tagging_ and steps_ % regrid_interval_ == 0)
        regrid_(currentTime_);
    return dt_;
}

template<auto opts>
void Simulator<opts>::advance_level_(int const level_number, double const t, double const dt)
{
    auto const ln = static_cast<std::size_t>(level_number);
    predictor1_(ln, t, dt);
    average_(ln, t, dt);
    move_ions_(ln, t, dt, core::UpdaterMode::domain_only);
    predictor2_(ln, t, dt);
    average_(ln, t, dt);
    move_ions_(ln, t, dt, core::UpdaterMode::all);
    corrector_(ln, t, dt);
    if (ln + 1 == levels_.size()) // the finest; coarser ones after their synchronization
        fine_dump_(ln, t + dt);

    // the finest level's share of the coarser level's reflux, per substep: the time average of
    //  E over them (coarser finer levels accumulate theirs when synchronized)
    if (ln > 0 and ln + 1 == levels_.size())
        accumulate_flux_sum_(ln, 1. / time_ratio());
}

// E and B at the start of the step, for the finer level's time interpolation and the reflux
template<auto opts>
void Simulator<opts>::prepare_step_(int const ln, double)
{
    per_patch_(static_cast<std::size_t>(ln), [](auto& patch) {
        for (std::size_t i = 0; i < 3; ++i)
        {
            patch.oldB[i].copyData(patch.emB[i]);
            patch.oldE[i].copyData(patch.emE[i]);
        }
    });
}

// the coarser level is at coarse_t + coarse_dt: the level ghost particles of then, split from
// its domain particles
template<auto opts>
void Simulator<opts>::first_step_(int const level_number, double, double const coarse_t,
                                  double const coarse_dt)
{
    auto const ln = static_cast<std::size_t>(level_number);
    auto& l       = levels_[ln];
    l.coarse_t    = coarse_t;
    l.coarse_dt   = coarse_dt;
    per_patch_(ln, [](auto& patch) {
        for (auto& pop : patch.ions)
            pop.levelGhostParticlesNew().clear();
        for (auto& e : patch.fluxSumE)
            e.zero();
    });
    for (auto const& name : population_names_)
        refine_(ln, amr::ParticleSplitter<Splitter_t>{}, damn::amr::LevelRegion::LevelGhosts,
                {{name + "_domain", name + "_level_ghost_new", ""}});
}

template<auto opts>
void Simulator<opts>::last_step_(int const ln)
{
    per_patch_(static_cast<std::size_t>(ln), [](auto& patch) {
        for (auto& pop : patch.ions)
        {
            std::swap(pop.levelGhostParticlesNew(), pop.levelGhostParticlesOld());
            pop.levelGhostParticlesNew().clear();
            pop.levelGhostParticles() = pop.levelGhostParticlesOld();
        }
    });
}

// level ln onto ln - 1, both at t: E and the total ion charge density and flux coarsened (as
// PHARE's synchronize: the populations' moments, the mass density and the bulk velocity stay the
// coarser level's own), then the coarser level's B recomputed from the fine time average of E
// (reflux), then its ghosts refilled
template<auto opts>
void Simulator<opts>::synchronize_(int const level_number, double const t)
{
    DAMN_LOG_SCOPE(1, "Simulator::synchronize_");
    auto const ln     = static_cast<std::size_t>(level_number);
    auto const coarse = ln - 1;

    coarsen_(ln, amr::ElectricFieldCoarsener{}, amr::vector_pairs("EM_E", "EM_E"));
    auto moments = amr::vector_pairs("ions_charge_flux", "ions_charge_flux");
    moments.push_back({"ions_charge_density", "ions_charge_density", ""});
    coarsen_(ln, amr::MomentsCoarsener{}, moments);

    // reflux: the coarser Eavg under the fine level is the fine one's time average
    coarsen_(ln, amr::ElectricFieldCoarsener{}, amr::vector_pairs("EMAvg_E", "EM_fluxSum_E"));
    copy_(coarse, damn::amr::fill_ghosts, "EMAvg_E");
    auto const dt = levels_[ln].coarse_dt;
    per_patch_(coarse, [&](auto& patch) {
        core::FaradayTransformer{}(patch.layout, patch.Bold, patch.emAvg.E, patch.em.B, dt);
    });
    fill_magnetic_ghosts_(coarse, "EM_B", t);
    if (coarse > 0)
        accumulate_flux_sum_(coarse, 1. / time_ratio());

    // ghosts after the coarsening
    fill_electric_ghosts_(coarse, "EM_E", t);
    copy_(coarse, damn::amr::fill_ghosts, "IonMoments");
    fine_dump_(coarse, t);
}

template<auto opts>
void Simulator<opts>::accumulate_flux_sum_(std::size_t const ln, double const coef)
{
    per_patch_(ln, [&](auto& patch) {
        for (std::size_t i = 0; i < 3; ++i)
        {
            auto* sum         = patch.fluxSumE[i].data();
            auto const* avg   = patch.avgE[i].data();
            auto const values = patch.fluxSumE[i].size();
            for (std::size_t k = 0; k < values; ++k)
                sum[k] += coef * avg[k];
        }
    });
}

// the ions' totals from the populations' (complete) moments
template<auto opts>
void Simulator<opts>::compute_moments_(std::size_t const ln)
{
    DAMN_LOG_SCOPE(1, "Simulator::compute_moments_");
    per_patch_(ln, [](auto& patch) { patch.ions.update(); });
}

template<auto opts>
void Simulator<opts>::fill_ion_borders_(std::size_t const ln)
{
    // the first exchange after the push: waits here are other ranks' longer pushes
    DAMN_LOG_SCOPE(1, "Simulator::fill_ion_borders_");
    copy_(ln, damn::amr::sum_borders, "IonPopBorderSum");
}

// the level border's moments: the level ghost particles split from the coarser level at both
// ends of its step, weighted linearly in time (PHARE's fillIonPopMomentGhosts), each by the
// patch owning its cell only (PHAREHUB/PHARE#1580: patches meeting on the level border hold
// the same ones), before the border sums complete every patch's copy
template<auto opts>
void Simulator<opts>::deposit_level_ghosts_(std::size_t const ln, double const t)
{
    if (ln == 0)
        return;
    auto alpha = alpha_(ln, t);
    if (alpha < -1e-10 or alpha > 1 + 1e-10)
        throw std::runtime_error("plamn Simulator: level ghost time interpolation coefficient "
                                 + std::to_string(alpha));
    alpha = std::clamp(alpha, 0., 1.);
    per_patch_(ln, [&](auto& patch) {
        Interpolator interpolate;
        interpolate.skip_cells(patch.level_ghost_not_owned);
        for (auto& pop : patch.ions)
        {
            interpolate(pop.levelGhostParticlesOld(), pop.particleDensity(), pop.flux(),
                        patch.layout, 1. - alpha);
            interpolate(pop.levelGhostParticlesNew(), pop.particleDensity(), pop.flux(),
                        patch.layout, alpha);
        }
    });
}

// J from B (Ampere, on the physical box + 1: what Ohm reads, so J is never exchanged), the
// electrons from the ions and J, then E from Ohm's law; electromag: the HybridPatch member
// holding the B used and the E computed, the model's (em) or the predicted one (emPred)
template<auto opts>
template<auto electromag>
void Simulator<opts>::ampere_ohm_(std::size_t const ln)
{
    // one loop each, so each has its own scope (time, counters)
    {
        DAMN_LOG_SCOPE(1, "Simulator::ampere");
        per_patch_(ln, [&](auto& patch) {
            core::AmpereTransformer{}(patch.layout, (patch.*electromag).B, patch.J);
        });
    }
    {
        DAMN_LOG_SCOPE(1, "Simulator::electrons");
        per_patch_(ln, [&](auto& patch) { patch.electrons().update(patch.layout); });
    }
    DAMN_LOG_SCOPE(1, "Simulator::ohm");
    per_patch_(ln, [&](auto& patch) {
        auto& em = patch.*electromag;
        core::OhmTransformer{*ohm_info_}(patch.layout, patch.electrons().density(),
                                         patch.electrons().velocity(), patch.electrons().pressure(),
                                         em.B, patch.J, em.E);
    });
}

template<auto opts>
void Simulator<opts>::average_(std::size_t const ln, double const t, double const dt)
{
    DAMN_LOG_SCOPE(1, "Simulator::average_");
    per_patch_(ln, [](auto& patch) {
        for (std::size_t i = 0; i < 3; ++i)
        {
            core::average(patch.em.B[i], patch.emPred.B[i], patch.emAvg.B[i]);
            core::average(patch.em.E[i], patch.emPred.E[i], patch.emAvg.E[i]);
        }
    });
    fill_electric_ghosts_(ln, "EMAvg_E", t + dt / 2);
}

template<auto opts>
void Simulator<opts>::move_ions_(std::size_t const ln, double const t, double const dt,
                                 core::UpdaterMode const mode)
{
    DAMN_LOG_SCOPE(1, "Simulator::move_ions");
    {
        DAMN_LOG_SCOPE(1, "Simulator::move_ions::push"); // and deposit
        per_patch_(ln, [&](auto& patch) {
            mover_.updatePopulations(patch.ions, patch.emAvg, *patch.boxing, dt, mode);
        });
    }
    if (mode != core::UpdaterMode::domain_only)
    {
        // as PHAREHUB/PHARE#1441 (and the level initialization): the particles that left their
        //  patch go to the patch owning their cell first, then each patch deposits its domain
        //  particles, every particle once (the domain_only predictor deposits its pushed copies
        //  in the updater)
        copy_(ln, damn::amr::gather_leaving, "IonGhostParticles");
        per_patch_(ln, [](auto& patch) {
            for (auto& pop : patch.ions)
                pop.patchGhostParticles().clear();
            core::deposit_domain<dimension, interp_order>(patch.ions, patch.layout);
        });
    }

    // as PHAREHUB/PHARE#1580: each level ghost particle deposited once, by the patch owning its
    //  cell, then the border sums: every patch's copy of a node the same
    deposit_level_ghosts_(ln, t + dt);
    fill_ion_borders_(ln);
    compute_moments_(ln);
}

template<auto opts>
void Simulator<opts>::predictor1_(std::size_t const ln, double const t, double const dt)
{
    DAMN_LOG_SCOPE(1, "Simulator::predictor1_");
    {
        DAMN_LOG_SCOPE(1, "Simulator::faraday");
        per_patch_(ln, [&](auto& patch) {
            core::FaradayTransformer{}(patch.layout, patch.em.B, patch.em.E, patch.emPred.B, dt);
        });
    }
    fill_magnetic_ghosts_(ln, "EMPred_B", t + dt);
    ampere_ohm_<&Patch_t::emPred>(ln);
}

template<auto opts>
void Simulator<opts>::predictor2_(std::size_t const ln, double const t, double const dt)
{
    DAMN_LOG_SCOPE(1, "Simulator::predictor2_");
    {
        DAMN_LOG_SCOPE(1, "Simulator::faraday");
        per_patch_(ln, [&](auto& patch) {
            core::FaradayTransformer{}(patch.layout, patch.em.B, patch.emAvg.E, patch.emPred.B, dt);
        });
    }
    fill_magnetic_ghosts_(ln, "EMPred_B", t + dt);
    ampere_ohm_<&Patch_t::emPred>(ln);
}

template<auto opts>
void Simulator<opts>::corrector_(std::size_t const ln, double const t, double const dt)
{
    DAMN_LOG_SCOPE(1, "Simulator::corrector_");
    {
        DAMN_LOG_SCOPE(1, "Simulator::faraday");
        per_patch_(ln, [&](auto& patch) {
            core::FaradayTransformer{}(patch.layout, patch.em.B, patch.emAvg.E, patch.em.B, dt);
        });
    }
    fill_magnetic_ghosts_(ln, "EM_B", t + dt);
    ampere_ohm_<&Patch_t::em>(ln);
    fill_electric_ghosts_(ln, "EM_E", t + dt);
}

} // namespace plamn

#endif // PLAMN_SIMULATOR_SIMULATOR_ADVANCE_HPP
