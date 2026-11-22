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

#include "plamn/core/data/field/level_fields.hpp"
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
#include "damn/core/def.hpp"

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

namespace detail
{
    // J = curl B on every patch of the level: unified memory, one launch for the level, its item
    //  list kept (no allocation once warm: an item per patch, or per tile of a tiled patch)
    template<auto electromag, auto opts>
        requires(opts.alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
    void ampere_level(auto& level)
    {
        using Patch_t       = HybridModel<opts>::Patch_t;
        using Transformer_t = core::AmpereTransformer<>;
        using Item_t = Transformer_t::item_t<typename Patch_t::Layout_t,
                                             std::decay_t<decltype(std::declval<Patch_t&>().J)>>;
        static thread_local damn::core::gpu::Vec_t<Item_t> items;
        items.clear();
        level.per_patch([&](auto& patch) {
            Transformer_t::gpu_items(patch.layout, (patch.*electromag).B, patch.J, items);
        });
        Transformer_t::run_gpu(items);
    }
    // Bnew = B - dt curl E on every patch of the level: unified memory (untiled), one launch, its
    //  item list kept; host memory or tiled fields per patch
    template<auto B_to, auto opts>
        requires(opts.alloc_mode == damn::AllocatorMode::GPU_UNIFIED
                 and !core::is_tiled(opts.layout_mode))
    void faraday_level(auto& level, auto const& E_of, double const dt)
    {
        auto const item_of = [&](auto& patch) {
            return core::FaradayTransformer::make_item(patch.layout, patch.em.B, E_of(patch),
                                                       (patch.*B_to).B);
        };
        using Patch_t = HybridModel<opts>::Patch_t;
        using Item_t  = decltype(item_of(std::declval<Patch_t&>()));
        static thread_local damn::core::gpu::Vec_t<Item_t> items;
        items.clear();
        level.per_patch([&](auto& patch) { items.push_back(item_of(patch)); });
        core::FaradayTransformer::run_gpu(items, dt);
    }
    template<auto B_to, auto opts>
        requires(opts.alloc_mode == damn::AllocatorMode::CPU or core::is_tiled(opts.layout_mode))
    void faraday_level(auto& level, auto const& E_of, double const dt)
    {
        level.per_patch([&](auto& patch) {
            core::FaradayTransformer{}(patch.layout, patch.em.B, E_of(patch), (patch.*B_to).B, dt);
        });
    }

    // E from Ohm's law on every patch of the level: unified memory, one launch for the level,
    //  its item list kept (an item per untiled patch; tiled patches through the transformer's
    //  own tiles path)
    template<auto electromag, auto opts>
        requires(opts.alloc_mode == damn::AllocatorMode::GPU_UNIFIED
                 and !core::is_tiled(opts.layout_mode))
    void ohm_level(HybridModel<opts> const& model, auto& level)
    {
        core::OhmTransformer const ohm{*model.ohm};
        auto const item_of = [&](auto& patch) {
            auto& em = patch.*electromag;
            return ohm.make_item(patch.layout, patch.electrons().density(),
                                 patch.electrons().velocity(), patch.electrons().pressure(), em.B,
                                 patch.J, em.E);
        };
        using Patch_t = HybridModel<opts>::Patch_t;
        using Item_t  = decltype(item_of(std::declval<Patch_t&>()));
        static thread_local damn::core::gpu::Vec_t<Item_t> items;
        items.clear();
        level.per_patch([&](auto& patch) { items.push_back(item_of(patch)); });
        ohm.run_gpu(items);
    }
    // host memory, or tiled fields (the transformer's tiles path): per patch
    template<auto electromag, auto opts>
        requires(opts.alloc_mode == damn::AllocatorMode::CPU or core::is_tiled(opts.layout_mode))
    void ohm_level(HybridModel<opts> const& model, auto& level)
    {
        level.per_patch([&](auto& patch) {
            auto& em = patch.*electromag;
            core::OhmTransformer{*model.ohm}(patch.layout, patch.electrons().density(),
                                             patch.electrons().velocity(),
                                             patch.electrons().pressure(), em.B, patch.J, em.E);
        });
    }

    // host memory: per patch
    template<auto electromag, auto opts>
        requires(opts.alloc_mode == damn::AllocatorMode::CPU)
    void ampere_level(auto& level)
    {
        level.per_patch([&](auto& patch) {
            core::AmpereTransformer{}(patch.layout, (patch.*electromag).B, patch.J);
        });
    }
} // namespace detail

// J from B (Ampere, on the physical box + 1: what Ohm reads, so J is never exchanged), the
// electrons from the ions and J, then E from Ohm's law; electromag: the HybridPatch member
// holding the B used and the E computed, the model's (em) or the predicted one (emPred)
template<auto electromag, auto opts>
void ampere_ohm(HybridModel<opts> const& model, auto&& level)
{
    // one loop each, so each has its own scope (time, counters)
    {
        DAMN_LOG_SCOPE(1, "Simulator::ampere");
        detail::ampere_level<electromag, opts>(level);
    }
    {
        DAMN_LOG_SCOPE(1, "Simulator::electrons");
        level.per_patch([&](auto& patch) { patch.electrons().update(patch.layout); });
    }
    DAMN_LOG_SCOPE(1, "Simulator::ohm");
    detail::ohm_level<electromag, opts>(model, level);
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
        core::DepositInterpolating<opts.dimension, opts.interp_order, opts.alloc_mode> interpolate;
        for (auto& pop : patch.ions)
        {
            interpolate(pop.levelGhostParticlesOld(), pop.particleDensity(), pop.flux(),
                        patch.layout, 1. - alpha);
            interpolate(pop.levelGhostParticlesNew(), pop.particleDensity(), pop.flux(),
                        patch.layout, alpha);
        }
    });
}

namespace detail
{
    // a patch's electromag (em, emPred or emAvg): B then E, component by component
    template<auto member>
    inline constexpr auto electromag_of = [](auto& patch, auto& fields) {
        fields.add((patch.*member).B);
        fields.add((patch.*member).E);
    };

    // emAvg = average(em, emPred): unified memory, the whole level's at once (one kernel)
    template<auto opts>
        requires(opts.alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
    void average_electromag(auto& level)
    {
        using Patch_t     = HybridModel<opts>::Patch_t;
        auto constexpr am = opts.alloc_mode;
        auto const em     = core::level_of<am>(level, electromag_of<&Patch_t::em>);
        auto const pred   = core::level_of<am>(level, electromag_of<&Patch_t::emPred>);
        core::level_of<am>(level, electromag_of<&Patch_t::emAvg>).average(em, pred);
    }
    // host memory: per patch
    template<auto opts>
        requires(opts.alloc_mode == damn::AllocatorMode::CPU)
    void average_electromag(auto& level)
    {
        level.per_patch([](auto& patch) {
            for (std::size_t i = 0; i < 3; ++i)
            {
                core::average(patch.em.B[i], patch.emPred.B[i], patch.emAvg.B[i]);
                core::average(patch.em.E[i], patch.emPred.E[i], patch.emAvg.E[i]);
            }
        });
    }
} // namespace detail

template<auto opts>
void average(auto& level, double const t, double const dt)
{
    DAMN_LOG_SCOPE(1, "Simulator::average_");
    detail::average_electromag<opts>(level);
    level.fill_electric_ghosts("EMAvg_E", t + dt / 2);
}

namespace detail
{
    // every patch's domain particles deposited (after the exchange), the moments zeroed first:
    //  unified memory, the whole level's at once (one kernel), then the device deposits
    template<auto opts>
        requires(opts.alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
    void deposit_domains(auto& level)
    {
        core::level_of<opts.alloc_mode>(level, core::domain_moments).zero();
        level.per_patch([](auto& patch) {
            for (auto& pop : patch.ions)
                pop.patchGhostParticles().clear();
            core::depositParticles(
                patch.ions, patch.layout,
                core::DepositInterpolating<opts.dimension, opts.interp_order, opts.alloc_mode>{},
                core::DomainDeposit{});
        });
    }
    // host memory: per patch, as deposit_domain
    template<auto opts>
        requires(opts.alloc_mode == damn::AllocatorMode::CPU)
    void deposit_domains(auto& level)
    {
        level.per_patch([](auto& patch) {
            for (auto& pop : patch.ions)
                pop.patchGhostParticles().clear();
            core::deposit_domain<opts.dimension, opts.interp_order>(patch.ions, patch.layout);
        });
    }
} // namespace detail

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
            // the patches' boxings (and their device box lists), kept per decomposition
            auto& boxings = model.level_boxings[static_cast<int>(level.number())];
            if (boxings.generation != level.generation())
                boxings.rebuild(level.generation(), accessor, data);
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
            level.copy(damn::amr::gather_leaving, model.particle_routes->gather_leaving);
        }
        DAMN_LOG_SCOPE(2, "Simulator::move_ions::deposit_domain");
        detail::deposit_domains<opts>(level);
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
        faraday_level<B_to, opts>(level, E_of, dt);
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
    DAMN_SUPER_DEBUG_DO(level.sums("predictor1", t);)
    d::average<opts>(level, t, dt);
    DAMN_SUPER_DEBUG_DO(level.sums("average1", t);)
    d::move_ions(model, level, t, dt, core::UpdaterMode::domain_only);
    DAMN_SUPER_DEBUG_DO(level.sums("moments1", t);)
    {
        DAMN_LOG_SCOPE(1, "Simulator::predictor2_");
        d::faraday_ampere_ohm<&Patch_t::emPred>(model, level, Eavg, "EMPred_B", t, dt);
    }
    DAMN_SUPER_DEBUG_DO(level.sums("predictor2", t);)
    d::average<opts>(level, t, dt);
    DAMN_SUPER_DEBUG_DO(level.sums("average2", t);)
    d::move_ions(model, level, t, dt, core::UpdaterMode::all);
    DAMN_SUPER_DEBUG_DO(level.sums("moments2", t);)
    {
        DAMN_LOG_SCOPE(1, "Simulator::corrector_");
        d::faraday_ampere_ohm<&Patch_t::em>(model, level, Eavg, "EM_B", t, dt);
        level.fill_electric_ghosts("EM_E", t + dt);
    }
    DAMN_SUPER_DEBUG_DO(level.sums("corrector", t);)
}

} // namespace plamn::solver

#endif // PLAMN_SOLVER_HYBRID_ADVANCE_HPP
