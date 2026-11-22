#ifndef PLAMN_SOLVER_HYBRID_INITIALIZE_HPP
#define PLAMN_SOLVER_HYBRID_INITIALIZE_HPP

// a hybrid level's data at its creation (PHARE's HybridLevelInitializer): the root level from the
//  user's functions, a refined level from its coarser level (B divergence free, E, the particles
//  split), and their moments; a level moved to other ranks (rebalance) keeps its moments
//
// level: the simulator's services for one level (solver/hybrid/advance.hpp, level_steps.hpp), and
//   check_set(key, where)                debug: throws if a ghost of key was not filled
//   mesh_size()                          the level's

#include "plamn/amr/untiled.hpp"
#include "damn/amr/copy.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/level_map.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "plamn/amr/level_pairs.hpp"
#include "plamn/amr/refine.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/models/electromag_initializer.hpp"
#include "plamn/core/models/quantities/quantity.hpp"
#include "plamn/core/numerics/ampere/ampere_transformer.hpp"
#include "plamn/core/numerics/interpolator/interpolating.hpp"
#include "plamn/core/numerics/moments/moments.hpp"
#include "plamn/core/numerics/particles_synchronised_mover/particles_synchronised_mover.hpp"
#include "plamn/solver/hybrid/advance.hpp"
#include "plamn/solver/hybrid/level_steps.hpp"
#include "plamn/solver/hybrid/model.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>
#include <format>
#include <string>
#include <type_traits>

namespace plamn::solver::hybrid
{
// the root level from the user's functions; sim: the input's "simulation"
template<auto opts>
void initialize_root(HybridModel<opts>& model, auto&& level, auto const& sim)
{
    using Patch_t = HybridModel<opts>::Patch_t;
    level.per_patch([&](auto& patch) {
        core::ElectromagInitializer<opts.dimension>{sim["electromag"]}(patch.em, patch.layout);
        for (auto& pop : patch.ions)
        {
            auto initializer = Patch_t::Types::ParticleInitializerFactory_t::create(
                pop.particleInitializerInfo(), pop.name());
            initializer->loadParticles(pop.domainParticles(), patch.layout);
        }
    });
    level.copy(damn::amr::fill_ghosts, "EM_B");
    // a node on a periodic boundary is initialized twice (at 0 and at L; they differ where the
    //  profile is not periodic): every patch holding it takes the max of its copies, once, after
    //  the ghost fill (the max runs over whole ghost overlaps), before anything reads B (PHARE's
    //  fillRootGhosts)
    level.copy(damn::amr::max_borders, "EM_B");
    level.copy(damn::amr::fill_ghosts, "EM_E");
    level.check_set("EM_B", "root initialization"); // E: from Ohm's law below

    // moments from the particles, completed across patch borders
    level.per_patch([&](auto& patch) {
        core::resetMoments(patch.ions);
        core::depositParticles(
            patch.ions, patch.layout,
            core::DepositInterpolating<opts.dimension, opts.interp_order, opts.alloc_mode>{},
            core::DomainDeposit{});
    });
    fill_ion_borders(level);
    compute_moments(level);

    // J from B, then E from Ohm's law
    ampere_ohm<&Patch_t::em>(model, level);
    level.copy(damn::amr::fill_ghosts, "EM_E");
    level.check_set("EM_E", "root initialization");
}

// a new level's B from the coarser level's: refined, then the divergence free touch up (its
//  ghosts are filled after, by the magnetic ghost fill)
template<auto opts>
void refine_magnetic(HybridModel<opts> const&, auto&& level)
{
    using Patch_t = HybridModel<opts>::Patch_t;
    level.refine(amr::FieldRefiner{}, damn::amr::LevelRegion::Interior,
                 amr::vector_pairs("EM_B", "EM_B"));
    auto& ld = level.data();
    for (std::size_t i = 0; i < damn::amr::size(ld); ++i)
    {
        auto& patch       = *ld.data[i];
        auto constexpr gw = Patch_t::field_ghost_width;
        amr::with_untiled_resources<gw>(patch.emB, [&](auto const& B) { // tiled: untiled copies
            amr::divergence_free_touch_up(B, ld.patches[i].box, ld.patches[i].box,
                                          level.mesh_size());
        });
    }
}

// SoAVXChunked: chunks before the first push
template<auto opts>
void sort_refined_particles(HybridModel<opts> const&, auto&& level)
{
    if constexpr (opts.layout_mode == core::LayoutMode::SoAVXChunked)
        level.per_patch([](auto& patch) {
            for (auto& pop : patch.ions)
                for (auto* ps : {&pop.domainParticles(), &pop.levelGhostParticles(),
                                 &pop.levelGhostParticlesOld()})
                    if (!ps->sorted())
                        ps->sort();
        });
}

// a refined level's moments and J from its particles: as the ions moved (advance.hpp), the
//  domain particles, the level ghosts (each held by one patch), then the border sums
template<auto opts>
void refined_moments(HybridModel<opts> const&, auto&& level)
{
    level.per_patch([&](auto& patch) {
        core::deposit_domain<opts.dimension, opts.interp_order>(patch.ions, patch.layout);
        core::depositParticles(
            patch.ions, patch.layout,
            core::DepositInterpolating<opts.dimension, opts.interp_order, opts.alloc_mode>{},
            core::LevelGhostDeposit{});
    });
    fill_ion_borders(level);
    compute_moments(level);

    // J from B; E was refined
    level.per_patch(
        [&](auto& patch) { core::AmpereTransformer{}(patch.layout, patch.em.B, patch.J); });
}

// a refined level from its coarser level at t: B, E and the particles, the level ghost particles
//  split from it, then the moments and J
template<auto opts>
void initialize_refined(HybridModel<opts> const& model, auto&& level, double const t)
{
    using damn::amr::LevelRegion;
    using Splitter_t = HybridModel<opts>::Splitter_t;
    refine_magnetic(model, level);
    level.refine(amr::FieldRefiner{}, LevelRegion::Interior, amr::vector_pairs("EM_E", "EM_E"));
    level.fill_magnetic_ghosts("EM_B", t);
    level.fill_electric_ghosts("EM_E", t);

    auto const& routes = *model.particle_routes;
    level.refine(amr::ParticleSplitter<Splitter_t>{}, LevelRegion::Interior,
                 routes.interior_splits);
    level.refine(amr::ParticleSplitter<Splitter_t>{}, LevelRegion::LevelGhosts,
                 routes.old_ghost_splits);
    own_level_ghosts(level, /*new_ghosts=*/false);
    level.per_patch([](auto& patch) {
        for (auto& pop : patch.ions)
            pop.levelGhostParticles() = pop.levelGhostParticlesOld();
    });
    sort_refined_particles(model, level);

    refined_moments(model, level);
}

// a level regridded (a rebalance: the same cells; a tagging regrid: new boxes): its domain
//  moments kept where the old level was (old_cells, its periodic images included: they hold the
//  synchronized moments, which a deposit would lose), from its particles elsewhere (cells new to
//  the level) and on its ghost nodes
template<auto opts>
void moved_moments(HybridModel<opts> const& model, auto&& level, auto const& old_cells)
{
    auto const each_moment = [](auto& patch, auto&& fn) {
        for (auto& pop : patch.populations)
        {
            fn(pop->density);
            fn(pop->charge_density);
            for (auto& f : pop->flux)
                fn(f);
        }
    };
    std::vector<std::vector<std::vector<double>>> kept; // per patch, per moment: its values
    level.per_patch([&](auto& patch) {
        auto& mine = kept.emplace_back();
        each_moment(patch, [&](auto const& f) {
            amr::with_untiled(
                f, [&](auto const& u) { mine.emplace_back(u.data(), u.data() + u.size()); });
        });
    });
    refined_moments(model, level); // the ghost nodes and new cells; the old domain put back below
    std::size_t i = 0;
    level.per_patch([&](auto& patch) {
        std::size_t m = 0;
        each_moment(patch, [&](auto& tiled) {
            amr::modify_untiled(tiled, [&](auto& f) {
                auto const& values = kept[i][m++];
                auto constexpr dim = std::decay_t<decltype(patch.layout)>::dimension;
                auto const cent    = core::centering<dim>(f);
                auto const was_old = [&](auto const& amr) { // a node of the old level's domain
                    return std::any_of(old_cells.begin(), old_cells.end(), [&](auto const& b) {
                        for (std::size_t d = 0; d < dim; ++d)
                            if (amr[d] < b.lower[d]
                                or amr[d] > b.upper[d] + (cent[d] == core::QtyCentering::primal))
                                return false;
                        return true;
                    });
                };
                damn::core::for_each_point(domain_box(patch.layout, f), [&](auto const& ijk) {
                    if (!was_old(core::to_amr(patch.layout, ijk)))
                        return;
                    auto& v = f(ijk);
                    v       = values[static_cast<std::size_t>(&v - f.data())];
                });
            });
        });
        ++i;
    });
    level.copy(damn::amr::fill_ghosts, "IonPopMoments");
    compute_moments(level);
}

} // namespace plamn::solver::hybrid

#endif // PLAMN_SOLVER_HYBRID_INITIALIZE_HPP
