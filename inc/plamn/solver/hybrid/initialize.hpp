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
#include "plamn/amr/level_pairs.hpp"
#include "plamn/amr/refine.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/models/electromag_initializer.hpp"
#include "plamn/core/numerics/ampere/ampere_transformer.hpp"
#include "plamn/core/numerics/interpolator/interpolating.hpp"
#include "plamn/core/numerics/moments/moments.hpp"
#include "plamn/core/numerics/particles_synchronised_mover/particles_synchronised_mover.hpp"
#include "plamn/solver/hybrid/advance.hpp"
#include "plamn/solver/hybrid/level_steps.hpp"
#include "plamn/solver/hybrid/model.hpp"

#include <cstddef>
#include <string>

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
        core::depositParticles(patch.ions, patch.layout,
                               core::Interpolating<opts.dimension, opts.interp_order>{},
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
        core::depositParticles(patch.ions, patch.layout,
                               core::Interpolating<opts.dimension, opts.interp_order>{},
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

    for (auto const& name : model.population_names)
    {
        level.refine(amr::ParticleSplitter<Splitter_t>{}, LevelRegion::Interior,
                     {{name + "_domain", name + "_domain", ""}});
        level.refine(amr::ParticleSplitter<Splitter_t>{}, LevelRegion::LevelGhosts,
                     {{name + "_domain", name + "_level_ghost_old", ""}});
    }
    own_level_ghosts(level, /*new_ghosts=*/false);
    level.per_patch([](auto& patch) {
        for (auto& pop : patch.ions)
            pop.levelGhostParticles() = pop.levelGhostParticlesOld();
    });
    sort_refined_particles(model, level);

    refined_moments(model, level);
}

} // namespace plamn::solver::hybrid

#endif // PLAMN_SOLVER_HYBRID_INITIALIZE_HPP
