#ifndef PLAMN_SOLVER_HYBRID_RESTARTS_HPP
#define PLAMN_SOLVER_HYBRID_RESTARTS_HPP

// what a restart holds of a hybrid level, and what the level rebuilds from it (the simulator
//  writes and reads the levels' resources by these names, simulator_restarts.hpp)

#include "damn/amr/level_map.hpp"
#include "damn/core/def.hpp"

#include "plamn/amr/refine.hpp"
#include "plamn/core/data/field/field_tiles.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/numerics/ampere/ampere_transformer.hpp"
#include "plamn/solver/hybrid/level_steps.hpp"
#include "plamn/solver/hybrid/model.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>
#include <format>
#include <initializer_list>

namespace plamn::solver::hybrid
{
// the resources a restart stores
template<auto opts>
NO_DISCARD std::vector<std::string> restart_names(HybridModel<opts> const& model)
{
    std::vector<std::string> names;
    for (auto const* em : {"EM_B", "EM_E"})
        for (auto const* c : {"_x", "_y", "_z"})
            names.push_back(std::string{em} + c);
    // the totals a synchronization coarsened: not the populations' sum on a coarser level
    names.push_back("ions_charge_density");
    for (auto const* c : {"_x", "_y", "_z"})
        names.push_back(std::string{"ions_charge_flux"} + c);
    for (auto const& pop : model.population_names)
    {
        names.push_back(pop + "_domain");
        for (auto const* m : {"_density", "_charge_density", "_flux_x", "_flux_y", "_flux_z"})
            names.push_back(pop + m);
    }
    return names;
}

// around a restart's writing (writing: true before, false after). SoAVXChunked: the particles
//  written without the domains' gaps, laid out again after, as loading does (the same memory
//  order then on both sides of a restart)
template<auto opts>
void restart_writing(HybridModel<opts> const&, auto&& level, bool const writing)
{
    using Particles_t = HybridModel<opts>::Patch_t::Particles_t;
    if constexpr (Particles_t::layout_mode == core::LayoutMode::SoAVXChunked)
        level.per_patch([&](auto& patch) {
            for (auto& pop : patch.ions)
                writing ? pop.domainParticles().compact() : pop.domainParticles().sort();
        });
}

// what the running level holds between steps, from what was stored: the particles' maps, the
//  level ghost particles (split from the coarser level, as first_step does), the mass density and
//  bulk velocity (from the populations'), J (from B)
template<auto opts>
void restart_loaded(HybridModel<opts> const& model, auto&& level)
{
    using damn::amr::LevelRegion;
    using Splitter_t = HybridModel<opts>::Splitter_t;
    level.per_patch([](auto& patch) {
        for (auto& pop : patch.ions)
        {
            if constexpr (requires { pop.domainParticles().remap(); })
                pop.domainParticles().remap();
            if constexpr (requires { pop.domainParticles().compact(); }) // SoAVXChunked
                pop.domainParticles().sort(); // the gaps as at initialization
        }
    });
    if (level.number() > 0)
    {
        for (auto const& name : model.population_names)
            level.refine(amr::ParticleSplitter<Splitter_t>{}, LevelRegion::LevelGhosts,
                         {{name + "_domain", name + "_level_ghost_old", ""}});
        own_level_ghosts(level, /*new_ghosts=*/false);
        level.per_patch([](auto& patch) {
            for (auto& pop : patch.ions)
                pop.levelGhostParticles() = pop.levelGhostParticlesOld();
        });
    }
    // the mass density and bulk velocity from the populations' (stored); the charge density and
    //  flux as stored (coarsened where a finer level was synchronized)
    level.per_patch([&](auto& patch) {
        std::vector<std::vector<double>> stored;
        auto const each = [&](auto&& fn) {
            fn(patch.chargeDensity);
            for (auto& c : patch.chargeFlux)
                fn(c);
        };
        each([&](auto const& g) {
            auto& values = stored.emplace_back();
            core::for_each_buffer(g, [&](auto const* data, auto const size) {
                values.insert(values.end(), data, data + size);
            });
        });
        patch.ions.update();
        std::size_t i = 0;
        each([&](auto& g) {
            auto const* from = stored[i++].data();
            core::for_each_buffer(g, [&](auto* data, auto const size) {
                std::copy(from, from + size, data);
                from += size;
            });
        });
    });
    level.per_patch(
        [&](auto& patch) { core::AmpereTransformer{}(patch.layout, patch.em.B, patch.J); });
}

} // namespace plamn::solver::hybrid

#endif // PLAMN_SOLVER_HYBRID_RESTARTS_HPP
