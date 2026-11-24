#ifndef PLAMN_SOLVER_HYBRID_MODEL_HPP
#define PLAMN_SOLVER_HYBRID_MODEL_HPP

// the hybrid model of a simulation: its patch data per level number, for the levels it is active
//  on (the hierarchy's geometry, neighbours and maps between levels are the simulator's, shared
//  by the models). Empty if the build does not enable it

#include "plamn/amr/field_access.hpp"    // IWYU pragma: keep (a member's type)
#include "plamn/amr/particle_access.hpp" // IWYU pragma: keep (a member's type)
#include "damn/amr/level_data.hpp"
#include "damn/core/def/constants.hpp"
#include "plamn/amr/split/split.hpp" // IWYU pragma: keep (the Splitter specializations)
#include "plamn/amr/split/splitter.hpp"
#include "plamn/core/data/electrons/closures.hpp"
#include "plamn/core/numerics/ohm/ohm.hpp"
#include "plamn/diagnostic/quantities.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"
#include "plamn/simulator/hybrid_patch.hpp"
#include "plamn/simulator/hybrid_patch_view.hpp"
#include "plamn/solver/computed.hpp"
#include "plamn/solver/hybrid/level_boxings.hpp" // IWYU pragma: keep (a member's type)
#include "plamn/solver/model_def.hpp"

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <format>


namespace plamn::solver
{
// ctx: the context of the levels it owns (simulator/level_context.hpp): their patches' storage
template<auto opts, simulator::LevelContext ctx = simulator::LevelContext{},
         bool enabled = has_hybrid_v<opts>>
struct HybridModel
{
};

template<auto opts, simulator::LevelContext ctx>
struct HybridModel<opts, ctx, true>
{
    auto static constexpr context = ctx;
    using Patch_t     = simulator::HybridPatch<opts, ctx>;
    using LevelData_t = damn::amr::LevelData<opts.dimension, std::unique_ptr<Patch_t>>;
    using Mover_t     = Patch_t::ParticlesSynchronisedMover_t;
    using Splitter_t  = amr::Splitter<damn::core::DimConst<opts.dimension>,
                                      damn::core::InterpConst<opts.interp_order>,
                                      amr::RefinedParticlesConst<opts.nbRefinedPart>>;

    std::map<int, LevelData_t> levels;
    // damn's field copies by accessors, per resource key: resolved once (Simulator::copy_)
    std::map<std::string, std::vector<amr::FieldAccess<Patch_t::field_ghost_width>>> field_access{};
    std::optional<core::OhmInfo> ohm; // from the input, at setup
    // the electrons' pressure closure, from the input at setup (solver/hybrid/electron/
    //  closures.hpp)
    std::optional<core::hybrid::ElectronClosure> electron_closure;
    std::vector<std::string> population_names{};
    // the particle schedules' pairs and accessors, from population_names (Simulator setup)
    std::optional<amr::ParticleRoutes<Patch_t>> particle_routes{};
    // per level, what its push needs of its decomposition (level_boxings.hpp), kept until a
    //  regrid / rebalance
    std::map<int, LevelBoxings<typename Patch_t::Boxing_t, opts.alloc_mode>> level_boxings{};
    using Grid_t     = Patch_t::Grid_t;
    using Computed_t = Computed<Patch_t, Grid_t, core::HybridQuantity::Scalar>;
    std::map<std::string, Computed_t> computed{}; // by key (solver/hybrid/computed.hpp)
    // the computed quantities asked for so far: their scratch grids on every patch (made
    //  with them, make_patch), the others on none. The diagnostics' at setup, any other at its
    //  first compute (Simulator::compute: on every patch then, collective)
    std::set<std::string> requested{};
    using View_t = decltype(simulator::view(std::declval<Patch_t&>())); // what diagnostics read
    std::map<std::string, diagnostic::Accessor<View_t>> quantities{};   // diagnostics' fields
    Mover_t mover{};

    // the levels from n on gone
    void truncate(std::size_t const n)
    {
        levels.erase(levels.lower_bound(static_cast<int>(n)), levels.end());
    }
};

} // namespace plamn::solver

#endif // PLAMN_SOLVER_HYBRID_MODEL_HPP
