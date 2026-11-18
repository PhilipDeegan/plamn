#ifndef PLAMN_SOLVER_HYBRID_MODEL_HPP
#define PLAMN_SOLVER_HYBRID_MODEL_HPP

// the hybrid model of a simulation: its patch data per level number, for the levels it is active
//  on (the hierarchy's geometry, neighbours and maps between levels are the simulator's, shared
//  by the models). Empty if the build does not enable it

#include "damn/amr/level_data.hpp"
#include "damn/core/def/constants.hpp"

#include "plamn/amr/split/split.hpp" // IWYU pragma: keep (the Splitter specializations)
#include "plamn/amr/split/splitter.hpp"
#include "plamn/core/numerics/ohm/ohm.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"
#include "plamn/simulator/hybrid_patch.hpp"
#include "plamn/simulator/hybrid_patch_view.hpp"
#include "plamn/solver/computed.hpp"
#include "plamn/solver/model_def.hpp"

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace plamn::solver
{
template<auto opts, bool enabled = has_hybrid_v<opts>>
struct HybridModel
{
};

template<auto opts>
struct HybridModel<opts, true>
{
    using Patch_t     = simulator::HybridPatch<opts>;
    using LevelData_t = damn::amr::LevelData<opts.dimension, std::unique_ptr<Patch_t>>;
    using Mover_t     = Patch_t::ParticlesSynchronisedMover_t;
    using Splitter_t  = amr::Splitter<damn::core::DimConst<opts.dimension>,
                                      damn::core::InterpConst<opts.interp_order>,
                                      amr::RefinedParticlesConst<opts.nbRefinedPart>>;

    std::map<int, LevelData_t> levels;
    std::optional<core::OhmInfo> ohm; // from the input, at setup
    std::vector<std::string> population_names{};
    using Grid_t     = Patch_t::Grid_t;
    using Computed_t = Computed<Patch_t, Grid_t, core::HybridQuantity::Scalar>;
    std::map<std::string, Computed_t> computed{}; // by key (solver/hybrid/computed.hpp)
    using View_t = decltype(simulator::view(std::declval<Patch_t&>())); // what diagnostics read
    Mover_t mover{};

    // the levels from n on gone
    void truncate(std::size_t const n)
    {
        levels.erase(levels.lower_bound(static_cast<int>(n)), levels.end());
    }
};

} // namespace plamn::solver

#endif // PLAMN_SOLVER_HYBRID_MODEL_HPP
