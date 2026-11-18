#ifndef PLAMN_SOLVER_HYBRID_MODEL_HPP
#define PLAMN_SOLVER_HYBRID_MODEL_HPP

// the hybrid model of a simulation: its patch data per level number, for the levels it is active
//  on (the hierarchy's geometry, neighbours and maps between levels are the simulator's, shared
//  by the models)

#include "damn/amr/level_data.hpp"

#include "plamn/simulator/hybrid_patch.hpp"

#include <cstddef>
#include <map>
#include <memory>

namespace plamn::solver
{
template<auto opts>
struct HybridModel
{
    static_assert(opts.hybrid_enabled);
    using Patch_t     = simulator::HybridPatch<opts>;
    using LevelData_t = damn::amr::LevelData<opts.dimension, std::unique_ptr<Patch_t>>;

    std::map<int, LevelData_t> levels;

    // the levels from n on gone
    void truncate(std::size_t const n)
    {
        levels.erase(levels.lower_bound(static_cast<int>(n)), levels.end());
    }
};

} // namespace plamn::solver

#endif // PLAMN_SOLVER_HYBRID_MODEL_HPP
