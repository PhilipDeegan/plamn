#ifndef PLAMN_SIMULATOR_LEVEL_CONTEXT_HPP
#define PLAMN_SIMULATOR_LEVEL_CONTEXT_HPP

// LevelContext: what a level's neighbours run, as far as its patches' storage cares, fixed by the
//  level's index (the input's max_mhd_level: decided at setup, unchanged by regrids). A patch
//  type takes it as a template argument and holds what its context needs (HybridStorage,
//  MHDStorage: empty members otherwise), so a level without such a neighbour pays nothing
//
//   coarser   the model of the level below: hybrid over MHD (the first hybrid level, X + 1)
//             holds what the hybrid -> MHD synchronization needs
//   finer     the model a finer level would be (not whether one exists now): MHD under hybrid
//             (the finest MHD level, X) holds what the hybrid level's ghosts read
//
// The default, {}, is a level whose neighbours run its own model (or none): every level of a
//  one model run. Code between two levels of one model takes any two contexts
//  (HybridPatch<opts, A>, HybridPatch<opts, B>): what differs is storage only

#include <cstdint>

namespace plamn::simulator
{
enum class LevelModel : std::uint8_t { same, mhd, hybrid };

struct LevelContext
{
    LevelModel coarser = LevelModel::same;
    LevelModel finer   = LevelModel::same;

    friend constexpr bool operator==(LevelContext const&, LevelContext const&) = default;
};

inline constexpr LevelContext hybrid_over_mhd{.coarser = LevelModel::mhd};
inline constexpr LevelContext mhd_under_hybrid{.finer = LevelModel::hybrid};

} // namespace plamn::simulator

#endif // PLAMN_SIMULATOR_LEVEL_CONTEXT_HPP
