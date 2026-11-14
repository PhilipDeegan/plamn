#ifndef PLAMN_AMR_LEVEL_PAIRS_HPP
#define PLAMN_AMR_LEVEL_PAIRS_HPP

// vector_pairs: the damn::amr::LevelPair of each of a vector's components (_x, _y, _z)

#include "damn/amr/level_map.hpp"
#include "damn/core/def.hpp"

#include <string>
#include <vector>
#include <initializer_list>

namespace plamn::amr
{
// the pairs of a vector's components between levels: coarse (and coarse_new, if not
//  empty) on the coarser level, fine on the finer
NO_DISCARD inline std::vector<damn::amr::LevelPair>
vector_pairs(std::string const& coarse, std::string const& fine, std::string const& coarse_new = "")
{
    std::vector<damn::amr::LevelPair> pairs;
    for (auto const* c : {"_x", "_y", "_z"})
        pairs.push_back({coarse + c, fine + c, coarse_new.empty() ? "" : coarse_new + c});
    return pairs;
}

} // namespace plamn::amr

#endif // PLAMN_AMR_LEVEL_PAIRS_HPP
