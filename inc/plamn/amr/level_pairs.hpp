#ifndef PLAMN_AMR_LEVEL_PAIRS_HPP
#define PLAMN_AMR_LEVEL_PAIRS_HPP

#include "damn/amr/level_map.hpp"
#include "damn/core/def.hpp"

#include <deque>
#include <string>
#include <vector>
#include <initializer_list>
#include <format>

namespace plamn::amr
{
// the pairs of a vector's components between levels: coarse (and coarse_new, if not
//  empty) on the coarser level, fine on the finer
MKN_KUL_FN_SIG_RET inline std::vector<damn::amr::LevelPair>
vector_pairs(std::string const& coarse, std::string const& fine, std::string const& coarse_new = "")
{
    std::vector<damn::amr::LevelPair> pairs;
    for (auto const* c : {"_x", "_y", "_z"})
        pairs.push_back({coarse + c, fine + c, coarse_new.empty() ? "" : coarse_new + c});
    return pairs;
}

// vector_pairs, built once per thread and names: the ghost fills of every step ask for the same
//  few. A deque: what it returned stays valid as it grows
MKN_KUL_FN_SIG_RET inline std::vector<damn::amr::LevelPair> const&
kept_vector_pairs(std::string const& coarse, std::string const& fine,
                  std::string const& coarse_new = "")
{
    struct Kept
    {
        std::string coarse, fine, coarse_new;
        std::vector<damn::amr::LevelPair> pairs;
    };
    static thread_local std::deque<Kept> kept;
    for (auto const& k : kept)
        if (k.coarse == coarse and k.fine == fine and k.coarse_new == coarse_new)
            return k.pairs;
    return kept.emplace_back(coarse, fine, coarse_new, vector_pairs(coarse, fine, coarse_new))
        .pairs;
}

} // namespace plamn::amr

#endif // PLAMN_AMR_LEVEL_PAIRS_HPP
