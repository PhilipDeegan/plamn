#ifndef PLAMN_CORE_NUMERICS_RECONSTRUCTIONS_STENCIL_HPP
#define PLAMN_CORE_NUMERICS_RECONSTRUCTIONS_STENCIL_HPP

// the cell values along a direction around a face, as the reconstructions read them: face i of
//  direction dir lies between cells i-1 and i
//   values<dir, from, to>(F, ijk)                        F at cells i+from .. i+to
//   centred_values<dir, from, to, src, dst>(U, ijk)      the same for U, a src quantity projected
//                                                        to dst (the cell centre) first

#include "damn/core/def.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"
#include "damn/core/def/damn_config.hpp"

#include <array>
#include <cstddef>
#include <utility>

namespace plamn::core::reconstruction
{
template<std::size_t dir>
MKN_KUL_FN_SIG_RET auto shifted(auto ijk, int const shift) DAMN_ALL
{
    ijk[dir] += shift;
    return ijk;
}

template<std::size_t dir, int from, int to>
MKN_KUL_FN_SIG_RET std::array<double, to - from + 1> values(auto const& F, auto const& ijk) DAMN_ALL
{
    return [&]<std::size_t... i>(std::index_sequence<i...>) {
        return std::array<double, to - from + 1>{
            F(shifted<dir>(ijk, from + static_cast<int>(i)))...};
    }(std::make_index_sequence<to - from + 1>{});
}

template<std::size_t dir, int from, int to, auto src, auto dst>
MKN_KUL_FN_SIG_RET std::array<double, to - from + 1> centred_values(auto const& U,
                                                                    auto const& ijk) DAMN_ALL
{
    return [&]<std::size_t... i>(std::index_sequence<i...>) {
        return std::array<double, to - from + 1>{
            project<src, dst>(U, shifted<dir>(ijk, from + static_cast<int>(i)))...};
    }(std::make_index_sequence<to - from + 1>{});
}

} // namespace plamn::core::reconstruction

#endif /* PLAMN_CORE_NUMERICS_RECONSTRUCTIONS_STENCIL_HPP */
