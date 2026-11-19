#ifndef PLAMN_CORE_NUMERICS_RECONSTRUCTIONS_CONSTANT_HPP
#define PLAMN_CORE_NUMERICS_RECONSTRUCTIONS_CONSTANT_HPP

// piecewise constant reconstruction, PHARE's core/numerics/reconstructions/constant.hpp: the
//  left and right states at face i are the values of cells i-1 and i

#include "damn/core/def.hpp"
#include "plamn/core/numerics/reconstructions/stencil.hpp"
#include "damn/core/def/damn_config.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace plamn::core
{
template<typename SlopeLimiter = void>
struct ConstantReconstruction
{
    static constexpr std::uint32_t nghosts = 1;

    template<std::size_t dir>
    NO_DISCARD static std::array<double, 2> reconstruct(auto const& F, auto const& ijk) DAMN_ALL
    {
        return reconstruction::values<dir, -1, 0>(F, ijk);
    }

    template<std::size_t dir, auto src, auto dst>
    NO_DISCARD static std::array<double, 2> center_reconstruct(auto const& U,
                                                               auto const& ijk) DAMN_ALL
    {
        return reconstruction::centred_values<dir, -1, 0, src, dst>(U, ijk);
    }
};

} // namespace plamn::core

#endif /* PLAMN_CORE_NUMERICS_RECONSTRUCTIONS_CONSTANT_HPP */
