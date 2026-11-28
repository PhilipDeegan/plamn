#ifndef PLAMN_CORE_NUMERICS_RECONSTRUCTIONS_LINEAR_HPP
#define PLAMN_CORE_NUMERICS_RECONSTRUCTIONS_LINEAR_HPP

#include "damn/core/def/damn_config.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"
#include "damn/core/def.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

// piecewise linear reconstruction with a slope limiter: the left and right states at face i of
//  direction dir, from the cell values i-2 .. i+1 along dir (face i lies between cells i-1 and i)
namespace plamn::core
{

struct VanLeerLimiter
{
    MKN_KUL_FN_SIG_RET static constexpr double limit(double const dl, double const dr) DAMN_ALL
    {
        return dl * dr > 0.0 ? 2.0 * dl * dr / (dl + dr) : 0.0;
    }
};

struct MinModLimiter
{
    MKN_KUL_FN_SIG_RET static constexpr double limit(double const dl, double const dr) DAMN_ALL
    {
        if (dl * dr <= 0.0)
            return 0.0;
        return (dl > 0.0 ? 1.0 : -1.0)
               * (dl * dl < dr * dr ? (dl > 0 ? dl : -dl) : (dr > 0 ? dr : -dr));
    }
};

template<typename SlopeLimiter = VanLeerLimiter>
struct LinearReconstruction
{
    static constexpr std::uint32_t nghosts = 2;

    // {left, right} states at face ijk of direction dir, from the cell centred field F
    template<std::size_t dir>
    MKN_KUL_FN_SIG_RET static std::array<double, 2> reconstruct(auto const& F,
                                                                auto const& ijk) DAMN_ALL
    {
        return from_values(at<dir, -2>(F, ijk), at<dir, -1>(F, ijk), F(ijk), at<dir, 1>(F, ijk));
    }

    // the same for U, a src quantity, projected to dst (the cell centre) first
    template<std::size_t dir, auto src, auto dst>
    MKN_KUL_FN_SIG_RET static std::array<double, 2> center_reconstruct(auto const& U,
                                                                       auto const& ijk) DAMN_ALL
    {
        auto const p = [&]<int shift>() { return project<src, dst>(U, shifted<dir, shift>(ijk)); };
        return from_values(p.template operator()<-2>(), p.template operator()<-1>(),
                           p.template operator()<0>(), p.template operator()<1>());
    }

    // u_2 .. u1: the values of cells i-2 .. i+1
    MKN_KUL_FN_SIG_RET static constexpr std::array<double, 2>
    from_values(double const u_2, double const u_1, double const u, double const u1) DAMN_ALL
    {
        return {u_1 + 0.5 * slope(u_2, u_1, u), u - 0.5 * slope(u_1, u, u1)};
    }

    MKN_KUL_FN_SIG_RET static constexpr double slope(double const ul, double const u,
                                                     double const ur) DAMN_ALL
    {
        return SlopeLimiter::limit(u - ul, ur - u);
    }

    template<std::size_t dir, int shift>
    MKN_KUL_FN_SIG_RET static auto shifted(auto ijk) DAMN_ALL
    {
        ijk[dir] += shift;
        return ijk;
    }

    template<std::size_t dir, int shift>
    MKN_KUL_FN_SIG_RET static double at(auto const& F, auto const& ijk) DAMN_ALL
    {
        return F(shifted<dir, shift>(ijk));
    }
};

} // namespace plamn::core

#endif /* PLAMN_CORE_NUMERICS_RECONSTRUCTIONS_LINEAR_HPP */
