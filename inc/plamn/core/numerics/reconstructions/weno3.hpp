#ifndef PLAMN_CORE_NUMERICS_RECONSTRUCTIONS_WENO3_HPP
#define PLAMN_CORE_NUMERICS_RECONSTRUCTIONS_WENO3_HPP

// third order WENO reconstruction, a fixed operation order (bit equal comparisons:
//  etc/compare/phare): the left and right states at face i from cells i-2 .. i+1

#include "damn/core/def.hpp"
#include "plamn/core/numerics/reconstructions/stencil.hpp"
#include "damn/core/def/damn_config.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace plamn::core
{
template<typename SlopeLimiter = void>
struct WENO3Reconstruction
{
    static constexpr std::uint32_t nghosts = 2;

    template<std::size_t dir>
    MKN_KUL_FN_SIG_RET static std::array<double, 2> reconstruct(auto const& F,
                                                                auto const& ijk) DAMN_ALL
    {
        return from_values(reconstruction::values<dir, -2, 1>(F, ijk));
    }

    template<std::size_t dir, auto src, auto dst>
    MKN_KUL_FN_SIG_RET static std::array<double, 2> center_reconstruct(auto const& U,
                                                                       auto const& ijk) DAMN_ALL
    {
        return from_values(reconstruction::centred_values<dir, -2, 1, src, dst>(U, ijk));
    }

    // v: cells i-2 .. i+1
    MKN_KUL_FN_SIG_RET static std::array<double, 2>
    from_values(std::array<double, 4> const& v) DAMN_ALL
    {
        auto const [u_2, u_1, u, u1] = v;
        return {left_(u_2, u_1, u), right_(u_1, u, u1)};
    }

private:
    MKN_KUL_FN_SIG_RET static double left_(double const ul, double const u,
                                           double const ur) DAMN_ALL
    {
        auto constexpr dL0    = 1. / 3.;
        auto constexpr dL1    = 2. / 3.;
        auto const [wL0, wL1] = weights_(ul, u, ur, dL0, dL1);
        return wL0 * (-0.5 * ul + 1.5 * u) + wL1 * (0.5 * u + 0.5 * ur);
    }

    MKN_KUL_FN_SIG_RET static double right_(double const ul, double const u,
                                            double const ur) DAMN_ALL
    {
        auto constexpr dR0    = 2. / 3.;
        auto constexpr dR1    = 1. / 3.;
        auto const [wR0, wR1] = weights_(ul, u, ur, dR0, dR1);
        return wR0 * (0.5 * u + 0.5 * ul) + wR1 * (-0.5 * ur + 1.5 * u);
    }

    MKN_KUL_FN_SIG_RET static std::array<double, 2> weights_(double const ul, double const u,
                                                             double const ur, double const d0,
                                                             double const d1) DAMN_ALL
    {
        auto constexpr eps   = 1.e-6;
        auto const beta0     = (u - ul) * (u - ul);
        auto const beta1     = (ur - u) * (ur - u);
        auto const alpha0    = d0 / ((beta0 + eps) * (beta0 + eps));
        auto const alpha1    = d1 / ((beta1 + eps) * (beta1 + eps));
        auto const sum_alpha = alpha0 + alpha1;
        return {alpha0 / sum_alpha, alpha1 / sum_alpha};
    }
};

} // namespace plamn::core

#endif /* PLAMN_CORE_NUMERICS_RECONSTRUCTIONS_WENO3_HPP */
