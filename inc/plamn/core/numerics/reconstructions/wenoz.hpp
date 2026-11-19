#ifndef PLAMN_CORE_NUMERICS_RECONSTRUCTIONS_WENOZ_HPP
#define PLAMN_CORE_NUMERICS_RECONSTRUCTIONS_WENOZ_HPP

// fifth order WENO-Z reconstruction, PHARE's core/numerics/reconstructions/wenoz.hpp (its
//  operation order: the same results): the left and right states at face i from cells
//  i-3 .. i+2

#include "damn/core/def.hpp"
#include "plamn/core/numerics/reconstructions/stencil.hpp"
#include "damn/core/def/damn_config.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>

namespace plamn::core
{
template<typename SlopeLimiter = void>
struct WENOZReconstruction
{
    static constexpr std::uint32_t nghosts = 3;

    template<std::size_t dir>
    NO_DISCARD static std::array<double, 2> reconstruct(auto const& F, auto const& ijk) DAMN_ALL
    {
        return from_values(reconstruction::values<dir, -3, 2>(F, ijk));
    }

    template<std::size_t dir, auto src, auto dst>
    NO_DISCARD static std::array<double, 2> center_reconstruct(auto const& U,
                                                               auto const& ijk) DAMN_ALL
    {
        return from_values(reconstruction::centred_values<dir, -3, 2, src, dst>(U, ijk));
    }

    // v: cells i-3 .. i+2
    NO_DISCARD static std::array<double, 2> from_values(std::array<double, 6> const& v) DAMN_ALL
    {
        auto const [u_3, u_2, u_1, u, u1, u2] = v;
        return {left_(u_3, u_2, u_1, u, u1), right_(u_2, u_1, u, u1, u2)};
    }

private:
    NO_DISCARD static double left_(double const ull, double const ul, double const u,
                                   double const ur, double const urr) DAMN_ALL
    {
        auto constexpr dL0         = 1. / 10.;
        auto constexpr dL1         = 3. / 5.;
        auto constexpr dL2         = 3. / 10.;
        auto const [wL0, wL1, wL2] = weights_(ull, ul, u, ur, urr, dL0, dL1, dL2);
        return wL0 * ((1. / 3.) * ull - (7. / 6.) * ul + (11. / 6.) * u)
               + wL1 * (-(1. / 6.) * ul + (5. / 6.) * u + (1. / 3.) * ur)
               + wL2 * ((1. / 3.) * u + (5. / 6.) * ur - (1. / 6.) * urr);
    }

    NO_DISCARD static double right_(double const ull, double const ul, double const u,
                                    double const ur, double const urr) DAMN_ALL
    {
        auto constexpr dR0         = 3. / 10.;
        auto constexpr dR1         = 3. / 5.;
        auto constexpr dR2         = 1. / 10.;
        auto const [wR0, wR1, wR2] = weights_(ull, ul, u, ur, urr, dR0, dR1, dR2);
        return wR0 * ((1. / 3.) * u + (5. / 6.) * ul - (1. / 6.) * ull)
               + wR1 * (-(1. / 6.) * ur + (5. / 6.) * u + (1. / 3.) * ul)
               + wR2 * ((1. / 3.) * urr - (7. / 6.) * ur + (11. / 6.) * u);
    }

    NO_DISCARD static std::array<double, 3> weights_(double const ull, double const ul,
                                                     double const u, double const ur,
                                                     double const urr, double const d0,
                                                     double const d1, double const d2) DAMN_ALL
    {
        auto constexpr eps   = 1.e-40;
        auto const beta0     = (13. / 12.) * (ull - 2. * ul + u) * (ull - 2. * ul + u)
                               + (1. / 4.) * (ull - 4. * ul + 3. * u) * (ull - 4. * ul + 3. * u);
        auto const beta1     = (13. / 12.) * (ul - 2. * u + ur) * (ul - 2. * u + ur)
                               + (1. / 4.) * (ul - ur) * (ul - ur);
        auto const beta2     = (13. / 12.) * (u - 2. * ur + urr) * (u - 2. * ur + urr)
                               + (1. / 4.) * (3. * u - 4. * ur + urr) * (3. * u - 4. * ur + urr);
        auto const tau5      = std::abs(beta0 - beta2);
        auto const alpha0    = d0 * (1. + tau5 / (beta0 + eps));
        auto const alpha1    = d1 * (1. + tau5 / (beta1 + eps));
        auto const alpha2    = d2 * (1. + tau5 / (beta2 + eps));
        auto const sum_alpha = alpha0 + alpha1 + alpha2;
        return {alpha0 / sum_alpha, alpha1 / sum_alpha, alpha2 / sum_alpha};
    }
};

} // namespace plamn::core

#endif /* PLAMN_CORE_NUMERICS_RECONSTRUCTIONS_WENOZ_HPP */
