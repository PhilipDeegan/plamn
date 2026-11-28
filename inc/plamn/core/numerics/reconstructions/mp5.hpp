#ifndef PLAMN_CORE_NUMERICS_RECONSTRUCTIONS_MP5_HPP
#define PLAMN_CORE_NUMERICS_RECONSTRUCTIONS_MP5_HPP

// fifth order monotonicity preserving reconstruction (Suresh and Huynh 1997), a fixed
//  operation order (bit equal comparisons: etc/compare/phare): the left and right states at face i
//  from cells i-3 .. i+2

#include "damn/core/def.hpp"
#include "plamn/core/numerics/reconstructions/stencil.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>

namespace plamn::core
{
// the min mod limiter over any number of values: 0 unless all have one sign, else the
//  one of least magnitude (the first of them)
template<typename... Rest>
MKN_KUL_FN_SIG_RET double min_mod(double const first, Rest const... rest)
{
    bool const all_positive = (first > 0) && ((rest > 0) && ...);
    bool const all_negative = (first < 0) && ((rest < 0) && ...);
    if (!all_positive && !all_negative)
        return 0.;
    return std::min({first, static_cast<double>(rest)...},
                    [](auto const a, auto const b) { return std::abs(a) < std::abs(b); });
}

template<typename SlopeLimiter = void>
struct MP5Reconstruction
{
    static constexpr std::uint32_t nghosts = 3;

    template<std::size_t dir>
    MKN_KUL_FN_SIG_RET static std::array<double, 2> reconstruct(auto const& F, auto const& ijk)
    {
        return from_values(reconstruction::values<dir, -3, 2>(F, ijk));
    }

    template<std::size_t dir, auto src, auto dst>
    MKN_KUL_FN_SIG_RET static std::array<double, 2> center_reconstruct(auto const& U,
                                                                       auto const& ijk)
    {
        return from_values(reconstruction::centred_values<dir, -3, 2, src, dst>(U, ijk));
    }

    // v: cells i-3 .. i+2
    MKN_KUL_FN_SIG_RET static std::array<double, 2> from_values(std::array<double, 6> const& v)
    {
        auto const [u_3, u_2, u_1, u, u1, u2] = v;
        return {impl_(u_3, u_2, u_1, u, u1), impl_(u2, u1, u, u_1, u_2)};
    }

private:
    MKN_KUL_FN_SIG_RET static double impl_(double const v_m2, double const v_m1, double const u,
                                           double const v_p1, double const v_p2)
    {
        auto constexpr alpha = 4.;
        auto const fi1_2     = (2. * v_m2 - 13. * v_m1 + 47. * u + 27. * v_p1 - 3. * v_p2) / 60.;
        auto const Dil       = u - v_m1;
        auto const Dir       = v_p1 - u;
        auto const fMP       = u + min_mod(Dir, alpha * Dil);
        auto const fUL       = u + alpha * Dil;
        auto const di        = Dir - Dil;
        auto const dir       = (v_p2 - v_p1) - Dir;
        auto const d4i1_2    = min_mod(4. * di - dir, 4. * dir - di, di, dir);
        auto const fMD       = (u + v_p1) / 2. - d4i1_2 / 2.;
        auto const dil       = Dil - (v_m1 - v_m2);
        auto const d4im1_2   = min_mod(4. * dil - di, 4. * di - dil, dil, di);
        auto const fLC       = u + Dil / 2. + (4. / 3.) * d4im1_2;
        auto const fmin      = std::max(std::min({u, v_p1, fMD}), std::min({u, fUL, fLC}));
        auto const fmax      = std::min(std::max({u, v_p1, fMD}), std::max({u, fUL, fLC}));
        return (fi1_2 - u) * (fi1_2 - fMP) < 0.0 ? fi1_2 : std::clamp(fi1_2, fmin, fmax);
    }
};

} // namespace plamn::core

#endif /* PLAMN_CORE_NUMERICS_RECONSTRUCTIONS_MP5_HPP */
