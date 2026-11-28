#ifndef PLAMN_CORE_NUMERICS_GODUNOV_HLL_HPP
#define PLAMN_CORE_NUMERICS_GODUNOV_HLL_HPP

// HLL flux at one face (a fixed operation order: bit equal comparisons, etc/compare/phare),
//  ideal MHD (Hall: not yet): the outermost wave speeds SL, SR (the fast magnetosonic one either
//  side), the flux of the state between them; the upwind constrained transport inputs from the
//  speeds clipped at 0 (Mignone & Del Zanna 2021, eqs 14, 32)

#include "damn/core/def.hpp"
#include "plamn/core/numerics/godunov/mhd_equations.hpp"
#include "plamn/core/numerics/godunov/mhd_point_state.hpp"
#include "plamn/core/numerics/godunov/rusanov.hpp"
#include "damn/core/def/damn_config.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

namespace plamn::core
{
template<std::size_t dir>
MKN_KUL_FN_SIG_RET GodunovFaceResult hll(double const gamma, MHDPrimitive const& uL,
                                         MHDPrimitive const& uR,
                                         FaceCurrents const* const j = nullptr) DAMN_ALL
{
    auto const b2   = [](auto const& B) { return B[0] * B[0] + B[1] * B[1] + B[2] * B[2]; };
    auto const cfL  = fast_magnetosonic(gamma, uL.rho, uL.B[dir], b2(uL.B), uL.P);
    auto const cfR  = fast_magnetosonic(gamma, uR.rho, uR.B[dir], b2(uR.B), uR.P);
    auto const VL   = uL.V[dir];
    auto const VR   = uR.V[dir];
    auto const SL   = std::min(VL - cfL, VR - cfR);
    auto const SR   = std::max(VL + cfL, VR + cfR);
    auto const fL   = j ? hall_mhd_flux<dir>(gamma, uL, j->L) : ideal_mhd_flux<dir>(gamma, uL);
    auto const fR   = j ? hall_mhd_flux<dir>(gamma, uR, j->R) : ideal_mhd_flux<dir>(gamma, uR);
    auto const cL   = to_conservative(gamma, uL);
    auto const cR   = to_conservative(gamma, uR);
    auto const flux = [&](double const ul, double const ur, double const fl, double const fr) {
        if (SL > 0.0)
            return fl;
        if (SR < 0.0)
            return fr;
        return (SR * fl - SL * fr + SL * SR * (ur - ul)) / (SR - SL);
    };

    GodunovFaceResult r{};
    r.F.rho = flux(cL.rho, cR.rho, fL.rho, fR.rho);
    for (std::size_t c = 0; c < 3; ++c)
    {
        r.F.rhoV[c] = flux(cL.rhoV[c], cR.rhoV[c], fL.rhoV[c], fR.rhoV[c]);
        r.F.B[c]    = flux(cL.B[c], cR.B[c], fL.B[c], fR.B[c]);
    }
    r.F.Etot = flux(cL.Etot, cR.Etot, fL.Etot, fR.Etot);

    // the speeds clipped at 0: a supersonic face takes the upwind state
    auto const sl  = std::min(0.0, SL);
    auto const sr  = std::max(0.0, SR);
    auto const inv = 1.0 / (sr - sl);
    auto const d   = -sr * sl * inv;
    r.uct_coefs    = {sr * inv, -sl * inv, d, d};
    for (std::size_t c = 0; c < 3; ++c)
        r.vt[c] = (sr * uL.V[c] - sl * uR.V[c]) * inv;
    if (j) // the Riemann averages with the clipped speeds
    {
        for (std::size_t c = 0; c < 3; ++c)
            r.jt[c] = (sr * j->L[c] - sl * j->R[c]) * inv;
        r.rhot = (sr * uL.rho - sl * uR.rho) * inv;
    }
    return r;
}

struct RusanovRiemann
{
    template<std::size_t dir>
    MKN_KUL_FN_SIG_RET static GodunovFaceResult
    solve(double const gamma, MHDPrimitive const& uL, MHDPrimitive const& uR,
          FaceCurrents const* const j = nullptr) DAMN_ALL
    {
        return rusanov<dir>(gamma, uL, uR, j);
    }
};

struct HLLRiemann
{
    template<std::size_t dir>
    MKN_KUL_FN_SIG_RET static GodunovFaceResult
    solve(double const gamma, MHDPrimitive const& uL, MHDPrimitive const& uR,
          FaceCurrents const* const j = nullptr) DAMN_ALL
    {
        return hll<dir>(gamma, uL, uR, j);
    }
};

} // namespace plamn::core

#endif /* PLAMN_CORE_NUMERICS_GODUNOV_HLL_HPP */
