#ifndef PLAMN_CORE_NUMERICS_GODUNOV_RUSANOV_HPP
#define PLAMN_CORE_NUMERICS_GODUNOV_RUSANOV_HPP

#include "damn/core/def/damn_config.hpp"
#include "plamn/core/numerics/godunov/mhd_equations.hpp"
#include "plamn/core/numerics/godunov/mhd_point_state.hpp"
#include "damn/core/def.hpp"

#include <array>
#include <cmath>
#include <cstdlib>

// Rusanov (local Lax-Friedrichs) flux at one face, PHARE's core/numerics/riemann_solvers/
//  rusanov.hpp and mhd_speeds.hpp, ideal MHD (Hall: not yet). PHARE's solver keeps the upwind
//  constrained transport inputs (vt, uct_coefs) as members after solve(): here they are returned
//  with the flux, so every face is independent (tiles, GPU threads)
namespace plamn::core
{

NO_DISCARD inline double fast_magnetosonic(double const gamma, double const rho, double const Bn,
                                           double const BdotB, double const P) DAMN_ALL
{
    auto const sound      = std::sqrt((gamma * P) / rho);
    auto const alfven_dir = std::sqrt(Bn * Bn / rho);
    auto const alfven     = std::sqrt(BdotB / rho);
    auto const c02        = sound * sound;
    auto const cA2        = alfven * alfven;
    auto const cAdir2     = alfven_dir * alfven_dir;
    return std::sqrt((c02 + cA2) * 0.5
                     + std::sqrt((c02 + cA2) * (c02 + cA2) - 4.0 * c02 * cAdir2) * 0.5);
}

// a face's flux and what upwind constrained transport needs from it
struct GodunovFaceResult
{
    MHDConservative F;
    std::array<double, 3> vt;        // transverse velocity: average of the left / right V
    std::array<double, 4> uct_coefs; // aL, aR (advective), dL, dR (diffusive)
    std::array<double, 3> jt{};      // Hall: the current's average (as vt), for the CT
    double rhot = 0;                 // Hall: rho's average, for the CT
};

// Hall: the current reconstructed to the face, left and right (PHARE's jL, jR); a Riemann
//  solver given them uses the Hall flux (hall_mhd_flux) and averages them (jt, rhot)
struct FaceCurrents
{
    std::array<double, 3> L, R;
};

template<std::size_t dir>
NO_DISCARD GodunovFaceResult rusanov(double const gamma, MHDPrimitive const& uL,
                                     MHDPrimitive const& uR,
                                     FaceCurrents const* const j = nullptr) DAMN_ALL
{
    auto const b2  = [](auto const& B) { return B[0] * B[0] + B[1] * B[1] + B[2] * B[2]; };
    auto const cfL = fast_magnetosonic(gamma, uL.rho, uL.B[dir], b2(uL.B), uL.P);
    auto const cfR = fast_magnetosonic(gamma, uR.rho, uR.B[dir], b2(uR.B), uR.P);
    auto const sL  = std::abs(uL.V[dir]) + cfL;
    auto const sR  = std::abs(uR.V[dir]) + cfR;
    auto const S   = sL > sR ? sL : sR;

    auto const fL = j ? hall_mhd_flux<dir>(gamma, uL, j->L) : ideal_mhd_flux<dir>(gamma, uL);
    auto const fR = j ? hall_mhd_flux<dir>(gamma, uR, j->R) : ideal_mhd_flux<dir>(gamma, uR);
    auto const cL = to_conservative(gamma, uL);
    auto const cR = to_conservative(gamma, uR);

    auto const flux = [&](double const fl, double const fr, double const ul, double const ur) {
        return (fl + fr) * 0.5 - S * (ur - ul) * 0.5;
    };

    GodunovFaceResult r{};
    r.F.rho = flux(fL.rho, fR.rho, cL.rho, cR.rho);
    for (std::size_t c = 0; c < 3; ++c)
    {
        r.F.rhoV[c] = flux(fL.rhoV[c], fR.rhoV[c], cL.rhoV[c], cR.rhoV[c]);
        r.F.B[c]    = flux(fL.B[c], fR.B[c], cL.B[c], cR.B[c]);
        r.vt[c]     = 0.5 * (uL.V[c] + uR.V[c]);
    }
    r.F.Etot    = flux(fL.Etot, fR.Etot, cL.Etot, cR.Etot);
    r.uct_coefs = {0.5, 0.5, 0.5 * S, 0.5 * S};
    if (j) // PHARE's riemann_averaging: 0.5 (L + R)
    {
        for (std::size_t c = 0; c < 3; ++c)
            r.jt[c] = 0.5 * (j->L[c] + j->R[c]);
        r.rhot = 0.5 * (uL.rho + uR.rho);
    }
    return r;
}

} // namespace plamn::core

#endif /* PLAMN_CORE_NUMERICS_GODUNOV_RUSANOV_HPP */
