#ifndef PLAMN_CORE_NUMERICS_GODUNOV_HLLD_HPP
#define PLAMN_CORE_NUMERICS_GODUNOV_HLLD_HPP

// HLLD flux at one face (Miyoshi & Kusano 2005) (a fixed operation order: bit equal
//  comparisons, etc/compare/phare), ideal MHD (Hall: not yet): the outer fast speeds SL, SR (as
//  HLL), the contact SM, the rotational SL*, SR*; the four intermediate states L*, L**, R**, R*,
//  and the flux of the region the face is in. Where the rotational waves (nearly) coincide with the
//  outer ones, an HLLC fallback (as idefix / PLUTO): the magnetic field of the star states from the
//  HLL average, SL* = SR* = SM for the flux. The upwind constrained transport inputs (Mignone & Del
//  Zanna 2021) from the speeds before that fallback; as HLL's where it applies.

#include "damn/core/def.hpp"
#include "damn/core/def/damn_config.hpp"
#include "plamn/core/numerics/godunov/mhd_equations.hpp"
#include "plamn/core/numerics/godunov/mhd_point_state.hpp"
#include "plamn/core/numerics/godunov/rusanov.hpp"
#include "plamn/core/numerics/primitive_conservative/primitive_conservative.hpp"

#include <algorithm>
#include <cmath>
#include <array>
#include <cstdlib>

namespace plamn::core
{
template<std::size_t dir>
MKN_KUL_FN_SIG_RET GodunovFaceResult hlld(double const gamma, MHDPrimitive const& uL,
                                          MHDPrimitive const& uR,
                                          FaceCurrents const* const j = nullptr) DAMN_ALL
{
    std::size_t constexpr t0 = dir == 0 ? 1 : 0; // the transverse directions, in order
    std::size_t constexpr t1 = dir == 2 ? 1 : 2;
    auto const sgn           = [](double const x) { return (x > 0.0) - (x < 0.0); };

    // the speeds (hlld_speeds_)
    auto const BdotBL         = uL.B[0] * uL.B[0] + uL.B[1] * uL.B[1] + uL.B[2] * uL.B[2];
    auto const BdotBR         = uR.B[0] * uR.B[0] + uR.B[1] * uR.B[1] + uR.B[2] * uR.B[2];
    auto const VnL            = uL.V[dir];
    auto const VnR            = uR.V[dir];
    auto const cfastL         = fast_magnetosonic(gamma, uL.rho, uL.B[dir], BdotBL, uL.P);
    auto const cfastR         = fast_magnetosonic(gamma, uR.rho, uR.B[dir], BdotBR, uR.P);
    auto const SL             = std::min(VnL - cfastL, VnR - cfastR);
    auto const SR             = std::max(VnL + cfastL, VnR + cfastR);
    auto const PtL            = uL.P + 0.5 * BdotBL;
    auto const PtR            = uR.P + 0.5 * BdotBR;
    auto const Bn             = uL.B[dir]; // the same on both sides
    auto const SM_numerator   = uR.rho * VnR * (SR - VnR) - uL.rho * VnL * (SL - VnL) - PtR + PtL;
    auto const SM_denominator = uR.rho * (SR - VnR) - uL.rho * (SL - VnL);
    auto const SM             = SM_numerator / SM_denominator;
    auto const rhoL_s         = uL.rho * (SL - VnL) / (SL - SM);
    auto const rhoR_s         = uR.rho * (SR - VnR) / (SR - SM);
    auto SL_s                 = SM - std::abs(Bn) / std::sqrt(rhoL_s);
    auto SR_s                 = SM + std::abs(Bn) / std::sqrt(rhoR_s);

    GodunovFaceResult r{};

    // the constrained transport inputs, from these speeds (uct_coefs_): the speeds clipped at 0
    //  (as HLL); HLL's where the rotational waves meet the outer ones, else the HLLD ones
    auto const fallback = (SL_s - SL) < 1.0e-4 * (SM - SL) or (SR_s - SR) > -1.0e-4 * (SR - SM);
    {
        auto const sl  = std::min(0.0, SL);
        auto const sr  = std::max(0.0, SR);
        auto const inv = 1.0 / (sr - sl);
        if (fallback)
        {
            auto const d = -sr * sl * inv;
            r.uct_coefs  = {sr * inv, -sl * inv, d, d};
        }
        else
        {
            auto const nuL  = (SL_s + SL) / (std::abs(SL_s) + std::abs(SL));
            auto const nuR  = (SR_s + SR) / (std::abs(SR_s) + std::abs(SR));
            auto const nu_s = std::abs(SR_s - SL_s) > 1.0e-9 * std::abs(SR - SL)
                                  ? (SR_s + SL_s) / (std::abs(SR_s) + std::abs(SL_s))
                                  : 0.0;
            auto const xiL  = ((VnL - SM) * (SL - SM)) / (SL_s + SL - 2. * SM);
            auto const xiR  = ((VnR - SM) * (SR - SM)) / (SR_s + SR - 2. * SM);
            r.uct_coefs     = {(1 + nu_s) * 0.5, (1 - nu_s) * 0.5,
                               0.5 * (nuL - nu_s) * xiL + 0.5 * (std::abs(SL_s) - nu_s * SL_s),
                               0.5 * (nuR - nu_s) * xiR + 0.5 * (std::abs(SR_s) - nu_s * SR_s)};
        }
        for (std::size_t c = 0; c < 3; ++c)
            r.vt[c] = (sr * uL.V[c] - sl * uR.V[c]) * inv;
        if (j) // the Riemann averages with the clipped speeds
        {
            for (std::size_t c = 0; c < 3; ++c)
                r.jt[c] = (sr * j->L[c] - sl * j->R[c]) * inv;
            r.rhot = (sr * uL.rho - sl * uR.rho) * inv;
        }
    }

    // the intermediate states (hlld_intermediate_states_)
    auto const& [vxL, vyL, vzL] = uL.V;
    auto const& [bxL, byL, bzL] = uL.B;
    auto const& [vxR, vyR, vzR] = uR.V;
    auto const& [bxR, byR, bzR] = uR.B;
    auto const etotL            = eos_p_to_etot(gamma, uL.rho, vxL, vyL, vzL, bxL, byL, bzL, uL.P);
    auto const etotR            = eos_p_to_etot(gamma, uR.rho, vxR, vyR, vzR, bxR, byR, bzR, uR.P);
    auto const p_tot_L = uL.P + 0.5 * (uL.B[0] * uL.B[0] + uL.B[1] * uL.B[1] + uL.B[2] * uL.B[2]);
    auto const p_tot_R = uR.P + 0.5 * (uR.B[0] * uR.B[0] + uR.B[1] * uR.B[1] + uR.B[2] * uR.B[2]);
    auto const fL      = j ? hall_mhd_flux<dir>(gamma, uL, j->L) : ideal_mhd_flux<dir>(gamma, uL);
    auto const fR      = j ? hall_mhd_flux<dir>(gamma, uR, j->R) : ideal_mhd_flux<dir>(gamma, uR);

    auto const B_hll = [&](std::size_t const t) {
        return (SR * uR.B[t] - SL * uL.B[t] + fL.B[t] - fR.B[t]) / (SR - SL);
    };
    auto const Bt_s = [&](MHDPrimitive const& u, double const S, std::size_t const t) {
        return u.B[t] * (u.rho * (S - u.V[dir]) * (S - u.V[dir]) - Bn * Bn)
               / (u.rho * (S - u.V[dir]) * (S - SM) - Bn * Bn);
    };
    auto const vn = SM;
    auto Bt0L_s   = Bt_s(uL, SL, t0);
    auto Bt1L_s   = Bt_s(uL, SL, t1);
    auto Bt0R_s   = Bt_s(uR, SR, t0);
    auto Bt1R_s   = Bt_s(uR, SR, t1);
    auto Bn_s     = Bn;
    if (fallback) // HLLC, as idefix / PLUTO
    {
        Bn_s   = B_hll(dir);
        Bt0L_s = Bt0R_s = B_hll(t0);
        Bt1L_s = Bt1R_s = B_hll(t1);
        SL_s = SR_s = SM;
    }

    auto const vt0L_s = uL.V[t0] - Bn * (Bt0L_s - uL.B[t0]) / (uL.rho * (SL - uL.V[dir]));
    auto const vt1L_s = uL.V[t1] - Bn * (Bt1L_s - uL.B[t1]) / (uL.rho * (SL - uL.V[dir]));
    auto const vt0R_s = uR.V[t0] - Bn * (Bt0R_s - uR.B[t0]) / (uR.rho * (SR - uR.V[dir]));
    auto const vt1R_s = uR.V[t1] - Bn * (Bt1R_s - uR.B[t1]) / (uR.rho * (SR - uR.V[dir]));

    auto const p_tot_s
        = ((SR - uR.V[dir]) * uR.rho * p_tot_L - (SL - uL.V[dir]) * uL.rho * p_tot_R
           + uL.rho * uR.rho * (SL - uL.V[dir]) * (SR - uR.V[dir]) * (uR.V[dir] - uL.V[dir]))
          / (uR.rho * (SR - uR.V[dir]) - uL.rho * (SL - uL.V[dir]));

    auto const EtotL_s = ((SL - uL.V[dir]) * etotL - p_tot_L * uL.V[dir] + p_tot_s * SM
                          + Bn
                                * (Bn * uL.V[dir] + uL.B[t0] * uL.V[t0] + uL.B[t1] * uL.V[t1]
                                   - (Bn * vn + vt0L_s * Bt0L_s + vt1L_s * Bt1L_s)))
                         / (SL - SM);
    auto const EtotR_s = ((SR - uR.V[dir]) * etotR - p_tot_R * uR.V[dir] + p_tot_s * SM
                          + Bn
                                * (Bn * uR.V[dir] + uR.B[t0] * uR.V[t0] + uR.B[t1] * uR.V[t1]
                                   - (Bn * vn + vt0R_s * Bt0R_s + vt1R_s * Bt1R_s)))
                         / (SR - SM);

    auto const vt0_ss
        = (std::sqrt(rhoL_s) * vt0L_s + std::sqrt(rhoR_s) * vt0R_s + (Bt0R_s - Bt0L_s) * sgn(Bn))
          / (std::sqrt(rhoL_s) + std::sqrt(rhoR_s));
    auto const vt1_ss
        = (std::sqrt(rhoL_s) * vt1L_s + std::sqrt(rhoR_s) * vt1R_s + (Bt1R_s - Bt1L_s) * sgn(Bn))
          / (std::sqrt(rhoL_s) + std::sqrt(rhoR_s));
    auto const Bt0_ss = (std::sqrt(rhoL_s) * Bt0R_s + std::sqrt(rhoR_s) * Bt0L_s
                         + std::sqrt(rhoL_s * rhoR_s) * (vt0R_s - vt0L_s) * sgn(Bn))
                        / (std::sqrt(rhoL_s) + std::sqrt(rhoR_s));
    auto const Bt1_ss = (std::sqrt(rhoL_s) * Bt1R_s + std::sqrt(rhoR_s) * Bt1L_s
                         + std::sqrt(rhoL_s * rhoR_s) * (vt1R_s - vt1L_s) * sgn(Bn))
                        / (std::sqrt(rhoL_s) + std::sqrt(rhoR_s));

    auto const EtotL_ss = EtotL_s
                          - std::sqrt(rhoL_s)
                                * (vn * Bn + vt0L_s * Bt0L_s + vt1L_s * Bt1L_s
                                   - (vn * Bn + vt0_ss * Bt0_ss + vt1_ss * Bt1_ss))
                                * sgn(Bn);
    auto const EtotR_ss = EtotR_s
                          + std::sqrt(rhoR_s)
                                * (vn * Bn + vt0R_s * Bt0R_s + vt1R_s * Bt1R_s
                                   - (vn * Bn + vt0_ss * Bt0_ss + vt1_ss * Bt1_ss))
                                * sgn(Bn);

    auto const state = [&](double const rho, double const vt0, double const vt1, double const bn,
                           double const bt0, double const bt1, double const etot) {
        MHDConservative u{};
        u.rho       = rho;
        u.rhoV[dir] = rho * vn;
        u.rhoV[t0]  = rho * vt0;
        u.rhoV[t1]  = rho * vt1;
        u.B[dir]    = bn;
        u.B[t0]     = bt0;
        u.B[t1]     = bt1;
        u.Etot      = etot;
        return u;
    };
    auto const uL_s  = state(rhoL_s, vt0L_s, vt1L_s, Bn_s, Bt0L_s, Bt1L_s, EtotL_s);
    auto const uR_s  = state(rhoR_s, vt0R_s, vt1R_s, Bn_s, Bt0R_s, Bt1R_s, EtotR_s);
    auto const uL_ss = state(rhoL_s, vt0_ss, vt1_ss, Bn, Bt0_ss, Bt1_ss, EtotL_ss);
    auto const uR_ss = state(rhoR_s, vt0_ss, vt1_ss, Bn, Bt0_ss, Bt1_ss, EtotR_ss);
    auto const cL    = to_conservative(gamma, uL);
    auto const cR    = to_conservative(gamma, uR);

    // the flux of the face's region (hlld_), with the fallback's SL* = SR* = SM
    auto const flux
        = [&](double const ul, double const ul_s, double const ul_ss, double const ur_ss,
              double const ur_s, double const ur, double const fl, double const fr) {
              if (SL >= 0.0) // L
                  return fl;
              if (SR <= 0.0) // R
                  return fr;
              if (SL_s >= 0) // L*
                  return fl + SL * ul_s - SL * ul;
              if (SR_s <= 0) // R*
                  return fr + SR * ur_s - SR * ur;
              if (SM >= 0) // L**
                  return fl + SL_s * ul_ss - (SL_s - SL) * ul_s - SL * ul;
              return fr + SR_s * ur_ss - (SR_s - SR) * ur_s - SR * ur; // R**
          };
    r.F.rho = flux(cL.rho, uL_s.rho, uL_ss.rho, uR_ss.rho, uR_s.rho, cR.rho, fL.rho, fR.rho);
    for (std::size_t c = 0; c < 3; ++c)
    {
        r.F.rhoV[c] = flux(cL.rhoV[c], uL_s.rhoV[c], uL_ss.rhoV[c], uR_ss.rhoV[c], uR_s.rhoV[c],
                           cR.rhoV[c], fL.rhoV[c], fR.rhoV[c]);
        r.F.B[c]    = flux(cL.B[c], uL_s.B[c], uL_ss.B[c], uR_ss.B[c], uR_s.B[c], cR.B[c], fL.B[c],
                           fR.B[c]);
    }
    r.F.Etot
        = flux(cL.Etot, uL_s.Etot, uL_ss.Etot, uR_ss.Etot, uR_s.Etot, cR.Etot, fL.Etot, fR.Etot);
    return r;
}

struct HLLDRiemann
{
    template<std::size_t dir>
    MKN_KUL_FN_SIG_RET static GodunovFaceResult
    solve(double const gamma, MHDPrimitive const& uL, MHDPrimitive const& uR,
          FaceCurrents const* const j = nullptr) DAMN_ALL
    {
        return hlld<dir>(gamma, uL, uR, j);
    }
};

} // namespace plamn::core

#endif /* PLAMN_CORE_NUMERICS_GODUNOV_HLLD_HPP */
