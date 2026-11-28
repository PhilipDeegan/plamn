#ifndef PLAMN_CORE_NUMERICS_GODUNOV_MHD_EQUATIONS_HPP
#define PLAMN_CORE_NUMERICS_GODUNOV_MHD_EQUATIONS_HPP

#include "damn/core/def/damn_config.hpp"
#include "plamn/core/numerics/godunov/mhd_point_state.hpp"
#include "plamn/core/numerics/primitive_conservative/primitive_conservative.hpp"
#include "damn/core/def.hpp"

#include <cstddef>
#include <array>

// MHD physical flux of a primitive state along direction dir: ideal_mhd_flux, and hall_mhd_flux
//  (the Hall term added, with the current J at the same point)
namespace plamn::core
{

template<std::size_t dir>
MKN_KUL_FN_SIG_RET constexpr MHDConservative ideal_mhd_flux(double const gamma,
                                                            MHDPrimitive const& u) DAMN_ALL
{
    static_assert(dir < 3);
    auto const& V = u.V;
    auto const& B = u.B;

    auto const b2    = B[0] * B[0] + B[1] * B[1] + B[2] * B[2];
    auto const Ptot  = u.P + 0.5 * b2; // generalised pressure
    auto const Etot  = eos_p_to_etot(gamma, u.rho, V[0], V[1], V[2], B[0], B[1], B[2], u.P);
    auto const VdotB = V[0] * B[0] + V[1] * B[1] + V[2] * B[2];
    auto const vn    = V[dir];
    auto const bn    = B[dir];

    // a fixed operation order, term for term: bit equal comparisons (etc/compare/phare)
    MHDConservative F{};
    F.rho = u.rho * vn;
    for (std::size_t c = 0; c < 3; ++c)
    {
        if (c == dir)
        {
            F.rhoV[c] = u.rho * vn * vn + Ptot - bn * bn;
            F.B[c]    = 0.0;
        }
        else
        {
            F.rhoV[c] = u.rho * vn * V[c] - bn * B[c];
            F.B[c]    = B[c] * vn - V[c] * bn;
        }
    }
    F.Etot = (Etot + Ptot) * vn - bn * VdotB;
    return F;
}

// the ideal flux plus the Hall term (a fixed operation order, as above): the induction flux's
//  (J x B) / rho, the energy flux's (B.J B_n - B.B J_n) / rho
template<std::size_t dir>
MKN_KUL_FN_SIG_RET constexpr MHDConservative
hall_mhd_flux(double const gamma, MHDPrimitive const& u, std::array<double, 3> const& J) DAMN_ALL
{
    auto F            = ideal_mhd_flux<dir>(gamma, u);
    auto const& B     = u.B;
    auto const invRho = 1.0 / u.rho;
    auto const JxB_x  = J[1] * B[2] - J[2] * B[1];
    auto const JxB_y  = J[2] * B[0] - J[0] * B[2];
    auto const JxB_z  = J[0] * B[1] - J[1] * B[0];
    auto const BdotJ  = B[0] * J[0] + B[1] * J[1] + B[2] * J[2];
    auto const BdotB  = B[0] * B[0] + B[1] * B[1] + B[2] * B[2];
    if constexpr (dir == 0)
    {
        F.B[1] += -JxB_z * invRho;
        F.B[2] += JxB_y * invRho;
        F.Etot += (BdotJ * B[0] - BdotB * J[0]) * invRho;
    }
    else if constexpr (dir == 1)
    {
        F.B[0] += JxB_z * invRho;
        F.B[2] += -JxB_x * invRho;
        F.Etot += (BdotJ * B[1] - BdotB * J[1]) * invRho;
    }
    else
    {
        F.B[0] += -JxB_y * invRho;
        F.B[1] += JxB_x * invRho;
        F.Etot += (BdotJ * B[2] - BdotB * J[2]) * invRho;
    }
    return F;
}

} // namespace plamn::core

#endif /* PLAMN_CORE_NUMERICS_GODUNOV_MHD_EQUATIONS_HPP */
