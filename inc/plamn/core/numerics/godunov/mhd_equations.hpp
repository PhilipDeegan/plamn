#ifndef PLAMN_CORE_NUMERICS_GODUNOV_MHD_EQUATIONS_HPP
#define PLAMN_CORE_NUMERICS_GODUNOV_MHD_EQUATIONS_HPP

#include "damn/core/def/damn_config.hpp"
#include "plamn/core/numerics/godunov/mhd_point_state.hpp"
#include "plamn/core/numerics/primitive_conservative/primitive_conservative.hpp"
#include "damn/core/def.hpp"

#include <cstddef>
#include <array>

// ideal MHD physical flux of a primitive state along direction dir, PHARE's
//  core/numerics/MHD_equations (Hall: not yet)
namespace plamn::core
{

template<std::size_t dir>
NO_DISCARD constexpr MHDConservative ideal_mhd_flux(double const gamma,
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

    // PHARE's expressions term for term (same rounding): its results are the reference
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

} // namespace plamn::core

#endif /* PLAMN_CORE_NUMERICS_GODUNOV_MHD_EQUATIONS_HPP */
