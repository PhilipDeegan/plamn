#ifndef PLAMN_CORE_NUMERICS_GODUNOV_MHD_POINT_STATE_HPP
#define PLAMN_CORE_NUMERICS_GODUNOV_MHD_POINT_STATE_HPP

#include "damn/core/def/damn_config.hpp"
#include "plamn/core/numerics/primitive_conservative/primitive_conservative.hpp"
#include "damn/core/def.hpp"

#include <array>

// MHD state at one point (a face's left / right state, a flux): values, not views, so a face
//  functor can return them (PHARE's godunov_utils PerIndex / PerIndexVector)
namespace plamn::core
{

// rho, V, B, P
struct MHDPrimitive
{
    double rho;
    std::array<double, 3> V, B;
    double P;
};

// rho, rhoV, B, Etot: also the layout of a flux (F_rho, F_rhoV, F_B, F_Etot)
struct MHDConservative
{
    double rho;
    std::array<double, 3> rhoV, B;
    double Etot;
};

NO_DISCARD constexpr MHDConservative to_conservative(double const gamma,
                                                     MHDPrimitive const& u) DAMN_ALL
{
    auto const& [vx, vy, vz] = u.V;
    auto const& [bx, by, bz] = u.B;
    return {u.rho, v_to_rhov(u.rho, vx, vy, vz), u.B,
            eos_p_to_etot(gamma, u.rho, vx, vy, vz, bx, by, bz, u.P)};
}

} // namespace plamn::core

#endif /* PLAMN_CORE_NUMERICS_GODUNOV_MHD_POINT_STATE_HPP */
