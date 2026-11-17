#ifndef PLAMN_CORE_NUMERICS_PRIMITIVE_CONSERVATIVE_PRIMITIVE_CONSERVATIVE_HPP
#define PLAMN_CORE_NUMERICS_PRIMITIVE_CONSERVATIVE_PRIMITIVE_CONSERVATIVE_HPP

#include "damn/core/def/damn_config.hpp"

#include "plamn/core/data/grid/patch_layout.hpp"
#include "plamn/core/models/quantities/mhd_quantities.hpp"

#include <array>

// MHD primitive (rho, V, B, P) <-> conservative (rho, rhoV, B, Etot) variables, PHARE's
//  core/numerics/primite_conservative_converter: per value functions, and per cell functors on
//  the cell centred grid (B, face centred, projected there). Where and how they are evaluated
//  (boxes, tiles, GPU) is up to the caller, see primitive_conservative_transformer.hpp
namespace plamn::core
{

NO_DISCARD constexpr auto v_to_rhov(auto const rho, auto const vx, auto const vy,
                                    auto const vz) DAMN_ALL
{
    return std::array{rho * vx, rho * vy, rho * vz};
}

NO_DISCARD constexpr auto rhov_to_v(auto const rho, auto const rhovx, auto const rhovy,
                                    auto const rhovz) DAMN_ALL
{
    return std::array{rhovx / rho, rhovy / rho, rhovz / rho};
}

// total energy from the pressure, ideal gas of heat capacity ratio gamma
NO_DISCARD constexpr auto eos_p_to_etot(double const gamma, auto const rho, auto const vx,
                                        auto const vy, auto const vz, auto const bx, auto const by,
                                        auto const bz, auto const p) DAMN_ALL
{
    auto const v2 = vx * vx + vy * vy + vz * vz;
    auto const b2 = bx * bx + by * by + bz * bz;
    return p / (gamma - 1.0) + 0.5 * rho * v2 + 0.5 * b2;
}

// pressure from the total energy, the inverse of eos_p_to_etot
NO_DISCARD constexpr auto eos_etot_to_p(double const gamma, auto const rho, auto const vx,
                                        auto const vy, auto const vz, auto const bx, auto const by,
                                        auto const bz, auto const etot) DAMN_ALL
{
    auto const v2 = vx * vx + vy * vy + vz * vz;
    auto const b2 = bx * bx + by * by + bz * bz;
    return (gamma - 1.0) * (etot - 0.5 * rho * v2 - 0.5 * b2);
}

// B on the cell centred grid at ijk
NO_DISCARD auto b_at_cell(auto const& B, auto const& ijk) DAMN_ALL
{
    using enum MHDQuantity::Scalar;
    return std::array{project<Bx, rho>(B[0], ijk), project<By, rho>(B[1], ijk),
                      project<Bz, rho>(B[2], ijk)};
}

// rho, V, B, P -> rhoV, Etot at one cell
struct ToConservative
{
    double gamma;

    void operator()(auto const& ijk, auto const& rho, auto const& V, auto const& B, auto const& P,
                    auto& rhoV, auto& Etot) const DAMN_ALL
    {
        auto const r               = rho(ijk);
        auto const vx              = V[0](ijk);
        auto const vy              = V[1](ijk);
        auto const vz              = V[2](ijk);
        auto const [bx, by, bz]    = b_at_cell(B, ijk);
        auto const [rvx, rvy, rvz] = v_to_rhov(r, vx, vy, vz);
        rhoV[0](ijk)               = rvx;
        rhoV[1](ijk)               = rvy;
        rhoV[2](ijk)               = rvz;
        Etot(ijk)                  = eos_p_to_etot(gamma, r, vx, vy, vz, bx, by, bz, P(ijk));
    }
};

// rho, rhoV, B, Etot -> V, P at one cell
struct ToPrimitive
{
    double gamma;

    void operator()(auto const& ijk, auto const& rho, auto const& rhoV, auto const& B,
                    auto const& Etot, auto& V, auto& P) const DAMN_ALL
    {
        auto const r            = rho(ijk);
        auto const [vx, vy, vz] = rhov_to_v(r, rhoV[0](ijk), rhoV[1](ijk), rhoV[2](ijk));
        auto const [bx, by, bz] = b_at_cell(B, ijk);
        V[0](ijk)               = vx;
        V[1](ijk)               = vy;
        V[2](ijk)               = vz;
        P(ijk)                  = eos_etot_to_p(gamma, r, vx, vy, vz, bx, by, bz, Etot(ijk));
    }
};

} // namespace plamn::core

#endif /* PLAMN_CORE_NUMERICS_PRIMITIVE_CONSERVATIVE_PRIMITIVE_CONSERVATIVE_HPP */
