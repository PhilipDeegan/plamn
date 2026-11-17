#ifndef PLAMN_CORE_NUMERICS_FINITE_VOLUME_FINITE_VOLUME_EULER_HPP
#define PLAMN_CORE_NUMERICS_FINITE_VOLUME_FINITE_VOLUME_EULER_HPP

#include "damn/core/def.hpp"
#include "damn/core/def/damn_config.hpp"

#include <array>
#include <cstddef>

// one conserved quantity's finite volume (forward Euler) update in one cell from its face fluxes,
//  PHARE's core/numerics/finite_volume_euler/finite_volume_euler_per_field.hpp, its expression
//  order: Unew = U - (dt / dx) (F_x(i + 1) - F_x(i)) - (dt / dy) (...) - ...
//  The face i of a flux field is the lower face of cell i (GodunovFace's indexing). Where and how
//  it is evaluated (boxes, tiles, GPU) is up to the caller, see finite_volume_euler_transformer.hpp
namespace plamn::core
{

template<std::size_t dim>
struct FiniteVolumeEuler
{
    double dt;
    std::array<double, dim> inv_dx; // 1 / dx, as PHARE's inverseMeshSize

    NO_DISCARD double operator()(auto const& ijk, auto const& U, auto const& Fx, auto const& Fy,
                                 auto const& Fz) const DAMN_ALL
    {
        auto const diff = [&](auto const& F, auto const d) {
            auto next = ijk;
            next[d] += 1;
            return F(next) - F(ijk);
        };
        if constexpr (dim == 1)
            return U(ijk) - (dt * inv_dx[0]) * diff(Fx, 0);
        else if constexpr (dim == 2)
            return U(ijk) - (dt * inv_dx[0]) * diff(Fx, 0) - (dt * inv_dx[1]) * diff(Fy, 1);
        else
            return U(ijk) - (dt * inv_dx[0]) * diff(Fx, 0) - (dt * inv_dx[1]) * diff(Fy, 1)
                   - (dt * inv_dx[2]) * diff(Fz, 2);
    }

    template<typename Layout>
    NO_DISCARD static FiniteVolumeEuler from(Layout const& layout, double const dt)
    {
        FiniteVolumeEuler fve{dt, {}};
        for (std::size_t d = 0; d < dim; ++d)
            fve.inv_dx[d] = 1.0 / layout.dx[d];
        return fve;
    }
};

} // namespace plamn::core

#endif /* PLAMN_CORE_NUMERICS_FINITE_VOLUME_FINITE_VOLUME_EULER_HPP */
