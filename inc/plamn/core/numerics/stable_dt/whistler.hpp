#ifndef PLAMN_CORE_NUMERICS_STABLE_DT_WHISTLER_HPP
#define PLAMN_CORE_NUMERICS_STABLE_DT_WHISTLER_HPP

// the hybrid stable time step, the whistler wave CFL bound (PHARE's SolverPPC::computeStableDt
// and core::compute_whistler_, behind master's "not ready in hybrid" throw): on every node of
// the mass density, B projected onto it,
//
//   dt <= cfl_wave / sum_d (pi * vw_d / dx_d),  vw_d = |B| / rho * (sqrt(1 + 1 / (4 dx_d^2))
//                                                                     + 1 / (2 dx_d))
//
// vw_d the whistler phase speed at the grid's shortest wavelength along d

#include "damn/core/def.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/point.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>

namespace plamn::core
{
NO_DISCARD inline double whistler_speed(double const inv_dx, double const rho, double const BdotB)
{
    auto const vw = std::sqrt(1 + 0.25 * inv_dx * inv_dx) + 0.5 * inv_dx;
    return std::sqrt(BdotB) * vw / rho;
}

// a patch's bound: rho its mass density, B its magnetic field components (patch grids, local
//  indexes)
template<typename Layout>
NO_DISCARD double whistler_dt(Layout const& layout, auto const& rho, auto const& B,
                              double const cfl_wave)
{
    auto constexpr dim = Layout::dimension;
    using Scalar       = std::decay_t<decltype(Layout::options)>::FieldOptions::Scalar;

    double dt = std::numeric_limits<double>::max();
    damn::core::for_each_point(domain_box(layout, Scalar::rho), [&](auto const& point) {
        damn::core::Point<std::size_t, dim> index;
        for (std::size_t d = 0; d < dim; ++d)
            index[d] = static_cast<std::size_t>(point[d]);

        auto const bx    = project<Scalar::Bx, Scalar::rho>(B[0], index);
        auto const by    = project<Scalar::By, Scalar::rho>(B[1], index);
        auto const bz    = project<Scalar::Bz, Scalar::rho>(B[2], index);
        auto const BdotB = bx * bx + by * by + bz * bz;
        auto const n     = rho(index.toArray());

        double inv_dt = 0;
        for (std::size_t d = 0; d < dim; ++d)
            inv_dt += std::numbers::pi * whistler_speed(1. / layout.dx[d], n, BdotB) / layout.dx[d];
        dt = std::min(dt, cfl_wave / inv_dt);
    });
    return dt;
}

} // namespace plamn::core

#endif // PLAMN_CORE_NUMERICS_STABLE_DT_WHISTLER_HPP
