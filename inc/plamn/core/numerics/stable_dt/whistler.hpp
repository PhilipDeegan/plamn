#ifndef PLAMN_CORE_NUMERICS_STABLE_DT_WHISTLER_HPP
#define PLAMN_CORE_NUMERICS_STABLE_DT_WHISTLER_HPP

// the hybrid stable time step, the whistler wave CFL bound (PHARE's SolverPPC::computeStableDt
// and core::compute_whistler_, behind master's "not ready in hybrid" throw): on every node of
// the mass density, B projected onto it,
//
//   dt <= cfl_wave / sum_d (vw_d / dx_d),  vw_d = vA (sqrt(1 + x_d^2) + x_d),
//                                           vA = |B| / sqrt(rho),  x_d = pi / (2 dx_d sqrt(rho))
//
// vw_d the whistler phase speed at the grid's shortest wavelength along d (k = pi / dx_d): the
//  Alfven speed on coarse grids, pi / dx_d (Hall dominated) on fine ones

#include "plamn/core/data/field/field_tiles.hpp"
#include "damn/core/def.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/point.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <algorithm>
#include <type_traits>

namespace plamn::core
{
MKN_KUL_FN_SIG_RET inline double whistler_speed(double const inv_dx, double const rho,
                                                double const BdotB)
{
    auto const sqrt_rho = std::sqrt(rho);
    auto const vA       = std::sqrt(BdotB) / sqrt_rho;
    auto const x        = std::numbers::pi * inv_dx / (2.0 * sqrt_rho);
    return vA * (std::sqrt(1.0 + x * x) + x);
}

// a patch's bound: rho its mass density, B its magnetic field components (patch grids, local
//  indexes)
template<typename Layout>
MKN_KUL_FN_SIG_RET double whistler_dt(Layout const& layout, auto const& rho, auto const& B,
                                      double const cfl_wave)
    requires(!TiledField<decltype(rho)>)
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
            inv_dt += whistler_speed(1. / layout.dx[d], n, BdotB) / layout.dx[d];
        dt = std::min(dt, cfl_wave / inv_dt);
    });
    return dt;
}

// tiled: tile by tile, on each tile's layout and views (B's tile ghosts current after its ghost
//  fill)
template<typename Layout>
MKN_KUL_FN_SIG_RET double whistler_dt(Layout const&, TiledField auto const& rho, auto const& B,
                                      double const cfl_wave)
{
    double dt = std::numeric_limits<double>::max();
    for (std::size_t ti = 0; ti < rho().size(); ++ti)
    {
        std::array const Bt{B[0]()[ti](), B[1]()[ti](), B[2]()[ti]()};
        dt = std::min(dt, whistler_dt(rho()[ti].layout(), rho()[ti](), Bt, cfl_wave));
    }
    return dt;
}

} // namespace plamn::core

#endif // PLAMN_CORE_NUMERICS_STABLE_DT_WHISTLER_HPP
