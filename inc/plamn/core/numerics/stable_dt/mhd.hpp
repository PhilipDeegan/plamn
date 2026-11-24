#ifndef PLAMN_CORE_NUMERICS_STABLE_DT_MHD_HPP
#define PLAMN_CORE_NUMERICS_STABLE_DT_MHD_HPP

// the MHD stable time step (PHARE's SolverMHD::computeStableDt): on every cell of the mass
//  density, B projected onto it,
//
//   dt <= cfl_wave / sum_d (|v_d| + cf_d + cw_d) / dx_d   cf_d the fast magnetosonic speed along d,
//                                                         cw_d = vw_d - vA (Hall), 0 (ideal)
//   dt <= cfl_diffusive / (2 eta sum_d 1 / dx_d^2)        with a resistivity eta > 0
//
// vw_d the whistler phase speed at the grid's shortest wavelength (stable_dt/whistler.hpp). A
//  non-finite speed: the first such cell logged, -1 returned (the caller reduces over the ranks
//  and stops every one: a throw here would leave the others in the reduction)

#include "plamn/core/data/field/field_tiles.hpp"
#include "plamn/core/numerics/godunov/rusanov.hpp"
#include "plamn/core/numerics/primitive_conservative/primitive_conservative.hpp"
#include "plamn/core/numerics/stable_dt/whistler.hpp"
#include "damn/core/def.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/point.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <algorithm>
#include <type_traits>
#include <cstdlib>

namespace plamn::core
{
// a level's MHD model and the run's CFL numbers
struct MHDStableDt
{
    double gamma         = 5. / 3.;
    double resistivity   = 0;
    bool hall            = false;
    double cfl_wave      = 0;
    double cfl_diffusive = 0;
};

// a patch's bound: its state's rho, rhoV, B, Etot (patch grids, local indexes)
template<typename Layout>
MKN_KUL_FN_SIG_RET double mhd_dt(Layout const& layout, auto const& rho, auto const& rhoV,
                                 auto const& B, auto const& Etot, MHDStableDt const& in)
    requires(!TiledField<decltype(rho)>)
{
    auto constexpr dim = Layout::dimension;
    using Scalar       = std::decay_t<decltype(Layout::options)>::FieldOptions::Scalar;

    double dt = std::numeric_limits<double>::max();
    if (in.resistivity > 0)
    {
        double inv_dx2 = 0;
        for (std::size_t d = 0; d < dim; ++d)
            inv_dx2 += 1.0 / (layout.dx[d] * layout.dx[d]);
        dt = std::min(dt, in.cfl_diffusive / (2.0 * in.resistivity * inv_dx2));
    }

    bool non_finite = false;
    damn::core::for_each_point(domain_box(layout, Scalar::rho), [&](auto const& point) {
        if (non_finite)
            return;
        damn::core::Point<std::size_t, dim> index;
        for (std::size_t d = 0; d < dim; ++d)
            index[d] = static_cast<std::size_t>(point[d]);
        auto const at = index.toArray();

        auto const r  = rho(at);
        auto const vx = rhoV[0](at) / r;
        auto const vy = rhoV[1](at) / r;
        auto const vz = rhoV[2](at) / r;

        auto const bx = project<Scalar::Bx, Scalar::rho>(B[0], index);
        auto const by = project<Scalar::By, Scalar::rho>(B[1], index);
        auto const bz = project<Scalar::Bz, Scalar::rho>(B[2], index);

        auto const P     = eos_etot_to_p(in.gamma, r, vx, vy, vz, bx, by, bz, Etot(at));
        auto const BdotB = bx * bx + by * by + bz * bz;

        std::array<double, 3> const v{vx, vy, vz};
        std::array<double, 3> const b{bx, by, bz};

        double inv_dt = 0;
        for (std::size_t d = 0; d < dim; ++d)
        {
            auto const cfast = fast_magnetosonic(in.gamma, r, b[d], BdotB, P);
            auto const cw
                = in.hall ? whistler_speed(1.0 / layout.dx[d], r, BdotB) - std::sqrt(BdotB / r)
                          : 0.0;
            inv_dt += (std::abs(v[d]) + cfast + cw) / layout.dx[d];
        }
        if (!std::isfinite(inv_dt))
        {
            std::cerr << "plamn: non-finite MHD wave speed at local cell (";
            for (std::size_t d = 0; d < dim; ++d)
                std::cerr << (d ? ", " : "") << point[d];
            std::cerr << "): rho = " << r << ", P = " << P << ", |B| = " << std::sqrt(BdotB)
                      << "\n";
            non_finite = true; // the first one only
            return;
        }
        dt = std::min(dt, in.cfl_wave / inv_dt);
    });
    return non_finite ? -1.0 : dt;
}

// tiled: tile by tile, on each tile's layout and views
template<typename Layout>
MKN_KUL_FN_SIG_RET double mhd_dt(Layout const&, TiledField auto const& rho, auto const& rhoV,
                                 auto const& B, auto const& Etot, MHDStableDt const& in)
{
    double dt = std::numeric_limits<double>::max();
    for (std::size_t ti = 0; ti < rho().size(); ++ti)
    {
        std::array const rhoVt{rhoV[0]()[ti](), rhoV[1]()[ti](), rhoV[2]()[ti]()};
        std::array const Bt{B[0]()[ti](), B[1]()[ti](), B[2]()[ti]()};
        auto const tile_dt = mhd_dt(rho()[ti].layout(), rho()[ti](), rhoVt, Bt, Etot()[ti](), in);
        if (tile_dt < 0)
            return -1.0;
        dt = std::min(dt, tile_dt);
    }
    return dt;
}

} // namespace plamn::core

#endif // PLAMN_CORE_NUMERICS_STABLE_DT_MHD_HPP
