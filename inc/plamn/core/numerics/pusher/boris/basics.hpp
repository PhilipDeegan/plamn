#ifndef PLAMN_CORE_PUSHER_BORIS_BASICS_HPP
#define PLAMN_CORE_PUSHER_BORIS_BASICS_HPP

#include "damn/core/def/damn_config.hpp"
#include "mkn/kul/throw.hpp"
#include "plamn/core/data/grid/ghost_width_calculator.hpp"

// #include "plamn/core/data/particles/particle.hpp"
// #include "plamn/core/data/electromag/electromag.hpp"
// #include "plamn/core/data/tensorfield/tensorfield.hpp"

#include <cmath>
#include <tuple>
#include <exception>
#include <array>
#include <cstdlib>
#include <type_traits>

namespace plamn::core::boris
{
struct MoveTwoCellException : std::exception
{
    MoveTwoCellException(double const d, double const v)
        : delta{d}
        , vel{v}
    {
    }

    double delta, vel;
};


// always inlined: called per particle twice per push, out of line it costs a call and the
//  particle's fields are reloaded through memory (gcc's LTO kept it out of line)
template<auto alloc, typename Particle_t, typename Float, std::size_t dim>
[[gnu::always_inline]] inline auto static advance(
    Particle_t& p, std::array<Float, dim> const& halfDtOverDl) DAMN_ALL
{
    auto newCell = p.iCell();
    for (std::size_t iDim = 0; iDim < dim; ++iDim)
    {
        Float delta = p.delta()[iDim] + static_cast<Float>(halfDtOverDl[iDim] * p.v()[iDim]);
        if constexpr (alloc == damn::AllocatorMode::CPU)
            if (std::abs(delta) > particle_half_push_limit)
                mkn::kul::throw_exception<MoveTwoCellException>(delta, p.v()[iDim]);

        Float iCell     = std::floor(delta);
        p.delta()[iDim] = delta - iCell;
        newCell[iDim]   = static_cast<int>(iCell + p.iCell()[iDim]);
    }

    return newCell;
}


template<typename Particle, typename ParticleEB, typename Float>
[[gnu::always_inline]] inline void accelerate(Particle& p, ParticleEB const& eb,
                                              Float const coef1) DAMN_ALL
{
    static constexpr Float one = 1;
    static constexpr Float two = 2;


    auto& [pE, pB]        = eb;
    auto& [pEx, pEy, pEz] = pE;
    auto& [pBx, pBy, pBz] = pB;

    // We now apply the 3 steps of the BORIS PUSHER
    // 1st half push of the electric field
    Float const velx1 = p.v()[0] + coef1 * pEx;
    Float const vely1 = p.v()[1] + coef1 * pEy;
    Float const velz1 = p.v()[2] + coef1 * pEz;

    // preparing variables for magnetic rotation
    Float const rx = coef1 * pBx;
    Float const ry = coef1 * pBy;
    Float const rz = coef1 * pBz;

    Float const rx2  = rx * rx;
    Float const ry2  = ry * ry;
    Float const rz2  = rz * rz;
    Float const rxry = rx * ry;
    Float const rxrz = rx * rz;
    Float const ryrz = ry * rz;

    Float const invDet = one / (one + rx2 + ry2 + rz2);

    // preparing rotation matrix due to the magnetic field
    // m = invDet*(I + r*r - r x I) - I where x denotes the cross product
    Float const mxx = one + rx2 - ry2 - rz2;
    Float const mxy = two * (rxry + rz);
    Float const mxz = two * (rxrz - ry);

    Float const myx = two * (rxry - rz);
    Float const myy = one + ry2 - rx2 - rz2;
    Float const myz = two * (ryrz + rx);

    Float const mzx = two * (rxrz + ry);
    Float const mzy = two * (ryrz - rx);
    Float const mzz = one + rz2 - rx2 - ry2;

    // magnetic rotation
    Float const velx2 = (mxx * velx1 + mxy * vely1 + mxz * velz1) * invDet;
    Float const vely2 = (myx * velx1 + myy * vely1 + myz * velz1) * invDet;
    Float const velz2 = (mzx * velx1 + mzy * vely1 + mzz * velz1) * invDet;

    // 2nd half push of the electric field / Update particle velocity
    p.v()[0] = velx2 + coef1 * pEx;
    p.v()[1] = vely2 + coef1 * pEy;
    p.v()[2] = velz2 + coef1 * pEz;
}


template<typename Particles_t, typename... Args>
auto advance_noavx(Particles_t& ps, Args const&... args) DAMN_ALL
{
    auto const& [pidx, halfDtOverDl] = std::forward_as_tuple(args...);
    using Float                      = std::decay_t<decltype(halfDtOverDl)>::value_type;

    auto newCell = ps.iCell(pidx);

    for (std::size_t iDim = 0; iDim < Particles_t::dimension; ++iDim)
    {
        auto& deldim     = ps.delta()[iDim][pidx];
        auto const delta = deldim + static_cast<Float>(halfDtOverDl[iDim] * ps.v()[iDim][pidx]);
        if constexpr (Particles_t::alloc_mode == damn::AllocatorMode::CPU)
            if (std::abs(delta) > particle_half_push_limit)
                mkn::kul::throw_runtime_error("Error, particle moves more than "
                                              "particle_half_push_limit cells in a half push");

        auto iCell = std::floor(delta);
        deldim     = delta - iCell;
        newCell[iDim] += static_cast<int>(iCell);
    }

    ps.iCell(pidx) = newCell;
}

template<typename Particles_t, typename... Args>
void accelerate_noavx(Particles_t& ps, Args&... args) DAMN_ALL
{
    auto const& [pidx, em, interp, layout, coef1] = std::forward_as_tuple(args...);
    using Float                                   = std::decay_t<decltype(coef1)>;
    static constexpr Float one                    = 1;
    static constexpr Float two                    = 2;

    // index-based: SoA has no flat iterator
    auto const& eb        = interp.m2p(ps, em, layout, pidx);
    auto& [pE, pB]        = eb;
    auto& [pEx, pEy, pEz] = pE;
    auto& [pBx, pBy, pBz] = pB;

    // We now apply the 3 steps of the BORIS PUSHER
    // 1st half push of the electric field
    Float const velx1 = ps.v()[0][pidx] + coef1 * pEx;
    Float const vely1 = ps.v()[1][pidx] + coef1 * pEy;
    Float const velz1 = ps.v()[2][pidx] + coef1 * pEz;

    // preparing variables for magnetic rotation
    Float const rx = coef1 * pBx;
    Float const ry = coef1 * pBy;
    Float const rz = coef1 * pBz;

    Float const rx2  = rx * rx;
    Float const ry2  = ry * ry;
    Float const rz2  = rz * rz;
    Float const rxry = rx * ry;
    Float const rxrz = rx * rz;
    Float const ryrz = ry * rz;

    Float const invDet = one / (one + rx2 + ry2 + rz2);

    // preparing rotation matrix due to the magnetic field
    // m = invDet*(I + r*r - r x I) - I where x denotes the cross product
    Float const mxx = one + rx2 - ry2 - rz2;
    Float const mxy = two * (rxry + rz);
    Float const mxz = two * (rxrz - ry);

    Float const myx = two * (rxry - rz);
    Float const myy = one + ry2 - rx2 - rz2;
    Float const myz = two * (ryrz + rx);

    Float const mzx = two * (rxrz + ry);
    Float const mzy = two * (ryrz - rx);
    Float const mzz = one + rz2 - rx2 - ry2;

    // magnetic rotation
    Float const velx2 = (mxx * velx1 + mxy * vely1 + mxz * velz1) * invDet;
    Float const vely2 = (myx * velx1 + myy * vely1 + myz * velz1) * invDet;
    Float const velz2 = (mzx * velx1 + mzy * vely1 + mzz * velz1) * invDet;

    // 2nd half push of the electric field / Update particle velocity
    ps.v()[0][pidx] = velx2 + coef1 * pEx;
    ps.v()[1][pidx] = vely2 + coef1 * pEy;
    ps.v()[2][pidx] = velz2 + coef1 * pEz;
}



} // namespace plamn::core::boris


#endif /*PLAMN_CORE_PUSHER_BORIS_BASICS_HPP*/
