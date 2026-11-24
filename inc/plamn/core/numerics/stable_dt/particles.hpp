#ifndef PLAMN_CORE_NUMERICS_STABLE_DT_PARTICLES_HPP
#define PLAMN_CORE_NUMERICS_STABLE_DT_PARTICLES_HPP

// the hybrid stable time step's particle bound (plamn's: PHARE's hybrid has only the whistler
//  one): no domain particle moves more than cfl_particles of the allowed cell jump in a step,
//
//   dt <= cfl_particles * cell_jump * dx_d / max |v_d|    over the array's particles, per direction
//
// (cell_jump: core::particle_allowed_cell_jump, what the ghost widths are made for; a domain
//  particle moving further throws, the cell jump limit). Gap slots (weight 0, SoAVXChunked) are
//  not particles: skipped
//
//   max_abs_velocity(particles)          max |v_d| per direction (3 components)
//   particles_dt(dx, particles, cfl)     the bound (no particle, or at rest: no bound, max())

#include "plamn/core/data/grid/ghost_width_calculator.hpp"
#include "plamn/core/data/particles/particle_components.hpp"
#include "plamn/core/data/particles/particle_iteration.hpp"
#include "damn/core/def.hpp"

#include <array>
#include <limits>
#include <algorithm>
#include <cstdlib>

namespace plamn::core
{

// the columns (or strided members) themselves: one pass per velocity component
template<typename Particles>
    requires HasComponents<Particles>
MKN_KUL_FN_SIG_RET std::array<double, 3> max_abs_velocity(Particles const& particles)
{
    std::array<double, 3> m{0, 0, 0};
    auto const c = components(particles);
    auto const n = c.weight.size();
    for (std::size_t c3 = 0; c3 < 3; ++c3)
        for (std::size_t i = 0; i < n; ++i)
            if (c.weight[i] != 0)
                m[c3] = std::max(m[c3], std::abs(c.v[c3][i]));
    return m;
}

// any other layout (tiled ones tile by tile): particle by particle
template<typename Particles>
    requires(!HasComponents<Particles>)
MKN_KUL_FN_SIG_RET std::array<double, 3> max_abs_velocity(Particles const& particles)
{
    std::array<double, 3> m{0, 0, 0};
    for_each_particle(particles, [&](auto const& p) {
        if (p.weight() == 0)
            return;
        for (std::size_t c3 = 0; c3 < 3; ++c3)
            m[c3] = std::max(m[c3], std::abs(p.v()[c3]));
    });
    return m;
}

template<std::size_t dim>
MKN_KUL_FN_SIG_RET double particles_dt(std::array<double, dim> const& dx, auto const& particles,
                                       double const cfl_particles)
{
    auto const v = max_abs_velocity(particles);
    double dt    = std::numeric_limits<double>::max();
    for (std::size_t d = 0; d < dim; ++d)
        if (v[d] > 0)
            dt = std::min(dt, cfl_particles * particle_allowed_cell_jump * dx[d] / v[d]);
    return dt;
}

} // namespace plamn::core

#endif // PLAMN_CORE_NUMERICS_STABLE_DT_PARTICLES_HPP
