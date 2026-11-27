#ifndef PLAMN_ETC_BENCH_PARTICLES_BENCH_PARTICLES_HPP
#define PLAMN_ETC_BENCH_PARTICLES_BENCH_PARTICLES_HPP

// the particles benches run on: ppc per cell, deltas and velocities from seed

#include "plamn/core/data/particles/appending/particles_appending.hpp"

#include "tst/plamn/core/data/particles/test_particles.hpp"
#include "tst/plamn/core/data/ion_population/test_ion_population_fixtures.hpp"

#include <cstddef>

namespace plamn::core::bench
{

template<auto opts>
auto make_ions(auto const& layout, std::size_t const ppc, std::size_t const seed) // AoSMapped
{
    UsableIons_t<opts> ions{layout, "protons"};
    auto& particles = ions.populations[0].particles;
    particles.domain_particles.reserve(layout.cells.size() * ppc);
    core::add_particles(particles.domain_particles, layout.cells, ppc);
    add_ghost_particles(particles.level_ghost_particles, layout.cells, ppc, 1);
    delta_disperse(particles.domain_particles, seed);
    vary_velocity(particles.domain_particles, -6, 6, seed);
    delta_disperse(particles.level_ghost_particles, seed);
    vary_velocity(particles.level_ghost_particles, -6, 6, seed);
    return ions;
}

// from's particles in opts' layout
template<auto opts>
auto from_ions(auto const& layout, auto const& from, std::size_t const ppc)
{
    UsableIons_t<opts> ions{layout, "protons"};
    auto& dst = ions.populations[0].particles;
    auto& src = from.populations[0].particles;
    reserve(dst.domain_particles, layout.cells, ppc);
    append_particles<ParticleType::Domain>(src.domain_particles, dst.domain_particles);
    append_particles<ParticleType::LevelGhost>(src.level_ghost_particles,
                                               dst.level_ghost_particles);
    return ions;
}

} // namespace plamn::core::bench

#endif // PLAMN_ETC_BENCH_PARTICLES_BENCH_PARTICLES_HPP
