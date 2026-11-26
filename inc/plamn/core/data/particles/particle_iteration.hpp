#ifndef PLAMN_CORE_DATA_PARTICLES_PARTICLE_ITERATION_HPP
#define PLAMN_CORE_DATA_PARTICLES_PARTICLE_ITERATION_HPP

// the particles of an array whatever its layout

#include "plamn/core/data/particles/particles_def.hpp"

#include <cstddef>
#include <stdexcept>

namespace plamn::core
{

// tile by tile; columns (the SoA family) as a Particle copy, SoAVXChunked's gaps (weight 0) too
void for_each_particle(TiledParticles auto const& particles, auto&& fn)
{
    for (auto const& tile : particles())
        for_each_particle(tile(), fn);
}
void for_each_particle(auto const& particles, auto&& fn)
    requires(!TiledParticles<decltype(particles)>)
{
    for (std::size_t i = 0; i < particles.size(); ++i)
        if constexpr (requires { particles.copy(i); }) // columns (SoA family): a Particle
            fn(particles.copy(i));
        else
            fn(particles[i]);
}

// in for_each_particle's order. Tiled: a walk over the tiles' sizes, for occasional access only
decltype(auto) particle_at(TiledParticles auto& particles, std::size_t i)
{
    for (auto& tile : particles())
    {
        if (i < tile().size())
            return tile()[i];
        i -= tile().size();
    }
    throw std::out_of_range("plamn::core::particle_at: past the last tile");
}
decltype(auto) particle_at(auto& particles, std::size_t const i)
    requires(!TiledParticles<decltype(particles)>)
{
    return particles[i];
}

} // namespace plamn::core

#endif // PLAMN_CORE_DATA_PARTICLES_PARTICLE_ITERATION_HPP
