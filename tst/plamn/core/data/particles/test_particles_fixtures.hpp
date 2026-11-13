#ifndef PLAMN_TEST_CORE_DATA_PARTICLES_TEST_PARTICLES_FIXTURES_HPP
#define PLAMN_TEST_CORE_DATA_PARTICLES_TEST_PARTICLES_FIXTURES_HPP

#include "plamn/core/def.hpp"


#include "plamn/core/data/particles/particle.hpp"
#include "plamn/core/data/ions/ion_population/particle_pack.hpp"

#include "test_particles.hpp"


#include <string>
#include <cassert>

namespace plamn::core
{

template<typename Particles_t>
struct UsableParticlesPopulation
{
    template<typename Layout_t>
    UsableParticlesPopulation(std::string const& _name, Layout_t const& layout)
        : name{_name}
        , domain_particles{make_particles<Particles_t>(layout)}
    {
    }

    UsableParticlesPopulation(UsableParticlesPopulation const& that)
        : name{that.name}
        , domain_particles{that.domain_particles}
        , patch_ghost_particles{that.patch_ghost_particles}
        , level_ghost_particles{that.level_ghost_particles}
        , levelGhostParticlesOld{that.levelGhostParticlesOld}
        , levelGhostParticlesNew{that.levelGhostParticlesNew}
    {
    }

    auto& pack() { return particles_pack; }
    auto& pack() const { return particles_pack; }

    std::string name;
    Particles_t domain_particles;
    Particles_t patch_ghost_particles  = domain_particles;
    Particles_t level_ghost_particles  = domain_particles;
    Particles_t levelGhostParticlesOld = domain_particles;
    Particles_t levelGhostParticlesNew = domain_particles;

    core::ParticlesPack<Particles_t> particles_pack{name,
                                                        &domain_particles,
                                                        &patch_ghost_particles,
                                                        &level_ghost_particles,
                                                        &levelGhostParticlesOld,
                                                        &levelGhostParticlesNew};
};




} // namespace plamn::core


#endif /* PLAMN_TEST_CORE_DATA_PARTICLES_TEST_PARTICLES_FIXTURES_HPP */
