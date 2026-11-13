#ifndef PLAMN_PARTICLE_INITIALIZER_HPP
#define PLAMN_PARTICLE_INITIALIZER_HPP

#include "plamn/core/def.hpp"


namespace plamn
{
namespace core
{
    template<typename Particles, typename Layout>
    class ParticleInitializer
    {
    public:
        virtual void loadParticles(Particles& particles, Layout const& layout) const = 0;
        virtual ~ParticleInitializer() = default;
    };

} // namespace core
} // namespace plamn

#endif
