#ifndef PLAMN_PARTICLE_INITIALIZER_HPP
#define PLAMN_PARTICLE_INITIALIZER_HPP

#include "plamn/core/def.hpp"

#include <cstddef>

namespace plamn
{
namespace core
{
    template<typename Particles, typename Layout>
    class ParticleInitializer
    {
    public:
        virtual void loadParticles(Particles& particles, Layout const& layout) const = 0;

        // how many particles loadParticles would load on layout's cells, e.g. a patch's load
        //  balancing weight before it is allocated
        NO_DISCARD virtual std::size_t nbrParticles(Layout const& layout) const = 0;
        virtual ~ParticleInitializer()                                          = default;
    };

} // namespace core
} // namespace plamn

#endif
