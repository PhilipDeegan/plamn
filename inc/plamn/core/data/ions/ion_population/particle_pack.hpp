#ifndef PLAMN_CORE_DATA_PARTICLES_PARTICLE_PACK_HPP
#define PLAMN_CORE_DATA_PARTICLES_PARTICLE_PACK_HPP

#include "plamn/core/def.hpp"
#include "plamn/core/data/particles/particles.hpp"
#include "damn/core/meta/type_meta.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace plamn
{
namespace core
{
    //! ParticlePack is used to pack three Particles together
    /**
     * For convenience, particles are often stored in arrays specific to their
     * location. Ghost particles are particles coming from neighbor patches of
     * the same level, incoming particles are particles coming from coarse-to-fine
     * boundaries, and interior particles are those within the patch physical domain.
     * ParticlesPack conveniently store the three arrays together.
     */
    template<typename Particles>
    struct ParticlesPack
    {
        static constexpr std::size_t dimension = Particles::dimension;
        using particles_type              = Particles;
        using pointer_tuple_t                  = damn::core::tuple_fixed_type<Particles*, 3>;

        void setBuffer(ParticlesPack* const source)
        {
            (*this) = source ? *source : ParticlesPack{_name};
        }

        std::string _name;
        Particles* _domainParticles{nullptr};
        Particles* _patchGhostParticles{nullptr};
        Particles* _levelGhostParticles{nullptr};
        Particles* _levelGhostParticlesOld{nullptr};
        Particles* _levelGhostParticlesNew{nullptr};

        auto& name() const { return _name; }


        NO_DISCARD bool isUsable() const
        {
            return damn::core::isUsable(_domainParticles, _patchGhostParticles, _levelGhostParticles);
        }


        NO_DISCARD bool isSettable() const
        {
            return damn::core::isSettable(_domainParticles, _patchGhostParticles, _levelGhostParticles);
        }


        NO_DISCARD auto& domainParticles() const
        {
            if (_domainParticles)
                return *_domainParticles;
            throw std::runtime_error("Error - cannot provide access to domainParticles");
        }
        NO_DISCARD auto& domainParticles()
        {
            return const_cast<Particles&>(
                static_cast<ParticlesPack const*>(this)->domainParticles());
        }


        NO_DISCARD auto& patchGhostParticles() const
        {
            if (_patchGhostParticles)
                return *_patchGhostParticles;
            throw std::runtime_error("Error - cannot provide access to patchGhostParticles");
        }
        NO_DISCARD auto& patchGhostParticles()
        {
            return const_cast<Particles&>(
                static_cast<ParticlesPack const*>(this)->patchGhostParticles());
        }


        NO_DISCARD auto& levelGhostParticles() const
        {
            if (_levelGhostParticles)
                return *_levelGhostParticles;
            throw std::runtime_error("Error - cannot provide access to levelGhostParticles");
        }
        NO_DISCARD auto& levelGhostParticles()
        {
            return const_cast<Particles&>(
                static_cast<ParticlesPack const*>(this)->levelGhostParticles());
        }


        NO_DISCARD Particles& levelGhostParticlesOld()
        {
            if (_levelGhostParticlesOld)
                return *_levelGhostParticlesOld;
            throw std::runtime_error("Error - cannot provide access to levelGhostParticlesOld");
        }
        NO_DISCARD Particles const& levelGhostParticlesOld() const
        {
            return const_cast<ParticlesPack*>(this)->levelGhostParticlesOld();
        }


        NO_DISCARD Particles& levelGhostParticlesNew()
        {
            if (_levelGhostParticlesNew)
                return *_levelGhostParticlesNew;
            throw std::runtime_error("Error - cannot provide access to levelGhostParticlesNew");
        }
        NO_DISCARD Particles const& levelGhostParticlesNew() const
        {
            return const_cast<ParticlesPack*>(this)->levelGhostParticlesNew();
        }
    };


} // namespace core
} // namespace plamn

#endif // PARTICLE_PACK_HPP
