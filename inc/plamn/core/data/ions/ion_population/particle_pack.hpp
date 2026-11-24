#ifndef PLAMN_CORE_DATA_PARTICLES_PARTICLE_PACK_HPP
#define PLAMN_CORE_DATA_PARTICLES_PARTICLE_PACK_HPP

#include "mkn/kul/meta.hpp"

#include "plamn/core/def.hpp"

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
        using particles_type                   = Particles;
        using pointer_tuple_t                  = mkn::kul::tuple_fixed_type<Particles*, 3>;

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


        MKN_KUL_FN_SIG_RET bool isUsable() const
        {
            return damn::core::isUsable(_domainParticles, _patchGhostParticles, _levelGhostParticles);
        }


        MKN_KUL_FN_SIG_RET bool isSettable() const
        {
            return damn::core::isSettable(_domainParticles, _patchGhostParticles, _levelGhostParticles);
        }


        MKN_KUL_FN_SIG_RET auto& domainParticles() const
        {
            if (_domainParticles)
                return *_domainParticles;
            throw std::runtime_error("Error - cannot provide access to domainParticles");
        }
        MKN_KUL_FN_SIG_RET auto& domainParticles()
        {
            return const_cast<Particles&>(
                static_cast<ParticlesPack const*>(this)->domainParticles());
        }


        MKN_KUL_FN_SIG_RET auto& patchGhostParticles() const
        {
            if (_patchGhostParticles)
                return *_patchGhostParticles;
            throw std::runtime_error("Error - cannot provide access to patchGhostParticles");
        }
        MKN_KUL_FN_SIG_RET auto& patchGhostParticles()
        {
            return const_cast<Particles&>(
                static_cast<ParticlesPack const*>(this)->patchGhostParticles());
        }


        MKN_KUL_FN_SIG_RET auto& levelGhostParticles() const
        {
            if (_levelGhostParticles)
                return *_levelGhostParticles;
            throw std::runtime_error("Error - cannot provide access to levelGhostParticles");
        }
        MKN_KUL_FN_SIG_RET auto& levelGhostParticles()
        {
            return const_cast<Particles&>(
                static_cast<ParticlesPack const*>(this)->levelGhostParticles());
        }


        MKN_KUL_FN_SIG_RET Particles& levelGhostParticlesOld()
        {
            if (_levelGhostParticlesOld)
                return *_levelGhostParticlesOld;
            throw std::runtime_error("Error - cannot provide access to levelGhostParticlesOld");
        }
        MKN_KUL_FN_SIG_RET Particles const& levelGhostParticlesOld() const
        {
            return const_cast<ParticlesPack*>(this)->levelGhostParticlesOld();
        }


        MKN_KUL_FN_SIG_RET Particles& levelGhostParticlesNew()
        {
            if (_levelGhostParticlesNew)
                return *_levelGhostParticlesNew;
            throw std::runtime_error("Error - cannot provide access to levelGhostParticlesNew");
        }
        MKN_KUL_FN_SIG_RET Particles const& levelGhostParticlesNew() const
        {
            return const_cast<ParticlesPack*>(this)->levelGhostParticlesNew();
        }
    };


} // namespace core
} // namespace plamn

#endif // PARTICLE_PACK_HPP
