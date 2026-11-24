#ifndef PLAMN_ION_POPULATION_HPP
#define PLAMN_ION_POPULATION_HPP

#include "plamn/core/def.hpp"
#include "plamn/initializer/data_provider.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"

#include "particle_pack.hpp"
#include "dict.hpp"

#include <cassert>
#include <tuple>
#include <string>
#include <sstream>
#include <cstddef>

namespace plamn
{
namespace core
{
    template<typename Particles, typename VecField, typename TensorField>
    class IonPopulation
    {
    public:
        using field_type                       = VecField::field_type;
        static constexpr std::size_t dimension = VecField::dimension;
        using particles_type              = Particles;
        using vecfield_type                    = VecField;
        using tensorfield_type                 = TensorField;


        IonPopulation(plamn::initializer::Dict const& initializer)
            : name_{initializer["name"].template to<std::string>()}
            , mass_{initializer["mass"].template to<double>()}
            , charge_{initializer["particle_initializer"]["charge"].template to<double>()}
            , flux_{name_ + "_flux", HybridQuantity::Vector::V}
            , momentumTensor_{name_ + "_momentumTensor", HybridQuantity::Tensor::M}
            , particleDensity_{name_ + "_particleDensity", HybridQuantity::Scalar::rho}
            , chargeDensity_{name_ + "_chargeDensity", HybridQuantity::Scalar::rho}
            , particles_{name_}
            , particleInitializerInfo_{initializer["particle_initializer"]}
        {
        }


        MKN_KUL_FN_SIG_RET double mass() const { return mass_; }

        //! the charge of every particle of the population
        MKN_KUL_FN_SIG_RET double charge() const { return charge_; }

        MKN_KUL_FN_SIG_RET std::string const& name() const { return name_; }




        MKN_KUL_FN_SIG_RET auto const& particleInitializerInfo() const
        {
            assert(particleInitializerInfo_.contains("density"));
            return particleInitializerInfo_;
        }


        MKN_KUL_FN_SIG_RET bool isUsable() const
        {
            // not the momentum tensor: bound only when computed (plamn's computed quantities)
            return damn::core::isUsable(particles_, particleDensity_, chargeDensity_, flux_);
        }

        MKN_KUL_FN_SIG_RET bool isSettable() const
        {
            return damn::core::isSettable(particles_, particleDensity_, chargeDensity_, flux_,
                                    momentumTensor_);
        }

        MKN_KUL_FN_SIG_RET auto& domainParticles() const { return particles_.domainParticles(); }
        MKN_KUL_FN_SIG_RET auto& domainParticles() { return particles_.domainParticles(); }

        MKN_KUL_FN_SIG_RET auto& patchGhostParticles() const
        {
            return particles_.patchGhostParticles();
        }
        MKN_KUL_FN_SIG_RET auto& patchGhostParticles() { return particles_.patchGhostParticles(); }

        MKN_KUL_FN_SIG_RET auto& levelGhostParticles() const
        {
            return particles_.levelGhostParticles();
        }
        MKN_KUL_FN_SIG_RET auto& levelGhostParticles() { return particles_.levelGhostParticles(); }

        MKN_KUL_FN_SIG_RET auto& levelGhostParticlesOld()
        {
            return particles_.levelGhostParticlesOld();
        }
        MKN_KUL_FN_SIG_RET auto& levelGhostParticlesOld() const
        {
            return particles_.levelGhostParticlesOld();
        }

        MKN_KUL_FN_SIG_RET auto& levelGhostParticlesNew()
        {
            return particles_.levelGhostParticlesNew();
        }
        MKN_KUL_FN_SIG_RET auto& levelGhostParticlesNew() const
        {
            return particles_.levelGhostParticlesNew();
        }

        MKN_KUL_FN_SIG_RET field_type const& particleDensity() const { return particleDensity_; }
        MKN_KUL_FN_SIG_RET field_type& particleDensity() { return particleDensity_; }

        MKN_KUL_FN_SIG_RET field_type const& chargeDensity() const { return chargeDensity_; }
        MKN_KUL_FN_SIG_RET field_type& chargeDensity() { return chargeDensity_; }

        MKN_KUL_FN_SIG_RET VecField const& flux() const { return flux_; }
        MKN_KUL_FN_SIG_RET VecField& flux() { return flux_; }

        MKN_KUL_FN_SIG_RET TensorField const& momentumTensor() const { return momentumTensor_; }
        MKN_KUL_FN_SIG_RET TensorField& momentumTensor() { return momentumTensor_; }




        //-------------------------------------------------------------------------
        //                  start the ResourcesUser interface
        //-------------------------------------------------------------------------




        MKN_KUL_FN_SIG_RET auto getCompileTimeResourcesViewList()
        {
            return std::forward_as_tuple(flux_, momentumTensor_, particleDensity_, chargeDensity_,
                                         particles_);
        }


        //-------------------------------------------------------------------------
        //                  ends the ResourcesUser interface
        //-------------------------------------------------------------------------



        MKN_KUL_FN_SIG_RET std::string to_str()
        {
            std::stringstream ss;
            ss << "Ions Population\n";
            ss << "------------------------------------\n";
            ss << "name                : " << name() << "\n";
            return ss.str();
        }

    private:
        std::string name_;
        double mass_;
        double charge_;
        VecField flux_;
        TensorField momentumTensor_;
        field_type particleDensity_;
        field_type chargeDensity_;
        ParticlesPack<Particles> particles_;
        plamn::initializer::Dict const& particleInitializerInfo_;
    };
} // namespace core
} // namespace plamn



#endif
