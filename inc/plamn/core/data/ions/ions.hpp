#ifndef PLAMN_IONS_HPP
#define PLAMN_IONS_HPP

#include "plamn/core/def.hpp"
#include "plamn/core/data/field/field_algorithm.hpp"
#include "plamn/core/data/field/field_tiles.hpp"
#include "plamn/initializer/data_provider.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"
#include "plamn/core/data/particles/exporting/particles_exporting.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "dict.hpp"
#include "mkn/kul/for.hpp"
// #include "particle_initializers/particle_initializer_factory.hpp"

#include <cassert>
#include <vector>
#include <string>
#include <sstream>
#include <iterator>
#include <stdexcept>
#include <functional>
#include <cstddef>
#include <format>
#include <tuple>
#include <type_traits>

namespace plamn
{
namespace core
{
    template<typename IonPopulation, typename Layout>
    class Ions
    {
    public:
        using value_type                = IonPopulation;
        using field_type                = IonPopulation::field_type;
        using vecfield_type             = IonPopulation::vecfield_type;
        using Float                     = field_type::type;
        using tensorfield_type          = IonPopulation::tensorfield_type;
        using particles_type       = IonPopulation::particles_type;
        using layout_type               = Layout;
        static constexpr auto dimension = Layout::dimension;


        Ions(Ions const&) = default;
        Ions(Ions&&)      = default;


        explicit Ions(plamn::initializer::Dict const& dict)
            : massDensity_{massDensityName(), HybridQuantity::Scalar::rho}
            , chargeDensity_{chargeDensityName(), HybridQuantity::Scalar::rho}
            , bulkVelocity_{"bulkVel", HybridQuantity::Vector::V}
            , populations_{mkn::kul::generate_from(
                  [&dict](auto ipop) { //
                      return IonPopulation{dict["pop" + std::to_string(ipop)]};
                  },
                  dict["nbrPopulations"].template to<std::size_t>())}
            , momentumTensor_{"momentumTensor", HybridQuantity::Tensor::M}
            , chargeFlux_{"chargeFlux", HybridQuantity::Vector::V}
        {
        }


        MKN_KUL_FN_SIG_RET auto nbrPopulations() const { return populations_.size(); }
        MKN_KUL_FN_SIG_RET auto size() const { return nbrPopulations(); }

        MKN_KUL_FN_SIG_RET field_type const& massDensity() const { return massDensity_; }
        MKN_KUL_FN_SIG_RET field_type& massDensity() { return massDensity_; }

        MKN_KUL_FN_SIG_RET field_type const& chargeDensity() const { return chargeDensity_; }
        MKN_KUL_FN_SIG_RET field_type& chargeDensity() { return chargeDensity_; }

        MKN_KUL_FN_SIG_RET vecfield_type const& velocity() const { return bulkVelocity_; }
        MKN_KUL_FN_SIG_RET vecfield_type& velocity() { return bulkVelocity_; }

        MKN_KUL_FN_SIG_RET vecfield_type const& chargeFlux() const { return chargeFlux_; }
        MKN_KUL_FN_SIG_RET vecfield_type& chargeFlux() { return chargeFlux_; }

        MKN_KUL_FN_SIG_RET std::string static chargeDensityName() { return "chargeDensity"; }
        MKN_KUL_FN_SIG_RET std::string static massDensityName() { return "massDensity"; }

        tensorfield_type const& momentumTensor() const { return momentumTensor_; }
        tensorfield_type& momentumTensor() { return momentumTensor_; }

        // the compute functions below work on the whole field, or only on tile tile_idx of tiled
        // fields, distinct tiles can then be computed concurrently

        // the charge density of a population is its charge times its particle density,
        // which is complete (border sums, level ghost particles) when this is called
        void computeChargeDensity(auto const&... tile_idx)
        {
            auto&& chargeDensity = maybe_tile_at(chargeDensity_, tile_idx...);
            chargeDensity.zero();

            for (auto& pop : populations_)
            {
                // we sum over all nodes contiguously, including ghosts
                // nodes. This is more efficient and easier to code as we don't
                // have to account for the field dimensionality.

                auto&& particleDensity = maybe_tile_at(pop.particleDensity(), tile_idx...);
                auto&& popDensity      = maybe_tile_at(pop.chargeDensity(), tile_idx...);
                core::transform(particleDensity, popDensity,
                                [q = pop.charge()](Float const& n) { return q * n; });
                core::transform(chargeDensity, popDensity, chargeDensity, std::plus<Float>{});
            }
        }

        void computeMassDensity(auto const&... tile_idx)
        {
            check();
            auto&& massDensity = maybe_tile_at(massDensity_, tile_idx...);
            massDensity.zero();

            for (auto const& pop : populations_)
            {
                // we sum over all nodes contiguously, including ghosts
                // nodes. This is more efficient and easier to code as we don't
                // have to account for the field dimensionality.

                auto&& popDensity = maybe_tile_at(pop.particleDensity(), tile_idx...);
                core::transform(
                    massDensity, popDensity, massDensity,
                    [&pop](auto const& n, auto const& pop_n) { return n + pop_n * pop.mass(); });
            }
        }


        void computeBulkVelocity(auto const&... tile_idx)
        {
            check();
            computeMassDensity(tile_idx...);

            auto&& massDensity  = maybe_tile_at(massDensity_, tile_idx...);
            auto&& bulkVelocity = maybe_tile_at(bulkVelocity_, tile_idx...);
            bulkVelocity.zero();
            auto& vx = bulkVelocity[0];
            auto& vy = bulkVelocity[1];
            auto& vz = bulkVelocity[2];

            for (auto& pop : populations_)
            {
                // account for mass only if populations have different masses
                std::function<Float(Float, Float)> plusMass
                    = [&pop](Float const& v, Float const& f) { return v + f * pop.mass(); };

                auto&& flux = maybe_tile_at(pop.flux(), tile_idx...);

                core::transform(vy, flux[1], vy, plusMass);
                core::transform(vx, flux[0], vx, plusMass);
                core::transform(vz, flux[2], vz, plusMass);
            }

            core::transform(vx, massDensity, vx, std::divides<Float>{});
            core::transform(vy, massDensity, vy, std::divides<Float>{});
            core::transform(vz, massDensity, vz, std::divides<Float>{});
        }


        // sum_s q_s F_s, the charge flux entering the electron velocity
        void computeChargeFlux(auto const&... tile_idx)
        {
            auto&& chargeFlux = maybe_tile_at(chargeFlux_, tile_idx...);
            chargeFlux.zero();

            for (auto const& pop : populations_)
            {
                auto const plusCharge
                    = [q = pop.charge()](Float const& c, Float const& f) { return c + q * f; };

                auto&& flux = maybe_tile_at(pop.flux(), tile_idx...);

                core::transform(chargeFlux[0], flux[0], chargeFlux[0], plusCharge);
                core::transform(chargeFlux[1], flux[1], chargeFlux[1], plusCharge);
                core::transform(chargeFlux[2], flux[2], chargeFlux[2], plusCharge);
            }
        }


        void computeFullMomentumTensor()
        {
            assert(momentumTensor_.isUsable());
            momentumTensor_.zero();
            auto& mom = momentumTensor_;

            for (auto& pop : populations_)
            {
                assert(pop.momentumTensor().isUsable());
                auto& p_mom = pop.momentumTensor();
                auto p_mij  = p_mom.begin();
                auto mij    = mom.begin();
                for (; p_mij != p_mom.end(); ++p_mij, ++mij)
                    core::transform(*mij, *p_mij, *mij, std::plus<typename field_type::type>{});
            }
        }


        void update(auto const&... tile_idx)
        {
            computeChargeDensity(tile_idx...);
            computeBulkVelocity(tile_idx...);
            computeChargeFlux(tile_idx...);
        }


        // drops the domain particles of all populations located in `boxes`
        void eraseDomainParticlesIn(auto const& boxes)
        {
            if (boxes.empty())
                return;

            for (auto& pop : populations_)
            {
                delete_particles_in(pop.domainParticles(), boxes);
                using Particles_t = std::decay_t<decltype(pop.domainParticles())>;
                if constexpr (is_chunked(Particles_t::layout_mode))
                    pop.domainParticles().template on_moved<ParticleType::Domain>();
            }
        }


        MKN_KUL_FN_SIG_RET auto begin() { return std::begin(populations_); }
        MKN_KUL_FN_SIG_RET auto end() { return std::end(populations_); }
        MKN_KUL_FN_SIG_RET auto begin() const { return std::begin(populations_); }
        MKN_KUL_FN_SIG_RET auto end() const { return std::end(populations_); }

        MKN_KUL_FN_SIG_RET auto& population(std::size_t const i)
        {
            if (i >= populations_.size())
                throw std::out_of_range("Ions population index out of range");
            return populations_[i];
        }

        MKN_KUL_FN_SIG_RET auto const& population(std::size_t const i) const
        {
            if (i >= populations_.size())
                throw std::out_of_range("Ions population index out of range");
            return populations_[i];
        }

        MKN_KUL_FN_SIG_RET bool isUsable() const
        {
            // not the momentum tensor: bound only when computed (plamn's computed quantities)
            auto usable
                = damn::core::isUsable(chargeDensity_, bulkVelocity_, massDensity_, chargeFlux_);
            for (auto const& pop : populations_)
                usable &= usable and pop.isUsable();
            return usable;
        }



        MKN_KUL_FN_SIG_RET bool isSettable() const
        {
            auto settable = damn::core::isSettable(chargeDensity_, bulkVelocity_, momentumTensor_,
                                             massDensity_, chargeFlux_);
            for (auto const& pop : populations_)
                settable = settable and pop.isSettable();
            return settable;
        }



        //-------------------------------------------------------------------------
        //                  start the ResourcesUser interface
        //-------------------------------------------------------------------------



        MKN_KUL_FN_SIG_RET std::vector<IonPopulation>& getRunTimeResourcesViewList()
        {
            return populations_;
        }

        MKN_KUL_FN_SIG_RET auto getCompileTimeResourcesViewList()
        {
            return std::forward_as_tuple(bulkVelocity_, momentumTensor_, chargeDensity_,
                                         massDensity_, chargeFlux_);
        }



        //-------------------------------------------------------------------------
        //                  ends the ResourcesUser interface
        //-------------------------------------------------------------------------


        MKN_KUL_FN_SIG_RET std::string to_str()
        {
            std::stringstream ss;
            ss << "Ions\n";
            ss << "------------------------------------\n";
            ss << "number of populations  : " << nbrPopulations() << "\n";
            for (auto& pop : populations_)
                ss << core::to_str(pop);
            return ss.str();
        }


        auto& operator[](std::size_t const i) const { return population(i); }
        auto& operator[](std::size_t const i) { return population(i); }

    private:
        void check() const
        {
            // assert(no_nans(massDensity_));
            // assert(no_nans(chargeDensity_));
            // assert(no_nans(bulkVelocity_[0]));
            // assert(no_nans(bulkVelocity_[1]));
            // assert(no_nans(bulkVelocity_[2]));
        }


        field_type massDensity_;
        field_type chargeDensity_;
        vecfield_type bulkVelocity_;
        std::vector<IonPopulation> populations_;
        tensorfield_type momentumTensor_;
        vecfield_type chargeFlux_;
    };

} // namespace core
} // namespace plamn



#endif
