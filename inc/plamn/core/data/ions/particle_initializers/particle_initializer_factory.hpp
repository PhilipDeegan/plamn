#ifndef PLAMN_PARTICLE_INITIALIZER_FACTORY_HPP
#define PLAMN_PARTICLE_INITIALIZER_FACTORY_HPP


#include "plamn/core/def.hpp"
#include "plamn/initializer/data_provider.hpp"
#include "maxwellian_particle_initializer.hpp"
#include "particle_initializer.hpp"

#include <memory>
#include <string>

namespace plamn
{
namespace core
{
    template<typename Particles, typename Layout>
    class ParticleInitializerFactory
    {
        using ParticleInitializerT      = ParticleInitializer<Particles, Layout>;
        static constexpr auto dimension = Layout::dimension;


    public:
        NO_DISCARD static std::unique_ptr<ParticleInitializerT>
        create(plamn::initializer::Dict const& dict, std::string const& populationName = "")
        {
            using FunctionType = plamn::initializer::InitFunction<dimension>;

            auto initializerName = dict["name"].template to<std::string>();

            if (initializerName == "maxwellian")
            {
                auto& density = dict["density"].template to<FunctionType>();

                auto& bulkVelx = dict["bulk_velocity_x"].template to<FunctionType>();

                auto& bulkVely = dict["bulk_velocity_y"].template to<FunctionType>();

                auto& bulkVelz = dict["bulk_velocity_z"].template to<FunctionType>();

                auto& vthx = dict["thermal_velocity_x"].template to<FunctionType>();

                auto& vthy = dict["thermal_velocity_y"].template to<FunctionType>();

                auto& vthz = dict["thermal_velocity_z"].template to<FunctionType>();

                auto densityCutOff = cppdict::get_value(dict, "density_cut_off", double{1e-16});

                auto nbrPartPerCell = dict["nbr_part_per_cell"].template to<int>();

                auto basisName = dict["basis"].template to<std::string>();

                std::array<FunctionType, 3> v = {bulkVelx, bulkVely, bulkVelz};

                std::array<FunctionType, 3> vth = {vthx, vthy, vthz};

                std::optional<std::size_t> seed;
                if (dict.contains("init") && dict["init"].contains("seed"))
                    seed = dict["init"]["seed"].template to<std::optional<std::size_t>>();

                std::array<FunctionType, 3> magneticField = {nullptr, nullptr, nullptr};

                if (basisName == "cartesian")
                {
                    return std::make_unique<
                        MaxwellianParticleInitializer<Particles, Layout>>(
                        density, v, vth, nbrPartPerCell, seed, Basis::Cartesian, magneticField,
                        densityCutOff, populationName);
                }
                else if (basisName == "magnetic")
                {
                    magneticField[0] = dict["magnetic_x"].template to<FunctionType>();
                    magneticField[1] = dict["magnetic_x"].template to<FunctionType>();
                    magneticField[2] = dict["magnetic_x"].template to<FunctionType>();

                    return std::make_unique<
                        MaxwellianParticleInitializer<Particles, Layout>>(
                        density, v, vth, nbrPartPerCell, seed, Basis::Magnetic, magneticField,
                        densityCutOff, populationName);
                }
            }
            // TODO throw?
            return nullptr;
        }
    };

} // namespace core

} // namespace plamn

#endif
