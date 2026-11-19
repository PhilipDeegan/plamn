// a Simulator of each set of models a build can enable: an MHD only build's simulator compiles
//  and refuses what it cannot do yet (its levels: no MHD patch yet); inputs giving levels to a
//  model the build does not enable refused at setup

#include "plamn/core/def.hpp"

#include "plamn/core/models/options/mhd_options_def.hpp"
#include "plamn/initializer/data_provider.hpp"
#include "plamn/simulator/simulator.hpp"
#include "plamn/simulator/simulator_def.hpp"

#include "gtest/gtest.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace plamn
{
namespace
{
    auto constexpr mhd_only = SimOpts{.dimension           = 1,
                                      .interp_order        = 0,
                                      .nbRefinedPart       = 0, // no hybrid: no split
                                      .reconstruction_type = MHDOpts::ReconstructionType::Linear,
                                      .slope_limiter_type  = MHDOpts::SlopeLimiterType::VanLeer,
                                      .riemann_solver_type = MHDOpts::RiemannSolverType::Rusanov};

    // a 1D periodic MHD input, as pyphare writes it
    initializer::Dict mhd_input(std::vector<std::string> const& models)
    {
        initializer::Dict dict;
        auto& sim                                       = dict["simulation"];
        sim["models"]                                   = models;
        sim["final_time"]                               = 1.;
        sim["time_step"]["value"]                       = .01;
        sim["grid"]["nbr_cells"]["x"]                   = 32;
        sim["grid"]["meshsize"]["x"]                    = .1;
        sim["grid"]["periodicities"]["x"]               = true;
        sim["AMR"]["max_nbr_levels"]                    = 1;
        sim["algo"]["fv_method"]["heat_capacity_ratio"] = 5. / 3.;
        sim["algo"]["time_integrator_type"] = static_cast<int>(MHDOpts::TimeIntegratorType::TVDRK2);
        return dict;
    }
} // namespace

TEST(SimulatorModels, mhdOnlyBuildSetsUp)
{
    Simulator<mhd_only> sim{mhd_input({"MHDModel"})};
    EXPECT_DOUBLE_EQ(sim.endTime(), 1.);
    EXPECT_THROW(sim.initialize(), std::runtime_error); // no MHD patch yet
}

TEST(SimulatorModels, modelsTheBuildLacksRefused)
{
    EXPECT_THROW(Simulator<mhd_only>{mhd_input({"HybridModel"})}, std::runtime_error);
}

} // namespace plamn

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
