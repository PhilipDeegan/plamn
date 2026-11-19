// an MHD only build's simulator: one MHD level from the user's functions, advanced by the model's
//  scheme over patches (ghosts: copies, then the physical boundaries); inputs giving levels to a
//  model the build does not enable refused at setup

#include "plamn/core/def.hpp"

#include "plamn/core/models/options/mhd_options_def.hpp"
#include "plamn/initializer/data_provider.hpp"
#include "plamn/simulator/simulator.hpp"
#include "plamn/simulator/simulator_def.hpp"

#include "gtest/gtest.h"

#include "damn/core/memory/span.hpp"
#include "plamn/core/boundary/boundary_defs.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <numeric>
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

    using InitFunction = initializer::InitFunction<1>;

    // f(x) at every point
    InitFunction function(double (*f)(double))
    {
        return [f](damn::core::Span<double const> const& x) {
            auto values = std::make_shared<damn::core::VectorSpan<double>>(x.size(), 0.);
            for (std::size_t i = 0; i < x.size(); ++i)
                (*values)[i] = f(x[i]);
            return values;
        };
    }
    InitFunction constant(double const value)
    {
        return [value](damn::core::Span<double const> const& x) {
            return std::make_shared<damn::core::VectorSpan<double>>(x.size(), value);
        };
    }

    std::uint32_t constexpr cells = 32;
    double constexpr dx           = .1;

    // a 1D MHD input, as pyphare writes it: periodic unless walls (reflective boundaries), the
    //  domain in patches of patch_cells, uniform but its density (rho)
    initializer::Dict mhd_input(std::vector<std::string> const& models, InitFunction rho,
                                int const patch_cells = cells, bool const walls = false)
    {
        initializer::Dict dict;
        auto& sim                         = dict["simulation"];
        sim["models"]                     = models;
        sim["final_time"]                 = 1.;
        sim["time_step"]["value"]         = .01;
        sim["grid"]["nbr_cells"]["x"]     = static_cast<int>(cells);
        sim["grid"]["meshsize"]["x"]      = dx;
        sim["grid"]["periodicities"]["x"] = !walls;
        if (walls)
            for (auto const* loc : {"xlower", "xupper"})
                sim["grid"]["domain_boundaries"][loc]["type"]
                    = static_cast<int>(core::BoundaryType::Reflective);
        sim["AMR"]["max_nbr_levels"]                    = 1;
        sim["AMR"]["largest_patch_size"]                = std::vector<int>{patch_cells};
        sim["algo"]["fv_method"]["heat_capacity_ratio"] = 5. / 3.;
        sim["algo"]["time_integrator_type"] = static_cast<int>(MHDOpts::TimeIntegratorType::TVDRK2);
        auto& state                         = sim["mhd_state"];
        state["density"]["initializer"]     = rho;
        state["pressure"]["initializer"]    = constant(.5);
        auto const vector = [&](std::string const& name, std::array<double, 3> const v) {
            for (std::size_t c = 0; c < 3; ++c)
                state[name]["initializer"][std::string{"xyz"[c]} + "_component"] = constant(v[c]);
        };
        vector("velocity", {.1, .2, -.1});
        vector("magnetic", {1., .5, .2});
        return dict;
    }

    double wave(double const x)
    {
        return 1. + .2 * std::sin(2 * std::numbers::pi * x / (cells * dx));
    }

    // rho's values after n steps
    std::vector<double> run(initializer::Dict const& input, std::size_t const n)
    {
        Simulator<mhd_only> sim{input};
        sim.initialize();
        for (std::size_t i = 0; i < n; ++i)
            sim.advance();
        return sim.domain_values("rho");
    }
} // namespace

TEST(SimulatorModels, mhdUniformStaysUniform)
{
    for (auto const rho : run(mhd_input({"MHDModel"}, constant(1.3), 8), 10))
        EXPECT_NEAR(rho, 1.3, 1e-13);
}

TEST(SimulatorModels, mhdIndependentOfThePatches)
{
    auto const one  = run(mhd_input({"MHDModel"}, function(wave)), 10);
    auto const four = run(mhd_input({"MHDModel"}, function(wave), 8), 10);
    ASSERT_EQ(one.size(), cells);
    EXPECT_EQ(one, four);                  // serial, same build: bit for bit
    EXPECT_NE(one.front(), wave(.5 * dx)); // the wave moved
}

TEST(SimulatorModels, mhdMassStaysBetweenWalls)
{
    auto const input = mhd_input({"MHDModel"}, function(wave), 8, /*walls=*/true);
    Simulator<mhd_only> sim{input};
    sim.initialize();
    auto const total = [&]() {
        auto const rho = sim.domain_values("rho");
        return std::accumulate(rho.begin(), rho.end(), 0.);
    };
    double const mass = total();
    for (std::size_t i = 0; i < 20; ++i)
        sim.advance();
    EXPECT_NEAR(total(), mass, 1e-12 * mass);
}

TEST(SimulatorModels, modelsTheBuildLacksRefused)
{
    EXPECT_THROW((Simulator<mhd_only>{mhd_input({"HybridModel"}, constant(1.))}),
                 std::runtime_error);
}

} // namespace plamn

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
