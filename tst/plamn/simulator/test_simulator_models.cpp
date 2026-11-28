// an MHD only build's simulator

#include "plamn/core/def.hpp"

#include "plamn/core/models/options/mhd_options_def.hpp"
#include "plamn/initializer/data_provider.hpp"
#include "plamn/simulator/restart_paths.hpp"
#include "plamn/simulator/simulator.hpp"
#include "plamn/simulator/simulator_def.hpp"

#include "gtest/gtest.h"
#include "highfive/H5File.hpp"
#include <mpi.h>

#include "damn/core/memory/span.hpp"
#include "plamn/core/boundary/boundary_defs.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <memory>
#include <iostream>
#include <cstdlib>
#include <utility>
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

// phareh5 files of an MHD level, the layout pharesee reads (/mhd/<quantity>, B as /EM_B): per
//  patch the field on its ghost box (the domain in the middle), V converted from the
//  conservative state
TEST(SimulatorModels, mhdDiagnosticsAsPhare)
{
    auto input            = mhd_input({"MHDModel"}, function(wave), 8);
    auto& diags           = input["simulation"]["diagnostics"];
    std::string const dir = ::testing::TempDir() + "plamn_mhd_diags";
    diags["format"]       = std::string{"phareh5"};
    diags["filePath"]     = dir;
    diags["mode"]         = std::string{"overwrite"};
    auto const add = [&](std::string const& type, std::size_t const n, std::string const& q) {
        auto& d               = diags[type][type + std::to_string(n)];
        d["quantity"]         = q;
        d["write_timestamps"] = std::vector<double>{0.};
    };
    add("mhd", 0, "/mhd/rho");
    add("mhd", 1, "/mhd/V");
    add("electromag", 0, "/EM_B");
    Simulator<mhd_only> sim{input};
    sim.initialize();
    ASSERT_TRUE(sim.dump_diagnostics(0., .01));

    // dataset name -> its domain values, patch by patch (in level order: 1D, one rank)
    auto const read = [&](std::string const& file_name, std::string const& name) {
        HighFive::File file{dir + "/" + file_name, HighFive::File::ReadOnly};
        auto const level = file.getGroup("/t/0.0000000000/pl0");
        std::vector<std::pair<int, std::vector<double>>> patches;
        for (auto const& p : level.listObjectNames())
        {
            auto const patch = level.getGroup(p);
            auto const lower = patch.getAttribute("lower").read<std::vector<int>>()[0];
            auto const n     = patch.getAttribute("nbrCells").read<std::vector<std::uint32_t>>()[0];
            auto const all   = patch.getDataSet(name).read<std::vector<double>>();
            auto const ghosts = (all.size() - n) / 2; // cell centred: n domain values
            patches.emplace_back(
                lower, std::vector<double>(all.begin() + ghosts, all.begin() + ghosts + n));
        }
        std::ranges::sort(patches);
        std::vector<double> values;
        for (auto const& [lower, v] : patches)
            values.insert(values.end(), v.begin(), v.end());
        return values;
    };
    auto const rho = read("mhd_rho.h5", "rho");
    ASSERT_EQ(rho.size(), cells);
    auto const expected = sim.domain_values("rho");
    for (std::size_t i = 0; i < cells; ++i)
        EXPECT_FLOAT_EQ(rho[i], expected[i]); // written as float (PLAMN_DIAG_DOUBLES 0)
    for (auto const vy : read("mhd_V.h5", "V_y"))
        EXPECT_NEAR(vy, .2, 1e-6);
    for (auto const bz : read("EM_B.h5", "EM_B_z"))
        EXPECT_NEAR(bz, .2, 1e-6);
}

// a restart of an MHD level: its conservative state, the run from it as the run that wrote it.
//  Not bit for bit: the time restarts from the restart's (start + the steps since), so
//  the step's t1 - t0 differs from the uninterrupted run's in its last bits
TEST(SimulatorModels, mhdRestartContinuesTheRun)
{
    std::string const dir = ::testing::TempDir() + "plamn_mhd_restarts";
    auto input            = mhd_input({"MHDModel"}, function(wave), 8);
    double const dt       = .01;
    double const t_write  = 5 * dt;

    input["simulation"]["restarts"]["filePath"]         = dir;
    input["simulation"]["restarts"]["write_timestamps"] = std::vector<double>{t_write};
    Simulator<mhd_only> first{input};
    first.initialize();
    for (std::size_t i = 0; i < 5; ++i)
        first.advance();
    ASSERT_TRUE(first.dump_restarts(first.currentTime(), dt));
    for (std::size_t i = 0; i < 5; ++i)
        first.advance();

    auto from                                      = mhd_input({"MHDModel"}, function(wave), 8);
    from["simulation"]["restarts"]["loadPath"]     = simulator::restart_dir_for_time(dir, t_write);
    from["simulation"]["restarts"]["restart_time"] = t_write;
    Simulator<mhd_only> second{from};
    second.initialize();
    EXPECT_DOUBLE_EQ(second.currentTime(), t_write);
    for (std::size_t i = 0; i < 5; ++i)
        second.advance();
    for (auto const* name : {"rho", "rhoV_x", "B_y", "Etot"})
    {
        auto const a = first.domain_values(name), b = second.domain_values(name);
        ASSERT_EQ(a.size(), b.size());
        double max = 0;
        for (std::size_t i = 0; i < a.size(); ++i)
            max = std::max(max, std::abs(a[i] - b[i]) / std::max(1., std::abs(a[i])));
        std::cout << name << ": max relative difference " << max << "\n";
        EXPECT_LT(max, 1e-13) << name;
    }
}

namespace
{
    // the input with a refined level over coarse cells lower..upper
    initializer::Dict with_level_1(initializer::Dict input, int const lower, int const upper)
    {
        auto& amr               = input["simulation"]["AMR"];
        amr["max_nbr_levels"]   = 2;
        amr["max_mhd_level"]    = 2;
        auto& box               = amr["refinement"]["boxes"]["L0"];
        box["nbr_boxes"]        = 1;
        box["B0"]["lower"]["x"] = lower;
        box["B0"]["upper"]["x"] = upper;
        return input;
    }
} // namespace

// a refined MHD level from the coarser one: its state
//  refined, each fine cell pair meaning back to its coarse cell (FieldRefiner's dual stencil)
TEST(SimulatorModels, mhdRefinedLevelFromTheCoarser)
{
    auto input              = mhd_input({"MHDModel"}, function(wave), 8);
    auto& amr               = input["simulation"]["AMR"];
    amr["max_nbr_levels"]   = 2;
    amr["max_mhd_level"]    = 2;
    auto& box               = amr["refinement"]["boxes"]["L0"];
    box["nbr_boxes"]        = 1;
    box["B0"]["lower"]["x"] = 8;
    box["B0"]["upper"]["x"] = 23;
    Simulator<mhd_only> sim{input};
    sim.initialize();
    ASSERT_EQ(sim.n_levels(), 2u);
    for (auto const* name : {"rho", "Etot", "rhoV_x", "B_y"})
    {
        auto const coarse = sim.level_values(name, 0);
        auto const fine   = sim.level_values(name, 1);
        ASSERT_EQ(fine.size(), 32u) << name;
        for (std::size_t i = 0; i < 16; ++i)
            EXPECT_NEAR(.5 * (fine[2 * i] + fine[2 * i + 1]), coarse[8 + i], 1e-14) << name;
    }
}


// a refined MHD level advanced in its coarser level's steps: substeps, level ghosts in time,
//  the reflux (the coarser level's fluxes under it replaced by its sums): a periodic level 0
//  keeps its mass, a uniform state stays
TEST(SimulatorModels, mhdRefinedLevelAdvanced)
{
    {
        Simulator<mhd_only> sim{with_level_1(mhd_input({"MHDModel"}, constant(1.3), 8), 8, 23)};
        sim.initialize();
        for (std::size_t i = 0; i < 5; ++i)
            sim.advance();
        for (std::size_t ln = 0; ln < 2; ++ln)
            for (auto const rho : sim.level_values("rho", ln))
                EXPECT_NEAR(rho, 1.3, 1e-13) << ln;
    }
    Simulator<mhd_only> sim{with_level_1(mhd_input({"MHDModel"}, function(wave), 8), 8, 23)};
    sim.initialize();
    auto const mass = [&]() {
        auto const rho = sim.level_values("rho", 0);
        return std::accumulate(rho.begin(), rho.end(), 0.);
    };
    double const m0 = mass();
    for (std::size_t i = 0; i < 10; ++i)
        sim.advance();
    std::cout << "level 0 mass relative change " << (mass() - m0) / m0 << "\n";
    EXPECT_NEAR(mass(), m0, 1e-13 * m0);
    auto const fine = sim.level_values("rho", 1);
    ASSERT_EQ(fine.size(), 32u);
    for (auto const rho : fine)
        EXPECT_TRUE(std::isfinite(rho));
}

// MHD levels from tags (the default tagger, mhd_method, on the MHD state's B): a B wave
//  tags cells, level 1 made from their clusters at initialization, regridded every 2 level 0
//  steps; level 0 keeps its mass (the reflux)
TEST(SimulatorModels, mhdTaggedLevels)
{
    auto input                                 = mhd_input({"MHDModel"}, function(wave), 8);
    auto& sim_                                 = input["simulation"];
    auto& amr                                  = sim_["AMR"];
    amr["max_nbr_levels"]                      = 2;
    amr["max_mhd_level"]                       = 2;
    amr["refinement"]["tagging"]["threshold"]  = 0.01;
    amr["refinement"]["tagging"]["mhd_method"] = std::string{"default"};
    sim_["mhd_state"]["magnetic"]["initializer"]["y_component"] = function(wave);
    Simulator<mhd_only> sim{input};
    sim.initialize();
    ASSERT_EQ(sim.n_levels(), 2u);
    auto const mass = [&]() {
        auto const rho = sim.level_values("rho", 0);
        return std::accumulate(rho.begin(), rho.end(), 0.);
    };
    double const m0 = mass();
    for (std::size_t i = 0; i < 10; ++i)
        sim.advance();
    EXPECT_NEAR(mass(), m0, 1e-13 * m0);
    for (std::size_t ln = 0; ln < sim.n_levels(); ++ln)
        for (auto const rho : sim.level_values("rho", ln))
            EXPECT_TRUE(std::isfinite(rho)) << ln;
}

TEST(SimulatorModels, modelsTheBuildLacksRefused)
{
    EXPECT_THROW((Simulator<mhd_only>{mhd_input({"HybridModel"}, constant(1.))}),
                 std::runtime_error);
}

} // namespace plamn

// MPI: the diagnostics' files are parallel HDF5's
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int const result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
