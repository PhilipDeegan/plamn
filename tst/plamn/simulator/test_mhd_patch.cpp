#include "plamn/core/def.hpp"

#include "plamn/core/models/options/mhd_options_def.hpp"
#include "plamn/simulator/mhd_patch.hpp"
#include "plamn/simulator/simulator_def.hpp"

#include "tst/plamn/core/data/grid/test_layout_fixtures.hpp"

#include "gtest/gtest.h"

#include <cstddef>
#include <set>
#include <string>
#include <vector>

namespace plamn::simulator
{
namespace
{
    auto constexpr opts = SimOpts{.dimension           = 2,
                                  .interp_order        = 0,
                                  .nbRefinedPart       = 0, // no hybrid: no split
                                  .reconstruction_type = MHDOpts::ReconstructionType::Linear,
                                  .slope_limiter_type  = MHDOpts::SlopeLimiterType::VanLeer,
                                  .riemann_solver_type = MHDOpts::RiemannSolverType::Rusanov};
    using Patch_t       = MHDPatch<opts>;
    using Layout_t      = Patch_t::Layout_t;

    std::vector<std::string> resource_names(Patch_t& patch)
    {
        std::vector<std::string> names;
        for_each_resource(patch, [&](std::string const& name, auto&) { names.push_back(name); });
        return names;
    }
} // namespace

TEST(MHDPatch, aStatePerSchemeStage)
{
    ::TestLayout<Layout_t> const layout{8};
    Patch_t patch{layout, 3};
    ASSERT_EQ(patch.states.size(), 3u);
    for (std::size_t s = 0; s < patch.states.size(); ++s)
    {
        auto& u = patch.states[s];
        EXPECT_EQ(u.rho.name(), Patch_t::state_name("rho", s));
        // the vector views write their own grids
        for (std::size_t c = 0; c < 3; ++c)
        {
            EXPECT_TRUE(u.rhoV[c].isUsable());
            EXPECT_EQ(u.B[c].data(), u.B.grids[c].data());
        }
    }
    EXPECT_EQ(patch.states[2].rhoV.grids[0].name(), "rhoV_s2_x");
    EXPECT_EQ(patch.E[1].data(), patch.E.grids[1].data());
}

TEST(MHDPatch, resourcesNamedOnce)
{
    ::TestLayout<Layout_t> const layout{8};
    Patch_t patch{layout, 2};
    auto const names = resource_names(patch);
    std::set<std::string> const unique{names.begin(), names.end()};
    EXPECT_EQ(unique.size(), names.size());
    // per state 3 scalars and 3 vectors, the old state's conservative 2 scalars and 2 vectors,
    //  then E
    EXPECT_EQ(names.size(), 2u * (3 + 3 * 3) + (2 + 2 * 3) + 3);
    for (auto const* name : {"rho", "rhoV_x", "B_z", "Etot", "V_y", "P", "rho_s1", "B_s1_x",
                             "rho_old", "rhoV_old_y", "B_old_z", "Etot_old", "E_x", "E_z"})
        EXPECT_TRUE(unique.contains(name)) << name;
}

} // namespace plamn::simulator

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
