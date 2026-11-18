#include "plamn/core/def.hpp"
// BoundaryManager (boundary.hpp): which adjacent face an edge or a corner takes the conditions of,
//  by direction (the highest) or by boundary type (none < reflective < inflow < open), and
//  validate_physical_boundaries: every non periodic direction's faces declare a boundary

#include "plamn/core/boundary/boundary.hpp"
#include "plamn/core/boundary/boundary_defs.hpp"
#include "plamn/core/models/quantities/mhd_quantities.hpp"
#include "plamn/initializer/data_provider.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace plamn::core
{
namespace
{
    std::size_t constexpr dimension    = 3;
    std::size_t constexpr NUM_3D_EDGES = 12;
    std::size_t constexpr NUM_3D_NODES = 8;
    using boundary_manager_type        = BoundaryManager<MHDQuantity>;

    boundary_manager_type createBoundaryManager()
    {
        initializer::Dict dict;
        dict["xlower"]["type"] = static_cast<int>(BoundaryType::None);
        dict["xupper"]["type"] = static_cast<int>(BoundaryType::None);
        dict["ylower"]["type"] = static_cast<int>(BoundaryType::Reflective);
        dict["yupper"]["type"] = static_cast<int>(BoundaryType::Reflective);
        dict["zlower"]["type"] = static_cast<int>(BoundaryType::Open);
        dict["zupper"]["type"] = static_cast<int>(BoundaryType::Open);
        return boundary_manager_type{dict, {}, {}};
    }

    // the master's type is at least each adjacent face's
    template<typename Location>
    void expectMasterHasHighestType(boundary_manager_type const& bm, Location const location)
    {
        auto const* masterBoundary = bm.boundary(bm.master(location));
        ASSERT_NE(masterBoundary, nullptr);
        for (auto const loc : adjacent_locations(location))
        {
            auto const* adjacentBoundary = bm.boundary(loc);
            ASSERT_NE(adjacentBoundary, nullptr);
            EXPECT_TRUE(masterBoundary->type >= adjacentBoundary->type);
        }
    }
} // namespace

TEST(BoundaryManager, hasPriorityPolicyByDirection)
{
    auto bm     = createBoundaryManager();
    bm.priority = boundary_manager_type::Priority::ByDirection;

    for (std::size_t i = 0; i < NUM_3D_EDGES; ++i)
    {
        auto const codim2loc = static_cast<Codim2BoundaryLocation>(i);
        EXPECT_EQ(bm.master(codim2loc), adjacent_locations(codim2loc)[1]);
    }

    for (std::size_t i = 0; i < NUM_3D_NODES; ++i)
    {
        auto const codim3loc = static_cast<Codim3BoundaryLocation>(i);
        EXPECT_EQ(bm.master(codim3loc), adjacent_locations(codim3loc)[2]);
    }
}

TEST(BoundaryManager, hasPriorityPolicyByBoundaryTypes)
{
    auto bm     = createBoundaryManager();
    bm.priority = boundary_manager_type::Priority::ByBoundaryType;

    for (std::size_t i = 0; i < NUM_3D_EDGES; ++i)
        expectMasterHasHighestType(bm, static_cast<Codim2BoundaryLocation>(i));

    for (std::size_t i = 0; i < NUM_3D_NODES; ++i)
        expectMasterHasHighestType(bm, static_cast<Codim3BoundaryLocation>(i));
}

// --- validate_physical_boundaries ---------------------------------------------------------

namespace
{
    // a grid sub-dict: per-direction periodicities from `periodicities`, and a boundaries entry
    //  per (location -> type) in `bcs`
    initializer::Dict makeGridDict(std::array<bool, 3> const& periodicities,
                                   std::vector<std::pair<std::string, BoundaryType>> const& bcs)
    {
        initializer::Dict grid;
        grid["periodicities"]["x"] = periodicities[0];
        grid["periodicities"]["y"] = periodicities[1];
        grid["periodicities"]["z"] = periodicities[2];
        for (auto const& [loc, type] : bcs)
            grid["domain_boundaries"][loc]["type"] = static_cast<int>(type);
        return grid;
    }
} // namespace

TEST(ValidatePhysicalBoundaries, acceptsDeclaredPhysicalFaces)
{
    auto const grid = makeGridDict(
        {false, true, true}, {{"xlower", BoundaryType::Open}, {"xupper", BoundaryType::Open}});
    EXPECT_NO_THROW(validate_physical_boundaries<dimension>(grid));
}

TEST(ValidatePhysicalBoundaries, acceptsAllPeriodic)
{
    auto const grid = makeGridDict({true, true, true}, {});
    EXPECT_NO_THROW(validate_physical_boundaries<dimension>(grid));
}

TEST(ValidatePhysicalBoundaries, throwsWhenPhysicalFaceMissing)
{
    // physical x, but only xlower declared -> xupper missing
    auto const grid = makeGridDict({false, true, true}, {{"xlower", BoundaryType::Open}});
    EXPECT_THROW(validate_physical_boundaries<dimension>(grid), std::runtime_error);
}

TEST(ValidatePhysicalBoundaries, throwsWhenPhysicalFaceIsNone)
{
    auto const grid = makeGridDict(
        {false, true, true}, {{"xlower", BoundaryType::Open}, {"xupper", BoundaryType::None}});
    EXPECT_THROW(validate_physical_boundaries<dimension>(grid), std::runtime_error);
}

TEST(ValidatePhysicalBoundaries, noPeriodicityKeyIsSkipped)
{
    initializer::Dict const grid; // minimal dict, no periodicities
    EXPECT_NO_THROW(validate_physical_boundaries<dimension>(grid));
}

} // namespace plamn::core

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
