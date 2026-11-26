#include "plamn/core/def.hpp"
// boundary locations numbered as SAMRAI's CartesianBoundaryDefines

#include "plamn/core/boundary/boundary_defs.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace plamn::core
{
namespace
{
    // SAMRAI's numbers for each location, the first `count` of the enum, in order
    template<typename Locations>
    void expectSameAsSamrai(Locations const& locations, std::size_t const samraiCount)
    {
        EXPECT_EQ(locations.size(), samraiCount);
        for (auto const& [location, samrai] : locations)
            EXPECT_EQ(static_cast<int>(location), samrai);
    }
} // namespace

TEST(BoundaryDefs, boundaryLocationMatchesSamraiFaces3D)
{
    std::array constexpr locations{
        std::pair{BoundaryLocation::XLower, 0}, // XLO
        std::pair{BoundaryLocation::XUpper, 1}, // XHI
        std::pair{BoundaryLocation::YLower, 2}, // YLO
        std::pair{BoundaryLocation::YUpper, 3}, // YHI
        std::pair{BoundaryLocation::ZLower, 4}, // ZLO
        std::pair{BoundaryLocation::ZUpper, 5}, // ZHI
    };
    expectSameAsSamrai(locations, 6); // NUM_3D_FACES
}

TEST(BoundaryDefs, boundaryLocationMatchesSamraiEdges2D)
{
    std::array constexpr locations{
        std::pair{BoundaryLocation::XLower, 0},
        std::pair{BoundaryLocation::XUpper, 1},
        std::pair{BoundaryLocation::YLower, 2},
        std::pair{BoundaryLocation::YUpper, 3},
    };
    expectSameAsSamrai(locations, 4); // NUM_2D_EDGES
}

TEST(BoundaryDefs, boundaryLocationMatchesSamraiNodes1D)
{
    std::array constexpr locations{
        std::pair{BoundaryLocation::XLower, 0},
        std::pair{BoundaryLocation::XUpper, 1},
    };
    expectSameAsSamrai(locations, 2); // NUM_1D_NODES
}

TEST(BoundaryDefs, codim2BoundaryLocationMatchesSamraiEdges3D)
{
    using enum Codim2BoundaryLocation;
    std::array constexpr locations{
        std::pair{XLower_YLower, 0},  // XLO_YLO
        std::pair{XUpper_YLower, 1},  // XHI_YLO
        std::pair{XLower_YUpper, 2},  // XLO_YHI
        std::pair{XUpper_YUpper, 3},  // XHI_YHI
        std::pair{XLower_ZLower, 4},  // XLO_ZLO
        std::pair{XUpper_ZLower, 5},  // XHI_ZLO
        std::pair{XLower_ZUpper, 6},  // XLO_ZHI
        std::pair{XUpper_ZUpper, 7},  // XHI_ZHI
        std::pair{YLower_ZLower, 8},  // YLO_ZLO
        std::pair{YUpper_ZLower, 9},  // YHI_ZLO
        std::pair{YLower_ZUpper, 10}, // YLO_ZHI
        std::pair{YUpper_ZUpper, 11}, // YHI_ZHI
    };
    expectSameAsSamrai(locations, 12); // NUM_3D_EDGES
}

TEST(BoundaryDefs, codim2BoundaryLocationMatchesSamraiNodes2D)
{
    using enum Codim2BoundaryLocation;
    std::array constexpr locations{
        std::pair{XLower_YLower, 0},
        std::pair{XUpper_YLower, 1},
        std::pair{XLower_YUpper, 2},
        std::pair{XUpper_YUpper, 3},
    };
    expectSameAsSamrai(locations, 4); // NUM_2D_NODES
}

TEST(BoundaryDefs, codim3BoundaryLocationMatchesSamraiNodes3D)
{
    using enum Codim3BoundaryLocation;
    std::array constexpr locations{
        std::pair{XLower_YLower_ZLower, 0}, // XLO_YLO_ZLO
        std::pair{XUpper_YLower_ZLower, 1}, // XHI_YLO_ZLO
        std::pair{XLower_YUpper_ZLower, 2}, // XLO_YHI_ZLO
        std::pair{XUpper_YUpper_ZLower, 3}, // XHI_YHI_ZLO
        std::pair{XLower_YLower_ZUpper, 4}, // XLO_YLO_ZHI
        std::pair{XUpper_YLower_ZUpper, 5}, // XHI_YLO_ZHI
        std::pair{XLower_YUpper_ZUpper, 6}, // XLO_YHI_ZHI
        std::pair{XUpper_YUpper_ZUpper, 7}, // XHI_YHI_ZHI
    };
    expectSameAsSamrai(locations, 8); // NUM_3D_NODES
}

// SAMRAI's counts are all there is: one past the last edge or corner is no location
TEST(BoundaryDefs, adjacentLocationsCoverSamraiCounts)
{
    EXPECT_THROW((void)adjacent_locations(static_cast<Codim2BoundaryLocation>(12)),
                 std::runtime_error);
    EXPECT_THROW((void)adjacent_locations(static_cast<Codim3BoundaryLocation>(8)),
                 std::runtime_error);
}

// an edge or corner joins the faces of its name: SAMRAI's XLO_YHI etc.
TEST(BoundaryDefs, adjacentLocationsMatchSamraiNames)
{
    using enum BoundaryLocation;
    using A2 = std::array<BoundaryLocation, 2>;
    using A3 = std::array<BoundaryLocation, 3>;
    std::array const edges{
        A2{XLower, YLower}, A2{XUpper, YLower}, A2{XLower, YUpper}, A2{XUpper, YUpper},
        A2{XLower, ZLower}, A2{XUpper, ZLower}, A2{XLower, ZUpper}, A2{XUpper, ZUpper},
        A2{YLower, ZLower}, A2{YUpper, ZLower}, A2{YLower, ZUpper}, A2{YUpper, ZUpper},
    };
    for (std::size_t i = 0; i < edges.size(); ++i)
        EXPECT_EQ(adjacent_locations(static_cast<Codim2BoundaryLocation>(i)), edges[i])
            << "edge " << i;

    std::array const corners{
        A3{XLower, YLower, ZLower}, A3{XUpper, YLower, ZLower}, A3{XLower, YUpper, ZLower},
        A3{XUpper, YUpper, ZLower}, A3{XLower, YLower, ZUpper}, A3{XUpper, YLower, ZUpper},
        A3{XLower, YUpper, ZUpper}, A3{XUpper, YUpper, ZUpper},
    };
    for (std::size_t i = 0; i < corners.size(); ++i)
        EXPECT_EQ(adjacent_locations(static_cast<Codim3BoundaryLocation>(i)), corners[i])
            << "corner " << i;
}

} // namespace plamn::core

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
