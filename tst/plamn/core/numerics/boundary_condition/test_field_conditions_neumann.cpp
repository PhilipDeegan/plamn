#include "plamn/core/def.hpp"
// Neumann (field_boundary_condition.hpp): a scalar's ghosts past a boundary take the interior
//  value, in 1D, 2D (x boundaries) and 3D (z boundaries)

#include "plamn/core/boundary/boundary_defs.hpp"
#include "plamn/core/numerics/boundary_condition/field_boundary_condition.hpp"

#include "tst/plamn/core/numerics/boundary_condition/bc_test_fixtures.hpp"

#include "gtest/gtest.h"

#include <cstdint>

namespace plamn::core::bc_test
{
TEST_F(FieldBC1D, NeumannSetsLowerGhostToInteriorValue)
{
    neumann(field, BoundaryLocation::XLower, ghost_cells(layout, BoundaryLocation::XLower), layout);

    for (std::uint32_t g = 0; g < gw; ++g)
        EXPECT_DOUBLE_EQ(field(g), interior_value);
}

TEST_F(FieldBC1D, NeumannSetsUpperGhostToInteriorValue)
{
    neumann(field, BoundaryLocation::XUpper, ghost_cells(layout, BoundaryLocation::XUpper), layout);

    for (std::uint32_t g = 0; g < gw; ++g)
        EXPECT_DOUBLE_EQ(field(alloc(0) - 1 - g), interior_value);
}


TEST_F(FieldBC2D, NeumannAtXBoundaries)
{
    neumann(field, BoundaryLocation::XLower, ghost_cells(layout, BoundaryLocation::XLower), layout);
    neumann(field, BoundaryLocation::XUpper, ghost_cells(layout, BoundaryLocation::XUpper), layout);

    auto const allocX = alloc(0);
    for (std::uint32_t g = 0; g < gw; ++g)
        for (auto iy = phys_start(1); iy <= phys_end(1); ++iy)
        {
            EXPECT_DOUBLE_EQ(field(g, iy), interior_value) << "lower g=" << g << " iy=" << iy;
            EXPECT_DOUBLE_EQ(field(allocX - 1 - g, iy), interior_value) << "upper g=" << g;
        }
}


TEST_F(FieldBC3D, NeumannAtZBoundaries)
{
    neumann(field, BoundaryLocation::ZLower, ghost_cells(layout, BoundaryLocation::ZLower), layout);
    neumann(field, BoundaryLocation::ZUpper, ghost_cells(layout, BoundaryLocation::ZUpper), layout);

    auto const allocZ = alloc(2);
    for (std::uint32_t g = 0; g < gw; ++g)
        for (auto ix = phys_start(0); ix <= phys_end(0); ++ix)
            for (auto iy = phys_start(1); iy <= phys_end(1); ++iy)
            {
                EXPECT_DOUBLE_EQ(field(ix, iy, g), interior_value) << "lower g=" << g;
                EXPECT_DOUBLE_EQ(field(ix, iy, allocZ - 1 - g), interior_value) << "upper g=" << g;
            }
}

} // namespace plamn::core::bc_test

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
