#include "plamn/core/def.hpp"
// a scalar's ghosts past a boundary at value v, by linear (2 v - the interior value) or constant
//  (v) extrapolation

#include "plamn/core/boundary/boundary_defs.hpp"
#include "plamn/core/numerics/boundary_condition/field_boundary_condition.hpp"

#include "tst/plamn/core/numerics/boundary_condition/bc_test_fixtures.hpp"

#include "gtest/gtest.h"

#include <cstdint>

namespace plamn::core::bc_test
{
using enum BoundaryLocation;
using enum DirichletExtrapolation;

TEST_F(FieldBC1D, DirichletSetsLowerGhostByLinearExtrapolation)
{
    double const value = 3.0;
    dirichlet(field, XLower, ghost_cells(layout, XLower), layout, value, Linear);

    double const expected = 2.0 * value - interior_value;
    for (std::uint32_t g = 0; g < gw; ++g)
        EXPECT_DOUBLE_EQ(field(g), expected);
}

TEST_F(FieldBC1D, DirichletSetsUpperGhostByLinearExtrapolation)
{
    double const value = 3.0;
    dirichlet(field, XUpper, ghost_cells(layout, XUpper), layout, value, Linear);

    double const expected = 2.0 * value - interior_value;
    for (std::uint32_t g = 0; g < gw; ++g)
        EXPECT_DOUBLE_EQ(field(alloc(0) - 1 - g), expected);
}

TEST_F(FieldBC1D, DirichletSetsGhostsToValueWithConstantExtrapolation)
{
    double const value = 3.0;
    dirichlet(field, XLower, ghost_cells(layout, XLower), layout, value, Constant);
    dirichlet(field, XUpper, ghost_cells(layout, XUpper), layout, value, Constant);

    for (std::uint32_t g = 0; g < gw; ++g)
    {
        EXPECT_DOUBLE_EQ(field(g), value);
        EXPECT_DOUBLE_EQ(field(alloc(0) - 1 - g), value);
    }
}


TEST_F(FieldBC2D, DirichletAtYBoundariesWithConstantExtrapolation)
{
    double const value = 3.0;
    dirichlet(field, YLower, ghost_cells(layout, YLower), layout, value, Constant);
    dirichlet(field, YUpper, ghost_cells(layout, YUpper), layout, value, Constant);

    auto const allocY = alloc(1);
    for (std::uint32_t g = 0; g < gw; ++g)
        for (auto ix = phys_start(0); ix <= phys_end(0); ++ix)
        {
            EXPECT_DOUBLE_EQ(field(ix, g), value) << "lower g=" << g << " ix=" << ix;
            EXPECT_DOUBLE_EQ(field(ix, allocY - 1 - g), value) << "upper g=" << g;
        }
}


TEST_F(FieldBC2D, DirichletAtXBoundaries)
{
    double const value    = 3.0;
    double const expected = 2.0 * value - interior_value;
    dirichlet(field, XLower, ghost_cells(layout, XLower), layout, value, Linear);
    dirichlet(field, XUpper, ghost_cells(layout, XUpper), layout, value, Linear);

    auto const allocX = alloc(0);
    for (std::uint32_t g = 0; g < gw; ++g)
        for (auto iy = phys_start(1); iy <= phys_end(1); ++iy)
        {
            EXPECT_DOUBLE_EQ(field(g, iy), expected) << "lower g=" << g << " iy=" << iy;
            EXPECT_DOUBLE_EQ(field(allocX - 1 - g, iy), expected) << "upper g=" << g;
        }
}

TEST_F(FieldBC2D, DirichletAtYBoundaries)
{
    double const value    = 3.0;
    double const expected = 2.0 * value - interior_value;
    dirichlet(field, YLower, ghost_cells(layout, YLower), layout, value, Linear);
    dirichlet(field, YUpper, ghost_cells(layout, YUpper), layout, value, Linear);

    auto const allocY = alloc(1);
    for (std::uint32_t g = 0; g < gw; ++g)
        for (auto ix = phys_start(0); ix <= phys_end(0); ++ix)
        {
            EXPECT_DOUBLE_EQ(field(ix, g), expected) << "lower g=" << g << " ix=" << ix;
            EXPECT_DOUBLE_EQ(field(ix, allocY - 1 - g), expected) << "upper g=" << g;
        }
}


TEST_F(FieldBC3D, DirichletAtZBoundaries)
{
    double const value    = 3.0;
    double const expected = 2.0 * value - interior_value;
    dirichlet(field, ZLower, ghost_cells(layout, ZLower), layout, value, Linear);
    dirichlet(field, ZUpper, ghost_cells(layout, ZUpper), layout, value, Linear);

    auto const allocZ = alloc(2);
    for (std::uint32_t g = 0; g < gw; ++g)
        for (auto ix = phys_start(0); ix <= phys_end(0); ++ix)
            for (auto iy = phys_start(1); iy <= phys_end(1); ++iy)
            {
                EXPECT_DOUBLE_EQ(field(ix, iy, g), expected) << "lower g=" << g;
                EXPECT_DOUBLE_EQ(field(ix, iy, allocZ - 1 - g), expected) << "upper g=" << g;
            }
}

} // namespace plamn::core::bc_test

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
