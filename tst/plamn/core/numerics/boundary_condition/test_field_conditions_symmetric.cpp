#include "plamn/core/def.hpp"
// a scalar's is Neumann; a vector's normal component is Dirichlet 0 (its boundary nodes 0, its
//  ghosts minus their mirrors), its tangential ones Neumann

#include "plamn/core/boundary/boundary_defs.hpp"
#include "plamn/core/numerics/boundary_condition/field_boundary_condition.hpp"

#include "tst/plamn/core/numerics/boundary_condition/bc_test_fixtures.hpp"

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>

namespace plamn::core::bc_test
{
using enum BoundaryLocation;

FieldCondition constexpr symmetric{.type = FieldConditionType::Symmetric};

TEST_F(FieldBC1D, SymmetricScalarEquivalentToNeumann)
{
    Grid_t ref{"rho_ref", layout, qty};
    fill_interior(ref, layout);
    neumann(ref, XLower, ghost_cells(layout, XLower), layout);
    neumann(ref, XUpper, ghost_cells(layout, XUpper), layout);

    apply_scalar(symmetric, field, XLower, ghost_cells(layout, XLower), layout);
    apply_scalar(symmetric, field, XUpper, ghost_cells(layout, XUpper), layout);

    for (std::uint32_t i = 0; i < alloc(0); ++i)
        EXPECT_DOUBLE_EQ(field(i), ref(i)) << "at index " << i;
}


TEST_F(VecFieldBC1D, SymmetricNormalComponentBxSetToDirichletZero)
{
    apply_vector(symmetric, B, XLower, ghost_cells(layout, XLower), layout);
    apply_vector(symmetric, B, XUpper, ghost_cells(layout, XUpper), layout);

    auto const& Bx = B[0];
    auto const ps  = phys_start(0, 0);
    auto const pe  = phys_end(0, 0);
    EXPECT_DOUBLE_EQ(Bx(ps - 1), -interior_value);
    EXPECT_DOUBLE_EQ(Bx(ps), 0.0);
    EXPECT_DOUBLE_EQ(Bx(pe), 0.0);
    EXPECT_DOUBLE_EQ(Bx(pe + 1), -interior_value);
}

TEST_F(VecFieldBC1D, SymmetricTangentialComponentsByBzSetToNeumann)
{
    apply_vector(symmetric, B, XLower, ghost_cells(layout, XLower), layout);
    apply_vector(symmetric, B, XUpper, ghost_cells(layout, XUpper), layout);

    auto const byPhysEnd = phys_end(1, 0);
    for (auto const& comp : {std::size_t{1}, std::size_t{2}})
    {
        auto const& f = B[comp];
        EXPECT_DOUBLE_EQ(f(0u), interior_value) << "component " << comp << " lower ghost";
        EXPECT_DOUBLE_EQ(f(byPhysEnd + 1), interior_value)
            << "component " << comp << " upper ghost";
    }
}


TEST_F(VecFieldBC2D, SymmetricAtXBoundaries)
{
    apply_vector(symmetric, B, XLower, ghost_cells(layout, XLower), layout);
    apply_vector(symmetric, B, XUpper, ghost_cells(layout, XUpper), layout);
    expect_reflected(B, layout, 0, -1., 1., true);
}

TEST_F(VecFieldBC2D, SymmetricAtYBoundaries)
{
    apply_vector(symmetric, B, YLower, ghost_cells(layout, YLower), layout);
    apply_vector(symmetric, B, YUpper, ghost_cells(layout, YUpper), layout);
    expect_reflected(B, layout, 1, -1., 1., true);
}


TEST_F(VecFieldBC3D, SymmetricAtZBoundaries)
{
    apply_vector(symmetric, B, ZLower, ghost_cells(layout, ZLower), layout);
    apply_vector(symmetric, B, ZUpper, ghost_cells(layout, ZUpper), layout);
    expect_reflected(B, layout, 2, -1., 1., true);
}

TEST_F(VecFieldBC3D, SymmetricAtXBoundaries)
{
    apply_vector(symmetric, B, XLower, ghost_cells(layout, XLower), layout);
    apply_vector(symmetric, B, XUpper, ghost_cells(layout, XUpper), layout);
    expect_reflected(B, layout, 0, -1., 1., true);
}

TEST_F(VecFieldBC3D, SymmetricAtYBoundaries)
{
    apply_vector(symmetric, B, YLower, ghost_cells(layout, YLower), layout);
    apply_vector(symmetric, B, YUpper, ghost_cells(layout, YUpper), layout);
    expect_reflected(B, layout, 1, -1., 1., true);
}

} // namespace plamn::core::bc_test

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
