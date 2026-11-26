#include "plamn/core/def.hpp"
// a scalar's is Dirichlet 0; a vector's normal component is Neumann, its tangential ones
//  Dirichlet 0 (ghosts minus their mirrors)

#include "plamn/core/boundary/boundary_defs.hpp"
#include "plamn/core/numerics/boundary_condition/field_boundary_condition.hpp"

#include "tst/plamn/core/numerics/boundary_condition/bc_test_fixtures.hpp"

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>

namespace plamn::core::bc_test
{
using enum BoundaryLocation;

FieldCondition constexpr antisymmetric{.type = FieldConditionType::AntiSymmetric};

TEST_F(FieldBC1D, AntiSymmetricScalarEquivalentToDirichletZero)
{
    Grid_t ref{"rho_ref", layout, qty};
    fill_interior(ref, layout);
    dirichlet(ref, XLower, ghost_cells(layout, XLower), layout, 0.0,
              DirichletExtrapolation::Linear);
    dirichlet(ref, XUpper, ghost_cells(layout, XUpper), layout, 0.0,
              DirichletExtrapolation::Linear);

    apply_scalar(antisymmetric, field, XLower, ghost_cells(layout, XLower), layout);
    apply_scalar(antisymmetric, field, XUpper, ghost_cells(layout, XUpper), layout);

    for (std::uint32_t i = 0; i < alloc(0); ++i)
        EXPECT_DOUBLE_EQ(field(i), ref(i)) << "at index " << i;
}


TEST_F(VecFieldBC1D, AntiSymmetricNormalComponentBxSetToNeumann)
{
    apply_vector(antisymmetric, B, XLower, ghost_cells(layout, XLower), layout);
    apply_vector(antisymmetric, B, XUpper, ghost_cells(layout, XUpper), layout);

    auto const& Bx = B[0];
    EXPECT_DOUBLE_EQ(Bx(phys_start(0, 0) - 1), interior_value);
    EXPECT_DOUBLE_EQ(Bx(phys_end(0, 0) + 1), interior_value);
}

TEST_F(VecFieldBC1D, AntiSymmetricTangentialComponentsByBzSetToDirichletZero)
{
    apply_vector(antisymmetric, B, XLower, ghost_cells(layout, XLower), layout);
    apply_vector(antisymmetric, B, XUpper, ghost_cells(layout, XUpper), layout);

    auto const byPhysEnd = phys_end(1, 0);
    for (auto const& comp : {std::size_t{1}, std::size_t{2}})
    {
        auto const& f = B[comp];
        EXPECT_DOUBLE_EQ(f(0u), -interior_value) << "component " << comp << " lower ghost";
        EXPECT_DOUBLE_EQ(f(byPhysEnd + 1), -interior_value)
            << "component " << comp << " upper ghost";
    }
}


TEST_F(VecFieldBC2D, AntiSymmetricAtXBoundaries)
{
    apply_vector(antisymmetric, B, XLower, ghost_cells(layout, XLower), layout);
    apply_vector(antisymmetric, B, XUpper, ghost_cells(layout, XUpper), layout);
    expect_reflected(B, layout, 0, 1., -1., false);
}

TEST_F(VecFieldBC2D, AntiSymmetricAtYBoundaries)
{
    apply_vector(antisymmetric, B, YLower, ghost_cells(layout, YLower), layout);
    apply_vector(antisymmetric, B, YUpper, ghost_cells(layout, YUpper), layout);
    expect_reflected(B, layout, 1, 1., -1., false);
}


TEST_F(VecFieldBC3D, AntiSymmetricAtZBoundaries)
{
    apply_vector(antisymmetric, B, ZLower, ghost_cells(layout, ZLower), layout);
    apply_vector(antisymmetric, B, ZUpper, ghost_cells(layout, ZUpper), layout);
    expect_reflected(B, layout, 2, 1., -1., false);
}

TEST_F(VecFieldBC3D, AntiSymmetricAtXBoundaries)
{
    apply_vector(antisymmetric, B, XLower, ghost_cells(layout, XLower), layout);
    apply_vector(antisymmetric, B, XUpper, ghost_cells(layout, XUpper), layout);
    expect_reflected(B, layout, 0, 1., -1., false);
}

TEST_F(VecFieldBC3D, AntiSymmetricAtYBoundaries)
{
    apply_vector(antisymmetric, B, YLower, ghost_cells(layout, YLower), layout);
    apply_vector(antisymmetric, B, YUpper, ghost_cells(layout, YUpper), layout);
    expect_reflected(B, layout, 1, 1., -1., false);
}

} // namespace plamn::core::bc_test

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
