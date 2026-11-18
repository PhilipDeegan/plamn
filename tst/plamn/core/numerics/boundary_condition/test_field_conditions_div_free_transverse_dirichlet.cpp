#include "plamn/core/def.hpp"
// DivergenceFreeTransverseDirichlet (field_boundary_condition.hpp): past each boundary B's
//  tangential components are Dirichlet at the condition's values (linear: 2 v - the interior
//  value), its normal one keeps every ghost cell's discrete divergence zero (with a uniform B,
//  the interior value), on isotropic and anisotropic meshes

#include "damn/core/geometry/box_spans.hpp"

#include "plamn/core/boundary/boundary_defs.hpp"
#include "plamn/core/numerics/boundary_condition/field_boundary_condition.hpp"

#include "tst/plamn/core/numerics/boundary_condition/bc_test_fixtures.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace plamn::core::bc_test
{
using enum BoundaryLocation;

FieldCondition constexpr div_free_dirichlet(std::array<double, 3> const& values)
{
    return FieldCondition{.type          = FieldConditionType::DivergenceFreeTransverseDirichlet,
                          .value         = values,
                          .extrapolation = DirichletExtrapolation::Linear};
}

// the two ghosts nearest each boundary along n, for each component over its domain nodes along
//  the other directions: the normal component's the interior value, the tangential ones 2 v - it
template<typename VecField, typename Layout_t>
void expect_transverse_dirichlet(VecField const& B, Layout_t const& layout, std::size_t const n,
                                 std::array<double, 3> const& values)
{
    static_assert(Layout_t::ghost_width >= 2);
    for (std::size_t c = 0; c < 3; ++c)
    {
        auto const box      = domain_box(layout, B[c]);
        auto const ps       = box.lower[n];
        auto const pe       = box.upper[n];
        auto const expected = c == n ? interior_value : 2.0 * values[c] - interior_value;
        auto plane          = box;
        plane.upper[n]      = plane.lower[n];
        damn::core::for_each_point(plane, [&](auto const& index) {
            auto const at = [&](std::uint32_t const i) {
                auto node = index;
                node[n]   = i;
                return B[c](node);
            };
            EXPECT_DOUBLE_EQ(at(ps - 2), expected) << "comp=" << c << " lower far ghost";
            EXPECT_DOUBLE_EQ(at(ps - 1), expected) << "comp=" << c << " lower near ghost";
            EXPECT_DOUBLE_EQ(at(pe + 1), expected) << "comp=" << c << " upper near ghost";
            EXPECT_DOUBLE_EQ(at(pe + 2), expected) << "comp=" << c << " upper far ghost";
        });
    }
}

// each ghost cell's discrete divergence, its differences divided by the mesh sizes if spaced
template<typename VecField, typename Layout_t>
void expect_ghost_cells_divergence_free(VecField const& B, Layout_t const& layout,
                                        BoundaryLocation const loc, bool const spaced)
{
    auto constexpr dim = Layout_t::dimension;
    damn::core::for_each_point(ghost_cells(layout, loc), [&](auto const& cell) {
        double div = 0.;
        for (std::size_t d = 0; d < dim; ++d)
        {
            auto up = cell;
            ++up[d];
            div += (B[d](up) - B[d](cell)) / (spaced ? layout.dx[d] : 1.);
        }
        EXPECT_NEAR(div, 0.0, 1e-12) << "divergence at " << static_cast<int>(loc);
    });
}


TEST_F(VecFieldBC1D, DivergenceFreeTransverseDirichletAtXBoundaries)
{
    std::array const values{123.0, 7.0, 11.0};
    apply_vector(div_free_dirichlet(values), B, XLower, ghost_cells(layout, XLower), layout);
    apply_vector(div_free_dirichlet(values), B, XUpper, ghost_cells(layout, XUpper), layout);
    expect_transverse_dirichlet(B, layout, 0, values);
}


TEST_F(VecFieldBC2D, DivergenceFreeTransverseDirichletAtXBoundaries)
{
    std::array const values{123.0, 7.0, 11.0};
    apply_vector(div_free_dirichlet(values), B, XLower, ghost_cells(layout, XLower), layout);
    apply_vector(div_free_dirichlet(values), B, XUpper, ghost_cells(layout, XUpper), layout);
    expect_transverse_dirichlet(B, layout, 0, values);
}

TEST_F(VecFieldBC2D, DivergenceFreeTransverseDirichletAtYBoundaries)
{
    std::array const values{3.0, 123.0, 11.0};
    apply_vector(div_free_dirichlet(values), B, YLower, ghost_cells(layout, YLower), layout);
    apply_vector(div_free_dirichlet(values), B, YUpper, ghost_cells(layout, YUpper), layout);
    expect_transverse_dirichlet(B, layout, 1, values);
}

TEST_F(VecFieldBC2DNonUniformBy, DivergenceFreeTransverseDirichletKeepsXGhostDivergenceZero)
{
    auto const condition = div_free_dirichlet({123.0, 0.0, 11.0});
    apply_vector(condition, B, XLower, ghost_cells(layout, XLower), layout);
    apply_vector(condition, B, XUpper, ghost_cells(layout, XUpper), layout);
    expect_ghost_cells_divergence_free(B, layout, XLower, false);
    expect_ghost_cells_divergence_free(B, layout, XUpper, false);
}


TEST_F(VecFieldBC2DNonUniformByAnisotropic,
       DivergenceFreeTransverseDirichletKeepsXGhostDivergenceZeroOnAnisotropicMesh)
{
    auto const condition = div_free_dirichlet({123.0, 0.0, 11.0});
    apply_vector(condition, B, XLower, ghost_cells(layout, XLower), layout);
    apply_vector(condition, B, XUpper, ghost_cells(layout, XUpper), layout);
    expect_ghost_cells_divergence_free(B, layout, XLower, true);
    expect_ghost_cells_divergence_free(B, layout, XUpper, true);
}


TEST_F(VecFieldBC3D, DivergenceFreeTransverseDirichletAtZBoundaries)
{
    std::array const values{3.0, 7.0, 123.0};
    apply_vector(div_free_dirichlet(values), B, ZLower, ghost_cells(layout, ZLower), layout);
    apply_vector(div_free_dirichlet(values), B, ZUpper, ghost_cells(layout, ZUpper), layout);
    expect_transverse_dirichlet(B, layout, 2, values);
}

} // namespace plamn::core::bc_test

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
