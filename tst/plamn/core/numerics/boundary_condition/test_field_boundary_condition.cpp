#include "plamn/core/def.hpp"
// a patch's ghosts past its lower and upper x boundaries, on MHD fields (rho: dual, Bx: primal
//  along x)

#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "plamn/core/boundary/boundary_defs.hpp"
#include "plamn/core/numerics/boundary_condition/field_boundary_condition.hpp"
#include "plamn/simulator/simulator_def.hpp"
#include "tst/plamn/core/data/grid/test_layout_fixtures.hpp"
#include "tst/plamn/core/data/vecfield/test_vecfield_fixtures.hpp"

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace plamn::core
{
template<std::size_t dim_>
struct BoundaryPatch
{
    auto constexpr static dim = dim_;
    auto constexpr static opts
        = SimOpts{.dimension           = dim,
                  .interp_order        = 1,
                  .reconstruction_type = MHDOpts::ReconstructionType::Linear,
                  .slope_limiter_type  = MHDOpts::SlopeLimiterType::VanLeer,
                  .riemann_solver_type = MHDOpts::RiemannSolverType::Rusanov};
    using Types                   = PLAMN_Types<opts>::MHD;
    using Layout_t                = Types::Layout_t;
    using Grid_t                  = Types::Grid_t;
    auto constexpr static vf_opts = TensorFieldOptions<Types>{};
    using VecField_t              = UsableVecField<vf_opts>;
    using S                       = MHDQuantity::Scalar;
    using V_                      = MHDQuantity::Vector;

    explicit BoundaryPatch(Layout_t const& layout_)
        : layout{layout_}
    {
        fill(rho);
        for (std::size_t c = 0; c < 3; ++c)
            fill(B[c]);
    }

    // distinct values everywhere, ghosts included (a filled ghost must change)
    void fill(auto& f) const
    {
        double v = 1.;
        damn::core::for_each_point(ghost_box(layout, f), [&](auto const& ijk) { f(ijk) = v++; });
    }

    // the ghost cells past the lower or upper x boundary (local index space), the whole patch
    //  along the other directions
    auto ghost_cells(Side const side) const
    {
        auto constexpr gw = Layout_t::ghost_width;
        auto const n      = layout.cells.shape();
        damn::core::Box<std::uint32_t, dim> box;
        for (std::size_t d = 0; d < dim; ++d)
        {
            box.lower[d] = 0;
            box.upper[d] = static_cast<std::uint32_t>(n[d]) + 2 * gw - 1;
        }
        box.lower[0] = side == Side::Lower ? 0 : gw + static_cast<std::uint32_t>(n[0]);
        box.upper[0] = side == Side::Lower ? gw - 1 : box.lower[0] + gw - 1;
        return box;
    }

    Layout_t layout;
    Grid_t rho{"rho", layout, S::rho};
    VecField_t B{"B", layout, V_::B};
};

template<typename P>
using Layout_of = typename P::Layout_t;

template<typename P>
struct FieldConditionTest : public ::testing::Test
{
    using Patch = P;
    TestLayout<typename P::Layout_t> const layout{10};
    P patch{layout};
};

using Dims_t = testing::Types<BoundaryPatch<1>, BoundaryPatch<2>>;
TYPED_TEST_SUITE(FieldConditionTest, Dims_t, );

// ghost i along x and its mirror: dual 2 gw - 1 - i (lower) / 2 (gw + n) - 1 - i (upper);
//  primal 2 gw - i (lower) / 2 (gw + n) - i (upper): the boundary node mirrors itself
template<typename Layout>
std::uint32_t mirror_x(Side const side, QtyCentering const c, std::uint32_t const i, auto n)
{
    auto constexpr gw = Layout::ghost_width;
    auto const b      = side == Side::Lower ? gw : gw + static_cast<std::uint32_t>(n);
    return c == QtyCentering::primal ? 2 * b - i : 2 * b - 1 - i;
}

TYPED_TEST(FieldConditionTest, neumann_copies_the_mirrors)
{
    auto& p      = this->patch;
    using Layout = typename TypeParam::Layout_t;
    auto const n = p.layout.cells.shape()[0];
    for (auto const& [side, loc] : {std::pair{Side::Lower, BoundaryLocation::XLower},
                                    std::pair{Side::Upper, BoundaryLocation::XUpper}})
    {
        auto const box = p.ghost_cells(side);
        neumann(p.rho, loc, box, p.layout);
        damn::core::for_each_point(box, [&](auto ijk) { // rho: dual
            auto m = ijk;
            m[0]   = mirror_x<Layout>(side, QtyCentering::dual, ijk[0], n);
            EXPECT_EQ(p.rho(ijk), p.rho(m));
        });
    }
}

TYPED_TEST(FieldConditionTest, dirichlet_sets_the_boundary_value)
{
    auto& p        = this->patch;
    using Layout   = typename TypeParam::Layout_t;
    auto const n   = p.layout.cells.shape()[0];
    double const v = 7.;
    for (auto const& [side, loc] : {std::pair{Side::Lower, BoundaryLocation::XLower},
                                    std::pair{Side::Upper, BoundaryLocation::XUpper}})
    {
        auto const box = p.ghost_cells(side);
        auto& bx       = p.B[0]; // primal along x: the boundary node is v, the ghosts 2 v - m
        dirichlet(bx, loc, box, p.layout, v);
        auto nodes = box;
        nodes.upper[0] += 1;
        damn::core::for_each_point(nodes, [&](auto ijk) {
            auto m = ijk;
            m[0]   = mirror_x<Layout>(side, QtyCentering::primal, ijk[0], n);
            if (m[0] == ijk[0])
                EXPECT_EQ(bx(ijk), v);
            else
                EXPECT_EQ(bx(ijk), 2 * v - bx(m));
        });
        dirichlet(p.rho, loc, box, p.layout, v, DirichletExtrapolation::Constant);
        damn::core::for_each_point(box, [&](auto ijk) { EXPECT_EQ(p.rho(ijk), v); });
    }
}

TYPED_TEST(FieldConditionTest, reflective_vector_conditions)
{
    // symmetric: the normal component Dirichlet 0, the tangential ones Neumann; antisymmetric:
    //  the other way around
    auto& p          = this->patch;
    using Layout     = typename TypeParam::Layout_t;
    auto const n     = p.layout.cells.shape()[0];
    auto const box   = p.ghost_cells(Side::Upper);
    auto const check = [&](std::size_t const c, bool const zero) {
        auto const& f = p.B[c];
        auto const cx = centering<TypeParam::dim>(f.physicalQuantity())[0];
        auto nodes    = box;
        if (cx == QtyCentering::primal)
            nodes.upper[0] += 1;
        damn::core::for_each_point(nodes, [&](auto ijk) {
            auto m = ijk;
            m[0]   = mirror_x<Layout>(Side::Upper, cx, ijk[0], n);
            if (zero)
                EXPECT_EQ(f(ijk), m[0] == ijk[0] ? 0. : -f(m)) << "component " << c;
            else
                EXPECT_EQ(f(ijk), f(m)) << "component " << c;
        });
    };
    apply_vector(FieldCondition{.type = FieldConditionType::Symmetric}, p.B,
                 BoundaryLocation::XUpper, box, p.layout);
    check(0, true);
    check(1, false);
    check(2, false);
    apply_vector(FieldCondition{.type = FieldConditionType::AntiSymmetric}, p.B,
                 BoundaryLocation::XUpper, box, p.layout);
    check(0, false);
    check(1, true);
    check(2, true);
}

TYPED_TEST(FieldConditionTest, divergence_free_transverse_conditions)
{
    // B's tangential components Neumann (or Dirichlet), then its normal one: every ghost cell
    //  past the boundary has a zero discrete divergence
    auto& p          = this->patch;
    auto constexpr d = TypeParam::dim;
    auto const div   = [&](auto const& cell) {
        double sum = 0;
        for (std::size_t t = 0; t < d; ++t)
        {
            auto up = cell;
            ++up[t];
            sum += (p.B[t](up) - p.B[t](cell)) / p.layout.dx[t];
        }
        return sum;
    };
    for (auto const type : {FieldConditionType::DivergenceFreeTransverseNeumann,
                            FieldConditionType::DivergenceFreeTransverseDirichlet})
        for (auto const& [side, loc] : {std::pair{Side::Lower, BoundaryLocation::XLower},
                                        std::pair{Side::Upper, BoundaryLocation::XUpper}})
        {
            p.fill(p.B[0]);
            p.fill(p.B[1]);
            p.fill(p.B[2]);
            auto const box = p.ghost_cells(side);
            apply_vector(FieldCondition{.type = type, .value = {0., .25, -.5}}, p.B, loc, box,
                         p.layout);
            // every ghost cell: a transverse component is primal along its direction, its
            //  upper face at the last cell is still allocated
            damn::core::for_each_point(box, [&](auto const& cell) {
                double scale = 1.;
                for (std::size_t t = 0; t < d; ++t)
                    scale = std::max(scale, std::abs(p.B[t](cell)) / p.layout.dx[t]);
                EXPECT_NEAR(div(cell), 0., 64 * std::numeric_limits<double>::epsilon() * scale);
            });
            if (type == FieldConditionType::DivergenceFreeTransverseDirichlet)
                damn::core::for_each_point(box, [&](auto const& cell) { // Bz: dual, dual
                    if (cell[0]
                        == (side == Side::Lower ? Layout_of<TypeParam>::ghost_width - 1
                                                : box.lower[0]))
                    {
                        auto m = cell;
                        m[0] += side == Side::Lower ? 1 : -1;
                        EXPECT_DOUBLE_EQ(p.B[2](cell), 2 * -.5 - p.B[2](m));
                    }
                });
        }
}

TEST(BoundaryDefs, locations)
{
    EXPECT_EQ(side_of(BoundaryLocation::YUpper), Side::Upper);
    EXPECT_EQ(direction_of(BoundaryLocation::ZLower), Direction::Z);
    EXPECT_EQ(boundary_location(Direction::Y, Side::Lower), BoundaryLocation::YLower);
    EXPECT_EQ(boundary_location("zupper"), BoundaryLocation::ZUpper);
    EXPECT_THROW((void)boundary_location("xmiddle"), std::runtime_error);
    auto const corner = adjacent_locations(Codim3BoundaryLocation::XUpper_YLower_ZUpper);
    EXPECT_EQ(corner[0], BoundaryLocation::XUpper);
    EXPECT_EQ(corner[1], BoundaryLocation::YLower);
    EXPECT_EQ(corner[2], BoundaryLocation::ZUpper);
}

} // namespace plamn::core

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
