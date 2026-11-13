#include "plamn/core/def.hpp"
//  tst/plamn/core/data/grid/test_layout.cpp
//
//  fixed values for the stateless layout functions (layout.hpp) and the hybrid Yee
//  centerings, worked out by hand from the Yee definitions: the reference for
//  replacing Layout with layout.hpp

#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"

#include "plamn/core/data/grid/layout.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cstdint>

namespace plamn::core
{
namespace
{
    auto constexpr P = QtyCentering::primal;
    auto constexpr D = QtyCentering::dual;

    using C3 = std::array<QtyCentering, 3>;

    template<typename T, std::size_t dim>
    auto box(std::array<T, dim> const& lower, std::array<T, dim> const& upper)
    {
        return damn::core::Box<T, dim>{lower, upper};
    }

    template<typename T, std::size_t dim>
    auto point(std::array<T, dim> const& p)
    {
        return damn::core::Point<T, dim>{p};
    }

    // n cells, gw ghosts, per direction
    std::uint32_t constexpr n  = 10;
    std::uint32_t constexpr gw = 2;
} // namespace


// Yee: B on faces, E and J on edges, moments on primal nodes
using S = HybridQuantity::Scalar;
static_assert(quantity(S::Bx).centering == C3{P, D, D});
static_assert(quantity(S::By).centering == C3{D, P, D});
static_assert(quantity(S::Bz).centering == C3{D, D, P});
static_assert(quantity(S::Ex).centering == C3{D, P, P});
static_assert(quantity(S::Ey).centering == C3{P, D, P});
static_assert(quantity(S::Ez).centering == C3{P, P, D});
static_assert(quantity(S::Jx).centering == C3{D, P, P});
static_assert(quantity(S::Jy).centering == C3{P, D, P});
static_assert(quantity(S::Jz).centering == C3{P, P, D});
static_assert(quantity(S::rho).centering == C3{P, P, P});
static_assert(quantity(S::Vx).centering == C3{P, P, P});
static_assert(quantity(S::P).centering == C3{P, P, P});
static_assert(quantity(S::Mxy).centering == C3{P, P, P});


// local index space: ghost inclusive, first ghost node is 0
// primal: n + 1 physical nodes, dual: n
static_assert(physical_start(gw) == 2);
static_assert(physical_end(P, n, gw) == 12);
static_assert(physical_end(D, n, gw) == 11);
static_assert(ghost_end(P, n, gw) == 14);
static_assert(ghost_end(D, n, gw) == 13);

static_assert(flip(P) == D && flip(D) == P);
static_assert(next_index(P, 4) == 5 && next_index(D, 4) == 4);
static_assert(prev_index(P, 4) == 4 && prev_index(D, 4) == 3);


TEST(Layout, nodeCountsAndAllocShape1D)
{
    auto const cells = box<int, 1>({0}, {9});
    EXPECT_EQ(physical_nodes(cells, {P}), (std::array<std::uint32_t, 1>{11}));
    EXPECT_EQ(physical_nodes(cells, {D}), (std::array<std::uint32_t, 1>{10}));
    EXPECT_EQ(alloc_shape(cells, {P}, gw), (std::array<std::uint32_t, 1>{15}));
    EXPECT_EQ(alloc_shape(cells, {D}, gw), (std::array<std::uint32_t, 1>{14}));
    EXPECT_EQ(alloc_shape(cells, {P}, 0), (std::array<std::uint32_t, 1>{11}));
}

TEST(Layout, allocShapeOfBx3D)
{
    auto const cells = box<int, 3>({0, 0, 0}, {9, 7, 5});
    auto const c     = quantity(S::Bx).centering;
    EXPECT_EQ(alloc_shape(cells, c, gw), (std::array<std::uint32_t, 3>{15, 12, 10}));
}


// patch away from the origin, negative lower bound, (primal, dual)
class Layout2D : public ::testing::Test
{
protected:
    damn::core::Box<int, 2> const cells = box<int, 2>({5, -3}, {14, 4}); // 10 x 8 cells
    std::array<QtyCentering, 2> const c{P, D};
};

TEST_F(Layout2D, allocShape)
{
    EXPECT_EQ(physical_nodes(cells, c), (std::array<std::uint32_t, 2>{11, 8}));
    EXPECT_EQ(alloc_shape(cells, c, gw), (std::array<std::uint32_t, 2>{15, 12}));
}

TEST_F(Layout2D, localBoxes)
{
    EXPECT_EQ(local_domain_box(cells, c, gw), (box<std::uint32_t, 2>({2, 2}, {12, 9})));
    EXPECT_EQ(local_ghost_box(cells, c, gw), (box<std::uint32_t, 2>({0, 0}, {14, 11})));
}

TEST_F(Layout2D, amrBoxes)
{
    EXPECT_EQ(amr_box(cells, c), (box<int, 2>({5, -3}, {15, 4})));
    EXPECT_EQ(amr_ghost_box(cells, c, gw), (box<int, 2>({3, -5}, {17, 6})));
}

TEST_F(Layout2D, amrLocalConversions)
{
    EXPECT_EQ(amr_to_local(cells, point<int, 2>({5, -3}), gw), (point<std::uint32_t, 2>({2, 2})));
    EXPECT_EQ(amr_to_local(cells, point<int, 2>({3, -5}), gw), (point<std::uint32_t, 2>({0, 0})));
    EXPECT_EQ(local_to_amr(cells, point<std::uint32_t, 2>({14, 11}), gw),
              (point<int, 2>({17, 6})));

    // the ghost box is the same box in both index spaces
    EXPECT_EQ(amr_to_local(cells, amr_ghost_box(cells, c, gw), gw), local_ghost_box(cells, c, gw));
    EXPECT_EQ(local_to_amr(cells, local_ghost_box(cells, c, gw), gw), amr_ghost_box(cells, c, gw));
}

TEST_F(Layout2D, coordinates)
{
    std::array<double, 2> const dx{0.1, 0.2};
    auto const amr = point<int, 2>({3, -2});

    auto const x = coordinate(dx, amr, c); // primal on the node, dual half a cell up
    EXPECT_DOUBLE_EQ(x[0], 0.3);
    EXPECT_DOUBLE_EQ(x[1], -0.3);

    auto const cc = cell_center(dx, amr);
    EXPECT_DOUBLE_EQ(cc[0], 0.35);
    EXPECT_DOUBLE_EQ(cc[1], -0.3);

    EXPECT_DOUBLE_EQ(cell_volume(dx), 0.02);
}

TEST_F(Layout2D, nextPrevious)
{
    auto const p = point<int, 2>({3, -2});
    EXPECT_EQ(next<Direction::X>(p), (point<int, 2>({4, -2})));
    EXPECT_EQ(previous<Direction::Y>(p), (point<int, 2>({3, -3})));
}


// ghosts mirrored across the physical boundary, local index space
TEST(Layout, boundaryMirrored)
{
    // lower boundary sits on primal node 2, dual nodes 1 and 2 straddle it
    EXPECT_EQ(boundary_mirrored(P, Side::Lower, n, gw, 1), 3u);
    EXPECT_EQ(boundary_mirrored(P, Side::Lower, n, gw, 0), 4u);
    EXPECT_EQ(boundary_mirrored(D, Side::Lower, n, gw, 1), 2u);
    EXPECT_EQ(boundary_mirrored(D, Side::Lower, n, gw, 0), 3u);

    // upper boundary sits on primal node 12, dual nodes 11 and 12 straddle it
    EXPECT_EQ(boundary_mirrored(P, Side::Upper, n, gw, 13), 11u);
    EXPECT_EQ(boundary_mirrored(P, Side::Upper, n, gw, 14), 10u);
    EXPECT_EQ(boundary_mirrored(D, Side::Upper, n, gw, 12), 11u);
    EXPECT_EQ(boundary_mirrored(D, Side::Upper, n, gw, 13), 10u);
}


// PatchLayout: the same 2D patch, through quantities instead of centerings
namespace
{
    struct TestOptions
    {
        std::size_t dimension;
        std::uint32_t field_ghost_width;
    };

    struct HasBx // stands in for a field
    {
        auto physicalQuantity() const { return S::Bx; }
    };
} // namespace

using Layout_t = PatchLayout<TestOptions{2, gw}>;

static_assert(centering<2>(S::Bx) == std::array{P, D});
static_assert(centering<2>(HybridQuantity::Vector::E)
              == std::array{std::array{D, P}, std::array{P, D}, std::array{P, P}});

class PatchLayout2D : public ::testing::Test
{
protected:
    Layout_t const L{box<int, 2>({5, -3}, {14, 4}), {0.1, 0.2}, {}, 1};
};

TEST_F(PatchLayout2D, centeringFromQuantityOrField)
{
    EXPECT_EQ(centering(L, S::Bx), (std::array{P, D}));
    EXPECT_EQ(centering(L, HasBx{}), (std::array{P, D}));
    EXPECT_EQ(centering(L, S::rho), (std::array{P, P}));
}

TEST_F(PatchLayout2D, sizesAndBoxes)
{
    EXPECT_EQ(cell_count(L), (std::array<std::uint32_t, 2>{10, 8}));
    EXPECT_EQ(physical_nodes(L, S::Bx), (std::array<std::uint32_t, 2>{11, 8}));
    EXPECT_EQ(alloc_shape(L, HasBx{}), (std::array<std::uint32_t, 2>{15, 12}));
    EXPECT_EQ(alloc_shape(L, S::rho), (std::array<std::uint32_t, 2>{15, 13}));
    EXPECT_EQ(domain_box(L, S::Bx), (box<std::uint32_t, 2>({2, 2}, {12, 9})));
    EXPECT_EQ(ghost_box(L, S::Bx), (box<std::uint32_t, 2>({0, 0}, {14, 11})));
    EXPECT_EQ(amr_box(L, S::Bx), (box<int, 2>({5, -3}, {15, 4})));
    EXPECT_EQ(amr_ghost_box(L, S::Bx), (box<int, 2>({3, -5}, {17, 6})));
}

TEST_F(PatchLayout2D, indexSpaces)
{
    EXPECT_EQ(to_local(L, point<int, 2>({5, -3})), (point<std::uint32_t, 2>({2, 2})));
    EXPECT_EQ(to_amr(L, point<std::uint32_t, 2>({14, 11})), (point<int, 2>({17, 6})));
    EXPECT_EQ(to_local(L, amr_ghost_box(L, S::Bx)), ghost_box(L, S::Bx));

    std::size_t n_points = 0;
    for_each_amr_local(L, amr_box(L, S::Bx), [&](auto const& amr, auto const& lcl) {
        EXPECT_EQ(to_local(L, amr), lcl);
        ++n_points;
    });
    EXPECT_EQ(n_points, 11u * 8u);
}

TEST_F(PatchLayout2D, physicalSpace)
{
    auto const x = coordinate(L, S::Bx, point<int, 2>({3, -2}));
    EXPECT_DOUBLE_EQ(x[0], 0.3);
    EXPECT_DOUBLE_EQ(x[1], -0.3);
    EXPECT_DOUBLE_EQ(cell_center(L, point<int, 2>({3, -2}))[0], 0.35);
    EXPECT_DOUBLE_EQ(cell_volume(L), 0.02);
    EXPECT_DOUBLE_EQ(inverse_mesh_size(L)[1], 5.0);
}

TEST_F(PatchLayout2D, subLayout)
{
    auto const sub = sub_layout(L, box<int, 2>({7, 0}, {9, 3}));
    EXPECT_EQ(sub.cells, (box<int, 2>({7, 0}, {9, 3})));
    EXPECT_EQ(sub.level, 1);
    EXPECT_DOUBLE_EQ(sub.origin[0], 0.7);
    EXPECT_DOUBLE_EQ(sub.origin[1], 0.0);
    EXPECT_EQ(alloc_shape(sub, S::Bx), (std::array<std::uint32_t, 2>{8, 8}));
}

} // namespace plamn::core


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
