// plamn's tagging criteria on one 2D patch of each Yee centering's B component:
//   cell_centred averages the primal directions over the cell's nodes
//   DefaultCriterion and LohnerCriterion are 0 on a constant, tag a step and nothing far from it
//   LohnerCriterion: 0 on a ramp, 1 / (1 + reltol) at most; a vector as a whole
//   tagged_cells: the union over quantities, each at its own threshold

#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/point.hpp"

#include "plamn/amr/resources.hpp"
#include "plamn/amr/tagging.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"
#include "plamn/simulator/hybrid_patch.hpp"
#include "plamn/simulator/simulator_def.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace plamn::amr
{
namespace
{
    auto constexpr opts = SimOpts{2, 1, 4};
    using Types         = simulator::HybridPatch<opts>::Types;
    using Layout_t      = Types::Layout_t;
    using Grid_t        = Types::Grid_t;
    using Box2          = damn::core::Box<int, 2>;
    using Point2        = damn::core::Point<int, 2>;
    using S             = core::HybridQuantity::Scalar;
    auto constexpr gw   = simulator::HybridPatch<opts>::field_ghost_width;
    using Resource      = FieldResource<Grid_t, gw>;

    Box2 const cells{std::array{0, 0}, std::array{19, 19}};
    std::array<double, 2> const dx{.1, .1};

    struct B
    {
        Layout_t layout{cells, dx, {0., 0.}, 0};
        Grid_t x{"EM_B_x", layout, S::Bx}, y{"EM_B_y", layout, S::By}, z{"EM_B_z", layout, S::Bz};

        // every point of every component from fn(component, x position of the point)
        void set(auto&& fn)
        {
            for (auto* g : {&x, &y, &z})
            {
                auto r               = field_resource<gw>(*g);
                auto const centering = core::centering<2>(*g);
                damn::core::for_each_point(ghost_box(r, cells), [&](auto const& p) {
                    double const at = p[0] + (centering[0] == core::QtyCentering::dual ? .5 : 0.);
                    value(r, cells, p) = fn(g == &x ? 0 : g == &y ? 1 : 2, at);
                });
            }
        }

        auto resources()
        {
            return [this](std::string const& name) {
                return field_resource<gw>(name == "EM_B_x" ? x : name == "EM_B_y" ? y : z);
            };
        }
    };

    std::vector<int> tagged_columns(std::vector<Point2> const& tagged)
    {
        std::vector<int> columns;
        for (auto const& c : tagged)
            if (c[1] == 10)
                columns.push_back(c[0]);
        return columns;
    }
} // namespace


TEST(Tagging, cellCentredAveragesThePrimalDirections)
{
    B b;
    b.set([](int, double x) { return x; }); // the x position of each point
    // Bx is primal in x: nodes 5 and 6 around cell 5's centre (5.5); By dual in x: 5.5 itself
    EXPECT_DOUBLE_EQ(cell_centred(field_resource<gw>(b.x), cells, Point2{std::array{5, 3}}), 5.5);
    EXPECT_DOUBLE_EQ(cell_centred(field_resource<gw>(b.y), cells, Point2{std::array{5, 3}}), 5.5);
}

TEST(Tagging, bothCriteriaTagAStepOnly)
{
    for (auto const method : {"default", "lohner-per-direction"})
    {
        Tagging t;
        if (std::string{method} == "default")
            t.criterion = DefaultCriterion{};
        else
            t.criterion = LohnerCriterion{};
        t.quantities.push_back({tagged_components("B"), 0.1});

        B b;
        b.set([](int, double) { return 1.; });
        EXPECT_TRUE(tagged_cells(t, cells, dx, b.resources()).empty()) << method;

        b.set([](int c, double x) { return c == 1 ? (x < 10 ? 0. : 1.) : 0.; }); // By steps at 10
        auto const columns = tagged_columns(tagged_cells(t, cells, dx, b.resources()));
        ASSERT_FALSE(columns.empty()) << method;
        for (auto const c : columns)
            EXPECT_LE(std::abs(c - 9.5), 3) << method << " column " << c;
    }
}

TEST(Tagging, lohnerIsZeroOnARampAndBoundedOnAStep)
{
    LohnerCriterion const l{.reltol = 0.01, .abstol = 0};
    B b;
    b.set([](int c, double x) { return c == 1 ? 3 + 0.5 * x : 1.; });
    std::vector<Resource> const by{field_resource<gw>(b.y)};
    EXPECT_NEAR(l.indicator(by, cells, Point2{std::array{10, 10}}, dx), 0, 1e-12);

    b.set([](int c, double x) { return c == 1 ? (x < 10 ? 0. : 1.) : 1.; });
    double max = 0;
    for (int i = 5; i < 15; ++i)
        max = std::max(max, l.indicator(by, cells, Point2{std::array{i, 10}}, dx));
    EXPECT_GT(max, 0.5);
    EXPECT_LE(max, 1 / (1 + l.reltol) + 1e-12);
}

TEST(Tagging, lohnerOnBAsAWholeIgnoresASmallComponentOnAStrongField)
{
    LohnerCriterion const l{.reltol = 0.01, .abstol = 1e-8};
    B b; // Bx = 1, By a small step: |B| hardly changes
    b.set([](int c, double x) { return c == 0 ? 1. : c == 1 ? (x < 10 ? -1e-3 : 1e-3) : 0.; });
    std::vector<Resource> const all{field_resource<gw>(b.x), field_resource<gw>(b.y),
                                    field_resource<gw>(b.z)};
    std::vector<Resource> const by{field_resource<gw>(b.y)};
    double whole = 0, alone = 0;
    for (int i = 5; i < 15; ++i)
    {
        whole = std::max(whole, l.indicator(all, cells, Point2{std::array{i, 10}}, dx));
        alone = std::max(alone, l.indicator(by, cells, Point2{std::array{i, 10}}, dx));
    }
    EXPECT_LT(whole, 0.1); // under the default threshold: about 2e-3 / (4 reltol)
    EXPECT_GT(alone, 0.5);
}

TEST(Tagging, eachQuantityAtItsOwnThreshold)
{
    B b; // By steps at 10 by 1, Bz at 15 by 0.1
    b.set([](int c, double x) {
        return c == 1 ? (x < 10 ? 0. : 1.) : c == 2 ? (x < 15 ? 0. : .1) : 0.;
    });
    Tagging t{DefaultCriterion{},
              {{tagged_components("By"), 0.5}, {tagged_components("Bz"), 0.05}}};
    auto const columns = tagged_columns(tagged_cells(t, cells, dx, b.resources()));
    bool near_10 = false, near_15 = false;
    for (auto const c : columns)
    {
        near_10 |= std::abs(c - 9.5) <= 3;
        near_15 |= std::abs(c - 14.5) <= 3;
    }
    EXPECT_TRUE(near_10 and near_15);

    t.quantities[1].threshold = 0.5; // Bz's step is under it now
    for (auto const c : tagged_columns(tagged_cells(t, cells, dx, b.resources())))
        EXPECT_LE(std::abs(c - 9.5), 3) << c;
}

} // namespace plamn::amr


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
