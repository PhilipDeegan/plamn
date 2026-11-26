// a periodic 2D coarse level of one patch and a fine patch inside it

#include "damn/amr/hierarchy.hpp"
#include "damn/amr/inter_level.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/level_map.hpp"
#include "damn/amr/topology.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/point.hpp"

#include "plamn/amr/coarsen.hpp"
#include "plamn/amr/refine.hpp"
#include "plamn/amr/resources.hpp"
#include "plamn/amr/split/split.hpp"
#include "plamn/core/data/particles/particle.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"
#include "plamn/simulator/hybrid_patch.hpp"
#include "plamn/simulator/simulator_def.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace plamn::amr
{
namespace
{
    auto constexpr opts = SimOpts{2, 1, 4};
    using Types         = simulator::HybridPatch<opts>::Types;
    using Layout_t      = Types::Layout_t;
    using Grid_t        = Types::Grid_t;
    using Particles_t   = Types::Particles_t;
    using Particle_t    = core::Particle<2>;
    using Box2          = damn::core::Box<int, 2>;
    using Point2        = damn::core::Point<int, 2>;
    using Splitter_t
        = Splitter<damn::core::DimConst<2>, damn::core::InterpConst<1>, RefinedParticlesConst<4>>;
    using S = core::HybridQuantity::Scalar;

    auto constexpr gw  = simulator::HybridPatch<opts>::field_ghost_width;
    auto constexpr pgw = simulator::HybridPatch<opts>::particle_ghost_width;

    std::vector<std::pair<std::string, S>> const qties{{"Bx", S::Bx},  {"By", S::By}, {"Bz", S::Bz},
                                                       {"Ex", S::Ex},  {"Ey", S::Ey}, {"Ez", S::Ez},
                                                       {"rho", S::rho}};

    struct PatchData
    {
        explicit PatchData(Layout_t const& l)
            : layout{l}
            , domain{core::make_particles<Particles_t>(l)}
            , level_ghost{core::make_particles<Particles_t>(l)}
        {
            for (auto const& [name, qty] : qties)
                grids.emplace_back(name, layout, qty);
        }

        Layout_t layout;
        std::vector<Grid_t> grids;
        Particles_t domain, level_ghost;
    };

    void for_each_resource(PatchData& data, auto&& fn)
    {
        for (std::size_t i = 0; i < qties.size(); ++i)
        {
            auto r = field_resource<gw>(data.grids[i]);
            fn(qties[i].first, r);
        }
        auto d = particle_resource<pgw>(data.domain);
        fn("domain", d);
        auto g = particle_resource<pgw>(data.level_ghost);
        fn("level_ghost", g);
    }

    using LevelData_t = damn::amr::LevelData<2, PatchData>;

    Box2 const fine_cells{std::array{8, 8}, std::array{23, 23}}; // coarse 4..11

    struct Levels
    {
        damn::amr::Hierarchy<2> h;
        LevelData_t coarse, fine;
        damn::amr::LevelMap<2> map;
    };

    Levels make_levels()
    {
        Levels l;
        l.h.domain     = Box2{std::array{0, 0}, std::array{15, 15}};
        l.h.dx         = {.4, .4};
        l.h.periodic   = {true, true};
        l.h.max_levels = 2;
        damn::amr::set_level(l.h, damn::amr::make_level(0, std::vector{l.h.domain}));
        damn::amr::set_level(l.h, damn::amr::make_level(1, std::vector{fine_cells}));
        for (int ln = 0; ln < 2; ++ln)
        {
            auto& ld = ln == 0 ? l.coarse : l.fine;
            ld       = damn::amr::allocate(l.h.levels[ln], 0, [&](auto const& p) {
                return PatchData{Layout_t{p.box, damn::amr::mesh_size(l.h, ln),
                                          damn::amr::position(l.h, ln, p.box.lower), ln}};
            });
        }
        l.map = damn::amr::make_level_map(l.h, 0, 0, damn::amr::single_node(), 8);
        return l;
    }

    // a quantity point's physical position on a level of mesh size dx
    std::array<double, 2> position(Grid_t const& g, Point2 const& p, double const dx)
    {
        auto const c = core::centering<2>(g);
        std::array<double, 2> x{};
        for (std::size_t d = 0; d < 2; ++d)
            x[d] = (p[d] + (c[d] == core::QtyCentering::dual ? .5 : 0.)) * dx;
        return x;
    }

    double linear(std::array<double, 2> const& x)
    {
        return 1 + 2 * x[0] + 3 * x[1];
    }

    // fn(resource, grid, cells) for every grid of every patch
    void for_each_grid(LevelData_t& ld, auto&& fn)
    {
        for (std::size_t i = 0; i < ld.data.size(); ++i)
            for (auto& g : ld.data[i].grids)
            {
                auto r = field_resource<gw>(g);
                fn(r, g, ld.patches[i].box);
            }
    }

    std::vector<std::string> names()
    {
        std::vector<std::string> n;
        for (auto const& [name, _] : qties)
            n.push_back(name);
        return n;
    }
} // namespace


TEST(FieldRefiner, exactForLinearFunctionsOnEveryCentering)
{
    for (auto const region :
         {damn::amr::LevelRegion::Interior, damn::amr::LevelRegion::LevelGhosts})
    {
        auto l = make_levels();
        for_each_grid(l.coarse, [&](auto& r, auto& g, auto const& cells) {
            damn::core::for_each_point(ghost_box(r, cells), [&](auto const& p) {
                value(r, cells, p) = linear(position(g, p, .4));
            });
        });
        damn::amr::refine(FieldRefiner{}, region, l.map, damn::amr::same_level_names(names()),
                          l.coarse, l.fine);

        std::size_t n = 0;
        for_each_grid(l.fine, [&](auto& r, auto& g, auto const& cells) {
            auto const box = region == damn::amr::LevelRegion::Interior ? domain_box(r, cells)
                                                                        : ghost_box(r, cells);
            damn::core::for_each_point(box, [&](auto const& p) {
                if (region == damn::amr::LevelRegion::LevelGhosts
                    and damn::core::isIn(p, domain_box(r, cells)))
                    return;
                ++n;
                EXPECT_NEAR(value(r, cells, p), linear(position(g, p, .2)), 1e-12)
                    << g.name() << " " << p;
            });
        });
        EXPECT_GT(n, 0u);
    }
}

TEST(FieldRefiner, divergenceFreeTouchUpKeepsDivBZero)
{
    auto l = make_levels();
    // B from Az on primal nodes: discretely div free on the coarse level
    double constexpr dx = .4, k = 2 * M_PI / (16 * dx);
    auto const Az
        = [&](Point2 const& p) { return std::sin(k * p[0] * dx) * std::sin(2 * k * p[1] * dx); };
    auto& c          = l.coarse.data[0];
    auto const cells = l.coarse.patches[0].box;
    {
        auto bx = field_resource<gw>(c.grids[0]), by = field_resource<gw>(c.grids[1]);
        damn::core::for_each_point(ghost_box(bx, cells), [&](auto const& p) {
            value(bx, cells, p) = (Az(p + Point2{std::array{0, 1}}) - Az(p)) / dx;
        });
        damn::core::for_each_point(ghost_box(by, cells), [&](auto const& p) {
            value(by, cells, p) = -(Az(p + Point2{std::array{1, 0}}) - Az(p)) / dx;
        });
        auto bz = field_resource<gw>(c.grids[2]);
        damn::core::for_each_point(ghost_box(bz, cells),
                                   [&](auto const& p) { value(bz, cells, p) = 0; });
    }
    damn::amr::refine(FieldRefiner{}, damn::amr::LevelRegion::Interior, l.map,
                      damn::amr::same_level_names({"Bx", "By", "Bz"}), l.coarse, l.fine);

    auto& f            = l.fine.data[0];
    auto fbx           = field_resource<gw>(f.grids[0]);
    auto fby           = field_resource<gw>(f.grids[1]);
    auto fbz           = field_resource<gw>(f.grids[2]);
    auto const max_div = [&] {
        double m = 0;
        damn::core::for_each_point(fine_cells, [&](auto const& p) {
            auto const div
                = (value(fbx, fine_cells, p + Point2{std::array{1, 0}}) - value(fbx, fine_cells, p))
                      / .2
                  + (value(fby, fine_cells, p + Point2{std::array{0, 1}})
                     - value(fby, fine_cells, p))
                        / .2;
            m = std::max(m, std::abs(div));
        });
        return m;
    };
    EXPECT_GT(max_div(), 1e-4); // stage 1 alone is not div free
    divergence_free_touch_up(std::array{&fbx, &fby, &fbz}, fine_cells, fine_cells,
                             std::array{.2, .2});
    EXPECT_LT(max_div(), 1e-11);
}

TEST(Coarsen, exactForLinearFunctionsOnEveryCentering)
{
    auto const check = [](auto const coarsener) {
        auto l = make_levels();
        for_each_grid(l.fine, [&](auto& r, auto& g, auto const& cells) {
            damn::core::for_each_point(ghost_box(r, cells), [&](auto const& p) {
                value(r, cells, p) = linear(position(g, p, .2));
            });
        });
        for_each_grid(l.coarse, [&](auto& r, auto&, auto const& cells) {
            damn::core::for_each_point(ghost_box(r, cells),
                                       [&](auto const& p) { value(r, cells, p) = -1; });
        });
        damn::amr::coarsen(coarsener, l.map, damn::amr::same_level_names(names()), l.fine,
                           l.coarse);

        auto const under = damn::core::coarsen(fine_cells, 2);
        for_each_grid(l.coarse, [&](auto& r, auto& g, auto const& cells) {
            damn::core::for_each_point(domain_box(r, cells), [&](auto const& q) {
                if (damn::core::isIn(q, domain_box(r, under)))
                    EXPECT_NEAR(value(r, cells, q), linear(position(g, q, .4)), 1e-12)
                        << g.name() << " " << q;
                else
                    EXPECT_EQ(value(r, cells, q), -1) << g.name() << " " << q;
            });
        });
    };
    check(DefaultFieldCoarsener{});
    check(ElectricFieldCoarsener{});
}

TEST(ParticleSplitter, childrenWhereAskedWeightConserved)
{
    auto l       = make_levels();
    auto& coarse = l.coarse.data[0].domain;
    damn::core::for_each_point(l.h.domain, [&](auto const& c) {
        coarse.emplace_back(Particle_t{1., {c[0], c[1]}, {.3, .6}, {.1, .2, .3}});
    });

    // interior: parents well inside have every child in the fine domain
    damn::amr::refine(ParticleSplitter<Splitter_t>{}, damn::amr::LevelRegion::Interior, l.map,
                      {{"domain", "domain", ""}}, l.coarse, l.fine);
    auto const& fine = l.fine.data[0].domain;
    double weight    = 0;
    for (std::size_t i = 0; i < fine.size(); ++i)
    {
        ASSERT_TRUE(damn::core::isIn(Point2{fine[i].iCell()}, fine_cells));
        weight += fine[i].weight();
    }
    // 8 x 8 coarse cells under the fine patch, children of parents on its edge may fall out or
    // in: the weight per coarse cell is 4 fine cells' worth
    EXPECT_NEAR(weight / (8 * 8 * 4), 1, .2);

    // level ghosts: every child of every parent in the ghost cells off the domain, none in it
    damn::amr::refine(ParticleSplitter<Splitter_t>{}, damn::amr::LevelRegion::LevelGhosts, l.map,
                      {{"domain", "level_ghost", ""}}, l.coarse, l.fine);
    auto const ghosts    = damn::core::grow(fine_cells, pgw);
    std::size_t expected = 0;
    Splitter_t split;
    std::array<Particle_t, 4> children;
    struct
    {
        std::array<Particle_t, 4>* a;
        auto begin() { return a->begin(); }
        auto size() const { return a->size(); }
    } buffer{&children};
    for (std::size_t i = 0; i < coarse.size(); ++i)
    {
        split(to_fine_grid(coarse[i]), coarse[i], buffer);
        for (auto const& child : children)
        {
            Point2 const cell{child.iCell()};
            expected += damn::core::isIn(cell, ghosts) and !damn::core::isIn(cell, fine_cells);
        }
    }
    auto const& lg = l.fine.data[0].level_ghost;
    EXPECT_EQ(lg.size(), expected);
    EXPECT_GT(lg.size(), 0u);
    for (std::size_t i = 0; i < lg.size(); ++i)
    {
        Point2 const cell{lg[i].iCell()};
        EXPECT_TRUE(damn::core::isIn(cell, ghosts) and !damn::core::isIn(cell, fine_cells)) << cell;
        EXPECT_DOUBLE_EQ(lg[i].weight(), 1.); // 4 children of a quarter each, times 4
    }
}

} // namespace plamn::amr


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
