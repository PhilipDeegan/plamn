// SoAVXChunked's gaps (particles_soavx_chunked.hpp): chunks per cell with free slots between them,
// so particles change cells without a full sort. No updater, no pusher: particles' cells are
// changed by hand, as a push would, then resettle() puts them back in their cells' chunks.
//
// every particle's weight is its id (1, 2, ...): after each operation, each chunk must hold the
// ids of exactly the particles in its cell (a reference id -> cell), real particles only (weight
// > 0) in chunks, gaps (weight 0) everywhere else, chunks in cell order without overlaps

#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/point.hpp"
#include "plamn/plamn_core.hpp"
#include "plamn/core/data/particles/particles.hpp"
#include "plamn/core/data/particles/particle_components.hpp"
#include "plamn/simulator/simulator_def.hpp"
#include "tst/plamn/core/data/grid/test_layout_fixtures.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <vector>

namespace plamn::core
{
template<std::size_t dim_>
struct GapsTest : public ::testing::Test
{
    auto constexpr static dim  = dim_;
    auto constexpr static opts = SimOpts{.dimension = dim, .layout_mode = LayoutMode::SoAVXChunked};
    using Layout_t             = PLAMN_Types<opts>::Hybrid::Layout_t;
    using Chunked_t            = Particles<ParticlesOptions{dim, LayoutMode::SoAVXChunked}>;
    using Box_t                = damn::core::Box<int, dim>;
    using Cell_t               = std::array<int, dim>;

    std::uint32_t const cells = 10;
    std::size_t const ppc     = 20;
    Layout_t const layout     = *TestLayout<Layout_t>{cells};
    Box_t const box           = layout.cells;
    std::map<double, Cell_t> where; // id -> cell
    std::mt19937 rng{1234};

    Chunked_t make(double const slack)
    {
        auto ps = make_particles<Chunked_t>(layout);
        damn::core::for_each_point(box, [&](auto const& point) {
            for (std::size_t k = 0; k < ppc; ++k)
            {
                Particle<dim> p;
                p.weight() = static_cast<double>(where.size() + 1);
                for (std::size_t d = 0; d < dim; ++d)
                {
                    p.iCell()[d] = point[d];
                    p.delta()[d] = .5;
                }
                p.v()             = {.1, .2, .3};
                where[p.weight()] = p.iCell();
                ps.push_back(p);
            }
        });
        ps.set_slack(slack, box);
        ps.sort();
        return ps;
    }

    // the chunk of cell, its first slot and size
    static auto chunk(Chunked_t const& ps, Cell_t const& cell) { return ps.cell_range(cell); }

    void expect_consistent(Chunked_t const& ps, std::string const& when) const
    {
        ASSERT_TRUE(ps.sorted()) << when;
        std::vector<bool> in_chunk(ps.size(), false);
        std::size_t previous_end = 0, particles = 0;
        damn::core::for_each_point(ps.shell(), [&](auto const& point) {
            Cell_t cell;
            for (std::size_t d = 0; d < dim; ++d)
                cell[d] = point[d];
            auto const [start, size] = chunk(ps, cell);
            if (size == 0)
                return;
            EXPECT_GE(start, previous_end) << when << ": chunks overlap or out of order";
            previous_end = start + size;
            std::set<double> ids;
            for (std::size_t i = start; i < start + size; ++i)
            {
                in_chunk[i] = true;
                EXPECT_GT(ps.weight(i), 0) << when << ": a gap in a chunk";
                ids.insert(ps.weight(i));
                for (std::size_t d = 0; d < dim; ++d)
                    EXPECT_EQ(ps.iCell(i)[d], cell[d]) << when << ": a particle in another cell";
            }
            std::set<double> expected;
            for (auto const& [id, c] : where)
                if (c == cell)
                    expected.insert(id);
            EXPECT_EQ(ids, expected) << when << ": cell " << cell[0];
            particles += size;
        });
        EXPECT_EQ(particles, where.size()) << when;
        for (std::size_t i = 0; i < ps.size(); ++i)
            if (!in_chunk[i])
                EXPECT_EQ(ps.weight(i), 0) << when << ": a particle outside the chunks, slot " << i;
    }

    // n particles of random cells (in box) moved by +-1 cell, as a push: their new cells
    void move_some(Chunked_t& ps, std::size_t const n)
    {
        std::uniform_int_distribution<std::size_t> any(0, ps.size() - 1);
        std::uniform_int_distribution<int> step(0, 1);
        for (std::size_t k = 0; k < n;)
        {
            auto const i = any(rng);
            if (ps.weight(i) == 0)
                continue;
            auto const d = std::uniform_int_distribution<std::size_t>(0, dim - 1)(rng);
            components(ps).iCell[d][i] += step(rng) ? 1 : -1;
            where[ps.weight(i)] = ps.iCell(i);
            ++k;
        }
    }
};

using Gaps1D = GapsTest<1>;

TEST_F(Gaps1D, sortLeavesTheSlackBetweenChunksBorderCellsMore)
{
    auto const ps = make(.1); // 200 particles, 20 slots: border cells weight 3, 8 inside 1
    expect_consistent(ps, "sorted");
    EXPECT_EQ(ps.nbr_particles(), 200u);
    auto const gap_after = [&](int const x) {
        auto const [start, size] = ps.cell_range(Cell_t{x});
        auto const [next, _]     = ps.cell_range(Cell_t{x + 1});
        return next - (start + size);
    };
    EXPECT_EQ(gap_after(0), 4u); // 20 * 3 / 14
    EXPECT_EQ(gap_after(4), 1u); // 20 * 1 / 14
    EXPECT_EQ(ps.size(), 200u + 2 * 4 + 8 * 1);
}

TEST_F(Gaps1D, withoutSlackChunksAreContiguous)
{
    auto const ps = make(0);
    expect_consistent(ps, "sorted");
    EXPECT_EQ(ps.size(), 200u);
}

TEST_F(Gaps1D, aLeavingParticleFreesTheEndWithTheSmallerGap)
{
    auto ps          = make(.1);
    Cell_t const c5  = {5};
    auto const shell = ps.shell();
    auto const flat
        = [&](Cell_t const& c) { return static_cast<std::size_t>(c[0] - shell.lower[0]); };
    auto const gaps = [&]() {
        auto const [start, size] = ps.cell_range(c5);
        auto const [ps4, n4]     = ps.cell_range(Cell_t{4});
        auto const [ps6, n6]     = ps.cell_range(Cell_t{6});
        return std::array{start - (ps4 + n4), ps6 - (start + size)};
    };
    EXPECT_EQ(gaps(), (std::array<std::size_t, 2>{1, 1}));
    auto const [s0, n0] = ps.cell_range(c5);
    ps.remove(flat(c5), s0 + 3); // a tie: the tail grows
    EXPECT_EQ(gaps(), (std::array<std::size_t, 2>{1, 2}));
    auto const [s1, n1] = ps.cell_range(c5);
    ps.remove(flat(c5), s1 + 3); // the head is smaller now: it grows, both 2
    EXPECT_EQ(gaps(), (std::array<std::size_t, 2>{2, 2}));
    EXPECT_EQ(ps.cell_range(c5).second, n0 - 2);
}

TEST_F(Gaps1D, anArrivingParticleTakesTheLargerGapThenTheOtherThenNeedsASort)
{
    auto ps = make(.1);
    // cell 5's gaps: 1 before (cell 4's), 1 after (its own)
    Particle<1> p;
    p.iCell()  = {5};
    p.delta()  = {.5};
    p.v()      = {0, 0, 0};
    p.weight() = 1000;
    EXPECT_TRUE(ps.insert(p)); // tail (a tie)
    p.weight() = 1001;
    EXPECT_TRUE(ps.insert(p)); // head, cell 4's gap
    p.weight() = 1002;
    EXPECT_FALSE(ps.insert(p)); // none left
    where[1000] = where[1001] = {5};
    expect_consistent(ps, "inserted");
}

TEST_F(Gaps1D, resettleMovesParticlesBetweenCellsWithoutASort)
{
    auto ps = make(.1);
    for (std::size_t round = 0; round < 20; ++round)
    {
        // two particles each way between two cells: room enough
        move_some(ps, 2);
        std::vector<Particle<1>> left;
        bool const sorted = ps.resettle(box, [&](auto const& p) { left.push_back(p); });
        for (auto const& p : left) // out of the box: not this array's any more
        {
            EXPECT_FALSE(isIn(damn::core::Point<int, 1>{p.iCell()}, box));
            where.erase(p.weight());
        }
        expect_consistent(ps, "round " + std::to_string(round) + (sorted ? " (sorted)" : ""));
    }
}

TEST_F(Gaps1D, resettleSortsOnlyWhenAChunkHasNoRoom)
{
    auto ps           = make(.1);
    std::size_t sorts = 0;
    for (std::size_t round = 0; round < 50; ++round)
    {
        move_some(ps, 15);
        std::vector<Particle<1>> left;
        sorts += ps.resettle(box, [&](auto const& p) { left.push_back(p); });
        for (auto const& p : left)
            where.erase(p.weight());
        expect_consistent(ps, "round " + std::to_string(round));
    }
    EXPECT_LT(sorts, 50u); // a full sort only sometimes, not every resettle
    std::cout << "1D: " << sorts << " sorts in 50 resettles of 15 moves" << std::endl;
}

} // namespace plamn::core


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
