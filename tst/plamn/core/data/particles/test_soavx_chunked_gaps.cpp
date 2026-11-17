// SoAVXChunked's gaps (particles_soavx_chunked.hpp), 1D, 2D and 3D: chunks per cell with free slots
// between them, so particles change cells without a full sort. No updater, no pusher: particles'
// cells are changed by hand, as a push would, then resettle() puts them back in their cells'
// chunks.
//
// every particle's weight is its id (1, 2, ...): after each operation, each chunk must hold the
// ids of exactly the particles in its cell (a reference id -> cell), real particles only (weight
// > 0) in chunks, gaps (weight 0) everywhere else, chunks in cell order without overlaps

#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/sys/env.hpp"
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
#include <tuple>
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
    std::mt19937 rng{1234};         // NOLINT(bugprone-random-generator-seed): reproducible

    Chunked_t make(double const slack, double const border = 3)
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
        ps.set_slack(slack, box, border);
        ps.sort();
        return ps;
    }

    // the chunk of cell, its first slot and size
    static auto chunk(Chunked_t const& ps, Cell_t const& cell) { return ps.cell_range(cell); }

    void expect_consistent(Chunked_t const& ps, std::string const& when) const
    {
        ASSERT_TRUE(ps.sorted()) << when;
        std::map<Cell_t, std::set<double>> ids_of; // the reference, by cell
        for (auto const& [id, c] : where)
            ids_of[c].insert(id);
        std::vector<bool> in_chunk(ps.size(), false);
        std::size_t previous_end = 0, particles = 0;
        damn::core::for_each_point(ps.shell(), [&](auto const& point) {
            Cell_t cell;
            for (std::size_t d = 0; d < dim; ++d)
                cell[d] = point[d];
            auto const [start, size] = chunk(ps, cell);
            auto const expected      = ids_of.contains(cell) ? ids_of[cell] : std::set<double>{};
            EXPECT_EQ(size, expected.size()) << when;
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
            EXPECT_EQ(ids, expected) << when << ": cell " << cell[0];
            particles += size;
        });
        EXPECT_EQ(particles, where.size()) << when;
        for (std::size_t i = 0; i < ps.size(); ++i)
        {
            if (!in_chunk[i])
            {
                EXPECT_EQ(ps.weight(i), 0) << when << ": a particle outside the chunks, slot " << i;
            }
        }
    }

    // as move_some, the moves recorded as the push records them: distinct slots, ascending
    //  (chunk order), each with the cell it left
    void move_some_recorded(Chunked_t& ps, std::size_t const n)
    {
        std::uniform_int_distribution<std::size_t> any(0, ps.size() - 1);
        std::uniform_int_distribution<int> step(0, 1);
        std::set<std::size_t> slots;
        while (slots.size() < n)
            if (auto const i = any(rng); ps.weight(i) != 0)
                slots.insert(i);
        ps.begin_crossers();
        for (auto const i : slots)
        {
            auto const from = ps.iCell(i);
            auto const d    = std::uniform_int_distribution<std::size_t>(0, dim - 1)(rng);
            components(ps).iCell[d][i] += step(rng) ? 1 : -1;
            where[ps.weight(i)] = ps.iCell(i);
            ps.add_crosser(i, from);
        }
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

// a recorded resettle (SoAVXChunked::resettle from the push's crossings): the same particles
//  in the same chunks as the scan, rounds of moves with the slots recorded as the push does
template<typename Test>
void recorded_rounds(Test& t, std::size_t const rounds, std::size_t const moves)
{
    auto ps           = t.make(.1);
    std::size_t sorts = 0, left = 0;
    for (std::size_t round = 0; round < rounds; ++round)
    {
        t.move_some_recorded(ps, moves);
        std::vector<Particle<Test::dim>> out;
        sorts += ps.resettle(t.box, [&](auto const& p) { out.push_back(p); });
        for (auto const& p : out)
            t.where.erase(p.weight());
        left += out.size();
        t.expect_consistent(ps, "recorded round " + std::to_string(round));
    }
    std::cout << Test::dim << "D recorded: " << sorts << " sorts in " << rounds << " resettles of "
              << moves << " moves, " << left << " left" << std::endl;
}

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
    EXPECT_EQ(gap_after(0), 4u);     // 20 * 3 / 14, rounded
    EXPECT_EQ(gap_after(4), 1u);     // its share of the running total: 1 or 2
    EXPECT_EQ(ps.size(), 200u + 20); // the slack exactly
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

TEST_F(Gaps1D, anArrivingParticleTakesTheLargerGapThenTheOtherThenNeighboursSlide)
{
    auto ps = make(.1);
    // cell 5's gaps: 1 before (cell 4's), 1 after (its own); cell 6's after it 1, 4's before it 1
    Particle<1> p;
    p.iCell()         = {5};
    p.delta()         = {.5};
    p.v()             = {0, 0, 0};
    auto const arrive = [&](double const id) {
        p.weight()    = id;
        bool const in = ps.insert(p);
        if (in)
            where[id] = {5};
        return in;
    };
    EXPECT_TRUE(arrive(1000)); // tail (a tie)
    EXPECT_TRUE(arrive(1001)); // head, cell 4's gap
    EXPECT_TRUE(arrive(1002)); // cell 6 slides up into its tail gap
    expect_consistent(ps, "inserted");
    // then cascading slides, until no chunk within reach has a gap
    std::size_t in = 3;
    while (arrive(1000. + static_cast<double>(in)))
        ++in;
    expect_consistent(ps, "full");
    EXPECT_GT(in, 4u);
    EXPECT_LE(in, ps.size() - 200); // the gaps of the whole array at most
    EXPECT_EQ(ps.cell_range(Cell_t{5}).second, ppc + in);
}

// an overflowing arrival appended (push_back) until the next sort: the chunks no longer bound
//  the array, the last cell's tail "gap" would be the appended particles
TEST_F(Gaps1D, noInsertOnceAppendedToUntilSorted)
{
    auto ps = make(.1);
    Particle<1> p;
    p.iCell()  = {box.upper[0]};
    p.delta()  = {.5};
    p.v()      = {0, 0, 0};
    p.weight() = 1000;
    ps.push_back(p);
    where[p.weight()] = p.iCell();
    p.weight()        = 1001;
    EXPECT_FALSE(ps.insert(p));
    ps.push_back(p);
    where[p.weight()] = p.iCell();
    ps.sort();
    expect_consistent(ps, "appended then sorted");
    p.weight() = 1002;
    EXPECT_TRUE(ps.insert(p));
    where[p.weight()] = p.iCell();
    expect_consistent(ps, "inserted after the sort");
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


using Gaps2D = GapsTest<2>;

// the gap after a cell's chunk: up to the next chunk in memory (the next cell along y)
std::size_t gap_after(auto const& ps, std::array<int, 2> const& cell)
{
    auto const [start, size] = ps.cell_range(cell);
    auto const [next, _]     = ps.cell_range(std::array<int, 2>{cell[0], cell[1] + 1});
    return next - (start + size);
}

TEST_F(Gaps2D, sortLeavesTheSlackBetweenChunksBorderCellsMore)
{
    // 2000 particles, 200 slots: 36 border cells weight 3, 64 inside 1 (172)
    auto const ps = make(.1);
    expect_consistent(ps, "sorted");
    EXPECT_EQ(ps.nbr_particles(), 2000u);
    for (auto const& cell : {std::array{0, 0}, std::array{0, 4}}) // the x = 0 border
    {
        EXPECT_GE(gap_after(ps, cell), 3u); // 200 * 3 / 172: 3 or 4
        EXPECT_LE(gap_after(ps, cell), 4u);
    }
    EXPECT_GE(gap_after(ps, {4, 4}), 1u); // 200 / 172: 1 or 2
    EXPECT_LE(gap_after(ps, {4, 4}), 2u);
    EXPECT_EQ(ps.size(), 2000u + 200); // the slack exactly
}

TEST_F(Gaps2D, aMoveAlongXUsesTheGapsOfAChunkARowAway)
{
    auto ps = make(.1);
    // particles of (4, 4) to (5, 4), as many as (5, 4)'s gaps: its own and (5, 3)'s
    std::size_t const room   = gap_after(ps, {5, 3}) + gap_after(ps, {5, 4});
    auto const [start, size] = ps.cell_range(Cell_t{4, 4});
    for (std::size_t i = start; i < start + room; ++i)
    {
        components(ps).iCell[0][i] = 5;
        where[ps.weight(i)]        = ps.iCell(i);
    }
    EXPECT_FALSE(ps.resettle(box, [](auto const&) { FAIL() << "nothing leaves the box"; }));
    expect_consistent(ps, "moved along x");
    EXPECT_EQ(ps.cell_range(Cell_t{5, 4}).second, ppc + room);

    // the rest of (4, 4), one at a time: (5, 4)'s neighbours slide over, cascading
    std::size_t slid = 0, sorts = 0;
    while (ps.cell_range(Cell_t{4, 4}).second)
    {
        auto const [s2, n2]         = ps.cell_range(Cell_t{4, 4});
        components(ps).iCell[0][s2] = 5;
        where[ps.weight(s2)]        = ps.iCell(s2);
        sorts += ps.resettle(box, [](auto const&) {});
        expect_consistent(ps, "move " + std::to_string(++slid));
    }
    EXPECT_EQ(ps.cell_range(Cell_t{5, 4}).second, 2 * ppc);
    EXPECT_LT(sorts, slid); // most taken by slides
}

// the slack as a constant per cell (border weight 1) or weighted to the border cells (3)
struct Gaps2DWeights : public Gaps2D, public ::testing::WithParamInterface<double>
{
};

// random moves inside the box: about as many enter a cell as leave it
TEST_P(Gaps2DWeights, randomMovesSortRarely)
{
    auto ps           = make(.1, GetParam());
    std::size_t sorts = 0;
    for (std::size_t round = 0; round < 50; ++round)
    {
        move_some(ps, 60);
        std::vector<Particle<2>> out;
        sorts += ps.resettle(box, [&](auto const& p) { out.push_back(p); });
        for (auto const& p : out)
            where.erase(p.weight());
        expect_consistent(ps, "round " + std::to_string(round));
    }
    EXPECT_LT(sorts, 25u);
    std::cout << "2D random moves, border weight " << GetParam() << ": " << sorts
              << " sorts in 50 resettles" << std::endl;
}

// arrivals into the border cells, as from neighbour patches and level ghosts, as many leaving
//  them out of the box: the border cells' gaps fill
TEST_P(Gaps2DWeights, borderArrivalsSortLessWithBorderWeight)
{
    auto ps           = make(.1, GetParam());
    std::size_t sorts = 0;
    std::uniform_int_distribution<int> along(box.lower[1], box.upper[1]);
    double id = 1e6;
    for (std::size_t round = 0; round < 50; ++round)
    {
        for (std::size_t k = 0; k < 20; ++k)
        {
            Particle<2> p;
            p.iCell()  = {k % 2 ? box.lower[0] : box.upper[0], along(rng)}; // the x borders
            p.delta()  = {.5, .5};
            p.v()      = {0, 0, 0};
            p.weight() = ++id;
            where[id]  = p.iCell();
            if (!ps.insert(p))
            {
                ps.push_back(p);
                ps.sort();
                ++sorts;
            }
        }
        // as many leave: border cells' particles out of the box
        std::size_t left = 0;
        for (std::size_t i = 0; i < ps.size() and left < 20; ++i)
            if (ps.weight(i) != 0 and ps.iCell(i)[0] == (left % 2 ? box.lower[0] : box.upper[0]))
            {
                components(ps).iCell[0][i] += left % 2 ? -1 : 1;
                ++left;
            }
        std::vector<Particle<2>> out;
        sorts += ps.resettle(box, [&](auto const& p) { out.push_back(p); });
        for (auto const& p : out)
            where.erase(p.weight());
        expect_consistent(ps, "round " + std::to_string(round));
    }
    std::cout << "2D border arrivals, border weight " << GetParam() << ": " << sorts
              << " sorts in 50 rounds of 20 arrivals" << std::endl;
}

INSTANTIATE_TEST_SUITE_P(Gaps, Gaps2DWeights, ::testing::Values(1., 3.));

TEST_F(Gaps2D, resettleSortsOnlyWhenAChunkHasNoRoom)
{
    auto ps           = make(damn::core::get_env_as("PLAMN_GAPS_SLACK", .1),
                             damn::core::get_env_as("PLAMN_GAPS_BORDER", 3.));
    std::size_t sorts = 0, left = 0;
    for (std::size_t round = 0; round < 50; ++round)
    {
        move_some(ps, 60); // 3% of the particles change cell
        std::vector<Particle<2>> out;
        sorts += ps.resettle(box, [&](auto const& p) { out.push_back(p); });
        for (auto const& p : out)
        {
            EXPECT_FALSE(isIn(damn::core::Point<int, 2>{p.iCell()}, box));
            where.erase(p.weight());
        }
        left += out.size();
        expect_consistent(ps, "round " + std::to_string(round));
    }
    EXPECT_LT(sorts, 50u);
    EXPECT_GT(left, 0u); // border cells' particles left the box
    std::cout << "2D: " << sorts << " sorts in 50 resettles of 60 moves, " << left << " left"
              << std::endl;
}


using Gaps3D = GapsTest<3>;

TEST_F(Gaps3D, sortLeavesTheSlackBetweenChunksBorderCellsMore)
{
    // 20000 particles, 2000 slots: 488 border cells weight 3, 512 inside 1 (1976)
    auto const ps = make(.1);
    expect_consistent(ps, "sorted");
    EXPECT_EQ(ps.nbr_particles(), 20000u);
    auto const gap_after = [&](Cell_t const& cell) {
        auto const [start, size] = ps.cell_range(cell);
        auto const [next, _]     = ps.cell_range(Cell_t{cell[0], cell[1], cell[2] + 1});
        return next - (start + size);
    };
    EXPECT_GE(gap_after({0, 0, 0}), 3u); // 2000 * 3 / 1976: 3 or 4
    EXPECT_LE(gap_after({0, 0, 0}), 4u);
    EXPECT_GE(gap_after({4, 4, 4}), 1u); // 2000 / 1976: 1 or 2
    EXPECT_LE(gap_after({4, 4, 4}), 2u);
    EXPECT_EQ(ps.size(), 20000u + 2000); // the slack exactly
}

// constant (1) or border weighted (3) slack; bulk motion along x (cells a plane apart in memory)
//  or z (the contiguous direction)
struct Gaps3DMotion : public Gaps3D,
                      public ::testing::WithParamInterface<std::tuple<double, std::size_t>>
{
};

TEST_P(Gaps3DMotion, randomMoves)
{
    auto const [weight, _] = GetParam();
    auto ps                = make(.1, weight);
    std::size_t sorts      = 0;
    for (std::size_t round = 0; round < 20; ++round)
    {
        move_some(ps, 600); // 3%
        std::vector<Particle<3>> out;
        sorts += ps.resettle(box, [&](auto const& p) { out.push_back(p); });
        for (auto const& p : out)
            where.erase(p.weight());
        expect_consistent(ps, "round " + std::to_string(round));
    }
    std::cout << "3D random moves, border weight " << weight << ": " << sorts
              << " sorts in 20 resettles of 600 moves" << std::endl;
}

// bulk motion: each round 5% of the particles move one cell along dir, those past the box's
//  upper side leave, as many arrive at its lower side (from the upstream neighbour)
TEST_P(Gaps3DMotion, bulkMotion)
{
    auto const [weight, dir] = GetParam();
    auto ps                  = make(.1, weight);
    std::size_t sorts = 0, moved = 0;
    std::uniform_int_distribution<int> across(box.lower[0], box.upper[0]);
    double id = 1e7;
    for (std::size_t round = 0; round < 20; ++round)
    {
        std::uniform_int_distribution<std::size_t> any(0, ps.size() - 1);
        std::set<std::size_t> chosen;
        while (chosen.size() < 1000)
            if (auto const i = any(rng); ps.weight(i) != 0)
                chosen.insert(i);
        for (auto const i : chosen)
        {
            components(ps).iCell[dir][i] += 1;
            where[ps.weight(i)] = ps.iCell(i);
        }
        std::vector<Particle<3>> out;
        sorts += ps.resettle(box, [&](auto const& p) { out.push_back(p); });
        for (auto const& p : out)
            where.erase(p.weight());
        moved += chosen.size();
        for (std::size_t k = 0; k < out.size(); ++k) // as many in, upstream
        {
            Particle<3> p;
            for (std::size_t d = 0; d < 3; ++d)
                p.iCell()[d] = d == dir ? box.lower[d] : across(rng);
            p.delta()  = {.5, .5, .5};
            p.v()      = {0, 0, 0};
            p.weight() = ++id;
            where[id]  = p.iCell();
            if (!ps.insert(p))
            {
                ps.push_back(p);
                ps.sort();
                ++sorts;
            }
        }
        expect_consistent(ps, "round " + std::to_string(round));
    }
    std::cout << "3D bulk motion along " << "xyz"[dir] << ", border weight " << weight << ": "
              << sorts << " sorts in 20 rounds of 1000 moves" << std::endl;
}

INSTANTIATE_TEST_SUITE_P(Gaps, Gaps3DMotion,
                         ::testing::Combine(::testing::Values(1., 3.),
                                            ::testing::Values(std::size_t{0}, std::size_t{2})));

TEST_F(Gaps1D, resettleFromTheRecordedCrossings)
{
    recorded_rounds(*this, 50, 15);
}
TEST_F(Gaps2D, resettleFromTheRecordedCrossings)
{
    recorded_rounds(*this, 50, 60);
}
TEST_F(Gaps3D, resettleFromTheRecordedCrossings)
{
    recorded_rounds(*this, 20, 600);
}

} // namespace plamn::core


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
