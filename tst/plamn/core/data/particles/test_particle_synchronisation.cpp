
#include "damn/core/logger.hpp"
#include "mkn/kul/meta.hpp"
#include "mkn/kul/env.hpp"

#include "plamn/core/data/particles/particles.hpp"
#include "plamn/core/data/particles/appending/particles_appending.hpp"
#include "plamn/core/data/particles/comparing/particles_comparing.hpp"
#include "plamn/core/data/particles/converting/particles_converting.hpp"
#include "plamn/core/data/particles/exporting/particles_exporting.hpp"
#include "plamn/core/data/particles/sorting/particles_sorting.hpp"
#include "plamn/simulator/simulator_def.hpp"

#include "tst/plamn/core/data/grid/test_layout_fixtures.hpp"
#include "tst/plamn/core/data/particles/test_particles.hpp"

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <limits>
#include <string>
#include <vector>


namespace plamn::core
{
using damn::core::for_each_point;
using damn::core::grow;
using damn::core::isIn;
using damn::core::Point;
using mkn::kul::all_are;
using mkn::kul::any_in;

auto static const cells = mkn::kul::env::get_as("PLAMN_CELLS", std::uint32_t{14});
auto static const ppc   = mkn::kul::env::get_as("PLAMN_PPC", std::size_t{1000});


template<std::size_t _dim, auto lm, auto am>
struct TestParam
{
    static_assert(all_are<LayoutMode>(lm));
    static_assert(all_are<damn::AllocatorMode>(am));

    auto constexpr static dim         = _dim;
    auto constexpr static layout_mode = lm;
    auto constexpr static alloc_mode  = am;
};


template<typename Param>
struct ParticleArrayConstructionTest : public ::testing::Test
{
    auto constexpr static dim         = Param::dim;
    auto constexpr static layout_mode = Param::layout_mode;
    auto constexpr static alloc_mode  = Param::alloc_mode;
    auto constexpr static sim_opts
        = SimOpts{.dimension = dim, .layout_mode = layout_mode, .alloc_mode = alloc_mode};

    using Layout_t = PLAMN_Types<sim_opts>::Hybrid::Layout_t;
    using ParticleArray_t
        = Particles<ParticlesOptions{dim, layout_mode, StorageMode::VECTOR, alloc_mode}>;

    Layout_t const layout = *TestLayout<Layout_t>{cells};

    MKN_KUL_FN_SIG_RET ParticleArray_t setup_particles() const // test movable
    {
        auto ps = make_particles<ParticleArray_t>(layout);
        add_particles(ps, layout.cells, ppc);
        delta_disperse(ps);
        vary_velocity(ps, -6, 6);
        return ps;
    }
};



// clang-format off
using Permutations_t = testing::Types< // ! notice commas !

    TestParam<1, LayoutMode::AoS, damn::AllocatorMode::CPU>
   ,TestParam<1, LayoutMode::AoSMapped, damn::AllocatorMode::CPU>
   ,TestParam<2, LayoutMode::AoS, damn::AllocatorMode::CPU>
   ,TestParam<2, LayoutMode::AoSMapped, damn::AllocatorMode::CPU>
   ,TestParam<2, LayoutMode::AoSCMTS, damn::AllocatorMode::CPU>
   ,TestParam<3, LayoutMode::AoSCMTS, damn::AllocatorMode::CPU>

>;
// clang-format on



TYPED_TEST_SUITE(ParticleArrayConstructionTest, Permutations_t, );


// 2^dim "corner" neighbours only (every coordinate moves by +/-1), ordered so that
// treating -1 as bit 0 and +1 as bit 1 counts up from (-1,-1,-1) to (1,1,1).
template<std::size_t dim>
auto corner_offsets()
{
    std::array<std::array<int, dim>, (std::size_t{1} << dim)> offsets{};
    for (std::size_t idx = 0; idx < offsets.size(); ++idx)
        for (std::size_t d = 0; d < dim; ++d)
            offsets[idx][d] = ((idx >> (dim - 1 - d)) & 1) ? 1 : -1;
    return offsets;
}

template<typename ICell>
auto add_icell(ICell const& a, ICell const& b)
{
    ICell out;
    for (std::size_t d = 0; d < a.size(); ++d)
        out[d] = a[d] + b[d];
    return out;
}


// function templates can't be partially specialised (layout_mode fixed, Particles deduced): the
// per-layout dispatch is a class template, one specialisation per layout_mode
template<auto layout_mode>
struct MoveParticles;

template<>
struct MoveParticles<LayoutMode::AoS>
{
    // no registration needed, flat array has no cell/tile structure to maintain
    template<typename Particles, typename Offsets>
    static void apply(Particles& particles, Offsets const& offsets)
    {
        std::size_t counter = 0;
        for (auto& p : particles)
        {
            p.iCell() = add_icell(p.iCell(), offsets[counter % offsets.size()]);
            ++counter;
        }
    }
};

template<>
struct MoveParticles<LayoutMode::AoSMapped>
{
    template<typename Particles, typename Offsets>
    static void apply(Particles& particles, Offsets const& offsets)
    {
        std::size_t counter = 0;
        for (std::size_t idx = 0; idx < particles.size(); ++idx)
        {
            auto const newcell
                = add_icell(particles[idx].iCell(), offsets[counter % offsets.size()]);
            ++counter;
            particles.change_icell(newcell, idx);
        }
    }
};

template<>
struct MoveParticles<LayoutMode::AoSCMTS>
{
    template<typename Particles, typename Offsets>
    static void apply(Particles& particles, Offsets const& offsets)
    {
        auto constexpr dim  = Particles::dimension;
        std::size_t counter = 0;
        for (auto& tile : particles())
        {
            auto& tile_particles = tile();
            auto const tile_cell = particles.local_cell(tile.lower);
            auto const n         = tile_particles.size();
            for (std::size_t i = 0; i < n; ++i)
            {
                auto& p            = tile_particles[i];
                auto const oldcell = p.iCell();
                auto const newcell = add_icell(oldcell, offsets[counter % offsets.size()]);
                ++counter;
                p.iCell() = newcell;
                TiledParticleTracker<dim> const pt{oldcell, tile_cell};
                particles.template move_check<ParticleType::Domain>(pt, i, p);
            }
        }
        particles.template on_moved<ParticleType::Domain>();
    }
};

template<>
struct MoveParticles<LayoutMode::AoSTS>
{
    template<typename Particles, typename Offsets>
    static void apply(Particles& particles, Offsets const& offsets)
    {
        auto constexpr dim  = Particles::dimension;
        std::size_t counter = 0;
        for (auto& tile : particles())
        {
            auto& tile_particles = tile();
            auto const n         = tile_particles.size();
            for (std::size_t i = 0; i < n; ++i)
            {
                auto& p              = tile_particles[i];
                auto const oldcell   = p.iCell();
                auto const tile_cell = particles.local_tile_cell(oldcell);
                auto const newcell   = add_icell(oldcell, offsets[counter % offsets.size()]);
                ++counter;
                p.iCell() = newcell;
                // as MultiBoris: only particles staying in the patch box register; patch
                // leavers stay put in their tile with a ghost iCell
                if (isIn(newcell, particles.box()))
                {
                    TiledParticleTracker<dim> const pt{oldcell, tile_cell};
                    particles.template move_check<ParticleType::Domain>(pt, i, p);
                }
            }
        }
        particles.template on_moved<ParticleType::Domain>();
    }
};

template<>
struct MoveParticles<LayoutMode::AoSPCTS>
{
    // registration goes through the span (move_check lives on PCTileSetSpan); the view's
    // gap arrays alias the vector's, so on_moved() on the vector sees every registration
    template<typename Particles, typename Offsets>
    static void apply(Particles& particles, Offsets const& offsets)
    {
        auto constexpr dim  = Particles::dimension;
        std::size_t counter = 0;
        auto view           = particles.view();
        for (auto& tile : view())
        {
            auto& tile_particles = tile();
            for_each_point(tile_particles.local_box(), [&](auto const& bix) {
                auto& cell_particles = tile_particles(bix);
                auto const n         = cell_particles.size();
                for (std::size_t i = 0; i < n; ++i)
                {
                    auto& p              = cell_particles[i];
                    auto const oldcell   = p.iCell();
                    auto const tile_cell = view.local_tile_cell(oldcell);
                    auto const newcell   = add_icell(oldcell, offsets[counter % offsets.size()]);
                    ++counter;
                    p.iCell() = newcell;
                    TiledParticleTracker<dim> const pt{oldcell, tile_cell};
                    view.template move_check<ParticleType::Domain>(pt, i, p);
                }
            });
        }
        particles.template on_moved<ParticleType::Domain>();
    }
};


template<typename Particles>
void move_particles(Particles& particles)
{
    MoveParticles<Particles::layout_mode>::apply(particles, corner_offsets<Particles::dimension>());
}


// level ghost arrays: particles live in the ghost layer (clamp-owner tile for tiled layouts);
// movers re-bucket anywhere inside the ghost box (including into the domain), ghost-box
// leavers are deleted at sync
template<auto layout_mode>
struct MoveLevelGhostParticles;

template<>
struct MoveLevelGhostParticles<LayoutMode::AoS>
{
    // no registration: move directly, then apply the deletion contract by hand
    template<typename Particles, typename Offsets, typename Box_t>
    static void apply(Particles& particles, Offsets const& offsets, Box_t const& ghost_box)
    {
        std::size_t counter = 0;
        for (auto& p : particles)
        {
            p.iCell() = add_icell(p.iCell(), offsets[counter % offsets.size()]);
            ++counter;
        }
        std::erase_if(particles.vector(),
                      [&](auto const& p) { return not isIn(p.iCell(), ghost_box); });
    }
};

template<>
struct MoveLevelGhostParticles<LayoutMode::AoSCMTS>
{
    // registers on the vector, as for the domain mover above
    template<typename Particles, typename Offsets, typename Box_t>
    static void apply(Particles& particles, Offsets const& offsets, Box_t const&)
    {
        auto constexpr dim  = Particles::dimension;
        std::size_t counter = 0;
        for (auto& tile : particles())
        {
            auto& tile_particles = tile();
            auto const tile_cell = particles.local_cell(tile.lower);
            auto const n         = tile_particles.size();
            for (std::size_t i = 0; i < n; ++i)
            {
                auto& p            = tile_particles[i];
                auto const oldcell = p.iCell();
                p.iCell()          = add_icell(oldcell, offsets[counter % offsets.size()]);
                ++counter;
                auto const pt
                    = make_particle_tracker<LayoutMode::AoSCMTS, ParticleType::LevelGhost, dim>(
                        oldcell, tile_cell);
                particles.template move_check<ParticleType::LevelGhost>(pt, i, p);
            }
        }
        particles.template on_moved<ParticleType::LevelGhost>();
    }
};

template<>
struct MoveLevelGhostParticles<LayoutMode::AoSPC>
{
    template<typename Particles, typename Offsets, typename Box_t>
    static void apply(Particles& particles, Offsets const& offsets, Box_t const&)
    {
        auto constexpr dim  = Particles::dimension;
        std::size_t counter = 0;
        for_each_point(particles.local_box(), [&](auto const& bix) {
            auto& cell_particles = particles(bix);
            auto const n         = cell_particles.size();
            for (std::size_t i = 0; i < n; ++i)
            {
                auto& p            = cell_particles[i];
                auto const oldcell = p.iCell();
                p.iCell()          = add_icell(oldcell, offsets[counter % offsets.size()]);
                ++counter;
                ParticleTracker<dim> const pt{oldcell};
                particles.template move_check<ParticleType::LevelGhost>(pt, i, p);
            }
        });
        particles.template on_moved<ParticleType::LevelGhost>();
    }
};

template<>
struct MoveLevelGhostParticles<LayoutMode::AoSPCTS>
{
    // registration goes through the span, as for the domain mover above
    template<typename Particles, typename Offsets, typename Box_t>
    static void apply(Particles& particles, Offsets const& offsets, Box_t const&)
    {
        auto constexpr dim  = Particles::dimension;
        std::size_t counter = 0;
        auto view           = particles.view();
        for (auto& tile : view())
        {
            auto& tile_particles = tile();
            for_each_point(tile_particles.local_box(), [&](auto const& bix) {
                auto& cell_particles = tile_particles(bix);
                auto const n         = cell_particles.size();
                for (std::size_t i = 0; i < n; ++i)
                {
                    auto& p              = cell_particles[i];
                    auto const oldcell   = p.iCell();
                    auto const tile_cell = view.local_tile_cell(oldcell);
                    p.iCell()            = add_icell(oldcell, offsets[counter % offsets.size()]);
                    ++counter;
                    auto const pt
                        = make_particle_tracker<LayoutMode::AoSPCTS, ParticleType::LevelGhost, dim>(
                            oldcell, tile_cell);
                    view.template move_check<ParticleType::LevelGhost>(pt, i, p);
                }
            });
        }
        particles.template on_moved<ParticleType::LevelGhost>();
    }
};


// AoS reference for move_in_domain: split the level ghosts that entered the domain box out
// into their own (domain) array
template<typename Particles, typename Box_t>
auto split_into_domain(Particles& ghosts, Box_t const& domain_box)
{
    auto domain = ghosts;
    std::erase_if(domain.vector(), [&](auto const& p) { return not isIn(p.iCell(), domain_box); });
    std::erase_if(ghosts.vector(), [&](auto const& p) { return isIn(p.iCell(), domain_box); });
    return domain;
}


// only particles outside the patch domain box may sit outside their tile's box —
// a particle inside the patch domain must never be stored in a tile's ghost cells
template<typename Particles>
void check_tile_ownership(Particles const& particles)
{
    using enum LayoutMode;

    if constexpr (any_in(Particles::layout_mode, AoSTS, AoSCMTS, AoSPCTS))
    {
        auto const check = [&](auto const& tile, auto const& p) {
            if (isIn(p.iCell(), particles.box()))
            {
                EXPECT_TRUE(isIn(p.iCell(), tile))
                    << "domain particle in tile ghost cells: " << Point{p.iCell()}
                    << " tile: " << tile;
            }
        };

        if constexpr (any_in(Particles::layout_mode, AoSTS, AoSCMTS))
        {
            for (auto const& tile : particles())
                for (auto const& p : tile())
                    check(tile, p);
        }
        else
        {
            for (auto const& tile : particles())
            {
                auto const& cps = tile();
                for_each_point(cps.local_box(), [&](auto const& bix) {
                    for (auto const& p : cps(bix))
                        check(tile, p);
                });
            }
        }
    }
}



TYPED_TEST(ParticleArrayConstructionTest, test_move_sync_works)
{
    using ParticleArray_t    = TestFixture::ParticleArray_t;
    using AoSParticleArray_t = ParticlesAoS<TestFixture::dim>;

    DAMN_LOG_LINE_SS(ParticleArray_t::type_id);

    auto particles = make_particles<ParticleArray_t>(this->layout);
    add_particles(particles, this->layout.cells, ppc);

    // independent AoS ground truth, same initial particles, moved directly (no
    // registration needed for AoS) so it never depends on the layout under test
    auto reference = convert_particles<AoSParticleArray_t>(particles, this->layout);

    move_particles(particles);
    move_particles(reference);

    check_tile_ownership(particles);
    if constexpr (ParticleArray_t::layout_mode == LayoutMode::AoSCMTS)
    {
        EXPECT_TRUE(particles.is_consistent()); // every particle mapped at its own cell
    }

    // particles have moved out of AMRBox into the first ghost layer; sorting must use the
    // grown box or distinct out-of-box cells alias to the same flat index and, with all
    // deltas identical, the two arrays end up in different orders
    auto const sort_box = grow(this->layout.cells, 1);
    auto converted      = convert_particles<AoSParticleArray_t>(particles, this->layout);
    sort_particles(converted, sort_box);
    sort_particles(reference, sort_box);

    auto const report = compare_particles(reference, converted);
    EXPECT_TRUE(report) << report.why();
}


// level ghost move_check is only supported for AoS, AoSPC, AoSPCTS and AoSCMTS on CPU
template<typename Param>
struct LevelGhostSyncTest : public ParticleArrayConstructionTest<Param>
{
};

// clang-format off
using LevelGhostPermutations_t = testing::Types< // ! notice commas !

    TestParam<1, LayoutMode::AoS, damn::AllocatorMode::CPU>
   ,TestParam<2, LayoutMode::AoSCMTS, damn::AllocatorMode::CPU>
   ,TestParam<3, LayoutMode::AoSCMTS, damn::AllocatorMode::CPU>

>;
// clang-format on

TYPED_TEST_SUITE(LevelGhostSyncTest, LevelGhostPermutations_t, );

// level ghost arrays fill the ghost layer; after a move they may re-bucket anywhere in the
// ghost box (including into the domain), ghost-box leavers must be deleted, and those that
// entered the domain must then be moved into the domain array by move_in_domain
// (as ParallelIonUpdater::post_move_sync does)
TYPED_TEST(LevelGhostSyncTest, test_level_ghost_move_sync_works)
{
    using ParticleArray_t    = TestFixture::ParticleArray_t;
    using AoSParticleArray_t = ParticlesAoS<TestFixture::dim>;
    using enum LayoutMode;

    auto constexpr static dim = TestFixture::dim;

    DAMN_LOG_LINE_SS(ParticleArray_t::type_id);

    static_assert(ParticleArray_t::alloc_mode == damn::AllocatorMode::CPU
                  and any_in(ParticleArray_t::layout_mode, AoS, AoSPC, AoSPCTS, AoSCMTS));
    {
        auto constexpr static ghosts = TestFixture::Layout_t::options.particle_ghost_width;
        auto const domain_box        = this->layout.cells;
        auto const ghost_box         = grow(domain_box, ghosts);

        auto particles = make_particles<ParticleArray_t>(this->layout);
        add_ghost_particles(particles, domain_box, ppc, ghosts);

        auto reference = convert_particles<AoSParticleArray_t>(particles, this->layout);

        auto const offsets = corner_offsets<dim>();
        MoveLevelGhostParticles<ParticleArray_t::layout_mode>::apply(particles, offsets, ghost_box);
        MoveLevelGhostParticles<AoS>::apply(reference, offsets, ghost_box);

        check_tile_ownership(particles);
        if constexpr (ParticleArray_t::layout_mode == AoSCMTS)
        {
            EXPECT_TRUE(particles.is_consistent());
        }

        auto const compare = [&](auto& ref, auto const& cmp, auto const& name) {
            auto converted = convert_particles<AoSParticleArray_t>(cmp, this->layout);
            sort_particles(converted, ghost_box);
            sort_particles(ref, ghost_box);
            EXPECT_EQ(ref.size(), converted.size()) << name;
            auto const report = compare_particles(ref, converted);
            EXPECT_TRUE(report) << name << ": " << report.why();
        };

        compare(reference, particles, "level ghosts after sync");

        // move_in_domain: level ghosts that entered the domain box go to the domain array
        // (AoSPC has no move_in_domain exporter)
        if constexpr (ParticleArray_t::layout_mode != AoSPC)
        {
            auto ref_domain = split_into_domain(reference, domain_box);
            if constexpr (ParticleArray_t::layout_mode == AoS)
            {
                auto domain = split_into_domain(particles, domain_box);
                compare(ref_domain, domain, "domain after move_in_domain");
            }
            else
            {
                auto domain = make_particles<ParticleArray_t>(this->layout);
                move_in_domain(domain, particles, domain_box);
                compare(ref_domain, domain, "domain after move_in_domain");
            }
            compare(reference, particles, "level ghosts after move_in_domain");
        }
    }
}



// SoAVXChunked's converters (it has no element iteration): AoS -> SoAVXChunked sorts, its
//  offsets give each cell's particles; back to AoS, cell by cell through them, the same particles
template<typename Dim>
struct SoAVXChunkedConvertingTest : public ::testing::Test
{
};
using ChunkedDims_t
    = testing::Types<std::integral_constant<std::size_t, 1>, std::integral_constant<std::size_t, 2>,
                     std::integral_constant<std::size_t, 3>>;
TYPED_TEST_SUITE(SoAVXChunkedConvertingTest, ChunkedDims_t, );

TYPED_TEST(SoAVXChunkedConvertingTest, roundTripKeepsParticlesAndOffsetsCountCells)
{
    auto constexpr dim  = TypeParam::value;
    auto constexpr opts = SimOpts{.dimension = dim, .layout_mode = LayoutMode::SoAVXChunked};
    using Layout_t      = PLAMN_Types<opts>::Hybrid::Layout_t;
    using AoS_t         = ParticlesAoS<dim>;
    using Chunked_t     = Particles<ParticlesOptions{dim, LayoutMode::SoAVXChunked}>;

    Layout_t const layout = *TestLayout<Layout_t>{std::uint32_t{6}};
    auto aos              = make_particles<AoS_t>(layout);
    add_particles(aos, layout.cells, 7);
    add_ghost_particles(aos, layout.cells, 3, 1); // and in the shell (box + 1) the offsets cover
    delta_disperse(aos);
    vary_velocity(aos, -6, 6);

    auto const chunked = convert_particles<Chunked_t>(aos, layout);
    ASSERT_TRUE(chunked.sorted());
    ASSERT_EQ(chunked.size(), aos.size());
    damn::core::for_each_point(chunked.shell(), [&](auto const& cell) {
        auto const expected = std::count_if(
            aos.begin(), aos.end(), [&](auto const& p) { return Point{p.iCell()} == Point{cell}; });
        EXPECT_EQ(chunked.cell_range(cell).second, static_cast<std::size_t>(expected))
            << Point{cell};
    });

    auto back           = convert_particles<AoS_t>(chunked, layout);
    auto const sort_box = grow(layout.cells, 1);
    auto reference      = aos;
    sort_particles(back, sort_box);
    sort_particles(reference, sort_box);
    auto const report
        = compare_particles(reference, back, std::numeric_limits<double>::denorm_min());
    EXPECT_TRUE(report) << report.why();
}


// SoAVX's exporting partitions column by column, stable: move_in_domain gives dst the domain's
//  particles and leaves src the others, delete_particles_not_in keeps the domain's, each in their
//  order (std::stable_partition's), every column moved with its particle (weight: the id)
template<typename Param>
struct SoAVXExportingTest : public ::testing::Test
{
};
template<std::size_t dim_, auto alloc_mode_>
struct ExportParam
{
    auto constexpr static dim        = dim_;
    auto constexpr static alloc_mode = alloc_mode_;
};
using ExportParams_t = testing::Types<ExportParam<1, damn::AllocatorMode::CPU>,
                                      ExportParam<2, damn::AllocatorMode::CPU>,
                                      ExportParam<3, damn::AllocatorMode::CPU>
#if DAMN_HAVE_GPU
                                      ,
                                      ExportParam<1, damn::AllocatorMode::GPU_UNIFIED>,
                                      ExportParam<3, damn::AllocatorMode::GPU_UNIFIED>
#endif
                                      >;
TYPED_TEST_SUITE(SoAVXExportingTest, ExportParams_t, );

TYPED_TEST(SoAVXExportingTest, partitionsAreStableAndMoveWholeParticles)
{
    auto constexpr dim  = TypeParam::dim;
    auto constexpr opts = SimOpts{.dimension = dim};
    using Layout_t      = PLAMN_Types<opts>::Hybrid::Layout_t;
    using AoS_t         = ParticlesAoS<dim>;
    using SoAVX_t       = Particles<ParticlesOptions{
        .dim = dim, .layout_mode = LayoutMode::SoAVX, .alloc_mode = TypeParam::alloc_mode}>;

    Layout_t const layout = *TestLayout<Layout_t>{std::uint32_t{5}};
    auto aos              = make_particles<AoS_t>(layout);
    add_particles(aos, layout.cells, 3);
    add_ghost_particles(aos, layout.cells, 2, 1);
    delta_disperse(aos);
    vary_velocity(aos, -6, 6);

    std::size_t const n = aos.size();
    ASSERT_GT(n, 0u);
    SoAVX_t src = make_particles<SoAVX_t>(layout);
    src.resize(n);
    std::vector<std::size_t> in, out; // ids, in order
    for (std::size_t i = 0; i < n; ++i)
    {
        auto const& p = aos[i];
        for (std::size_t d = 0; d < dim; ++d)
        {
            src.iCell_[d][i] = p.iCell()[d];
            src.delta_[d][i] = p.delta()[d];
        }
        for (std::size_t d = 0; d < 3; ++d)
            src.v_[d][i] = p.v()[d];
        src.weight_[i] = static_cast<double>(i + 1);
        src.v_[0][i]   = 10. * static_cast<double>(i);
        std::array<int, dim> cell;
        for (std::size_t d = 0; d < dim; ++d)
            cell[d] = src.iCell_[d][i];
        (isIn(cell, layout.cells) ? in : out).push_back(i);
    }
    ASSERT_FALSE(in.empty());
    ASSERT_FALSE(out.empty());
    auto const check = [&](auto const& ps, auto const& ids, std::string const& what) {
        ASSERT_EQ(ps.size(), ids.size()) << what;
        for (std::size_t k = 0; k < ids.size(); ++k)
        {
            EXPECT_EQ(ps.weight_[k], static_cast<double>(ids[k] + 1)) << what << " " << k;
            EXPECT_EQ(ps.v_[0][k], 10. * static_cast<double>(ids[k])) << what << " " << k;
        }
    };

    SoAVX_t kept = src;
    SoAVX_t dst  = make_particles<SoAVX_t>(layout);
    move_in_domain(dst, kept, layout.cells);
    check(dst, in, "move_in_domain dst");
    check(kept, out, "move_in_domain src");

    SoAVX_t domain = src;
    delete_particles_not_in(domain, layout.cells);
    check(domain, in, "delete_particles_not_in");
}

} // namespace plamn::core


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
