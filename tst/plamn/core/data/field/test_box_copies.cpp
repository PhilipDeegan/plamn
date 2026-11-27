// box copies (tiles <-> untiled temporaries) against the per tile host functions they replace;
//  tiles of uneven sizes, each tile's values its own (overlapping ghosts differ)

#include "plamn/core/def.hpp"

#include "damn/core/def/damn_config.hpp"
#include "plamn/core/data/field/box_copies.hpp"
#include "plamn/core/data/field/field_box.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"
#include "plamn/plamn_core.hpp"
#include "plamn/simulator/simulator_def.hpp"

#include "tst/plamn/core/data/grid/test_layout_fixtures.hpp"

#include "gtest/gtest.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <algorithm>

namespace plamn::core
{

bool static const premain = [] {
    setenv("DAMN_TILING_MAX_TILE_SIZE", "5", 1); // 12 cells: tiles of 5, 5, 2
    return true;
}();

template<std::size_t _dim, auto _layout_mode, auto _alloc_mode>
struct TestParam
{
    auto constexpr static dim         = _dim;
    auto constexpr static layout_mode = _layout_mode;
    auto constexpr static alloc_mode  = _alloc_mode;
};

template<typename Param>
struct BoxCopiesTest : public ::testing::Test
{
    auto constexpr static opts = SimOpts{.dimension    = Param::dim,
                                         .interp_order = 1,
                                         .layout_mode  = Param::layout_mode,
                                         .alloc_mode   = Param::alloc_mode};
    using Hybrid_t             = PLAMN_Types<opts>::Hybrid;
    using Layout_t             = Hybrid_t::Layout_t;
    using Tiles_t              = Hybrid_t::Grid_t;
    using Grid_t               = Tiles_t::grid_type;

    static constexpr auto pq = HybridQuantity::Scalar::rho; // primal

    static void set_tiles(Tiles_t& tiles, double const offset)
    {
        for (std::size_t t = 0; t < tiles().size(); ++t)
        {
            auto& tile = tiles()[t]();
            for (std::size_t i = 0; i < tile.size(); ++i)
                tile.data()[i] = offset + 1000. * (t + 1) + std::sin(.1 * i);
        }
    }

    // expected: rel 0 on the host; GPU sums are atomic (order varies): a few ulp
    static void expect_near(auto const* a, auto const* b, std::size_t const n, double const rel)
    {
        double max_diff = 0;
        for (std::size_t i = 0; i < n; ++i)
            max_diff = std::max(max_diff, std::abs(a[i] - b[i]) / std::max(1., std::abs(a[i])));
        EXPECT_LE(max_diff, rel) << "max diff " << max_diff;
    }

    double const sum_rel = Param::alloc_mode == damn::AllocatorMode::CPU ? 0 : 1e-14;

    TestLayout<Layout_t> const layout{12u};
    Tiles_t tiles{"rho", layout, pq};
};

// clang-format off
using Permutations_t = testing::Types< // ! notice commas !
     TestParam<1, LayoutMode::AoSCMTS, damn::AllocatorMode::CPU>
    ,TestParam<2, LayoutMode::AoSCMTS, damn::AllocatorMode::CPU>
    ,TestParam<3, LayoutMode::AoSCMTS, damn::AllocatorMode::CPU>
DAMN_WITH_GPU(
    ,TestParam<2, LayoutMode::AoSPCTS, damn::AllocatorMode::GPU_UNIFIED>
    ,TestParam<3, LayoutMode::AoSPCTS, damn::AllocatorMode::GPU_UNIFIED>
)
>;
// clang-format on

TYPED_TEST_SUITE(BoxCopiesTest, Permutations_t, );

TYPED_TEST(BoxCopiesTest, owned_into)
{
    using Grid_t = TestFixture::Grid_t;
    this->set_tiles(this->tiles, 0);
    Grid_t ref{"ref", this->layout, this->pq}, cmp{"cmp", this->layout, this->pq};
    reduce_single(*ref, this->tiles);
    (*cmp).fill(-1.); // every value overwritten
    auto& copies = box_copies<TestFixture::Tiles_t::alloc_mode, TestFixture::Tiles_t::dimension>();
    owned_into(copies, *cmp, this->tiles);
    copies.run();
    this->expect_near((*ref).data(), (*cmp).data(), (*ref).size(), 0);
}

TYPED_TEST(BoxCopiesTest, summed_into)
{
    using Grid_t = TestFixture::Grid_t;
    this->set_tiles(this->tiles, 0);
    Grid_t ref{"ref", this->layout, this->pq}, cmp{"cmp", this->layout, this->pq};
    reduce_into(*ref, this->tiles);
    (*cmp).fill(-1.); // zeroed by the run
    auto& copies = box_copies<TestFixture::Tiles_t::alloc_mode, TestFixture::Tiles_t::dimension>();
    summed_into(copies, *cmp, this->tiles);
    copies.run();
    this->expect_near((*ref).data(), (*cmp).data(), (*ref).size(), this->sum_rel);
}

TYPED_TEST(BoxCopiesTest, into_tile_ghosts)
{
    using Tiles_t = TestFixture::Tiles_t;
    using Grid_t  = TestFixture::Grid_t;
    Grid_t src{"src", this->layout, this->pq};
    for (std::size_t i = 0; i < (*src).size(); ++i)
        (*src).data()[i] = std::cos(.1 * i);
    Tiles_t ref{"ref", this->layout, this->pq}, cmp{"cmp", this->layout, this->pq};
    this->set_tiles(ref, 0);
    this->set_tiles(cmp, 0); // values outside the ghost boxes kept: the same in both
    copy_fields(ref, *src);
    auto& copies = box_copies<Tiles_t::alloc_mode, Tiles_t::dimension>();
    into_tile_ghosts(copies, cmp, *src);
    copies.run();
    for (std::size_t t = 0; t < ref().size(); ++t)
        this->expect_near(ref()[t]().data(), cmp()[t]().data(), ref()[t]().size(), 0);
}

} // namespace plamn::core

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
