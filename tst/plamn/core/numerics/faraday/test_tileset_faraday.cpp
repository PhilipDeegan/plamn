#include "plamn/core/def.hpp"
//  tests/core/numerics/faraday/test_tileset_faraday.cpp
//
//  requires
//  - cmake:    -DwithPhlop
//  - cxxflags: -DDAMN_LOG_LEVEL=1
//  - env:      PLAMN_SCOPE_TIMING=1

// USE HIP_VISIBLE_DEVICES OR CUDA_VISIBLE_DEVICES env vars
// #define DAMN_UNDEF_ASSERT

#include "damn/core/def/damn_config.hpp"

#include "mkn/kul/std/array.hpp"
#include "mkn/kul/math.hpp"
#include "mkn/kul/float.hpp"
#include "mkn/kul/int.hpp"
#include "mkn/kul/for.hpp"
#include "mkn/kul/meta.hpp"
#include "mkn/kul/tuple.hpp"
#include "mkn/kul/zip.hpp"
#include "mkn/kul/string.hpp"
#include "mkn/kul/env.hpp"
#include "plamn/core/numerics/faraday/faraday_transformer.hpp"

#include "plamn/simulator/simulator_def.hpp"

#include "tst/plamn/core/data/grid/test_layout_fixtures.hpp"
#include "tst/plamn/core/data/vecfield/test_vecfield_fixtures.hpp"
#include "tst/plamn/core/data/electromag/test_electromag_fixtures.hpp"

#include "gtest/gtest.h"

#include <cstddef>

#include "BS_thread_pool.hpp"

namespace plamn::core
{

auto static const cells = mkn::kul::env::get_as("PLAMN_CELLS", std::uint32_t{4});
auto static const dt    = mkn::kul::env::get_as("PLAMN_TIMESTEP", double{.001});

static ::BS::thread_pool pool{4};

void ref_do(auto& patch)
{
    FaradayTransformer{}(patch.layout, patch.em.B, patch.em.E, patch.emNew.B, dt);
}

template<typename Patch>
void cmp_do(Patch& patch)
{
    FaradayTransformer const faraday{};

    if constexpr (Patch::alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
    {
        faraday(patch.layout, patch.em.B, patch.em.E, patch.emNew.B, dt); // one launch
        return;
    }

    // a pool task per tile, as the level transformer
    for (std::size_t ti = 0; ti < tile_count(patch.emNew.B); ++ti)
        pool.detach_task(
            [&, ti]() { faraday.on_tile(ti, patch.em.B, patch.em.E, patch.emNew.B, dt); });
    pool.wait();
}



template<auto alloc_mode>
void compare(auto const& /*layout*/, auto& ref, auto& cmp)
{
    using enum LayoutMode;
    using enum damn::AllocatorMode;
    auto constexpr static n_components = 3;


    for (std::size_t c = 0; c < n_components; ++c)
    {
        auto const eq = compare_fields(ref.emNew.B[c], cmp.emNew.B[c]);
        DAMN_LOG_LINE_SS(eq.why());
        EXPECT_TRUE(eq) << "Failure for B New: " << eq.why();

        // auto const& ref_Bxyz = ref.emNew.B[c];
        // auto const& cmp_Bxyz = cmp.emNew.B[c];

        // for (auto const& tile : cmp_Bxyz)
        // {
        //     auto const& tile_field = tile();
        //     for (auto const& tix : *tile)
        //     {
        //         auto const tlix = tix - tile.lower;
        //         auto const plix = tix - patch_box.lower;
        //         EXPECT_TRUE(float_equals(tile_field(tlix), ref_Bxyz(plix), diff));
        //     }
        // }
    }
}


template<std::size_t _dim, auto _layout_mode, auto _alloc_mode>
struct TestParam
{
    static_assert(mkn::kul::all_are<LayoutMode>(_layout_mode));
    static_assert(mkn::kul::all_are<damn::AllocatorMode>(_alloc_mode));

    auto constexpr static dim         = _dim;
    auto constexpr static layout_mode = _layout_mode;
    auto constexpr static alloc_mode  = _alloc_mode;
};


template<typename Param>
struct FaradayTileTest : public ::testing::Test
{
    struct TileSetPatch
    {
        auto constexpr static dim         = Param::dim;
        auto constexpr static interp      = 1;
        auto constexpr static layout_mode = Param::layout_mode;
        auto constexpr static alloc_mode  = Param::alloc_mode;
        auto constexpr static opts        = SimOpts{.dimension    = dim,
                                                    .interp_order = interp,
                                                    .layout_mode  = layout_mode,
                                                    .alloc_mode   = alloc_mode};

        using DamnTypes                 = PLAMN_Types<opts>;
        using Layout_t                   = DamnTypes::Hybrid::Layout_t;
        using Field_t                    = DamnTypes::Hybrid::Grid_t;
        using Hybrid_t                   = DamnTypes::Hybrid;
        auto constexpr static field_opts = TensorFieldOptions<Hybrid_t>{};
        using UsableElectromag_t         = UsableElectromag<field_opts>;
        using UsableVecField_t           = UsableVecField<field_opts>;


        TileSetPatch(Layout_t const& layout_)
            : layout{layout_}
        {
        }

        Layout_t layout;
        UsableElectromag_t em{layout}, emNew{layout};
    };

    struct ContiguousPatch
    {
        // reference (untiled) is always run on CPU, compared against the tiled/possibly-GPU patch
        auto constexpr static dim    = Param::dim;
        auto constexpr static interp = 1;
        auto constexpr static opts   = SimOpts{.dimension = dim, .interp_order = interp};

        using DamnTypes                 = PLAMN_Types<opts>;
        using Layout_t                   = DamnTypes::Hybrid::Layout_t;
        using Hybrid_t                   = DamnTypes::Hybrid;
        auto constexpr static field_opts = TensorFieldOptions<Hybrid_t>{};
        using UsableElectromag_t         = UsableElectromag<field_opts>;
        using Electromag_t               = UsableElectromag_t::Super;
        using Grid_t                     = DamnTypes::Hybrid::Grid_t;
        using UsableVecField_t           = UsableVecField<field_opts>;

        ContiguousPatch(Layout_t const& layout_)
            : layout{layout_}
        {
        }

        Layout_t layout;
        UsableElectromag_t em{layout}, emNew{layout};
    };

    using RefPatch = ContiguousPatch;
    using CmpPatch = TileSetPatch;

    FaradayTileTest() {}

    void init_check() const
    {
        bool static constexpr init_value_check = 0;
        if constexpr (init_value_check)
        {
            // auto ref = make_ions<RefParticles_t>(layout);
            // auto cmp = from_ions<CmpParticles_t>(layout, *ref);
            // compare(*this->layout, *ref, *cmp);
        }
    }

    void do_compare() const { compare<CmpPatch::alloc_mode>(*cmp_layout, ref_patch, cmp_patch); }

    TestLayout<typename RefPatch::Layout_t> const ref_layout{cells};
    TestLayout<typename CmpPatch::Layout_t> const cmp_layout{cells};
    RefPatch ref_patch{ref_layout};
    CmpPatch cmp_patch{cmp_layout};
};

// clang-format off
using Permutations_t = testing::Types<
     TestParam<3, LayoutMode::AoSCMTS, damn::AllocatorMode::CPU>
    ,TestParam<3, LayoutMode::AoSPCTS, damn::AllocatorMode::CPU>
DAMN_WITH_GPU(
    ,TestParam<3, LayoutMode::AoSPCTS, damn::AllocatorMode::GPU_UNIFIED>
)
>;
// clang-format on

TYPED_TEST_SUITE(FaradayTileTest, Permutations_t, );

auto run(auto& self)
{
    ref_do(self.ref_patch);
    cmp_do(self.cmp_patch);
    self.do_compare();
}

TYPED_TEST(FaradayTileTest, dispatch)
{
    run(*this);
}

} // namespace plamn::core

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
