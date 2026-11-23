#include "plamn/core/def.hpp"
//  tests/core/numerics/ampere/test_tileset_ampere.cpp
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
#include "damn/core/exec/thread_pool.hpp"
#include "plamn/core/numerics/ampere/ampere_transformer.hpp"
#include "plamn/core/numerics/faraday/faraday_transformer.hpp"
#include "plamn/core/data/field/field_tiles_gpu.hpp"
#include "plamn/core/data/field/field_algorithm.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"

#include "plamn/simulator/simulator_def.hpp"

#include "tst/plamn/core/data/grid/test_layout_fixtures.hpp"
#include "tst/plamn/core/data/vecfield/test_vecfield_fixtures.hpp"
#include "tst/plamn/core/data/electromag/test_electromag_fixtures.hpp"
#include "tst/plamn/core/data/tensorfield/test_tensorfield_fixtures.hpp"

#include "gtest/gtest.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <cstddef>
#include <algorithm>

namespace plamn::core
{
auto static const cells = mkn::kul::env::get_as("PLAMN_CELLS", std::uint32_t{4});

bool static const premain = []() {
    damn::core::ThreadPool::threads_per_pool = 4;
    return true;
}();

auto& pool = *damn::core::ThreadPool::INSTANCE().thread_pools[0];


void ref_do(auto& patch)
{
    AmpereTransformer<>{}(patch.layout, patch.em.B, patch.J);
}

template<typename Patch>
void cmp_do(Patch& patch)
{
    AmpereTransformer<> const ampere{};

    if constexpr (Patch::alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
    {
        ampere(patch.layout, patch.em.B, patch.J); // one launch, block per tile
        return;
    }

    // a pool task per tile, as the level transformer
    for (std::size_t ti = 0; ti < tile_count(patch.J); ++ti)
        pool.detach_task([&, ti]() { ampere.on_tile(ti, patch.em.B, patch.J); });
    pool.wait();
}



template<auto alloc_mode>
void compare(auto const& layout, auto& ref, auto& cmp)
{
    using enum LayoutMode;
    using enum damn::AllocatorMode;
    auto constexpr static n_components = 3;

    for (std::size_t c = 0; c < n_components; ++c)
    {
        {
            auto const eq = compare_fields(ref.em.B[c], cmp.em.B[c]);
            EXPECT_TRUE(eq) << "Failure for B New: " << eq.why();
        }
        {
            auto const eq = compare_fields(ref.em.E[c], cmp.em.E[c]);
            EXPECT_TRUE(eq) << "Failure for E New: " << eq.why();
        }
        auto const eq = compare_fields(ref.J[c], cmp.J[c]);
        EXPECT_TRUE(eq) << "Failure for J: " << eq.why();
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
struct AmpereTileTest : public ::testing::Test
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
        UsableElectromag_t em{layout};
        UsableVecField_t J{"J", layout, HybridQuantity::Vector::J, 0};
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
        UsableElectromag_t em{layout};
        UsableVecField_t J{"J", layout, HybridQuantity::Vector::J, 0};
    };

    using RefPatch = ContiguousPatch;
    using CmpPatch = TileSetPatch;

    AmpereTileTest() {}

    void do_compare() const { compare<CmpPatch::alloc_mode>(*cmp_layout, ref_patch, cmp_patch); }

    TestLayout<typename RefPatch::Layout_t> const ref_layout{cells};
    TestLayout<typename CmpPatch::Layout_t> const cmp_layout{cells};
    RefPatch ref_patch{ref_layout};
    CmpPatch cmp_patch{cmp_layout};
};

// clang-format off
using Permutations_t = testing::Types< // ! notice commas !
     TestParam<3, LayoutMode::AoSCMTS, damn::AllocatorMode::CPU>
    ,TestParam<3, LayoutMode::AoSPCTS, damn::AllocatorMode::CPU>
DAMN_WITH_GPU(
    ,TestParam<3, LayoutMode::AoSPCTS, damn::AllocatorMode::GPU_UNIFIED>
)
>;
// clang-format on

TYPED_TEST_SUITE(AmpereTileTest, Permutations_t, );

auto run(auto& self)
{
    ref_do(self.ref_patch);
    cmp_do(self.cmp_patch);
    self.do_compare();
}

TYPED_TEST(AmpereTileTest, dispatch)
{
    run(*this);
}

// tile buffers of tiled vecfields, every element (ghosts included)
void copy_tiles(auto& dst, auto const& src)
{
    for (std::size_t c = 0; c < dst.size(); ++c)
        for (std::size_t ti = 0; ti < dst[c]().size(); ++ti)
        {
            auto& d       = dst[c]()[ti]();
            auto const& s = src[c]()[ti]();
            std::copy(s.data(), s.data() + s.size(), d.data());
        }
}

void expect_same_tiles(auto const& a, auto const& b, std::string const& what)
{
    std::size_t n_diff = 0, n = 0;
    for (std::size_t c = 0; c < a.size(); ++c)
        for (std::size_t ti = 0; ti < a[c]().size(); ++ti)
        {
            auto const& x = a[c]()[ti]();
            auto const& y = b[c]()[ti]();
            for (std::size_t i = 0; i < x.size(); ++i, ++n)
                n_diff += !(x.data()[i] == y.data()[i]
                            or (std::isnan(x.data()[i]) and std::isnan(y.data()[i])));
        }
    EXPECT_EQ(n_diff, 0u) << what << ": " << n_diff << " of " << n << " elements differ";
}

// device sync_inner_ghosts / average against their host versions, same input, exact
TYPED_TEST(AmpereTileTest, device_ops)
{
    using Patch = TestFixture::CmpPatch;
    if constexpr (Patch::alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
    {
        auto& host = this->cmp_patch;
        Patch dev{this->cmp_layout};
        // a value per (component, tile, element): ghosts differ from their neighbours' domains
        for (std::size_t c = 0; c < 3; ++c)
            for (std::size_t ti = 0; ti < host.J[c]().size(); ++ti)
            {
                auto& f = host.J[c]()[ti]();
                for (std::size_t i = 0; i < f.size(); ++i)
                    f.data()[i] = c * 1e6 + ti * 1e3 + i * 1e-3;
            }
        copy_tiles(dev.J, host.J);
        copy_tiles(dev.em.E, host.em.E);
        copy_tiles(dev.em.B, host.em.B);

        sync_inner_ghosts(host.J);
        sync_inner_ghosts_on_device(dev.J);
        expect_same_tiles(host.J, dev.J, "sync_inner_ghosts");

        { // the host sync did write ghosts, else the comparison above proves nothing
            Patch untouched{this->cmp_layout};
            copy_tiles(untouched.J, dev.J);
            for (std::size_t c = 0; c < 3; ++c)
                for (std::size_t ti = 0; ti < untouched.J[c]().size(); ++ti)
                {
                    auto& f = untouched.J[c]()[ti]();
                    for (std::size_t i = 0; i < f.size(); ++i)
                        f.data()[i] = c * 1e6 + ti * 1e3 + i * 1e-3;
                }
            std::size_t n_synced = 0;
            for (std::size_t c = 0; c < 3; ++c)
                for (std::size_t ti = 0; ti < host.J[c]().size(); ++ti)
                    for (std::size_t i = 0; i < host.J[c]()[ti]().size(); ++i)
                        n_synced += host.J[c]()[ti]().data()[i]
                                    != untouched.J[c]()[ti]().data()[i];
            if (host.J[0]().size() > 1)
            {
                EXPECT_GT(n_synced, 0u);
            }
        }

        for (std::size_t c = 0; c < 3; ++c) // host reference
            for (std::size_t ti = 0; ti < host.em.B[c]().size(); ++ti)
            {
                auto const& x0 = host.em.E[c]()[ti]();
                auto const& x1 = host.J[c]()[ti]();
                auto& a        = host.em.B[c]()[ti]();
                for (std::size_t i = 0; i < a.size(); ++i)
                    a.data()[i] = (x0.data()[i] + x1.data()[i]) * .5;
            }
        core::average(dev.em.E, dev.J, dev.em.B); // GPU_UNIFIED: device jobs, see algorithm.hpp
        expect_same_tiles(host.em.B, dev.em.B, "average");
    }
}

// Faraday then Ampere, as the solver within a patch (faraday, fill_magnetic_ghosts, ampere: the
//  level fill only brings the patch's outer ghosts): tiled against untiled. Each tile's J is on its
//  ghost box less 1 (AmpereMode::ShrinkedGhost), from its B ghosts. Faraday writes each tile's
//  Bnew without its ghosts between tiles: without sync_inner_ghosts(Bnew) they are stale, with it
//  J must match the untiled one. Enough cells for tiles between tiles (PLAMN_CELLS is 4: one tile)
TYPED_TEST(AmpereTileTest, faraday_then_ampere)
{
    using Ref       = TestFixture::RefPatch;
    using Cmp       = TestFixture::CmpPatch;
    auto const n    = std::max<std::uint32_t>(cells, 20);
    double const dt = .001;

    struct Pair
    {
        Pair(std::uint32_t const cells_)
            : ref_layout{cells_}
            , cmp_layout{cells_}
        {
        }

        TestLayout<typename Ref::Layout_t> const ref_layout;
        TestLayout<typename Cmp::Layout_t> const cmp_layout;
        Ref ref{ref_layout};
        Cmp cmp{cmp_layout};
        typename Ref::UsableElectromag_t ref_new{*ref_layout};
        typename Cmp::UsableElectromag_t cmp_new{*cmp_layout};
    };

    auto const run_pair = [&](bool const sync) {
        auto pair     = std::make_unique<Pair>(n);
        auto& ref     = pair->ref;
        auto& cmp     = pair->cmp;
        auto& ref_new = pair->ref_new;
        auto& cmp_new = pair->cmp_new;
        if (tile_count(cmp.J) < 2)
            return std::size_t{0};

        FaradayTransformer{}(ref.layout, ref.em.B, ref.em.E, ref_new.B, dt);
        AmpereTransformer<>{}(ref.layout, ref_new.B, ref.J);

        FaradayTransformer const faraday{};
        AmpereTransformer<> const ampere{};
        if constexpr (Cmp::alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
        {
            faraday(cmp.layout, cmp.em.B, cmp.em.E, cmp_new.B, dt);
            if (sync)
                sync_inner_ghosts_on_device(cmp_new.B);
            ampere(cmp.layout, cmp_new.B, cmp.J);
        }
        else
        {
            for (std::size_t ti = 0; ti < tile_count(cmp_new.B); ++ti)
                faraday.on_tile(ti, cmp.em.B, cmp.em.E, cmp_new.B, dt);
            if (sync)
                sync_inner_ghosts(cmp_new.B);
            for (std::size_t ti = 0; ti < tile_count(cmp.J); ++ti)
                ampere.on_tile(ti, cmp_new.B, cmp.J);
        }

        // the GPU's device code contracts to FMAs its own way: ulps apart (1.8e-15 seen)
        double const diff = Cmp::alloc_mode == damn::AllocatorMode::GPU_UNIFIED ? 1e-14 : 1e-15;
        std::size_t n_bad = 0;
        for (std::size_t c = 0; c < 3; ++c)
        {
            auto const eq = compare_fields(ref.J[c], cmp.J[c], diff);
            if (!eq)
            {
                ++n_bad;
                DAMN_LOG_LINE_SS((sync ? "synced" : "unsynced") << " J" << c << ": " << eq.why());
            }
        }
        return n_bad;
    };

    auto const unsynced = run_pair(false);
    auto const synced   = run_pair(true);
    DAMN_LOG_LINE_SS("faraday_then_ampere: J components differing, without the inner ghost sync "
                     << unsynced << ", with it " << synced);
    EXPECT_EQ(synced, 0u);
    if (unsynced)
        std::printf("faraday_then_ampere: %zu of 3 J components differ without "
                    "sync_inner_ghosts(Bnew) (the solver's path within a patch)\n",
                    unsynced);
}

} // namespace plamn::core

int main(int argc, char** argv)
{
    // assert(phlop::ScopeTimerMan::INSTANCE().active);
    ::testing::InitGoogleTest(&argc, argv);
    auto r = RUN_ALL_TESTS();
    DAMN_WITH_PHLOP(if (phlop::scope_timer().active) phlop::shutdown_scope_timer();)
    return r;
}
