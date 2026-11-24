#include "plamn/core/def.hpp"
//  tests/core/numerics/ohm/test_tileset_ohm.cpp
//
// #define DAMN_UNDEF_ASSERT

#include "damn/core/logger.hpp"    // scope timing
#include "damn/core/def/phlop.hpp" // scope timing

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
#include "plamn/core/numerics/ohm/ohm_transformer.hpp"
#include "plamn/core/data/grid/grid_tiles.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"


#include "plamn/simulator/simulator_def.hpp"

#include "tst/plamn/core/data/field/test_field_fixtures.hpp"
#include "tst/plamn/core/data/grid/test_layout_fixtures.hpp"
#include "tst/plamn/core/data/vecfield/test_vecfield_fixtures.hpp"
#include "tst/plamn/core/data/electromag/test_electromag_fixtures.hpp"

#include "gtest/gtest.h"

#include <cstddef>


namespace plamn::core
{
double static constexpr eta = 1;
double static constexpr nu  = 1;
auto static const cells     = mkn::kul::env::get_as("PLAMN_CELLS", std::uint32_t{22});

void ref_do(auto& patch)
{
    OhmTransformer{{eta, nu}}(patch.layout, patch.n, patch.V, patch.P, patch.em.B, patch.J,
                              patch.emNew.E);
}

void cmp_do(auto& patch)
{
    // CPU: tile by tile, GPU: one launch, block per tile
    OhmTransformer{{eta, nu}}(patch.layout, *patch.n, *patch.V, *patch.P, *patch.em.B, *patch.J,
                              *patch.emNew.E);
}

template<auto alloc_mode>
void compare(auto const& /*layout*/, auto& ref, auto& cmp)
{
    using enum LayoutMode;
    using enum damn::AllocatorMode;
    auto constexpr static n_components = 3;


    for (std::size_t c = 0; c < n_components; ++c)
    {
        auto const eq = compare_fields(ref.emNew.E[c], cmp.emNew.E[c]);
        DAMN_LOG_LINE_SS(eq.why());
        EXPECT_TRUE(eq) << "Failure for E New: " << eq.why();
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
struct OhmTileTest : public ::testing::Test
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
            J.fill(.001);
            V.fill(.001);
            n.fill(.001);
            P.fill(.001);
        }

        Layout_t layout;
        UsableElectromag_t em{layout}, emNew{layout};
        UsableVecField_t J{"J", layout, HybridQuantity::Vector::J};
        UsableVecField_t V{"V", layout, HybridQuantity::Vector::V};
        Field_t n{"rho", layout, HybridQuantity::Scalar::rho};
        Field_t P{"P", layout, HybridQuantity::Scalar::P};
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
            J.fill(.001);
            V.fill(.001);
            n.fill(.001);
            P.fill(.001);
        }

        Layout_t layout;
        UsableElectromag_t em{layout}, emNew{layout};
        UsableVecField_t J{"J", layout, HybridQuantity::Vector::J};
        UsableVecField_t V{"V", layout, HybridQuantity::Vector::V};
        Grid_t n{"rho", layout, HybridQuantity::Scalar::rho};
        Grid_t P{"P", layout, HybridQuantity::Scalar::P};
    };



    using RefPatch = ContiguousPatch;
    using CmpPatch = TileSetPatch;

    OhmTileTest() {}

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

TYPED_TEST_SUITE(OhmTileTest, Permutations_t, );

auto run(auto& self)
{
    ref_do(self.ref_patch);
    cmp_do(self.cmp_patch);
    self.do_compare();
}


TYPED_TEST(OhmTileTest, dispatch)
{
    run(*this);
}

} // namespace plamn::core


int main(int argc, char** argv)
{
    DAMN_WITH_PHLOP(phlop::scope_timer().init();)
    ::testing::InitGoogleTest(&argc, argv);
    auto const ret = RUN_ALL_TESTS();
    DAMN_WITH_PHLOP(if (phlop::scope_timer().active) phlop::shutdown_scope_timer();)
    return ret;
}
