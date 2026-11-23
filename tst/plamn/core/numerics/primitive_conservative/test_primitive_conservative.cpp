#include "plamn/core/def.hpp"
//  MHD primitive <-> conservative conversion (PrimitiveConservativeTransformer): values against the
//  formulas, the round trip, and tiled (per tile pool tasks, or one GPU launch) == untiled

#include "damn/core/def/damn_config.hpp"
#include "mkn/kul/for.hpp"
#include "mkn/kul/env.hpp"

#include "plamn/core/numerics/primitive_conservative/primitive_conservative_transformer.hpp"
#include "plamn/simulator/simulator_def.hpp"

#include "tst/plamn/core/data/field/test_field_fixtures.hpp"
#include "tst/plamn/core/data/grid/test_grid_fixtures.hpp"
#include "tst/plamn/core/data/grid/test_layout_fixtures.hpp"
#include "tst/plamn/core/data/vecfield/test_vecfield_fixtures.hpp"

#include "gtest/gtest.h"

#include "BS_thread_pool.hpp"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace plamn::core
{
auto static const cells    = mkn::kul::env::get_as("PLAMN_CELLS", std::uint32_t{10});
double static const gamma_ = 5. / 3.;
static ::BS::thread_pool pool{4};

template<std::size_t dim_, auto layout_mode_ = LayoutMode::AoSMapped,
         auto alloc_mode_ = damn::AllocatorMode::CPU>
struct MHDPatch
{
    auto constexpr static dim         = dim_;
    auto constexpr static layout_mode = layout_mode_;
    auto constexpr static alloc_mode  = alloc_mode_;
    auto constexpr static opts
        = SimOpts{.dimension           = dim,
                  .interp_order        = 1,
                  .layout_mode         = layout_mode,
                  .alloc_mode          = alloc_mode,
                  .reconstruction_type = MHDOpts::ReconstructionType::Linear,
                  .slope_limiter_type  = MHDOpts::SlopeLimiterType::VanLeer,
                  .riemann_solver_type = MHDOpts::RiemannSolverType::Rusanov};
    using Types                   = PLAMN_Types<opts>::MHD;
    using Layout_t                = Types::Layout_t;
    using Grid_t                  = Types::Grid_t;
    auto constexpr static vf_opts = TensorFieldOptions<Types>{};
    using VecField_t              = UsableVecField<vf_opts>;

    explicit MHDPatch(Layout_t const& layout_)
        : layout{layout_}
    {
    }

    Layout_t layout;
    Grid_t rho{"rho", layout, MHDQuantity::Scalar::rho}, P{"P", layout, MHDQuantity::Scalar::P};
    Grid_t Etot{"Etot", layout, MHDQuantity::Scalar::Etot};
    Grid_t P2{"P2", layout, MHDQuantity::Scalar::P};
    VecField_t V{"V", layout, MHDQuantity::Vector::V}, B{"B", layout, MHDQuantity::Vector::B};
    VecField_t rhoV{"rhoV", layout, MHDQuantity::Vector::rhoV};
    VecField_t V2{"V2", layout, MHDQuantity::Vector::V};
};

// a smooth function of the AMR index, the same on every tile
double wave(auto const& amr, double const base, double const k)
{
    double v = base;
    for (std::size_t d = 0; d < amr.size(); ++d)
        v += .1 * std::sin(k * (amr[d] + 1) * (d + 1));
    return v;
}

// f(amr index) on the ghost box of field, tile by tile if tiled
template<typename Patch>
void fill(Patch const& patch, auto& field, double const base, double const k)
{
    auto const one = [&](auto& f, auto const& layout) {
        damn::core::for_each_point(ghost_box(layout, f), [&](auto const& ijk) {
            f(ijk) = wave(to_amr(layout, ijk), base, k);
        });
    };
    if constexpr (is_tiled(Patch::layout_mode))
        for (auto& tile : field())
            one(tile, tile.layout());
    else
        one(field, patch.layout);
}

template<typename Patch>
void fill_primitive(Patch& p)
{
    fill(p, p.rho, 1., .7);
    fill(p, p.P, .5, .3);
    for (std::size_t c = 0; c < 3; ++c)
    {
        fill(p, p.V[c], .1 * c, .5 + .1 * c);
        fill(p, p.B[c], -.2 + .1 * c, .4 + .2 * c);
    }
}

template<typename Patch>
void convert(Patch& p)
{
    PrimitiveConservativeTransformer const pc{gamma_};
    if constexpr (not is_tiled(Patch::layout_mode)
                  or Patch::alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
    {
        pc.to_conservative(p.layout, p.rho, p.V, p.B, p.P, p.rhoV, p.Etot);
        pc.to_primitive(p.layout, p.rho, p.rhoV, p.B, p.Etot, p.V2, p.P2);
    }
    else
    {
        // a pool task per tile, as the level transformer
        for (std::size_t ti = 0; ti < tile_count(p.rho); ++ti)
            pool.detach_task([&, ti]() {
                pc.to_conservative_on_tile(ti, p.rho, p.V, p.B, p.P, p.rhoV, p.Etot);
                pc.to_primitive_on_tile(ti, p.rho, p.rhoV, p.B, p.Etot, p.V2, p.P2);
            });
        pool.wait();
    }
}


template<typename Param>
struct PrimitiveConservativeTest : public ::testing::Test
{
};

template<std::size_t dim_, auto layout_mode_ = LayoutMode::AoSMapped,
         auto alloc_mode_ = damn::AllocatorMode::CPU>
struct TestParam
{
    auto constexpr static dim         = dim_;
    auto constexpr static layout_mode = layout_mode_;
    auto constexpr static alloc_mode  = alloc_mode_;
};

using Dims_t = testing::Types<TestParam<1>, TestParam<2>, TestParam<3>>;
TYPED_TEST_SUITE(PrimitiveConservativeTest, Dims_t, );

// to_conservative against the formulas (B averaged from its two faces along its own direction,
//  the cell value in the others), then to_primitive gives back V and P
TYPED_TEST(PrimitiveConservativeTest, values)
{
    using Patch_t             = MHDPatch<TypeParam::dim>;
    auto constexpr static dim = Patch_t::dim;
    TestLayout<typename Patch_t::Layout_t> const layout{cells};
    Patch_t p{layout};
    fill_primitive(p);
    convert(p);

    std::size_t n = 0;
    damn::core::for_each_point(ghost_box(layout, p.rho), [&](auto const& ijk) {
        ++n;
        double b[3];
        for (std::size_t c = 0; c < 3; ++c)
        {
            b[c] = p.B[c](ijk);
            if (c < dim)
            {
                auto next = ijk;
                ++next[c];
                b[c] = .5 * (b[c] + p.B[c](next));
            }
        }
        double const r = p.rho(ijk), vx = p.V[0](ijk), vy = p.V[1](ijk), vz = p.V[2](ijk);
        double const etot = p.P(ijk) / (gamma_ - 1) + .5 * r * (vx * vx + vy * vy + vz * vz)
                            + .5 * (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
        EXPECT_DOUBLE_EQ(p.rhoV[0](ijk), r * vx);
        EXPECT_DOUBLE_EQ(p.rhoV[1](ijk), r * vy);
        EXPECT_DOUBLE_EQ(p.rhoV[2](ijk), r * vz);
        EXPECT_NEAR(p.Etot(ijk), etot, 1e-14 * std::abs(etot));
        for (std::size_t c = 0; c < 3; ++c)
            EXPECT_NEAR(p.V2[c](ijk), p.V[c](ijk), 1e-14);
        EXPECT_NEAR(p.P2(ijk), p.P(ijk), 1e-13);
    });
    EXPECT_GT(n, 0u);
}


template<typename Param>
struct PrimitiveConservativeTileTest : public ::testing::Test
{
};

// clang-format off
using Tiled_t = testing::Types<
     TestParam<2, LayoutMode::AoSPCTS>
    ,TestParam<3, LayoutMode::AoSPCTS>
DAMN_WITH_GPU(
    ,TestParam<2, LayoutMode::AoSPCTS, damn::AllocatorMode::GPU_UNIFIED>
    ,TestParam<3, LayoutMode::AoSPCTS, damn::AllocatorMode::GPU_UNIFIED>
)
>;
// clang-format on
TYPED_TEST_SUITE(PrimitiveConservativeTileTest, Tiled_t, );

// every point of every tile's ghost box against the untiled field at the same AMR index: points
//  (not compare_fields' reduce, which sums the tiles' overlapping ghosts as for deposits), within
//  rel_tol of max(|ref|, 1) (0: bit for bit)
double static max_diff = 0;
std::size_t tile_mismatches(auto const& ref_layout, auto const& ref, auto const& cmp,
                            double const rel_tol)
{
    std::size_t n = 0, bad = 0;
    for (auto const& tile : cmp())
    {
        auto const& layout = tile.layout();
        damn::core::for_each_point(ghost_box(layout, tile), [&](auto const& ijk) {
            ++n;
            double const t = tile(ijk), r = ref(to_local(ref_layout, to_amr(layout, ijk)));
            double const d = t == r ? 0 : std::abs(t - r) / std::max(std::abs(r), 1.0);
            max_diff       = std::max(max_diff, d);
            bad += d > rel_tol;
        });
    }
    EXPECT_GT(n, 0u);
    return bad;
}

// tiled (per tile pool tasks, or one GPU launch) == untiled: bit for bit on the CPU (the same
//  code per point); on the GPU within 1e-14 (device code contracts to FMAs its own way)
TYPED_TEST(PrimitiveConservativeTileTest, tiled_matches_untiled)
{
    using Ref_t = MHDPatch<TypeParam::dim>;
    using Cmp_t = MHDPatch<TypeParam::dim, TypeParam::layout_mode, TypeParam::alloc_mode>;
    TestLayout<typename Ref_t::Layout_t> const ref_layout{cells};
    TestLayout<typename Cmp_t::Layout_t> const cmp_layout{cells};
    Ref_t ref{ref_layout};
    Cmp_t cmp{cmp_layout};
    fill_primitive(ref);
    fill_primitive(cmp);
    convert(ref);
    convert(cmp);

    double const tol = TypeParam::alloc_mode == damn::AllocatorMode::GPU_UNIFIED ? 1e-14 : 0;
    max_diff         = 0;
    EXPECT_EQ(tile_mismatches(ref_layout, ref.Etot, cmp.Etot, tol), 0u) << "Etot";
    EXPECT_EQ(tile_mismatches(ref_layout, ref.P2, cmp.P2, tol), 0u) << "P";
    for (std::size_t c = 0; c < 3; ++c)
    {
        EXPECT_EQ(tile_mismatches(ref_layout, ref.rhoV[c], cmp.rhoV[c], tol), 0u) << "rhoV " << c;
        EXPECT_EQ(tile_mismatches(ref_layout, ref.V2[c], cmp.V2[c], tol), 0u) << "V " << c;
    }
    std::printf("primitive / conservative: tiled vs untiled max relative diff %g\n", max_diff);
}

} // namespace plamn::core


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
