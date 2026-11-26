#include "plamn/core/def.hpp"
// B's update is the hybrid's FaradayTransformer, on the MHD layout

#include "damn/core/def/damn_config.hpp"
#include "mkn/kul/env.hpp"

#include "plamn/core/numerics/faraday/faraday_transformer.hpp"
#include "plamn/core/numerics/finite_volume/finite_volume_euler_transformer.hpp"
#include "plamn/simulator/simulator_def.hpp"

#include "tst/plamn/core/data/grid/test_layout_fixtures.hpp"
#include "tst/plamn/core/data/vecfield/test_vecfield_fixtures.hpp"

#include "gtest/gtest.h"

#include "BS_thread_pool.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace plamn::core
{
auto static const cells = mkn::kul::env::get_as("PLAMN_CELLS", std::uint32_t{10});
static ::BS::thread_pool pool{4};
double static const dt = 0.01;

template<std::size_t dim_, auto layout_mode_ = LayoutMode::AoSMapped,
         auto alloc_mode_ = damn::AllocatorMode::CPU>
struct FVPatch
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
    using S                       = MHDQuantity::Scalar;
    using V_                      = MHDQuantity::Vector;

    explicit FVPatch(Layout_t const& layout_)
        : layout{layout_}
    {
    }

    Layout_t layout;
    Grid_t U{"U", layout, S::rho}, Unew{"Unew", layout, S::rho};
    std::array<Grid_t, 3> F{Grid_t{"Fx", layout, S::ScalarFlux_x},
                            Grid_t{"Fy", layout, S::ScalarFlux_y},
                            Grid_t{"Fz", layout, S::ScalarFlux_z}};
    VecField_t B{"B", layout, V_::B}, E{"E", layout, V_::E}, Bnew{"Bnew", layout, V_::B};
};

// fn(amr index) on the ghost box of field, tile by tile if tiled
template<typename Patch>
void fill(Patch const& patch, auto& field, auto&& fn)
{
    auto const one = [&](auto& f, auto const& layout) {
        damn::core::for_each_point(ghost_box(layout, f),
                                   [&](auto const& ijk) { f(ijk) = fn(to_amr(layout, ijk)); });
    };
    if constexpr (is_tiled(Patch::layout_mode))
        for (auto& tile : field())
            one(tile, tile.layout());
    else
        one(field, patch.layout);
}

double wave(auto const& amr, double const base, double const k)
{
    double v = base;
    for (std::size_t d = 0; d < amr.size(); ++d)
        v += .1 * std::sin(k * (amr[d] + 1) * (d + 1));
    return v;
}

template<typename Patch>
void update(Patch& p)
{
    FiniteVolumeEulerTransformer const fve{};
    FaradayTransformer const faraday{};
    auto& [Fx, Fy, Fz] = p.F;
    if constexpr (not is_tiled(Patch::layout_mode)
                  or Patch::alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
    {
        fve(p.layout, dt, p.U, Fx, Fy, Fz, p.Unew);
        faraday(p.layout, p.B, p.E, p.Bnew, dt);
    }
    else
    {
        // a pool task per tile, as the level transformer
        for (std::size_t ti = 0; ti < tile_count(p.Unew); ++ti)
            pool.detach_task([&, ti]() {
                fve.on_tile(ti, dt, p.U, Fx, Fy, Fz, p.Unew);
                faraday.on_tile(ti, p.B, p.E, p.Bnew, dt);
            });
        pool.wait();
    }
}

template<std::size_t dim_, auto layout_mode_ = LayoutMode::AoSMapped,
         auto alloc_mode_ = damn::AllocatorMode::CPU>
struct TestParam
{
    auto constexpr static dim         = dim_;
    auto constexpr static layout_mode = layout_mode_;
    auto constexpr static alloc_mode  = alloc_mode_;
};

template<typename Param>
struct FiniteVolumeTest : public ::testing::Test
{
};
using Dims_t = testing::Types<TestParam<1>, TestParam<2>, TestParam<3>>;
TYPED_TEST_SUITE(FiniteVolumeTest, Dims_t, );

// F_d = s_d * (amr index along d): div F = sum_d s_d / dx_d in every cell
TYPED_TEST(FiniteVolumeTest, linear_fluxes)
{
    using Patch_t             = FVPatch<TypeParam::dim>;
    auto constexpr static dim = Patch_t::dim;
    TestLayout<typename Patch_t::Layout_t> const layout{cells};
    Patch_t p{layout};

    std::array<double, 3> const s{.7, -1.3, 2.1};
    fill(p, p.U, [](auto const& a) { return wave(a, 1., .4); });
    fill(p, p.Unew, [](auto const&) { return 42.; });
    for (std::size_t d = 0; d < 3; ++d)
        fill(p, p.F[d], [&](auto const& a) { return d < dim ? s[d] * a[d] : 1e9; });
    update(p);

    double div = 0;
    for (std::size_t d = 0; d < dim; ++d)
        div += s[d] / layout.dx[d];
    std::size_t n = 0;
    damn::core::for_each_point(FiniteVolumeEulerTransformer::box_for(layout, p.Unew),
                               [&](auto const& ijk) {
                                   ++n;
                                   EXPECT_NEAR(p.Unew(ijk), p.U(ijk) - dt * div, 1e-12);
                               });
    EXPECT_GT(n, 0u);
}


// every point of every tile's box against the untiled field at the same AMR index, within rel_tol
//  of max(|ref|, 1): tiles and patch are separate instantiations, with FMA contraction (gcc
//  -ffp-contract=fast by default, device compilers) each fuses its own way, ~ulps apart
double static max_diff = 0;
std::size_t tile_mismatches(auto const& ref_layout, auto const& ref, auto const& cmp, auto&& box_of,
                            double const rel_tol)
{
    std::size_t n = 0, bad = 0;
    for (auto const& tile : cmp())
    {
        auto const& layout = tile.layout();
        damn::core::for_each_point(box_of(layout, tile), [&](auto const& ijk) {
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

template<typename Param>
struct FiniteVolumeTileTest : public ::testing::Test
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
TYPED_TEST_SUITE(FiniteVolumeTileTest, Tiled_t, );

template<typename Patch>
void fill_waves(Patch& p)
{
    fill(p, p.U, [](auto const& a) { return wave(a, 1., .4); });
    for (std::size_t d = 0; d < 3; ++d)
        fill(p, p.F[d], [&](auto const& a) { return wave(a, .1 * d, .3 + .2 * d); });
    for (std::size_t c = 0; c < 3; ++c)
    {
        fill(p, p.B[c], [&](auto const& a) { return wave(a, -.2 + .1 * c, .4 + .2 * c); });
        fill(p, p.E[c], [&](auto const& a) { return wave(a, .3 * c, .5 + .1 * c); });
    }
}

TYPED_TEST(FiniteVolumeTileTest, tiled_matches_untiled)
{
    using Ref_t = FVPatch<TypeParam::dim>;
    using Cmp_t = FVPatch<TypeParam::dim, TypeParam::layout_mode, TypeParam::alloc_mode>;
    TestLayout<typename Ref_t::Layout_t> const ref_layout{cells};
    TestLayout<typename Cmp_t::Layout_t> const cmp_layout{cells};
    Ref_t ref{ref_layout};
    Cmp_t cmp{cmp_layout};
    fill_waves(ref);
    fill_waves(cmp);
    update(ref);
    update(cmp);

    double const tol = 1e-14;
    auto const fv_box
        = [](auto const& l, auto const& f) { return FiniteVolumeEulerTransformer::box_for(l, f); };
    auto const b_box
        = [](auto const& l, auto const& f) { return FaradayTransformer::box_for(l, f); };
    EXPECT_EQ(tile_mismatches(ref_layout, ref.Unew, cmp.Unew, fv_box, tol), 0u) << "Unew";
    for (std::size_t c = 0; c < 3; ++c)
        EXPECT_EQ(tile_mismatches(ref_layout, ref.Bnew[c], cmp.Bnew[c], b_box, tol), 0u)
            << "Bnew[" << c << "]";
    std::cout << "max difference tiled vs untiled: " << max_diff << std::endl;
}

} // namespace plamn::core


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
