#include "plamn/core/def.hpp"
//  upwind constrained transport (UpwindCTTransformer): ideal Ohm's law E = -v x B on a uniform
//  state, and tiled (per tile pool tasks, or one GPU launch) == untiled

#include "damn/core/def/damn_config.hpp"
#include "damn/core/meta/for_N.hpp"
#include "damn/core/sys/env.hpp"

#include "plamn/core/numerics/constrained_transport/upwind_ct_transformer.hpp"
#include "plamn/core/numerics/reconstructions/linear.hpp"
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
auto static const cells = damn::core::get_env_as("PLAMN_CELLS", std::uint32_t{10});
static ::BS::thread_pool pool{4};
using Linear_t = LinearReconstruction<VanLeerLimiter>;
using CT_t     = UpwindCTTransformer<Linear_t>;

template<std::size_t dim_, auto layout_mode_ = LayoutMode::AoSMapped,
         auto alloc_mode_ = damn::AllocatorMode::CPU>
struct CTPatch
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

    struct Faces // one face direction's CT fields
    {
        Faces(Layout_t const& layout, S const sf, V_ const vf)
            : aL{"aL", layout, sf}
            , aR{"aR", layout, sf}
            , dL{"dL", layout, sf}
            , dR{"dR", layout, sf}
            , vt{"vt", layout, vf}
        {
        }
        Grid_t aL, aR, dL, dR;
        VecField_t vt;
    };

    explicit CTPatch(Layout_t const& layout_)
        : layout{layout_}
    {
    }

    Layout_t layout;
    VecField_t B{"B", layout, V_::B}, E{"E", layout, V_::E};
    std::array<Faces, 3> f{Faces{layout, S::ScalarFlux_x, V_::VecFlux_x},
                           Faces{layout, S::ScalarFlux_y, V_::VecFlux_y},
                           Faces{layout, S::ScalarFlux_z, V_::VecFlux_z}};
};

// f(amr index) on the ghost box of field, tile by tile if tiled
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
void ct(Patch& p)
{
    CT_t const upwind{};
    auto& [x, y, z] = p.f;
    if constexpr (not is_tiled(Patch::layout_mode)
                  or Patch::alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
        upwind(p.layout, p.B, x.vt, x.aL, x.aR, x.dL, x.dR, y.vt, y.aL, y.aR, y.dL, y.dR, z.vt,
               z.aL, z.aR, z.dL, z.dR, p.E);
    else
    {
        // a pool task per tile, as the level transformer
        for (std::size_t ti = 0; ti < tile_count(p.E); ++ti)
            pool.detach_task([&, ti]() {
                upwind.on_tile(ti, p.B, x.vt, x.aL, x.aR, x.dL, x.dR, y.vt, y.aL, y.aR, y.dL, y.dR,
                               z.vt, z.aL, z.aR, z.dL, z.dR, p.E);
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
struct UpwindCTTest : public ::testing::Test
{
};
using Dims_t = testing::Types<TestParam<1>, TestParam<2>, TestParam<3>>;
TYPED_TEST_SUITE(UpwindCTTest, Dims_t, );

// a uniform state with symmetric coefficients (aL = aR = 1/2, dL = dR): the dissipative terms
//  cancel and E = -v x B, for every component with an equation; in 1D E_x, E_y are left as is
TYPED_TEST(UpwindCTTest, ideal_ohm_uniform)
{
    using Patch_t             = CTPatch<TypeParam::dim>;
    auto constexpr static dim = Patch_t::dim;
    TestLayout<typename Patch_t::Layout_t> const layout{cells};
    Patch_t p{layout};

    std::array<double, 3> const v{.3, -.2, .5}, b{.7, .4, -.6};
    auto const constant = [](double const c) { return [c](auto const&) { return c; }; };
    for (std::size_t c = 0; c < 3; ++c)
    {
        fill(p, p.B[c], constant(b[c]));
        fill(p, p.E[c], constant(42.));
    }
    for (auto& face : p.f)
    {
        fill(p, face.aL, constant(.5));
        fill(p, face.aR, constant(.5));
        fill(p, face.dL, constant(1.3));
        fill(p, face.dR, constant(1.3));
        for (std::size_t c = 0; c < 3; ++c)
            fill(p, face.vt[c], constant(v[c]));
    }
    ct(p);

    std::array<double, 3> const minus_vxB{
        -(v[1] * b[2] - v[2] * b[1]), -(v[2] * b[0] - v[0] * b[2]), -(v[0] * b[1] - v[1] * b[0])};
    damn::core::for_N<3>([&](auto c) {
        double const expect
            = UpwindCT<Linear_t>::template has_equation<c, dim> ? minus_vxB[c] : 42.;
        std::size_t n = 0;
        damn::core::for_each_point(CT_t::box_for(layout, p.E[c]), [&](auto const& ijk) {
            ++n;
            EXPECT_NEAR(p.E[c](ijk), expect, 1e-15) << "E[" << c << "]";
        });
        EXPECT_GT(n, 0u);
    });
}


// every point of every tile's box against the untiled field at the same AMR index: equal, or
//  within rel_tol of max(|ref|, 1) (GPU: the device compiler fuses multiply-adds)
double static max_diff = 0;
std::size_t tile_mismatches(auto const& ref_layout, auto const& ref, auto const& cmp,
                            double const rel_tol)
{
    std::size_t n = 0, bad = 0;
    for (auto const& tile : cmp())
    {
        auto const& layout = tile.layout();
        damn::core::for_each_point(CT_t::box_for(layout, tile), [&](auto const& ijk) {
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
struct UpwindCTTileTest : public ::testing::Test
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
TYPED_TEST_SUITE(UpwindCTTileTest, Tiled_t, );

template<typename Patch>
void fill_waves(Patch& p)
{
    for (std::size_t c = 0; c < 3; ++c)
        fill(p, p.B[c], [&](auto const& a) { return wave(a, -.2 + .1 * c, .4 + .2 * c); });
    double k = .3;
    for (auto& face : p.f)
    {
        k += .05;
        fill(p, face.aL, [&](auto const& a) { return wave(a, .5, k); });
        fill(p, face.aR, [&](auto const& a) { return wave(a, .5, k + .1); });
        fill(p, face.dL, [&](auto const& a) { return wave(a, 1., k + .2); });
        fill(p, face.dR, [&](auto const& a) { return wave(a, 1., k + .3); });
        for (std::size_t c = 0; c < 3; ++c)
            fill(p, face.vt[c], [&](auto const& a) { return wave(a, .1 * c, k + .1 * c); });
    }
}

// tiled == untiled within ~ulps (FMA contraction, see tol)
TYPED_TEST(UpwindCTTileTest, tiled_matches_untiled)
{
    using Ref_t = CTPatch<TypeParam::dim>;
    using Cmp_t = CTPatch<TypeParam::dim, TypeParam::layout_mode, TypeParam::alloc_mode>;
    TestLayout<typename Ref_t::Layout_t> const ref_layout{cells};
    TestLayout<typename Cmp_t::Layout_t> const cmp_layout{cells};
    Ref_t ref{ref_layout};
    Cmp_t cmp{cmp_layout};
    fill_waves(ref);
    fill_waves(cmp);
    ct(ref);
    ct(cmp);

    // tiles and patch are separate instantiations: with FMA contraction (gcc -ffp-contract=fast
    //  by default, device compilers) each fuses its own way, ~ulps apart; exact with it off
    double const tol = 1e-14;
    for (std::size_t c = 0; c < 3; ++c)
        EXPECT_EQ(tile_mismatches(ref_layout, ref.E[c], cmp.E[c], tol), 0u) << "E[" << c << "]";
    std::cout << "max difference tiled vs untiled: " << max_diff << std::endl;
}

} // namespace plamn::core


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
