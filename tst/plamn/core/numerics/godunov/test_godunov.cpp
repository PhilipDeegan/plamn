#include "plamn/core/def.hpp"
//  MHD faces: Linear + VanLeer reconstruction, ideal fluxes, Rusanov; GodunovTransformer per face
//  against GodunovFace, and tiled (per tile pool tasks, or one GPU launch) == untiled

#include "damn/core/def/damn_config.hpp"
#include "mkn/kul/for.hpp"
#include "mkn/kul/env.hpp"

#include "plamn/core/numerics/godunov/godunov_transformer.hpp"
#include "plamn/core/numerics/reconstructions/linear.hpp"
#include "plamn/simulator/simulator_def.hpp"

#include "tst/plamn/core/data/grid/test_layout_fixtures.hpp"
#include "tst/plamn/core/data/vecfield/test_vecfield_fixtures.hpp"

#include "gtest/gtest.h"

#include "BS_thread_pool.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <iostream>

namespace plamn::core
{
auto static const cells    = mkn::kul::env::get_as("PLAMN_CELLS", std::uint32_t{10});
double static const gamma_ = 5. / 3.;
static ::BS::thread_pool pool{4};
using Linear_t = LinearReconstruction<VanLeerLimiter>;

TEST(GodunovPoint, linear_reconstruction)
{
    // linear data: both sides of the face are its exact value, the midpoint of u_1 and u
    auto const [l, r] = Linear_t::from_values(1., 2., 3., 4.);
    EXPECT_DOUBLE_EQ(l, 2.5);
    EXPECT_DOUBLE_EQ(r, 2.5);
    // an extremum: no slope, each side its cell's value
    auto const [l2, r2] = Linear_t::from_values(1., 3., 1., 3.);
    EXPECT_DOUBLE_EQ(l2, 3.);
    EXPECT_DOUBLE_EQ(r2, 1.);
}

TEST(GodunovPoint, rusanov_consistency)
{
    // equal left and right states: no dissipation, the physical flux
    MHDPrimitive const u{1.2, {.3, -.2, .1}, {.7, .4, -.5}, .9};
    mkn::kul::for_N<3>([&](auto dir) {
        auto const r = rusanov<dir>(gamma_, u, u);
        auto const f = ideal_mhd_flux<dir>(gamma_, u);
        EXPECT_DOUBLE_EQ(r.F.rho, f.rho);
        EXPECT_DOUBLE_EQ(r.F.Etot, f.Etot);
        for (std::size_t c = 0; c < 3; ++c)
        {
            EXPECT_DOUBLE_EQ(r.F.rhoV[c], f.rhoV[c]);
            EXPECT_DOUBLE_EQ(r.F.B[c], f.B[c]);
            EXPECT_DOUBLE_EQ(r.vt[c], u.V[c]);
        }
        EXPECT_EQ(r.F.B[dir], 0.);
        double const b2 = .7 * .7 + .4 * .4 + .5 * .5;
        double const S  = std::abs(u.V[dir]) + fast_magnetosonic(gamma_, u.rho, u.B[dir], b2, u.P);
        EXPECT_DOUBLE_EQ(r.uct_coefs[0], .5);
        EXPECT_DOUBLE_EQ(r.uct_coefs[1], .5);
        EXPECT_DOUBLE_EQ(r.uct_coefs[2], .5 * S);
        EXPECT_DOUBLE_EQ(r.uct_coefs[3], .5 * S);
    });
}


template<std::size_t dim_, auto layout_mode_ = LayoutMode::AoSMapped,
         auto alloc_mode_ = damn::AllocatorMode::CPU>
struct FacePatch
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

    // the fields of the faces of one direction
    struct Faces
    {
        Faces(Layout_t const& layout, S const sf, V_ const vf)
            : F_rho{"F_rho", layout, sf}
            , F_Etot{"F_Etot", layout, sf}
            , aL{"aL", layout, sf}
            , aR{"aR", layout, sf}
            , dL{"dL", layout, sf}
            , dR{"dR", layout, sf}
            , F_rhoV{"F_rhoV", layout, vf}
            , F_B{"F_B", layout, vf}
            , vt{"vt", layout, vf}
        {
        }
        Grid_t F_rho, F_Etot, aL, aR, dL, dR;
        VecField_t F_rhoV, F_B, vt;
    };

    explicit FacePatch(Layout_t const& layout_)
        : layout{layout_}
    {
    }

    Layout_t layout;
    Grid_t rho{"rho", layout, S::rho}, P{"P", layout, S::P};
    VecField_t V{"V", layout, V_::V}, B{"B", layout, V_::B};
    std::array<Faces, 3> faces{Faces{layout, S::ScalarFlux_x, V_::VecFlux_x},
                               Faces{layout, S::ScalarFlux_y, V_::VecFlux_y},
                               Faces{layout, S::ScalarFlux_z, V_::VecFlux_z}};
};

double wave(auto const& amr, double const base, double const k)
{
    double v = base;
    for (std::size_t d = 0; d < amr.size(); ++d)
        v += .1 * std::sin(k * (amr[d] + 1) * (d + 1));
    return v;
}

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

template<std::size_t dir, typename Patch>
void faces(Patch& p)
{
    GodunovTransformer<Linear_t> const godunov{gamma_};
    auto& f = p.faces[dir];
    if constexpr (not is_tiled(Patch::layout_mode)
                  or Patch::alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
        godunov.template operator()<dir>(p.layout, p.rho, p.V, p.B, p.P, f.F_rho, f.F_rhoV, f.F_B,
                                         f.F_Etot, f.vt, f.aL, f.aR, f.dL, f.dR);
    else
    {
        // a pool task per tile, as the level transformer
        for (std::size_t ti = 0; ti < tile_count(p.rho); ++ti)
            pool.detach_task([&, ti]() {
                godunov.template on_tile<dir>(ti, p.rho, p.V, p.B, p.P, f.F_rho, f.F_rhoV, f.F_B,
                                              f.F_Etot, f.vt, f.aL, f.aR, f.dL, f.dR);
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
struct GodunovTest : public ::testing::Test
{
};
using Dims_t = testing::Types<TestParam<1>, TestParam<2>, TestParam<3>>;
TYPED_TEST_SUITE(GodunovTest, Dims_t, );

// the transformer's stored values at every face of the box are GodunovFace's at that face
TYPED_TEST(GodunovTest, transformer_matches_face)
{
    using Patch_t = FacePatch<TypeParam::dim>;
    TestLayout<typename Patch_t::Layout_t> const layout{cells};
    Patch_t p{layout};
    fill_primitive(p);
    GodunovFace<Linear_t> const face{gamma_};

    mkn::kul::for_N<Patch_t::dim>([&](auto dir) {
        faces<dir>(p);
        auto const& f = p.faces[dir];
        std::size_t n = 0;
        damn::core::for_each_point(
            GodunovTransformer<Linear_t>::template box_for<dir>(layout, f.F_rho),
            [&](auto const& ijk) {
                ++n;
                auto const r = face.template operator()<dir>(ijk, p.rho, p.V, p.B, p.P);
                EXPECT_EQ(f.F_rho(ijk), r.F.rho);
                EXPECT_EQ(f.F_Etot(ijk), r.F.Etot);
                EXPECT_EQ(f.dR(ijk), r.uct_coefs[3]);
                for (std::size_t c = 0; c < 3; ++c)
                {
                    EXPECT_EQ(f.F_rhoV[c](ijk), r.F.rhoV[c]);
                    EXPECT_EQ(f.F_B[c](ijk), r.F.B[c]);
                    EXPECT_EQ(f.vt[c](ijk), r.vt[c]);
                }
                EXPECT_TRUE(std::isfinite(r.F.Etot));
            });
        EXPECT_GT(n, 0u);
    });
}


// every point of every tile's face box against the untiled field at the same AMR index: equal, or
//  within rel_tol of max(|ref|, 1) (GPU: the device compiler fuses multiply-adds where the host's
//  does not; near zero fluxes, cancellations, make a plain relative difference meaningless)
double static max_rel_diff = 0;
std::size_t tile_mismatches(auto const& ref_layout, auto const& ref, auto const& cmp, auto&& box_of,
                            double const rel_tol = 0)
{
    std::size_t n = 0, bad = 0;
    for (auto const& tile : cmp())
    {
        auto const& layout = tile.layout();
        damn::core::for_each_point(box_of(layout, tile), [&](auto const& ijk) {
            ++n;
            double const t = tile(ijk), r = ref(to_local(ref_layout, to_amr(layout, ijk)));
            double const rel = t == r ? 0 : std::abs(t - r) / std::max(std::abs(r), 1.0);
            max_rel_diff     = std::max(max_rel_diff, rel);
            bad += rel > rel_tol;
        });
    }
    EXPECT_GT(n, 0u);
    return bad;
}

template<typename Param>
struct GodunovTileTest : public ::testing::Test
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
TYPED_TEST_SUITE(GodunovTileTest, Tiled_t, );

// tiled (per tile pool tasks, or one GPU launch) == untiled within ~ulps (FMA contraction, see
// tol), every direction
TYPED_TEST(GodunovTileTest, tiled_matches_untiled)
{
    using Ref_t = FacePatch<TypeParam::dim>;
    using Cmp_t = FacePatch<TypeParam::dim, TypeParam::layout_mode, TypeParam::alloc_mode>;
    TestLayout<typename Ref_t::Layout_t> const ref_layout{cells};
    TestLayout<typename Cmp_t::Layout_t> const cmp_layout{cells};
    Ref_t ref{ref_layout};
    Cmp_t cmp{cmp_layout};
    fill_primitive(ref);
    fill_primitive(cmp);

    mkn::kul::for_N<Ref_t::dim>([&](auto dir) {
        faces<dir>(ref);
        faces<dir>(cmp);
        auto const box_of = [dir](auto const& layout, auto const& f) {
            return GodunovTransformer<Linear_t>::template box_for<dir>(layout, f);
        };
        auto const& r = ref.faces[dir];
        auto const& c = cmp.faces[dir];
        // tiles and patch are separate instantiations: with FMA contraction (gcc -ffp-contract=fast
        //  by default, device compilers) each fuses its own way, ~ulps apart; exact with it off
        double const tol = 1e-14;
        EXPECT_EQ(tile_mismatches(ref_layout, r.F_rho, c.F_rho, box_of, tol), 0u)
            << "F_rho " << dir;
        EXPECT_EQ(tile_mismatches(ref_layout, r.F_Etot, c.F_Etot, box_of, tol), 0u)
            << "F_Etot " << dir;
        EXPECT_EQ(tile_mismatches(ref_layout, r.dL, c.dL, box_of, tol), 0u) << "dL " << dir;
        for (std::size_t k = 0; k < 3; ++k)
        {
            EXPECT_EQ(tile_mismatches(ref_layout, r.F_rhoV[k], c.F_rhoV[k], box_of, tol), 0u)
                << "F_rhoV";
            EXPECT_EQ(tile_mismatches(ref_layout, r.F_B[k], c.F_B[k], box_of, tol), 0u) << "F_B";
            EXPECT_EQ(tile_mismatches(ref_layout, r.vt[k], c.vt[k], box_of, tol), 0u) << "vt";
        }
    });
    std::cout << "max relative difference tiled vs untiled: " << max_rel_diff << std::endl;
}

} // namespace plamn::core


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
