#include "plamn/core/def.hpp"
//  MHD time integrators (Euler, TVDRK2, TVDRK3, SSPRK4_5) on one periodic patch, 1D 2D 3D:
//  a uniform state stays uniform; mass, momentum, energy are conserved; div B stays ~0

#include "mkn/kul/for.hpp"
#include "mkn/kul/env.hpp"

#include "plamn/core/models/mhd_state.hpp"
#include "plamn/core/numerics/reconstructions/linear.hpp"
#include "plamn/core/numerics/time_integrator/mhd_time_integrator.hpp"
#include "plamn/simulator/simulator_def.hpp"

#include "tst/plamn/core/data/grid/test_layout_fixtures.hpp"
#include "tst/plamn/core/numerics/time_integrator/mhd_patch_fixtures.hpp"
#include "tst/plamn/core/data/vecfield/test_vecfield_fixtures.hpp"

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numbers>
#include <string>
#include <type_traits>
#include <utility>

namespace plamn::core
{
auto static const cells    = mkn::kul::env::get_as("PLAMN_CELLS", std::uint32_t{10});
double static const gamma_ = 5. / 3.;
using Linear_t             = LinearReconstruction<VanLeerLimiter>;


// the AMR index wrapped into [0, cells)
auto wrap(auto amr)
{
    for (std::size_t d = 0; d < amr.size(); ++d)
        amr[d] = ((amr[d] % static_cast<int>(cells)) + static_cast<int>(cells))
                 % static_cast<int>(cells);
    return amr;
}

// periodic ghosts: every ghost point (outside the field's domain box) a copy of its wrapped
//  domain point
void periodic(auto const& layout, auto& f)
{
    auto const domain = domain_box(layout, f);
    damn::core::for_each_point(ghost_box(layout, f), [&](auto const& ijk) {
        bool inside = true;
        for (std::size_t d = 0; d < ijk.size(); ++d)
            inside = inside and ijk[d] >= domain.lower[d] and ijk[d] <= domain.upper[d];
        if (not inside)
            f(ijk) = f(to_local(layout, wrap(to_amr(layout, ijk))));
    });
}

template<typename Patch>
struct OnePatchLevel
{
    Patch& p;
    std::size_t state_fills = 0, electric_fills = 0;

    void per_patch(auto&& fn) { fn(p); }
    void fill_state_ghosts(std::size_t const s, double)
    {
        ++state_fills;
        auto& u = p.states[s];
        periodic(p.layout, u.rho);
        periodic(p.layout, u.Etot);
        for (std::size_t c = 0; c < 3; ++c)
        {
            periodic(p.layout, u.rhoV[c]);
            periodic(p.layout, u.B[c]);
        }
    }
    void fill_electric_ghosts(double)
    {
        ++electric_fills;
        for (std::size_t c = 0; c < 3; ++c)
            periodic(p.layout, p.E[c]);
    }
};

// f(x) at the point ijk of field f (wrapped first: periodic copies are bit for bit), x in [0, 1)
//  per direction, + 0.5 cell for dual directions
void set(auto const& layout, auto& f, auto&& fn)
{
    auto constexpr static dim = std::decay_t<decltype(layout)>::dimension;
    auto const centering      = centering_of<dim>(f);
    damn::core::for_each_point(ghost_box(layout, f), [&](auto const& ijk) {
        auto const amr = wrap(to_amr(layout, ijk));
        std::array<double, dim> x{};
        for (std::size_t d = 0; d < dim; ++d)
            x[d] = (amr[d] + (centering[d] == QtyCentering::dual ? .5 : 0.)) / cells;
        f(ijk) = fn(x);
    });
}

double constexpr static two_pi = 2 * std::numbers::pi;

// smooth periodic state; B = B_mean + curl A on the Yee mesh: discretely divergence free
template<typename Patch>
void init_waves(Patch& p)
{
    auto constexpr static dim = Patch::dim;
    auto& u                   = p.states[0];
    auto const& l             = p.layout;
    auto const s              = [](auto const& x, std::size_t const d, double const ph = 0) {
        return d < x.size() ? std::sin(two_pi * x[d] + ph) : 0.;
    };
    set(l, u.rho,
        [&](auto const& x) { return 1. + .2 * s(x, 0) + .1 * s(x, 1, 1.) + .1 * s(x, 2, 2.); });
    set(l, u.P,
        [&](auto const& x) { return 1. + .1 * s(x, 0, .5) + .1 * s(x, 1) + .05 * s(x, 2); });
    for (std::size_t c = 0; c < 3; ++c)
        set(l, u.V[c],
            [&](auto const& x) { return .1 * c + .1 * s(x, (c + 1) % 3) + .05 * s(x, c, .3); });

    // A on the edges (A_c along c: dual along c, primal across), B_c from its differences
    auto const A = [&](std::size_t const c, auto x /*node*/) {
        // a smooth periodic function per component
        double const a = .05 * (c + 1);
        return a * std::sin(two_pi * (x[0] + .1 * c))
               * (dim > 1 ? std::cos(two_pi * x[dim > 1 ? 1 : 0]) : 1.)
               * (dim > 2 ? std::cos(two_pi * x[dim > 2 ? 2 : 0] + c) : 1.);
    };
    std::array<double, 3> const B_mean{.8, .4, -.3};
    double const dx = 1. / cells;
    for (std::size_t c = 0; c < 3; ++c)
        set(l, u.B[c], [&](auto const& x) {
            // B_c = d_{c+1} A_{c+2} - d_{c+2} A_{c+1}, differences along existing directions
            auto const e = [&](std::size_t const k, std::size_t const along) {
                if (along >= dim)
                    return 0.;
                // A_k at this B point shifted to the edges +- half a cell along `along`
                auto xp = x, xm = x;
                xp[along] += .5 * dx;
                xm[along] -= .5 * dx;
                // A_k is dual along k: B_c is dual along k (k != c), already at the dual offset
                return (A(k, xp) - A(k, xm)) / dx;
            };
            std::size_t const c1 = (c + 1) % 3, c2 = (c + 2) % 3;
            return B_mean[c] + e(c2, c1) - e(c1, c2);
        });

    PrimitiveConservativeTransformer{gamma_}.to_conservative(l, u.rho, u.V, u.B, u.P, u.rhoV,
                                                             u.Etot);
}

template<typename Patch>
void init_uniform(Patch& p)
{
    auto& u       = p.states[0];
    auto const& l = p.layout;
    set(l, u.rho, [](auto const&) { return 1.3; });
    set(l, u.P, [](auto const&) { return .7; });
    for (std::size_t c = 0; c < 3; ++c)
    {
        set(l, u.V[c], [&](auto const&) { return .2 - .1 * c; });
        set(l, u.B[c], [&](auto const&) { return .5 + .2 * c; });
    }
    PrimitiveConservativeTransformer{gamma_}.to_conservative(l, u.rho, u.V, u.B, u.P, u.rhoV,
                                                             u.Etot);
}

// sum over the domain
double total(auto const& layout, auto const& f)
{
    double sum = 0;
    damn::core::for_each_point(domain_box(layout, f), [&](auto const& ijk) { sum += f(ijk); });
    return sum;
}

// max |div B| over the cells (B_d at the cell's faces, B dual across d)
template<typename Patch>
double max_div_B(Patch const& p)
{
    auto constexpr static dim = Patch::dim;
    auto const& B             = p.states[0].B;
    double m                  = 0;
    damn::core::for_each_point(domain_box(p.layout, p.states[0].rho), [&](auto const& ijk) {
        double div = 0;
        for (std::size_t d = 0; d < dim; ++d)
        {
            auto next = ijk;
            next[d] += 1;
            div += (B[d](next) - B[d](ijk)) * cells;
        }
        m = std::max(m, std::abs(div));
    });
    return m;
}

template<std::size_t dim_, std::size_t which_>
struct TestParam
{
    auto constexpr static dim   = dim_;
    auto constexpr static which = which_;
};

template<std::size_t which>
auto integrator()
{
    MHDSteps<Linear_t> const steps{gamma_};
    if constexpr (which == 0)
        return EulerIntegrator<Linear_t>{steps};
    else if constexpr (which == 1)
        return TVDRK2Integrator<Linear_t>{steps};
    else if constexpr (which == 2)
        return TVDRK3Integrator<Linear_t>{steps};
    else
        return SSPRK4_5Integrator<Linear_t>{steps};
}

template<typename Param>
struct MHDIntegratorTest : public ::testing::Test
{
    using Patch_t = IntegratorPatch<Param::dim>;
    TestLayout<typename Patch_t::Layout_t> const layout{cells};
    Patch_t patch{layout};
    OnePatchLevel<Patch_t> level{patch};
    double const dt = .2 / cells / 3.; // CFL ~ .2 for speeds up to ~3

    void run(std::size_t const n_steps)
    {
        level.fill_state_ghosts(0, 0);
        auto const integrate = integrator<Param::which>();
        static_assert(std::remove_cvref_t<decltype(integrate)>::n_states <= Patch_t::n_states);
        for (std::size_t i = 0; i < n_steps; ++i)
            integrate(level, i * dt, (i + 1) * dt);
    }
};
// clang-format off
using Params_t = testing::Types<
    TestParam<1, 0>, TestParam<2, 0>, TestParam<3, 0>,
    TestParam<1, 1>, TestParam<2, 1>, TestParam<3, 1>,
    TestParam<1, 2>, TestParam<2, 2>, TestParam<3, 2>,
    TestParam<1, 3>, TestParam<2, 3>, TestParam<3, 3>>;
// clang-format on
TYPED_TEST_SUITE(MHDIntegratorTest, Params_t, );

TYPED_TEST(MHDIntegratorTest, uniform_stays_uniform)
{
    auto& p = this->patch;
    init_uniform(p);
    auto const& u     = p.states[0];
    auto const first  = domain_box(p.layout, u.rho).lower;
    double const rho  = u.rho(first);
    double const Etot = u.Etot(first);
    this->run(3);
    damn::core::for_each_point(domain_box(p.layout, u.rho), [&](auto const& ijk) {
        EXPECT_NEAR(u.rho(ijk), rho, 1e-14);
        EXPECT_NEAR(u.Etot(ijk), Etot, 1e-14);
    });
    for (std::size_t c = 0; c < 3; ++c)
        damn::core::for_each_point(domain_box(p.layout, u.B[c]), [&](auto const& ijk) {
            EXPECT_NEAR(u.B[c](ijk), .5 + .2 * c, 1e-14);
        });
}

TYPED_TEST(MHDIntegratorTest, conservation_and_div_B)
{
    auto& p = this->patch;
    init_waves(p);
    auto const& u = p.states[0];
    auto const& l = p.layout;

    auto const totals = [&]() {
        return std::array<double, 5>{total(l, u.rho), total(l, u.rhoV[0]), total(l, u.rhoV[1]),
                                     total(l, u.rhoV[2]), total(l, u.Etot)};
    };
    auto const before = totals();
    double const div0 = max_div_B(p);
    this->run(5);
    auto const after = totals();

    for (std::size_t i = 0; i < before.size(); ++i)
        EXPECT_NEAR(after[i], before[i], 1e-12 * std::max(1., std::abs(before[i])))
            << "quantity " << i;
    damn::core::for_each_point(domain_box(l, u.rho), [&](auto const& ijk) {
        ASSERT_TRUE(std::isfinite(u.Etot(ijk)));
        ASSERT_GT(u.rho(ijk), 0.);
    });
    double const div1 = max_div_B(p);
    EXPECT_LT(div0, 1e-12);
    EXPECT_LT(div1, 1e-12);
    std::cout << "div B before " << div0 << " after " << div1 << ", fills: state "
              << this->level.state_fills << " electric " << this->level.electric_fills << std::endl;
}

} // namespace plamn::core


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
