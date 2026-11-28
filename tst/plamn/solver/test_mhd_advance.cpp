#include "plamn/core/def.hpp"
// one 1D patch covering a non periodic domain, its ghosts past the domain filled by the model's
//  physical boundaries

#include "damn/amr/hierarchy.hpp"
#include "damn/amr/physical_boundary.hpp"
#include "damn/core/geometry/box_spans.hpp"

#include "plamn/core/boundary/boundary_defs.hpp"
#include "plamn/core/boundary/physical_boundaries.hpp"
#include "plamn/core/data/grid/layout.hpp"
#include "plamn/core/models/options/mhd_options_def.hpp"
#include "plamn/core/numerics/primitive_conservative/primitive_conservative.hpp"
#include "plamn/initializer/data_provider.hpp"
#include "plamn/solver/advance.hpp"
#include "plamn/solver/mhd/model.hpp"

#include "tst/plamn/core/data/grid/test_layout_fixtures.hpp"
#include "tst/plamn/core/numerics/time_integrator/mhd_patch_fixtures.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace plamn::solver
{
namespace
{
    using Patch_t                 = core::IntegratorPatch<1>;
    using Layout_t                = Patch_t::Layout_t;
    using Model_t                 = MHDModel<Patch_t::opts>;
    using Scheme                  = MHDOpts::TimeIntegratorType;
    std::uint32_t constexpr cells = 32;
    double constexpr dt           = .2 / cells / 3.;
    std::array constexpr all_schemes
        = {Scheme::Euler, Scheme::TVDRK2, Scheme::TVDRK3, Scheme::SSPRK4_5};

    initializer::Dict input(Scheme const scheme, core::BoundaryType const type)
    {
        initializer::Dict sim;
        sim["algo"]["fv_method"]["heat_capacity_ratio"] = 5. / 3.;
        sim["algo"]["time_integrator_type"]             = static_cast<int>(scheme);
        sim["grid"]["periodicities"]["x"]               = false;
        for (auto const* loc : {"xlower", "xupper"})
            sim["grid"]["domain_boundaries"][loc]["type"] = static_cast<int>(type);
        return sim;
    }

    // one patch, the whole domain: no neighbour, its ghosts past the domain the boundaries'
    struct WalledLevel
    {
        Patch_t& p;
        Model_t const& model;
        std::vector<damn::amr::PhysicalBoundaryRegion<1>> regions;

        WalledLevel(Patch_t& p_, Model_t const& model_)
            : p{p_}
            , model{model_}
        {
            damn::amr::Hierarchy<1> h;
            h.domain   = p.layout.cells;
            h.dx       = p.layout.dx;
            h.periodic = {false};
            h.levels.resize(1);
            regions
                = damn::amr::physical_boundary_regions(h, 0, p.layout.cells, Layout_t::ghost_width);
        }

        void per_patch(auto&& fn) { fn(p); }
        void fill_state_ghosts(std::size_t const s, double)
        {
            auto& u = p.states[s];
            core::fill_physical_boundaries(model.boundaries, regions, p.layout, u.rho, u.Etot,
                                           u.rhoV, u.B);
        }
        void fill_electric_ghosts(double)
        {
            core::fill_physical_boundaries(model.boundaries, regions, p.layout, p.E);
        }
    };

    // the primitive state at the points of the patch's fields (x in [0, 1)), to conservative
    void init(Patch_t& p, double const gamma, auto&& rho, auto&& P, auto&& vx, auto&& By)
    {
        auto& u        = p.states[0];
        auto const& l  = p.layout;
        auto const set = [&](auto& f, auto&& fn) {
            auto const centering = core::centering_of<1>(f);
            damn::core::for_each_point(core::ghost_box(l, f), [&](auto const& ijk) {
                auto const amr = core::to_amr(l, ijk);
                f(ijk)
                    = fn((amr[0] + (centering[0] == core::QtyCentering::dual ? .5 : 0.)) / cells);
            });
        };
        set(u.rho, rho);
        set(u.P, P);
        set(u.V[0], vx);
        set(u.V[1], [](double) { return 0.; });
        set(u.V[2], [](double) { return 0.; });
        set(u.B[0], [](double) { return 0.; });
        set(u.B[1], By);
        set(u.B[2], [](double) { return .3; });
        core::PrimitiveConservativeTransformer{gamma}.to_conservative(l, u.rho, u.V, u.B, u.P,
                                                                      u.rhoV, u.Etot);
    }

    double total(auto const& layout, auto const& f)
    {
        double sum = 0;
        damn::core::for_each_point(core::domain_box(layout, f),
                                   [&](auto const& ijk) { sum += f(ijk); });
        return sum;
    }
} // namespace

TEST(MHDModel, fromInput)
{
    auto const m = Model_t::from(input(Scheme::TVDRK3, core::BoundaryType::Reflective));
    EXPECT_DOUBLE_EQ(m.gamma, 5. / 3.);
    EXPECT_EQ(m.scheme, Scheme::TVDRK3);
    EXPECT_EQ(m.n_states(), 3u);
    for (auto const loc : {core::BoundaryLocation::XLower, core::BoundaryLocation::XUpper})
    {
        auto const* b = m.boundaries.boundary(loc);
        ASSERT_NE(b, nullptr);
        EXPECT_EQ(b->type, core::BoundaryType::Reflective);
    }
}

// hyper resistivity only alongside the Hall term
TEST(MHDModel, hyperResistivityNeedsHall)
{
    auto sim                                      = input(Scheme::TVDRK2, core::BoundaryType::Open);
    sim["algo"]["fv_method"]["hyper_resistivity"] = .01;
    EXPECT_THROW((void)Model_t::from(sim), std::runtime_error);
    sim["algo"]["fv_method"]["hyper_resistivity"] = 0.;
    sim["algo"]["fv_method"]["resistivity"]       = .01;
    EXPECT_DOUBLE_EQ(Model_t::from(sim).resistivity, .01);
}

// Hall is a build's (a template flag): an input asking for it on an ideal build fails
TEST(MHDModel, hallMustMatchTheBuild)
{
    auto sim                         = input(Scheme::TVDRK2, core::BoundaryType::Open);
    sim["algo"]["fv_method"]["hall"] = true;
    EXPECT_THROW((void)Model_t::from(sim), std::runtime_error);
}

TEST(MHDModel, needsSchemeAndBoundaries)
{
    EXPECT_THROW((void)Model_t::from(input(Scheme::MHDOff, core::BoundaryType::Open)),
                 std::runtime_error);
    EXPECT_THROW((void)Model_t::from(input(Scheme::Euler, core::BoundaryType::None)),
                 std::runtime_error);
}

TEST(MHDAdvance, uniformStaysUniformBetweenOpenBoundaries)
{
    for (auto const scheme : all_schemes)
    {
        auto const model = Model_t::from(input(scheme, core::BoundaryType::Open));
        ::TestLayout<Layout_t> const layout{cells};
        Patch_t p{layout};
        init(
            p, model.gamma, [](double) { return 1.3; }, [](double) { return .7; },
            [](double) { return .2; }, [](double) { return .5; });
        WalledLevel level{p, model};
        level.fill_state_ghosts(0, 0);
        auto const& u     = p.states[0];
        auto const first  = core::domain_box(layout, u.rho).lower;
        double const rho  = u.rho(first);
        double const Etot = u.Etot(first);
        double const rhoV = u.rhoV[0](first);
        for (std::size_t i = 0; i < 5; ++i)
            advance(model, level, i * dt, (i + 1) * dt);
        damn::core::for_each_point(core::domain_box(layout, u.rho), [&](auto const& ijk) {
            EXPECT_NEAR(u.rho(ijk), rho, 1e-14) << static_cast<int>(scheme);
            EXPECT_NEAR(u.Etot(ijk), Etot, 1e-14) << static_cast<int>(scheme);
            EXPECT_NEAR(u.rhoV[0](ijk), rhoV, 1e-14) << static_cast<int>(scheme);
        });
    }
}

TEST(MHDAdvance, nothingCrossesReflectiveWalls)
{
    for (auto const scheme : all_schemes)
    {
        auto const model = Model_t::from(input(scheme, core::BoundaryType::Reflective));
        ::TestLayout<Layout_t> const layout{cells};
        Patch_t p{layout};
        // a pressure and field pulse at rest, its waves reaching both walls, a flow into them
        auto const bump = [](double const x) { return std::exp(-60. * (x - .4) * (x - .4)); };
        init(
            p, model.gamma, [&](double const x) { return 1. + .5 * bump(x); },
            [&](double const x) { return .6 + bump(x); },
            [](double const x) { return .3 * std::sin(6.283185307179586 * x); },
            [&](double const x) { return .4 + .3 * bump(x); });
        WalledLevel level{p, model};
        level.fill_state_ghosts(0, 0);
        auto const& u     = p.states[0];
        auto const totals = [&]() {
            return std::array{total(layout, u.rho), total(layout, u.Etot), total(layout, u.B[1])};
        };
        auto const before = totals();
        auto const rho0   = u.rho(core::domain_box(layout, u.rho).upper);
        for (std::size_t i = 0; i < 40; ++i)
            advance(model, level, i * dt, (i + 1) * dt);
        auto const after = totals();
        for (std::size_t q = 0; q < before.size(); ++q)
            EXPECT_NEAR(after[q], before[q], 1e-12 * std::abs(before[q]))
                << "scheme " << static_cast<int>(scheme) << " quantity " << q;
        EXPECT_NE(u.rho(core::domain_box(layout, u.rho).upper), rho0); // something moved
        damn::core::for_each_point(core::domain_box(layout, u.rho), [&](auto const& ijk) {
            ASSERT_GT(u.rho(ijk), 0.);
            ASSERT_TRUE(std::isfinite(u.Etot(ijk)));
        });
    }
}

} // namespace plamn::solver

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
