// the flux on the E nodes is the transpose of the E gather: for any E,
//  sum_p w_p v_p . E(x_p) (m2p) = sum_nodes fluxE . E
// so the ion current Ohm reads does exactly the work E does on the particles. The moment flux
//  averaged to the E nodes fails the same identity

#include "plamn/core/def.hpp"
#include "plamn/plamn_core.hpp"
#include "plamn/simulator/simulator_def.hpp"
#include "plamn/core/numerics/interpolator/interpolator.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"

#include "gtest/gtest.h"

#include "tst/plamn/core/data/grid/test_layout_fixtures.hpp"
#include "tst/plamn/core/data/vecfield/test_vecfield_fixtures.hpp"
#include "damn/core/def/constants.hpp"
#include "damn/core/geometry/box_spans.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

using namespace plamn::core;

namespace
{
template<std::size_t dim>
struct TestParticle
{
    std::array<int, dim> iCell_{};
    std::array<double, dim> delta_{};
    double weight_ = 0;
    std::array<double, 3> v_{};

    auto& iCell() const { return iCell_; }
    auto& delta() const { return delta_; }
    auto weight() const { return weight_; }
    auto& v() const { return v_; }
};

template<typename VecField>
struct TestEM
{
    VecField& e;
    VecField& b;
    auto& E() const { return e(); } // the components, as m2p destructures them
    auto& B() const { return b(); }
};
} // namespace


template<typename TypeInfo>
struct FluxETest : public ::testing::Test
{
    static constexpr auto dim    = typename TypeInfo::first_type{}();
    static constexpr auto interp = typename TypeInfo::second_type{}();

    using HybridTypes = PLAMN_Types<plamn::SimOpts{dim, interp}>::Hybrid;
    using Layout_t    = HybridTypes::Layout_t;
    using Scalar      = HybridQuantity::Scalar;
    using Grid_t      = Grid<damn::core::NdArrayVector<dim>, Scalar>;
    using VecField_t  = UsableVecField<TensorFieldOptions<HybridTypes>{}>;

    static constexpr std::uint32_t cells = dim == 1 ? 20 : dim == 2 ? 12 : 8;
    TestLayout<Layout_t> layout{0.1, cells};

    Grid_t density{"density", Scalar::rho, alloc_shape(layout, Scalar::rho)};
    VecField_t flux{"flux", layout, HybridQuantity::Vector::V};
    VecField_t fluxE{"fluxE", layout, HybridQuantity::Vector::E};
    VecField_t E{"E", layout, HybridQuantity::Vector::E};
    VecField_t B{"B", layout, HybridQuantity::Vector::B};
    std::vector<TestParticle<dim>> particles;

    FluxETest()
    {
        std::mt19937_64 gen{17};
        std::uniform_real_distribution<double> u{0., 1.}, s{-1., 1.};
        auto const zero_all = [](auto& field) { field.zero(); };
        zero_all(density);
        for (std::uint8_t c = 0; c < 3; ++c)
        {
            flux[c].zero();
            fluxE[c].zero();
            B[c].zero();
            damn::core::for_each_point(ghost_box(layout, E[c]),
                                       [&](auto const& lcl) { E[c](lcl) = s(gen); });
        }
        // interior cells: every stencil (orders 1 to 3) inside the ghost box
        std::uniform_int_distribution<int> cell{2, int(cells) - 3};
        particles.resize(200);
        for (auto& p : particles)
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                p.iCell_[d] = cell(gen);
                p.delta_[d] = u(gen);
            }
            p.weight_ = 0.5 + u(gen);
            p.v_      = {s(gen), s(gen), s(gen)};
        }
    }

    // sum_p w_p v_p . E(x_p), with the production gather
    double particles_work()
    {
        Interpolator<dim, interp> interpolator;
        TestEM<VecField_t> const em{E, B};
        double work = 0;
        for (auto const& p : particles)
        {
            auto const [pE, pB] = interpolator.m2p(p, em, layout);
            for (std::size_t c = 0; c < 3; ++c)
                work += p.weight() * p.v()[c] * pE[c];
        }
        return work;
    }

    // sum over each E component's nodes of values . E, and the scale (sum of |.|)
    std::pair<double, double> grid_work(auto const& values)
    {
        double work = 0, scale = 0;
        for (std::uint8_t c = 0; c < 3; ++c)
            damn::core::for_each_point(ghost_box(layout, E[c]), [&](auto const& lcl) {
                work += values(c, lcl) * E[c](lcl);
                scale += std::abs(values(c, lcl) * E[c](lcl));
            });
        return {work, scale};
    }

    // the moment flux of the particles, with the production deposit
    void deposit_moments()
    {
        Interpolator<dim, interp> interpolator;
        for (auto const& p : particles)
            interpolator.particleToMesh(p, density, flux, layout);
    }
};

using FluxETupleInfos
    = testing::Types<std::pair<damn::core::DimConst<1>, damn::core::InterpConst<1>>,
                     std::pair<damn::core::DimConst<1>, damn::core::InterpConst<3>>,
                     std::pair<damn::core::DimConst<2>, damn::core::InterpConst<1>>,
                     std::pair<damn::core::DimConst<2>, damn::core::InterpConst<2>>,
                     std::pair<damn::core::DimConst<3>, damn::core::InterpConst<1>>>;

TYPED_TEST_SUITE(FluxETest, FluxETupleInfos);


TYPED_TEST(FluxETest, ThatTheEFluxIsTheTransposeOfTheGather)
{
    using Layout_t = TestFixture::Layout_t;
    using Scalar   = TestFixture::Scalar;
    Interpolator<TestFixture::dim, TestFixture::interp> interpolator;
    for (auto const& p : this->particles)
        interpolator.fluxE_to_mesh(p, this->fluxE, this->layout);

    // the E components on the moment nodes are the moment flux (complete_fluxE)
    this->deposit_moments();
    auto const on_moments = [&]<auto quantity>(std::uint8_t const c) {
        if constexpr (on_moment_nodes<Layout_t::dimension, quantity>())
            damn::core::for_each_point(ghost_box(this->layout, this->E[c]), [&](auto const& lcl) {
                this->fluxE[c](lcl) = this->flux[c](lcl);
            });
    };
    on_moments.template operator()<Scalar::Ex>(0);
    on_moments.template operator()<Scalar::Ey>(1);
    on_moments.template operator()<Scalar::Ez>(2);

    auto const by_particles = this->particles_work();
    auto const [by_grid, scale]
        = this->grid_work([&](auto const c, auto const& lcl) { return this->fluxE[c](lcl); });
    EXPECT_GT(scale, 0.);
    EXPECT_NEAR(by_particles, by_grid, 1e-12 * scale);
}


TYPED_TEST(FluxETest, ThatTheMomentFluxAveragedToTheENodesIsNot)
{
    using Scalar = TestFixture::Scalar;
    this->deposit_moments();
    auto const averaged = [&](auto const c, auto const& lcl) {
        auto const& F = this->flux[c];
        if (c == 0)
            return project<Scalar::Vx, Scalar::Ex>(F, lcl);
        if (c == 1)
            return project<Scalar::Vy, Scalar::Ey>(F, lcl);
        return project<Scalar::Vz, Scalar::Ez>(F, lcl);
    };
    // only where the average's stencil is inside the moment flux's nodes: the E domain boxes
    double work = 0, scale = 0;
    for (std::uint8_t c = 0; c < 3; ++c)
        damn::core::for_each_point(domain_box(this->layout, this->E[c]), [&](auto const& lcl) {
            work += averaged(c, lcl) * this->E[c](lcl);
            scale += std::abs(averaged(c, lcl) * this->E[c](lcl));
        });
    EXPECT_GT(std::abs(this->particles_work() - work), 1e-6 * scale);
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
