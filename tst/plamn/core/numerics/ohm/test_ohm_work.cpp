// Ohm's ideal term does no work on the electrons (energy conservation of the field solve)

#include "plamn/core/def.hpp"
#include "plamn/plamn_core.hpp"
#include "plamn/simulator/simulator_def.hpp"
#include "plamn/core/numerics/ohm/ohm_transformer.hpp"
#include "plamn/core/data/grid/layout_defs.hpp"

#include "damn/core/geometry/box_spans.hpp"

#include "gtest/gtest.h"

#include "tst/plamn/core/data/grid/test_layout_fixtures.hpp"
#include "tst/plamn/core/data/vecfield/test_vecfield_fixtures.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <utility>

using namespace plamn::core;


// ---------------------------------------------------------------------------------------------
// The ideal term does no work on the electrons: with G = F - J the electron flux on the E nodes,
// E = -G x B / n gives sum_i G_i E_i = 0 over a periodic grid, for any n > 0, F, J, B. The fields
// are periodic noise of the AMR index (period = number of cells), so ghost nodes are periodic
// copies. G is built with the test's own averaging, not Ohm's. The former stencil (Ve and B
// averaged to the E node, then multiplied) runs in the same harness and must fail it.
// ---------------------------------------------------------------------------------------------

template<typename TypeInfo>
struct OhmWorkTest : public ::testing::Test
{
    static constexpr auto dim    = typename TypeInfo::first_type{}();
    static constexpr auto interp = typename TypeInfo::second_type{}();

    using HybridTypes = plamn::core::PLAMN_Types<plamn::SimOpts{dim, interp}>::Hybrid;
    using GridYee     = HybridTypes::Layout_t;
    using Scalar      = HybridQuantity::Scalar;
    using Grid_t      = Grid<damn::core::NdArrayVector<dim>, Scalar>;
    using VecField_t  = UsableVecField<plamn::core::TensorFieldOptions<HybridTypes>{}>;
    using Point_t     = damn::core::Point<std::size_t, dim>;

    // the same number of cells along each direction: the period of the noise
    static constexpr std::uint32_t cells = dim == 1 ? 50 : dim == 2 ? 24 : 12;
    TestLayout<GridYee> layout{0.1, cells};
    std::array<long, dim> const period = mkn::kul::ConstArray<long, dim>(cells);

    Grid_t n{"n", Scalar::rho, alloc_shape(layout, Scalar::rho)};
    Grid_t P{"P", Scalar::P, alloc_shape(layout, Scalar::P)};
    VecField_t F{"F", layout, HybridQuantity::Vector::V};
    VecField_t B{"B", layout, HybridQuantity::Vector::B};
    VecField_t J{"J", layout, HybridQuantity::Vector::J};
    VecField_t E{"E", layout, HybridQuantity::Vector::E};

    // uniform in [-1, 1), periodic in the AMR index (splitmix64 of the wrapped index)
    double noise(std::uint64_t const seed, Point_t const& lcl) const
    {
        std::uint64_t h = seed * 0x9E3779B97F4A7C15ull;
        for (std::size_t d = 0; d < dim; ++d)
        {
            auto const amr     = long(lcl[d]) - long(GridYee::ghost_width);
            auto const wrapped = ((amr % period[d]) + period[d]) % period[d];
            h ^= std::uint64_t(wrapped) + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
            h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ull;
            h = (h ^ (h >> 27)) * 0x94D049BB133111EBull;
            h ^= h >> 31;
        }
        return 2. * double(h >> 11) / double(1ull << 53) - 1.;
    }

    void fill(auto& field, std::uint64_t const seed, auto&& fn)
    {
        damn::core::for_each_point(ghost_box(layout, field),
                                   [&](auto const& lcl) { field(lcl) = fn(noise(seed, lcl)); });
    }

    OhmWorkTest()
    {
        // n log-normal over 0.05 to 20, B and F with nonzero means
        fill(n, 1, [](double u) { return std::exp(3. * u); });
        fill(P, 2, [](double) { return 0.; });
        double const Bmean[3] = {0.3, -0.7, 1.1}, Fmean[3] = {0.2, 0.5, -0.4};
        for (std::uint8_t c = 0; c < 3; ++c)
        {
            fill(B[c], 10 + c, [&](double u) { return Bmean[c] + u; });
            fill(F[c], 20 + c, [&](double u) { return Fmean[c] + u; });
            fill(J[c], 30 + c, [](double u) { return u; });
            fill(E[c], 50 + c, [](double) { return 0.; });
        }
    }

    // G_c on the E_c nodes, the test's own averaging: E_c is dual along c (c < dim) and primal
    //  elsewhere, as are the moments; dual node i sits between primal nodes i and i + 1
    double G(std::uint8_t const c, Point_t const& lcl) const
    {
        auto f = F[c](lcl);
        if (c < dim)
        {
            auto up = lcl;
            ++up[c];
            f = 0.5 * (f + F[c](up));
        }
        return f - J[c](lcl);
    }

    // sum_i G_i E_i over each E component's own periodic nodes (upper primal node excluded),
    //  and the sum of |G_i E_i| as the scale
    std::pair<double, double> work() const
    {
        double sum = 0., scale = 0.;
        for (std::uint8_t c = 0; c < 3; ++c)
            damn::core::for_each_point(domain_box(layout, E[c]), [&](auto const& lcl) {
                for (std::size_t d = 0; d < dim; ++d)
                    if (long(lcl[d]) - long(GridYee::ghost_width) == period[d])
                        return;
                auto const ge = G(c, lcl) * E[c](lcl);
                sum += ge;
                scale += std::abs(ge);
            });
        return {sum, scale};
    }

    // the former ideal term: Ve = (F - Proj(J)) / n on the moment nodes, Ve and B averaged
    //  separately to each E node and multiplied there
    void formerIdealTerm()
    {
        auto const Ve = [&]<std::uint8_t c, auto Jq, auto Vq>() {
            return [&](Point_t const& i) { return (F[c](i) - project<Jq, Vq>(J[c], i)) / n(i); };
        };
        auto const Vex = Ve.template operator()<0, Scalar::Jx, Scalar::Vx>();
        auto const Vey = Ve.template operator()<1, Scalar::Jy, Scalar::Vy>();
        auto const Vez = Ve.template operator()<2, Scalar::Jz, Scalar::Vz>();
        damn::core::for_each_point(domain_box(layout, E[0]), [&](Point_t const& i) {
            E[0](i) = -project<Scalar::Vy, Scalar::Ex>(Vey, i)
                          * project<Scalar::Bz, Scalar::Ex>(B[2], i)
                      + project<Scalar::Vz, Scalar::Ex>(Vez, i)
                            * project<Scalar::By, Scalar::Ex>(B[1], i);
        });
        damn::core::for_each_point(domain_box(layout, E[1]), [&](Point_t const& i) {
            E[1](i) = -project<Scalar::Vz, Scalar::Ey>(Vez, i)
                          * project<Scalar::Bx, Scalar::Ey>(B[0], i)
                      + project<Scalar::Vx, Scalar::Ey>(Vex, i)
                            * project<Scalar::Bz, Scalar::Ey>(B[2], i);
        });
        damn::core::for_each_point(domain_box(layout, E[2]), [&](Point_t const& i) {
            E[2](i) = -project<Scalar::Vx, Scalar::Ez>(Vex, i)
                          * project<Scalar::By, Scalar::Ez>(B[1], i)
                      + project<Scalar::Vy, Scalar::Ez>(Vey, i)
                            * project<Scalar::Bx, Scalar::Ez>(B[0], i);
        });
    }
};

using OhmWorkTupleInfos
    = testing::Types<std::pair<damn::core::DimConst<1>, damn::core::InterpConst<1>>,
                     std::pair<damn::core::DimConst<1>, damn::core::InterpConst<3>>,
                     std::pair<damn::core::DimConst<2>, damn::core::InterpConst<1>>,
                     std::pair<damn::core::DimConst<3>, damn::core::InterpConst<1>>>;

TYPED_TEST_SUITE(OhmWorkTest, OhmWorkTupleInfos);


TYPED_TEST(OhmWorkTest, ThatTheIdealTermDoesNoWorkOnTheElectrons)
{
    plamn::initializer::Dict dict;
    dict["resistivity"]       = 0.0;
    dict["hyper_resistivity"] = 0.0;

    OhmTransformer{OhmInfo::FROM(dict)}(this->layout, this->n, this->F, this->P, this->B, this->J,
                                        this->E);
    auto const [sum, scale] = this->work();
    EXPECT_GT(scale, 0.);
    EXPECT_LT(std::abs(sum), 1e-13 * scale) << "relative work " << sum / scale;
}


TYPED_TEST(OhmWorkTest, ThatTheFormerIdealTermDoesWork)
{
    this->formerIdealTerm();
    auto const [sum, scale] = this->work();
    EXPECT_GT(std::abs(sum), 1e-6 * scale) << "relative work " << sum / scale;
}



int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
