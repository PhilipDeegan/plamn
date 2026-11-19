// the MHD reconstructions (PHARE's): the left and right states at a face from the cells around it,
//  on 1D data given point by point: exact for constant (all) and linear (all but constant) data,
//  converging at their order on a smooth profile

#include "plamn/core/numerics/reconstructions/constant.hpp"
#include "plamn/core/numerics/reconstructions/linear.hpp"
#include "plamn/core/numerics/reconstructions/mp5.hpp"
#include "plamn/core/numerics/reconstructions/weno3.hpp"
#include "plamn/core/numerics/reconstructions/wenoz.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>

namespace plamn::core
{
namespace
{
    // cell i's value: f's mean over the cell [i h, (i + 1) h] (f' its antiderivative)
    struct Cells
    {
        double h;
        double (*F)(double); // an antiderivative of the profile
        double operator()(std::array<int, 1> const& i) const
        {
            return (F((i[0] + 1) * h) - F(i[0] * h)) / h;
        }
    };

    double constant_F(double const x)
    {
        return 2. * x;
    }
    double linear_F(double const x)
    {
        return 2. * x + .75 * x * x;
    } // f = 2 + 1.5 x
    double smooth_F(double const x)
    {
        return -std::cos(2 * std::numbers::pi * x) / (2 * std::numbers::pi);
    }
    double smooth_f(double const x)
    {
        return std::sin(2 * std::numbers::pi * x);
    }

    // the largest error of both states against f at the faces of cells 0.25..0.75 (any order)
    template<typename R>
    double face_error(double const h, double (*F)(double), double (*f)(double))
    {
        Cells const cells{h, F};
        double err = 0;
        for (int i = static_cast<int>(.25 / h); i <= static_cast<int>(.75 / h); ++i)
        {
            auto const [l, r] = R::template reconstruct<0>(cells, std::array<int, 1>{i});
            err               = std::max({err, std::abs(l - f(i * h)), std::abs(r - f(i * h))});
        }
        return err;
    }

    template<typename R>
    struct ReconstructionTest : ::testing::Test
    {
    };
    using All = ::testing::Types<ConstantReconstruction<>, LinearReconstruction<VanLeerLimiter>,
                                 WENO3Reconstruction<>, WENOZReconstruction<>, MP5Reconstruction<>>;
    TYPED_TEST_SUITE(ReconstructionTest, All, );
} // namespace

TYPED_TEST(ReconstructionTest, constantDataExact)
{
    EXPECT_LT((face_error<TypeParam>(.05, constant_F, [](double) { return 2.; })), 1e-14);
}

TYPED_TEST(ReconstructionTest, linearDataExact)
{
    if constexpr (std::is_same_v<TypeParam, ConstantReconstruction<>>)
        GTEST_SKIP() << "first order";
    else
        EXPECT_LT((face_error<TypeParam>(.05, linear_F, [](double x) { return 2. + 1.5 * x; })),
                  1e-13);
}

// the error ratio when h halves: 2^order (smooth data: the limiters at rest)
TEST(Reconstructions, convergeAtTheirOrder)
{
    auto const ratio = [](auto r) {
        using R = decltype(r);
        return face_error<R>(1. / 64, smooth_F, smooth_f)
               / face_error<R>(1. / 128, smooth_F, smooth_f);
    };
    EXPECT_GT(ratio(ConstantReconstruction<>{}), 1.8);
    EXPECT_GT(ratio(WENO3Reconstruction<>{}), 3.);
    EXPECT_GT(ratio(WENOZReconstruction<>{}), 24.);
    EXPECT_GT(ratio(MP5Reconstruction<>{}), 24.);
}

} // namespace plamn::core

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
