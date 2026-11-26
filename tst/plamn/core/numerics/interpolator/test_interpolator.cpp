#include "plamn/core/numerics/interpolator/interpolator.hpp"

#include "damn/core/geometry/box.hpp"

#include "gtest/gtest.h"

#include <array>
#include <random>
#include <cstddef>

namespace
{
using plamn::core::QtyCentering;

// the protected start index and weights of a particle at cell (AMR) of a patch from lower
template<std::size_t interp>
struct Weights : plamn::core::Interpolator<1, interp>
{
    template<auto centering>
    auto operator()(int const lower, int const cell, double const delta)
    {
        damn::core::Box<int, 1> const box{{lower}, {lower + 100}};
        this->template indexAndWeights_<centering>(box, std::array{cell}, std::array{delta});
        if constexpr (centering == QtyCentering::dual)
            return std::pair{static_cast<int>(this->dual_startIndex_[0]) + lower,
                             this->dual_weights_[0]};
        else
            return std::pair{static_cast<int>(this->primal_startIndex_[0]) + lower,
                             this->primal_weights_[0]};
    }
};

// the weights from delta alone: the same bits at the same delta in any cell of any patch, the
// start index shifted with the cell
template<std::size_t interp, auto centering>
void same_weights_in_every_cell()
{
    Weights<interp> w;
    std::mt19937_64 rng{42};
    std::uniform_real_distribution<double> uniform{0., 1.};
    for (int n = 0; n < 10000; ++n)
    {
        double const delta            = uniform(rng);
        auto const [start0, weights0] = w.template operator()<centering>(0, 3, delta);
        for (int const lower : {0, -7, 13, 1000})
            for (int const cell : {lower + 3, lower + 17, lower + 61, lower + 97})
            {
                auto const [start, weights] = w.template operator()<centering>(lower, cell, delta);
                EXPECT_EQ(start - cell, start0 - 3);
                for (std::size_t i = 0; i < weights.size(); ++i)
                    ASSERT_EQ(weights[i], weights0[i])
                        << "delta " << delta << " lower " << lower << " cell " << cell;
            }
    }
}

TEST(Interpolator, weightsAreTheSameInEveryCell)
{
    same_weights_in_every_cell<1, QtyCentering::primal>();
    same_weights_in_every_cell<1, QtyCentering::dual>();
    same_weights_in_every_cell<2, QtyCentering::primal>();
    same_weights_in_every_cell<2, QtyCentering::dual>();
    same_weights_in_every_cell<3, QtyCentering::primal>();
    same_weights_in_every_cell<3, QtyCentering::dual>();
}

// order 1: primal 1 - delta, delta (exact); dual delta + .5 or delta - .5 from the start node
TEST(Interpolator, order1WeightsAreTheClosedForms)
{
    Weights<1> w;
    for (double const delta : {0., 0.1, 0.25, 0.3, 0.5, 0.7, 0.99})
    {
        auto const [ps, pw] = w.operator()<QtyCentering::primal>(500, 523, delta);
        EXPECT_EQ(ps, 523);
        EXPECT_EQ(pw[1], delta);
        EXPECT_EQ(pw[0], 1. - delta);

        auto const [ds, dw] = w.operator()<QtyCentering::dual>(500, 523, delta);
        EXPECT_EQ(ds, delta < .5 ? 522 : 523);
        double const rel = delta < .5 ? delta + .5 : delta - .5;
        EXPECT_EQ(dw[1], rel);
        EXPECT_EQ(dw[0], 1. - rel);
    }
}

} // namespace

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
