// aos_transpose.hpp: 8 particle structures to columns and back by the AVX-512 transpose (or per
//  particle without it), against the per particle copy: every dimension, every block size n, the
//  slots past n (to_columns: weight 0; from_columns: particle n untouched)

#include "plamn/core/data/particles/aos_transpose.hpp"
#include "plamn/core/data/particles/particle.hpp"

#include "gtest/gtest.h"

#include <cstddef>
#include <cstring>
#include <random>

namespace plamn::core
{
template<std::size_t dim>
void check_both_ways()
{
    std::mt19937 rng{7}; // NOLINT(bugprone-random-generator-seed): reproducible
    std::uniform_real_distribution<double> u(-1, 1);
    std::uniform_int_distribution<int> cell(-50, 5000);
    for (std::size_t n = 1; n <= 8; ++n)
        for (int rep = 0; rep < 50; ++rep)
        {
            Particle<dim> ps[9], ref[9];
            for (auto& p : ps)
            {
                p.weight() = u(rng) + 2;
                for (std::size_t k = 0; k < dim; ++k)
                {
                    p.iCell()[k] = cell(rng);
                    p.delta()[k] = u(rng);
                }
                for (std::size_t k = 0; k < 3; ++k)
                    p.v()[k] = u(rng);
            }
            alignas(64) double w[8], d[dim][8], v[3][8], w2[8], d2[dim][8], v2[3][8];
            alignas(64) int c[dim][8], c2[dim][8];
            aos_transpose::Cols<dim> A{w, {}, {}, {v[0], v[1], v[2]}};
            aos_transpose::Cols<dim> B{w2, {}, {}, {v2[0], v2[1], v2[2]}};
            for (std::size_t k = 0; k < dim; ++k)
            {
                A.c[k] = c[k];
                A.d[k] = d[k];
                B.c[k] = c2[k];
                B.d[k] = d2[k];
            }
            aos_transpose::to_columns<dim>(ps, n, A);
            aos_transpose::to_columns_each<dim>(ps, n, B);
            for (std::size_t l = 0; l < 8; ++l)
            {
                EXPECT_EQ(w[l], w2[l]);
                if (l >= n)
                    continue;
                for (std::size_t k = 0; k < dim; ++k)
                {
                    EXPECT_EQ(c[k][l], c2[k][l]);
                    EXPECT_EQ(d[k][l], d2[k][l]);
                }
                for (std::size_t k = 0; k < 3; ++k)
                    EXPECT_EQ(v[k][l], v2[k][l]);
            }

            for (std::size_t l = 0; l < 8; ++l) // the columns changed, written back
            {
                w[l] += 1;
                for (std::size_t k = 0; k < dim; ++k)
                {
                    c[k][l] += 3;
                    d[k][l] *= .5;
                }
                for (std::size_t k = 0; k < 3; ++k)
                    v[k][l] -= .25;
            }
            std::memcpy(static_cast<void*>(ref), ps, sizeof ps);
            aos_transpose::from_columns<dim>(ps, n, A);
            aos_transpose::from_columns_each<dim>(ref, n, A);
            for (std::size_t l = 0; l < 9; ++l)
            {
                EXPECT_EQ(ps[l].weight(), ref[l].weight());
                for (std::size_t k = 0; k < dim; ++k)
                {
                    EXPECT_EQ(ps[l].iCell()[k], ref[l].iCell()[k]);
                    EXPECT_EQ(ps[l].delta()[k], ref[l].delta()[k]);
                }
                for (std::size_t k = 0; k < 3; ++k)
                    EXPECT_EQ(ps[l].v()[k], ref[l].v()[k]);
            }
        }
}

TEST(AoSTranspose, oneD)
{
    check_both_ways<1>();
}
TEST(AoSTranspose, twoD)
{
    check_both_ways<2>();
}
TEST(AoSTranspose, threeD)
{
    check_both_ways<3>();
}

} // namespace plamn::core


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
