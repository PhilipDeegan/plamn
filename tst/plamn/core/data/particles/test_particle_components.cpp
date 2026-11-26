// compared as sets: SoAVXChunked keeps the particles by cell

#include "plamn/core/data/particles/particle_components.hpp"

#include "tst/plamn/core/data/particles/test_particles.hpp"

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace plamn::core
{
template<std::size_t dim_, LayoutMode layout_>
struct Case
{
    auto static constexpr dim    = dim_;
    auto static constexpr layout = layout_;
};

template<typename C>
struct ParticleComponentsTest : public ::testing::Test
{
    using Particles_t = Particles<ParticlesOptions{C::dim, C::layout}>;
};

using Cases
    = ::testing::Types<Case<1, LayoutMode::AoS>, Case<2, LayoutMode::AoS>, Case<3, LayoutMode::AoS>,
                       Case<1, LayoutMode::AoSMapped>, Case<2, LayoutMode::AoSMapped>,
                       Case<3, LayoutMode::AoSMapped>, Case<1, LayoutMode::SoA>,
                       Case<2, LayoutMode::SoA>, Case<3, LayoutMode::SoA>,
                       Case<1, LayoutMode::SoAVX>, Case<2, LayoutMode::SoAVX>,
                       Case<3, LayoutMode::SoAVX>, Case<1, LayoutMode::SoAVXChunked>,
                       Case<2, LayoutMode::SoAVXChunked>, Case<3, LayoutMode::SoAVXChunked>>;
TYPED_TEST_SUITE(ParticleComponentsTest, Cases, );

namespace
{
    // add_particles, through AoSMapped for SoAVX (it has no emplace_back)
    template<typename Particles_t>
    void fill(Particles_t& particles, auto const& box, std::size_t const ppc)
    {
        if constexpr (Particles_t::layout_mode == LayoutMode::SoAVX)
        {
            Particles<ParticlesOptions{Particles_t::dimension, LayoutMode::AoSMapped}> staging;
            add_particles(staging, box, ppc);
            append_particles<ParticleType::Domain>(staging, particles);
        }
        else
            add_particles(particles, box, ppc);
    }

    // every particle of the views, as comparable tuples, sorted
    template<std::size_t dim>
    auto rows(auto const& c)
    {
        using Row = std::tuple<double, std::array<int, dim>, std::array<double, dim>,
                               std::array<double, 3>>;
        std::vector<Row> out(c.weight.size());
        for (std::size_t k = 0; k < out.size(); ++k)
        {
            auto& [w, cell, delta, v] = out[k];
            w                         = c.weight[k];
            for (std::size_t i = 0; i < dim; ++i)
            {
                cell[i]  = c.iCell[i][k];
                delta[i] = c.delta[i][k];
            }
            for (std::size_t i = 0; i < 3; ++i)
                v[i] = c.v[i][k];
        }
        std::sort(out.begin(), out.end());
        return out;
    }
} // namespace

TYPED_TEST(ParticleComponentsTest, readsAndWritesTheArraysOwnStorage)
{
    using Particles_t  = TestFixture::Particles_t;
    auto constexpr dim = TypeParam::dim;
    using Reference_t  = Particles<ParticlesOptions{dim, LayoutMode::AoS}>;

    damn::core::Box<int, dim> const box{mkn::kul::ConstArray<int, dim>(0),
                                        mkn::kul::ConstArray<int, dim>(3)};
    auto particles = [&] { // SoAVXChunked's offsets cover a box
        if constexpr (std::is_constructible_v<Particles_t, decltype(box)>)
            return Particles_t{box};
        else
            return Particles_t{};
    }();
    EXPECT_EQ(components(particles).weight.size(), 0u);
    fill(particles, box, 3);
    Reference_t reference;
    add_particles(reference, box, 3);
    ASSERT_GT(particles.size(), 0u);
    ASSERT_EQ(particles.size(), reference.size());

    auto const c = components(particles);
    ASSERT_EQ(c.weight.size(), particles.size());
    EXPECT_EQ(rows<dim>(c), rows<dim>(components(reference)));

    for (std::size_t k = 0; k < c.v[1].size(); ++k)
        c.v[1][k] = 42. + k;
    c.weight[0]            = -1.;
    Particles_t const copy = particles; // the writes are in the array's storage
    auto const cc          = components(copy);
    static_assert(std::is_const_v<std::remove_reference_t<decltype(cc.weight[0])>>);
    for (std::size_t k = 0; k < cc.v[1].size(); ++k)
        EXPECT_EQ(cc.v[1][k], 42. + k);
    EXPECT_EQ(cc.weight[0], -1.);

    std::size_t n = 0;
    cc.for_each([&](auto const& name, auto const& s) {
        EXPECT_FALSE(name.empty());
        EXPECT_EQ(s.size(), particles.size());
        ++n;
    });
    EXPECT_EQ(n, 1 + 2 * dim + 3);
}

} // namespace plamn::core


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
