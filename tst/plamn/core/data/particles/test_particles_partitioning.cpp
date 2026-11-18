#include "plamn/core/def.hpp"

#include "plamn/plamn_core.hpp"
#include "mkn/kul/time.hpp"

#include "plamn/core/data/particles/partitioning/particles_partitioning.hpp"

#include "tst/plamn/core/data/vecfield/test_vecfield_fixtures.hpp"
#include "tst/plamn/core/data/electromag/test_electromag_fixtures.hpp"
#include "tst/plamn/core/data/tensorfield/test_tensorfield_fixtures.hpp"
#include "tst/plamn/core/data/ion_population/test_ion_population_fixtures.hpp"

#include "tst/damn/core/geometry/test_box_fixtures.hpp"

#include "gtest/gtest.h"

using namespace plamn::core;

std::size_t constexpr static dim = 3; // only 3d
using Box_t                      = damn::core::Box<int, dim>;
using Particles_t                = Particles<ParticlesOptions{dim, LayoutMode::AoS}>;


struct StaticParticlesDataFixture
{
    std::size_t constexpr static ghost_cells = 1;
    std::size_t constexpr static ppc         = 125;
    std::size_t constexpr static box_size    = 50;

    static StaticParticlesDataFixture instance()
    {
        static StaticParticlesDataFixture i;
        return i;
    }

    auto b() const { return true; }
    auto& middle() const { return std::get<0>(box_tuple); }
    auto& neighbours() const { return std::get<1>(box_tuple); }

    StaticParticlesDataFixture()
        : box_tuple{damn::core::get_some_boxes(box_size)}
    {
        add_particles(particles, damn::core::grow(middle(), ghost_cells), ppc);
    }

    std::tuple<Box_t, std::vector<Box_t>> const box_tuple;
    Particles_t particles;
};
static auto is_initialized = StaticParticlesDataFixture::instance().b();

template<typename Particles, std::size_t ghost_cells = 2>
struct ParticlesData
{
    auto static constexpr dim = Particles::dimension;
    ParticlesData(damn::core::Box<int, dim> const& box_)
        : box{box_}
    {
    }
    damn::core::Box<int, dim> box;
    damn::core::Box<int, dim> ghost_box = damn::core::grow(box, ghost_cells);
    Particles domainParticles;
};


template<typename Particles_, std::size_t impl_ = 0>
struct ParticlesDataFixture
{
    std::size_t constexpr static ghost_cells = StaticParticlesDataFixture::ghost_cells;
    std::size_t constexpr static ppc         = StaticParticlesDataFixture::ppc;
    std::size_t constexpr static box_size    = StaticParticlesDataFixture::box_size;
    std::size_t constexpr static impl        = impl_;

    using Particles_t     = Particles_;
    using ParticlesData_t = ParticlesData<Particles_t, ghost_cells>;

    ParticlesDataFixture()
        : middle{StaticParticlesDataFixture::instance().middle()}
        , neighbours{mkn::kul::generate_from(
              [&](auto& box) { return std::make_unique<ParticlesData_t>(box); },
              StaticParticlesDataFixture::instance().neighbours())}
    {
        auto& particles = middle.domainParticles;
        for (auto const& particle : StaticParticlesDataFixture::instance().particles)
            particles.push_back(particle);
    }

    ParticlesData_t middle;
    std::vector<std::unique_ptr<ParticlesData_t>> neighbours;
};


template<typename ParticlesData>
struct ParticlesDataTest : public ::testing::Test, public ParticlesData
{
};


using ParticlesDatas = testing::Types< //
    ParticlesDataFixture<Particles<ParticlesOptions{dim, LayoutMode::AoS}>>

    >;

TYPED_TEST_SUITE(ParticlesDataTest, ParticlesDatas, );

TYPED_TEST(ParticlesDataTest, works)
{
    using Test            = TypeParam;
    using Particles_t = typename TestFixture::Particles_t;
    using Partitioner     = ParticlesPartitioner<Particles_t::alloc_mode, Particles_t>;
    auto& middle          = this->middle;
    auto& particles       = middle.domainParticles;

    for (std::size_t i = 0; i < particles.size(); ++i)
    {
        EXPECT_TRUE(isIn(particles.iCell(i), middle.ghost_box));
    }

    auto before_size = particles.size();
    auto start       = mkn::kul::Now::MICROS();
    auto iterator    = Partitioner{particles}(middle.box);
    auto end         = mkn::kul::Now::MICROS();

    EXPECT_EQ(before_size, particles.size());
    EXPECT_EQ(iterator.size(), std::pow(Test::box_size, dim) * Test::ppc);

    for (std::size_t i = 0; i < static_cast<std::size_t>(iterator.size()); ++i)
        EXPECT_TRUE(isIn(particles.iCell(i), middle.box));

    for (std::size_t i = iterator.size(); i < particles.size(); ++i)
    {
        auto const& iCell = particles.iCell(i);
        EXPECT_TRUE(isIn(iCell, middle.ghost_box));
        EXPECT_TRUE(not isIn(iCell, middle.box));
    }
}


TYPED_TEST(ParticlesDataTest, worksVector)
{
    using Test            = TypeParam;
    using Particles_t = typename TestFixture::Particles_t;
    using Partitioner     = ParticlesPartitioner<Particles_t::alloc_mode, Particles_t>;
    auto& middle          = this->middle;
    auto& particles       = middle.domainParticles;

    for (std::size_t i = 0; i < particles.size(); ++i)
    {
        EXPECT_TRUE(isIn(particles.iCell(i), middle.ghost_box));
    }

    auto before_size = particles.size();
    auto start       = mkn::kul::Now::MICROS();
    auto iterators   = Partitioner{particles}(std::vector{middle.box});
    auto end         = mkn::kul::Now::MICROS();

    auto iterator = iterators.back();

    EXPECT_EQ(before_size, particles.size());
    EXPECT_EQ(iterator.size(), std::pow(Test::box_size, dim) * Test::ppc);

    for (std::size_t i = 0; i < static_cast<std::size_t>(iterator.size()); ++i)
        EXPECT_TRUE(isIn(particles.iCell(i), middle.box));

    for (std::size_t i = iterator.size(); i < particles.size(); ++i)
    {
        auto const& iCell = particles.iCell(i);
        EXPECT_TRUE(isIn(iCell, middle.ghost_box));
        EXPECT_TRUE(not isIn(iCell, middle.box));
    }
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
