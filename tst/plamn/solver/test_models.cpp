#include "plamn/core/def.hpp"

#include "plamn/core/models/options/mhd_options_def.hpp"
#include "plamn/initializer/data_provider.hpp"
#include "plamn/simulator/simulator_def.hpp"
#include "plamn/solver/models.hpp"

#include "gtest/gtest.h"

#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace plamn::solver
{
namespace
{
    auto constexpr linear  = MHDOpts::ReconstructionType::Linear;
    auto constexpr vanleer = MHDOpts::SlopeLimiterType::VanLeer;
    auto constexpr rusanov = MHDOpts::RiemannSolverType::Rusanov;

    auto constexpr hybrid_only = SimOpts{.dimension = 1, .interp_order = 1};
    auto constexpr mhd_only    = SimOpts{.dimension           = 1,
                                         .interp_order        = 0,
                                         .nbRefinedPart       = 0, // no hybrid: no split
                                         .reconstruction_type = linear,
                                         .slope_limiter_type  = vanleer,
                                         .riemann_solver_type = rusanov};
    auto constexpr both        = SimOpts{.dimension           = 1,
                                         .interp_order        = 1,
                                         .reconstruction_type = linear,
                                         .slope_limiter_type  = vanleer,
                                         .riemann_solver_type = rusanov};

    // what visit gave level ln: 'h' hybrid, 'm' MHD
    template<auto opts>
    char owner(Models<opts>& models, std::size_t const ln)
    {
        return models.visit(ln, [](auto& model) {
            using Model_t = std::remove_cvref_t<decltype(model)>;
            static_assert(is_hybrid_model_v<Model_t> or is_mhd_model_v<Model_t>);
            return is_mhd_model_v<Model_t> ? 'm' : 'h';
        });
    }

    initializer::Dict amr(int const max_mhd_level)
    {
        initializer::Dict sim;
        sim["AMR"]["max_mhd_level"] = max_mhd_level;
        return sim;
    }
} // namespace

TEST(Models, disabledModelsAreEmpty)
{
    static_assert(std::is_empty_v<MHDModel<hybrid_only>>);
    static_assert(std::is_empty_v<HybridModel<mhd_only>>);
    static_assert(!std::is_empty_v<HybridModel<both>> and !std::is_empty_v<MHDModel<both>>);
}

TEST(Models, oneModelOwnsEveryLevel)
{
    Models<hybrid_only> hybrid;
    hybrid.set_levels(initializer::Dict{}, 3);
    Models<mhd_only> mhd;
    mhd.set_levels(initializer::Dict{}, 3); // default: every level MHD
    EXPECT_EQ(mhd.mhd_levels, 3);
    for (std::size_t ln = 0; ln < 3; ++ln)
    {
        EXPECT_EQ(owner(hybrid, ln), 'h');
        EXPECT_FALSE(hybrid.is_mhd(ln));
        EXPECT_EQ(owner(mhd, ln), 'm');
        EXPECT_TRUE(mhd.is_mhd(ln));
    }
}

TEST(Models, mhdLevelsBelowHybridOnes)
{
    Models<both> models;
    models.set_levels(initializer::Dict{}, 3); // default: no MHD level
    EXPECT_EQ(owner(models, 0), 'h');
    models.set_levels(amr(2), 3);
    EXPECT_EQ(owner(models, 0), 'm');
    EXPECT_EQ(owner(models, 1), 'm');
    EXPECT_EQ(owner(models, 2), 'h');
    models.set_levels(amr(3), 3);
    EXPECT_EQ(owner(models, 2), 'm');
}

TEST(Models, levelsOfADisabledModelRefused)
{
    Models<hybrid_only> hybrid;
    EXPECT_THROW(hybrid.set_levels(amr(1), 3), std::runtime_error);
    Models<mhd_only> mhd;
    EXPECT_THROW(mhd.set_levels(amr(2), 3), std::runtime_error);
    Models<both> models;
    EXPECT_THROW(models.set_levels(amr(4), 3), std::runtime_error);
    EXPECT_THROW(models.set_levels(amr(-1), 3), std::runtime_error);
}

} // namespace plamn::solver

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
