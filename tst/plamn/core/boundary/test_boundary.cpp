#include "plamn/core/def.hpp"
// boundaries (boundary.hpp): each type's conditions per quantity, an inflow's values, the
//  manager from the input's dict, an edge or corner's face

#include "plamn/core/boundary/boundary.hpp"
#include "plamn/core/boundary/boundary_defs.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"
#include "plamn/core/models/quantities/mhd_quantities.hpp"
#include "plamn/core/numerics/boundary_condition/field_boundary_condition.hpp"
#include "plamn/core/numerics/primitive_conservative/primitive_conservative.hpp"
#include "plamn/initializer/data_provider.hpp"

#include "gtest/gtest.h"

#include <array>
#include <stdexcept>
#include <vector>

namespace plamn::core
{
using S = MHDQuantity::Scalar;
using V = MHDQuantity::Vector;
using enum FieldConditionType;
std::vector<S> const scalars{S::rho, S::Etot, S::P};
std::vector<V> const vectors{V::rhoV, V::B, V::E, V::V};

auto type_of(auto const& b, auto const q)
{
    auto const* c = b.condition(q);
    return c ? c->type : None;
}

TEST(Boundary, openAndReflectiveConditions)
{
    auto const open = make_boundary<MHDQuantity>(BoundaryLocation::XLower, BoundaryType::Open,
                                                 scalars, vectors);
    EXPECT_EQ(type_of(open, S::rho), Neumann);
    EXPECT_EQ(type_of(open, S::Etot), Neumann);
    EXPECT_EQ(type_of(open, S::P), None);
    EXPECT_EQ(type_of(open, V::rhoV), Neumann);
    EXPECT_EQ(type_of(open, V::B), DivergenceFreeTransverseNeumann);
    EXPECT_EQ(type_of(open, V::E), None);
    EXPECT_EQ(type_of(open, V::V), None);

    auto const reflective = make_boundary<MHDQuantity>(BoundaryLocation::YUpper,
                                                       BoundaryType::Reflective, scalars, vectors);
    EXPECT_EQ(type_of(reflective, S::rho), Neumann);
    EXPECT_EQ(type_of(reflective, S::P), Neumann); // every scalar
    EXPECT_EQ(type_of(reflective, V::rhoV), Symmetric);
    EXPECT_EQ(type_of(reflective, V::B), DivergenceFreeTransverseNeumann);
    EXPECT_EQ(type_of(reflective, V::E), AntiSymmetric);
    EXPECT_EQ(type_of(reflective, V::V), None);
}

TEST(Boundary, inflowValues)
{
    double const gamma = 5. / 3.;
    InflowState const in{
        .density = 2., .pressure = .5, .velocity = {1.5, .25, -.5}, .B = {.3, -.2, .1}};
    auto const b
        = make_boundary<MHDQuantity>(BoundaryLocation::XLower, BoundaryType::SuperMagnetofastInflow,
                                     scalars, vectors, in, gamma);
    auto const& [rho, P, v, B] = in;
    EXPECT_EQ(b.condition(S::rho)->value[0], rho);
    EXPECT_EQ(b.condition(S::rho)->extrapolation, DirichletExtrapolation::Constant);
    // computed here and in make_boundary: two compile contexts (FMA contraction may differ)
    EXPECT_DOUBLE_EQ(b.condition(S::Etot)->value[0],
                     eos_p_to_etot(gamma, rho, v[0], v[1], v[2], B[0], B[1], B[2], P));
    EXPECT_EQ(type_of(b, S::P), None);
    for (std::size_t c = 0; c < 3; ++c)
    {
        EXPECT_EQ(b.condition(V::rhoV)->value[c], rho * v[c]);
        EXPECT_EQ(b.condition(V::B)->value[c], B[c]);
    }
    EXPECT_EQ(type_of(b, V::B), DivergenceFreeTransverseDirichlet);
    auto const& E = b.condition(V::E)->value; // -v x B
    EXPECT_DOUBLE_EQ(E[0], v[2] * B[1] - v[1] * B[2]);
    EXPECT_DOUBLE_EQ(E[1], v[0] * B[2] - v[2] * B[0]);
    EXPECT_DOUBLE_EQ(E[2], v[1] * B[0] - v[0] * B[1]);

    EXPECT_THROW((void)make_boundary<MHDQuantity>(BoundaryLocation::XLower,
                                                  BoundaryType::SuperMagnetofastInflow, scalars,
                                                  vectors, in, 1.),
                 std::runtime_error); // gamma > 1
    EXPECT_THROW((void)make_boundary<MHDQuantity>(BoundaryLocation::XLower,
                                                  BoundaryType::SuperMagnetofastInflow, scalars,
                                                  vectors, std::nullopt, gamma),
                 std::runtime_error); // no state
}

TEST(Boundary, physicalBoundariesAreMHDsOnly)
{
    using H = HybridQuantity;
    EXPECT_THROW((void)make_boundary<H>(BoundaryLocation::XLower, BoundaryType::Open,
                                        {H::Scalar::rho}, {H::Vector::B}),
                 std::runtime_error);
    auto const none = make_boundary<H>(BoundaryLocation::XLower, BoundaryType::None,
                                       {H::Scalar::rho}, {H::Vector::B});
    EXPECT_EQ(none.condition(H::Vector::B)->type, None);
}

initializer::Dict domain_boundaries()
{
    initializer::Dict d;
    d["xlower"]["type"]          = static_cast<int>(BoundaryType::SuperMagnetofastInflow);
    d["xlower"]["density"]       = 1.;
    d["xlower"]["pressure"]      = 1.;
    d["xlower"]["velocity"]["x"] = 2.;
    d["xlower"]["velocity"]["y"] = 0.;
    d["xlower"]["velocity"]["z"] = 0.;
    d["xlower"]["B"]["x"]        = .5;
    d["xlower"]["B"]["y"]        = .1;
    d["xlower"]["B"]["z"]        = 0.;
    d["xupper"]["type"]          = static_cast<int>(BoundaryType::Open);
    d["ylower"]["type"]          = static_cast<int>(BoundaryType::Reflective);
    d["yupper"]["type"]          = static_cast<int>(BoundaryType::Reflective);
    return d;
}

TEST(BoundaryManager, fromTheInput)
{
    BoundaryManager<MHDQuantity> const m{domain_boundaries(), scalars, vectors, 5. / 3.};
    ASSERT_NE(m.boundary(BoundaryLocation::XLower), nullptr);
    EXPECT_EQ(m.boundary(BoundaryLocation::XLower)->type, BoundaryType::SuperMagnetofastInflow);
    EXPECT_EQ(m.boundary(BoundaryLocation::XLower)->condition(V::rhoV)->value[0], 2.);
    EXPECT_EQ(m.boundary(BoundaryLocation::XUpper)->type, BoundaryType::Open);
    EXPECT_EQ(m.boundary(BoundaryLocation::ZLower), nullptr);

    // a corner: by direction, the y face; by type, the highest (open > inflow > reflective)
    EXPECT_EQ(m.master(Codim2BoundaryLocation::XUpper_YLower), BoundaryLocation::YLower);
    EXPECT_EQ(m.master(BoundaryLocation::XUpper), BoundaryLocation::XUpper);
    auto by_type     = m;
    by_type.priority = BoundaryManager<MHDQuantity>::Priority::ByBoundaryType;
    EXPECT_EQ(by_type.master(Codim2BoundaryLocation::XUpper_YLower), BoundaryLocation::XUpper);
    EXPECT_EQ(by_type.master(Codim2BoundaryLocation::XLower_YUpper), BoundaryLocation::XLower);

    BoundaryManager<MHDQuantity> const periodic{initializer::Dict{}, scalars, vectors};
    EXPECT_EQ(periodic.boundary(BoundaryLocation::XLower), nullptr);
}

TEST(BoundaryManager, physicalDirectionsDeclareTheirFaces)
{
    initializer::Dict grid;
    grid["periodicities"]["x"] = false;
    grid["periodicities"]["y"] = true;
    EXPECT_THROW(validate_physical_boundaries<2>(grid), std::runtime_error);
    grid["domain_boundaries"]["xlower"]["type"] = static_cast<int>(BoundaryType::Open);
    EXPECT_THROW(validate_physical_boundaries<2>(grid), std::runtime_error); // xupper missing
    grid["domain_boundaries"]["xupper"]["type"] = static_cast<int>(BoundaryType::Open);
    EXPECT_NO_THROW(validate_physical_boundaries<2>(grid));
}

} // namespace plamn::core

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
