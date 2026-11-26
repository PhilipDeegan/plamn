#include "plamn/core/def.hpp"

#include "plamn/core/boundary/boundary.hpp"
#include "plamn/core/boundary/boundary_defs.hpp"
#include "plamn/core/models/quantities/mhd_quantities.hpp"
#include "plamn/core/numerics/boundary_condition/field_boundary_condition.hpp"
#include "plamn/initializer/data_provider.hpp"

#include "tst/plamn/core/numerics/boundary_condition/bc_test_fixtures.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace plamn::core::bc_test
{
namespace
{
    using Scalar = MHDQuantity::Scalar;
    using Vector = MHDQuantity::Vector;
    using FBC    = FieldConditionType;

    double constexpr heatCapacityRatio = 5.0 / 3.0;

    std::vector<Scalar> const mhdScalars{Scalar::rho, Scalar::Etot};
    std::vector<Vector> const mhdVectors{Vector::B, Vector::E, Vector::rhoV};

    void fillInflowData(initializer::Dict& dict)
    {
        dict["density"]  = 1.0;
        dict["pressure"] = 2.0;

        dict["velocity"]["x"] = 3.0;
        dict["velocity"]["y"] = 0.0;
        dict["velocity"]["z"] = 0.0;

        dict["B"]["x"] = 0.75;
        dict["B"]["y"] = 1.0;
        dict["B"]["z"] = 0.0;
    }

    initializer::Dict dictFor(BoundaryType const type)
    {
        initializer::Dict dict;
        dict["type"] = static_cast<int>(type);
        if (type == BoundaryType::SuperMagnetofastInflow)
            fillInflowData(dict);
        return dict;
    }

    auto create(BoundaryType const type)
    {
        return make_boundary<MHDQuantity>(BoundaryLocation::XLower, dictFor(type), mhdScalars,
                                          mhdVectors, heatCapacityRatio);
    }

    struct ExpectedDispatch
    {
        std::map<Scalar, FBC> scalars;
        std::map<Vector, FBC> vectors;
    };

    void checkDispatch(BoundaryType const type, ExpectedDispatch const& expected)
    {
        auto const boundary = create(type);
        auto const check    = [&](auto const& quantities, char const* kind) {
            for (auto const& [qty, expectedType] : quantities)
            {
                auto const* bc = boundary.condition(qty);
                ASSERT_NE(bc, nullptr)
                    << "type " << static_cast<int>(type) << " left " << kind << " quantity "
                    << static_cast<int>(qty) << " with no condition";
                EXPECT_EQ(bc->type, expectedType)
                    << "type " << static_cast<int>(type) << ", " << kind << " quantity "
                    << static_cast<int>(qty) << ": got " << static_cast<int>(bc->type)
                    << ", expected " << static_cast<int>(expectedType);
            }
        };
        check(expected.scalars, "scalar");
        check(expected.vectors, "vector");
    }
} // namespace

TEST(BoundaryFactory, NoneLeavesEveryQuantityUntouched)
{
    checkDispatch(BoundaryType::None,
                  {{{Scalar::rho, FBC::None}, {Scalar::Etot, FBC::None}},
                   {{Vector::B, FBC::None}, {Vector::E, FBC::None}, {Vector::rhoV, FBC::None}}});
}

TEST(BoundaryFactory, Reflective)
{
    checkDispatch(BoundaryType::Reflective,
                  {{{Scalar::rho, FBC::Neumann}, {Scalar::Etot, FBC::Neumann}},
                   {{Vector::B, FBC::DivergenceFreeTransverseNeumann},
                    {Vector::E, FBC::AntiSymmetric},
                    {Vector::rhoV, FBC::Symmetric}}});
}

TEST(BoundaryFactory, Open)
{
    checkDispatch(BoundaryType::Open, {{{Scalar::rho, FBC::Neumann}, {Scalar::Etot, FBC::Neumann}},
                                       {{Vector::B, FBC::DivergenceFreeTransverseNeumann},
                                        {Vector::E, FBC::None},
                                        {Vector::rhoV, FBC::Neumann}}});
}

TEST(BoundaryFactory, SuperMagnetofastInflow)
{
    checkDispatch(BoundaryType::SuperMagnetofastInflow,
                  {{{Scalar::rho, FBC::Dirichlet}, {Scalar::Etot, FBC::Dirichlet}},
                   {{Vector::B, FBC::DivergenceFreeTransverseDirichlet},
                    {Vector::E, FBC::Dirichlet},
                    {Vector::rhoV, FBC::Dirichlet}}});
}

TEST(BoundaryFactory, SuperMagnetofastInflowImposesTotalEnergyFromInflowState)
{
    using Types         = MHDTypes<1>;
    using Layout_t      = Types::Layout_t;
    auto constexpr gw   = Layout_t::ghost_width;
    auto const boundary = create(BoundaryType::SuperMagnetofastInflow);
    auto const* bc      = boundary.condition(Scalar::Etot);
    ASSERT_NE(bc, nullptr);

    double const rho = 1.0, P = 2.0, vx = 3.0, Bx = 0.75, By = 1.0;
    // computed here and in make_boundary: two compile contexts (FMA contraction may differ)
    double const expected
        = P / (heatCapacityRatio - 1.0) + 0.5 * rho * vx * vx + 0.5 * (Bx * Bx + By * By);

    auto const layout = make_layout<Layout_t>(uniform_dx<1>());
    Types::Grid_t Etot{"Etot", layout, Scalar::Etot};
    for (std::uint32_t i = 0; i <= ghost_box(layout, Etot).upper[0]; ++i)
        Etot(i) = expected;
    for (std::uint32_t i = 0; i < gw; ++i)
        Etot(i) = -1.0;

    apply_scalar(*bc, Etot, BoundaryLocation::XLower, ghost_cells(layout, BoundaryLocation::XLower),
                 layout);

    for (std::uint32_t i = 0; i < gw; ++i)
        EXPECT_DOUBLE_EQ(Etot(i), expected);
}

TEST(BoundaryFactory, SuperMagnetofastInflowImposesMotionalElectricField)
{
    using Types         = MHDTypes<1>;
    using Layout_t      = Types::Layout_t;
    auto constexpr gw   = Layout_t::ghost_width;
    auto const boundary = create(BoundaryType::SuperMagnetofastInflow);
    auto const* bc      = boundary.condition(Vector::E);
    ASSERT_NE(bc, nullptr);

    double const vx = 3.0, Bx = 0.75, By = 1.0;
    std::array<double, 3> const expected{0.0, 0.0, Bx * 0.0 - By * vx};

    auto const layout = make_layout<Layout_t>(uniform_dx<1>());
    VecField_t<Types> E{"bc_test_E", layout, Vector::E};
    for (std::size_t c = 0; c < 3; ++c)
        for (std::uint32_t i = 0; i <= ghost_box(layout, E[c]).upper[0]; ++i)
            E[c](i) = (i < gw) ? -1.0 : expected[c];

    apply_vector(*bc, E, BoundaryLocation::XLower, ghost_cells(layout, BoundaryLocation::XLower),
                 layout);

    for (std::size_t c = 0; c < 3; ++c)
        for (std::uint32_t i = 0; i <= gw; ++i)
            EXPECT_DOUBLE_EQ(E[c](i), expected[c]) << "component " << c << " index " << i;
}

} // namespace plamn::core::bc_test

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
