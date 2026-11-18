#include "plamn/core/def.hpp"
// the None field condition (field_boundary_condition.hpp): leaves a field's ghosts untouched

#include "plamn/core/boundary/boundary_defs.hpp"
#include "plamn/core/numerics/boundary_condition/field_boundary_condition.hpp"

#include "tst/plamn/core/numerics/boundary_condition/bc_test_fixtures.hpp"

#include "gtest/gtest.h"

#include <cstdint>

namespace plamn::core::bc_test
{
TEST_F(FieldBC1D, NoneDoesNotModifyGhostCells)
{
    FieldCondition const none{.type = FieldConditionType::None};
    apply_scalar(none, field, BoundaryLocation::XLower,
                 ghost_cells(layout, BoundaryLocation::XLower), layout);
    apply_scalar(none, field, BoundaryLocation::XUpper,
                 ghost_cells(layout, BoundaryLocation::XUpper), layout);

    for (std::uint32_t g = 0; g < gw; ++g)
    {
        EXPECT_DOUBLE_EQ(field(g), ghost_sentinel);
        EXPECT_DOUBLE_EQ(field(alloc(0) - 1 - g), ghost_sentinel);
    }
}

} // namespace plamn::core::bc_test

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
