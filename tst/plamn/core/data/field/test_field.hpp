#ifndef PLAMN_TEST_CORE_FIELD_TEST_HPP
#define PLAMN_TEST_CORE_FIELD_TEST_HPP

#include "plamn/core/def.hpp"

#include "plamn/core/data/field/field.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"

#include "tst/plamn/core/data/field/test_field_fixtures.hpp"

#include "gtest/gtest.h" // EXPECT_FLOAT_EQ

#include <cassert>

namespace plamn::core
{

template<std::size_t dim>
struct FieldMock
{
    static auto constexpr dimension  = dim;
    auto constexpr static alloc_mode = damn::AllocatorMode::CPU;

    FieldMock() = default;

    auto& operator()(auto&&...) { return data; }
    auto& operator()(auto&&...) const { return data; }
    auto physicalQuantity() const { return qty; }
    std::string name() const { return "FieldMock"; }

    double data;
    HybridQuantity::Scalar qty = HybridQuantity::Scalar::Ex;
};


template<typename Layout, typename Field0, typename Field1>
void test_fields(Layout const& layout, Field0 const& field0, Field1 const& field1)
{
    EXPECT_EQ(field0.shape(), field1.shape());

    auto const eq = valid_ghost_box(field0.physicalQuantity())
                        ? compare_fields(field0, field1)
                        : compare_field_domains(layout, field0, field1);
    EXPECT_TRUE(eq) << eq.why();
}


template<typename Layout, typename NdArrayImpl>
void test(Layout const& layout,
          Field<Layout::dimension, HybridQuantity::Scalar> const& field0,
          Field<Layout::dimension, HybridQuantity::Scalar> const& field1)
{
    test_fields(layout, field0, field1);
}


template<typename Layout, typename Field, typename T>
void test(Layout const& layout, Field const& field0, std::vector<T> const& fieldV)
{
    EXPECT_EQ(field0.size(), fieldV.size());
    test_fields(layout, field0, make_field_from(field0, fieldV.data()));
}


template<typename Layout, typename Field0, typename Field1>
void test(Layout const& layout, Field0 const& field0, Field1 const& field1)
{
    test_fields(layout, field0, field1);
}


} // namespace plamn::core


#endif /* PLAMN_TEST_CORE_FIELD_TEST_HPP */
