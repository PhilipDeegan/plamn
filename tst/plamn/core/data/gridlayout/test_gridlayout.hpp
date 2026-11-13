#include "plamn/core/def.hpp"
#ifndef TESTS_CORE_DATA_GRIDLAYOUT_TEST_GRIDLAYOUT_HPP
#define TESTS_CORE_DATA_GRIDLAYOUT_TEST_GRIDLAYOUT_HPP

#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"

#include <cstdint>
#include <type_traits>
#include "plamn/core/data/grid/gridlayout.hpp"
#include "damn/core/math/array.hpp"


template<typename GridLayout>
class TestGridLayout : public GridLayout
{ // to expose a default constructor
public:
    using Super               = GridLayout;
    auto static constexpr dim = GridLayout::dimension;

    TestGridLayout() = default;

    TestGridLayout(double dl, std::uint32_t cells)
        : GridLayout{damn::core::ConstArray<double, dim>(dl),
                     damn::core::ConstArray<std::uint32_t, dim>(cells),
                     damn::core::Point<double, dim>{damn::core::ConstArray<double, dim>(0)}}
    {
    }

    TestGridLayout(std::uint32_t const cells)
        : GridLayout{damn::core::ConstArray<double, dim>(1.0 / cells),
                     damn::core::ConstArray<std::uint32_t, dim>(cells),
                     damn::core::Point<double, dim>{damn::core::ConstArray<double, dim>(0)}}
    {
    }


    TestGridLayout(damn::core::Box<int, dim> const& amrbox)
        : GridLayout{damn::core::ConstArray<double, dim>(.1),
                     amrbox.shape().template toArray<std::uint32_t>(),
                     damn::core::Point<double, dim>{damn::core::ConstArray<double, dim>(0)},
                     amrbox}
    {
    }

    template<typename... Args>
    TestGridLayout(Args&&... args)
        requires(std::is_constructible_v<GridLayout, Args...>)
        : GridLayout{args...}
    {
    }

    auto static make(std::uint32_t cells) { return TestGridLayout{cells}; }
    auto static make(double dl, std::uint32_t cells) { return TestGridLayout{dl, cells}; }
    auto static make(damn::core::Box<int, dim> const& amrbox) { return TestGridLayout{amrbox}; }

    Super& operator*() { return *this; }
    Super const& operator*() const { return *this; }

    operator Super&() { return *this; }
    operator Super const&() const { return *this; }
};

#endif /*TESTS_CORE_DATA_GRIDLAYOUT_TEST_GRIDLAYOUT_HPP*/
