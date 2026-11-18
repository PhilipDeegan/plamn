#include "plamn/core/def.hpp"
#ifndef PLAMN_TEST_CORE_DATA_GRID_TEST_LAYOUT_FIXTURES_HPP
#define PLAMN_TEST_CORE_DATA_GRID_TEST_LAYOUT_FIXTURES_HPP

#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "mkn/kul/std/array.hpp"

#include <cstdint>


// a Layout (PatchLayout) made from a cell count and mesh size, origin 0, level 0
//   cells only: mesh size 1/cells, AMR box from 0
//   amrbox only: mesh size .1
//   amrbox and mesh size: a patch anywhere in a level of that mesh size (a refined level's)
template<typename Layout>
class TestLayout : public Layout
{
public:
    using Super               = Layout;
    auto static constexpr dim = Layout::dimension;

    TestLayout() = default;

    TestLayout(Super const& layout)
        : Super{layout}
    {
    }

    TestLayout(double dl, std::uint32_t cells)
        : Super{damn::core::Box<int, dim>{mkn::kul::ConstArray<int, dim>(0),
                                          mkn::kul::ConstArray<int, dim>(cells - 1)},
                mkn::kul::ConstArray<double, dim>(dl)}
    {
    }

    TestLayout(std::uint32_t const cells)
        : TestLayout{1.0 / cells, cells}
    {
    }

    TestLayout(damn::core::Box<int, dim> const& amrbox)
        : Super{amrbox, mkn::kul::ConstArray<double, dim>(.1)}
    {
    }

    TestLayout(damn::core::Box<int, dim> const& amrbox, double const dl)
        : Super{amrbox, mkn::kul::ConstArray<double, dim>(dl)}
    {
    }

    auto static make(std::uint32_t cells) { return TestLayout{cells}; }
    auto static make(double dl, std::uint32_t cells) { return TestLayout{dl, cells}; }
    auto static make(damn::core::Box<int, dim> const& amrbox) { return TestLayout{amrbox}; }

    Super& operator*() { return *this; }
    Super const& operator*() const { return *this; }
};

#endif /*PLAMN_TEST_CORE_DATA_GRID_TEST_LAYOUT_FIXTURES_HPP*/
