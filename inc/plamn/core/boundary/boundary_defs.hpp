#ifndef PLAMN_CORE_BOUNDARY_BOUNDARY_DEFS_HPP
#define PLAMN_CORE_BOUNDARY_BOUNDARY_DEFS_HPP

// physical boundaries of the domain (PHARE's core/boundary/boundary_defs.hpp): their kinds, as
//  pyphare writes them (simulation/grid/domain_boundaries/<location>/type), and their locations:
//  codimension 1 (a face in 3D, an edge in 2D, an end in 1D), 2 (an edge in 3D, a corner in
//  2D) and 3 (a corner in 3D), an edge or corner adjacent to the faces it joins

#include "damn/core/def.hpp"

#include "plamn/core/data/grid/layout_defs.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace plamn::core
{
enum class BoundaryType { None, Reflective, SuperMagnetofastInflow, Open };

enum class BoundaryLocation : std::uint8_t {
    XLower = 0,
    XUpper = 1,
    YLower = 2,
    YUpper = 3,
    ZLower = 4,
    ZUpper = 5
};

MKN_KUL_FN_SIG_RET constexpr Side side_of(BoundaryLocation const location)
{
    return static_cast<std::uint8_t>(location) % 2 == 0 ? Side::Lower : Side::Upper;
}

MKN_KUL_FN_SIG_RET constexpr Direction direction_of(BoundaryLocation const location)
{
    return static_cast<Direction>(static_cast<std::uint8_t>(location) / 2);
}

MKN_KUL_FN_SIG_RET constexpr BoundaryLocation boundary_location(Direction const d, Side const side)
{
    return static_cast<BoundaryLocation>(2 * static_cast<std::uint8_t>(d)
                                         + (side == Side::Lower ? 0 : 1));
}

enum class Codim2BoundaryLocation : std::uint8_t {
    XLower_YLower = 0,
    XUpper_YLower = 1,
    XLower_YUpper = 2,
    XUpper_YUpper = 3,
    XLower_ZLower = 4,
    XUpper_ZLower = 5,
    XLower_ZUpper = 6,
    XUpper_ZUpper = 7,
    YLower_ZLower = 8,
    YUpper_ZLower = 9,
    YLower_ZUpper = 10,
    YUpper_ZUpper = 11
};

enum class Codim3BoundaryLocation : std::uint8_t {
    XLower_YLower_ZLower = 0,
    XUpper_YLower_ZLower = 1,
    XLower_YUpper_ZLower = 2,
    XUpper_YUpper_ZLower = 3,
    XLower_YLower_ZUpper = 4,
    XUpper_YLower_ZUpper = 5,
    XLower_YUpper_ZUpper = 6,
    XUpper_YUpper_ZUpper = 7
};

// the faces an edge or a corner joins
MKN_KUL_FN_SIG_RET constexpr std::array<BoundaryLocation, 2>
adjacent_locations(Codim2BoundaryLocation const location)
{
    using enum BoundaryLocation;
    constexpr std::array<std::array<BoundaryLocation, 2>, 12> adjacent{{
        {XLower, YLower},
        {XUpper, YLower},
        {XLower, YUpper},
        {XUpper, YUpper},
        {XLower, ZLower},
        {XUpper, ZLower},
        {XLower, ZUpper},
        {XUpper, ZUpper},
        {YLower, ZLower},
        {YUpper, ZLower},
        {YLower, ZUpper},
        {YUpper, ZUpper},
    }};
    auto const i = static_cast<std::size_t>(location);
    if (i >= adjacent.size())
        throw std::runtime_error("plamn: invalid codimension 2 boundary location");
    return adjacent[i];
}

MKN_KUL_FN_SIG_RET constexpr std::array<BoundaryLocation, 3>
adjacent_locations(Codim3BoundaryLocation const location)
{
    using enum BoundaryLocation;
    constexpr std::array<std::array<BoundaryLocation, 3>, 8> adjacent{{
        {XLower, YLower, ZLower},
        {XUpper, YLower, ZLower},
        {XLower, YUpper, ZLower},
        {XUpper, YUpper, ZLower},
        {XLower, YLower, ZUpper},
        {XUpper, YLower, ZUpper},
        {XLower, YUpper, ZUpper},
        {XUpper, YUpper, ZUpper},
    }};
    auto const i = static_cast<std::size_t>(location);
    if (i >= adjacent.size())
        throw std::runtime_error("plamn: invalid codimension 3 boundary location");
    return adjacent[i];
}

// "xlower", ..., "zupper" (pyphare's domain_boundaries keys)
MKN_KUL_FN_SIG_RET inline BoundaryLocation boundary_location(std::string const& name)
{
    constexpr std::array<char const*, 6> names{"xlower", "xupper", "ylower",
                                               "yupper", "zlower", "zupper"};
    for (std::size_t i = 0; i < names.size(); ++i)
        if (name == names[i])
            return static_cast<BoundaryLocation>(i);
    throw std::runtime_error("plamn: unknown boundary location " + name);
}

template<std::size_t codim>
    requires(codim > 0 and codim < 4)
using CodimNBoundaryLocation = std::tuple_element_t<
    codim - 1, std::tuple<BoundaryLocation, Codim2BoundaryLocation, Codim3BoundaryLocation>>;

} // namespace plamn::core

#endif // PLAMN_CORE_BOUNDARY_BOUNDARY_DEFS_HPP
