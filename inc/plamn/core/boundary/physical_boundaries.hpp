#ifndef PLAMN_CORE_BOUNDARY_PHYSICAL_BOUNDARIES_HPP
#define PLAMN_CORE_BOUNDARY_PHYSICAL_BOUNDARIES_HPP

// a patch's ghosts past the domain's physical boundaries, filled from its domain values: each
//  region of damn::amr::physical_boundary_regions (a face, an edge or a corner) takes the
//  conditions of its boundary (an edge or corner: one adjacent face's, the manager's choice),
//  each field the condition of its quantity (none: left as is). Faces first: an edge or a corner
//  mirrors along its face's direction into ghosts of the other faces

#include "damn/core/def.hpp"

#include "plamn/core/boundary/boundary_defs.hpp"
#include "plamn/core/data/grid/layout.hpp"
#include "plamn/core/data/grid/layout_defs.hpp"
#include "plamn/core/numerics/boundary_condition/field_boundary_condition.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace damn::amr
{
template<std::size_t dim>
struct PhysicalBoundaryRegion;
} // namespace damn::amr
namespace damn::core
{
template<typename Type, std::size_t dim>
struct Box;
} // namespace damn::core

namespace plamn::core
{
template<typename Quantity>
struct Boundary;
template<typename Quantity>
struct BoundaryManager;

// the face whose conditions a region takes (its sides: -1 lower, +1 upper, 0 within)
template<typename Quantity, std::size_t dim>
NO_DISCARD BoundaryLocation region_location(BoundaryManager<Quantity> const& manager,
                                            std::array<std::int8_t, dim> const& sides)
{
    std::array<std::size_t, 3> dirs{};
    std::array<bool, 3> upper{};
    std::size_t n = 0;
    for (std::size_t d = 0; d < dim; ++d)
        if (sides[d] != 0)
        {
            dirs[n]  = d;
            upper[n] = sides[d] > 0;
            ++n;
        }
    if (n == 1)
        return boundary_location(static_cast<Direction>(dirs[0]),
                                 upper[0] ? Side::Upper : Side::Lower);
    if (n == 2) // xy: 0.., xz: 4.., yz: 8..; then the first side, twice the second
    {
        std::size_t const base = dirs[0] == 0 ? (dirs[1] == 1 ? 0 : 4) : 8;
        return manager.master(static_cast<Codim2BoundaryLocation>(base + upper[0] + 2 * upper[1]));
    }
    if (n == 3)
        return manager.master(
            static_cast<Codim3BoundaryLocation>(upper[0] + 2 * upper[1] + 4 * upper[2]));
    throw std::runtime_error("plamn: a physical boundary region without a side");
}

namespace boundary::detail
{
    template<typename Quantity, typename Field, typename Layout>
    void fill(Boundary<Quantity> const& boundary, Field& field, BoundaryLocation const location,
              damn::core::Box<std::uint32_t, Layout::dimension> const& ghost_cells,
              Layout const& layout)
    {
        auto const* condition = boundary.condition(field.physicalQuantity());
        if (!condition)
            return;
        using Q = std::decay_t<decltype(field.physicalQuantity())>;
        if constexpr (std::is_same_v<Q, typename Quantity::Scalar>)
            apply_scalar(*condition, field, location, ghost_cells, layout);
        else
            apply_vector(*condition, field, location, ghost_cells, layout);
    }
} // namespace boundary::detail

// fields: the patch's scalar and vector fields (each by its physical quantity)
template<typename Quantity, typename Layout, typename... Fields>
void fill_physical_boundaries(
    BoundaryManager<Quantity> const& manager,
    std::vector<damn::amr::PhysicalBoundaryRegion<Layout::dimension>> regions, Layout const& layout,
    Fields&... fields)
{
    std::ranges::stable_sort(regions, {}, [](auto const& r) { return r.codimension(); });
    for (auto const& region : regions)
    {
        auto const location = region_location(manager, region.sides);
        auto const* b       = manager.boundary(location);
        if (!b)
            throw std::runtime_error("plamn: a physical boundary without its conditions");
        auto const ghost_cells = amr_to_local(layout.cells, region.box, Layout::ghost_width);
        (boundary::detail::fill(*b, fields, location, ghost_cells, layout), ...);
    }
}

} // namespace plamn::core

#endif // PLAMN_CORE_BOUNDARY_PHYSICAL_BOUNDARIES_HPP
