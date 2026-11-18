#ifndef PLAMN_CORE_DATA_GRID_LAYOUT_HPP
#define PLAMN_CORE_DATA_GRID_LAYOUT_HPP

// stateless mesh/layout functions over (cells box, centering, ghost width, mesh size)
//   cells: the patch's cell box in the level (AMR) index space
//   local index space: ghost inclusive, starts at 0 on the first ghost node
//   gw: field ghost width, the same for primal and dual

#include "mkn/kul/math.hpp"
#include "mkn/kul/float.hpp"
#include "mkn/kul/int.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/def/damn_config.hpp"
#include "mkn/kul/std/array.hpp"

#include "plamn/core/def.hpp"
#include "plamn/core/data/grid/layout_defs.hpp"
#include "plamn/core/models/quantities/quantity.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace plamn::core
{
NO_DISCARD constexpr QtyCentering flip(QtyCentering const c) DAMN_ALL
{
    return c == QtyCentering::primal ? QtyCentering::dual : QtyCentering::primal;
}

template<std::size_t dim>
NO_DISCARD constexpr auto primal_centering() DAMN_ALL
{
    return mkn::kul::ConstArray<QtyCentering, dim>(QtyCentering::primal);
}

template<std::size_t dim>
NO_DISCARD constexpr auto dual_centering() DAMN_ALL
{
    return mkn::kul::ConstArray<QtyCentering, dim>(QtyCentering::dual);
}

NO_DISCARD constexpr std::uint32_t dual_offset(QtyCentering const c) DAMN_ALL
{
    return c == QtyCentering::dual ? 1 : 0;
}


// per direction, n = number of cells
NO_DISCARD constexpr std::uint32_t physical_start(std::uint32_t const gw) DAMN_ALL
{
    return gw;
}

NO_DISCARD constexpr std::uint32_t physical_end(QtyCentering const c, std::uint32_t const n,
                                                std::uint32_t const gw) DAMN_ALL
{
    return gw + n - dual_offset(c);
}

NO_DISCARD constexpr std::uint32_t ghost_end(QtyCentering const c, std::uint32_t const n,
                                             std::uint32_t const gw) DAMN_ALL
{
    return physical_end(c, n, gw) + gw;
}

// index of the node of centering c next to (right of) / previous to (left of) a node of the
// opposite centering
NO_DISCARD constexpr std::uint32_t next_index(QtyCentering const c, std::uint32_t const i) DAMN_ALL
{
    return c == QtyCentering::primal ? i + 1 : i;
}

NO_DISCARD constexpr std::uint32_t prev_index(QtyCentering const c, std::uint32_t const i) DAMN_ALL
{
    return c == QtyCentering::primal ? i : i - 1;
}


template<std::size_t dim>
NO_DISCARD constexpr auto physical_nodes(damn::core::Box<int, dim> const& cells,
                                         std::array<QtyCentering, dim> const& centering) DAMN_ALL
{
    auto const shape = cells.shape();
    std::array<std::uint32_t, dim> nodes;
    for (std::size_t i = 0; i < dim; ++i)
        nodes[i] = shape[i] + 1 - dual_offset(centering[i]);
    return nodes;
}

// total number of nodes, ghosts included: what to allocate
template<std::size_t dim>
NO_DISCARD constexpr auto alloc_shape(damn::core::Box<int, dim> const& cells,
                                      std::array<QtyCentering, dim> const& centering,
                                      std::uint32_t const gw) DAMN_ALL
{
    auto nodes = physical_nodes(cells, centering);
    for (auto& n : nodes)
        n += 2 * gw;
    return nodes;
}


template<std::size_t dim>
NO_DISCARD constexpr auto local_domain_box(damn::core::Box<int, dim> const& cells,
                                           std::array<QtyCentering, dim> const& centering,
                                           std::uint32_t const gw)
{
    auto const shape = cells.shape();
    damn::core::Box<std::uint32_t, dim> box;
    for (std::size_t i = 0; i < dim; ++i)
    {
        box.lower[i] = physical_start(gw);
        box.upper[i] = physical_end(centering[i], shape[i], gw);
    }
    return box;
}

template<std::size_t dim>
NO_DISCARD constexpr auto local_ghost_box(damn::core::Box<int, dim> const& cells,
                                          std::array<QtyCentering, dim> const& centering,
                                          std::uint32_t const gw)
{
    auto const shape = cells.shape();
    damn::core::Box<std::uint32_t, dim> box;
    for (std::size_t i = 0; i < dim; ++i)
    {
        box.lower[i] = 0;
        box.upper[i] = ghost_end(centering[i], shape[i], gw);
    }
    return box;
}


template<std::size_t dim>
NO_DISCARD constexpr auto qty_box(damn::core::Box<int, dim> const& cells,
                                  std::array<QtyCentering, dim> const& centering)
{
    return toCenteredBox(cells, centering);
}

template<std::size_t dim>
NO_DISCARD constexpr auto qty_ghost_box(damn::core::Box<int, dim> const& cells,
                                        std::array<QtyCentering, dim> const& centering,
                                        std::uint32_t const gw)
{
    return damn::core::grow(qty_box(cells, centering),
                            mkn::kul::ConstArray<std::uint32_t, dim>(gw));
}


// cell (or node, same shift for both centerings) index conversions
template<typename T, std::size_t dim>
NO_DISCARD constexpr auto amr_to_local(damn::core::Box<int, dim> const& cells, damn::core::Point<T, dim> const& amr,
                                       std::uint32_t const gw) DAMN_ALL
{
    static_assert(std::is_integral_v<T>);
    damn::core::Point<std::uint32_t, dim> local;
    for (std::size_t i = 0; i < dim; ++i)
    {
        int const l = static_cast<int>(amr[i]) - (cells.lower[i] - static_cast<int>(gw));
        assert(l >= 0);
        local[i] = static_cast<std::uint32_t>(l);
    }
    return local;
}

template<typename T, std::size_t dim>
NO_DISCARD constexpr auto amr_to_local(damn::core::Box<int, dim> const& cells, damn::core::Box<T, dim> const& amr,
                                       std::uint32_t const gw)
{
    return damn::core::Box<std::uint32_t, dim>{amr_to_local(cells, amr.lower, gw),
                                   amr_to_local(cells, amr.upper, gw)};
}

template<typename T, std::size_t dim>
NO_DISCARD constexpr auto local_to_amr(damn::core::Box<int, dim> const& cells, damn::core::Point<T, dim> const& local,
                                       std::uint32_t const gw)
{
    static_assert(std::is_integral_v<T>);
    damn::core::Point<int, dim> amr;
    for (std::size_t i = 0; i < dim; ++i)
        amr[i] = static_cast<int>(local[i]) + (cells.lower[i] - static_cast<int>(gw));
    return amr;
}

template<typename T, std::size_t dim>
NO_DISCARD constexpr auto local_to_amr(damn::core::Box<int, dim> const& cells, damn::core::Box<T, dim> const& local,
                                       std::uint32_t const gw)
{
    return damn::core::Box<int, dim>{local_to_amr(cells, local.lower, gw),
                         local_to_amr(cells, local.upper, gw)};
}


// physical position of an AMR index, dual nodes sit half a cell up
template<std::size_t dim>
NO_DISCARD constexpr auto coordinate(std::array<double, dim> const& dx,
                                     damn::core::Point<int, dim> const& amr,
                                     std::array<QtyCentering, dim> const& centering) DAMN_ALL
{
    damn::core::Point<double, dim> x;
    for (std::size_t i = 0; i < dim; ++i)
        x[i] = (static_cast<double>(amr[i]) + 0.5 * dual_offset(centering[i])) * dx[i];
    return x;
}

template<std::size_t dim>
NO_DISCARD constexpr auto cell_center(std::array<double, dim> const& dx,
                                      damn::core::Point<int, dim> const& amr) DAMN_ALL
{
    return coordinate(dx, amr, mkn::kul::ConstArray<QtyCentering, dim>(QtyCentering::dual));
}

template<std::size_t dim>
NO_DISCARD constexpr double cell_volume(std::array<double, dim> const& dx) DAMN_ALL
{
    return mkn::kul::math::product(dx, 1.0);
}


// index mirrored across the lower/upper physical boundary, local index space
NO_DISCARD constexpr std::uint32_t boundary_mirrored(QtyCentering const c, Side const side,
                                                     std::uint32_t const n,
                                                     std::uint32_t const gw,
                                                     std::uint32_t const index)
{
    auto const s = static_cast<std::int32_t>(side);
    auto const i = static_cast<std::int32_t>(index);
    auto const b = static_cast<std::int32_t>(side == Side::Lower ? physical_start(gw)
                                                                 : physical_end(c, n, gw));
    return static_cast<std::uint32_t>(c == QtyCentering::primal ? i - 2 * (i - b)
                                                                : i - 2 * (i - b) + s);
}


// point shifted by +1/-1 along direction d
template<auto d, typename T, std::size_t dim>
NO_DISCARD constexpr auto next(damn::core::Point<T, dim> p) DAMN_ALL
{
    ++p[static_cast<std::size_t>(d)];
    return p;
}

template<auto d, typename T, std::size_t dim>
NO_DISCARD constexpr auto previous(damn::core::Point<T, dim> p) DAMN_ALL
{
    --p[static_cast<std::size_t>(d)];
    return p;
}

} // namespace plamn::core

#endif // PLAMN_CORE_DATA_GRID_LAYOUT_HPP
