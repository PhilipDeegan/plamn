#ifndef PLAMN_CORE_DATA_GRID_PATCH_LAYOUT_HPP
#define PLAMN_CORE_DATA_GRID_PATCH_LAYOUT_HPP

// PatchLayout: the data describing one patch's mesh, and free functions over it
//   cells:  the patch's cell box in the level (AMR) index space
//   dx:     mesh size
//   origin: physical position of the patch's first cell's lower corner
//   level:  AMR level number
// ghost width and dimension are compile time, from the options
//
// quantity centerings come from the model's quantity tables (hybrid::quantities,
// mhd::quantities): pass a quantity, or anything with physicalQuantity()

#include "plamn/core/def.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "plamn/core/data/field/field.hpp"
#include "plamn/core/data/grid/layout.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"
#include "plamn/core/models/quantities/mhd_quantities.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <concepts>

namespace plamn::core
{
template<auto options_>
struct PatchLayout
{
    auto static constexpr options                   = options_;
    std::size_t static constexpr dimension          = options.dimension;
    std::uint32_t static constexpr ghost_width      = options.field_ghost_width;
    using AMRBox_t                                  = damn::core::Box<int, dimension>;

    AMRBox_t cells{};
    std::array<double, dimension> dx{};
    damn::core::Point<double, dimension> origin{};
    int level = 0;
};

template<typename T>
concept IsPatchLayout = requires { T::options; }
                        && std::derived_from<std::decay_t<T>, PatchLayout<T::options>>;


// ---------------------------------------------------------------- centering

template<std::size_t dim, typename Scalar>
    requires requires(Scalar s) { quantity(s); }
NO_DISCARD constexpr std::array<QtyCentering, dim> centering(Scalar const qty) DAMN_ALL
{
    auto const c = quantity(qty).centering;
    std::array<QtyCentering, dim> out{};
    for (std::size_t i = 0; i < dim; ++i)
        out[i] = c[i];
    return out;
}

template<std::size_t dim, typename Components>
NO_DISCARD constexpr auto components_centering(Components const& qties) DAMN_ALL
{
    std::array<std::array<QtyCentering, dim>, std::tuple_size_v<Components>> out{};
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = centering<dim>(qties[i]);
    return out;
}

template<std::size_t dim>
NO_DISCARD constexpr auto centering(HybridQuantity::Vector const qty)
{
    return components_centering<dim>(HybridQuantity::componentsQuantities(qty));
}

template<std::size_t dim>
NO_DISCARD constexpr auto centering(HybridQuantity::Tensor const qty)
{
    return components_centering<dim>(HybridQuantity::componentsQuantities(qty));
}

template<std::size_t dim>
NO_DISCARD constexpr auto centering(MHDQuantity::Vector const qty)
{
    return components_centering<dim>(MHDQuantity::componentsQuantities(qty));
}

// a quantity, or anything carrying one
template<std::size_t dim, typename HasQuantity>
    requires HasPhysicalQuantity<HasQuantity>
NO_DISCARD constexpr auto centering(HasQuantity const& hasQuantity) DAMN_ALL
{
    return centering<dim>(hasQuantity.physicalQuantity());
}

template<auto o>
NO_DISCARD constexpr auto centering(PatchLayout<o> const&, auto const& qty) DAMN_ALL
{
    return centering<PatchLayout<o>::dimension>(qty);
}


// ---------------------------------------------------------------- sizes and boxes

template<auto o>
NO_DISCARD constexpr auto cell_count(PatchLayout<o> const& L) DAMN_ALL
{
    return L.cells.shape().template toArray<std::uint32_t>();
}

// what to allocate for qty: ghost nodes included
template<auto o>
NO_DISCARD constexpr auto alloc_shape(PatchLayout<o> const& L, auto const& qty) DAMN_ALL
{
    return alloc_shape(L.cells, centering(L, qty), PatchLayout<o>::ghost_width);
}

template<auto o>
NO_DISCARD constexpr auto physical_nodes(PatchLayout<o> const& L, auto const& qty) DAMN_ALL
{
    return physical_nodes(L.cells, centering(L, qty));
}

// local index space, ghost inclusive, first ghost node at 0
template<auto o>
NO_DISCARD constexpr auto domain_box(PatchLayout<o> const& L, auto const& qty) DAMN_ALL
{
    return local_domain_box(L.cells, centering(L, qty), PatchLayout<o>::ghost_width);
}

template<auto o>
NO_DISCARD constexpr auto ghost_box(PatchLayout<o> const& L, auto const& qty) DAMN_ALL
{
    return local_ghost_box(L.cells, centering(L, qty), PatchLayout<o>::ghost_width);
}

// AMR index space
template<auto o>
NO_DISCARD constexpr auto amr_box(PatchLayout<o> const& L, auto const& qty) DAMN_ALL
{
    return amr_box(L.cells, centering(L, qty));
}

template<auto o>
NO_DISCARD constexpr auto amr_ghost_box(PatchLayout<o> const& L, auto const& qty) DAMN_ALL
{
    return amr_ghost_box(L.cells, centering(L, qty), PatchLayout<o>::ghost_width);
}


// ---------------------------------------------------------------- index spaces

template<auto o, typename T>
NO_DISCARD constexpr auto to_local(PatchLayout<o> const& L,
                                   damn::core::Point<T, PatchLayout<o>::dimension> const& amr)
    DAMN_ALL
{
    return amr_to_local(L.cells, amr, PatchLayout<o>::ghost_width);
}

template<auto o, typename T>
NO_DISCARD constexpr auto to_local(PatchLayout<o> const& L,
                                   damn::core::Box<T, PatchLayout<o>::dimension> const& amr)
{
    return amr_to_local(L.cells, amr, PatchLayout<o>::ghost_width);
}

template<auto o, typename T>
NO_DISCARD constexpr auto to_amr(PatchLayout<o> const& L,
                                 damn::core::Point<T, PatchLayout<o>::dimension> const& local)
{
    return local_to_amr(L.cells, local, PatchLayout<o>::ghost_width);
}

template<auto o, typename T>
NO_DISCARD constexpr auto to_amr(PatchLayout<o> const& L,
                                 damn::core::Box<T, PatchLayout<o>::dimension> const& local)
{
    return local_to_amr(L.cells, local, PatchLayout<o>::ghost_width);
}

// fn(amr_index, local_index) for each point of an AMR box
template<auto o>
void for_each_amr_local(PatchLayout<o> const& L, auto const& amr, auto&& fn)
{
    damn::core::for_each_point_zip(amr, to_local(L, amr), fn);
}


// ---------------------------------------------------------------- physical space

// position of an AMR index of qty
template<auto o>
NO_DISCARD constexpr auto coordinate(PatchLayout<o> const& L, auto const& qty,
                                     damn::core::Point<int, PatchLayout<o>::dimension> const& amr)
    DAMN_ALL
{
    return coordinate(L.dx, amr, centering(L, qty));
}

template<auto o>
NO_DISCARD constexpr auto
cell_center(PatchLayout<o> const& L, damn::core::Point<int, PatchLayout<o>::dimension> const& amr)
    DAMN_ALL
{
    return cell_center(L.dx, amr);
}

template<auto o>
NO_DISCARD constexpr double cell_volume(PatchLayout<o> const& L) DAMN_ALL
{
    return cell_volume(L.dx);
}

template<auto o>
NO_DISCARD constexpr auto inverse_mesh_size(PatchLayout<o> const& L) DAMN_ALL
{
    std::array<double, PatchLayout<o>::dimension> inv{};
    for (std::size_t i = 0; i < inv.size(); ++i)
        inv[i] = 1. / L.dx[i];
    return inv;
}


// ---------------------------------------------------------------- derived layouts

// the layout of a sub box of the patch's cells, same level and mesh size
template<auto o>
NO_DISCARD PatchLayout<o> sub_layout(PatchLayout<o> const& L,
                                     damn::core::Box<int, PatchLayout<o>::dimension> const& box)
{
    return {box, L.dx, damn::core::Point<double, PatchLayout<o>::dimension>{L.dx} * box.lower,
            L.level};
}

} // namespace plamn::core

#endif // PLAMN_CORE_DATA_GRID_PATCH_LAYOUT_HPP
