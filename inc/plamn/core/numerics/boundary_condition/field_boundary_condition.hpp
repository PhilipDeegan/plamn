#ifndef PLAMN_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_BOUNDARY_CONDITION_HPP
#define PLAMN_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_BOUNDARY_CONDITION_HPP

// a field's ghosts past a physical boundary, filled from its domain values:
//  doc/core/numerics/boundary_condition/field_boundary_condition.md

#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/point.hpp"

#include "plamn/core/def.hpp"
#include "plamn/core/boundary/boundary_defs.hpp"
#include "plamn/core/data/grid/layout.hpp"
#include "plamn/core/data/grid/layout_defs.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace damn::core
{
template<typename Type, std::size_t dim>
struct Box;
} // namespace damn::core

namespace plamn::core
{
enum class QtyCentering : std::uint8_t;

enum class FieldConditionType : std::uint8_t {
    None,
    Dirichlet,
    AntiSymmetric,
    Symmetric,
    Neumann,
    DivergenceFreeTransverseNeumann,
    DivergenceFreeTransverseDirichlet
};

enum class DirichletExtrapolation : std::uint8_t { Constant = 0, Linear = 1 };

// plain data: many boundaries carry one per quantity
struct FieldCondition
{
    FieldConditionType type              = FieldConditionType::None;
    std::array<double, 3> value          = {0., 0., 0.}; // Dirichlet: one per component
    DirichletExtrapolation extrapolation = DirichletExtrapolation::Linear;
};

namespace boundary::detail
{
    // the index mirroring `index` across the boundary, along d (local index space)
    template<typename Layout, std::size_t dim>
    MKN_KUL_FN_SIG_RET auto mirrored(Layout const& layout, Direction const d, Side const side,
                                     QtyCentering const c,
                                     damn::core::Point<std::uint32_t, dim> index)
    {
        auto const i = static_cast<std::size_t>(d);
        auto const n
            = static_cast<std::uint32_t>(layout.cells.upper[i] - layout.cells.lower[i] + 1);
        index[i] = boundary_mirrored(c, side, n, Layout::ghost_width, index[i]);
        return index;
    }

    // the field's nodes over the ghost cells, at its centering
    template<std::size_t dim>
    MKN_KUL_FN_SIG_RET auto field_box(damn::core::Box<std::uint32_t, dim> const& ghost_cells,
                                      auto const& field)
    {
        return toCenteredBox(ghost_cells, centering<dim>(field.physicalQuantity()));
    }
} // namespace boundary::detail


namespace boundary::detail
{
    // B's normal component past the boundary, given its tangential ones there: face by face away
    //  from the domain (the boundary face is the domain's), each ghost cell's discrete divergence
    //  zero: Bn(next face) = Bn(face) - dn * sum over tangential directions of dBt/dt
    template<typename VecField, typename Layout>
    void
    divergence_free_normal(VecField& B, BoundaryLocation const location,
                           damn::core::Box<std::uint32_t, Layout::dimension> const& ghost_cells,
                           Layout const& layout)
    {
        auto constexpr dim = Layout::dimension;
        auto const n       = static_cast<std::size_t>(direction_of(location));
        auto const side    = side_of(location);
        auto& Bn           = B[n];
        auto const one     = [&](auto const& cell) {
            double div = 0.;
            for (std::size_t t = 0; t < dim; ++t)
                if (t != n)
                {
                    auto up = cell;
                    ++up[t];
                    div += (B[t](up) - B[t](cell)) / layout.dx[t];
                }
            auto outer = cell; // the cell's face away from the domain, and the inner one
            auto inner = cell;
            if (side == Side::Upper)
                ++outer[n];
            else
                ++inner[n];
            Bn(outer) = Bn(inner) + (side == Side::Upper ? -1. : 1.) * layout.dx[n] * div;
        };
        if (side == Side::Upper) // nearest the domain first: each face reads one already set
            damn::core::for_each_point(ghost_cells, one);
        else
            damn::core::for_each_point_reverse(ghost_cells, one);
    }
} // namespace boundary::detail


// zero gradient: ghosts take their mirrors' values
template<typename Field, typename Layout>
void neumann(Field& field, BoundaryLocation const location,
             damn::core::Box<std::uint32_t, Layout::dimension> const& ghost_cells,
             Layout const& layout)
{
    auto constexpr dim = Layout::dimension;
    auto const d       = direction_of(location);
    auto const side    = side_of(location);
    auto const c       = centering<dim>(field.physicalQuantity())[static_cast<std::size_t>(d)];
    damn::core::for_each_point(
        boundary::detail::field_box(ghost_cells, field), [&](auto const& index) {
            field(index) = field(boundary::detail::mirrored(layout, d, side, c, index));
        });
}

// value v at the boundary: v on a boundary node or extrapolating constantly, else 2 v - mirror
template<typename Field, typename Layout>
void dirichlet(Field& field, BoundaryLocation const location,
               damn::core::Box<std::uint32_t, Layout::dimension> const& ghost_cells,
               Layout const& layout, double const value,
               DirichletExtrapolation const extrapolation = DirichletExtrapolation::Linear)
{
    auto constexpr dim = Layout::dimension;
    auto const d       = direction_of(location);
    auto const i       = static_cast<std::size_t>(d);
    auto const side    = side_of(location);
    auto const c       = centering<dim>(field.physicalQuantity())[i];
    damn::core::for_each_point(
        boundary::detail::field_box(ghost_cells, field), [&](auto const& index) {
            auto const mirror = boundary::detail::mirrored(layout, d, side, c, index);
            field(index)
                = (extrapolation == DirichletExtrapolation::Constant or mirror[i] == index[i])
                      ? value
                      : 2. * value - field(mirror);
        });
}

template<typename Field, typename Layout>
void apply_scalar(FieldCondition const& condition, Field& field, BoundaryLocation const location,
                  damn::core::Box<std::uint32_t, Layout::dimension> const& ghost_cells,
                  Layout const& layout)
{
    using enum FieldConditionType;
    switch (condition.type)
    {
        case None: return;
        case Neumann:
        case Symmetric: return neumann(field, location, ghost_cells, layout);
        case Dirichlet:
            return dirichlet(field, location, ghost_cells, layout, condition.value[0],
                             condition.extrapolation);
        case AntiSymmetric: return dirichlet(field, location, ghost_cells, layout, 0.);
        default: throw std::runtime_error("plamn: a vector field condition on a scalar field");
    }
}

// a vector field's condition, component by component (c: the component, normal if it is the
//  boundary's direction)
template<typename VecField, typename Layout>
void apply_vector(FieldCondition const& condition, VecField& vecfield,
                  BoundaryLocation const location,
                  damn::core::Box<std::uint32_t, Layout::dimension> const& ghost_cells,
                  Layout const& layout)
{
    using enum FieldConditionType;
    auto const normal = static_cast<std::size_t>(direction_of(location));
    for (std::size_t c = 0; c < 3; ++c)
    {
        auto& field = vecfield[c];
        switch (condition.type)
        {
            case None: break;
            case Neumann: neumann(field, location, ghost_cells, layout); break;
            case Dirichlet:
                dirichlet(field, location, ghost_cells, layout, condition.value[c],
                          condition.extrapolation);
                break;
            case Symmetric:
                if (c == normal)
                    dirichlet(field, location, ghost_cells, layout, 0.);
                else
                    neumann(field, location, ghost_cells, layout);
                break;
            case AntiSymmetric:
                if (c == normal)
                    neumann(field, location, ghost_cells, layout);
                else
                    dirichlet(field, location, ghost_cells, layout, 0.);
                break;
            case DivergenceFreeTransverseNeumann:
                if (c != normal)
                    neumann(field, location, ghost_cells, layout);
                break;
            case DivergenceFreeTransverseDirichlet:
                if (c != normal)
                    dirichlet(field, location, ghost_cells, layout, condition.value[c],
                              condition.extrapolation);
                break;
        }
    }
    if (condition.type == DivergenceFreeTransverseNeumann
        or condition.type == DivergenceFreeTransverseDirichlet)
        boundary::detail::divergence_free_normal(vecfield, location, ghost_cells, layout);
}

} // namespace plamn::core

#endif // PLAMN_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_BOUNDARY_CONDITION_HPP
