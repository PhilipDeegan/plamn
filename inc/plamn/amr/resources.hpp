#ifndef PLAMN_AMR_RESOURCES_HPP
#define PLAMN_AMR_RESOURCES_HPP

// plamn data as damn::amr resources, for damn's schedules
//
//   field_resource<gw>(field):     BoxData, values on the field's nodes (its centering), gw
//                                  ghost nodes each side; a C ordered field (not tiled) also
//                                  gives damn::hdf5's checkpoints its buffer (box_span)
//   particle_resource<gw>(array):  ParticleData, particles in cells, gw ghost cells; a layout
//                                  with components() also gives checkpoints its columns
//
// both are small handles over the data, made on the fly in a for_each_resource

#include "damn/core/geometry/box.hpp"
#include "damn/core/memory/box_span.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/math/array.hpp"

#include "plamn/core/data/particles/particle_components.hpp"
#include "plamn/core/def.hpp"
#include "plamn/core/data/grid/layout.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"

#include <cstddef>
#include <stdexcept>
#include <cstdint>
#include <utility>

namespace plamn::core
{
template<size_t dim>
struct Particle;
} // namespace plamn::core

namespace plamn::amr
{
template<typename Field, std::uint32_t gw>
struct FieldResource
{
    auto static constexpr dimension = Field::dimension;
    Field* field;
};

template<std::uint32_t gw, typename Field>
NO_DISCARD auto field_resource(Field& field)
{
    return FieldResource<Field, gw>{std::addressof(field)}; // Grid overloads operator&
}

template<typename Field, std::uint32_t gw>
NO_DISCARD auto domain_box(FieldResource<Field, gw> const& r,
                           damn::core::Box<int, Field::dimension> const& cells)
{
    return core::qty_box(cells, core::centering<Field::dimension>(*r.field));
}

template<typename Field, std::uint32_t gw>
NO_DISCARD auto ghost_box(FieldResource<Field, gw> const& r,
                          damn::core::Box<int, Field::dimension> const& cells)
{
    return damn::core::Box<int, Field::dimension>{
        core::qty_ghost_box(cells, core::centering<Field::dimension>(*r.field), gw)};
}

template<typename Field, std::uint32_t gw>
NO_DISCARD auto& value(FieldResource<Field, gw> const& r,
                       damn::core::Box<int, Field::dimension> const& cells,
                       damn::core::Point<int, Field::dimension> const& amr)
{
    return (*r.field)(core::amr_to_local(cells, amr, gw).toArray());
}


// what checkpoints hold of a field: its ghosts too (a refined level's level ghosts were filled
//  from the coarser level before its synchronization: they cannot be rebuilt bit for bit)
template<typename Field, std::uint32_t gw>
NO_DISCARD auto checkpoint_box(FieldResource<Field, gw> const& r,
                               damn::core::Box<int, Field::dimension> const& cells)
{
    return ghost_box(r, cells);
}

// a C ordered field's values over its ghost box, written and read in place by checkpoints
template<typename Field, std::uint32_t gw>
    requires(Field::is_c_ordered)
NO_DISCARD auto box_span(FieldResource<Field, gw> const& r,
                         damn::core::Box<int, Field::dimension> const& cells)
{
    auto const ghost = ghost_box(r, cells);
    if (r.field->size() != ghost.size())
        throw std::runtime_error("plamn::amr::box_span: a field not sized for its ghost box");
    return damn::core::BoxSpan<typename Field::value_type, Field::dimension>{r.field->data(),
                                                                             ghost};
}


template<typename Particles, std::uint32_t gw>
struct ParticleResource
{
    using value_type                = Particles::value_type;
    auto static constexpr dimension = Particles::dimension;
    Particles* particles;
};

template<std::uint32_t gw, typename Particles>
NO_DISCARD auto particle_resource(Particles& particles)
{
    return ParticleResource<Particles, gw>{std::addressof(particles)};
}

template<typename Particles, std::uint32_t gw>
NO_DISCARD auto domain_box(ParticleResource<Particles, gw> const&,
                           damn::core::Box<int, Particles::dimension> const& cells)
{
    return cells;
}

template<typename Particles, std::uint32_t gw>
NO_DISCARD auto ghost_box(ParticleResource<Particles, gw> const&,
                          damn::core::Box<int, Particles::dimension> const& cells)
{
    return damn::core::grow(cells, damn::core::ConstArray<int, Particles::dimension>(gw));
}

template<typename Particles, std::uint32_t gw>
void for_each_item(ParticleResource<Particles, gw> const& r, auto&& fn)
{
    auto const& particles = *r.particles;
    for (std::size_t i = 0; i < particles.size(); ++i)
        fn(particles[i]);
}

template<typename Particles, std::uint32_t gw>
void append(ParticleResource<Particles, gw>& r, typename Particles::value_type particle,
            damn::core::Point<int, Particles::dimension> const& shift)
{
    for (std::size_t i = 0; i < Particles::dimension; ++i)
        particle.iCell()[i] += shift[i];
    r.particles->emplace_back(particle);
}

// the particles' components as checkpoint columns (weight, iCell_x.., delta_x.., v_x..)
template<typename Particles, std::uint32_t gw>
    requires core::HasComponents<Particles>
void for_each_column(ParticleResource<Particles, gw> const& r, auto&& fn)
{
    core::components(*r.particles).for_each(fn);
}

template<typename Particles, std::uint32_t gw>
    requires core::HasComponents<Particles>
void resize_columns(ParticleResource<Particles, gw> const& r, std::size_t const n)
{
    r.particles->resize(n); // the caller rebuilds the layout's maps (cell map, offsets) after
}

} // namespace plamn::amr


namespace plamn::core
{
// a particle's cell, for damn::amr::ParticleData (found by ADL on the particle)
template<std::size_t dim>
NO_DISCARD auto cell(Particle<dim> const& particle)
{
    return damn::core::Point<int, dim>{particle.iCell()};
}
} // namespace plamn::core

#endif // PLAMN_AMR_RESOURCES_HPP
