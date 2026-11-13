#ifndef PLAMN_AMR_RESOURCES_HPP
#define PLAMN_AMR_RESOURCES_HPP

// plamn data as damn::amr resources, for damn's schedules
//
//   field_resource<gw>(field):     BoxData, values on the field's nodes (its centering), gw
//                                  ghost nodes each side
//   particle_resource<gw>(array):  ParticleData, particles in cells, gw ghost cells
//
// both are small handles over the data, made on the fly in a for_each_resource

#include "plamn/core/def.hpp"
#include "plamn/core/data/grid/layout.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"

#include <memory>
#include <cstddef>
#include <cstdint>

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
    return core::amr_box(cells, core::centering<Field::dimension>(*r.field));
}

template<typename Field, std::uint32_t gw>
NO_DISCARD auto ghost_box(FieldResource<Field, gw> const& r,
                          damn::core::Box<int, Field::dimension> const& cells)
{
    return damn::core::Box<int, Field::dimension>{
        core::amr_ghost_box(cells, core::centering<Field::dimension>(*r.field), gw)};
}

template<typename Field, std::uint32_t gw>
NO_DISCARD auto& value(FieldResource<Field, gw> const& r,
                       damn::core::Box<int, Field::dimension> const& cells,
                       damn::core::Point<int, Field::dimension> const& amr)
{
    return (*r.field)(core::amr_to_local(cells, amr, gw).toArray());
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
