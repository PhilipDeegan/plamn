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
#include "mkn/kul/std/array.hpp"

#include "plamn/core/data/particles/particle_components.hpp"
#include "plamn/core/def.hpp"
#include "plamn/core/data/grid/layout.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"
#include "plamn/core/data/particles/particle_iteration.hpp"
#include "plamn/core/data/particles/particles_def.hpp"

#include <cstddef>
#include <stdexcept>
#include <cstdint>
#include <vector>
#include <cstring>
#include <utility>
#include <format>
#include <concepts>
#include <type_traits>

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
    return damn::core::grow(cells, mkn::kul::ConstArray<int, Particles::dimension>(gw));
}

template<typename Particles, std::uint32_t gw>
void for_each_item(ParticleResource<Particles, gw> const& r, auto&& fn)
{
    core::for_each_particle(*r.particles, fn); // tiled arrays tile by tile
}

template<typename Particles, std::uint32_t gw>
void append(ParticleResource<Particles, gw>& r, typename Particles::value_type particle,
            damn::core::Point<int, Particles::dimension> const& shift)
{
    for (std::size_t i = 0; i < Particles::dimension; ++i)
        particle.iCell()[i] += shift[i];
    r.particles->emplace_back(particle);
}

// SoAVX, SoAVXChunked (host memory): moved between patches column by column, damn's
//  particle_items hooks (damn/amr/particle_items.hpp). The bytes: the number of particles, then
//  each component's values contiguously, in components()' order (weight, iCell_x.., delta_x..,
//  v_x, v_y, v_z). A selection reads the iCell columns only, or SoAVXChunked's per cell ranges
//  when sorted; appends are column wise, SoAVXChunked sorted once after all of them
//  A resource over Particles const (a read only source, amr/particle_access.hpp) packs too
template<typename Particles>
concept ParticleColumns = core::HasComponents<std::remove_const_t<Particles>>
                          and core::is_soavx(Particles::layout_mode);

namespace detail
{
    // the particles whose cell + shift is in box, by index
    template<typename Particles>
    std::vector<std::size_t> selected(Particles const& particles,
                                      damn::core::Box<int, Particles::dimension> const& box,
                                      damn::core::Point<int, Particles::dimension> const& shift)
    {
        auto constexpr dim = Particles::dimension;
        std::vector<std::size_t> out;
        if constexpr (Particles::layout_mode == core::LayoutMode::SoAVXChunked)
            if (particles.sorted())
            {
                damn::core::Point<int, dim> back; // box in the particles' cells
                for (std::size_t d = 0; d < dim; ++d)
                    back[d] = -shift[d];
                auto const cells = damn::core::shift(box, back) * particles.shell();
                if (cells)
                    particles.for_each_range(*cells, [&](auto const start, auto const size) {
                        for (std::size_t i = start; i < start + size; ++i)
                            out.push_back(i);
                    });
                return out;
            }
        auto const c = core::components(particles);
        for (std::size_t i = 0; i < particles.size(); ++i)
        {
            if (c.weight[i] == 0) // SoAVXChunked's gaps
                continue;
            damn::core::Point<int, dim> cell;
            for (std::size_t d = 0; d < dim; ++d)
                cell[d] = c.iCell[d][i] + shift[d];
            if (isIn(cell, box))
                out.push_back(i);
        }
        return out;
    }

    // fn(dst span, src span) for each component pair, the iCells with their direction
    template<std::size_t dim>
    void for_each_column_pair(auto const& dst, auto const& src, auto&& fn)
    {
        fn(dst.weight, src.weight, -1);
        for (std::size_t d = 0; d < dim; ++d)
            fn(dst.iCell[d], src.iCell[d], static_cast<int>(d));
        for (std::size_t d = 0; d < dim; ++d)
            fn(dst.delta[d], src.delta[d], -1);
        for (std::size_t d = 0; d < 3; ++d)
            fn(dst.v[d], src.v[d], -1);
    }
} // namespace detail

template<typename Particles, std::uint32_t gw>
    requires ParticleColumns<Particles>
void pack_items(ParticleResource<Particles, gw> const& r,
                damn::core::Box<int, Particles::dimension> const& box,
                damn::core::Point<int, Particles::dimension> const& shift,
                std::vector<std::byte>& out)
{
    auto const& particles = *r.particles;
    auto const index      = detail::selected(particles, box, shift);
    std::uint64_t const n = index.size();
    auto at               = out.size();
    std::size_t bytes     = sizeof(n);
    core::components(particles).for_each(
        [&](auto const&, auto const& span) { bytes += n * sizeof(span[0]); });
    out.resize(at + bytes);
    std::memcpy(out.data() + at, &n, sizeof(n));
    at += sizeof(n);
    core::components(particles).for_each([&](auto const&, auto const& span) {
        for (auto const i : index)
        {
            std::memcpy(out.data() + at, &span[i], sizeof(span[i]));
            at += sizeof(span[i]);
        }
    });
}

template<typename Particles, std::uint32_t gw>
    requires ParticleColumns<Particles>
void unpack_items(ParticleResource<Particles, gw>& r, std::byte const*& at,
                  damn::core::Point<int, Particles::dimension> const& shift)
{
    auto& particles = *r.particles;
    std::uint64_t n;
    std::memcpy(&n, at, sizeof(n));
    at += sizeof(n);
    if (n == 0)
        return;
    if constexpr (Particles::layout_mode == core::LayoutMode::SoAVXChunked)
        if (particles.sorted() and particles.has_slack())
        { // into their chunks' gaps, one at a time: a particle from each component's stream
            auto constexpr dim      = Particles::dimension;
            std::byte const* weight = at;
            std::byte const* icell  = weight + n * sizeof(double);
            std::byte const* delta  = icell + n * dim * sizeof(int);
            std::byte const* v      = delta + n * dim * sizeof(double);
            for (std::size_t k = 0; k < n; ++k)
            {
                typename Particles::value_type p;
                std::memcpy(&p.weight(), weight + k * sizeof(double), sizeof(double));
                for (std::size_t d = 0; d < dim; ++d)
                {
                    std::memcpy(&p.iCell()[d], icell + (d * n + k) * sizeof(int), sizeof(int));
                    p.iCell()[d] += shift[d];
                    std::memcpy(&p.delta()[d], delta + (d * n + k) * sizeof(double),
                                sizeof(double));
                }
                for (std::size_t d = 0; d < 3; ++d)
                    std::memcpy(&p.v()[d], v + (d * n + k) * sizeof(double), sizeof(double));
                if (!particles.insert(p))
                    particles.push_back(p); // a sort after all of them (items_appended)
            }
            at = v + n * 3 * sizeof(double);
            return;
        }
    auto const was = particles.size();
    particles.resize(was + n);
    auto const c = core::components(particles);
    c.for_each([&](auto const&, auto const& span) {
        for (std::size_t k = 0; k < n; ++k)
        {
            std::memcpy(&span[was + k], at, sizeof(span[was + k]));
            at += sizeof(span[was + k]);
        }
    });
    for (std::size_t d = 0; d < Particles::dimension; ++d)
        for (std::size_t k = 0; k < n; ++k)
            c.iCell[d][was + k] += shift[d];
}

// as pack_items then unpack_items, without the bytes (src may be dst: its first particles stay;
//  src may be over Particles const)
template<typename Particles, typename Src, std::uint32_t gw>
    requires ParticleColumns<Particles> and std::same_as<std::remove_const_t<Src>, Particles>
void append_items_from(ParticleResource<Particles, gw>& dst, ParticleResource<Src, gw> const& src,
                       damn::core::Box<int, Particles::dimension> const& box,
                       damn::core::Point<int, Particles::dimension> const& shift)
{
    auto const index = detail::selected(*src.particles, box, shift);
    if (index.empty())
        return;
    if constexpr (Particles::layout_mode == core::LayoutMode::SoAVXChunked)
        if (dst.particles->sorted() and dst.particles->has_slack()
            and src.particles != dst.particles)
        { // into their chunks' gaps
            for (auto const i : index)
            {
                auto p = src.particles->copy(i);
                for (std::size_t d = 0; d < Particles::dimension; ++d)
                    p.iCell()[d] += shift[d];
                if (!dst.particles->insert(p))
                    dst.particles->push_back(p); // a sort after all of them (items_appended)
            }
            return;
        }
    auto& out      = *dst.particles;
    auto const was = out.size();
    out.resize(was + index.size()); // src's columns too if the same: read after
    detail::for_each_column_pair<Particles::dimension>(
        core::components(out), core::components(std::as_const(*src.particles)),
        [&](auto const& to, auto const& from, int const d) {
            for (std::size_t k = 0; k < index.size(); ++k)
                to[was + k] = from[index[k]] + (d < 0 ? 0 : shift[static_cast<std::size_t>(d)]);
        });
}

// what pack_items wrote, as particles (refine's coarse particles, split into new ones)
template<typename Particles, std::uint32_t gw>
    requires ParticleColumns<Particles>
std::vector<typename Particles::value_type> read_items(ParticleResource<Particles, gw> const&,
                                                       std::byte const*& at)
{
    auto constexpr dim = Particles::dimension;
    std::uint64_t n;
    std::memcpy(&n, at, sizeof(n));
    at += sizeof(n);
    std::vector<typename Particles::value_type> out(n);
    auto const read = [&](auto& to) {
        std::memcpy(&to, at, sizeof(to));
        at += sizeof(to);
    };
    for (auto& p : out)
        read(p.weight());
    for (std::size_t d = 0; d < dim; ++d)
        for (auto& p : out)
            read(p.iCell()[d]);
    for (std::size_t d = 0; d < dim; ++d)
        for (auto& p : out)
            read(p.delta()[d]);
    for (std::size_t d = 0; d < 3; ++d)
        for (auto& p : out)
            read(p.v()[d]);
    return out;
}

// after a copy or a regrid appended to it: SoAVXChunked's offsets rebuilt, once
template<typename Particles, std::uint32_t gw>
    requires ParticleColumns<Particles>
void items_appended(ParticleResource<Particles, gw>& r)
{
    if constexpr (Particles::layout_mode == core::LayoutMode::SoAVXChunked)
        if (!r.particles->sorted()) // appended to, not only inserted into its gaps
            r.particles->sort();
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
