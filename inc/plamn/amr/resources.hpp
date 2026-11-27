#ifndef PLAMN_AMR_RESOURCES_HPP
#define PLAMN_AMR_RESOURCES_HPP

// plamn data as damn::amr resources, small handles made on the fly: doc/amr/resources.md

#include "damn/core/geometry/box.hpp"
#include "damn/core/memory/box_span.hpp"
#include "damn/core/memory/scratch.hpp"
#include "damn/core/geometry/point.hpp"
#include "mkn/kul/for.hpp"
#include "mkn/kul/std/array.hpp"
#include "plamn/core/data/particles/particle_components.hpp"
#include "plamn/core/def.hpp"
#include "plamn/core/data/grid/layout.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"
#include "plamn/core/data/particles/particle_iteration.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/data/particles/selecting/particles_selecting.hpp"
#include "damn/core/def/damn_config.hpp"

#include <array>
#include <cstddef>
#include <algorithm>
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
    // in unified memory: damn's same rank copies run on the device over box_span (data + box)
    bool static constexpr on_device = Field::alloc_mode == damn::AllocatorMode::GPU_UNIFIED;
    Field* field;
};

template<std::uint32_t gw, typename Field>
MKN_KUL_FN_SIG_RET auto field_resource(Field& field)
{
    return FieldResource<Field, gw>{std::addressof(field)}; // Grid overloads operator&
}

template<typename Field, std::uint32_t gw>
MKN_KUL_FN_SIG_RET auto domain_box(FieldResource<Field, gw> const& r,
                                   damn::core::Box<int, Field::dimension> const& cells)
{
    return core::qty_box(cells, core::centering<Field::dimension>(*r.field));
}

template<typename Field, std::uint32_t gw>
MKN_KUL_FN_SIG_RET auto ghost_box(FieldResource<Field, gw> const& r,
                                  damn::core::Box<int, Field::dimension> const& cells)
{
    return damn::core::Box<int, Field::dimension>{
        core::qty_ghost_box(cells, core::centering<Field::dimension>(*r.field), gw)};
}

template<typename Field, std::uint32_t gw>
MKN_KUL_FN_SIG_RET auto& value(FieldResource<Field, gw> const& r,
                               damn::core::Box<int, Field::dimension> const& cells,
                               damn::core::Point<int, Field::dimension> const& amr)
{
    return (*r.field)(core::amr_to_local(cells, amr, gw).toArray());
}


// ghosts too: a refined level's level ghosts were filled from the coarser level before its
//  synchronization, they cannot be rebuilt bit for bit
template<typename Field, std::uint32_t gw>
MKN_KUL_FN_SIG_RET auto checkpoint_box(FieldResource<Field, gw> const& r,
                                       damn::core::Box<int, Field::dimension> const& cells)
{
    return ghost_box(r, cells);
}

// a C ordered field's values over its ghost box, written and read in place by checkpoints
template<typename Field, std::uint32_t gw>
    requires(Field::is_c_ordered)
MKN_KUL_FN_SIG_RET auto box_span(FieldResource<Field, gw> const& r,
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
    // in unified memory: damn moves its leaving particles on the device (device_items.hpp)
    bool static constexpr on_device = Particles::alloc_mode == damn::AllocatorMode::GPU_UNIFIED;
    Particles* particles;
};

template<std::uint32_t gw, typename Particles>
MKN_KUL_FN_SIG_RET auto particle_resource(Particles& particles)
{
    return ParticleResource<Particles, gw>{std::addressof(particles)};
}

template<typename Particles, std::uint32_t gw>
MKN_KUL_FN_SIG_RET auto domain_box(ParticleResource<Particles, gw> const&,
                                   damn::core::Box<int, Particles::dimension> const& cells)
{
    return cells;
}

template<typename Particles, std::uint32_t gw>
MKN_KUL_FN_SIG_RET auto ghost_box(ParticleResource<Particles, gw> const&,
                                  damn::core::Box<int, Particles::dimension> const& cells)
{
    return damn::core::grow(cells, mkn::kul::ConstArray<int, Particles::dimension>(gw));
}

template<typename Particles, std::uint32_t gw>
void for_each_item(ParticleResource<Particles, gw> const& r, auto&& fn)
{
    if constexpr (core::is_chunked(std::remove_const_t<Particles>::layout_mode))
        core::for_each_particle(*r.particles, [&](auto const& p) {
            if (p.weight() != 0) // not a chunks' gap
                fn(p);
        });
    else
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

// SoAVX, SoAVXChunked (host memory): moved column by column, the bytes: doc/amr/resources.md
template<typename Particles>
concept ParticleColumns = core::HasComponents<std::remove_const_t<Particles>>
                          and Particles::layout_mode == core::LayoutMode::SoAVXChunked;

namespace detail
{
    // box in the particles' own cells
    template<std::size_t dim>
    auto unshifted(damn::core::Box<int, dim> const& box, damn::core::Point<int, dim> const& shift)
    {
        damn::core::Point<int, dim> back;
        for (std::size_t d = 0; d < dim; ++d)
            back[d] = -shift[d];
        return damn::core::shift(box, back);
    }

    template<typename Particles>
    void for_each_selected(Particles const& particles,
                           damn::core::Box<int, Particles::dimension> const& box,
                           damn::core::Point<int, Particles::dimension> const& shift, auto&& fn)
    {
        core::for_each_range_in(particles, unshifted(box, shift), fn);
    }

    template<typename Particles>
    std::size_t count_selected(Particles const& particles,
                               damn::core::Box<int, Particles::dimension> const& box,
                               damn::core::Point<int, Particles::dimension> const& shift)
    {
        return core::nbr_particles_in(particles, unshifted(box, shift));
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
    auto const c          = core::components(particles);
    std::uint64_t const n = detail::count_selected(particles, box, shift);
    auto const at         = out.size() + sizeof(n);
    std::size_t bytes     = sizeof(n);
    c.for_each([&](auto const&, auto const& span) { bytes += n * sizeof(span[0]); });
    out.resize(out.size() + bytes);
    std::memcpy(out.data() + at - sizeof(n), &n, sizeof(n));
    std::size_t k = 0;
    detail::for_each_selected(
        particles, box, shift, [&](std::size_t const start, std::size_t const size) {
            auto column = at;
            c.for_each([&](auto const&, auto const& span) {
                auto constexpr bytes = sizeof(span[0]);
                std::memcpy(out.data() + column + k * bytes, &span[start], size * bytes);
                column += n * bytes;
            });
            k += size;
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
    bool const same = static_cast<void const*>(src.particles) == dst.particles;
    if constexpr (Particles::layout_mode == core::LayoutMode::SoAVXChunked)
        if (!same and dst.particles->sorted() and dst.particles->has_slack())
        { // into their chunks' gaps
            detail::for_each_selected(
                *src.particles, box, shift, [&](std::size_t const start, std::size_t const size) {
                    for (std::size_t i = start; i < start + size; ++i)
                    {
                        auto p = src.particles->copy(i);
                        for (std::size_t d = 0; d < Particles::dimension; ++d)
                            p.iCell()[d] += shift[d];
                        if (!dst.particles->insert(p))
                            dst.particles->push_back(p); // a sort after all (items_appended)
                    }
                });
            return;
        }
    auto const n = detail::count_selected(*src.particles, box, shift);
    if (n == 0)
        return;
    // the same array: its ranges taken before the resize, whose growth moves the chunk offsets
    damn::core::ScratchVector<std::array<std::size_t, 2>> kept;
    if (same)
        detail::for_each_selected(*src.particles, box, shift,
                                  [&](std::size_t const start, std::size_t const size) {
                                      kept.push_back({start, size});
                                  });
    auto& out      = *dst.particles;
    auto const was = out.size();
    out.resize(was + n); // src's columns too if the same: read after
    auto const to   = core::components(out);
    auto const from = core::components(std::as_const(*src.particles));
    std::size_t k   = 0;
    auto const copy = [&](std::size_t const start, std::size_t const size) {
        detail::for_each_column_pair<Particles::dimension>(
            to, from, [&](auto const& t, auto const& f, int const d) {
                auto const by = d < 0 ? 0 : shift[static_cast<std::size_t>(d)];
                for (std::size_t j = 0; j < size; ++j)
                    t[was + k + j] = f[start + j] + by;
            });
        k += size;
    };
    if (same)
        for (auto const& [start, size] : kept)
            copy(start, size);
    else
        detail::for_each_selected(*src.particles, box, shift, copy);
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

// AoSMapped, AoS4SoAVXChunked (host memory): moved as whole structures, the bytes:
//  doc/amr/resources.md
template<typename Particles>
concept ParticleStructs
    = mkn::kul::any_in(std::remove_const_t<Particles>::layout_mode, core::LayoutMode::AoSMapped,
                       core::LayoutMode::AoS4SoAVXChunked);

namespace detail
{
    bool has_gaps(auto const&)
    {
        return false;
    }

    template<typename Particles>
        requires(core::is_chunked(Particles::layout_mode))
    bool has_gaps(Particles const& particles)
    {
        return particles.sorted() and particles.has_slack();
    }

    template<typename Particles>
    auto shifted(typename Particles::value_type p,
                 damn::core::Point<int, Particles::dimension> const& shift)
    {
        for (std::size_t d = 0; d < Particles::dimension; ++d)
            p.iCell()[d] += shift[d];
        return p;
    }

    template<typename Particles>
    void arrive(Particles& dst, typename Particles::value_type const& p,
                damn::core::Point<int, Particles::dimension> const& shift, bool /*gaps*/)
    {
        dst.push_back(shifted<Particles>(p, shift));
    }

    // its chunk's gaps if it can, else appended
    template<typename Particles>
        requires(core::is_chunked(Particles::layout_mode))
    void arrive(Particles& dst, typename Particles::value_type const& p,
                damn::core::Point<int, Particles::dimension> const& shift, bool const gaps)
    {
        auto const q = shifted<Particles>(p, shift);
        if (!gaps or !dst.insert(q))
            dst.push_back(q);
    }
} // namespace detail

template<typename Particles, std::uint32_t gw>
    requires ParticleStructs<Particles>
void pack_items(ParticleResource<Particles, gw> const& r,
                damn::core::Box<int, Particles::dimension> const& box,
                damn::core::Point<int, Particles::dimension> const& shift,
                std::vector<std::byte>& out)
{
    using P               = typename Particles::value_type;
    auto const& particles = *r.particles;
    auto const count_at   = out.size(); // the count is known only once visited
    out.resize(count_at + sizeof(std::uint64_t));
    std::uint64_t n = 0;
    detail::for_each_selected(particles, box, shift,
                              [&](std::size_t const start, std::size_t const size) {
                                  auto const at = out.size();
                                  out.resize(at + size * sizeof(P));
                                  std::memcpy(out.data() + at, &particles[start], size * sizeof(P));
                                  n += size;
                              });
    std::memcpy(out.data() + count_at, &n, sizeof(n));
}

template<typename Particles, std::uint32_t gw>
    requires ParticleStructs<Particles>
void unpack_items(ParticleResource<Particles, gw>& r, std::byte const*& at,
                  damn::core::Point<int, Particles::dimension> const& shift)
{
    using P         = typename Particles::value_type;
    auto& particles = *r.particles;
    std::uint64_t n;
    std::memcpy(&n, at, sizeof(n));
    at += sizeof(n);
    bool const gaps = detail::has_gaps(particles);
    for (std::uint64_t k = 0; k < n; ++k)
    {
        P p;
        std::memcpy(&p, at, sizeof(P));
        at += sizeof(P);
        detail::arrive(particles, p, shift, gaps);
    }
}

// as pack_items then unpack_items, without the bytes (src may be dst: the selection copied first)
template<typename Particles, typename Src, std::uint32_t gw>
    requires ParticleStructs<Particles> and std::same_as<std::remove_const_t<Src>, Particles>
void append_items_from(ParticleResource<Particles, gw>& dst, ParticleResource<Src, gw> const& src,
                       damn::core::Box<int, Particles::dimension> const& box,
                       damn::core::Point<int, Particles::dimension> const& shift)
{
    using P         = typename Particles::value_type;
    bool const same = static_cast<void const*>(src.particles) == dst.particles;
    bool const gaps = !same and detail::has_gaps(*dst.particles);
    if (!same)
    {
        detail::for_each_selected(
            *src.particles, box, shift, [&](std::size_t const start, std::size_t const size) {
                for (std::size_t i = start; i < start + size; ++i)
                    detail::arrive(*dst.particles, (*src.particles)[i], shift, gaps);
            });
        return;
    }
    // the same array: the selection copied first, appending moves it
    damn::core::Scratch<P> moving{detail::count_selected(*src.particles, box, shift)};
    std::size_t k = 0;
    detail::for_each_selected(*src.particles, box, shift,
                              [&](std::size_t const start, std::size_t const size) {
                                  std::copy_n(&(*src.particles)[start], size, moving.data() + k);
                                  k += size;
                              });
    for (auto const& p : moving)
        detail::arrive(*dst.particles, p, shift, gaps);
}

// what pack_items wrote, as particles (refine's coarse particles, split into new ones)
template<typename Particles, std::uint32_t gw>
    requires ParticleStructs<Particles>
std::vector<typename Particles::value_type> read_items(ParticleResource<Particles, gw> const&,
                                                       std::byte const*& at)
{
    using P = typename Particles::value_type;
    std::uint64_t n;
    std::memcpy(&n, at, sizeof(n));
    at += sizeof(n);
    std::vector<P> out(n);
    if (n)
        std::memcpy(out.data(), at, n * sizeof(P));
    at += n * sizeof(P);
    return out;
}

// after a copy or a regrid appended to it: a chunked array's offsets rebuilt, once
template<typename Particles, std::uint32_t gw>
    requires(core::is_chunked(Particles::layout_mode))
void items_appended(ParticleResource<Particles, gw>& r)
{
    if (!r.particles->sorted()) // appended to, not only inserted into its gaps
        r.particles->sort();
}

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
// a particle's cell, for damn::amr::CellItemData (found by ADL on the particle)
template<std::size_t dim>
MKN_KUL_FN_SIG_RET auto cell(Particle<dim> const& particle)
{
    return damn::core::Point<int, dim>{particle.iCell()};
}
} // namespace plamn::core

// after the resources it uses
#include "plamn/amr/device_items.hpp" // IWYU pragma: export

#endif // PLAMN_AMR_RESOURCES_HPP
