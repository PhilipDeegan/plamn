#ifndef PLAMN_AMR_DEVICE_ITEMS_HPP
#define PLAMN_AMR_DEVICE_ITEMS_HPP

// damn's device item hooks for unified memory SoAVXChunked particles: their columns, the cells
//  their chunks: doc/amr/device_items.md

#include "damn/core/def/damn_config.hpp"

#if DAMN_HAVE_GPU && DAMN_HAVE_THRUST

#include "damn/amr/copy.hpp" // damn::amr::DeviceItems (device_items.hpp, after copy.hpp)
#include "plamn/amr/resources.hpp"
#include "plamn/core/data/particles/arrays/particles_chunked.hpp"
#include "plamn/core/data/particles/particle_components.hpp"
#include "plamn/core/def.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace plamn::amr
{
namespace detail
{
    template<typename P>
    bool constexpr is_device_chunked = P::layout_mode == core::LayoutMode::SoAVXChunked
                                       and P::alloc_mode == damn::AllocatorMode::GPU_UNIFIED;

    // weight, delta.., v..: an item's columns, iCell left out (its chunk's cell)
    template<std::size_t dim>
    damn::amr::ItemColumns constexpr columns_layout()
    {
        damn::amr::ItemColumns layout;
        for (std::size_t c = 0; c < 1 + dim + 3; ++c)
            layout.bytes[layout.n++] = sizeof(double);
        return layout;
    }

    // addresses only: nothing read on the host
    template<std::size_t dim>
    void columns_of(auto const& c, auto& columns)
    {
        using byte_t = std::remove_pointer_t<std::remove_cvref_t<decltype(columns[0])>>;
        columns[0]   = reinterpret_cast<byte_t*>(&c.weight[0]);
        for (std::size_t k = 0; k < dim; ++k)
            columns[1 + k] = reinterpret_cast<byte_t*>(&c.delta[k][0]);
        for (std::size_t k = 0; k < 3; ++k)
            columns[1 + dim + k] = reinterpret_cast<byte_t*>(&c.v[k][0]);
    }
} // namespace detail

template<typename Particles, std::uint32_t gw>
    requires(detail::is_device_chunked<std::remove_const_t<Particles>>)
damn::amr::DeviceItems<Particles::dimension> device_items(ParticleResource<Particles, gw> const& r)
{
    auto constexpr dim = Particles::dimension;
    auto const& p      = *r.particles;
    if (!p.sorted()) // its chunks give a cell's particles only once sorted
        throw std::runtime_error("plamn::amr::device_items: particles not sorted");
    damn::amr::DeviceItems<dim> items{
        p.shell(), p.starts_data(), p.sizes_data(), {}, detail::columns_layout<dim>()};
    detail::columns_of<dim>(core::components(p), items.columns);
    return items;
}

// unsorted after: items_appended sorts
template<typename Particles, std::uint32_t gw>
    requires(detail::is_device_chunked<Particles>)
damn::amr::DeviceItemsOut<Particles::dimension>
device_items_appended(ParticleResource<Particles, gw>& r, std::size_t const n)
{
    auto constexpr dim = Particles::dimension;
    auto& p            = *r.particles;
    auto const was     = p.size();
    p.resize(was + n); // on the device (mkn::gpu::resize)
    damn::amr::DeviceItemsOut<dim> out{{}, detail::columns_layout<dim>(), {}};
    auto const c = core::components(p);
    detail::columns_of<dim>(c, out.columns);
    for (std::size_t k = 0; k < out.layout.n; ++k)
        out.columns[k] += was * sizeof(double);
    for (std::size_t d = 0; d < dim; ++d)
        out.cell[d] = &c.iCell[d][0] + was;
    return out;
}

} // namespace plamn::amr

#endif // DAMN_HAVE_GPU && DAMN_HAVE_THRUST

#endif /* PLAMN_AMR_DEVICE_ITEMS_HPP */
