#ifndef PLAMN_CORE_DATA_PARTICLES_PARTICLE_COMPONENTS_HPP
#define PLAMN_CORE_DATA_PARTICLES_PARTICLE_COMPONENTS_HPP

// components(particles): every particle component as a damn::core::StridedSpan over the array's
// own memory, whatever the layout:
//
//   AoS, AoSMapped        a member of Particle            stride sizeof(Particle)
//   SoA                   a component of a member's array stride sizeof(std::array<., n>)
//   SoAVX, SoAVXChunked   a column                        contiguous
//
// what the DataWrangler hands python (live numpy views) and what restarts write and read in
// place, so neither copies nor needs to know the layout. Components in file order: weight,
// iCell_x.., delta_x.., v_x, v_y, v_z (for_each). Tiled layouts (one span per tile) and device
// memory: not yet

#include "damn/core/def.hpp"
#include "damn/core/def/damn_config.hpp"
#include "damn/core/memory/strided_span.hpp"

#include "plamn/core/data/particles/particles_def.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <type_traits>

namespace plamn::core
{
template<std::size_t dim, bool is_const>
struct ParticleComponents
{
    template<typename T>
    using Span_t = damn::core::StridedSpan<std::conditional_t<is_const, T const, T>>;

    Span_t<double> weight;
    std::array<Span_t<int>, dim> iCell;
    std::array<Span_t<double>, dim> delta;
    std::array<Span_t<double>, 3> v;

    // fn(name, span) for each component, in file order
    void for_each(auto&& fn) const
    {
        std::array<char const*, 3> constexpr xyz{"_x", "_y", "_z"};
        fn(std::string{"weight"}, weight);
        for (std::size_t i = 0; i < dim; ++i)
            fn("iCell" + std::string{xyz[i]}, iCell[i]);
        for (std::size_t i = 0; i < dim; ++i)
            fn("delta" + std::string{xyz[i]}, delta[i]);
        for (std::size_t i = 0; i < 3; ++i)
            fn("v" + std::string{xyz[i]}, v[i]);
    }
};

template<typename Particles>
concept HasComponents = std::decay_t<Particles>::options.alloc_mode == damn::AllocatorMode::CPU
                        and (std::decay_t<Particles>::layout_mode == LayoutMode::AoS
                             or std::decay_t<Particles>::layout_mode == LayoutMode::AoSMapped
                             or std::decay_t<Particles>::layout_mode == LayoutMode::SoA
                             or std::decay_t<Particles>::layout_mode == LayoutMode::SoAVX
                             or std::decay_t<Particles>::layout_mode == LayoutMode::SoAVXChunked);

template<typename Particles>
    requires HasComponents<Particles>
NO_DISCARD auto components(Particles& particles)
{
    using P                 = std::decay_t<Particles>;
    auto constexpr dim      = P::dimension;
    auto constexpr is_const = std::is_const_v<Particles>;
    auto constexpr layout   = P::layout_mode;
    using damn::core::strided_span;

    ParticleComponents<dim, is_const> c;
    auto const n = particles.size();
    if (n == 0)
        return c;

    if constexpr (layout == LayoutMode::AoS or layout == LayoutMode::AoSMapped)
    {
        auto* const p         = particles.data();
        auto constexpr stride = sizeof(*p);
        c.weight              = strided_span(&p->weight(), n, stride);
        for (std::size_t i = 0; i < dim; ++i)
        {
            c.iCell[i] = strided_span(&p->iCell()[i], n, stride);
            c.delta[i] = strided_span(&p->delta()[i], n, stride);
        }
        for (std::size_t i = 0; i < 3; ++i)
            c.v[i] = strided_span(&p->v()[i], n, stride);
    }
    else if constexpr (layout == LayoutMode::SoA)
    {
        c.weight = strided_span(particles.weight_.data(), n, sizeof(double));
        for (std::size_t i = 0; i < dim; ++i)
        {
            c.iCell[i]
                = strided_span(&particles.iCell_.data()[0][i], n, sizeof(particles.iCell_[0]));
            c.delta[i]
                = strided_span(&particles.delta_.data()[0][i], n, sizeof(particles.delta_[0]));
        }
        for (std::size_t i = 0; i < 3; ++i)
            c.v[i] = strided_span(&particles.v_.data()[0][i], n, sizeof(particles.v_[0]));
    }
    else // SoAVX, SoAVXChunked: a column per component
    {
        c.weight = strided_span(particles.weight_.data(), n, sizeof(double));
        for (std::size_t i = 0; i < dim; ++i)
        {
            c.iCell[i] = strided_span(particles.iCell_[i].data(), n, sizeof(int));
            c.delta[i] = strided_span(particles.delta_[i].data(), n, sizeof(double));
        }
        for (std::size_t i = 0; i < 3; ++i)
            c.v[i] = strided_span(particles.v_[i].data(), n, sizeof(double));
    }
    return c;
}

} // namespace plamn::core

#endif // PLAMN_CORE_DATA_PARTICLES_PARTICLE_COMPONENTS_HPP
