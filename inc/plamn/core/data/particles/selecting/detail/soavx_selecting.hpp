// IWYU pragma: private, include "plamn/core/data/particles/selecting/particles_selecting.hpp"
#ifndef PLAMN_CORE_DATA_PARTICLES_SELECTING_DETAIL_SOAVX_SELECTING
#define PLAMN_CORE_DATA_PARTICLES_SELECTING_DETAIL_SOAVX_SELECTING

#include "damn/core/def/damn_config.hpp"
#include "damn/core/geometry/point.hpp"
#include "plamn/core/data/particles/particle_components.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/data/particles/selecting/detail/def_selecting.hpp" // IWYU pragma: keep
#include "plamn/core/data/particles/selecting/particles_selecting.hpp"

#include <cstddef>

namespace damn::core
{
template<typename Type, std::size_t dim>
struct Box;
} // namespace damn::core

namespace plamn::core
{
namespace detail
{
    // the chunked layouts' gaps (weight 0) are not particles
    template<typename Src, std::size_t dim, typename Fn>
    void visit_by_columns(Src const& src, damn::core::Box<int, dim> const& box, Fn&& fn)
    {
        auto const c = components(src);
        detail::IndexRuns<Fn> runs{fn};
        for (std::size_t i = 0; i < src.size(); ++i)
        {
            if (c.weight[i] == 0)
                continue;
            damn::core::Point<int, dim> cell;
            for (std::size_t d = 0; d < dim; ++d)
                cell[d] = c.iCell[d][i];
            if (isIn(cell, box))
                runs(i);
        }
        runs.flush();
    }

    template<typename Src, std::size_t dim, typename Fn>
    void visit_by_chunks(Src const& src, damn::core::Box<int, dim> const& box, Fn&& fn)
    {
        if (!src.sorted())
            return visit_by_columns(src, box, fn);
        if (auto const cells = box * src.shell())
            src.for_each_range(*cells, fn);
    }
} // namespace detail

template<>
template<typename Src, std::size_t dim, typename Fn>
void ParticlesSelector<LayoutMode::SoAVX, damn::AllocatorMode::CPU>::operator()(
    Src const& src, damn::core::Box<int, dim> const& box, Fn&& fn) const
{
    detail::visit_by_columns(src, box, fn);
}

template<>
template<typename Src, std::size_t dim, typename Fn>
void ParticlesSelector<LayoutMode::SoAVXChunked, damn::AllocatorMode::CPU>::operator()(
    Src const& src, damn::core::Box<int, dim> const& box, Fn&& fn) const
{
    detail::visit_by_chunks(src, box, fn);
}

template<>
template<typename Src, std::size_t dim, typename Fn>
void ParticlesSelector<LayoutMode::AoS4SoAVXChunked, damn::AllocatorMode::CPU>::operator()(
    Src const& src, damn::core::Box<int, dim> const& box, Fn&& fn) const
{
    detail::visit_by_chunks(src, box, fn);
}

// unified memory: read on the host, as on the CPU
template<>
template<typename Src, std::size_t dim, typename Fn>
void ParticlesSelector<LayoutMode::SoAVX, damn::AllocatorMode::GPU_UNIFIED>::operator()(
    Src const& src, damn::core::Box<int, dim> const& box, Fn&& fn) const
{
    detail::visit_by_columns(src, box, fn);
}

template<>
template<typename Src, std::size_t dim, typename Fn>
void ParticlesSelector<LayoutMode::SoAVXChunked, damn::AllocatorMode::GPU_UNIFIED>::operator()(
    Src const& src, damn::core::Box<int, dim> const& box, Fn&& fn) const
{
    detail::visit_by_chunks(src, box, fn);
}

template<>
template<typename Src, std::size_t dim, typename Fn>
void ParticlesSelector<LayoutMode::AoS4SoAVXChunked, damn::AllocatorMode::GPU_UNIFIED>::operator()(
    Src const& src, damn::core::Box<int, dim> const& box, Fn&& fn) const
{
    detail::visit_by_chunks(src, box, fn);
}

} // namespace plamn::core


#endif /* PLAMN_CORE_DATA_PARTICLES_SELECTING_DETAIL_SOAVX_SELECTING */
