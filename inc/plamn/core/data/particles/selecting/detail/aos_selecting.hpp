// IWYU pragma: private, include "plamn/core/data/particles/selecting/particles_selecting.hpp"
#ifndef PLAMN_CORE_DATA_PARTICLES_SELECTING_DETAIL_AOS_SELECTING
#define PLAMN_CORE_DATA_PARTICLES_SELECTING_DETAIL_AOS_SELECTING

#include "damn/core/def/damn_config.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/data/particles/selecting/detail/def_selecting.hpp" // IWYU pragma: keep
#include "plamn/core/data/particles/selecting/particles_selecting.hpp"

#include <cstddef>

namespace plamn::core
{

template<>
template<typename Src, std::size_t dim, typename Fn>
void ParticlesSelector<LayoutMode::AoSMapped, damn::AllocatorMode::CPU>::operator()(
    Src const& src, damn::core::Box<int, dim> const& box, Fn&& fn) const
{
    auto const cells = box * damn::core::grow(src.box(), 1); // the CellMap's span: the safe shell
    if (!cells)
        return;
    detail::IndexRuns<Fn> runs{fn};
    damn::core::for_each_point(*cells, [&](auto const& cell) {
        for (auto const i : src.map(cell)) // map(cell) takes AMR indexes
            runs(i);
    });
    runs.flush();
}

template<>
template<typename Src, std::size_t dim>
std::size_t ParticlesSelector<LayoutMode::AoSMapped, damn::AllocatorMode::CPU>::count(
    Src const& src, damn::core::Box<int, dim> const& box) const
{
    return src.nbr_particles_in(box);
}

// unified memory: read on the host, as on the CPU
template<>
template<typename Src, std::size_t dim, typename Fn>
void ParticlesSelector<LayoutMode::AoSMapped, damn::AllocatorMode::GPU_UNIFIED>::operator()(
    Src const& src, damn::core::Box<int, dim> const& box, Fn&& fn) const
{
    ParticlesSelector<LayoutMode::AoSMapped, damn::AllocatorMode::CPU>{}(src, box, fn);
}

template<>
template<typename Src, std::size_t dim>
std::size_t ParticlesSelector<LayoutMode::AoSMapped, damn::AllocatorMode::GPU_UNIFIED>::count(
    Src const& src, damn::core::Box<int, dim> const& box) const
{
    return src.nbr_particles_in(box);
}

} // namespace plamn::core


#endif /* PLAMN_CORE_DATA_PARTICLES_SELECTING_DETAIL_AOS_SELECTING */
