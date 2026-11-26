// IWYU pragma: private, include "plamn/core/data/particles/selecting/particles_selecting.hpp"
#ifndef PLAMN_CORE_DATA_PARTICLES_SELECTING_DETAIL_CHUNKED_SELECTING
#define PLAMN_CORE_DATA_PARTICLES_SELECTING_DETAIL_CHUNKED_SELECTING

#include "damn/core/def/damn_config.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/data/particles/selecting/detail/def_selecting.hpp" // IWYU pragma: keep
#include "plamn/core/data/particles/selecting/particles_selecting.hpp"

#include <cstddef>
#include <stdexcept>

namespace damn::core
{
template<typename Type, std::size_t dim>
struct Box;
} // namespace damn::core

namespace plamn::core
{
namespace detail
{
    // the chunk offsets are stale once the array is appended to, until its next sort: no scan
    //  of the particles instead
    void check_sorted(auto const& src)
    {
        if (!src.sorted())
            throw std::runtime_error("plamn: particles selected from an unsorted chunked array");
    }

    template<typename Src, std::size_t dim, typename Fn>
    void visit_by_chunks(Src const& src, damn::core::Box<int, dim> const& box, Fn&& fn)
    {
        check_sorted(src);
        if (auto const cells = box * src.shell())
            src.for_each_range(*cells, fn);
    }

    template<typename Src, std::size_t dim>
    std::size_t count_by_chunks(Src const& src, damn::core::Box<int, dim> const& box)
    {
        check_sorted(src);
        return src.nbr_particles_in(box);
    }
} // namespace detail

template<>
template<typename Src, std::size_t dim, typename Fn>
void ParticlesSelector<LayoutMode::SoAVXChunked, damn::AllocatorMode::CPU>::operator()(
    Src const& src, damn::core::Box<int, dim> const& box, Fn&& fn) const
{
    detail::visit_by_chunks(src, box, fn);
}

template<>
template<typename Src, std::size_t dim>
std::size_t ParticlesSelector<LayoutMode::SoAVXChunked, damn::AllocatorMode::CPU>::count(
    Src const& src, damn::core::Box<int, dim> const& box) const
{
    return detail::count_by_chunks(src, box);
}

template<>
template<typename Src, std::size_t dim, typename Fn>
void ParticlesSelector<LayoutMode::SoAVXChunked, damn::AllocatorMode::GPU_UNIFIED>::operator()(
    Src const& src, damn::core::Box<int, dim> const& box, Fn&& fn) const
{
    detail::visit_by_chunks(src, box, fn);
}

template<>
template<typename Src, std::size_t dim>
std::size_t ParticlesSelector<LayoutMode::SoAVXChunked, damn::AllocatorMode::GPU_UNIFIED>::count(
    Src const& src, damn::core::Box<int, dim> const& box) const
{
    return detail::count_by_chunks(src, box);
}

template<>
template<typename Src, std::size_t dim, typename Fn>
void ParticlesSelector<LayoutMode::AoS4SoAVXChunked, damn::AllocatorMode::CPU>::operator()(
    Src const& src, damn::core::Box<int, dim> const& box, Fn&& fn) const
{
    detail::visit_by_chunks(src, box, fn);
}

template<>
template<typename Src, std::size_t dim>
std::size_t ParticlesSelector<LayoutMode::AoS4SoAVXChunked, damn::AllocatorMode::CPU>::count(
    Src const& src, damn::core::Box<int, dim> const& box) const
{
    return detail::count_by_chunks(src, box);
}

template<>
template<typename Src, std::size_t dim, typename Fn>
void ParticlesSelector<LayoutMode::AoS4SoAVXChunked, damn::AllocatorMode::GPU_UNIFIED>::operator()(
    Src const& src, damn::core::Box<int, dim> const& box, Fn&& fn) const
{
    detail::visit_by_chunks(src, box, fn);
}

template<>
template<typename Src, std::size_t dim>
std::size_t
ParticlesSelector<LayoutMode::AoS4SoAVXChunked, damn::AllocatorMode::GPU_UNIFIED>::count(
    Src const& src, damn::core::Box<int, dim> const& box) const
{
    return detail::count_by_chunks(src, box);
}

} // namespace plamn::core


#endif /* PLAMN_CORE_DATA_PARTICLES_SELECTING_DETAIL_CHUNKED_SELECTING */
