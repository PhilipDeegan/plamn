#include "plamn/core/def.hpp"
// IWYU pragma: private, include "core/data/particles/appending/particles_appending.hpp"
#ifndef PLAMN_CORE_DATA_PARTICLES_APPENDING_DETAIL_DEF_APPENDING
#define PLAMN_CORE_DATA_PARTICLES_APPENDING_DETAIL_DEF_APPENDING

#include "plamn/core/data/particles/particles_def.hpp"

#include <cstddef>

namespace plamn::core
{

template<auto src_layout_mde, auto src_alloc_mde, auto dst_layout_mde, auto dst_alloc_mde>
struct ParticlesAppender
{
    static_assert(damn::core::all_are<LayoutMode>(src_layout_mde, dst_layout_mde));
    static_assert(damn::core::all_are<damn::AllocatorMode>(src_alloc_mde, dst_alloc_mde));

    auto constexpr static src_layout_mode = src_layout_mde;
    auto constexpr static src_alloc_mode  = src_alloc_mde;

    auto constexpr static dst_layout_mode = dst_layout_mde;
    auto constexpr static dst_alloc_mode  = dst_alloc_mde;

    template<auto type, typename Src, typename Dst>
    void operator()(Src const& src, Dst& dst);

    std::size_t const start;
    std::size_t const end;
};




} // namespace plamn::core


#endif /* PLAMN_CORE_DATA_PARTICLES_APPENDING_DETAIL_DEF_APPENDING */
