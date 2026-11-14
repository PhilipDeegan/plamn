// IWYU pragma: private, include "plamn/core/data/particles/converting/particles_converting.hpp"
#ifndef PLAMN_CORE_DATA_PARTICLES_CONVERTING_DETAIL_DEF_CONVERTING
#define PLAMN_CORE_DATA_PARTICLES_CONVERTING_DETAIL_DEF_CONVERTING

#include "damn/core/meta/type_meta.hpp"

#include <cstdint>

namespace damn
{
enum class AllocatorMode : std::uint16_t;
} // namespace damn

namespace plamn::core
{
enum class LayoutMode : std::uint16_t;


template<auto src_layout_mde, auto src_alloc_mde, auto dst_layout_mde, auto dst_alloc_mde>
struct ParticlesConverter
{
    static_assert(damn::core::all_are<LayoutMode>(src_layout_mde, dst_layout_mde));
    static_assert(damn::core::all_are<damn::AllocatorMode>(src_alloc_mde, dst_alloc_mde));

    auto constexpr static src_layout_mode = src_layout_mde;
    auto constexpr static src_alloc_mode  = src_alloc_mde;

    auto constexpr static dst_layout_mode = dst_layout_mde;
    auto constexpr static dst_alloc_mode  = dst_alloc_mde;

    template<typename Dst, typename Src, typename Layout>
    Dst operator()(Src const& src, Layout const& layout);
};


} // namespace plamn::core


#endif /* PLAMN_CORE_DATA_PARTICLES_CONVERTING_DETAIL_DEF_CONVERTING */
