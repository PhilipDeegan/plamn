#ifndef PLAMN_CORE_DATA_PARTICLES_COMPARING_DETAIL_DEF_COMPARING
#define PLAMN_CORE_DATA_PARTICLES_COMPARING_DETAIL_DEF_COMPARING

#include "damn/core/meta/type_meta.hpp"
#include "damn/core/math/equality.hpp"

#include "plamn/core/def.hpp"
#include "plamn/core/data/particles/particles_def.hpp"

#include <cstdint>
#include <cstddef>

namespace damn
{
enum class AllocatorMode : std::uint16_t;
} // namespace damn
namespace damn::core
{
struct EqualityReport;
} // namespace damn::core

namespace plamn::core
{
enum class LayoutMode : std::uint16_t;


template<auto src_layout_mde, auto src_alloc_mde, auto dst_layout_mde, auto dst_alloc_mde>
struct ParticlesComparator
{
    static_assert(damn::core::all_are<LayoutMode>(src_layout_mde, dst_layout_mde));
    static_assert(damn::core::all_are<damn::AllocatorMode>(src_alloc_mde, dst_alloc_mde));

    auto constexpr static src_layout_mode = src_layout_mde;
    auto constexpr static src_alloc_mode  = src_alloc_mde;

    auto constexpr static dst_layout_mode = dst_layout_mde;
    auto constexpr static dst_alloc_mode  = dst_alloc_mde;

    template<typename PS0, typename PS1>
    damn::core::EqualityReport operator()(PS0 const& ps0, PS1 const& ps1, double const atol = 1e-15);
};


} // namespace plamn::core


#endif /* PLAMN_CORE_DATA_PARTICLES_COMPARING_DETAIL_DEF_COMPARING */
