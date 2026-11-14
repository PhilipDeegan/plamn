#ifndef PLAMN_CORE_DATA_PARTICLES_CONVERTING_PARTICLES_CONVERTING
#define PLAMN_CORE_DATA_PARTICLES_CONVERTING_PARTICLES_CONVERTING

#include "damn/core/geometry/box.hpp"
#include "damn/core/logger.hpp"
#include "damn/core/sys/strings.hpp"

#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/data/particles/converting/detail/def_converting.hpp"
#include "plamn/core/data/particles/converting/detail/aos_converting.hpp"
#include "plamn/core/data/particles/converting/detail/soa_converting.hpp"
#include "plamn/core/data/particles/sorting/particles_sorting.hpp"

#include "phlop/timing/scope.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string_view>
#include <type_traits>

namespace plamn::core
{
template<auto opts>
class Particles;
template<auto src_layout_mde, auto src_alloc_mde, auto dst_layout_mde, auto dst_alloc_mde>
struct ParticlesConverter;

// generic fallthrough
template<auto src_layout_mde, auto src_alloc_mde, auto dst_layout_mde, auto dst_alloc_mde>
template<typename Dst, typename Src, typename Layout>
Dst ParticlesConverter<src_layout_mde, src_alloc_mde, dst_layout_mde, dst_alloc_mde>::operator()(
    Src const& src, Layout const& layout)
{
    auto dst = make_particles<Dst>(layout);


    // not optimized but *should work*
    dst.reserve(dst.size() + src.size());
    if constexpr (src_layout_mde == LayoutMode::SoA) // no iteration, see Particles::begin
        for (std::size_t i = 0; i < src.size(); ++i)
            dst.push_back(src.copy(i));
    else
        std::copy(src.begin(), src.end(), std::back_inserter(dst));

    return dst;
}


} // namespace plamn::core

namespace plamn::core
{


template<typename Dst, typename Src, typename Layout>
auto static convert_particles_from(Src const& src, Layout const& layout)
{
    using Converter
        = ParticlesConverter<Src::layout_mode, Src::alloc_mode, Dst::layout_mode, Dst::alloc_mode>;

    std::string_view constexpr static FN_ID = "convert_particles,";
    auto constexpr function_id              = damn::core::join_string_views_v<FN_ID, Dst::type_id>;
    DAMN_LOG_SCOPE(1, function_id);

    return Converter{}.template operator()<Dst>(src, layout);
}

template<typename Dst, typename Src, typename Layout>
auto static convert_particles(Src const& src, Layout const& layout)
{
    if constexpr (std::is_same_v<Dst, Src>)
        return src;
    else
        return convert_particles_from<Dst>(src, layout);
}

template<typename Dst, typename Src, typename Layout>
auto static convert_particles_and_sort(Src const& src, Layout const& layout)
{
    std::string_view constexpr static FN_ID = "convert_particles_and_sort,";
    auto constexpr function_id              = damn::core::join_string_views_v<FN_ID, Dst::type_id>;
    DAMN_LOG_SCOPE(1, function_id);

    auto out = convert_particles<Dst>(src, layout);
    sort_particles(out, damn::core::grow(layout.cells, Layout::options.particle_ghost_width));
    return out;
}

template<auto layout_mode, auto O, typename Layout>
auto convert_to(Particles<O> const& src, Layout const& layout)
{
    using Parts = Particles<O.with_layout(layout_mode).with_storage(StorageMode::VECTOR)>;

    return convert_particles<Parts>(src, layout);
}

} // namespace plamn::core


#endif /* PLAMN_CORE_DATA_PARTICLES_CONVERTING_PARTICLES_CONVERTING */
