#ifndef PLAMN_CORE_DATA_PARTICLES_APPENDING_PARTICLES_APPENDING
#define PLAMN_CORE_DATA_PARTICLES_APPENDING_PARTICLES_APPENDING

#include "damn/core/logger.hpp"
#include "damn/core/sys/strings.hpp"
#include "damn/core/geometry/box_spans.hpp"

#include "plamn/core/def.hpp"
#include "plamn/core/data/particles/appending/detail/def_appending.hpp"
#include "plamn/core/data/particles/appending/detail/aos_appending.hpp"
#include "plamn/core/data/particles/appending/detail/soa_appending.hpp"
#include "plamn/core/data/particles/appending/detail/soavx_chunked_appending.hpp"
#include "plamn/core/data/particles/particles_def.hpp"

#include <cassert>
#include <string_view>
#include <iterator>
#include <cstddef>

namespace plamn::core
{

// generic fallthrough
// template<auto src_layout_mde, auto src_alloc_mde, auto dst_layout_mde, auto dst_alloc_mde>
// template<auto type, typename Src, typename Dst>
// void ParticlesAppender<src_layout_mde, src_alloc_mde, dst_layout_mde, dst_alloc_mde>::operator()(
//     Src const& src, Dst& dst)
// {
//     // not optimized but *should work*
//     // if it doesn't, make a new impl

//     // assert(dst.capacity() > dst.size() + src.size());
//     dst.reserve(dst.size() + src.size()); // BAD!
//     std::copy(src.begin(), src.end(), std::back_inserter(dst));
// }


} // namespace plamn::core

namespace plamn::core
{
template<auto src_layout_mde, auto src_alloc_mde, auto dst_layout_mde, auto dst_alloc_mde>
struct ParticlesAppender;


// Deals with one box at a time - no ParticleType, no domain/ghost distinction, so callers
// passing overlapping boxes across multiple calls will over-reserve the overlap. Resolves
// cells through the container's own local_cell/local_tile_cell (not a box*tile guess), so
// ghost-layer cells clamp-owned by a border tile (see TileSet::tag_cells_) land correctly.
template<typename Dst, typename Box_t>
void reserve(Dst& dst, Box_t const& box, std::size_t const& ppc)
{
    using enum LayoutMode;

    if constexpr (Dst::layout_mode == AoSPC)
    {
        // PerCellVector exposes indexed per-cell access directly, ghost cells included
        // (its NdArray spans the whole ghost_box, no tiling to worry about).
        for_each_point(box, [&](auto const& bix) { dst(dst.local_cell(*bix)).reserve(ppc); });
    }
    else if constexpr (Dst::layout_mode == AoSPCTS)
    {
        // tiles are per-cell (PerCellVector) here, not flat - no tile-level reserve exists,
        // reserve each cell in the overlap directly instead. Approximate: box*tile misses
        // ghost cells clamp-owned outside a tile's own box, but reserve is just a hint.
        for (auto& tile : dst())
            if (auto const overlap = box * tile)
                damn::core::for_each_point(tile().local_box(*overlap),
                               [&](auto const& bix) { tile()(bix).reserve(ppc); });
    }
    else if constexpr (Dst::layout_mode == AoSTS or Dst::layout_mode == AoSCMTS)
    {
        // one flat vector per tile. Approximate: box*tile misses ghost cells clamp-owned
        // outside a tile's own box, but reserve is just a hint
        for (auto& tile : dst())
            if (auto const overlap = box * tile)
                tile().reserve(tile().size() + ppc * overlap->size());
    }
    else
    {
        dst.reserve(ppc * box.size());
    }
}


// reserves box, then for every cell in box emplaces ppc particles built by fn(icell, id) -
// id is a running counter across the whole box, not reset per cell. emplace_back already
// resolves the owning cell/tile per particle (including clamp-owned ghost cells for tiled
// layouts, via the same particles_.at() lookup the span side uses) - no faster path exists
// on the vector side for tiled layouts, which expose that lookup only through their span.
template<auto type, typename Dst, typename Box_t>
void add_particles(Dst& dst, Box_t const& box, std::size_t const& ppc, auto&& fn)
{
    reserve(dst, box, ppc);

    std::size_t id = dst.size();
    damn::core::for_each_point(box, [&](auto const& bix) {
        for (std::size_t i = 0; i < ppc; ++i)
            dst.emplace_back(fn(*bix, id++));
    });

    dst.template on_appended<type>();
}


template<auto type, typename Src, typename Dst>
void append_particles(Src const& src, Dst& dst)
{
    using Appending
        = ParticlesAppender<Src::layout_mode, Src::alloc_mode, Dst::layout_mode, Dst::alloc_mode>;

    DAMN_DEBUG_DO(int const old_size = dst.size();)

    std::string_view constexpr static FN_ID     = "append_particles,";
    [[maybe_unused]] auto constexpr function_id = damn::core::join_string_views_v<FN_ID, Dst::type_id>;
    DAMN_LOG_SCOPE(3, function_id);

    Appending{0, src.size()}.template operator()<type>(src, dst);

    assert(dst.size() == old_size + src.size());
}


} // namespace plamn::core


#endif /* PLAMN_CORE_DATA_PARTICLES_APPENDING_PARTICLES_APPENDING */
