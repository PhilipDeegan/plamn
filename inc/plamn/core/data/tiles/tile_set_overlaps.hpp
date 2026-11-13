#ifndef PLAMN_CORE_DATA_TILES_TILE_SET_OVERLAPS_HPP
#define PLAMN_CORE_DATA_TILES_TILE_SET_OVERLAPS_HPP

// damn's tile overlap maps, with ghost boxes from a quantity's centering or from the
// grid layout's particle ghost width

#include "plamn/core/def.hpp"
#include "damn/core/data/tiles/tile_set_overlaps.hpp"


namespace plamn::core
{

template<typename Layout_t>
auto make_nd_span_set_for_qty(Layout_t const& layout, auto const pq)
{
    return damn::core::make_nd_span_set_from(layout,
                                 [&](auto const& layout) { return ghost_box(layout, pq); });
}


template<auto quantity, typename TileSet_t, typename Layout_t>
auto make_qty_nd_span_set_from(TileSet_t& tiles, Layout_t const& layout)
{
    return damn::core::make_nd_span_set_from(tiles, layout, [&](auto const& box) {
        return amr_ghost_box(sub_layout(layout, box), quantity);
    });
}

template<typename TileSet_t, typename Layout_t>
auto make_qty_nd_span_set_from(TileSet_t& tiles, Layout_t const& layout, auto const qty)
{
    return damn::core::make_nd_span_set_from(
        tiles, layout, [&](auto const& box) { return amr_ghost_box(sub_layout(layout, box), qty); });
}


template<typename TileSet_t, typename Layout_t>
auto make_particle_nd_span_set_from(TileSet_t& tiles, Layout_t const& layout)
{
    return damn::core::make_nd_span_set_from(tiles, layout, [](auto const& box) {
        return damn::core::grow(box, Layout_t::options.particle_ghost_width);
    });
}

} // namespace plamn::core


#endif /* PLAMN_CORE_DATA_TILES_TILE_SET_OVERLAPS_HPP */
