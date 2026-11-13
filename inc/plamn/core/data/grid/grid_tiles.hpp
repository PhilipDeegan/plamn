#ifndef PLAMN_CORE_DATA_GRID_GRID_TILES_HPP
#define PLAMN_CORE_DATA_GRID_GRID_TILES_HPP

#include "plamn/core/def.hpp"
#include "damn/core/data/tiles/tile_set.hpp"
#include "plamn/core/data/field/field_tiles.hpp"

#include <cmath>
#include <tuple>
#include <string>
#include <optional>
#include <stdexcept>

namespace plamn::core
{

template<bool hasGhosts = true>
auto static grid_cells(auto&&... args)
{
    auto const& [layout, qty] = std::forward_as_tuple(args...);

    if constexpr (hasGhosts)
        return layout.allocSize(qty);
    else
        return *(layout.AMRBox().shape().as_unsigned() - 1);
}

template<typename GridLayout_t, typename Grid_t, typename Field_t>
struct GridTile : public FieldTile<GridLayout_t, Field_t>
{
    auto constexpr static dimension = GridLayout_t::dimension;
    using Super                     = FieldTile<GridLayout_t, Field_t>;
    using value_type                = Grid_t::Super;
    using NdArray_t                 = Grid_t::Super;
    using Super::operator();

    GridTile(auto const& layout, auto const& pq)
        : Super{layout, Field_t{pq}}
        , arr{grid_cells(layout, pq)}
    {
        reset();
    }

    // called when making tiles
    GridTile(auto const& box, auto const& patch_layout, auto const& pq)
        : GridTile(patch_layout.copy_as(box), pq)
    {
    }

    GridTile(GridTile const& that)
        : Super{that.layout(), Field_t{(*that).physicalQuantity()}}
        , arr{that.arr}
    {
        reset();
    }

    GridTile(GridTile&&)                 = default;
    GridTile& operator=(GridTile const&) = delete;
    GridTile& operator=(GridTile&&)      = default;

    Super& operator*() DAMN_ALL { return *this; }
    Super const& operator*() const DAMN_ALL { return *this; }

    NO_DISCARD auto physicalQuantity() const DAMN_ALL { return (**this).physicalQuantity(); }

private:
    void reset() { (**this)() = Field_t{(**this).physicalQuantity(), arr.data(), arr.shape()}; }

    Super& super() { return *this; }
    Super const& super() const { return *this; }

    NdArray_t arr;
};

template<typename GridLayout_t, typename Grid_t, typename Field_t>
class GridTileSet : public damn::core::TileSet<GridTile<GridLayout_t, Grid_t, Field_t>, Grid_t::alloc_mode>,
                    public FieldTileSet<GridLayout_t, Grid_t, Field_t>

{
public:
    using grid_type  = Grid_t;
    using field_type = Field_t;
    using Super      = damn::core::TileSet<GridTile<GridLayout_t, Grid_t, Field_t>, Grid_t::alloc_mode>;
    using View       = FieldTileSet<GridLayout_t, Grid_t, Field_t>;
    using type       = Grid_t::value_type;
    using physical_quantity_type = Grid_t::physical_quantity_type;
    using value_type             = GridTile<GridLayout_t, Grid_t, Field_t>;
    using Super::operator[];
    using View::operator();
    using View::box;

    auto constexpr static alloc_mode = Grid_t::alloc_mode;
    auto constexpr static dimension  = Grid_t::dimension;

    GridTileSet(std::string const& name, GridLayout_t const& layout,
                physical_quantity_type const qty, std::optional<type> val = std::nullopt)
        : Super{layout.AMRBox(), layout, qty}
        , View{name, qty}
        , name_{name}
        , layout_{layout}
        , max_tile_size_{get_max_tile_size()}
    {
        check_tile_widths_();
        Super::build_links(super());
        assert(this->super()[0]().data());
        View::setBuffer(this);
        assert(View::isUsable());
        assert((**this)[0]().data());
        if (val)
            fill(*val);
        // ptr = &TileOverlaps_t::getOrCreateQuantity(layout_, *this);
    }

    GridTileSet(GridTileSet const& that)
        : Super{that.super()}
        , View{that}
        , name_{that.name_}
        , layout_{that.layout_}
        , max_tile_size_{get_max_tile_size()}
    {
        Super::build_links(super());
        View::setBuffer(this);
        assert((**this)[0]().data());
        assert(View::isUsable());
        // ptr = &TileOverlaps_t::getOrCreateQuantity(layout_, *this);
    }

    GridTileSet(GridTileSet&&) = default;

    GridTileSet& operator=(GridTileSet const&) = delete;
    GridTileSet& operator=(GridTileSet&&)      = delete;

    template<typename T>
    void fill(T const v)
    {
        for (auto& tile : *this)
            tile().fill(v);
    }

    auto& operator()() { return super()(); }
    auto& operator()() const { return super()(); }

    NO_DISCARD auto at(auto const&... args) { return super().at(args...); }
    NO_DISCARD auto at(auto const&... args) const { return super().at(args...); }

    View& operator*() { return *this; }
    View const& operator*() const { return *this; }

    auto data() { return super().data(); }
    auto data() const { return super().data(); }
    NO_DISCARD auto begin() { return super().begin(); }
    NO_DISCARD auto begin() const { return super().begin(); }
    NO_DISCARD auto end() { return super().end(); }
    NO_DISCARD auto end() const { return super().end(); }
    NO_DISCARD auto size() const DAMN_ALL { return View::size(); }
    auto& layout() const { return layout_; }

    // patch layout and each tile's sub layout from a corrected patch layout (restarts)
    void reset_layout(GridLayout_t const& layout)
    {
        layout_ = layout;
        for (auto& tile : *this)
            tile.reset_layout(layout.copy_as(tile.layout().AMRBox()));
    }

    Super& super() { return *this; }
    Super const& super() const { return *this; }

    NO_DISCARD auto& name() const { return name_; }
    auto max_tile_size() const { return max_tile_size_; }

private:
    // a tile's halo is refreshed from its direct neighbours only (traverse_tile_neighbours), so a
    // tile with a neighbour in a direction must be at least field_ghost_width wide in it, or the
    // halo of the next tile over reaches past it. The default tiling options satisfy this, small
    // DAMN_TILING_* env values may not
    void check_tile_widths_() const
    {
        auto constexpr ghost_width = GridLayout_t::options.field_ghost_width;
        auto const patch_shape     = layout_.AMRBox().shape();
        for (auto const& tile : super()())
        {
            auto const tile_shape = static_cast<damn::core::Box<int, dimension>>(tile).shape();
            for (std::size_t d = 0; d < dimension; ++d)
                if (tile_shape[d] < patch_shape[d]
                    and static_cast<std::size_t>(tile_shape[d]) < ghost_width)
                    throw std::runtime_error(
                        "GridTileSet: tile width " + std::to_string(tile_shape[d])
                        + " < field ghost width " + std::to_string(ghost_width)
                        + ", increase DAMN_TILING_MIN_TILE_SIZE/DAMN_TILING_MAX_TILE_SIZE");
        }
    }

    auto get_max_tile_size() const
    {
        std::uint32_t size = 0;
        for (auto const& tile : super()())
            if (std::uint32_t tile_size = damn::core::product(tile.ghost_box().shape()); tile_size > size)
                size = tile_size;
        return size;
    }

    std::string name_;
    GridLayout_t layout_;
    std::uint32_t max_tile_size_;

    // FieldTileOverlaps<field_opts>::Level::Patch* ptr = nullptr;
};



template<typename GridLayout_t, typename Grid_t, typename Field_t>
inline std::ostream& operator<<(std::ostream& out,
                                FieldTileSet<GridLayout_t, Grid_t, Field_t> const& ts)
{
    for (auto const& tile : ts())
        out << tile();

    return out;
}


template<typename GridLayout_t, typename Grid_t, typename Field_t>
inline auto sum_field(FieldTileSet<GridLayout_t, Grid_t, Field_t> const& ts)
{
    return damn::core::sum_from(ts(), [](auto const& tile) { return sum(tile()); });
}

template<typename GridLayout_t, typename Grid_t, typename Field_t>
inline auto sum_not_nan(FieldTileSet<GridLayout_t, Grid_t, Field_t> const& ts)
{
    return damn::core::sum_from(ts(), [](auto const& tile) {
        typename Grid_t::value_type s = 0;
        for (auto const& v : tile())
            if (!std::isnan(v))
                s += v;
        return s;
    });
}


template<typename GridLayout_t, typename Grid_t, typename Field_t>
void check_field(basic::FieldTileSet<GridLayout_t, Grid_t, Field_t> const& f,
                 GridLayout_t const& layout)
{
#if DAMN_DEBUG
    auto const domainBox = layout.AMRBoxFor(f);

    for (auto& tile : f())
        if (auto const overlap = **tile * domainBox)
            damn::core::for_each_point(tile.layout().AMRToLocal(*overlap),
                           [&](auto const& bix) { assert(not std::isnan(tile()(bix))); });
#endif // DAMN_DEBUG
}

template<typename GridLayout_t, typename Grid_t, typename Field_t>
void check_field(basic::FieldTileSet<GridLayout_t, Grid_t, Field_t> const& f)
{
#if DAMN_DEBUG
    for (auto& tile : f())
        check_field(tile());
#endif // DAMN_DEBUG
}


} // namespace plamn::core

#endif // PLAMN_CORE_DATA_GRID_GRID_TILES_HPP
