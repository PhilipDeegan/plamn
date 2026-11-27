#ifndef PLAMN_CORE_DATA_FIELD_BOX_COPIES_HPP
#define PLAMN_CORE_DATA_FIELD_BOX_COPIES_HPP

// box copies between fields of different shapes (tiles <-> untiled temporaries), collected over a
//  level and run at once: doc/core/data/field/box_copies.md

#include "damn/core/def/damn_config.hpp"
#include "damn/core/exec/kernels.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/operators.hpp"
#include "damn/core/vector.hpp"
#include "plamn/core/data/field/field_box.hpp"
#include "plamn/core/data/field/field_tiles.hpp"
#include "plamn/core/data/field/level_fields.hpp"
#include "mkn/kul/std/array.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <type_traits>

namespace plamn::core
{

// dst[dst_lo + p] = (or +=) src[src_lo + p] for p in [0, extent), both C ordered
template<std::size_t dim, typename T>
struct BoxCopy
{
    T* dst;
    T const* src;
    std::array<std::uint32_t, dim> dst_shape, src_shape, dst_lo, src_lo, extent;
    std::size_t first; // its first value in the run's range
    bool add;
};

template<auto alloc_mode, std::size_t dim, typename T = double>
class BoxCopies
{
    using Box_t = damn::core::Box<std::uint32_t, dim>;

public:
    // zeroed at the start of the next run, before any copy
    void zero(auto& field) { zeros_.add(field); }

    void add(auto& dst, Box_t const& dst_box, auto const& src, Box_t const& src_box,
             bool const sum = false)
    {
        static_assert(detail::is_c_ordered_v<std::decay_t<decltype(dst)>>);
        static_assert(detail::is_c_ordered_v<std::decay_t<decltype(src)>>);
        assert(dst_box.shape() == src_box.shape());
        auto const size = dst_box.size();
        if (size == 0)
            return;
        BoxCopy<dim, T> c{dst.data(), src.data(), {}, {}, {}, {}, {}, total_, sum};
        auto const ds = dst.shape();
        auto const ss = src.shape();
        auto const ex = dst_box.shape();
        for (std::size_t d = 0; d < dim; ++d)
        {
            c.dst_shape[d] = ds[d];
            c.src_shape[d] = ss[d];
            c.dst_lo[d]    = dst_box.lower[d];
            c.src_lo[d]    = src_box.lower[d];
            c.extent[d]    = ex[d];
        }
        items_.push_back(c);
        total_ += size;
    }

    void run()
    {
        zeros_.zero();
        zeros_.clear();
        if (total_ > 0)
            run_();
        items_.clear();
        total_ = 0;
    }

private:
    struct At
    {
        T* dst;
        T const* src;
        bool add;
    };

    // value i of the range: the last item whose first <= i, then its point in C order
    static At at_(BoxCopy<dim, T> const* const items, std::size_t const n,
                  std::size_t const i) DAMN_ALL
    {
        std::size_t lo = 0, hi = n;
        while (hi - lo > 1)
            if (auto const mid = (lo + hi) / 2; items[mid].first <= i)
                lo = mid;
            else
                hi = mid;
        auto const& c = items[lo];
        std::array<std::uint32_t, dim> p;
        std::size_t j = i - c.first;
        for (std::size_t d = dim; d-- > 0;)
        {
            p[d] = static_cast<std::uint32_t>(j % c.extent[d]);
            j /= c.extent[d];
        }
        std::size_t dst_at = 0, src_at = 0;
        for (std::size_t d = 0; d < dim; ++d)
        {
            dst_at = dst_at * c.dst_shape[d] + c.dst_lo[d] + p[d];
            src_at = src_at * c.src_shape[d] + c.src_lo[d] + p[d];
        }
        return {c.dst + dst_at, c.src + src_at, c.add};
    }

    // one launch over every item's values; sums overlap (tile ghost boxes): atomic
    void run_()
        requires(alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
    {
        auto const* const items = items_.data();
        auto const n            = items_.size();
        damn::kernel::launch(total_, [=] DAMN_ALL() {
            auto const v = at_(items, n, damn::kernel::idx());
            if (T& dst = *v.dst; v.add)
                damn::core::Operators<T, true, true>{dst} += *v.src;
            else
                dst = *v.src;
        });
    }

    // a span (the last dimension) at a time, items in order
    void run_()
        requires(alloc_mode == damn::AllocatorMode::CPU)
    {
        for (auto const& c : items_)
        {
            auto const span    = c.extent[dim - 1];
            std::size_t n_rows = 1;
            for (std::size_t d = 0; d + 1 < dim; ++d)
                n_rows *= c.extent[d];
            for (std::size_t r = 0; r < n_rows; ++r)
            {
                auto const v = at_(&c, 1, c.first + r * span);
                if (c.add)
                    for (std::uint32_t k = 0; k < span; ++k)
                        v.dst[k] += v.src[k];
                else
                    std::copy_n(v.src, span, v.dst);
            }
        }
    }

    typename damn::Vector<BoxCopy<dim, T>, alloc_mode>::vector_t items_
        = damn::Vector<BoxCopy<dim, T>, alloc_mode>::make(0);
    std::size_t total_ = 0;
    LevelFields<alloc_mode, T> zeros_;
};

// the copies of a thread, kept between runs (no allocation once large enough)
template<auto alloc_mode, std::size_t dim, typename T = double>
BoxCopies<alloc_mode, dim, T>& box_copies()
{
    static thread_local BoxCopies<alloc_mode, dim, T> copies;
    return copies;
}

// tmp's every value from the tile that owns it (border tiles out to the patch ghost box): the
//  tiles' owned boxes partition tmp, nothing zeroed
template<typename Tiles>
void owned_into(auto& copies, auto& tmp, Tiles const& tiles)
    requires(is_field_tile_set_v<Tiles>)
{
    if (tiles().size() == 0)
        return;
    auto const pq           = tiles.physicalQuantity();
    auto const patch_layout = sub_layout(tiles[0].layout(), tiles.box());
    auto const ghost_box    = qty_ghost_box(patch_layout, pq);
    auto const expander     = BoxExpander{ghost_box, patch_layout.cells};
    assert(mkn::kul::array_equals(tiles.shape(), tmp.shape()));

    for (auto const& tile : tiles())
    {
        auto const tile_gb              = qty_ghost_box(tile.layout(), pq);
        auto const is_border            = *(tile_gb * patch_layout.cells) != tile_gb;
        decltype(tile_gb) const& tile_b = *tile;
        auto const box                  = is_border ? expander(tile_b) : tile_b;
        copies.add(tmp, damn::core::as_unsigned(shift(box, ghost_box.lower * -1)), tile(),
                   damn::core::as_unsigned(shift(box, tile_gb.lower * -1)));
    }
}

// tmp zeroed, then every tile's ghost box added (their partial ghost contributions included)
template<typename Tiles>
void summed_into(auto& copies, auto& tmp, Tiles const& tiles)
    requires(is_field_tile_set_v<Tiles>)
{
    copies.zero(tmp);
    if (tiles().size() == 0)
        return;
    auto const pq           = tiles.physicalQuantity();
    auto const patch_layout = sub_layout(tiles[0].layout(), tiles.box());
    assert(mkn::kul::array_equals(tiles.shape(), tmp.shape()));

    for (auto const& tile : tiles())
    {
        auto const tile_gb = qty_ghost_box(tile.layout(), pq);
        copies.add(tmp, to_local(patch_layout, tile_gb), tile(), to_local(tile.layout(), tile_gb),
                   /*sum=*/true);
    }
}

// every tile's ghost box (outer and inner ghosts) from tmp
template<typename Tiles>
void into_tile_ghosts(auto& copies, Tiles& tiles, auto const& tmp)
    requires(is_field_tile_set_v<Tiles>)
{
    if (tiles().size() == 0)
        return;
    auto const patch_layout = sub_layout(tiles[0].layout(), tiles.box());
    for (auto& tile : tiles())
        copies.add(tile(), to_local(tile.layout(), tile.ghost_box()), tmp,
                   to_local(patch_layout, tile.ghost_box()));
}

} // namespace plamn::core

#endif // PLAMN_CORE_DATA_FIELD_BOX_COPIES_HPP
