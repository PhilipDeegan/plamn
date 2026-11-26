#ifndef PLAMN_CORE_DATA_FIELD_FIELD_TILES_HPP
#define PLAMN_CORE_DATA_FIELD_FIELD_TILES_HPP

#include "damn/core/vector.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/data/tiles/tile_set.hpp"
#include "damn/core/data/tiles/tile_set_traversal.hpp"
#include "mkn/kul/std/array.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/def/damn_config.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/memory/span.hpp"
#include "damn/core/memory/memory.hpp"
#include "plamn/core/def.hpp"

#include <tuple>
#include <utility>
#include <optional>
#include <cassert>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <type_traits>
#include <array>
#include <cstdlib>
#include <iosfwd>
#include <version>

// defined in tensorfield.hpp, which includes this file
namespace plamn::core::basic
{
template<typename Field_t, std::size_t rank>
struct TensorField;
} // namespace plamn::core::basic

namespace plamn::core
{
namespace detail
{
    template<typename T>
    constexpr bool is_c_ordered_v = [] {
        if constexpr (requires { T::is_c_ordered; })
            return T::is_c_ordered;
        else
            return false;
    }();
} // namespace detail

// Operator{dst}(src) over two equally shaped local boxes, one contiguous span at a time rather
// than an ndarray lookup per index
template<typename Operator, typename Dst, typename Src, std::size_t dim>
void operate_on_spans(Dst& dst, damn::core::Box<std::uint32_t, dim> const& dst_lcl_box, Src const& src,
                      damn::core::Box<std::uint32_t, dim> const& src_lcl_box)
{
    using value_type = Dst::value_type;

    auto const on_span = [](auto* d, auto const* s, std::uint32_t const size) {
        if constexpr (requires { Operator::span(d, s, size); })
            Operator::span(d, s, size);
        else if constexpr (std::is_same_v<Operator, mkn::kul::SetEqual<value_type>>)
            std::copy_n(s, size, d);
        else
            for (std::uint32_t i = 0; i < size; ++i)
                Operator{d[i]}(s[i]);
    };

    if constexpr (dim > 1 and detail::is_c_ordered_v<Dst> and detail::is_c_ordered_v<Src>)
    {
        // thin ghost slabs have spans of a few values: index the two corners once, then step
        // the pointers by each array's own row/plane strides instead of an ndarray lookup
        // (both points) per span
        auto const end = dst_lcl_box.upper + 1;
        auto const& lo = dst_lcl_box.lower;
        for (std::size_t d = 0; d < dim; ++d)
            if (end[d] <= lo[d])
                return;

        std::uint32_t const size = end[dim - 1] - lo[dim - 1];
        auto const& dshape       = dst.shape();
        auto const& sshape       = src.shape();
        auto* d0                 = &dst(dst_lcl_box.lower);
        auto const* s0           = &src(src_lcl_box.lower);

        if constexpr (dim == 2)
            for (std::uint32_t x = 0; x < end[0] - lo[0]; ++x)
                on_span(d0 + x * dshape[1], s0 + x * sshape[1], size);

        if constexpr (dim == 3)
        {
            std::size_t const d_plane = std::size_t{dshape[1]} * dshape[2];
            std::size_t const s_plane = std::size_t{sshape[1]} * sshape[2];
            for (std::uint32_t x = 0; x < end[0] - lo[0]; ++x)
                for (std::uint32_t y = 0; y < end[1] - lo[1]; ++y)
                    on_span(d0 + x * d_plane + y * dshape[2], s0 + x * s_plane + y * sshape[2],
                            size);
        }
    }
    else
        damn::core::for_each_span(dst_lcl_box, src_lcl_box,
                      [&](auto const& dst_start, auto const& src_start, std::uint32_t const size) {
                          on_span(&dst(dst_start), &src(src_start), size);
                      });
}

template<typename Layout_t, typename Grid_t, typename Field_t>
struct GridTile;

template<typename Layout_t, typename Field_t>
class FieldTile : public damn::core::Box<std::int32_t, Layout_t::dimension>
{
public:
    auto constexpr static dimension    = Layout_t::dimension;
    auto constexpr static alloc_mode   = Field_t::alloc_mode;
    auto constexpr static is_c_ordered = Field_t::is_c_ordered;
    using Super                        = damn::core::Box<std::int32_t, dimension>;
    using value_type                 = Field_t;
    using type                       = Field_t::type;

    template<typename... Args>
    FieldTile(Layout_t const& layout, Field_t const& field)
        : Super{layout.cells}
        , field_{field}
        , layout_{layout}
    {
    }

    auto& operator()() DAMN_ALL { return field_; }
    auto& operator()() const DAMN_ALL { return field_; }
    Super& operator*() DAMN_ALL { return *this; }
    Super const& operator*() const DAMN_ALL { return *this; }
    auto& layout() const DAMN_ALL { return layout_; }

    // same box, other mesh size/origin
    void reset_layout(Layout_t const& layout)
    {
        assert(layout.cells == layout_.cells);
        layout_ = layout;
    }

    auto ghost_box() const DAMN_ALL { return ghost_box_; }
    auto field_box() const DAMN_ALL
    {
        return damn::core::shrink(ghost_box(), Layout_t::options.field_ghost_width);
    }

    template<template<typename, std::size_t> typename Point_t>
    auto& operator()(Point_t<std::uint32_t, dimension> const& point) DAMN_ALL
    {
        return field_(point);
    }
    template<template<typename, std::size_t> typename Point_t>
    auto& operator()(Point_t<std::uint32_t, dimension> const& point) const DAMN_ALL
    {
        return field_(point);
    }
    template<typename... IJK>
    auto& operator()(IJK const&... ijk)
        requires(sizeof...(IJK) == dimension)
    DAMN_ALL
    {
        return field_(damn::core::to_point<std::uint32_t>(ijk...));
    }
    template<typename... IJK>
    auto& operator()(IJK const&... ijk) const
        requires(sizeof...(IJK) == dimension)
    DAMN_ALL
    {
        return field_(damn::core::to_point<std::uint32_t>(ijk...));
    }

    MKN_KUL_FN_SIG_RET auto physicalQuantity() const DAMN_ALL { return field_.physicalQuantity(); }

    bool isUsable() const { return field_.isUsable(); }
    bool isSettable() const { return !isUsable(); }

    auto& links() { return _links; }
    auto& links() const { return _links; }
    auto& link(std::size_t const idx) { return _links[idx]; }

private:
    Field_t field_;
    Layout_t layout_;
    damn::core::Box<int, dimension> ghost_box_ = qty_ghost_box(layout_, physicalQuantity());
    std::array<FieldTile*, 7> _links           = mkn::kul::ConstArray<FieldTile*, 7>(nullptr);
};

template<typename Layout_t, typename Grid_t, typename Field_t>
struct FieldTileSetTypes
{
    using Tile_t     = GridTile<Layout_t, Grid_t, Field_t>;
    using Tile_vt    = FieldTile<Layout_t, Field_t>;
    using Span_t     = damn::core::ViewSpan<Tile_vt, Tile_t>;
    using Span_pt    = damn::core::ViewSpan<Tile_vt*, Tile_t*>;
    using value_type = damn::core::TileSetView<Tile_vt, Span_t, Span_pt>;
};

} // namespace plamn::core

namespace plamn::core::basic
{
template<typename Layout_, typename Grid_t, typename Field_t>
class FieldTileSet : public FieldTileSetTypes<Layout_, Grid_t, Field_t>::value_type
{
    using local_types = FieldTileSetTypes<Layout_, Grid_t, Field_t>;

public:
    using Layout_t                   = Layout_;
    using grid_type                  = Grid_t;
    using field_type                 = Field_t;
    using Super                      = local_types::value_type;
    using value_type                 = local_types::Tile_vt;
    using physical_quantity_type     = Grid_t::physical_quantity_type;
    using type                       = Grid_t::type;
    auto constexpr static dimension  = Layout_t::dimension;
    auto constexpr static alloc_mode = Grid_t::alloc_mode;

    FieldTileSet(auto physicalQuantity) DAMN_ALL : Super{{}, nullptr, 0, nullptr, {}},
                                                         qty_{physicalQuantity}
    {
    }

    FieldTileSet(FieldTileSet const&) DAMN_ALL            = default;
    FieldTileSet(FieldTileSet&&) DAMN_ALL                 = delete;
    FieldTileSet& operator=(FieldTileSet const&) DAMN_ALL = delete;
    FieldTileSet& operator=(FieldTileSet&&) DAMN_ALL      = delete;

    void setBuffer(std::nullptr_t /*ptr*/) DAMN_ALL
    {
        super() = Super{{}, nullptr, 0, nullptr, {}};
    }

    template<typename FieldLike>
    void setBuffer(FieldLike* const field) DAMN_ALL
    {
        auto data = field ? field->data() : nullptr;
        if (data)
        {
            super()        = field->super().as([](auto&&... args) mutable {
                auto&& [box, tiles_data, n_tiles, cells_data, cells_shape]
                    = std::forward_as_tuple(args...);
                assert(cells_data);
                return Super{box, &*tiles_data[0], n_tiles,
                             reinterpret_cast<FieldTile<Layout_t, Field_t>**>(cells_data),
                             cells_shape};
            });
            ghost_box_     = qty_ghost_box(field->layout(), qty_);
            max_tile_size_ = field->max_tile_size();
        }
        else
            super() = Super{{}, nullptr, 0, nullptr, {}};
    }

    void zero()
    {
        if (!Super::data())
            throw std::runtime_error("invalid state");

        using vec_helper = damn::Vector<type, alloc_mode>;
        for (auto& tile : *this)
            vec_helper::fill(tile().data(), tile().size(), 0);
    }

    void fill(auto const v)
    {
        if (!Super::data())
            throw std::runtime_error("invalid state");

        using vec_helper = damn::Vector<type, alloc_mode>;
        for (auto& tile : *this)
            vec_helper::fill(tile().data(), tile().size(), v);
    }

    auto& operator()() DAMN_ALL
    {
        DAMN_ASSERT(isUsable());
        return super()();
    }
    auto& operator()() const DAMN_ALL
    {
        DAMN_ASSERT(isUsable());
        return super()();
    }

    void setData(auto* const /*data*/) DAMN_ALL
    {
        throw std::runtime_error("fix?");
        /*Super::setBuffer(data);*/
    }

    bool isUsable() const { return Super::data() != nullptr; }
    bool isSettable() const { return !isUsable(); }

    // tile by tile, the allocator's copy (on the device for unified memory)
    void copyData(FieldTileSet const& that)
    {
        for (std::size_t tidx = 0; tidx < super().size(); ++tidx)
            damn::core::mem::copy<alloc_mode>(super()[tidx]().data(), that[tidx]().data(),
                                              that[tidx]().size());
    }
    MKN_KUL_FN_SIG_RET auto size() const DAMN_ALL { return ghost_box_.size(); } // NOT ntiles!
    MKN_KUL_FN_SIG_RET auto shape() const DAMN_ALL { return *ghost_box_.shape().as_unsigned(); }
    MKN_KUL_FN_SIG_RET auto ghost_box() const DAMN_ALL { return ghost_box_; }

    MKN_KUL_FN_SIG_RET auto at(auto const&... args) { return super().at(args...); }
    MKN_KUL_FN_SIG_RET auto at(auto const&... args) const { return super().at(args...); }

    void check(bool const /*nan*/ = false) const
    {
        // for (auto& tile : super())
        // {
        //     assert(tile().size() < static_cast<std::size_t>(1e6));
        //     tile().check();
        // }
    }

    void sync_inner_ghosts()
    {
        damn::core::visit_tile_neighbours(super(), [](auto& t0, auto& t1) { sync_ghosts_from_(t0, t1); });
    }

    // only writes the ghosts of tile `tile_idx` (reading its neighbours' domains), so distinct
    // tiles can be synced concurrently
    void sync_inner_ghosts(std::size_t const tile_idx)
    {
        for_each_inner_ghost_overlap(
            tile_idx, [](auto& t0, auto const& t1, auto const& dst, auto const& src) {
                operate_on_spans<mkn::kul::SetEqual<typename Field_t::value_type>>(t0(), dst, t1(),
                                                                                   src);
            });
    }

    // fn(t0, t1, dst, src) for each neighbour t1 of tile `tile_idx` (t0) whose field box t0's
    //  ghosts overlap: dst / src the overlap as local boxes of t0 / t1
    void for_each_inner_ghost_overlap(std::size_t const tile_idx, auto&& fn)
    {
        auto& t0 = super()[tile_idx];
        damn::core::traverse_tile_neighbours(super(), t0, [&](auto& t1) {
            if (auto const boxes = inner_ghost_overlap_(t0, t1))
                fn(t0, t1, boxes->first, boxes->second);
        });
    }


    void notZero() const
    {
        for (auto& tile : super())
            for (auto const& e : tile())
                if (std::abs(e) < 1e-15)
                    throw std::runtime_error("ZERO");
    }


    auto max_tile_size() const { return max_tile_size_; }
    auto ntiles() const { return Super::size(); }

    template<typename, typename, typename>
    friend std::ostream& operator<<(std::ostream&, FieldTileSet const&);

    MKN_KUL_FN_SIG_RET auto physicalQuantity() const DAMN_ALL { return qty_; }

private:
    Super& super() { return *this; }
    Super const& super() const { return *this; }

    // the overlap stops short of t0's own field_box in each direction t1 lies in (the opposite
    //  of BoxExpander): shared primal wall nodes are computed by both tiles, copying them would
    //  race with t1 being synced concurrently from t0. Nodes this drops off edge/corner overlaps
    //  outside t0's field_box come from the face neighbours, tiles being a tensor product
    static void sync_ghosts_from_(auto& t0, auto const& t1)
    {
        if (auto const boxes = inner_ghost_overlap_(t0, t1))
            operate_on_spans<mkn::kul::SetEqual<typename Field_t::value_type>>(t0(), boxes->first,
                                                                               t1(), boxes->second);
    }

    // as local boxes of t0 (dst) and t1 (src)
    static auto inner_ghost_overlap_(auto const& t0, auto const& t1)
    {
        using Box_t   = decltype(damn::core::as_unsigned(shift(t0.ghost_box(), t0.ghost_box().lower * -1)));
        using Boxes_t = std::optional<std::pair<Box_t, Box_t>>;

        auto const t0_fb = t0.field_box();
        auto const t1_fb = t1.field_box();
        auto overlap     = t0.ghost_box() * t1_fb;
        if (!overlap)
            return Boxes_t{};

        for (std::size_t d = 0; d < dimension; ++d)
            if (t1_fb.lower[d] > t0_fb.lower[d])
                overlap->lower[d] = std::max(overlap->lower[d], t0_fb.upper[d] + 1);
            else if (t1_fb.lower[d] < t0_fb.lower[d])
                overlap->upper[d] = std::min(overlap->upper[d], t0_fb.lower[d] - 1);

        for (std::size_t d = 0; d < dimension; ++d)
            if (overlap->lower[d] > overlap->upper[d])
                return Boxes_t{};

        return Boxes_t{std::pair{damn::core::as_unsigned(shift(*overlap, t0.ghost_box().lower * -1)),
                                 damn::core::as_unsigned(shift(*overlap, t1.ghost_box().lower * -1))}};
    }

    physical_quantity_type qty_;
    damn::core::Box<int, dimension> ghost_box_{};
    std::uint32_t max_tile_size_;
};


} // namespace plamn::core::basic

namespace plamn::core
{

template<typename Layout_t, typename Grid_t, typename Field_t>
class FieldTileSet : public basic::FieldTileSet<Layout_t, Grid_t, Field_t>
{
public:
    using Super = basic::FieldTileSet<Layout_t, Grid_t, Field_t>;

    FieldTileSet(std::string const& name, auto physicalQuantity) DAMN_ALL
        : Super{physicalQuantity},
          name_{name}
    {
    }

    MKN_KUL_FN_SIG_RET auto& name() const { return name_; }

private:
    Super& super() { return *this; }
    Super const& super() const { return *this; }

    std::string const name_;
};


template<typename T>
concept is_field_tile_set_c = requires(T* p) {
    []<typename GL, typename Arr, typename PQ>(basic::FieldTileSet<GL, Arr, PQ> const*) {}(p);
};

template<typename T>
inline constexpr bool is_field_tile_set_v = is_field_tile_set_c<T>;

// a tile set, whatever its cv/ref
template<typename T>
concept TiledField = is_field_tile_set_v<std::decay_t<T>>;

template<typename T>
concept has_tiled_field_type_c = requires {
    typename std::remove_cvref_t<T>::field_type;
} and is_field_tile_set_v<typename std::remove_cvref_t<T>::field_type>;

// structs of (vec)fields naming their view type over a given vecfield type, tiled through their
// as<V>(fn) member
template<typename T>
concept has_view_t_c = requires {
    typename std::remove_cvref_t<T>::vecfield_type;
    typename std::remove_cvref_t<T>::template view_t<
        typename std::remove_cvref_t<T>::vecfield_type>;
};

namespace basic
{
    template<typename VecFieldT>
    class Electromag; // IWYU pragma: keep (electromag.hpp)
} // namespace basic

// a (vec)field as plain device copyable views (basic:: types, no names): an untiled one, or a
//  tiled one as its tile set view (basic::FieldTileSet: pointers to its tiles)
template<typename T>
auto untiled_view(T&& untiled)
{
    using T_t = std::remove_cvref_t<T>;
    if constexpr (requires {
                      typename T_t::vecfield_type;
                      untiled.E;
                      untiled.B;
                  }) // an electromag: its vecfields' views (no name() of its own)
    {
        using V_t = decltype(untiled_view(untiled.E));
        return basic::Electromag<V_t>{untiled_view(untiled.E), untiled_view(untiled.B)};
    }
    else if constexpr (not requires { untiled.name(); }) // already a basic:: view
        return T_t{untiled};
    else if constexpr (requires { T_t::rank; }) // a tensor field
    {
        using Field_t = typename T_t::field_type::Super;
        using V_t     = basic::TensorField<Field_t, T_t::rank>;
        return untiled.template as<V_t>(
            [](auto& c) { return Field_t{static_cast<Field_t const&>(c)}; });
    }
    else // a field: its view base (a cast: a tile set's base also takes a quantity)
        return typename T_t::Super{static_cast<typename T_t::Super const&>(untiled)};
}

template<typename T>
auto tile_at(T&& tiled, std::size_t const i)
{
    using Tiled_t = std::remove_cvref_t<T>;
    if constexpr (is_field_tile_set_v<Tiled_t>)
        return tiled()[i]();
    else if constexpr (has_view_t_c<Tiled_t>) // before has_tiled_field_type_c, which it may match
    {
        using VecField_t = Tiled_t::vecfield_type;
        using TileVF_t   = decltype(tile_at(std::declval<VecField_t&>(), i));
        using View_t     = Tiled_t::template view_t<TileVF_t>;
        return tiled.template as<View_t>([&](auto& member) { return tile_at(member, i); });
    }
    else if constexpr (has_tiled_field_type_c<T>)
    {
        using Field_t = typename Tiled_t::field_type;
        using Tile_t  = typename Field_t::value_type::value_type;
        using V_t     = basic::TensorField<Tile_t, Tiled_t::rank>;
        return tiled.template as<V_t>([&](auto& c) { return c()[i](); });
    }
    else
        return std::forward<T>(tiled);
}

// tile tile_idx of x (a view), or x itself when no tile_idx is given (whole field)
template<typename T>
decltype(auto) maybe_tile_at(T&& x, auto const&... tile_idx)
{
    static_assert(sizeof...(tile_idx) < 2);
    if constexpr (sizeof...(tile_idx) == 0)
        return (x);
    else
        return tile_at(std::forward<T>(x), tile_idx...);
}

template<typename... Tiled>
auto tiles_at(std::size_t const i, Tiled&&... tiled)
{
    return std::make_tuple(tile_at(std::forward<Tiled>(tiled), i)...);
}

template<typename T>
auto tile_count(T const& tiled)
{
    if constexpr (is_field_tile_set_v<std::remove_cvref_t<T>>)
        return tiled().size();
    else
        return tiled[0]().size();
}

template<typename T>
auto& tile_layout(T const& tiled, std::size_t const i)
{
    if constexpr (is_field_tile_set_v<std::remove_cvref_t<T>>)
        return tiled()[i].layout();
    else // tensorfield expected
        return tiled[0]()[i].layout();
}

template<typename Fn, typename Tiled0, typename... Args>
void tile_exec_with_layout_at(std::size_t const i, Fn&& fn, Tiled0&& tiled0, Args&&... args)
{
    fn(tile_layout(tiled0, i), tile_at(tiled0, i), tile_at(args, i)...);
}

template<typename Fn, typename Tiled0, typename... Args>
void tile_exec_with_layout(Fn&& fn, Tiled0&& tiled0, Args&&... args)
{
    for (std::size_t i = 0; i < tile_count(tiled0); ++i)
        tile_exec_with_layout_at(i, fn, tiled0, args...);
}

// dst = src for same shape (vec)fields, whole, or only tile tile_idx of tiled ones
void copy_data(auto& dst, auto& src, auto const&... tile_idx)
{
    if constexpr (sizeof...(tile_idx) == 0)
        dst.copyData(src);
    else
    {
        auto const copy = [](auto& d, auto const& s) { // the allocator's copy
            assert(d.size() == s.size());
            damn::core::mem::copy<std::remove_cvref_t<decltype(d)>::alloc_mode>(d.data(), s.data(),
                                                                                s.size());
        };
        auto d = tile_at(dst, tile_idx...);
        auto s = tile_at(src, tile_idx...);
        if constexpr (is_field_tile_set_v<std::remove_cvref_t<decltype(dst)>>)
            copy(d, s);
        else
            for (std::size_t c = 0; c < d.size(); ++c)
                copy(d[c], s[c]);
    }
}

// fn(data, size) for each contiguous buffer of a field: a tiled one's tiles (ghosts included) in
//  tile order, an untiled one's single buffer. Two fields of one tiling give the same buffers
void for_each_buffer(TiledField auto& field, auto&& fn)
{
    for (auto& tile : field())
        fn(tile().data(), tile().size());
}
void for_each_buffer(auto& field, auto&& fn)
    requires(!TiledField<decltype(field)>)
{
    fn(field.data(), field.size());
}

void sync_inner_ghosts(auto& vf, auto const&... tile_idx)
{
    if constexpr (is_field_tile_set_v<std::remove_cvref_t<decltype(vf)>>)
        vf.sync_inner_ghosts(tile_idx...);
    else
        for (auto& c : vf)
            c.sync_inner_ghosts(tile_idx...);
}


} // namespace plamn::core

#endif // PLAMN_CORE_DATA_FIELD_FIELD_TILES_HPP
