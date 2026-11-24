#ifndef PLAMN_CORE_DATA_PARTICLES_CELLMAP_HPP
#define PLAMN_CORE_DATA_PARTICLES_CELLMAP_HPP

#include "damn/core/data/ndarray/ndarray_view.hpp"
#include "damn/core/def/damn_config.hpp" // IWYU pragma: keep
#include "damn/core/logger.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/range.hpp"
#include "mkn/kul/meta.hpp"
#include "damn/core/data/ndarray/ndarray_vector.hpp"
#include "damn/core/geometry/point.hpp"

#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/def.hpp"
#include "plamn/core/geometry/indexer.hpp"

#include <cassert>
#include <array>
#include <cstddef>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace plamn::core
{

struct CellMapOptions
{
    using value_type = int;

    std::size_t dimension;
    damn::AllocatorMode alloc_mode = damn::AllocatorMode::CPU;
    StorageMode storage_mode = StorageMode::VECTOR;

    CellMapOptions static FROM(auto opts)
    {
        return {opts.dimension, opts.alloc_mode, opts.storage_mode};
    }
};


template<auto storage_mode, auto opts>
struct CellMapStorage;


// SPAN — non-owning view; Indexer<SPAN> objects live inside the VECTOR storage
template<auto opts>
struct CellMapStorage<StorageMode::SPAN, opts>
{
    bool constexpr static c_order = true;
    using Options                 = decltype(opts);
    using cell_index_t            = Options::value_type;

    CellMapStorage() = default;

    using idx_span_t   = Indexer<IndexerOptions{opts.alloc_mode, StorageMode::SPAN}>;
    using idx_vector_t = Indexer<IndexerOptions{opts.alloc_mode, StorageMode::VECTOR}>;

    CellMapStorage(auto& cellmap)
        : box_{cellmap.box_}
        , cellIndexes_{static_cast<idx_span_t*>(cellmap.cellIndexes_.data()),
                       cellmap.cellIndexes_.shape()}
    {
    }

    void reset_from(auto& cellmap)
    {
        cellIndexes_ = damn::core::NdArrayViewSpan<opts.dimension, idx_span_t, idx_vector_t>{
            static_cast<idx_span_t*>(cellmap.cellIndexes_.data()), cellmap.cellIndexes_.shape()};
    }

    damn::core::Box<cell_index_t, opts.dimension> box_;
    damn::core::NdArrayViewSpan<opts.dimension, idx_span_t, idx_vector_t> cellIndexes_;
};

// VECTOR — owns the single NdArrayVector; its Indexer<VECTOR> elements inherit Indexer<SPAN>,
// so the SPAN base above can alias them directly via static_cast with no extra allocation.
template<auto opts>
struct CellMapStorage<StorageMode::VECTOR, opts>
    : CellMapStorage<StorageMode::SPAN,
                     CellMapOptions{opts.dimension, opts.alloc_mode, StorageMode::SPAN}>
{
    bool constexpr static c_order = true;
    using Options                 = decltype(opts);
    using cell_index_t            = Options::value_type;
    using SpanBase
        = CellMapStorage<StorageMode::SPAN,
                         CellMapOptions{opts.dimension, opts.alloc_mode, StorageMode::SPAN}>;

    using idx_span_t   = Indexer<IndexerOptions{opts.alloc_mode, StorageMode::SPAN}>;
    using idx_vector_t = Indexer<IndexerOptions{opts.alloc_mode, StorageMode::VECTOR}>;

    CellMapStorage(damn::core::Box<cell_index_t, opts.dimension> const& box)
        : SpanBase{}
        , cellIndexes_{box.shape().template toArray<std::uint32_t>()}
    {
        SpanBase::box_ = box;
        SpanBase::cellIndexes_
            = {static_cast<idx_span_t*>(cellIndexes_.data()), cellIndexes_.shape()};
    }

    damn::core::NdArrayVector<opts.dimension, idx_vector_t, c_order, opts.alloc_mode> cellIndexes_;
};


template<auto opts>
class CellMap : public CellMapStorage<opts.storage_mode, opts>
{
    using Super = CellMapStorage<opts.storage_mode, opts>;
    using Super::box_;
    using Super::cellIndexes_;
    using Options      = decltype(opts);
    using cell_index_t = Options::value_type;

    template<auto, auto>
    friend struct CellMapStorage;

private:
    using cell_t = std::array<cell_index_t, opts.dimension>;
    using box_t  = damn::core::Box<cell_index_t, opts.dimension>;

public:
    CellMap(damn::core::Box<cell_index_t, opts.dimension> const& box)
        requires(opts.storage_mode == StorageMode::VECTOR)
        : Super{box}
    {
    }

    CellMap(CellMap const& from)            = default;
    CellMap(CellMap&& from)                 = default;
    CellMap& operator=(CellMap const& from) = default;
    CellMap& operator=(CellMap&& from)      = default;


    template<typename... Args>
    CellMap(Args&&... args)
        requires std::is_constructible_v<Super, Args&&...>
    DAMN_ALL : Super{std::forward<Args>(args)...}
    {
    }

    // produce a non-owning span view of this CellMap for GPU kernel dispatch.
    // Syncs every Indexer's span before building the view so GPU sees valid pointers.
    auto view()
        requires(opts.storage_mode == StorageMode::VECTOR)
    {
        for (auto& idx : cellIndexes_)
            idx.sync();
        constexpr auto span_opts
            = CellMapOptions{opts.dimension, opts.alloc_mode, StorageMode::SPAN};
        return CellMap<span_opts>{*this};
    }

    // refresh span pointer after source CellMap's NdArrayVector may have reallocated
    void reset_from(auto& from)
        requires(opts.storage_mode == StorageMode::SPAN)
    {
        Super::reset_from(from);
    }



    MKN_KUL_FN_SIG_RET auto nbr_cells() const { return cellIndexes_.size(); }


    MKN_KUL_FN_SIG_RET bool check_unique() const
    {
        std::unordered_map<std::size_t, std::size_t> counts;
        damn::core::for_each_point(box_, [&](auto const& cell) {
            auto& itemIndexes = cellIndexes_(local_(cell));
            for (auto const& index : itemIndexes)
            {
                if (counts.count(index) == 0)
                    counts[index] = 0;
                counts[index]++;
                if (counts[index] > 1)
                {
                    std::cout << "Oops " << cell << " " << index
                              << " appears mapped more than once\n";
                }
            }
        });
        bool unique = true;
        for (auto const& [index, count] : counts)
            if (count > 1)
            {
                std::cout << "CellMap CHECKUNIQUE : " << index << "appears " << count << " times\n";
                unique = false;
            }
        return unique;
    }


    // add a single index to the cellmap with the specified cell
    template<typename CellIndex>
    void addToCell(CellIndex const& cell, std::size_t itemIndex);

    static auto constexpr default_extractor
        = [](auto const& item) -> auto& { return item.iCell(); };
    using DefaultExtractor = decltype(default_extractor);


    // same as above but cell is found with the CellExtractor
    template<mkn::kul::Iterable Array, typename CellExtractor = DefaultExtractor>
    void add(Array const& items, std::size_t itemIndex, CellExtractor extract = default_extractor);



    // add all given items indexes to the cellmap.
    // items will be added to the cell they are in, given the CellExtractor
    template<mkn::kul::Iterable Array, typename CellExtractor = DefaultExtractor>
    void add(Array const& items, CellExtractor extract = default_extractor);



    // same as above but for indexes within the given range
    template<mkn::kul::Iterable Array, typename CellExtractor = DefaultExtractor>
    void add(Array const& items, std::size_t first, std::size_t last,
             CellExtractor extract = default_extractor);



    // number of indexes stored in that cell of the cellmap
    MKN_KUL_FN_SIG_RET std::size_t size(cell_t cell) const
    {
        return cellIndexes_(local_(cell)).size();
    }
    MKN_KUL_FN_SIG_RET std::size_t size(cell_t cell) { return cellIndexes_(local_(cell)).size(); }

    // total number of mapped indexes
    MKN_KUL_FN_SIG_RET auto size() const;

    // number of indexes mapped in the given box
    MKN_KUL_FN_SIG_RET auto size(box_t const& box) const;

    // total capacity over all cells
    MKN_KUL_FN_SIG_RET auto capacity() const;

    // remove all indexes
    void clear()
    {
        for (auto& cell : cellIndexes_)
            cell.clear();
    }

    void empty();

    MKN_KUL_FN_SIG_RET bool is_empty() const { return size() == 0; }


    MKN_KUL_FN_SIG_RET float used_mem_ratio() const
    {
        return static_cast<float>(size()) / capacity();
    }


    // export from 'from' into 'dest' items indexed in the map found withing 'box'
    template<typename Src, typename Dst>
    void export_to(box_t const& box, Src const& from, Dst& dest) const;


    // same as previous but applies a transformation to the items before exporting to 'dest'
    template<typename Src, typename Dst, typename Transformation>
    void export_to(box_t const& box, Src const& from, Dst& dest, Transformation&& Fn) const;


    // export items satisfying Predicate in 'from' into 'dest'
    template<typename Src, typename Dst, typename Predicate>
    void export_if(Src const& from, Dst& dest, Predicate&& pred) const;




    // item at itemIndex in items is registered at 'oldCell' but is now in a
    // different one. This method changes the cellmap to move the index to its new cell
    // new cell is obtained from the item
    template<typename Array, typename CellIndex, typename CellExtractor = DefaultExtractor>
    void update(Array& items, std::size_t itemIndex, CellIndex const& oldCell,
                CellExtractor extract = default_extractor);


    // re-orders the array so that elements satisfying the predicate are found first
    // and element not satisfying after. Returns the pivot index, which is the first
    // element not to satisfy the predicate
    // Ensures the cellmap and the re-ordered array are still consistent.
    template<typename Range, typename Predicate, typename CellExtractor = DefaultExtractor>
    auto partition(Range range, Predicate&& pred, CellExtractor = default_extractor);


    void erase(auto& items, box_t const& box);


    // erase all items indexed in the given range from both the cellmap and the
    // array the range is for.
    template<typename Range>
    void erase(Range range);


    // erase items indexes from the cellmap
    template<mkn::kul::Iterable Array, typename CellExtractor = DefaultExtractor>
    void erase(Array const& items, std::size_t itemIndex,
               CellExtractor extract = default_extractor);


    template<mkn::kul::Iterable Array>
    void erase(Array const& items, cell_t const oldCell, std::size_t const itemIndex);


    // sort all cell indexes
    void sort();

    template<typename CellIndex>
    void print(CellIndex const& cell) const;

    MKN_KUL_FN_SIG_RET auto& box() { return box_; }
    MKN_KUL_FN_SIG_RET auto const& box() const { return box_; }

    MKN_KUL_FN_SIG_RET auto begin() { return cellIndexes_.begin(); }
    MKN_KUL_FN_SIG_RET auto end() { return cellIndexes_.end(); }


    template<template<typename, std::size_t> typename Arr>
    MKN_KUL_FN_SIG_RET auto& operator()(Arr<std::uint32_t, opts.dimension> const& arr)
    {
        return cellIndexes_(arr);
    }

    template<template<typename, std::size_t> typename Arr>
    MKN_KUL_FN_SIG_RET auto& operator()(Arr<std::uint32_t, opts.dimension> const& arr) const
    {
        return cellIndexes_(arr);
    }

    template<typename Cell>
    MKN_KUL_FN_SIG_RET auto& operator()(Cell const& cell)
    {
        return cellIndexes_(local_(cell));
    }

    template<typename Cell>
    MKN_KUL_FN_SIG_RET auto const& operator()(Cell const& cell) const
    {
        return cellIndexes_(local_(cell));
    }

    void swap(cell_t const a, cell_t const b, std::size_t const i0, std::size_t i1);

private:
    template<typename Cell>
    auto local_(Cell const& cell) const
    {
        return (damn::core::Point{cell} - box_.lower).as_unsigned();
    }


    // Box<cell_index_t, dim> box_;
    // bool constexpr static c_order = true;
    // NdArrayVector<dim, Indexer, c_order, default_allocator_mode()> cellIndexes_;
};


template<auto opts>
void CellMap<opts>::swap(cell_t const a, cell_t const b, std::size_t const i0, std::size_t i1)
{
    assert(cellIndexes_(local_(a)).is_indexed(i0));
    assert(cellIndexes_(local_(b)).is_indexed(i1));
    cellIndexes_(local_(a)).updateIndex(i0, i1);
    cellIndexes_(local_(b)).updateIndex(i1, i0);
}


template<auto opts>
template<typename CellIndex>
inline void CellMap<opts>::addToCell(CellIndex const& cell, std::size_t itemIndex)
{
    if (isIn(damn::core::Point{cell}, box_))
        cellIndexes_(local_(cell)).add(itemIndex);
}


template<auto opts>
template<mkn::kul::Iterable Array, typename CellExtractor>
inline void CellMap<opts>::add(Array const& items, CellExtractor extract)
{
    for (std::size_t itemIndex = 0; itemIndex < items.size(); ++itemIndex)
    {
        addToCell(extract(items[itemIndex]), itemIndex);
    }
}

template<auto opts>
template<mkn::kul::Iterable Array, typename CellExtractor>
inline void CellMap<opts>::add(Array const& items, std::size_t first, std::size_t last,
                               CellExtractor extract)

{
    for (auto itemIndex = first; itemIndex <= last; ++itemIndex)
    {
        addToCell(extract(items[itemIndex]), itemIndex);
    }
}


template<auto opts>
template<mkn::kul::Iterable Array, typename CellExtractor>
inline void CellMap<opts>::add(Array const& items, std::size_t itemIndex, CellExtractor extract)
{
    addToCell(extract(items[itemIndex]), itemIndex);
}



template<auto opts>
template<mkn::kul::Iterable Array, typename CellExtractor>
inline void CellMap<opts>::erase(Array const& items, std::size_t itemIndex, CellExtractor extract)
{
    auto& blist = cellIndexes_(local_(extract(items[itemIndex])));
    blist.remove(itemIndex);
}


template<auto opts>
template<mkn::kul::Iterable Array>
inline void CellMap<opts>::erase(Array const& items, cell_t const oldCell,
                                 std::size_t const itemIndex)
{
    auto& blist = cellIndexes_(local_(oldCell));
    blist.remove(itemIndex);
}


template<auto opts>
inline auto CellMap<opts>::size() const
{
    DAMN_LOG_SCOPE(3, "CellMap::size()");
    std::size_t s = 0;
    for (auto const& cell : cellIndexes_)
    {
        s += cell.size();
    }
    return s;
}

template<auto opts>
inline auto CellMap<opts>::size(box_t const& box) const
{
    DAMN_LOG_SCOPE(3, "CellMap::size(box)");
    std::size_t s      = 0;
    auto const overlap = box_ * box; // box may extend past this cellmap's own range
    if (not overlap)
        return s;
    damn::core::for_each_point(*overlap, [&](auto const& cell) {
        auto& blist = cellIndexes_(local_(cell));
        s += blist.size();
    });
    return s;
}



template<auto opts>
template<typename CellIndex>
inline void CellMap<opts>::print(CellIndex const& cell) const
{
    auto& blist = cellIndexes_(local_(cell));
    for (auto itemIndex : blist)
    {
        std::cout << itemIndex << "\n";
    }
}


template<auto opts>
template<typename Src, typename Dst>
inline void CellMap<opts>::export_to(box_t const& box, Src const& from, Dst& dest) const
{
    damn::core::for_each_point(box, [&](auto const& cell) {
        auto& blist = cellIndexes_(local_(cell));
        for (auto itemIndex : blist)
        {
            dest.push_back(from[itemIndex]);
        }
    });
}

template<auto opts>
template<typename Src, typename Dst, typename Transformation>
inline void CellMap<opts>::export_to(box_t const& box, Src const& from, Dst& dest,
                                     Transformation&& Fn) const
{
    damn::core::for_each_point(box, [&](auto const& cell) {
        auto& blist = cellIndexes_(local_(cell));
        for (auto itemIndex : blist)
        {
            dest.push_back(Fn(from[itemIndex]));
        }
    });
}

template<auto opts>
template<typename Src, typename Dst, typename Predicate>
inline void CellMap<opts>::export_if(Src const& from, Dst& dest, Predicate&& pred) const
{
    damn::core::for_each_point(box_, [&](auto const& cell) {
        auto& blist = cellIndexes_(local_(cell));
        if (pred(cell))
        {
            for (auto itemIndex : blist)
            {
                dest.push_back(from[itemIndex]);
            }
        }
    });
}




template<auto opts>
inline auto CellMap<opts>::capacity() const
{
    std::size_t tot = 0;
    damn::core::for_each_point(box_, [&](auto const& cell) { tot += cellIndexes_(local_(cell)).capacity(); });
    return tot;
}

template<auto opts>
inline void CellMap<opts>::empty()
{
    damn::core::for_each_point(box_, [&](auto const& cell) {
        cellIndexes_(local_(cell)).empty();
        assert(cellIndexes_(local_(cell)).is_empty());
    });
}



template<auto opts>
template<typename Array, typename CellIndex, typename CellExtractor>
inline void CellMap<opts>::update(Array& items, std::size_t itemIndex, CellIndex const& oldCell,
                                  CellExtractor /*extract*/)
{
    // we want to check first if the particle is in the map
    // already. if is, needs to remove it before inserting it again
    // with at its right cell.
    // It could not be in the map at 'oldCell', in that case we want to add it.
    auto& oldList = cellIndexes_(local_(oldCell));
    if (oldList.is_indexed(itemIndex))
        oldList.remove(itemIndex);
    add(items, itemIndex);
}




template<auto opts>
template<typename Range, typename Predicate, typename CellExtractor>
inline auto CellMap<opts>::partition(Range range, Predicate&& pred, CellExtractor extract)
{
    std::size_t toSwapIndex = range.iend() - 1;
    auto pivot              = range.iend();
    damn::core::for_each_point(box_, [&](auto const& cell) {
        auto& itemIndexes = cellIndexes_(local_(cell));
        if (!pred(cell))
        {
            for (auto currentIdx : itemIndexes)
            {
                // partition only indexes in range
                if (currentIdx >= range.ibegin() and currentIdx < range.iend())
                {
                    assert(pivot > 0);
                    --pivot;
                    if (currentIdx < toSwapIndex)
                    {
                        while (!pred(extract(range.array()[toSwapIndex]))
                               and toSwapIndex > currentIdx)
                        {
                            --toSwapIndex;
                        }
                        if (toSwapIndex > currentIdx)
                        {
                            assert(pred(extract(range.array()[toSwapIndex])));
                            assert(toSwapIndex >= range.ibegin());
                            itemIndexes.updateIndex(currentIdx, toSwapIndex);
                            auto& l = cellIndexes_(local_(extract(range.array()[toSwapIndex])));
                            l.updateIndex(toSwapIndex, currentIdx);
                            std::swap(range.array()[currentIdx], range.array()[toSwapIndex]);
                            --toSwapIndex;
                        }
                    }
                }
            }
        }
    });

    return damn::core::makeRange(range.array(), range.ibegin(), pivot);
}


template<auto opts>
void CellMap<opts>::erase(auto& items, box_t const& box)
{
    std::size_t prev_idx = 0; // to throw if not sorted

    damn::core::for_each_point(box, [&](auto const& cell) {
        auto& blist = cellIndexes_(local_(cell));
        if (blist.size() == 0)
            return;
        prev_idx = blist.back();
        for (std::size_t i = blist.size(); i-- > 0;) // reverse loop
        {
            auto const& idx = blist[i];
            DAMN_LOG_LINE_SS(i << " " << idx << " " << items.size());
            if (idx > prev_idx)
                throw std::runtime_error("Index list is not sorted");
            if (idx > items.size())
                throw std::runtime_error("Index list is invalid");
            items.erase(items.begin() + idx);
            prev_idx = idx;
        }
        blist.clear();
    });
}


template<auto opts>
template<typename Range>
inline void CellMap<opts>::erase(Range range)
{
    auto& items = range.array();

    // first erase indexes from the cellmap
    // then items from the array
    for (std::size_t i = range.ibegin(); i < range.iend(); ++i)
    {
        erase(items, i);
    }
    items.erase(range.begin(), range.end());
}




template<auto opts>
inline void CellMap<opts>::sort()
{
    damn::core::for_each_point(box_, [&](auto const& cell) { cellIndexes_(local_(cell)).sort(); });
}


#if 0
// we don't need it now but may become handy in the future so leave it here
template<typename CellMap, typename Box, typename Particles>
void get_sorted(CellMap const& cm, Box const& box, Particles& dest)
{
    std::size_t pidx = 0;
    damn::core::for_each_point(box, [&](auto const& cell) {
        auto bucketlist = cm.list_at(cell);
        if (bucketlist)
        {
            for (auto const& part_ptr : bucketlist->get())
            {
                dest[pidx++] = *part_ptr;
            }
        }
    });
}
#endif

} // namespace plamn::core

#endif
