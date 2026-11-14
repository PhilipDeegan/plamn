#ifndef PLAMN_CORE_DATA_PARTICLES_PARTICLE_TRANSLATION_TRACKING_HPP
#define PLAMN_CORE_DATA_PARTICLES_PARTICLE_TRANSLATION_TRACKING_HPP

#include "damn/core/vector.hpp"
#include "damn/core/memory/span.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/data/ndarray/ndarray_vector.hpp"
#include "damn/core/data/ndarray/ndarray_view.hpp"
#include "damn/core/def/damn_config.hpp"

#include "plamn/core/data/particles/particles_def.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace damn::core
{
template<typename Type, std::size_t dim>
class Point;
} // namespace damn::core

namespace plamn::core
{

// Shared per-cell bookkeeping for the owning (VECTOR) side of PerCellVector / TileSetVector /
// PCTileSetVector: the gap/arrival/capacity counters used to register and apply particle moves,
// plus the box/local-cell plumbing they're indexed by. Identical across all three layouts.
//
// What is NOT here on purpose: sync/sync_moved/on_appended/move_check and friends. Those walk
// genuinely different index spaces per layout (per-cell, per-tile via a representative slot,
// per-tile-wall-cell layered on a nested per-cell level) and stay implemented separately.
template<std::size_t dim, auto alloc_mode>
class ParticleTranslationTracker
{
public:
    using SIZE_T  = damn::core::default_span_size_t;
    using box_t   = damn::core::Box<int, dim>;
    using lobox_t = damn::core::Box<std::uint32_t, dim>;

    template<typename T>
    using nd_array_t = damn::core::NdArrayVector<dim, T, /*c_order=*/true, alloc_mode>;

    template<typename T>
    using vec_helper = damn::Vector<T, alloc_mode, 1>;

    using size_t_vec_helper = vec_helper<std::size_t>;
    using size_t_vector     = typename size_t_vec_helper::vector_t;

    auto& box() const DAMN_ALL { return box_; }
    auto& ghost_box() const DAMN_ALL { return ghost_box_; }

    auto local_cell(std::array<int, dim> const& icell) const DAMN_ALL
    {
        return as_local_cell(ghost_box_, icell);
    }
    auto local_cell(damn::core::Point<int, dim> const& icell) const DAMN_ALL
    {
        return local_cell(icell.toArray());
    }

    auto local_box() const DAMN_ALL
    {
        return damn::core::box_from_zero_to_upper_minus_one(
            ghost_box_.shape().template toArray<std::uint32_t>());
    }
    auto local_box(damn::core::Box<int, dim> const& from) const DAMN_ALL
    {
        return lobox_t{local_cell(from.lower), local_cell(from.upper)};
    }

protected:
    ParticleTranslationTracker(box_t const& box, std::size_t const ghost_cells)
        : ghost_cells_{ghost_cells}
        , box_{box}
        , ghost_box_{damn::core::grow(box, ghost_cells)}
    {
        zero_bookkeeping();
    }

    void zero_bookkeeping()
    {
        cell_size_.zero();
        gap_idx_.zero();
        add_into_.zero();
        left_.zero();
        cap_.zero();
    }

    std::size_t ghost_cells_;
    damn::core::Box<int, dim> box_, ghost_box_;

    nd_array_t<size_t_vector> gaps_{local_box().shape()};
    nd_array_t<damn::core::Span<std::size_t>> gap_views_{local_box().shape()};
    nd_array_t<SIZE_T> gap_idx_{local_box().shape()};
    nd_array_t<SIZE_T> add_into_{local_box().shape()};
    nd_array_t<SIZE_T> left_{local_box().shape()};
    nd_array_t<SIZE_T> cap_{local_box().shape()};
    nd_array_t<std::size_t> cell_size_{local_box().shape()};

    std::size_t total_size = 0;
};


// Span-side counterpart of ParticleTranslationTracker: the same box/local-cell plumbing plus
// the subset of bookkeeping views identical across PerCellSpan / TileSetSpan / PCTileSetSpan
// (gaps_/gap_idx_/add_into_/cap_/left_, size_, box_/ghost_box_/local_ghost_box_).
//
// cell_size_ is deliberately excluded: PerCellSpan has no such member (it tracks per-cell size
// via each cell's own vector), so it isn't identical across all three - it stays declared
// locally on TileSetSpan/PCTileSetSpan instead. Same reasoning as PerCellVector's off_sets_ on
// the VECTOR side above.
template<std::size_t dim>
class ParticleTranslationTrackerSpan
{
public:
    using SIZE_T  = damn::core::default_span_size_t;
    using lobox_t = damn::core::Box<std::uint32_t, dim>;

    auto size() const DAMN_ALL { return size_; }

    auto& box() const DAMN_ALL { return box_; }
    auto& ghost_box() const DAMN_ALL { return ghost_box_; }

    auto local_cell(std::array<int, dim> const& icell) const DAMN_ALL
    {
        return as_local_cell(ghost_box_, icell);
    }
    auto local_cell(damn::core::Point<int, dim> const& icell) const DAMN_ALL
    {
        return local_cell(icell.toArray());
    }

    auto& local_box() const DAMN_ALL { return local_ghost_box_; }
    auto local_box(damn::core::Box<int, dim> const& from) const DAMN_ALL
    {
        return lobox_t{local_cell(from.lower), local_cell(from.upper)};
    }

protected:
    // named fields (rather than positional ctor args) so callers can't silently swap two
    // same-typed values (e.g. box vs ghost_box, or the several std::size_t views)
    struct Init
    {
        damn::core::NdArrayView<dim, damn::core::Span<std::size_t>> gaps;
        damn::core::NdArrayView<dim, SIZE_T> gap_idx, add_into, cap, left;
        std::size_t size;
        damn::core::Box<int, dim> box, ghost_box;
        lobox_t local_ghost_box;
    };

    ParticleTranslationTrackerSpan(Init const& init)
        : gaps_{init.gaps}
        , gap_idx_{init.gap_idx}
        , add_into_{init.add_into}
        , cap_{init.cap}
        , left_{init.left}
        , size_{init.size}
        , box_{init.box}
        , ghost_box_{init.ghost_box}
        , local_ghost_box_{init.local_ghost_box}
    {
    }

    damn::core::NdArrayView<dim, damn::core::Span<std::size_t>> gaps_;
    damn::core::NdArrayView<dim, SIZE_T> gap_idx_, add_into_, cap_, left_;
    std::size_t size_;

    damn::core::Box<int, dim> box_, ghost_box_;
    lobox_t local_ghost_box_;
};

} // namespace plamn::core

#endif /* PLAMN_CORE_DATA_PARTICLES_PARTICLE_TRANSLATION_TRACKING_HPP */
