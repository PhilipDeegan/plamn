#ifndef PLAMN_CORE_DATA_PARTICLES_PARTICLES_SOAVX_CHUNKED_HPP
#define PLAMN_CORE_DATA_PARTICLES_PARTICLES_SOAVX_CHUNKED_HPP

// SoAVXChunked: SoAVX sorted by cell, with per cell chunks (particles_chunked.hpp: the chunks'
//  bookkeeping), as AoSMapped is AoS + CellMap. The columns' part: slots moved and written column
//  by column, sort() scattering all columns into a second set of columns, swapped in

#include "damn/core/vector.hpp"
#include "damn/core/memory/memory.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/def/damn_config.hpp"
#include "mkn/kul/for.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/data/particles/arrays/particles_chunked.hpp"
#include "plamn/core/data/particles/arrays/particles_soavx.hpp"
#include "plamn/core/data/particles/particle.hpp"

#include <tuple>
#include <array>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

#if DAMN_HAVE_THRUST
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#endif

namespace plamn::core
{

template<std::size_t dim>
using SoAVXChunks = Chunks<dim>;


template<std::size_t dim, auto alloc_mode_, auto _const_ = 0>
struct SoAVXChunkedSpan : public SoAVXSpan<dim, alloc_mode_, _const_>
{
    using Super = SoAVXSpan<dim, alloc_mode_, _const_>;
    using box_t = damn::core::Box<int, dim>;

    SoAVXChunkedSpan() = default;

    template<typename Particles_t>
    SoAVXChunkedSpan(Particles_t& particles)
        : Super{particles, 0, particles.size()}
        , box_{particles.box()}
        , starts_{particles.starts_data()}
        , sizes_{particles.sizes_data()}
        , sorted_{particles.sorted()}
    {
    }

    auto& box() const DAMN_ALL { return box_; }
    auto starts_data() const DAMN_ALL { return starts_; }
    auto sizes_data() const DAMN_ALL { return sizes_; }
    bool sorted() const DAMN_ALL { return sorted_; }

    box_t box_;
    chunk_offset_t const* starts_ = nullptr;
    chunk_offset_t const* sizes_  = nullptr;
    bool sorted_                  = false;
};


template<std::size_t dim, auto alloc_mode_>
struct SoAVXChunkedVector
    : public SoAVXVector<dim, alloc_mode_>,
      public ChunkState<dim, SoAVXVector<dim, alloc_mode_>::template container_t>
{
    using Super  = SoAVXVector<dim, alloc_mode_>;
    using Chunks = ChunkState<dim, Super::template container_t>;
    using box_t  = damn::core::Box<int, dim>;

    template<typename T>
    using container_t = Super::template container_t<T>;

    SoAVXChunkedVector(box_t const& box = {}, std::size_t const size = 0)
        : Super(size)
        , Chunks{box}
    {
    }

    bool sorted() const { return this->sorted_ and this->sorted_size_ == this->size(); }

    Super scratch_; // the columns sort() scatters into, then swaps in
};


template<typename Super_>
class SoAVXChunkedParticles
    : public SoAVXParticles<Super_>,
      public ChunkedOps<SoAVXChunkedParticles<Super_>, SoAVXParticles<Super_>::dimension,
                        SoAVXParticles<Super_>::alloc_mode>
{
    using Super = SoAVXParticles<Super_>;
    using Ops   = ChunkedOps<SoAVXChunkedParticles<Super_>, Super::dimension, Super::alloc_mode>;
    friend Ops;

public:
    auto static constexpr dimension    = Super::dimension;
    auto static constexpr alloc_mode   = Super::alloc_mode;
    auto static constexpr layout_mode  = LayoutMode::SoAVXChunked;
    auto static constexpr storage_mode = Super::storage_mode;

    using box_t = damn::core::Box<int, dimension>;
    using Ops::nbr_particles_in;
    using Super::size;

    template<typename... Args>
    SoAVXChunkedParticles(Args&&... args)
        requires std::is_constructible_v<Super, Args&&...>
        : Super{std::forward<Args>(args)...}
    {
    }

    SoAVXChunkedParticles(SoAVXChunkedParticles const&)            = default;
    SoAVXChunkedParticles(SoAVXChunkedParticles&&)                 = default;
    SoAVXChunkedParticles& operator=(SoAVXChunkedParticles const&) = default;
    SoAVXChunkedParticles& operator=(SoAVXChunkedParticles&&)      = default;

    void resize(std::size_t const s)
        requires(storage_mode == StorageMode::VECTOR)
    {
        this->unsort();
        Super_::resize(s);
    }

    template<typename Particle_t>
    void push_back(Particle_t const& particle)
        requires(storage_mode == StorageMode::VECTOR)
    {
        this->unsort();
        Super::push_back(particle);
    }

    void clear()
        requires(storage_mode == StorageMode::VECTOR)
    {
        this->unsort();
        Super_::clear();
    }

    // one particle: anything with weight()/iCell()/delta()/v(), or Particle_t's arguments
    template<typename... Args>
    void emplace_back(Args&&... args)
        requires(storage_mode == StorageMode::VECTOR)
    {
        if constexpr (sizeof...(Args) == 1)
            push_back(args...);
        else
            push_back(Particle_t{args...});
    }

    void reserve(std::size_t const s)
        requires(storage_mode == StorageMode::VECTOR)
    {
        std::apply([&](auto&... column) { ((column.reserve(s)), ...); }, this->as_tuple());
    }

    // the chunks are kept eagerly, as AoSMapped's CellMap: rebuilt after every change
    //  through these hooks (appends, the post push sync), queries assert sorted()
    template<auto type>
    void on_appended()
        requires(storage_mode == StorageMode::VECTOR)
    {
        this->sort();
    }

    template<auto type>
    void on_moved(auto&&... /*stream*/)
        requires(storage_mode == StorageMode::VECTOR)
    {
        this->sort();
    }

    // that's particles appended, column by column: one bulk copy per column whatever that's
    //  allocator (e.g. a CPU staging array into GPU_UNIFIED). Unsorted until on_appended
    template<typename That>
    void append_columns(That const& that)
        requires(storage_mode == StorageMode::VECTOR)
    {
        this->unsort();
        std::size_t const offset = size();
        std::size_t const n      = that.size();
        if (n == 0)
            return;

        auto columns            = this->as_tuple();
        auto const that_columns = that.as_tuple();
        mkn::kul::for_N<std::tuple_size_v<decltype(columns)>>([&](auto c) {
            auto& column = std::get<c>(columns);
            damn::resize(column, offset + n);
            damn::core::mem::copy<alloc_mode>(column.data() + offset,
                                              std::get<c>(that_columns).data(), n);
        });
    }

private:
    using Particle_t = Particle<dimension>;

    // ChunkedOps' storage hooks (particles_chunked.hpp): the columns
    double weight_at_(std::size_t const i) const { return this->weight_[i]; }
    void set_weight_(std::size_t const i, double const w) { this->weight_[i] = w; }
    std::array<int, dimension> cell_at_(std::size_t const i) const
    {
        std::array<int, dimension> cell;
        for (std::size_t d = 0; d < dimension; ++d)
            cell[d] = this->iCell_[d][i];
        return cell;
    }

    void move_slot_(std::size_t const to, std::size_t const from)
    {
        auto columns = this->as_tuple();
        mkn::kul::for_N<std::tuple_size_v<decltype(columns)>>(
            [&](auto k) { std::get<k>(columns)[to] = std::get<k>(columns)[from]; });
    }

    void write_slot_(std::size_t const i, auto const& p)
    {
        this->weight_[i] = p.weight();
        mkn::kul::for_N<dimension>([&](auto d) {
            this->iCell_[d][i] = p.iCell()[d];
            this->delta_[d][i] = p.delta()[d];
        });
        mkn::kul::for_N<3>([&](auto d) { this->v_[d][i] = p.v()[d]; });
    }

    void write_filler_(std::size_t const i, std::array<int, dimension> const& cell)
    {
        this->weight_[i] = 0;
        mkn::kul::for_N<dimension>([&](auto d) {
            this->iCell_[d][i] = cell[d];
            this->delta_[d][i] = 0;
        });
        mkn::kul::for_N<3>([&](auto d) { this->v_[d][i] = 0; });
    }

    auto slot_weights_() const
    {
        auto const* weights = this->weight_.data();
        return [=] DAMN_ALL(std::size_t const i) { return weights[i]; };
    }
    auto slot_cells_() const
    {
        auto cells_of = mkn::kul::for_N<dimension, mkn::kul::for_N_R_mode::make_array>(
            [&](auto d) -> int const* { return this->iCell_[d].data(); });
        return [=] DAMN_ALL(std::size_t const i) {
            return mkn::kul::for_N<dimension, mkn::kul::for_N_R_mode::make_array>(
                [&](auto d) { return cells_of[d][i]; });
        };
    }

    // every column scattered into the scratch columns' slots to(i), swapped in
    void scatter_sorted_(auto const& to, std::size_t const n, std::size_t const slots_total)
    {
        auto static constexpr gpu = alloc_mode == damn::AllocatorMode::GPU_UNIFIED;
        // the scratch columns with at least the live ones' capacity: swapped in, the sort keeps
        //  the headroom (the initializer's reserve, growth) instead of an exact fit
        auto const capacity = std::max(slots_total, this->weight_.capacity());
        std::apply([&](auto&... c) { (damn::reserve(c, capacity, false), ...); },
                   this->scratch_.as_tuple());
        damn::resize(this->scratch_.weight_, slots_total, false);
        mkn::kul::for_N<dimension>([&](auto d) {
            damn::resize(this->scratch_.iCell_[d], slots_total, false);
            damn::resize(this->scratch_.delta_[d], slots_total, false);
        });
        mkn::kul::for_N<3>([&](auto d) { damn::resize(this->scratch_.v_[d], slots_total, false); });
        auto const columns = [](auto& v) {
            return std::apply([](auto&... c) { return std::make_tuple(c.data()...); },
                              v.as_tuple());
        };
        auto const src     = columns(*this);
        auto const dst     = columns(this->scratch_);
        auto const scatter = [=] DAMN_ALL(std::size_t const i) {
            auto const slot = to(i);
            if (slot == std::numeric_limits<std::size_t>::max())
                return;
            mkn::kul::for_N<std::tuple_size_v<decltype(src)>>(
                [&](auto c) { std::get<c>(dst)[slot] = std::get<c>(src)[i]; });
        };
        if constexpr (gpu)
        {
#if DAMN_HAVE_THRUST
            auto const first = thrust::counting_iterator<std::size_t>(0);
            thrust::for_each(thrust::device, first, first + n, scatter);
#endif
        }
        else
            for (std::size_t i = 0; i < n; ++i)
                scatter(i);

        auto mine   = this->as_tuple();
        auto theirs = this->scratch_.as_tuple();
        mkn::kul::for_N<std::tuple_size_v<decltype(mine)>>(
            [&](auto c) { std::get<c>(mine).swap(std::get<c>(theirs)); });
    }
};


template<std::size_t dim, auto alloc_mode = damn::AllocatorMode::CPU>
using SoAVXChunkedVectorParticles = SoAVXChunkedParticles<SoAVXChunkedVector<dim, alloc_mode>>;


} // namespace plamn::core


#endif /* PLAMN_CORE_DATA_PARTICLES_PARTICLES_SOAVX_CHUNKED_HPP */
