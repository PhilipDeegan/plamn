#ifndef PLAMN_CORE_DATA_PARTICLES_PARTICLES_SOAVX_CHUNKED_HPP
#define PLAMN_CORE_DATA_PARTICLES_PARTICLES_SOAVX_CHUNKED_HPP

// SoAVXChunked: SoAVX sorted by cell, with per cell chunks, as AoSMapped is AoS + CellMap
//
//  starts_[c] is the first particle of cell c, sizes_[c] its count, cells c
//  flattened in C order (last direction fastest) over shell() = grow(box, 1) (as AoSMapped's
//  CellMap, the safe outer shell). A span of cells along the last direction is then one
//  contiguous range of particles: selecting / exporting a box is a list of ranges.
//
//  Each cell's chunk is starts_[c], sizes_[c]: chunks are in cell order, with gaps between them,
//  free slots holding fillers (weight 0, the cell of the chunk before them). sort() lays them out:
//  a counting sort (histogram with a slot per particle, exclusive scan of each cell's particles
//  and gap, one scatter of all columns into a second set of columns, swapped in), the gaps the
//  slack: a fraction of the particles spread over the patch's cells, border cells weighted (their
//  particles leave to and arrive from other patches; inside, about as many enter a cell as leave
//  it), none in the shell outside the box. Without slack (the default) chunks are contiguous.
//
//  Between sorts, particles change cells without one (resettle(), insert(), remove()): a chunk
//  grows into the gap at its tail or head, whichever is larger, and shrinks from the end with the
//  smaller gap (the gaps either side even out); an empty chunk takes the middle of the gap it is
//  in. A chunk with no room either side has a neighbour slide over by one into its other gap,
//  which, without one, first has its own neighbour slide (up to max_slides chunks each way). Only
//  when none of them has room is a full sort needed.
//
//  Any other mutation (push, append, erase, resize) leaves the chunks stale: sorted() is false
//  until the next sort(), range accessors assert sorted(). GPU_UNIFIED sorts on the device
//  (thrust), the order of particles within a cell is then not deterministic (atomic slots), CPU
//  is stable.

#include "damn/core/vector.hpp"
#include "damn/core/operators.hpp"
#include "damn/core/memory/memory.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/def/damn_config.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/logger.hpp"
#include "damn/core/meta/for_N.hpp"

#include "plamn/core/def.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/data/particles/arrays/particles_soavx.hpp"
#include "plamn/core/data/particles/particle.hpp"

#include "phlop/timing/scope.hpp"

#include <tuple>
#include <cstdint>
#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <vector>
#include <stdexcept>
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

// per cell offsets type: particles per patch fit in 32 bits, and device atomics on it are cheap
using chunk_offset_t = std::uint32_t;


// the chunks view shared by the vector and the span: shell box + starts and sizes pointers
template<std::size_t dim>
struct SoAVXChunks
{
    using box_t = damn::core::Box<int, dim>;

    // flat C order index of cell (x[, y[, z]]) in shell
    static std::size_t flat(box_t const& shell, auto const& cell) DAMN_ALL
    {
        std::size_t idx = 0;
        for (std::size_t d = 0; d < dim; ++d)
            idx = idx * (shell.upper[d] - shell.lower[d] + 1) + (cell[d] - shell.lower[d]);
        return idx;
    }

    // calls fn(start, size) per contiguous particle range of the cells of box (clipped to
    //  shell), adjacent ranges merged (no gap between them), empty ones skipped
    static void for_each_range(box_t const& shell, chunk_offset_t const* starts,
                               chunk_offset_t const* sizes, box_t const& box, auto&& fn)
    {
        auto const clipped = box * shell;
        if (!clipped)
            return;

        std::size_t start = 0, size = 0;
        damn::core::for_each_span(*clipped,
                                  [&](auto const& span_start, std::uint32_t const span_size) {
                                      auto const c0 = flat(shell, span_start);
                                      for (std::size_t c = c0; c < c0 + span_size; ++c)
                                      {
                                          std::size_t const s = starts[c], n = sizes[c];
                                          if (n == 0)
                                              continue;
                                          if (size > 0 and start + size == s)
                                              size += n;
                                          else
                                          {
                                              if (size > 0)
                                                  fn(start, size);
                                              start = s;
                                              size  = n;
                                          }
                                      }
                                  });
        if (size > 0)
            fn(start, size);
    }
};


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
struct SoAVXChunkedVector : public SoAVXVector<dim, alloc_mode_>
{
    using Super = SoAVXVector<dim, alloc_mode_>;
    using box_t = damn::core::Box<int, dim>;

    template<typename T>
    using container_t = Super::template container_t<T>;

    SoAVXChunkedVector(box_t const& box = {}, std::size_t const size = 0)
        : Super(size)
        , box_{box}
    {
    }

    auto& box() const { return box_; }
    auto starts_data() const { return starts_.data(); }
    auto sizes_data() const { return sizes_.data(); }
    bool sorted() const { return sorted_ and sorted_size_ == this->size(); }

    // the chunks are stale after any change to the particles but resettle / insert / remove,
    //  until the next sort()
    void unsort() { sorted_ = false; }

    // sort()'s gaps: slack (a fraction of the particles) over domain's cells (the patch's: box()
    //  is the cell map's, larger), its border cells weighted border to 1 inside
    void set_slack(double const fraction, box_t const& domain, double const border = 3)
    {
        slack_         = fraction;
        slack_domain_  = domain;
        border_weight_ = border;
    }

    box_t box_;
    container_t<chunk_offset_t> starts_; // per cell, see top of file
    container_t<chunk_offset_t> sizes_;  // per cell
    container_t<chunk_offset_t> gaps_;   // per cell, after its chunk, sort() scratch
    container_t<chunk_offset_t> slots_;  // per particle slot in its cell, sort() scratch
    double slack_         = 0;
    box_t slack_domain_   = {};
    double border_weight_ = 3;
    Super scratch_; // the columns sort() scatters into, then swaps in
    bool sorted_             = false;
    std::size_t sorted_size_ = 0;
};


template<typename Super_>
class SoAVXChunkedParticles : public SoAVXParticles<Super_>
{
    using Super  = SoAVXParticles<Super_>;
    using Chunks = SoAVXChunks<Super::dimension>;

public:
    auto static constexpr dimension    = Super::dimension;
    auto static constexpr alloc_mode   = Super::alloc_mode;
    auto static constexpr layout_mode  = LayoutMode::SoAVXChunked;
    auto static constexpr storage_mode = Super::storage_mode;

    using box_t = damn::core::Box<int, dimension>;
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

    // the cells the chunks cover
    auto shell() const { return damn::core::grow(this->box(), 1); }

    // particles of cell, as {start, size}
    auto cell_range(auto const& cell) const
    {
        DAMN_ASSERT(this->sorted());
        auto const c = Chunks::flat(shell(), cell);
        return std::pair<std::size_t, std::size_t>{this->starts_[c], this->sizes_[c]};
    }

    // fn(start, size) per contiguous range of the particles in box, see SoAVXChunks
    void for_each_range(box_t const& box, auto&& fn) const
    {
        DAMN_ASSERT(this->sorted());
        Chunks::for_each_range(shell(), this->starts_data(), this->sizes_data(), box, fn);
    }

    // the particles, not the gaps (sorted)
    NO_DISCARD std::size_t nbr_particles() const
    {
        std::size_t n = 0;
        for_each_range(shell(), [&](auto, auto const size) { n += size; });
        return n;
    }

    // a particle into cell's chunk, its tail or head gap, whichever is larger: false if neither
    //  has room (or cell is outside the shell); the chunks stay sorted
    bool insert(auto const& particle)
        requires(storage_mode == StorageMode::VECTOR);

    // the particle of slot i, in the chunk of cell c (flat), out: the end of the chunk with the
    //  smaller gap next to it fills the hole, so the gaps either side even out
    void remove(std::size_t c, std::size_t i)
        requires(storage_mode == StorageMode::VECTOR);

    // after particles' cells changed (a push): each chunk's particles no longer in its cell taken
    //  out; those whose cell is in keep inserted into their new chunk (else a full sort, they
    //  appended), the others given to leave(particle). Returns whether it sorted
    bool resettle(box_t const& keep, auto&& leave)
        requires(storage_mode == StorageMode::VECTOR);

    std::size_t count(box_t const& box) const
    {
        std::size_t n = 0;
        for_each_range(box, [&](auto, auto const size) { n += size; });
        return n;
    }

    // counting sort by cell over shell(), all columns moved together, chunks laid out with the
    //  slack's gaps between them (weight 0 slots, dropped: gaps)
    void sort()
        requires(storage_mode == StorageMode::VECTOR);

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
        sort();
    }

    template<auto type>
    void on_moved(auto&&... /*stream*/)
        requires(storage_mode == StorageMode::VECTOR)
    {
        sort();
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
        damn::core::for_N<std::tuple_size_v<decltype(columns)>>([&](auto c) {
            auto& column = std::get<c>(columns);
            damn::resize(column, offset + n);
            damn::core::mem::copy<alloc_mode>(column.data() + offset,
                                              std::get<c>(that_columns).data(), n);
        });
    }

    std::size_t nbr_particles_in(box_t const& box) const { return count(box); }

    // a gap slot of chunk c (flat): weight 0, the chunk's cell
    void filler_(std::size_t const i, std::size_t const c)
    {
        auto const cell  = cell_of_(c);
        this->weight_[i] = 0;
        damn::core::for_N<dimension>([&](auto d) {
            this->iCell_[d][i] = cell[d];
            this->delta_[d][i] = 0;
        });
        damn::core::for_N<3>([&](auto d) { this->v_[d][i] = 0; });
    }

    std::size_t nbr_particles_in(std::array<int, dimension> const& cell) const
    {
        return cell_range(cell).second;
    }

private:
    using Particle_t = Particle<dimension>;

    // the cell of flat index c in the shell
    std::array<int, dimension> cell_of_(std::size_t c) const
    {
        auto const shell = this->shell();
        std::array<int, dimension> cell;
        for (std::size_t d = dimension; d-- > 0;)
        {
            auto const n = static_cast<std::size_t>(shell.upper[d] - shell.lower[d] + 1);
            cell[d]      = shell.lower[d] + static_cast<int>(c % n);
            c /= n;
        }
        return cell;
    }

    // the end of the last non empty chunk before c, the start of the first after c
    NO_DISCARD std::size_t prev_end_(std::size_t const c) const
    {
        for (std::size_t k = c; k-- > 0;)
            if (this->sizes_[k])
                return this->starts_[k] + this->sizes_[k];
        return 0;
    }
    NO_DISCARD std::size_t next_start_(std::size_t const c) const
    {
        for (std::size_t k = c + 1; k < this->sizes_.size(); ++k)
            if (this->sizes_[k])
                return this->starts_[k];
        return size();
    }

    // chunks a slide may cascade through (each moves one particle), before a full sort
    static constexpr std::size_t max_slides = 8;

    // the next non empty chunk after c moved up one slot into its tail gap, making that gap first
    //  by sliding the chunk after it (and so on, up to depth chunks): a free slot after c
    bool slide_next_(std::size_t const c, std::size_t const depth = max_slides)
    {
        for (std::size_t k = c + 1; k < this->sizes_.size(); ++k)
            if (this->sizes_[k])
            {
                auto& start = this->starts_[k];
                auto& n     = this->sizes_[k];
                if (next_start_(k) == start + n and (depth == 1 or !slide_next_(k, depth - 1)))
                    return false;
                move_slot_(start + n, start);
                this->filler_(start, c);
                ++start;
                return true;
            }
        return false;
    }
    // the previous non empty chunk before c moved down one slot into its head gap, the same way
    bool slide_prev_(std::size_t const c, std::size_t const depth = max_slides)
    {
        for (std::size_t k = c; k-- > 0;)
            if (this->sizes_[k])
            {
                auto& start = this->starts_[k];
                auto& n     = this->sizes_[k];
                if (prev_end_(k) == start and (depth == 1 or !slide_prev_(k, depth - 1)))
                    return false;
                move_slot_(start - 1, start + n - 1);
                this->filler_(start + n - 1, k);
                --start;
                return true;
            }
        return false;
    }

    void move_slot_(std::size_t const to, std::size_t const from)
    {
        auto columns = this->as_tuple();
        damn::core::for_N<std::tuple_size_v<decltype(columns)>>(
            [&](auto k) { std::get<k>(columns)[to] = std::get<k>(columns)[from]; });
    }

    void write_slot_(std::size_t const i, auto const& p)
    {
        this->weight_[i] = p.weight();
        damn::core::for_N<dimension>([&](auto d) {
            this->iCell_[d][i] = p.iCell()[d];
            this->delta_[d][i] = p.delta()[d];
        });
        damn::core::for_N<3>([&](auto d) { this->v_[d][i] = p.v()[d]; });
    }
};


template<typename Super_>
bool SoAVXChunkedParticles<Super_>::insert(auto const& particle)
    requires(storage_mode == StorageMode::VECTOR)
{
    auto const shell = this->shell();
    damn::core::Point<int, dimension> cell;
    for (std::size_t d = 0; d < dimension; ++d)
        cell[d] = particle.iCell()[d];
    if (!isIn(cell, shell))
        return false;
    auto const c     = Chunks::flat(shell, cell);
    auto& start      = this->starts_[c];
    auto& n          = this->sizes_[c];
    auto const lower = prev_end_(c);
    auto const upper = next_start_(c);
    if (n == 0)
    { // the middle of the gap it is in
        if (upper == lower)
            return false;
        start = static_cast<chunk_offset_t>(lower + (upper - lower) / 2);
        write_slot_(start, particle);
        n = 1;
        return true;
    }
    std::size_t head = start - lower, tail = upper - (start + n);
    if (head == 0 and tail == 0)
    { // a neighbour chunk slides over by one into its other gap: its first particle to its tail
      //  (the next chunk), or its last to its head (the previous one)
        if (slide_next_(c))
            tail = 1;
        else if (slide_prev_(c))
            head = 1;
        else
            return false;
    }
    if (tail >= head)
        write_slot_(start + n, particle);
    else
        write_slot_(--start, particle);
    ++n;
    return true;
}

template<typename Super_>
void SoAVXChunkedParticles<Super_>::remove(std::size_t const c, std::size_t const i)
    requires(storage_mode == StorageMode::VECTOR)
{
    auto& start = this->starts_[c];
    auto& n     = this->sizes_[c];
    DAMN_ASSERT(n > 0 and i >= start and i < start + n);
    std::size_t const head = start - prev_end_(c), tail = next_start_(c) - (start + n);
    if (tail <= head)
    { // the last fills the hole, the tail gap grows
        auto const last = start + n - 1;
        if (i != last)
            move_slot_(i, last);
        this->filler_(last, c);
    }
    else
    { // the first fills it, the head gap grows
        if (i != start)
            move_slot_(i, start);
        this->filler_(start, c);
        ++start;
    }
    --n;
}

template<typename Super_>
bool SoAVXChunkedParticles<Super_>::resettle(box_t const& keep, auto&& leave)
    requires(storage_mode == StorageMode::VECTOR)
{
    DAMN_LOG_SCOPE(2, "SoAVXChunked::resettle");
    DAMN_ASSERT(this->sorted());
    std::vector<Particle_t> moved; // the particles that changed cell
    for (std::size_t c = 0; c < this->sizes_.size(); ++c)
    {
        auto const cell = cell_of_(c);
        for (std::size_t i = this->starts_[c]; i < this->starts_[c] + this->sizes_[c];)
        {
            bool stays = true;
            for (std::size_t d = 0; d < dimension; ++d)
                stays = stays and this->iCell_[d][i] == cell[d];
            if (stays)
            {
                ++i;
                continue;
            }
            moved.push_back(this->copy(i));
            remove(c, i); // slot i holds another particle of the chunk now, or is its head gap
            if (i < this->starts_[c])
                i = this->starts_[c];
        }
    }
    std::vector<Particle_t> overflow; // no room in their chunk
    for (auto const& p : moved)
    {
        if (!isIn(damn::core::Point<int, dimension>{p.iCell()}, keep))
            leave(p);
        else if (!insert(p))
            overflow.push_back(p);
    }
    if (overflow.empty())
        return false;
    for (auto const& p : overflow)
        this->push_back(p);
    sort();
    return true;
}


template<typename Super_>
void SoAVXChunkedParticles<Super_>::sort()
    requires(storage_mode == StorageMode::VECTOR)
{
    DAMN_LOG_SCOPE(2, "SoAVXChunked::sort");
    auto static constexpr gpu  = alloc_mode == damn::AllocatorMode::GPU_UNIFIED;
    using Op                   = damn::core::Operators<chunk_offset_t, /*atomic=*/gpu, /*GPU=*/gpu>;
    auto static constexpr none = std::numeric_limits<chunk_offset_t>::max(); // a gap's slot

    auto const shell        = this->shell();
    std::size_t const n     = size();
    std::size_t const cells = shell.size();

    damn::resize(this->starts_, cells, /*copy=*/false);
    damn::resize(this->sizes_, cells, /*copy=*/false);
    damn::resize(this->gaps_, cells, /*copy=*/false);
    damn::resize(this->slots_, n, /*copy=*/false);

    auto* const sizes   = this->sizes_.data();
    auto* const slots   = this->slots_.data();
    auto const* weights = this->weight_.data();
    auto cells_of       = damn::core::for_N<dimension, damn::core::for_N_R_mode::make_array>(
        [&](auto d) -> int const* { return this->iCell_[d].data(); });
    auto const cell_idx = [=] DAMN_ALL(std::size_t const i) {
        auto const cell = damn::core::for_N<dimension, damn::core::for_N_R_mode::make_array>(
            [&](auto d) { return cells_of[d][i]; });
        DAMN_ASSERT(isIn(damn::core::Point{cell}, shell));
        return Chunks::flat(shell, cell);
    };

    // 1. each particle's slot in its cell, the cells' sizes (gaps: weight 0, dropped)
    auto const count = [=] DAMN_ALL(std::size_t const i) {
        slots[i] = weights[i] == 0 ? none : Op{sizes[cell_idx(i)]}.increment_return_old();
    };
    std::fill(sizes, sizes + cells, chunk_offset_t{0});
    if constexpr (gpu)
    {
#if DAMN_HAVE_THRUST
        auto const first = thrust::counting_iterator<std::size_t>(0);
        thrust::for_each(thrust::device, first, first + n, count);
#else
        throw std::runtime_error("SoAVXChunked GPU sort needs thrust");
#endif
    }
    else
        for (std::size_t i = 0; i < n; ++i)
            count(i);

    // 2. the gaps (the slack over the box's cells, border cells weighted), the chunks' starts
    std::size_t particles = 0;
    for (std::size_t c = 0; c < cells; ++c)
        particles += sizes[c];
    auto* const gaps   = this->gaps_.data();
    auto* const starts = this->starts_.data();
    std::fill(gaps, gaps + cells, chunk_offset_t{0});
    if (auto const slack = static_cast<std::size_t>(this->slack_ * static_cast<double>(particles)))
    {
        auto const box    = this->slack_domain_;
        double total      = 0;
        auto const weight = [&](auto const& cell) {
            if (!isIn(damn::core::Point<int, dimension>{cell}, box))
                return 0.;
            for (std::size_t d = 0; d < dimension; ++d)
                if (cell[d] == box.lower[d] or cell[d] == box.upper[d])
                    return this->border_weight_;
            return 1.;
        };
        for (std::size_t c = 0; c < cells; ++c)
            total += weight(cell_of_(c));
        for (std::size_t c = 0; c < cells; ++c)
            gaps[c] = static_cast<chunk_offset_t>(static_cast<double>(slack) * weight(cell_of_(c))
                                                  / total);
    }
    std::size_t slots_total = 0;
    for (std::size_t c = 0; c < cells; ++c)
    {
        starts[c] = static_cast<chunk_offset_t>(slots_total);
        slots_total += sizes[c] + gaps[c];
    }

    // 3. the particles into their slots of the scratch columns, swapped in, the gaps filled
    damn::resize(this->scratch_.weight_, slots_total, false);
    damn::core::for_N<dimension>([&](auto d) {
        damn::resize(this->scratch_.iCell_[d], slots_total, false);
        damn::resize(this->scratch_.delta_[d], slots_total, false);
    });
    damn::core::for_N<3>([&](auto d) { damn::resize(this->scratch_.v_[d], slots_total, false); });
    auto const columns = [](auto& v) {
        return std::apply([](auto&... c) { return std::make_tuple(c.data()...); }, v.as_tuple());
    };
    auto const src     = columns(*this);
    auto const dst     = columns(this->scratch_);
    auto const scatter = [=] DAMN_ALL(std::size_t const i) {
        if (slots[i] == none)
            return;
        auto const to = starts[cell_idx(i)] + slots[i];
        damn::core::for_N<std::tuple_size_v<decltype(src)>>(
            [&](auto c) { std::get<c>(dst)[to] = std::get<c>(src)[i]; });
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
    damn::core::for_N<std::tuple_size_v<decltype(mine)>>(
        [&](auto c) { std::get<c>(mine).swap(std::get<c>(theirs)); });
    for (std::size_t c = 0; c < cells; ++c)
        for (std::size_t i = starts[c] + sizes[c]; i < starts[c] + sizes[c] + gaps[c]; ++i)
            this->filler_(i, c);

    this->sorted_      = true;
    this->sorted_size_ = slots_total;
}


template<std::size_t dim, auto alloc_mode = damn::AllocatorMode::CPU>
using SoAVXChunkedVectorParticles = SoAVXChunkedParticles<SoAVXChunkedVector<dim, alloc_mode>>;


} // namespace plamn::core


#endif /* PLAMN_CORE_DATA_PARTICLES_PARTICLES_SOAVX_CHUNKED_HPP */
