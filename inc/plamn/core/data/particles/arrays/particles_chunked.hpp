#ifndef PLAMN_CORE_DATA_PARTICLES_PARTICLES_CHUNKED_HPP
#define PLAMN_CORE_DATA_PARTICLES_PARTICLES_CHUNKED_HPP

// particles sorted by cell, in per cell chunks: the bookkeeping whatever the storage (columns:
//  SoAVXChunked, particle structures: AoS4SoAVXChunked)
//
//  starts_[c] is the first particle of cell c, sizes_[c] its count, cells c flattened in C order
//  (last direction fastest) over shell() = grow(box, 1) (as AoSMapped's CellMap, the safe outer
//  shell). A span of cells along the last direction is then one contiguous range of particles:
//  selecting / exporting a box is a list of ranges.
//
//  Each cell's chunk is starts_[c], sizes_[c]: chunks are in cell order, with gaps between them,
//  free slots holding fillers (weight 0, the cell of the chunk before them). sort() lays them out:
//  a counting sort (histogram with a slot per particle, exclusive scan of each cell's particles
//  and gap, one scatter of all particles into a second storage, swapped in), the gaps the
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
//
//  ChunkedOps<Derived> holds the algorithms; Derived (the particles), with a ChunkState in its
//  storage, provides the storage's part:
//    size(), copy(i), push_back(p)          as any particles
//    weight_at_(i), set_weight_(i, w)       a slot's weight (0: a gap)
//    cell_at_(i)                            a slot's cell (std::array<int, dim>)
//    move_slot_(to, from), write_slot_(i, p), write_filler_(i, cell)
//    slot_weights_(), slot_cells_()         sort()'s per slot functors (DAMN_ALL: device sorts)
//    scatter_sorted_(to, n, slots_total)    every slot i with to(i) != none into slot to(i) of a
//                                           scratch storage of slots_total slots, swapped in

#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/def/damn_config.hpp"
#include "damn/core/logger.hpp"
#include "damn/core/operators.hpp"
#include "damn/core/vector.hpp"
#include "plamn/core/def.hpp"
#include "plamn/core/data/particles/particle.hpp"
#include "phlop/timing/scope.hpp"
#include "damn/core/def/detail/mkn_gpu.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <format>

#if DAMN_HAVE_THRUST
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#endif

#if DAMN_HAVE_THRUST && DAMN_HAVE_MKN_GPU
namespace plamn::core::detail
{
// thrust's temporaries (exclusive_scan's) from one device buffer per host thread, grown as
//  needed and kept: no cudaMalloc / cudaFree (which synchronizes the device) per call. Not freed
//  at exit (the context may be gone by then: the process's memory). For mkn.gpu, eventually
struct ThrustScratch
{
    using value_type = char;
    char* data       = nullptr;
    std::size_t size = 0;

    char* allocate(std::ptrdiff_t const n)
    {
        auto const want = static_cast<std::size_t>(n);
        if (want > size)
        {
            if (data)
                mkn::gpu::destroy(data);
            size = want + want / 2;
            mkn::gpu::alloc(data, size);
        }
        return data;
    }
    void deallocate(char*, std::size_t) {}
};
inline ThrustScratch& thrust_scratch()
{
    static thread_local ThrustScratch scratch;
    return scratch;
}
} // namespace plamn::core::detail
#endif

namespace plamn::core
{

// per cell offsets type: particles per patch fit in 32 bits, and device atomics on it are cheap
using chunk_offset_t = std::uint32_t;


// the chunks view shared by the vector and the span: shell box + starts and sizes pointers
template<std::size_t dim>
struct Chunks
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


// what a chunked vector holds beside its particles (container_t: its allocator's vector)
template<std::size_t dim, template<typename> typename container_t>
struct ChunkState
{
    using box_t = damn::core::Box<int, dim>;

    ChunkState(box_t const& box = {})
        : box_{box}
    {
    }

    auto& box() const { return box_; }
    auto starts_data() const { return starts_.data(); }
    auto sizes_data() const { return sizes_.data(); }

    // the chunks are stale after any change to the particles but resettle / insert / remove,
    //  until the next sort()
    void unsort() { sorted_ = false; }

    // gaps between the chunks (set_slack): 0, none, a sort on every change of cell
    NO_DISCARD bool has_slack() const { return slack_ > 0; }

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
    container_t<chunk_offset_t> slots_;  // per particle slot in its cell, sort() scratch; the
                                         //  device sync's fates before it
    container_t<chunk_offset_t> counts_; // the device sync's counters (kept: no allocation)
    double slack_            = 0;
    box_t slack_domain_      = {};
    double border_weight_    = 3;
    bool sorted_             = false;
    std::size_t sorted_size_ = 0;
    // the host resettle's (slot, chunk) of particles changing chunk (add_crosser): with the owned
    //  state, not in ChunkedOps (spans stay plain data, device copyable)
    std::vector<std::pair<std::size_t, std::size_t>> crossers_;
    bool crossers_valid_ = false;
};


template<typename Derived, std::size_t dim, auto alloc_mode>
class ChunkedOps
{
public:
    using box_t      = damn::core::Box<int, dim>;
    using Particle_t = Particle<dim>;

    // the cells the chunks cover
    auto shell() const { return damn::core::grow(self().box(), 1); }

    // particles of cell, as {start, size}
    auto cell_range(auto const& cell) const
    {
        DAMN_ASSERT(self().sorted());
        auto const c = Chunks<dim>::flat(shell(), cell);
        return std::pair<std::size_t, std::size_t>{self().starts_[c], self().sizes_[c]};
    }

    // fn(start, size) per contiguous range of the particles in box, see Chunks
    void for_each_range(box_t const& box, auto&& fn) const
    {
        DAMN_ASSERT(self().sorted());
        Chunks<dim>::for_each_range(shell(), self().starts_data(), self().sizes_data(), box, fn);
    }

    // the particles, not the gaps (sorted)
    NO_DISCARD std::size_t nbr_particles() const
    {
        std::size_t n = 0;
        for_each_range(shell(), [&](auto, auto const size) { n += size; });
        return n;
    }

    std::size_t count(box_t const& box) const
    {
        std::size_t n = 0;
        for_each_range(box, [&](auto, auto const size) { n += size; });
        return n;
    }

    std::size_t nbr_particles_in(box_t const& box) const { return count(box); }
    std::size_t nbr_particles_in(std::array<int, dim> const& cell) const
    {
        return cell_range(cell).second;
    }

    // a particle into cell's chunk, its tail or head gap, whichever is larger: false if neither
    //  has room (or cell is outside the shell, or the array was appended to since its sort)
    bool insert(auto const& particle);

    // the particle of slot i, in the chunk of cell c (flat), out: the end of the chunk with the
    //  smaller gap next to it fills the hole, so the gaps either side even out
    void remove(std::size_t c, std::size_t i);

    // after particles' cells changed (a push): each chunk's particles no longer in its cell taken
    //  out; those keep(cell) (or in the box keep) inserted into their new chunk (else a full sort,
    //  they appended), the others given to leave(particle). Returns whether it sorted
    bool resettle(auto const& keep, auto&& leave);

    // counting sort by cell over shell(), the particles moved whole, chunks laid out with the
    //  slack's gaps between them (weight 0 slots, dropped: gaps)
    void sort();

    // sorted without gaps: the particles only, in cell order (e.g. to be written out); sort()
    //  lays the gaps out again
    void compact()
    {
        auto const slack = self().slack_;
        self().slack_    = 0;
        sort();
        self().slack_ = slack;
    }

    // the push's record of the particles that changed cell (their slot, the chunk of the cell
    //  they left): resettle() then takes them instead of scanning every chunk. Only a push of a
    //  sorted array records (every particle that crossed), sort() drops it, resettle() uses it up
    void begin_crossers()
    {
        self().crossers_.clear();
        self().crossers_valid_ = self().sorted();
    }
    void add_crosser(std::size_t const slot, auto const& from)
    {
        damn::core::Point<int, dim> cell;
        for (std::size_t d = 0; d < dim; ++d)
            cell[d] = from[d];
        self().crossers_.emplace_back(slot, Chunks<dim>::flat(shell(), cell));
    }

    // a gap slot of chunk c (flat): weight 0, the chunk's cell
    void filler_(std::size_t const i, std::size_t const c) { self().write_filler_(i, cell_of_(c)); }

protected:
    auto& self() { return static_cast<Derived&>(*this); }
    auto& self() const { return static_cast<Derived const&>(*this); }


    // take the recorded crossers out of their chunks: their slots filled from the chunks' tails
    //  (each chunk's crossers are consecutive and ascending: one pass per chunk)
    void take_crossers_(std::vector<Particle_t>& moved);

    // the cell of flat index c in the shell
    std::array<int, dim> cell_of_(std::size_t c) const
    {
        auto const shell = this->shell();
        std::array<int, dim> cell;
        for (std::size_t d = dim; d-- > 0;)
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
            if (self().sizes_[k])
                return self().starts_[k] + self().sizes_[k];
        return 0;
    }
    NO_DISCARD std::size_t next_start_(std::size_t const c) const
    {
        for (std::size_t k = c + 1; k < self().sizes_.size(); ++k)
            if (self().sizes_[k])
                return self().starts_[k];
        return self().size();
    }

    // chunks a slide may cascade through (each moves one particle), before a full sort
    static constexpr std::size_t max_slides = 8;

    // the next non empty chunk after c moved up one slot into its tail gap, making that gap first
    //  by sliding the chunk after it (and so on, up to depth chunks): a free slot after c
    bool slide_next_(std::size_t const c, std::size_t const depth = max_slides)
    {
        for (std::size_t k = c + 1; k < self().sizes_.size(); ++k)
            if (self().sizes_[k])
            {
                auto& start = self().starts_[k];
                auto& n     = self().sizes_[k];
                if (next_start_(k) == start + n and (depth == 1 or !slide_next_(k, depth - 1)))
                    return false;
                self().move_slot_(start + n, start);
                filler_(start, c);
                ++start;
                return true;
            }
        return false;
    }
    // the previous non empty chunk before c moved down one slot into its head gap, the same way
    bool slide_prev_(std::size_t const c, std::size_t const depth = max_slides)
    {
        for (std::size_t k = c; k-- > 0;)
            if (self().sizes_[k])
            {
                auto& start = self().starts_[k];
                auto& n     = self().sizes_[k];
                if (prev_end_(k) == start and (depth == 1 or !slide_prev_(k, depth - 1)))
                    return false;
                self().move_slot_(start - 1, start + n - 1);
                filler_(start + n - 1, k);
                --start;
                return true;
            }
        return false;
    }
};


template<typename Derived, std::size_t dim, auto alloc_mode>
bool ChunkedOps<Derived, dim, alloc_mode>::insert(auto const& particle)
{
    if (!self().sorted()) // appended to since: the chunks no longer bound the particles
        return false;
    auto const shell = this->shell();
    damn::core::Point<int, dim> cell;
    for (std::size_t d = 0; d < dim; ++d)
        cell[d] = particle.iCell()[d];
    if (!isIn(cell, shell))
        return false;
    auto const c     = Chunks<dim>::flat(shell, cell);
    auto& start      = self().starts_[c];
    auto& n          = self().sizes_[c];
    auto const lower = prev_end_(c);
    auto const upper = next_start_(c);
    if (n == 0)
    { // the middle of the gap it is in
        if (upper == lower)
            return false;
        start = static_cast<chunk_offset_t>(lower + (upper - lower) / 2);
        self().write_slot_(start, particle);
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
        self().write_slot_(start + n, particle);
    else
        self().write_slot_(--start, particle);
    ++n;
    return true;
}

template<typename Derived, std::size_t dim, auto alloc_mode>
void ChunkedOps<Derived, dim, alloc_mode>::remove(std::size_t const c, std::size_t const i)
{
    auto& start = self().starts_[c];
    auto& n     = self().sizes_[c];
    DAMN_ASSERT(n > 0 and i >= start and i < start + n);
    std::size_t const head = start - prev_end_(c), tail = next_start_(c) - (start + n);
    if (tail <= head)
    { // the last fills the hole, the tail gap grows
        auto const last = start + n - 1;
        if (i != last)
            self().move_slot_(i, last);
        filler_(last, c);
    }
    else
    { // the first fills it, the head gap grows
        if (i != start)
            self().move_slot_(i, start);
        filler_(start, c);
        ++start;
    }
    --n;
}

template<typename Derived, std::size_t dim, auto alloc_mode>
void ChunkedOps<Derived, dim, alloc_mode>::take_crossers_(std::vector<Particle_t>& moved)
{
    DAMN_LOG_SCOPE(3, "SoAVXChunked::take_crossers");
    self().crossers_valid_ = false;
    moved.reserve(self().crossers_.size());
    for (auto const& [slot, c] : self().crossers_)
    {
        moved.push_back(self().copy(slot));
        self().set_weight_(slot, 0); // a hole until filled
    }
    for (std::size_t k = 0; k < self().crossers_.size();)
    {
        auto const c = self().crossers_[k].second;
        auto& n      = self().sizes_[c];
        auto end     = static_cast<std::size_t>(self().starts_[c]) + n;
        for (; k < self().crossers_.size() and self().crossers_[k].second == c; ++k)
        {
            auto const hole = self().crossers_[k].first;
            while (end > hole and self().weight_at_(end - 1) == 0) // holes at the tail: gap
                filler_(--end, c);
            if (end <= hole)
                continue; // trimmed off with the tail
            self().move_slot_(hole, end - 1);
            filler_(--end, c);
        }
        n = static_cast<std::remove_reference_t<decltype(n)>>(end - self().starts_[c]);
    }
    self().crossers_.clear();
}


template<typename Derived, std::size_t dim, auto alloc_mode>
bool ChunkedOps<Derived, dim, alloc_mode>::resettle(auto const& keep, auto&& leave)
{
    auto const kept = [&](auto const& cell) {
        if constexpr (std::is_same_v<std::decay_t<decltype(keep)>, box_t>)
            return isIn(damn::core::Point<int, dim>{cell}, keep);
        else
            return keep(cell);
    };
    DAMN_LOG_SCOPE(2, "SoAVXChunked::resettle");
    DAMN_ASSERT(self().sorted());
    std::vector<Particle_t> moved; // the particles that changed cell
    if (self().crossers_valid_)
        take_crossers_(moved);
    else
        for (std::size_t c = 0; c < self().sizes_.size(); ++c)
        {
            auto const cell = cell_of_(c);
            for (std::size_t i = self().starts_[c]; i < self().starts_[c] + self().sizes_[c];)
            {
                if (self().cell_at_(i) == cell)
                {
                    ++i;
                    continue;
                }
                moved.push_back(self().copy(i));
                remove(c, i); // slot i holds another particle of the chunk now, or is its head gap
                if (i < self().starts_[c])
                    i = self().starts_[c];
            }
        }
    std::vector<Particle_t> overflow; // no room in their chunk
    for (auto const& p : moved)
    {
        if (!kept(p.iCell()))
            leave(p);
        else if (!insert(p))
            overflow.push_back(p);
    }
    if (overflow.empty())
        return false;
    for (auto const& p : overflow)
        self().push_back(p);
    sort();
    return true;
}


template<typename Derived, std::size_t dim, auto alloc_mode>
void ChunkedOps<Derived, dim, alloc_mode>::sort()
{
    DAMN_LOG_SCOPE(2, "SoAVXChunked::sort");
    self().crossers_valid_     = false; // the slots change
    auto static constexpr gpu  = alloc_mode == damn::AllocatorMode::GPU_UNIFIED;
    using Op                   = damn::core::Operators<chunk_offset_t, /*atomic=*/gpu, /*GPU=*/gpu>;
    auto static constexpr none = std::numeric_limits<chunk_offset_t>::max(); // a gap's slot

    auto& s                 = self();
    auto const shell        = this->shell();
    std::size_t const n     = s.size();
    std::size_t const cells = shell.size();

    damn::resize(s.starts_, cells, /*copy=*/false);
    damn::resize(s.sizes_, cells, /*copy=*/false);
    damn::resize(s.gaps_, cells, /*copy=*/false);
    damn::resize(s.slots_, n, /*copy=*/false);

    auto* const sizes   = s.sizes_.data();
    auto* const slots   = s.slots_.data();
    auto const weights  = s.slot_weights_();
    auto const cells_of = s.slot_cells_();
    auto const cell_idx = [=] DAMN_ALL(std::size_t const i) {
        auto const cell = cells_of(i);
        DAMN_ASSERT(isIn(damn::core::Point{cell}, shell));
        return Chunks<dim>::flat(shell, cell);
    };

    // 1. each particle's slot in its cell, the cells' sizes (gaps: weight 0, dropped)
    auto const count = [=] DAMN_ALL(std::size_t const i) {
        slots[i] = weights(i) == 0 ? none : Op{sizes[cell_idx(i)]}.increment_return_old();
    };
    auto* const starts = s.starts_.data();
    if constexpr (gpu)
    { // all on the device, no gaps (the slack serves the host's resettle): the host reads the
      //  total only
#if DAMN_HAVE_THRUST
        DAMN_ASSERT(s.slack_ == 0);
        thrust::fill(thrust::device, sizes, sizes + cells, chunk_offset_t{0});
        auto const first = thrust::counting_iterator<std::size_t>(0);
        thrust::for_each(thrust::device, first, first + n, count);
        thrust::exclusive_scan(thrust::device(detail::thrust_scratch()), sizes, sizes + cells,
                               starts);
        // the total: the last cell's start + size (no reduce: no launch, no temporaries)
        std::array<chunk_offset_t, 2> last{};
        if (cells)
        {
            mkn::gpu::take(starts + cells - 1, &last[0], 1); // a memcpy: no host page fault
            mkn::gpu::take(sizes + cells - 1, &last[1], 1);
        }
        std::size_t const slots_total = std::size_t{last[0]} + last[1];
        auto const to                 = [=] DAMN_ALL(std::size_t const i) -> std::size_t {
            return slots[i] == none ? std::numeric_limits<std::size_t>::max()
                                    : starts[cell_idx(i)] + slots[i];
        };
        s.scatter_sorted_(to, n, slots_total);
        s.sorted_      = true;
        s.sorted_size_ = slots_total;
        return;
#else
        throw std::runtime_error("SoAVXChunked GPU sort needs thrust");
#endif
    }
    std::fill(sizes, sizes + cells, chunk_offset_t{0});
    for (std::size_t i = 0; i < n; ++i)
        count(i);

    // 2. the gaps (the slack over the box's cells, border cells weighted), the chunks' starts
    std::size_t particles = 0;
    for (std::size_t c = 0; c < cells; ++c)
        particles += sizes[c];
    auto* const gaps = s.gaps_.data();
    std::fill(gaps, gaps + cells, chunk_offset_t{0});
    if (auto const slack = static_cast<std::size_t>(s.slack_ * static_cast<double>(particles)))
    {
        auto const box    = s.slack_domain_;
        double total      = 0;
        auto const weight = [&](auto const& cell) {
            if (!isIn(damn::core::Point<int, dim>{cell}, box))
                return 0.;
            for (std::size_t d = 0; d < dim; ++d)
                if (cell[d] == box.lower[d] or cell[d] == box.upper[d])
                    return s.border_weight_;
            return 1.;
        };
        for (std::size_t c = 0; c < cells; ++c)
            total += weight(cell_of_(c));
        // the running total rounded: slack slots in all, each cell its share (an interior cell
        //  whose share is under 1 has one now and then, not never)
        double running        = 0;
        std::size_t allocated = 0;
        for (std::size_t c = 0; c < cells; ++c)
        {
            running += static_cast<double>(slack) * weight(cell_of_(c)) / total;
            auto const upto = static_cast<std::size_t>(std::llround(running));
            gaps[c]         = static_cast<chunk_offset_t>(upto - allocated);
            allocated       = upto;
        }
    }
    std::size_t slots_total = 0;
    for (std::size_t c = 0; c < cells; ++c)
    {
        starts[c] = static_cast<chunk_offset_t>(slots_total);
        slots_total += sizes[c] + gaps[c];
    }

    // 3. the particles into their slots of the scratch storage, swapped in, the gaps filled
    auto const to = [=] DAMN_ALL(std::size_t const i) -> std::size_t {
        return slots[i] == none ? std::numeric_limits<std::size_t>::max()
                                : starts[cell_idx(i)] + slots[i];
    };
    s.scatter_sorted_(to, n, slots_total);
    for (std::size_t c = 0; c < cells; ++c)
        for (std::size_t i = starts[c] + sizes[c]; i < starts[c] + sizes[c] + gaps[c]; ++i)
            filler_(i, c);

    s.sorted_      = true;
    s.sorted_size_ = slots_total;
}

} // namespace plamn::core

#endif /* PLAMN_CORE_DATA_PARTICLES_PARTICLES_CHUNKED_HPP */
