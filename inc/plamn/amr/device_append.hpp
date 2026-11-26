#ifndef PLAMN_AMR_DEVICE_APPEND_HPP
#define PLAMN_AMR_DEVICE_APPEND_HPP

// damn's device Append hooks for unified memory SoAVXChunked particles, a few launches, the
//  particles never touched by the host: doc/amr/device_append.md
// sources must be sorted (their chunks give a cell's particles), else the host path runs

#include "damn/core/def/damn_config.hpp"

#if DAMN_HAVE_GPU && DAMN_HAVE_THRUST

#include "damn/amr/copy.hpp" // damn::amr::AppendPiece (device_copy.hpp, after copy.hpp)
#include "damn/core/def/detail/mkn_gpu.hpp"
#include "damn/core/exec/kernels.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/vector.hpp"
#include "plamn/amr/resources.hpp"
#include "plamn/core/data/particles/arrays/particles_chunked.hpp"
#include "plamn/core/data/particles/particle_components.hpp"
#include "plamn/core/def.hpp"

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace plamn::amr
{

namespace detail
{
    template<typename P>
    bool constexpr is_device_chunked = P::layout_mode == core::LayoutMode::SoAVXChunked
                                       and P::alloc_mode == damn::AllocatorMode::GPU_UNIFIED;

    template<std::size_t dim>
    struct AppendCells // a piece's source cells
    {
        core::chunk_offset_t const* starts;
        core::chunk_offset_t const* sizes;
        damn::core::Box<int, dim> shell; // its source's
        damn::core::Box<int, dim> cells; // the piece's, in its source's cells
        std::size_t first;               // its first cell in the launch's range
    };

    template<std::size_t dim>
    struct AppendColumns // a piece's source and destination columns
    {
        std::array<double const*, 1 + dim + 3> src_d; // weight, delta.., v..
        std::array<int const*, dim> src_i;
        std::array<double*, 1 + dim + 3> dst_d;
        std::array<int*, dim> dst_i;
        damn::core::Point<int, dim> shift;
        std::size_t dst_at; // its first particle's index in its destination
        std::size_t begin;  // its first particle in the launch's range
    };

    template<std::size_t dim>
    std::size_t flat_in(damn::core::Box<int, dim> const& box,
                        damn::core::Point<int, dim> const& p) DAMN_ALL
    {
        std::size_t at = 0;
        for (std::size_t d = 0; d < dim; ++d)
            at = at * static_cast<std::size_t>(box.upper[d] - box.lower[d] + 1)
                 + static_cast<std::size_t>(p[d] - box.lower[d]);
        return at;
    }
    template<std::size_t dim>
    damn::core::Point<int, dim> point_in(damn::core::Box<int, dim> const& box,
                                         std::size_t i) DAMN_ALL
    {
        damn::core::Point<int, dim> p;
        for (std::size_t d = dim; d-- > 0;)
        {
            auto const n = static_cast<std::size_t>(box.upper[d] - box.lower[d] + 1);
            p[d]         = box.lower[d] + static_cast<int>(i % n);
            i /= n;
        }
        return p;
    }
    template<std::size_t dim>
    auto columns_of(auto const& c)
    {
        std::array<decltype(&c.weight[0]), 1 + dim + 3> d{};
        std::array<decltype(&c.iCell[0][0]), dim> i{};
        d[0] = &c.weight[0]; // addresses only: nothing read on the host
        for (std::size_t k = 0; k < dim; ++k)
        {
            d[1 + k] = &c.delta[k][0];
            i[k]     = &c.iCell[k][0];
        }
        for (std::size_t k = 0; k < 3; ++k)
            d[1 + dim + k] = &c.v[k][0];
        return std::pair{d, i};
    }

    // the last piece k with pc[k].first <= i (pieces without cells skipped: a later one has the
    //  same first)
    template<std::size_t dim>
    std::size_t last_at_most_first(AppendCells<dim> const* pc, std::size_t const n,
                                   std::size_t const i) DAMN_ALL
    {
        std::size_t lo = 0, hi = n;
        while (hi - lo > 1)
            if (auto const mid = (lo + hi) / 2; pc[mid].first <= i)
                lo = mid;
            else
                hi = mid;
        return lo;
    }

    struct AppendBuffers // kept between copies (no allocation once large enough)
    {
        std::vector<std::size_t, mkn::gpu::NoConstructAllocator<std::size_t>> counts, scan, totals;
    };
    inline AppendBuffers& append_buffers()
    {
        static thread_local AppendBuffers buffers;
        return buffers;
    }
} // namespace detail

template<typename Particles, typename Src, std::uint32_t gw, std::size_t dim>
    requires(detail::is_device_chunked<Particles>)
bool can_append_items_on_device(
    std::vector<damn::amr::AppendPiece<ParticleResource<Particles, gw>, ParticleResource<Src, gw>,
                                       dim>> const& pieces)
{
    for (auto const& piece : pieces)
        if (!piece.src.particles->sorted())
            return false;
    return true;
}

template<typename Particles, typename Src, std::uint32_t gw, std::size_t dim>
    requires(detail::is_device_chunked<Particles>)
void append_items_on_device(
    std::vector<damn::amr::AppendPiece<ParticleResource<Particles, gw>, ParticleResource<Src, gw>,
                                       dim>> const& pieces)
{
    auto& buffers = detail::append_buffers();

    // 1. counts per (piece, cell)
    damn::core::gpu::Vec_t<detail::AppendCells<dim>> cells;
    cells.reserve(pieces.size());
    std::size_t ncells = 0;
    for (auto const& piece : pieces)
    {
        auto const& src = *piece.src.particles;
        damn::core::Point<int, dim> back;
        for (std::size_t d = 0; d < dim; ++d)
            back[d] = -piece.shift[d];
        auto const shell = src.shell();
        auto const in    = damn::core::shift(piece.box, back) * shell;
        auto const box   = in ? *in : damn::core::Box<int, dim>{};
        cells.push_back({src.starts_data(), src.sizes_data(), shell, box, ncells});
        ncells += in ? in->size() : 0;
    }
    if (ncells == 0)
        return;
    damn::resize(buffers.counts, ncells, false);
    damn::resize(buffers.scan, ncells, false);
    damn::resize(buffers.totals, pieces.size(), false);
    {
        auto const* const pc = cells.data();
        auto const n         = cells.size();
        auto* const counts   = buffers.counts.data();
        damn::kernel::launch(ncells, [=] DAMN_ALL() {
            auto const i = damn::kernel::idx();
            auto const k = detail::last_at_most_first(pc, n, i);
            auto const p = detail::point_in(pc[k].cells, i - pc[k].first);
            counts[i]    = pc[k].sizes[core::Chunks<dim>::flat(pc[k].shell, p)];
        });
    }

    // 2. scan, each piece's total
    thrust::exclusive_scan(thrust::device, buffers.counts.data(), buffers.counts.data() + ncells,
                           buffers.scan.data());

    // the last piece's end: ncells
    std::vector<std::size_t> total(pieces.size());
    {
        auto const* const counts = buffers.counts.data();
        auto const* const scan   = buffers.scan.data();
        auto* const totals       = buffers.totals.data();
        auto const* const pc     = cells.data();
        auto const n             = cells.size();
        damn::kernel::launch(n, [=] DAMN_ALL() {
            auto const k   = damn::kernel::idx();
            auto const end = k + 1 < n ? pc[k + 1].first : ncells;
            totals[k]
                = pc[k].first == end ? 0 : scan[end - 1] + counts[end - 1] - scan[pc[k].first];
        });
        for (std::size_t k = 0; k < n; ++k)
            total[k] = buffers.totals[k];
    }

    // 3. destinations resized, pieces placed in plan order
    std::unordered_map<Particles*, std::size_t> end_of;
    std::vector<std::size_t> dst_at(pieces.size());
    for (std::size_t k = 0; k < pieces.size(); ++k)
    {
        auto* dst        = pieces[k].dst.particles;
        auto [it, fresh] = end_of.try_emplace(dst, dst->size());
        dst_at[k]        = it->second;
        it->second += total[k];
    }
    for (auto const& [dst, end] : end_of)
        if (end != dst->size())
            dst->resize(end); // unsorted: items_appended sorts it

    // 4. a thread per particle
    damn::core::gpu::Vec_t<detail::AppendColumns<dim>> columns;
    columns.reserve(pieces.size());
    std::size_t nparticles = 0;
    for (std::size_t k = 0; k < pieces.size(); ++k)
    {
        auto const [sd, si] = detail::columns_of<dim>(core::components(*pieces[k].src.particles));
        auto const [dd, di] = detail::columns_of<dim>(core::components(*pieces[k].dst.particles));
        detail::AppendColumns<dim> col{};
        for (std::size_t c = 0; c < sd.size(); ++c)
        {
            col.src_d[c] = sd[c];
            col.dst_d[c] = dd[c];
        }
        for (std::size_t c = 0; c < dim; ++c)
        {
            col.src_i[c] = si[c];
            col.dst_i[c] = di[c];
        }
        col.shift  = pieces[k].shift;
        col.dst_at = dst_at[k];
        col.begin  = nparticles;
        columns.push_back(col);
        nparticles += total[k];
    }
    if (nparticles == 0)
        return;
    {
        auto const* const pc   = cells.data();
        auto const* const cols = columns.data();
        auto const* const scan = buffers.scan.data();
        auto const n           = cells.size();
        damn::kernel::launch(nparticles, [=] DAMN_ALL() {
            auto const i = damn::kernel::idx();
            // its cell: the last with scan <= i and a count (the first of equal scans is empty)
            std::size_t lo = 0, hi = ncells;
            while (hi - lo > 1)
                if (auto const mid = (lo + hi) / 2; scan[mid] <= i)
                    lo = mid;
                else
                    hi = mid;
            auto const cell = lo;
            auto const k    = detail::last_at_most_first(pc, n, cell);
            auto const p    = detail::point_in(pc[k].cells, cell - pc[k].first);
            auto const from
                = pc[k].starts[core::Chunks<dim>::flat(pc[k].shell, p)] + (i - scan[cell]);
            auto const& col = cols[k];
            auto const to   = col.dst_at + (i - col.begin);
            for (std::size_t c = 0; c < 1 + dim + 3; ++c)
                col.dst_d[c][to] = col.src_d[c][from];
            for (std::size_t d = 0; d < dim; ++d)
                col.dst_i[d][to] = col.src_i[d][from] + col.shift[d];
        });
    }
}

} // namespace plamn::amr

#endif // DAMN_HAVE_GPU && DAMN_HAVE_THRUST

#endif /* PLAMN_AMR_DEVICE_APPEND_HPP */
