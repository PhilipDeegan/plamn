#ifndef PLAMN_CORE_DATA_PARTICLES_EXPORTING_DETAIL_SOAVX_EXPORTING
#define PLAMN_CORE_DATA_PARTICLES_EXPORTING_DETAIL_SOAVX_EXPORTING

// SoAVX: untiled and untracked, particles leaving a box are partitioned column by column (keep
//  flags, their scan for the slots, every column scattered), then moved to dst / dropped from the
//  tail. GPU_UNIFIED partitions and moves on the device (kernels, device copies)

#include "damn/core/def/damn_config.hpp"
#include "damn/core/vector.hpp"
#include "mkn/kul/for.hpp"
#include "damn/core/memory/memory.hpp"
#include "damn/core/exec/kernels.hpp"
#include "damn/core/def/detail/mkn_gpu.hpp"
#include "mkn/gpu/scan.hpp"
#include "plamn/core/data/particles/arrays/particles_soavx.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "mkn/gpu.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

namespace damn::core
{
template<typename Type, std::size_t dim>
struct Box;
} // namespace damn::core


namespace plamn::core
{

template<auto layout_mde, auto alloc_mde>
struct ParticlesExporter;

template<auto alloc_mde>
struct ParticlesExporter<LayoutMode::SoAVX, alloc_mde>
{
    auto constexpr static layout_mode = LayoutMode::SoAVX;
    auto constexpr static alloc_mode  = alloc_mde;

    template<typename Src, std::size_t dim>
    void delete_particles_in(Src& src, damn::core::Box<int, dim> const& box)
    {
        src.resize(partition_(src, [=] DAMN_ALL(auto const& cell) { return !isIn(cell, box); }));
    }

    template<typename Src, typename Boxes>
    void delete_particles_in(Src& src, Boxes const& boxes)
    {
        auto const list = box_list_(boxes);
        auto const view = list.view();
        src.resize(
            partition_(src, [=] DAMN_ALL(auto const& cell) { return !view.contains(cell); }));
    }

    template<typename Src, std::size_t dim>
    void delete_particles_not_in(Src& src, damn::core::Box<int, dim> const& box)
    {
        src.resize(partition_(src, [=] DAMN_ALL(auto const& cell) { return isIn(cell, box); }));
    }

    template<typename Src, typename Boxes>
    void delete_particles_not_in(Src& src, Boxes const& boxes)
    {
        auto const list = box_list_(boxes);
        auto const view = list.view();
        src.resize(partition_(src, [=] DAMN_ALL(auto const& cell) { return view.contains(cell); }));
    }

    template<typename Dst, typename Src, std::size_t dim>
    void move_in_domain(Dst& dst, Src& src, damn::core::Box<int, dim> const& domain_box)
    {
        auto const kept
            = partition_(src, [=] DAMN_ALL(auto const& cell) { return !isIn(cell, domain_box); });
        move_tail_(dst, src, kept);
    }

    template<typename Dst, typename Src, std::size_t dim>
    void move_in_ghost_layer(Dst& dst, Src& src, damn::core::Box<int, dim> const& domain_box,
                             damn::core::Box<int, dim> const& ghost_box)
    {
        auto const list = box_list_(ghost_box.remove(domain_box));
        auto const view = list.view();
        auto const kept
            = partition_(src, [=] DAMN_ALL(auto const& cell) { return !view.contains(cell); });
        move_tail_(dst, src, kept);
    }

    template<typename Dst, typename Src, std::size_t dim, typename Boxes>
    void move_in_ghost_layer(Dst& dst, Src& src, damn::core::Box<int, dim> const& domain_box,
                             Boxes const& ghost_boxes)
    {
        for (auto const& gb : ghost_boxes)
            move_in_ghost_layer(dst, src, domain_box, gb);
    }

private:
    // a box list the partition predicate can read where it runs: copied into this allocator's
    //  memory (unified for GPU_UNIFIED), read through pointer and size
    template<typename Box_t>
    struct BoxListView
    {
        Box_t const* boxes;
        std::size_t size;

        bool contains(auto const& cell) const DAMN_ALL
        {
            for (std::size_t i = 0; i < size; ++i)
                if (isIn(cell, boxes[i]))
                    return true;
            return false;
        }
    };

    template<typename Boxes>
    static auto box_list_(Boxes const& boxes)
    {
        using Box_t    = std::decay_t<typename Boxes::value_type>;
        using Vector_t = typename damn::Vector<Box_t, alloc_mode>::vector_t;
        struct List
        {
            Vector_t boxes;
            auto view() const { return BoxListView<Box_t>{boxes.data(), boxes.size()}; }
        };
        List list{damn::Vector<Box_t, alloc_mode>::make(0)};
        list.boxes.reserve(boxes.size());
        for (auto const& box : boxes)
            list.boxes.push_back(box);
        return list;
    }

    // particles for which keep(iCell) first, then the others, each in their order (stable):
    //  returns how many keep. A particle's slot from the scan of the keep flags (kept before it,
    //  or all kept + the others before it), every column scattered into this thread's scratch
    //  columns, swapped in (no allocation once they fit). GPU_UNIFIED on the device: kernels on
    //  the default stream, the scan's total back on the host (its one wait), a sync at the end
    //  (the moves after it copy on the host's side)
    template<typename Src>
    static std::size_t partition_(Src& src, auto const keep)
    {
        auto constexpr static dim = Src::dimension;
        auto constexpr static gpu = alloc_mode == damn::AllocatorMode::GPU_UNIFIED;
        using Columns_t           = SoAVXVector<dim, alloc_mode>;
        using Slots_t =
            typename damn::Vector<std::uint32_t, alloc_mode, /*no construct*/ 1>::vector_t;

        std::size_t const n = src.size();
        if (n == 0)
            return 0;
        // this thread's, never freed (GPU: the context may be gone at exit)
        thread_local auto* const scratch = new Columns_t{};
        thread_local auto* const slots_v = new Slots_t{};
        damn::resize(*slots_v, n, /*copy=*/false);
        std::apply([&](auto&... c) { (damn::resize(c, n, /*copy=*/false), ...); },
                   scratch->as_tuple());

        // the columns as plain pointer arrays (a kernel captures trivially copyable data only):
        //  the doubles (weight, delta, v), the ints (iCell)
        std::size_t constexpr static n_reals = 1 + dim + 3;
        struct Columns
        {
            std::array<double*, n_reals> reals;
            std::array<int*, dim> cells;
        };
        auto const columns = [](auto& v) {
            Columns c{};
            c.reals[0] = v.weight_.data();
            for (std::size_t d = 0; d < dim; ++d)
            {
                c.reals[1 + d] = v.delta_[d].data();
                c.cells[d]     = v.iCell_[d].data();
            }
            for (std::size_t d = 0; d < 3; ++d)
                c.reals[1 + dim + d] = v.v_[d].data();
            return c;
        };
        auto const from = columns(src);
        auto const to   = columns(*scratch);
        auto* const pos = slots_v->data();
        auto const flag = [=] DAMN_ALL(std::size_t const i) {
            std::array<int, dim> cell;
            for (std::size_t d = 0; d < dim; ++d)
                cell[d] = from.cells[d][i];
            pos[i] = keep(cell) ? 1 : 0;
        };
        auto const scatter
            = [=] DAMN_ALL(std::size_t const i, std::size_t const kept, bool const kept_i) {
                  std::size_t const slot = kept_i ? pos[i] : kept + (i - pos[i]);
                  for (std::size_t c = 0; c < n_reals; ++c)
                      to.reals[c][slot] = from.reals[c][i];
                  for (std::size_t c = 0; c < dim; ++c)
                      to.cells[c][slot] = from.cells[c][i];
              };

        std::size_t kept = 0;
        if constexpr (gpu)
        {
            DAMN_WITH_MKN_GPU({
                using Launcher = damn::core::gpu::DeviceLauncher<mkn::gpu::Guess, /*sync=*/false>;
                Launcher{n}([=] DAMN_ALL() {
                    if (auto const i = damn::kernel::idx(); i < n)
                        flag(i);
                });
                // the flags scanned in place: flag i is then pos[i + 1] - pos[i] (the last's:
                //  kept - pos[n - 1])
                kept = mkn::gpu::exclusive_scan(pos, pos, n);
                Launcher{n}([=] DAMN_ALL() {
                    if (auto const i = damn::kernel::idx(); i < n)
                    {
                        bool const k = (i + 1 < n ? pos[i + 1] : kept) != pos[i];
                        scatter(i, kept, k);
                    }
                });
                mkn::gpu::sync();
            });
        }
        else
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                flag(i);
                auto const k = pos[i];
                pos[i]       = static_cast<std::uint32_t>(kept);
                kept += k;
            }
            for (std::size_t i = 0; i < n; ++i)
                scatter(i, kept, (i + 1 < n ? pos[i + 1] : kept) != pos[i]);
        }

        auto mine   = src.as_tuple();
        auto theirs = scratch->as_tuple();
        mkn::kul::for_N<std::tuple_size_v<decltype(mine)>>(
            [&](auto c) { std::get<c>(mine).swap(std::get<c>(theirs)); });
        return kept;
    }

    // src's particles from `from` onwards appended to dst, src keeps [0, from): dst presized,
    //  then per column the tail copied at its offset (on the device for GPU_UNIFIED)
    static void move_tail_(auto& dst, auto& src, std::size_t const from)
    {
        std::size_t const n      = src.size() - from;
        std::size_t const offset = dst.size();
        if (n > 0)
        {
            auto dst_columns       = dst.as_tuple();
            auto const src_columns = src.as_tuple();
            mkn::kul::for_N<std::tuple_size_v<decltype(dst_columns)>>([&](auto c) {
                auto& d       = std::get<c>(dst_columns);
                auto const& s = std::get<c>(src_columns);
                damn::resize(d, offset + n);
                damn::core::mem::copy<alloc_mode>(d.data() + offset, s.data() + from, n);
            });
        }
        src.resize(from);
    }
};

// SoAVXChunked: the same column moves, every one ends in a resize, which unsorts: the offsets
//  are rebuilt by the following on_moved / on_appended
template<auto alloc_mde>
struct ParticlesExporter<LayoutMode::SoAVXChunked, alloc_mde>
    : public ParticlesExporter<LayoutMode::SoAVX, alloc_mde>
{
    auto constexpr static layout_mode = LayoutMode::SoAVXChunked;
};

} // namespace plamn::core

#endif /* PLAMN_CORE_DATA_PARTICLES_EXPORTING_DETAIL_SOAVX_EXPORTING */
