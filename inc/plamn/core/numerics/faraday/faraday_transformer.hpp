#ifndef PLAMN_CORE_NUMERICS_FARADAY_FARADAY_TRANSFORMER_HPP
#define PLAMN_CORE_NUMERICS_FARADAY_FARADAY_TRANSFORMER_HPP

#include "mkn/kul/for.hpp"
#include "damn/core/def/damn_config.hpp"
#include "damn/core/exec/kernels.hpp" // IWYU pragma: keep (gpu::Vec_t, gpu::TileLauncher: GPU builds)
#include "plamn/core/data/field/field_tiles.hpp"
#include "plamn/core/numerics/faraday/faraday.hpp" // IWYU pragma: keep (Faraday: GPU builds)
#include "plamn/core/numerics/differential_operators.hpp"
#include "plamn/core/data/field/field_expr.hpp"
#include "plamn/core/data/grid/layout.hpp"

#include <array>
#include <stdexcept>
#include <cstddef>
#include <type_traits>

#include "damn/core/exec/kernels/region_chunks.hpp" // IWYU pragma: keep (run_gpu: GPU builds)

namespace plamn::core
{

// what one tile's block needs, built on the host: plain views, copyable to the device
template<typename Curl_t, typename VecField_t, typename Box_t>
struct FaradayTile
{
    Curl_t curl;
    VecField_t B, Bnew;
    std::array<Box_t, 3> boxes;
};


// Bnew = B - dt curl(E) over one patch: Faraday per point, on the domain box of each Bnew component
//  untiled: the patch's fields, or one tile's views (on_tile)
//  tiled:   every tile of the patch, CPU tile by tile, GPU one launch with a block per tile
//  Bnew ghosts between tiles are not synced (sync_inner_ghosts)
//  B and Bnew may be the same vecfield (corrector), curl(E) does not read B
class FaradayTransformer
{
public:
    template<typename Layout>
    static auto box_for(Layout const& layout, auto const& Bnc)
    {
        auto constexpr static dim = Layout::dimension;
        return local_domain_box(layout.cells, centering_of<dim>(Bnc),
                                Layout::options.field_ghost_width);
    }

    template<typename Layout, typename VecField>
    void operator()(Layout const& layout, VecField const& B, VecField const& E, VecField& Bnew,
                    double const dt) const
        requires(not has_tiled_field_type_c<VecField>)
    {
        auto constexpr static dim = Layout::dimension;
        auto const curlE          = curl<dim>(layout.dx, E);

        mkn::kul::for_N<3>([&](auto c) {
            assign(Bnew[c], box_for(layout, Bnew[c]), of(B[c]) - dt * curlE.template get<c>());
        });
    }

    template<typename Layout, typename VecField>
    void operator()(Layout const& /*patch layout*/, VecField const& B, VecField const& E,
                    VecField& Bnew, double const dt) const
        requires(has_tiled_field_type_c<VecField>)
    {
        if constexpr (VecField::field_type::alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
            on_tiles_gpu(B, E, Bnew, dt);
        else
            for (std::size_t ti = 0; ti < tile_count(Bnew); ++ti)
                on_tile(ti, B, E, Bnew, dt);
    }

    // GPU_UNIFIED, an item per untiled patch collected across a level (one launch per run_gpu)
    template<typename Layout>
    static auto make_item(Layout const& layout, auto const& B, auto const& E, auto& Bnew)
    {
        auto const Bv = untiled_view(B);
        auto const Ev = untiled_view(E);
        auto Bnv      = untiled_view(Bnew);
        auto const boxes
            = mkn::kul::for_N_make_array<3>([&](auto c) { return box_for(layout, Bnv[c]); });
        auto const curlE = curl(layout, Ev);
        using Box_t      = std::decay_t<decltype(boxes)>::value_type;
        return FaradayTile<std::decay_t<decltype(curlE)>, decltype(Bnv), Box_t>{curlE, Bv, Bnv,
                                                                                boxes};
    }

    // every item's Bnew components on the device, one launch (damn's region chunks)
    static void run_gpu([[maybe_unused]] auto& items, [[maybe_unused]] double const dt)
    {
#if DAMN_HAVE_GPU
        damn::core::run_region_chunks(items, [=] DAMN_ALL(auto& it, auto const c, auto const& ijk) {
            if (c == 0)
                Faraday{}(ijk, it.Bnew[0], it.B[0], it.curl.x, dt);
            else if (c == 1)
                Faraday{}(ijk, it.Bnew[1], it.B[1], it.curl.y, dt);
            else
                Faraday{}(ijk, it.Bnew[2], it.B[2], it.curl.z, dt);
        });
#else
        throw std::runtime_error("FaradayTransformer: run_gpu needs a GPU build");
#endif
    }

    void on_tile(std::size_t const tile_idx, auto const& B, auto const& E, auto& Bnew,
                 double const dt) const
    {
        tile_exec_with_layout_at(
            tile_idx,
            [&](auto& layout, auto const& B_t, auto const& E_t, auto Bnew_t) {
                (*this)(layout, B_t, E_t, Bnew_t, dt);
            },
            B, E, Bnew);
    }

private:
    // block per tile, block threads stride each Bnew component's box
    //  waits for the launch before returning: the per tile items live in this scope
    //   todo: async (no wait), the items must then outlive the launch, see .gpu_todo
    template<typename VecField>
    void on_tiles_gpu([[maybe_unused]] VecField const& B, [[maybe_unused]] VecField const& E,
                      [[maybe_unused]] VecField& Bnew, [[maybe_unused]] double const dt) const
    {
#if DAMN_HAVE_GPU
        using Launcher = damn::core::gpu::TileLauncher</*sync=*/true>;

        auto const make_item = [&](std::size_t const ti) {
            auto const& layout = Bnew[0][ti].layout();
            auto const Bt      = tile_at(B, ti);
            auto const Et      = tile_at(E, ti);
            auto Bnt           = tile_at(Bnew, ti);
            auto const boxes
                = mkn::kul::for_N_make_array<3>([&](auto c) { return box_for(layout, Bnt[c]); });
            auto const curlE = curl(layout, Et);
            using Box_t      = std::decay_t<decltype(boxes)>::value_type;
            using Item_t     = FaradayTile<std::decay_t<decltype(curlE)>, decltype(Bnt), Box_t>;
            return Item_t{curlE, Bt, Bnt, boxes};
        };
        using Item_t = decltype(make_item(0));

        auto const n_tiles = tile_count(Bnew);
        damn::core::gpu::Vec_t<Item_t> items;
        items.reserve(n_tiles);
        for (std::size_t ti = 0; ti < n_tiles; ++ti)
            items.emplace_back(make_item(ti));

        auto* const ip = items.data();
        Launcher{n_tiles}([=] DAMN_ALL() mutable {
            auto& item = ip[Launcher::tile_idx()];
            Launcher::for_each_point(item.boxes[0], [&](auto const& ijk) {
                Faraday{}(ijk, item.Bnew[0], item.B[0], item.curl.x, dt);
            });
            Launcher::for_each_point(item.boxes[1], [&](auto const& ijk) {
                Faraday{}(ijk, item.Bnew[1], item.B[1], item.curl.y, dt);
            });
            Launcher::for_each_point(item.boxes[2], [&](auto const& ijk) {
                Faraday{}(ijk, item.Bnew[2], item.B[2], item.curl.z, dt);
            });
        });
#else
        throw std::runtime_error("FaradayTransformer: GPU_UNIFIED tiles need a GPU build");
#endif
    }
};

} // namespace plamn::core

#endif /* PLAMN_CORE_NUMERICS_FARADAY_FARADAY_TRANSFORMER_HPP */
