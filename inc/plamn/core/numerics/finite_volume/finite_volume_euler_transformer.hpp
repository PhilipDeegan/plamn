#ifndef PLAMN_CORE_NUMERICS_FINITE_VOLUME_FINITE_VOLUME_EULER_TRANSFORMER_HPP
#define PLAMN_CORE_NUMERICS_FINITE_VOLUME_FINITE_VOLUME_EULER_TRANSFORMER_HPP

#include "damn/core/def/damn_config.hpp"
#include "damn/core/exec/kernels.hpp" // IWYU pragma: keep (gpu::Vec_t, gpu::TileLauncher: GPU builds)

#include "plamn/core/data/field/field_tiles.hpp"
#include "plamn/core/data/grid/detail/detail.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"
#include "plamn/core/numerics/finite_volume/finite_volume_euler.hpp"

#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace plamn::core
{

// what one tile's block needs, built on the host: plain views, copyable to the device
template<typename Fve_t, typename Field_t, typename Box_t>
struct FiniteVolumeEulerTile
{
    Fve_t fve;
    Field_t U, Fx, Fy, Fz, Unew;
    Box_t box;
};


// Unew = U - dt div(F) for one conserved field on its domain box (FiniteVolumeEuler per cell), as
//  PHARE's FiniteVolumeEulerPerField; called per conserved quantity (rho, rhoV components, Etot)
//  Fx, Fy, Fz: its face fluxes per direction (those of directions >= dim are not read)
//  U and Unew may be the same field: a cell only reads its own U
//  untiled: the patch's fields, or one tile's views (on_tile)
//  tiled:   every tile of the patch, CPU tile by tile, GPU one launch with a block per tile
class FiniteVolumeEulerTransformer
{
public:
    static auto box_for(auto const& layout, auto const& Unew) { return domain_box(layout, Unew); }

    template<typename Field>
    void operator()(auto const& layout, double const dt, Field const& U, Field const& Fx,
                    Field const& Fy, Field const& Fz, Field& Unew) const
    {
        if constexpr (not is_field_tile_set_v<Field>)
        {
            using Layout_t = std::decay_t<decltype(layout)>;
            auto const fve = FiniteVolumeEuler<Layout_t::dimension>::from(layout, dt);
            // captureless kernel: the fields and the functor travel as arguments, device-copyable
            eval_on_box(
                Unew, box_for(layout, Unew),
                [] DAMN_ALL(auto const& ijk, auto const& fve_, auto const& U_, auto const& Fx_,
                            auto const& Fy_, auto const& Fz_,
                            auto& Unew_) { Unew_(ijk) = fve_(ijk, U_, Fx_, Fy_, Fz_); },
                fve, U, Fx, Fy, Fz, Unew);
        }
        else if constexpr (Field::alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
            on_tiles_gpu_(dt, U, Fx, Fy, Fz, Unew);
        else
            for (std::size_t ti = 0; ti < tile_count(Unew); ++ti)
                on_tile(ti, dt, U, Fx, Fy, Fz, Unew);
    }

    void on_tile(std::size_t const tile_idx, double const dt, auto const& U, auto const& Fx,
                 auto const& Fy, auto const& Fz, auto& Unew) const
    {
        tile_exec_with_layout_at(
            tile_idx,
            [&](auto& layout, auto const& U_t, auto const& Fx_t, auto const& Fy_t, auto const& Fz_t,
                auto Unew_t) { (*this)(layout, dt, U_t, Fx_t, Fy_t, Fz_t, Unew_t); },
            U, Fx, Fy, Fz, Unew);
    }

private:
    // block per tile, block threads stride the tile's domain box
    //  waits for the launch before returning: the per tile items live in this scope
    template<typename Field>
    void on_tiles_gpu_([[maybe_unused]] double const dt, [[maybe_unused]] Field const& U,
                       [[maybe_unused]] Field const& Fx, [[maybe_unused]] Field const& Fy,
                       [[maybe_unused]] Field const& Fz, [[maybe_unused]] Field& Unew) const
    {
#if DAMN_HAVE_GPU
        using Launcher = damn::core::gpu::TileLauncher</*sync=*/true>;

        auto const make_item = [&](std::size_t const ti) {
            auto const& layout = Unew[ti].layout();
            using Layout_t     = std::decay_t<decltype(layout)>;
            auto const fve     = FiniteVolumeEuler<Layout_t::dimension>::from(layout, dt);
            auto Un            = tile_at(Unew, ti);
            auto const box     = box_for(layout, Un);
            using Item_t
                = FiniteVolumeEulerTile<std::decay_t<decltype(fve)>, decltype(Un), decltype(box)>;
            return Item_t{
                fve, tile_at(U, ti), tile_at(Fx, ti), tile_at(Fy, ti), tile_at(Fz, ti), Un, box};
        };
        using Item_t = decltype(make_item(0));

        auto const n_tiles = tile_count(Unew);
        damn::core::gpu::Vec_t<Item_t> items;
        items.reserve(n_tiles);
        for (std::size_t ti = 0; ti < n_tiles; ++ti)
            items.emplace_back(make_item(ti));
        auto* const ip = items.data();

        Launcher{n_tiles}([=] DAMN_ALL() mutable {
            auto& it = ip[Launcher::tile_idx()];
            Launcher::for_each_point(it.box, [&](auto const& ijk) {
                it.Unew(ijk) = it.fve(ijk, it.U, it.Fx, it.Fy, it.Fz);
            });
        });
#else
        throw std::runtime_error(
            "FiniteVolumeEulerTransformer: GPU_UNIFIED tiles need a GPU build");
#endif
    }
};

} // namespace plamn::core

#endif /* PLAMN_CORE_NUMERICS_FINITE_VOLUME_FINITE_VOLUME_EULER_TRANSFORMER_HPP */
