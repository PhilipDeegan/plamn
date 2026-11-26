#ifndef PLAMN_CORE_NUMERICS_PRIMITIVE_CONSERVATIVE_PRIMITIVE_CONSERVATIVE_TRANSFORMER_HPP
#define PLAMN_CORE_NUMERICS_PRIMITIVE_CONSERVATIVE_PRIMITIVE_CONSERVATIVE_TRANSFORMER_HPP

#include "damn/core/def/damn_config.hpp"
#include "damn/core/exec/kernels.hpp" // IWYU pragma: keep (gpu::Vec_t, gpu::TileLauncher: GPU builds)

#include "plamn/core/data/grid/detail/detail.hpp"
#include "plamn/core/data/field/field_tiles.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"
#include "plamn/core/numerics/primitive_conservative/primitive_conservative.hpp"

#include <cstddef>
#include <stdexcept>

namespace plamn::core
{

// what one tile's block needs, built on the host: plain views, copyable to the device
template<typename Fn_t, typename Field_t, typename VecField_t, typename Box_t>
struct ConvertTile
{
    Fn_t fn;
    Field_t a0;
    VecField_t v0, B;
    Field_t a1;
    VecField_t v1;
    Field_t a2;
    Box_t box;
};


// to_conservative: rho, V, B, P -> rhoV, Etot; to_primitive: rho, rhoV, B, Etot -> V, P
//  per cell (ToConservative / ToPrimitive) on the ghost box of rho, as PHARE's converters
//  untiled: the patch's fields, or one tile's views (on_tile)
//  tiled:   every tile of the patch, CPU tile by tile, GPU one launch with a block per tile
class PrimitiveConservativeTransformer
{
public:
    explicit PrimitiveConservativeTransformer(double const gamma)
        : gamma_{gamma}
    {
    }

    static auto box_for(auto const& layout, auto const& rho) { return ghost_box(layout, rho); }

    void to_conservative(auto const& layout, auto const& rho, auto const& V, auto const& B,
                         auto const& P, auto& rhoV, auto& Etot) const
    {
        apply_(ToConservative{gamma_}, layout, rho, V, B, P, rhoV, Etot);
    }

    void to_primitive(auto const& layout, auto const& rho, auto const& rhoV, auto const& B,
                      auto const& Etot, auto& V, auto& P) const
    {
        apply_(ToPrimitive{gamma_}, layout, rho, rhoV, B, Etot, V, P);
    }

    void to_conservative_on_tile(std::size_t const tile_idx, auto const& rho, auto const& V,
                                 auto const& B, auto const& P, auto& rhoV, auto& Etot) const
    {
        on_tile_(ToConservative{gamma_}, tile_idx, rho, V, B, P, rhoV, Etot);
    }

    void to_primitive_on_tile(std::size_t const tile_idx, auto const& rho, auto const& rhoV,
                              auto const& B, auto const& Etot, auto& V, auto& P) const
    {
        on_tile_(ToPrimitive{gamma_}, tile_idx, rho, rhoV, B, Etot, V, P);
    }

private:
    // fn(ijk, a0, v0, B, a1, v1, a2): a0, v0, B, a1 in, v1, a2 out
    template<typename Fn, typename Field, typename VecField>
    void apply_(Fn const fn, auto const& layout, Field const& a0, VecField const& v0,
                VecField const& B, Field const& a1, VecField& v1, Field& a2) const
    {
        if constexpr (not is_field_tile_set_v<Field>)
            // captureless kernel: the fields and fn travel as arguments, device-copyable
            eval_on_box(
                a0, box_for(layout, a0),
                [] DAMN_ALL(auto const& ijk, auto const& fn_, auto&&... args) {
                    fn_(ijk, args...);
                },
                fn, a0, v0, B, a1, v1, a2);
        else if constexpr (Field::alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
            on_tiles_gpu_(fn, a0, v0, B, a1, v1, a2);
        else
            for (std::size_t ti = 0; ti < tile_count(a0); ++ti)
                on_tile_(fn, ti, a0, v0, B, a1, v1, a2);
    }

    void on_tile_(auto const fn, std::size_t const tile_idx, auto const& a0, auto const& v0,
                  auto const& B, auto const& a1, auto& v1, auto& a2) const
    {
        tile_exec_with_layout_at(
            tile_idx,
            [&](auto& layout, auto const& a0_t, auto const& v0_t, auto const& B_t, auto const& a1_t,
                auto v1_t, auto a2_t) { apply_(fn, layout, a0_t, v0_t, B_t, a1_t, v1_t, a2_t); },
            a0, v0, B, a1, v1, a2);
    }

    // block per tile, block threads stride the tile's ghost box of a0
    //  waits for the launch before returning: the per tile items live in this scope
    template<typename Fn, typename Field, typename VecField>
    void on_tiles_gpu_([[maybe_unused]] Fn const fn, [[maybe_unused]] Field const& a0,
                       [[maybe_unused]] VecField const& v0, [[maybe_unused]] VecField const& B,
                       [[maybe_unused]] Field const& a1, [[maybe_unused]] VecField& v1,
                       [[maybe_unused]] Field& a2) const
    {
#if DAMN_HAVE_GPU
        using Launcher = damn::core::gpu::TileLauncher</*sync=*/true>;

        auto const make_item = [&](std::size_t const ti) {
            auto const a0t = tile_at(a0, ti);
            auto const box = box_for(a0[ti].layout(), a0t);
            using Item_t   = ConvertTile<Fn, std::decay_t<decltype(a0t)>,
                                         std::decay_t<decltype(tile_at(v0, ti))>, decltype(box)>;
            return Item_t{fn,
                          a0t,
                          tile_at(v0, ti),
                          tile_at(B, ti),
                          tile_at(a1, ti),
                          tile_at(v1, ti),
                          tile_at(a2, ti),
                          box};
        };
        using Item_t = decltype(make_item(0));

        auto const n_tiles = tile_count(a0);
        damn::core::gpu::Vec_t<Item_t> items;
        items.reserve(n_tiles);
        for (std::size_t ti = 0; ti < n_tiles; ++ti)
            items.emplace_back(make_item(ti));
        auto* const ip = items.data();

        Launcher{n_tiles}([=] DAMN_ALL() mutable {
            auto& it = ip[Launcher::tile_idx()];
            Launcher::for_each_point(it.box, [&](auto const& ijk) {
                it.fn(ijk, it.a0, it.v0, it.B, it.a1, it.v1, it.a2);
            });
        });
#else
        throw std::runtime_error(
            "PrimitiveConservativeTransformer: GPU_UNIFIED tiles need a GPU build");
#endif
    }

    double gamma_;
};

} // namespace plamn::core

#endif /* PLAMN_CORE_NUMERICS_PRIMITIVE_CONSERVATIVE_PRIMITIVE_CONSERVATIVE_TRANSFORMER_HPP */
