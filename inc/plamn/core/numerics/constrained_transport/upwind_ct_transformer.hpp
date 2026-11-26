#ifndef PLAMN_CORE_NUMERICS_CONSTRAINED_TRANSPORT_UPWIND_CT_TRANSFORMER_HPP
#define PLAMN_CORE_NUMERICS_CONSTRAINED_TRANSPORT_UPWIND_CT_TRANSFORMER_HPP

#include "damn/core/def/damn_config.hpp"
#include "damn/core/exec/kernels.hpp" // IWYU pragma: keep (gpu::Vec_t, gpu::TileLauncher: GPU builds)
#include "mkn/kul/for.hpp"
#include "plamn/core/data/grid/detail/detail.hpp"
#include "plamn/core/data/field/field_tiles.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"
#include "plamn/core/numerics/constrained_transport/upwind_ct.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace plamn::core
{

// what one tile's block needs, built on the host: plain views, copyable to the device
template<typename Field_t, typename VecField_t, typename Box_t>
struct UpwindCTTile
{
    VecField_t B, vtx;
    Field_t aLx, aRx, dLx, dRx;
    VecField_t vty;
    Field_t aLy, aRy, dLy, dRy;
    VecField_t vtz;
    Field_t aLz, aRz, dLz, dRz;
    VecField_t E;
    std::array<Box_t, 3> boxes;
};


// E on the edges by upwind constrained transport (UpwindCT), each component with an equation on
//  its domain box, as PHARE's UpwindConstrainedTransport::operator() (ideal)
//  x, y, z: each face direction's vt, aL, aR, dL, dR (GodunovTransformer's outputs)
//  untiled: the patch's fields, or one tile's views (on_tile)
//  tiled:   every tile of the patch, CPU tile by tile, GPU one launch with a block per tile
template<typename Reconstruction>
class UpwindCTTransformer
{
    using CT = UpwindCT<Reconstruction>;

public:
    static auto box_for(auto const& layout, auto const& Ec) { return domain_box(layout, Ec); }

    template<typename Field, typename VecField>
    void operator()(auto const& layout, VecField const& B, VecField const& vtx, Field const& aLx,
                    Field const& aRx, Field const& dLx, Field const& dRx, VecField const& vty,
                    Field const& aLy, Field const& aRy, Field const& dLy, Field const& dRy,
                    VecField const& vtz, Field const& aLz, Field const& aRz, Field const& dLz,
                    Field const& dRz, VecField& E) const
    {
        if constexpr (not is_field_tile_set_v<Field>)
        {
            auto constexpr static dim = std::decay_t<decltype(layout)>::dimension;
            mkn::kul::for_N<3>([&](auto c) {
                if constexpr (CT::template has_equation<c, dim>)
                    // captureless kernel: the fields travel as arguments, device-copyable
                    eval_on_box(
                        E[c], box_for(layout, E[c]),
                        [] DAMN_ALL(auto const& ijk, auto const& B_, auto const& vtx_,
                                    auto const& aLx_, auto const& aRx_, auto const& dLx_,
                                    auto const& dRx_, auto const& vty_, auto const& aLy_,
                                    auto const& aRy_, auto const& dLy_, auto const& dRy_,
                                    auto const& vtz_, auto const& aLz_, auto const& aRz_,
                                    auto const& dLz_, auto const& dRz_, auto& Ec) {
                            Ec(ijk) = CT::template E<decltype(c)::value, dim>(
                                ijk, B_, CTFace{vtx_, aLx_, aRx_, dLx_, dRx_},
                                CTFace{vty_, aLy_, aRy_, dLy_, dRy_},
                                CTFace{vtz_, aLz_, aRz_, dLz_, dRz_});
                        },
                        B, vtx, aLx, aRx, dLx, dRx, vty, aLy, aRy, dLy, dRy, vtz, aLz, aRz, dLz,
                        dRz, E[c]);
            });
        }
        else if constexpr (Field::alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
            on_tiles_gpu_(B, vtx, aLx, aRx, dLx, dRx, vty, aLy, aRy, dLy, dRy, vtz, aLz, aRz, dLz,
                          dRz, E);
        else
            for (std::size_t ti = 0; ti < tile_count(E); ++ti)
                on_tile(ti, B, vtx, aLx, aRx, dLx, dRx, vty, aLy, aRy, dLy, dRy, vtz, aLz, aRz, dLz,
                        dRz, E);
    }

    // Hall: as operator(), E_hall (the faces' jt, rhot read); x, y, z: each face direction's
    //  CTFace with its jt, rhot set; untiled fields only (tiles / GPU: not yet)
    template<typename Face>
    void hall(auto const& layout, auto const& B, Face const& x, Face const& y, Face const& z,
              auto& E) const
    {
        static_assert(not is_field_tile_set_v<std::decay_t<decltype(x.aL)>>,
                      "plamn: Hall MHD on tiles: not yet");
        auto constexpr static dim = std::decay_t<decltype(layout)>::dimension;
        mkn::kul::for_N<3>([&](auto c) {
            if constexpr (CT::template has_equation<c, dim>)
                eval_on_box(
                    E[c], box_for(layout, E[c]),
                    [] DAMN_ALL(auto const& ijk, auto const& B_, auto const& x_, auto const& y_,
                                auto const& z_, auto& Ec) {
                        Ec(ijk) = CT::template E_hall<decltype(c)::value, dim>(ijk, B_, x_, y_, z_);
                    },
                    B, x, y, z, E[c]);
        });
    }

    // operator()'s arguments without the layout: B, x's, y's, z's CT fields, E
    template<typename... Args>
    void on_tile(std::size_t const tile_idx, Args&&... args_then_E) const
    {
        tile_exec_with_layout_at(
            tile_idx, [&](auto& layout, auto&&... tile_views) { (*this)(layout, tile_views...); },
            args_then_E...);
    }

private:
    // block per tile, block threads stride each E component's box
    //  waits for the launch before returning: the per tile items live in this scope
    template<typename Field, typename VecField>
    void on_tiles_gpu_([[maybe_unused]] VecField const& B, [[maybe_unused]] VecField const& vtx,
                       [[maybe_unused]] Field const& aLx, [[maybe_unused]] Field const& aRx,
                       [[maybe_unused]] Field const& dLx, [[maybe_unused]] Field const& dRx,
                       [[maybe_unused]] VecField const& vty, [[maybe_unused]] Field const& aLy,
                       [[maybe_unused]] Field const& aRy, [[maybe_unused]] Field const& dLy,
                       [[maybe_unused]] Field const& dRy, [[maybe_unused]] VecField const& vtz,
                       [[maybe_unused]] Field const& aLz, [[maybe_unused]] Field const& aRz,
                       [[maybe_unused]] Field const& dLz, [[maybe_unused]] Field const& dRz,
                       [[maybe_unused]] VecField& E) const
    {
#if DAMN_HAVE_GPU
        using Launcher = damn::core::gpu::TileLauncher</*sync=*/true>;

        auto const make_item = [&](std::size_t const ti) {
            auto const& layout = E[0][ti].layout();
            auto Et            = tile_at(E, ti);
            auto const boxes
                = mkn::kul::for_N_make_array<3>([&](auto c) { return box_for(layout, Et[c]); });
            auto const s = [&](auto const& f) { return tile_at(f, ti); };
            using Item_t = UpwindCTTile<std::decay_t<decltype(s(aLx))>, decltype(Et),
                                        typename decltype(boxes)::value_type>;
            return Item_t{s(B),   s(vtx), s(aLx), s(aRx), s(dLx), s(dRx), s(vty), s(aLy), s(aRy),
                          s(dLy), s(dRy), s(vtz), s(aLz), s(aRz), s(dLz), s(dRz), Et,     boxes};
        };
        using Item_t = decltype(make_item(0));

        auto const n_tiles = tile_count(E);
        damn::core::gpu::Vec_t<Item_t> items;
        items.reserve(n_tiles);
        for (std::size_t ti = 0; ti < n_tiles; ++ti)
            items.emplace_back(make_item(ti));
        auto* const ip            = items.data();
        auto constexpr static dim = VecField::dimension;

        Launcher{n_tiles}([=] DAMN_ALL() mutable {
            auto& it = ip[Launcher::tile_idx()];
            mkn::kul::for_N<3>([&](auto c) {
                if constexpr (CT::template has_equation<c, dim>)
                    Launcher::for_each_point(it.boxes[c], [&](auto const& ijk) {
                        it.E[c](ijk) = CT::template E<c, dim>(
                            ijk, it.B, CTFace{it.vtx, it.aLx, it.aRx, it.dLx, it.dRx},
                            CTFace{it.vty, it.aLy, it.aRy, it.dLy, it.dRy},
                            CTFace{it.vtz, it.aLz, it.aRz, it.dLz, it.dRz});
                    });
            });
        });
#else
        throw std::runtime_error("UpwindCTTransformer: GPU_UNIFIED tiles need a GPU build");
#endif
    }
};

} // namespace plamn::core

#endif /* PLAMN_CORE_NUMERICS_CONSTRAINED_TRANSPORT_UPWIND_CT_TRANSFORMER_HPP */
