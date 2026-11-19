#ifndef PLAMN_CORE_NUMERICS_GODUNOV_GODUNOV_TRANSFORMER_HPP
#define PLAMN_CORE_NUMERICS_GODUNOV_GODUNOV_TRANSFORMER_HPP

#include "damn/core/def/damn_config.hpp"
#include "damn/core/exec/kernels.hpp" // IWYU pragma: keep (gpu::Vec_t, gpu::TileLauncher: GPU builds)
#include "damn/core/geometry/box.hpp"
#include "plamn/core/data/grid/detail/detail.hpp"
#include "plamn/core/data/field/field_tiles.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"
#include "plamn/core/numerics/godunov/godunov_face.hpp"
#include "plamn/core/numerics/godunov/hll.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace plamn::core
{

// what one tile's block needs, built on the host: plain views, copyable to the device
template<typename Face_t, typename Field_t, typename VecField_t, typename Box_t>
struct GodunovTile
{
    Face_t face;
    Field_t rho;
    VecField_t V, B;
    Field_t P, F_rho;
    VecField_t F_rhoV, F_B;
    Field_t F_Etot;
    VecField_t vt;
    Field_t aL, aR, dL, dR;
    Box_t box;
};


// the fluxes and upwind constrained transport inputs of every face of direction dir, from the
//  primitive fields: GodunovFace per face on the domain box of the direction's face quantity,
//  grown by the reconstruction's ghosts in the transverse directions (PHARE's evalOnBiggerBox
//  with getGrow), as PHARE's Godunov::operator() for one direction
//  untiled: the patch's fields, or one tile's views (on_tile)
//  tiled:   every tile of the patch, CPU tile by tile, GPU one launch with a block per tile
template<typename Reconstruction, typename Riemann = RusanovRiemann>
class GodunovTransformer
{
    using Face_t = GodunovFace<Reconstruction, Riemann>;

public:
    explicit GodunovTransformer(double const gamma)
        : face_{gamma}
    {
    }

    template<std::size_t dir>
    static auto box_for(auto const& layout, auto const& F_rho)
    {
        auto const domain = domain_box(layout, F_rho);
        using Box_t       = std::decay_t<decltype(domain)>;
        using Type        = std::decay_t<decltype(domain.lower[0])>;
        std::array<Type, Box_t::dimension> by{};
        for (std::size_t d = 0; d < Box_t::dimension; ++d)
            by[d] = d == dir ? 0 : Reconstruction::nghosts;
        return damn::core::grow(domain, by);
    }

    template<std::size_t dir, typename Field, typename VecField>
    void operator()(auto const& layout, Field const& rho, VecField const& V, VecField const& B,
                    Field const& P, Field& F_rho, VecField& F_rhoV, VecField& F_B, Field& F_Etot,
                    VecField& vt, Field& aL, Field& aR, Field& dL, Field& dR) const
    {
        if constexpr (not is_field_tile_set_v<Field>)
            // captureless kernel: the fields and the face travel as arguments, device-copyable
            eval_on_box(
                F_rho, box_for<dir>(layout, F_rho),
                [] DAMN_ALL(auto const& ijk, auto const& face, auto const& rho_, auto const& V_,
                            auto const& B_, auto const& P_, auto&... out) {
                    Face_t::store(ijk, face.template operator()<dir>(ijk, rho_, V_, B_, P_),
                                  out...);
                },
                face_, rho, V, B, P, F_rho, F_rhoV, F_B, F_Etot, vt, aL, aR, dL, dR);
        else if constexpr (Field::alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
            on_tiles_gpu_<dir>(rho, V, B, P, F_rho, F_rhoV, F_B, F_Etot, vt, aL, aR, dL, dR);
        else
            for (std::size_t ti = 0; ti < tile_count(F_rho); ++ti)
                on_tile<dir>(ti, rho, V, B, P, F_rho, F_rhoV, F_B, F_Etot, vt, aL, aR, dL, dR);
    }

    // Hall: as operator(), J in, the faces' current and rho averages (jt, rhot) out for the
    //  constrained transport; untiled fields only (the tiled / GPU paths: not yet)
    template<std::size_t dir, typename Field, typename VecField>
    void hall(auto const& layout, Field const& rho, VecField const& V, VecField const& B,
              Field const& P, VecField const& J, Field& F_rho, VecField& F_rhoV, VecField& F_B,
              Field& F_Etot, VecField& vt, Field& aL, Field& aR, Field& dL, Field& dR, VecField& jt,
              Field& rhot) const
    {
        static_assert(not is_field_tile_set_v<Field>, "plamn: Hall MHD on tiles: not yet");
        eval_on_box(
            F_rho, box_for<dir>(layout, F_rho),
            [] DAMN_ALL(auto const& ijk, auto const& face, auto const& rho_, auto const& V_,
                        auto const& B_, auto const& P_, auto const& J_, auto& jt_, auto& rhot_,
                        auto&... out) {
                auto const r = face.template hall<dir>(ijk, rho_, V_, B_, P_, J_);
                Face_t::store(ijk, r, out...);
                Face_t::store_hall(ijk, r, jt_, rhot_);
            },
            face_, rho, V, B, P, J, jt, rhot, F_rho, F_rhoV, F_B, F_Etot, vt, aL, aR, dL, dR);
    }

    // one tile of the tiled fields, eg as one task of a thread pool
    template<std::size_t dir>
    void on_tile(std::size_t const tile_idx, auto const& rho, auto const& V, auto const& B,
                 auto const& P, auto& F_rho, auto& F_rhoV, auto& F_B, auto& F_Etot, auto& vt,
                 auto& aL, auto& aR, auto& dL, auto& dR) const
    {
        tile_exec_with_layout_at(
            tile_idx,
            [&](auto& layout, auto const& rho_t, auto const& V_t, auto const& B_t, auto const& P_t,
                auto F_rho_t, auto F_rhoV_t, auto F_B_t, auto F_Etot_t, auto vt_t, auto aL_t,
                auto aR_t, auto dL_t, auto dR_t) {
                (*this).template operator()<dir>(layout, rho_t, V_t, B_t, P_t, F_rho_t, F_rhoV_t,
                                                 F_B_t, F_Etot_t, vt_t, aL_t, aR_t, dL_t, dR_t);
            },
            rho, V, B, P, F_rho, F_rhoV, F_B, F_Etot, vt, aL, aR, dL, dR);
    }

private:
    // block per tile, block threads stride the tile's face box
    //  waits for the launch before returning: the per tile items live in this scope
    template<std::size_t dir, typename Field, typename VecField>
    void on_tiles_gpu_([[maybe_unused]] Field const& rho, [[maybe_unused]] VecField const& V,
                       [[maybe_unused]] VecField const& B, [[maybe_unused]] Field const& P,
                       [[maybe_unused]] Field& F_rho, [[maybe_unused]] VecField& F_rhoV,
                       [[maybe_unused]] VecField& F_B, [[maybe_unused]] Field& F_Etot,
                       [[maybe_unused]] VecField& vt, [[maybe_unused]] Field& aL,
                       [[maybe_unused]] Field& aR, [[maybe_unused]] Field& dL,
                       [[maybe_unused]] Field& dR) const
    {
#if DAMN_HAVE_GPU
        using Launcher = damn::core::gpu::TileLauncher</*sync=*/true>;

        auto const make_item = [&](std::size_t const ti) {
            auto const F_rho_t = tile_at(F_rho, ti);
            auto const box     = box_for<dir>(F_rho[ti].layout(), F_rho_t);
            using Item_t
                = GodunovTile<Face_t, std::decay_t<decltype(F_rho_t)>,
                              std::decay_t<decltype(tile_at(V, ti))>, std::decay_t<decltype(box)>>;
            return Item_t{face_,
                          tile_at(rho, ti),
                          tile_at(V, ti),
                          tile_at(B, ti),
                          tile_at(P, ti),
                          F_rho_t,
                          tile_at(F_rhoV, ti),
                          tile_at(F_B, ti),
                          tile_at(F_Etot, ti),
                          tile_at(vt, ti),
                          tile_at(aL, ti),
                          tile_at(aR, ti),
                          tile_at(dL, ti),
                          tile_at(dR, ti),
                          box};
        };
        using Item_t = decltype(make_item(0));

        auto const n_tiles = tile_count(F_rho);
        damn::core::gpu::Vec_t<Item_t> items;
        items.reserve(n_tiles);
        for (std::size_t ti = 0; ti < n_tiles; ++ti)
            items.emplace_back(make_item(ti));
        auto* const ip = items.data();

        Launcher{n_tiles}([=] DAMN_ALL() mutable {
            auto& it = ip[Launcher::tile_idx()];
            Launcher::for_each_point(it.box, [&](auto const& ijk) {
                Face_t::store(ijk, it.face.template operator()<dir>(ijk, it.rho, it.V, it.B, it.P),
                              it.F_rho, it.F_rhoV, it.F_B, it.F_Etot, it.vt, it.aL, it.aR, it.dL,
                              it.dR);
            });
        });
#else
        throw std::runtime_error("GodunovTransformer: GPU_UNIFIED tiles need a GPU build");
#endif
    }

    Face_t face_;
};

} // namespace plamn::core

#endif /* PLAMN_CORE_NUMERICS_GODUNOV_GODUNOV_TRANSFORMER_HPP */
