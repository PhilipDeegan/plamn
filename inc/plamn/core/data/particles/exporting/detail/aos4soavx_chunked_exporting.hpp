#ifndef PLAMN_CORE_DATA_PARTICLES_EXPORTING_DETAIL_AOS4SOAVX_CHUNKED_EXPORTING
#define PLAMN_CORE_DATA_PARTICLES_EXPORTING_DETAIL_AOS4SOAVX_CHUNKED_EXPORTING

// AoS4SoAVXChunked (host): one pass over the particle structures, those staying compacted in
//  order, those moving appended to dst; the chunks' gaps (weight 0) never move, they are dropped
//  with the rest of src's tail. Every operation ends in a resize, which unsorts: the offsets are
//  rebuilt by the following on_moved / on_appended

#include "damn/core/def/damn_config.hpp"
#include "damn/core/geometry/box.hpp" // IWYU pragma: keep (isIn on its boxes)
#include "damn/core/geometry/point.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/data/particles/exporting/detail/def_exporting.hpp" // IWYU pragma: keep (the primary template)

#include <cstddef>

namespace plamn::core
{

template<>
struct ParticlesExporter<LayoutMode::AoS4SoAVXChunked, damn::AllocatorMode::CPU>
{
    auto constexpr static layout_mode = LayoutMode::AoS4SoAVXChunked;
    auto constexpr static alloc_mode  = damn::AllocatorMode::CPU;

    template<typename Src, std::size_t dim>
    void delete_particles_in(Src& src, damn::core::Box<int, dim> const& box)
    {
        keep_(src, [&](auto const& p) { return !in_(p, box); });
    }

    template<typename Src, typename Boxes>
    void delete_particles_in(Src& src, Boxes const& boxes)
    {
        keep_(src, [&](auto const& p) { return !in_any_(p, boxes); });
    }

    template<typename Src, std::size_t dim>
    void delete_particles_not_in(Src& src, damn::core::Box<int, dim> const& box)
    {
        keep_(src, [&](auto const& p) { return in_(p, box); });
    }

    template<typename Src, typename Boxes>
    void delete_particles_not_in(Src& src, Boxes const& boxes)
    {
        keep_(src, [&](auto const& p) { return in_any_(p, boxes); });
    }

    template<typename Dst, typename Src, std::size_t dim>
    void move_in_domain(Dst& dst, Src& src, damn::core::Box<int, dim> const& domain_box)
    {
        move_(dst, src, [&](auto const& p) { return in_(p, domain_box); });
    }

    template<typename Dst, typename Src, std::size_t dim>
    void move_in_ghost_layer(Dst& dst, Src& src, damn::core::Box<int, dim> const& domain_box,
                             damn::core::Box<int, dim> const& ghost_box)
    {
        move_(dst, src, [&](auto const& p) { return in_(p, ghost_box) and !in_(p, domain_box); });
    }

    template<typename Dst, typename Src, std::size_t dim, typename Boxes>
    void move_in_ghost_layer(Dst& dst, Src& src, damn::core::Box<int, dim> const& domain_box,
                             Boxes const& ghost_boxes)
    {
        for (auto const& gb : ghost_boxes)
            move_in_ghost_layer(dst, src, domain_box, gb);
    }

private:
    template<std::size_t dim>
    static bool in_(auto const& p, damn::core::Box<int, dim> const& box)
    {
        return isIn(damn::core::Point<int, dim>{p.iCell()}, box);
    }
    static bool in_any_(auto const& p, auto const& boxes)
    {
        for (auto const& box : boxes)
            if (in_(p, box))
                return true;
        return false;
    }

    // src's particles keep(p) compacted in order (gaps dropped), the rest gone
    static void keep_(auto& src, auto&& keep)
    {
        std::size_t k = 0;
        for (std::size_t i = 0; i < src.size(); ++i)
            if (src[i].weight() != 0 and keep(src[i]))
                src[k++] = src[i];
        src.resize(k);
    }

    // src's particles moves(p) appended to dst, the others compacted in order (gaps dropped)
    static void move_(auto& dst, auto& src, auto&& moves)
    {
        std::size_t k = 0;
        for (std::size_t i = 0; i < src.size(); ++i)
        {
            auto const p = src[i];
            if (p.weight() == 0)
                continue;
            if (moves(p))
                dst.push_back(p);
            else
                src[k++] = p;
        }
        src.resize(k);
    }
};

} // namespace plamn::core

#endif /* PLAMN_CORE_DATA_PARTICLES_EXPORTING_DETAIL_AOS4SOAVX_CHUNKED_EXPORTING */
