#ifndef PLAMN_CORE_DATA_PARTICLES_EXPORTING_DETAIL_DEF_EXPORTING
#define PLAMN_CORE_DATA_PARTICLES_EXPORTING_DETAIL_DEF_EXPORTING

#include "plamn/core/def.hpp"
#include "damn/core/meta/type_meta.hpp"

#include <cstdint>

namespace damn
{
enum class AllocatorMode : std::uint16_t;

namespace core
{
    template<typename Type, std::size_t dim>
    struct Box;
} // namespace core
} // namespace damn
namespace plamn::core
{
enum class LayoutMode : std::uint16_t;
} // namespace plamn::core

#include "plamn/core/data/particles/particles_def.hpp"

#include <cstddef>

namespace plamn::core
{

template<auto layout_mde, auto alloc_mde>
struct ParticlesExporter
{
    static_assert(damn::core::all_are<LayoutMode>(layout_mde));
    static_assert(damn::core::all_are<damn::AllocatorMode>(alloc_mde));

    auto constexpr static layout_mode = layout_mde;
    auto constexpr static alloc_mode  = alloc_mde;

    template<typename Src, typename Dst, typename Box_t>
    void move_particles(Src& src, Dst& dst, Box_t const& box, std::size_t const growby = 0);


    template<typename Src, std::size_t dim>
    void delete_particles_in(Src& src, damn::core::Box<int, dim> const& box);

    template<typename Src, typename Boxes>
    void delete_particles_in(Src& src, Boxes const& boxes);

    template<typename Src, std::size_t dim>
    void delete_particles_not_in(Src& src, damn::core::Box<int, dim> const& box);

    template<typename Src, typename Boxes>
    void delete_particles_not_in(Src& src, Boxes const& boxes);

    template<typename Dst, typename Src, std::size_t dim>
    void move_in_domain(Dst& dst, Src& src, damn::core::Box<int, dim> const& domain_box);

    template<typename Dst, typename Src, std::size_t dim>
    void move_in_ghost_layer(Dst& dst, Src& src, damn::core::Box<int, dim> const& domain_box,
                            damn::core::Box<int, dim> const& ghost_box);

    template<typename Dst, typename Src, std::size_t dim, typename Boxes>
    void move_in_ghost_layer(Dst& dst, Src& src, damn::core::Box<int, dim> const& domain_box,
                            Boxes const& ghost_boxes);

    // template<typename Src, typename Dst, typename Box_t, typename Refiner, typename Transformer>
    // void operator()(Src const&, Dst&, Box_t const&, Refiner, Transformer);
};


} // namespace plamn::core


#endif /* PLAMN_CORE_DATA_PARTICLES_EXPORTING_DETAIL_DEF_EXPORTING */
