#ifndef PLAMN_CORE_DATA_PARTICLES_EXPORTING_PARTICLES_EXPORTING
#define PLAMN_CORE_DATA_PARTICLES_EXPORTING_PARTICLES_EXPORTING

#include "damn/core/logger.hpp"
#include "damn/core/sys/strings.hpp"

#include "plamn/core/data/particles/exporting/detail/def_exporting.hpp"

#include <cstddef>
#include <string_view>

namespace damn::core
{
template<typename Type, std::size_t dim>
struct Box;
} // namespace damn::core

namespace plamn::core
{

template<typename Src, typename Box_t>
void delete_particles_in(Src& src, Box_t const& box)
{
    using Exporter = ParticlesExporter<Src::layout_mode, Src::alloc_mode>;

    std::string_view constexpr static FN_ID     = "delete_particles,";
    [[maybe_unused]] auto constexpr function_id = damn::core::join_string_views_v<FN_ID, Src::type_id>;
    DAMN_LOG_SCOPE(3, function_id);

    Exporter{}.delete_particles_in(src, box);
}

template<typename Src, typename Box_t>
void delete_particles_not_in(Src& src, Box_t const& box)
{
    using Exporter = ParticlesExporter<Src::layout_mode, Src::alloc_mode>;

    std::string_view constexpr static FN_ID     = "delete_particles,";
    [[maybe_unused]] auto constexpr function_id = damn::core::join_string_views_v<FN_ID, Src::type_id>;
    DAMN_LOG_SCOPE(3, function_id);

    Exporter{}.delete_particles_not_in(src, box);
}


template<typename Dst, typename Src, typename Box_t>
void move_in_domain(Dst& dst, Src& src, Box_t const& domain_box)
{
    static_assert(Dst::layout_mode == Src::layout_mode);
    using Exporter = ParticlesExporter<Src::layout_mode, Src::alloc_mode>;
    Exporter{}.move_in_domain(dst, src, domain_box);
}

template<typename Dst, typename Src, typename T, std::size_t dim> //
void move_in_ghost_layer(Dst& dst, Src& src, damn::core::Box<T, dim> const& domain_box,
                         damn::core::Box<T, dim> const& ghost_box)
{
    static_assert(Dst::layout_mode == Src::layout_mode);
    using Exporter = ParticlesExporter<Src::layout_mode, Src::alloc_mode>;
    Exporter{}.move_in_ghost_layer(dst, src, domain_box, ghost_box);
}

template<typename Dst, typename Src, typename T, std::size_t dim, typename Boxes>
void move_in_ghost_layer(Dst& dst, Src& src, damn::core::Box<T, dim> const& domain_box,
                         Boxes const& ghost_boxes)
{
    static_assert(Dst::layout_mode == Src::layout_mode);
    using Exporter = ParticlesExporter<Src::layout_mode, Src::alloc_mode>;
    Exporter{}.move_in_ghost_layer(dst, src, domain_box, ghost_boxes);
}


} // namespace plamn::core


#endif /* PLAMN_CORE_DATA_PARTICLES_EXPORTING_PARTICLES_EXPORTING */
