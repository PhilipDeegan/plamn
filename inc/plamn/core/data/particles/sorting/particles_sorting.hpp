#ifndef PLAMN_CORE_DATA_PARTICLES_SORTING_PARTICLES_SORTING_HPP
#define PLAMN_CORE_DATA_PARTICLES_SORTING_PARTICLES_SORTING_HPP

#include "damn/core/geometry/box.hpp"
#include "damn/core/def/detail/mkn_gpu.hpp"
#include "damn/core/logger.hpp"
#include "damn/core/meta/logic.hpp"
#include "damn/core/sys/strings.hpp"

#include "plamn/core/data/particles/particles_def.hpp"

#include "phlop/timing/scope.hpp"

#include <type_traits>
#include <cstddef>
#include <string_view>

namespace plamn::core::detail
{
template<auto alloc_mode, std::size_t impl, typename Particles>
class ParticleSorter;
} // namespace plamn::core::detail

// the specializations: after the primary template above
#include "particles_sorting_cpu.hpp" // IWYU pragma: keep

#if DAMN_HAVE_MKN_GPU
#include "particles_sorting_gpu_mkn.hpp" // IWYU pragma: keep
#else // no impl (yet)
namespace plamn::core::detail
{
template<typename Particles>
class ParticleSorter<damn::AllocatorMode::GPU_UNIFIED, /*impl = */ 0, Particles>;
// template<typename Particles>
// class ParticleSorter<alloc_mode::GPU, /*impl = */ 0, Particles>;
} // namespace plamn::core::detail
#endif



namespace plamn::core
{



template<typename Particles, std::size_t impl = 0>
struct ParticlesSorter
{
    using LM        = LayoutMode;
    using box_t     = damn::core::Box<int, Particles::dimension>;
    using sort_impl = detail::ParticleSorter<Particles::alloc_mode, impl, Particles>;

    bool constexpr static is_sortable()
    {
        return true; //! any_in(Particles::layout_mode, AoSTS, SoATS);
    }

    bool constexpr static is_cell_sortable()
    {
        return !damn::core::any_in(Particles::layout_mode, LM::AoSPC, LM::SoAPC, LM::AoSPCTS);
    }

    auto constexpr static sortable      = is_sortable();
    auto constexpr static cell_sortable = is_cell_sortable();

    void operator()()
    {
        if constexpr (sortable and cell_sortable)
            sort_impl{particles, box}();
    }

    void by_delta()
    {
        if constexpr (sortable)
            sort_impl{particles, box}.by_delta();
    }


    Particles& particles;
    box_t box;
};

} // namespace plamn::core




namespace plamn::core
{

struct Sortings // to be used in constexpr fashion
{
    bool by_delta = true;
};


template<Sortings S = Sortings{}, typename Particles_t, typename Box_t>
auto& sort_particles(Particles_t&& ps, Box_t const& box)
{
    using Particles                         = std::decay_t<Particles_t>;
    std::string_view constexpr static FN_ID = "sort,";
    auto constexpr function_id              = damn::core::join_string_views_v<FN_ID, Particles::type_id>;
    DAMN_LOG_SCOPE(1, function_id);

    if (ps.size() == 0)
        return ps;

    ParticlesSorter<Particles> sorter{ps, box};
    sorter();
    if constexpr (S.by_delta)
        sorter.by_delta();

    using enum LayoutMode;
    if constexpr (damn::core::any_in(Particles::layout_mode, AoSPC, SoAPC))
        ps.reset_index_wrapper_map();

    return ps;
}



} // namespace plamn::core


#endif /*PLAMN_CORE_DATA_PARTICLES_SORTING_PARTICLES_SORTING_HPP*/
