#ifndef PLAMN_CORE_DATA_PARTICLES_SELECTING_PARTICLES_SELECTING
#define PLAMN_CORE_DATA_PARTICLES_SELECTING_PARTICLES_SELECTING

#include "plamn/core/data/particles/selecting/detail/def_selecting.hpp"
#include "plamn/core/data/particles/selecting/detail/aos_selecting.hpp"   // IWYU pragma: keep
#include "plamn/core/data/particles/selecting/detail/soavx_selecting.hpp" // IWYU pragma: keep

#include <cstddef>
#include <type_traits>
#include <vector>

namespace plamn::core
{

// fn(start, size) per contiguous range of src's particles whose cells are in box, by the layout's
//  own means: doc/core/data/particles/selecting/particles_selecting.md
template<typename Src, typename Box_t, typename Fn>
void for_each_range_in(Src const& src, Box_t const& box, Fn&& fn)
{
    using Particles = std::remove_const_t<Src>;
    ParticlesSelector<Particles::layout_mode, Particles::alloc_mode>{}(src, box, fn);
}

template<typename Src, typename Box_t, typename Fn>
void for_each_particle_in(Src const& src, Box_t const& box, Fn&& fn)
{
    for_each_range_in(src, box, [&](std::size_t const start, std::size_t const size) {
        for (std::size_t i = start; i < start + size; ++i)
            fn(i);
    });
}

template<typename Src, typename Box_t>
std::vector<std::size_t> particle_indexes_in(Src const& src, Box_t const& box)
{
    std::vector<std::size_t> out;
    for_each_particle_in(src, box, [&](auto const i) { out.push_back(i); });
    return out;
}

} // namespace plamn::core


#endif /* PLAMN_CORE_DATA_PARTICLES_SELECTING_PARTICLES_SELECTING */
