#ifndef PLAMN_CORE_DATA_PARTICLES_PARTITIONING_PARTICLES_PARTITIONING_HPP
#define PLAMN_CORE_DATA_PARTICLES_PARTITIONING_PARTICLES_PARTITIONING_HPP

#include "damn/core/def/damn_config.hpp"
#include "damn/core/memory/scratch.hpp"
#include "plamn/core/data/particles/selecting/particles_selecting.hpp"

#include <algorithm>
#include <cstddef>
#include <format>
#include <vector>
#include <span>

namespace plamn::core
{
template<auto alloc_mode, typename Particles, std::size_t impl = 0>
class ParticlesPartitioner;
} // namespace plamn::core

// the specializations: after the primary template above
#include "particles_partitioning_cpu.hpp" // IWYU pragma: keep

#if DAMN_HAVE_GPU

#include "particles_partitioning_gpu.hpp"

#else // no impl (yet)

namespace plamn::core
{

// force compile error showing no impl
template<typename Particles>
class ParticlesPartitioner<damn::AllocatorMode::GPU_UNIFIED, Particles,
                           /*impl = */ 0>; // IWYU pragma: keep

} // namespace plamn::core

#endif

namespace plamn::core
{
// template<typename Particles, std::size_t impl = 0>
// struct ParticlesPartitioner
// {
//     using box_t = Box<int, Particles::dimension>;
//     using _impl = detail::ParticlesPartitioner<Particles::alloc_mode, impl,
//     Particles>;


//     template<typename... Args>
//     auto operator()(Args&&... args)
//     {
//         return _impl{particles}(std::forward<Args>(args)...);
//     }

//     template<typename... Args>
//     auto notIn(Args&&... args)
//     {
//         return _impl{particles}.notIn(std::forward<Args>(args)...);
//     }

//     Particles& particles;
//     // box_t box;
// };

template<typename Particles_t>
auto partition_particles(Particles_t& particles, auto const box)
{
    return ParticlesPartitioner<Particles_t::alloc_mode, Particles_t>{particles}(box);
}
template<typename Particles_t>
auto partition_particles_not_in(Particles_t& particles, auto const box)
{
    return ParticlesPartitioner<Particles_t::alloc_mode, Particles_t>{particles}.notIn(box);
}

// the particles in any of boxes last, the others first: returns the pivot, the first of the
//  former. Moved by the array's own swap: AoSMapped's keeps its CellMap
template<typename Particles_t, typename Box_t>
std::size_t partition_particles_not_in(Particles_t& particles, std::vector<Box_t> const& boxes)
{
    std::size_t n = 0; // boxes may overlap: an upper bound, deduplicated below
    for (auto const& box : boxes)
        n += nbr_particles_in(particles, box);
    damn::core::Scratch<std::size_t> buffer{n};
    std::size_t k = 0;
    for (auto const& box : boxes)
        for_each_particle_in(particles, box, [&](std::size_t const i) { buffer[k++] = i; });
    std::sort(buffer.begin(), buffer.end());
    auto const in = buffer.span().first(
        static_cast<std::size_t>(std::unique(buffer.begin(), buffer.end()) - buffer.begin()));
    auto pivot = particles.size();
    // the highest first: those after it in `in` are already past the pivot, so the particle it
    //  is swapped with is never one of them
    for (auto it = in.rbegin(); it != in.rend(); ++it)
        if (--pivot != *it)
            particles.swap(*it, pivot);
    return pivot;
}

} // namespace plamn::core


#endif /*PLAMN_CORE_DATA_PARTICLES_PARTITIONING_PARTICLES_PARTITIONING_HPP*/
