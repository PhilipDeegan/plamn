#ifndef PLAMN_CORE_DATA_PARTICLES_PARTITIONING_PARTICLES_PARTITIONING_HPP
#define PLAMN_CORE_DATA_PARTICLES_PARTITIONING_PARTICLES_PARTITIONING_HPP

#include "damn/core/def/damn_config.hpp"

#include <cstddef>

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

} // namespace plamn::core


#endif /*PLAMN_CORE_DATA_PARTICLES_PARTITIONING_PARTICLES_PARTITIONING_HPP*/
