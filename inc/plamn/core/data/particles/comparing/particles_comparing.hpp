#include "damn/core/math/equality.hpp"

#include "plamn/core/def.hpp"
#ifndef PLAMN_CORE_DATA_PARTICLES_COMPARING_PARTICLES_COMPARING
#define PLAMN_CORE_DATA_PARTICLES_COMPARING_PARTICLES_COMPARING

#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/data/particles/comparing/detail/def_comparing.hpp"
#include "plamn/core/data/particles/comparing/detail/aos_comparing.hpp"
#include "plamn/core/data/particles/comparing/detail/soa_comparing.hpp"

namespace plamn::core
{


template<auto src_layout_mde, auto src_alloc_mde, auto dst_layout_mde, auto dst_alloc_mde>
template<typename PS0, typename PS1>
damn::core::EqualityReport
ParticlesComparator<src_layout_mde, src_alloc_mde, dst_layout_mde, dst_alloc_mde>::operator()(
    PS0 const& ps0, PS1 const& ps1, double const atol)
{
    return particles_equals(ps0, ps1, atol);
}

} // namespace plamn::core

namespace plamn::core
{


template<typename PS0, typename PS1>
auto static compare_particles(PS0 const& ps0, PS1 const& ps1, double const atol = 1e-15)
{
    using Comparator
        = ParticlesComparator<PS0::layout_mode, PS0::alloc_mode, PS1::layout_mode, PS1::alloc_mode>;

    return Comparator{}(ps0, ps1, atol);
}




} // namespace plamn::core


#endif /* PLAMN_CORE_DATA_PARTICLES_COMPARING_PARTICLES_COMPARING */
