#include "plamn/core/def.hpp"

#ifndef PLAMN_CORE_DATA_PARTICLES_COMPARING_DETAIL_SOA_COMPARING
#define PLAMN_CORE_DATA_PARTICLES_COMPARING_DETAIL_SOA_COMPARING

#include "damn/core/memory/memory.hpp"
#include "damn/core/math/equality.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/data/particles/arrays/particles_soa.hpp"
#include "plamn/core/data/particles/comparing/detail/def_comparing.hpp"

namespace plamn::core
{
using enum LayoutMode;
using enum damn::AllocatorMode;


// compares via per-index accessors only (iCell(i)/v(i)/delta(i)) - AoS and SoA both
// provide them, regardless of storage layout, and SoA has no flat iterator
template<typename PS0, typename PS1>
damn::core::EqualityReport index_based_particles_equals(PS0 const& ps0, PS1 const& ps1, double const atol)
{
    if (ps0.size() != ps1.size())
        return damn::core::EqualityReport{false, "different sizes: " + std::to_string(ps0.size()) + " vs "
                                         + std::to_string(ps1.size())};

    for (std::size_t i = 0; i < ps0.size(); ++i)
    {
        std::string const idx = std::to_string(i);
        if (ps0.iCell(i) != ps1.iCell(i))
            return damn::core::EqualityReport{false, "icell mismatch at index: " + idx, i};

        if (!damn::core::float_equals(ps0.v(i), ps1.v(i), atol))
            return damn::core::EqualityReport{false, "v mismatch at index: " + idx, i};

        if (!damn::core::float_equals(ps0.delta(i), ps1.delta(i), atol))
            return damn::core::EqualityReport{false, "delta mismatch at index: " + idx, i};
    }

    return damn::core::EqualityReport{true};
}


template<>
template<typename PS0, typename PS1>
damn::core::EqualityReport ParticlesComparator<SoA, CPU, SoA, CPU>::operator()(PS0 const& ps0, PS1 const& ps1,
                                                                   double const atol)
{
    return index_based_particles_equals(ps0, ps1, atol);
}

template<>
template<typename PS0, typename PS1>
damn::core::EqualityReport ParticlesComparator<SoA, CPU, AoS, CPU>::operator()(PS0 const& ps0, PS1 const& ps1,
                                                                   double const atol)
{
    return index_based_particles_equals(ps0, ps1, atol);
}

template<>
template<typename PS0, typename PS1>
damn::core::EqualityReport ParticlesComparator<AoS, CPU, SoA, CPU>::operator()(PS0 const& ps0, PS1 const& ps1,
                                                                   double const atol)
{
    return index_based_particles_equals(ps0, ps1, atol);
}

template<>
template<typename PS0, typename PS1>
damn::core::EqualityReport ParticlesComparator<SoAVX, CPU, AoS, CPU>::operator()(PS0 const& ps0, PS1 const& ps1,
                                                                     double const atol)
{
    return damn::core::EqualityReport{};
}
template<>
template<typename PS0, typename PS1>
damn::core::EqualityReport ParticlesComparator<AoS, CPU, SoAVX, CPU>::operator()(PS0 const& ps0, PS1 const& ps1,
                                                                     double const atol)
{
    return ParticlesComparator<SoAVX, CPU, AoS, CPU>{}(ps1, ps0, atol);
}


} // namespace plamn::core


#endif /* PLAMN_CORE_DATA_PARTICLES_COMPARING_DETAIL_SOA_COMPARING */
