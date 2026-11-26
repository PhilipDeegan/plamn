#ifndef PLAMN_CORE_NUMERICS_AMPERE_AMPERE_HPP
#define PLAMN_CORE_NUMERICS_AMPERE_AMPERE_HPP

#include "damn/core/def/damn_config.hpp"

#include <cstdint>

namespace plamn::core
{

enum class AmpereMode : std::uint16_t { ShrinkedGhost, GrownPhysical };

struct AmpereBox // structural type — usable as NTTP
{
    AmpereMode mode      = AmpereMode::ShrinkedGhost;
    std::uint32_t amount = 1;
};


// J = curl(B), one component at one point: curlc is that component of curl(layout, B)
struct Ampere
{
    void operator()(auto const& ijk, auto& Jc, auto const& curlc) const DAMN_ALL
    {
        Jc(ijk) = curlc(ijk);
    }
};

} // namespace plamn::core

#endif
