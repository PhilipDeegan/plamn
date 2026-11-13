#include "plamn/core/def.hpp"
// IWYU pragma: private, include "core/data/particles/serializing/particles_serializing.hpp"


#ifndef PLAMN_CORE_DATA_PARTICLES_SERIALIZING_DETAIL_SOA_SERIALIZING
#define PLAMN_CORE_DATA_PARTICLES_SERIALIZING_DETAIL_SOA_SERIALIZING


#include "plamn/core/data/particles/serializing/detail/def_serializing.hpp"


namespace plamn::core
{

using enum LayoutMode;
using enum damn::AllocatorMode;

// SoA
template<>
template<typename Src>
void ParticlesSerializer<LayoutMode::SoA, CPU>::operator()(std::string const& file_name,
                                                           Src const& src)
{
}


template<>
template<typename Dst, typename Src>
void ParticlesDeserializer<LayoutMode::SoA, CPU>::operator()(std::string const& file_name, Dst& dst)
{
    using enum LayoutMode;

    static_assert(Dst::alloc_mode == Src::alloc_mode);
    static_assert(layout_mode == Src::layout_mode
                  || damn::core::any_in(Src::layout_mode, AoS, AoSMapped, AoSTS, AoSPCTS));

    if constexpr (layout_mode == Src::layout_mode)
    {
        throw std::runtime_error("no impl");
    }
    else if (damn::core::any_in(Src::layout_mode, AoS, AoSMapped))
    {
        //
    }
    else
        throw std::runtime_error("no impl");
}

// SoAMapped
template<>
template<typename Src>
void ParticlesSerializer<LayoutMode::SoAVX, CPU>::operator()(std::string const& file_name,
                                                             Src const& src)
{
}


template<>
template<typename Dst, typename Src>
void ParticlesDeserializer<LayoutMode::SoAVX, CPU>::operator()(std::string const& file_name,
                                                               Dst& dst)
{
    using enum LayoutMode;

    static_assert(Dst::alloc_mode == Src::alloc_mode);
    static_assert(layout_mode == Src::layout_mode
                  || damn::core::any_in(Src::layout_mode, AoS, AoSMapped, AoSTS, AoSPCTS));

    ParticlesDeserializer<Src::layout_mode, CPU>{}.template readN<Src>(
        file_name, [&](auto const& particles) {
            for (auto const& p : particles)
                dst.push_back(p);
        });
}

} // namespace plamn::core


#endif /* PLAMN_CORE_DATA_PARTICLES_SERIALIZING_DETAIL_SOA_SERIALIZING */
