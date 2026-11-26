// IWYU pragma: private, include "plamn/core/data/particles/serializing/particles_serializing.hpp"
#ifndef PLAMN_CORE_DATA_PARTICLES_SERIALIZING_DETAIL_TS_SERIALIZING
#define PLAMN_CORE_DATA_PARTICLES_SERIALIZING_DETAIL_TS_SERIALIZING

#include "damn/core/def/damn_config.hpp"

#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/data/particles/particles_detail.hpp"
#include "plamn/core/data/particles/particles.hpp"
#include "plamn/core/data/particles/serializing/detail/def_serializing.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace plamn::core
{

using enum LayoutMode;
using enum damn::AllocatorMode;

// AoSTS / AoSPCTS: no single contiguous buffer (tiles, AoSPCTS per cell buckets), one particle
//  at a time, no flat staging buffer. The same Particle_t: the file format is the flat AoS one
template<auto layout_mode>
struct TiledParticlesSerializing
{
    template<typename Src>
    static void write(std::string const& file_name, Src const& src, auto& self)
    {
        auto file = self.open_file_to_write(file_name);
        per_particle(src, [&](auto const& p) { file.write(&p, 1); });
    }

    template<typename Dst, typename Src>
    static void read(std::string const& file_name, Dst& dst, auto& self)
    {
        static_assert(Dst::alloc_mode == Src::alloc_mode);
        using Particle_t = typename Dst::value_type;

        auto file             = self.open_file_from_start(file_name);
        auto const nbr_parts  = file.size() / sizeof(Particle_t);
        Particle_t p;
        for (std::size_t i = 0; i < nbr_parts; ++i)
            dst.emplace_back(*file.read(&p, 1));

        dst.template on_appended<ParticleType::Domain>();
    }
};


template<>
template<typename Src>
void ParticlesSerializer<AoSTS, CPU>::operator()(std::string const& file_name, Src const& src)
{
    TiledParticlesSerializing<AoSTS>::write(file_name, src, *this);
}

template<>
template<typename Dst, typename Src>
void ParticlesDeserializer<AoSTS, CPU>::operator()(std::string const& file_name, Dst& dst)
{
    TiledParticlesSerializing<AoSTS>::template read<Dst, Src>(file_name, dst, *this);
}

// the flat AoS file format: their chunked reader
template<>
template<typename Src, std::uint16_t N, typename Fn>
void ParticlesDeserializer<AoSTS, CPU>::readN(std::string const& file_name, Fn fn)
{
    ParticlesDeserializer<AoS, CPU>{}.template readN<Src, N>(file_name, fn);
}


template<>
template<typename Src>
void ParticlesSerializer<AoSPCTS, CPU>::operator()(std::string const& file_name, Src const& src)
{
    TiledParticlesSerializing<AoSPCTS>::write(file_name, src, *this);
}

template<>
template<typename Dst, typename Src>
void ParticlesDeserializer<AoSPCTS, CPU>::operator()(std::string const& file_name, Dst& dst)
{
    TiledParticlesSerializing<AoSPCTS>::template read<Dst, Src>(file_name, dst, *this);
}

template<>
template<typename Src, std::uint16_t N, typename Fn>
void ParticlesDeserializer<AoSPCTS, CPU>::readN(std::string const& file_name, Fn fn)
{
    ParticlesDeserializer<AoS, CPU>{}.template readN<Src, N>(file_name, fn);
}


} // namespace plamn::core


#endif /* PLAMN_CORE_DATA_PARTICLES_SERIALIZING_DETAIL_TS_SERIALIZING */
