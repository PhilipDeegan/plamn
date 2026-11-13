#ifndef PLAMN_CORE_DATA_PARTICLES_SERIALIZING_PARTICLES_SERIALIZING
#define PLAMN_CORE_DATA_PARTICLES_SERIALIZING_PARTICLES_SERIALIZING

#include "plamn/core/def.hpp"
#include "plamn/core/data/particles/serializing/detail/def_serializing.hpp"
#include "plamn/core/data/particles/serializing/detail/aos_serializing.hpp"
#include "plamn/core/data/particles/serializing/detail/soa_serializing.hpp"
#include "plamn/core/data/particles/serializing/detail/ts_serializing.hpp"

#include <string>

namespace plamn::core
{



// template<auto src_layout_mde, auto src_alloc_mde>
// template<typename Dst, typename Src>
// void ParticlesDeserializer<src_layout_mde, src_alloc_mde>::operator()(std::string const&
// file_name,
//                                                                       Dst& dst)
// {
//     ParticlesDeserializer<Src::layout_mode, CPU>{}.template readN<Src>(
//         file_name, [&](auto const& particles) {
//             for (auto const& p : particles)
//                 dst.push_back(p);
//         });
// }

} // namespace plamn::core

namespace plamn::core
{


template<typename Src>
void serialize_particles(std::string const& file_name, Src const& src)
{
    using Serializer = ParticlesSerializer<Src::layout_mode, Src::alloc_mode>;

    Serializer{}.operator()(file_name, src);
}

template<typename Dst, typename Src = Dst>
void deserialize_particles(std::string const& file_name, Dst& dst)
{
    using Deserializer = ParticlesDeserializer<Dst::layout_mode, Dst::alloc_mode>;

    Deserializer{}.template operator()<Dst, Src>(file_name, dst);
}

} // namespace plamn::core


#endif /* PLAMN_CORE_DATA_PARTICLES_SERIALIZING_PARTICLES_SERIALIZING */
