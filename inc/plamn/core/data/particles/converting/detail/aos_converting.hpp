#include "plamn/core/def.hpp"
// IWYU pragma: private, include "core/data/particles/converting/particles_converting.hpp"

#ifndef PLAMN_CORE_DATA_PARTICLES_CONVERTING_DETAIL_AOS_CONVERTING
#define PLAMN_CORE_DATA_PARTICLES_CONVERTING_DETAIL_AOS_CONVERTING

#include "damn/core/memory/memory.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "plamn/core/data/particles/converting/detail/def_converting.hpp"

namespace plamn::core
{

using enum LayoutMode;
using enum damn::AllocatorMode;

template<>
template<typename Dst, typename Src, typename Layout>
Dst ParticlesConverter<AoS, CPU, AoS, CPU>::operator()(Src const& src, Layout const& layout)
{
    return src;
}

template<>
template<typename Dst, typename Src, typename Layout>
Dst ParticlesConverter<AoSTS, CPU, AoS, CPU>::operator()(Src const& src, Layout const& layout)
{
    auto out = make_particles<Dst>(layout);
    out.reserve(src.size());

    for (auto const& tile : src())
        std::copy(tile().begin(), tile().end(), std::back_inserter(out));

    return out;
}
template<>
template<typename Dst, typename Src, typename Layout>
Dst ParticlesConverter<AoSCMTS, CPU, AoS, CPU>::operator()(Src const& src, Layout const& layout)
{
    auto out = make_particles<Dst>(layout);
    out.reserve(src.size());
    for (auto const& tile : src())
        std::copy(tile().begin(), tile().end(), std::back_inserter(out));
    return out;
}

template<>
template<typename Dst, typename Src, typename Layout>
Dst ParticlesConverter<AoSPCTS, CPU, AoS, CPU>::operator()(Src const& src, Layout const& layout)
{
    auto out = make_particles<Dst>(layout);
    out.reserve(src.size());
    for (auto const& tile : src())
    {
        // full per-tile box: particles that left the patch domain live in tile ghost
        // cells; tile ghost cells inside the patch stay empty (owner tiles hold those)
        auto const& cps = tile();
        damn::core::for_each_point(cps.local_box(), [&](auto const& bix) {
            std::copy(cps(bix).begin(), cps(bix).end(), std::back_inserter(out));
        });
    }
    return out;
}

template<> // unified memory is host readable, so the CPU path works as is
template<typename Dst, typename Src, typename Layout>
Dst ParticlesConverter<AoSPCTS, GPU_UNIFIED, AoS, CPU>::operator()(Src const& src,
                                                                   Layout const& layout)
{
    return ParticlesConverter<AoSPCTS, CPU, AoS, CPU>{}.template operator()<Dst>(src, layout);
}

template<>
template<typename Dst, typename Src, typename Layout>
Dst ParticlesConverter<AoSTS, GPU_UNIFIED, AoS, CPU>::operator()( //
    Src const& src, Layout const& layout)
{
    auto out = make_particles<Dst>(layout);
    out.resize(src.size());
    std::size_t offset = 0;

    for (auto const& tile : src())
    {
        auto& particles = tile();
        if (particles.size() == 0)
            continue;
        damn::core::gpu::copy(out.data() + offset, particles.data(), particles.size());
        offset += particles.size();
    }
    return out;
}




} // namespace plamn::core


#endif /* PLAMN_CORE_DATA_PARTICLES_CONVERTING_DETAIL_AOS_CONVERTING */
