// IWYU pragma: private, include "plamn/core/data/particles/converting/particles_converting.hpp"
#ifndef PLAMN_CORE_DATA_PARTICLES_CONVERTING_DETAIL_SOAVX_CHUNKED_CONVERTING
#define PLAMN_CORE_DATA_PARTICLES_CONVERTING_DETAIL_SOAVX_CHUNKED_CONVERTING

// SoAVXChunked (SoAVX sorted by cell, per cell offsets, see particles_soavx_chunked.hpp), CPU:
//
//   from it   cell by cell through its offsets (the layout's order), a range at a time; while
//             its offsets are stale (a change since the last sort()) in memory order instead
//   into it   appended, then sort(): the result's offsets are valid
//
// it has no element iteration (Particles::begin), so the generic converter cannot read it, and
// would leave it unsorted when writing it

#include "damn/core/def/damn_config.hpp"

#include "plamn/core/data/particles/converting/particles_converting.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/data/particles/arrays/particles_soavx_chunked.hpp"
#include "plamn/core/data/particles/converting/detail/def_converting.hpp" // IWYU pragma: keep

#include <cassert>
#include <cstddef>

namespace plamn::core
{
using enum LayoutMode;
using enum damn::AllocatorMode;

namespace detail
{
    template<typename Dst, typename Src>
    void soavx_chunked_into(Src const& src, Dst& out)
    {
        out.reserve(out.size() + src.size());
        if (!src.sorted())
        {
            for (std::size_t i = 0; i < src.size(); ++i)
                out.emplace_back(src.copy(i));
            return;
        }
        src.for_each_range(src.shell(), [&](auto const start, auto const size) {
            for (std::size_t i = start; i < start + size; ++i)
                out.emplace_back(src.copy(i));
        });
    }

    template<typename Dst, typename Src, typename Layout>
    Dst into_soavx_chunked(Src const& src, Layout const& layout)
    {
        auto out = make_particles<Dst>(layout);
        out.reserve(src.size());
        for (auto const& p : src)
            out.emplace_back(p);
        out.sort();
        return out;
    }
} // namespace detail


template<>
template<typename Dst, typename Src, typename Layout>
Dst ParticlesConverter<SoAVXChunked, CPU, AoS, CPU>::operator()( //
    Src const& src, Layout const& layout)
{
    auto out = make_particles<Dst>(layout);
    detail::soavx_chunked_into(src, out);
    assert(src.size() == out.size()); // every particle is in the shell the offsets cover
    return out;
}

template<>
template<typename Dst, typename Src, typename Layout>
Dst ParticlesConverter<AoS, CPU, SoAVXChunked, CPU>::operator()( //
    Src const& src, Layout const& layout)
{
    return detail::into_soavx_chunked<Dst>(src, layout);
}

template<>
template<typename Dst, typename Src, typename Layout>
Dst ParticlesConverter<AoSMapped, CPU, SoAVXChunked, CPU>::operator()( //
    Src const& src, Layout const& layout)
{
    return detail::into_soavx_chunked<Dst>(src, layout);
}

} // namespace plamn::core

#endif /* PLAMN_CORE_DATA_PARTICLES_CONVERTING_DETAIL_SOAVX_CHUNKED_CONVERTING */
