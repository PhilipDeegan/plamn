#include "plamn/core/def.hpp"

#ifndef PLAMN_CORE_DATA_PARTICLES_COMPARING_DETAIL_AOS_COMPARING
#define PLAMN_CORE_DATA_PARTICLES_COMPARING_DETAIL_AOS_COMPARING

#include "damn/core/def/damn_config.hpp"
#include "damn/core/memory/memory.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/math/equality.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/data/particles/comparing/detail/def_comparing.hpp"
#include "plamn/core/data/particles/arrays/particles_pc_ts.hpp"

// #include <format>
#include <optional>

namespace plamn::core
{

using enum LayoutMode;
using enum damn::AllocatorMode;

template<>
template<typename PS0, typename PS1>
damn::core::EqualityReport ParticlesComparator<AoS, CPU, AoS, CPU>::operator()(PS0 const& ps0, PS1 const& ps1,
                                                                   double const atol)
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
damn::core::EqualityReport ParticlesComparator<AoSMapped, CPU, AoS, CPU>::operator()(PS0 const& ps0,
                                                                         PS1 const& ps1,
                                                                         double const atol)
{
    return ParticlesComparator<AoS, CPU, AoS, CPU>{}(ps0, ps1, atol);
}


template<>
template<typename PS0, typename PS1>
damn::core::EqualityReport ParticlesComparator<AoSMapped, CPU, SoA, CPU>::operator()(PS0 const& ps0,
                                                                         PS1 const& ps1,
                                                                         double const atol)
{
    throw std::runtime_error("finish");
}


template<>
template<typename PS0, typename PS1>
damn::core::EqualityReport ParticlesComparator<AoSMapped, CPU, SoAVX, CPU>::operator()(PS0 const& ps0,
                                                                           PS1 const& ps1,
                                                                           double const atol)
{
    static_assert(PS0::dimension == PS1::dimension);

    if (ps0.size() != ps1.size())
        return damn::core::EqualityReport{false, "different sizes: " + std::to_string(ps0.size()) + " vs "
                                         + std::to_string(ps1.size())};

    for (std::size_t i = 0; i < ps0.size(); ++i)
    {
        std::string idx = std::to_string(i);
        if (ps0.iCell(i) != ps1.iCell(i))
            return damn::core::EqualityReport{false, "icell mismatch at index: " + idx, i};

        for (std::size_t d = 0; d < 3; ++d)
            if (!damn::core::float_equals(ps0.v(i)[d], ps1.v()[d][i], atol))
                return damn::core::EqualityReport{false, "v mismatch at index: " + idx, i};

        for (std::size_t d = 0; d < PS0::dimension; ++d)
            if (!damn::core::float_equals(ps0.delta(i)[d], ps1.delta()[d][i], atol))
                return damn::core::EqualityReport{false, "delta mismatch at index: " + idx + ", dim "
                                                 + std::to_string(d)};
    }

    return damn::core::EqualityReport{true};
}


template<>
template<typename PS0, typename PS1>
damn::core::EqualityReport ParticlesComparator<AoSTS, CPU, AoSTS, CPU>::operator()(PS0 const& ps0,
                                                                       PS1 const& ps1,
                                                                       double const atol)
{
    if (ps0.size() != ps1.size())
        return damn::core::EqualityReport{false, "different sizes: " + std::to_string(ps0.size()) + " vs "
                                         + std::to_string(ps1.size())};

    if (ps0().size() != ps1().size())
        return damn::core::EqualityReport{false, "different tile sizes: " + std::to_string(ps0().size())
                                         + " vs " + std::to_string(ps1().size())};

    ParticlesComparator<AoS, CPU, AoS, CPU> comparator;
    for (std::size_t i = 0; i < ps0().size(); ++i)
        if (auto eq = comparator(ps0()[i](), ps1()[i](), atol); !eq)
            return eq;

    return damn::core::EqualityReport{true};
}


template<>
template<typename PS0, typename PS1>
damn::core::EqualityReport ParticlesComparator<AoSTS, GPU_UNIFIED, AoSTS, GPU_UNIFIED>::operator()(
    PS0 const& ps0, PS1 const& ps1, double const atol)
{
    return ParticlesComparator<AoSTS, CPU, AoSTS, CPU>{}(ps0, ps1, atol); // do better
}

template<>
template<typename PS0, typename PS1>
damn::core::EqualityReport ParticlesComparator<AoSCMTS, CPU, AoSCMTS, CPU>::operator()(PS0 const& ps0,
                                                                           PS1 const& ps1,
                                                                           double const atol)
{
    if (ps0.size() != ps1.size())
        return damn::core::EqualityReport{false, "different sizes: " + std::to_string(ps0.size()) + " vs "
                                         + std::to_string(ps1.size())};

    if (ps0().size() != ps1().size())
        return damn::core::EqualityReport{false, "different tile counts: " + std::to_string(ps0().size())
                                         + " vs " + std::to_string(ps1().size())};

    ParticlesComparator<AoS, CPU, AoS, CPU> comparator;
    for (std::size_t ti = 0; ti < ps0().size(); ++ti)
        if (auto eq = comparator(ps0()[ti](), ps1()[ti](), atol); !eq)
            return eq;

    return damn::core::EqualityReport{true};
}


template<>
template<typename PS0, typename PS1>
damn::core::EqualityReport ParticlesComparator<AoSPC, CPU, AoSPC, CPU>::operator()(PS0 const& ps0,
                                                                       PS1 const& ps1,
                                                                       double const atol)
{
    if (ps0.size() != ps1.size())
        return damn::core::EqualityReport{false, "different sizes: " + std::to_string(ps0.size()) + " vs "
                                         + std::to_string(ps1.size())};

    ParticlesComparator<AoS, CPU, AoS, CPU> comparator;
    std::optional<damn::core::EqualityReport> first_diff; // EqualityReport is not assignable
    damn::core::for_each_point(ps0.local_box(), [&](auto const& cell) {
        if (first_diff)
            return;
        if (auto eq = comparator(ps0(cell), ps1(cell), atol); !eq)
            first_diff.emplace(eq);
    });
    if (first_diff)
        return *first_diff;

    return damn::core::EqualityReport{true};
}


template<>
template<typename PS0, typename PS1>
damn::core::EqualityReport ParticlesComparator<AoSPCTS, CPU, AoSPCTS, CPU>::operator()(PS0 const& ps0,
                                                                           PS1 const& ps1,
                                                                           double const atol)
{
    if (ps0.size() != ps1.size())
        return damn::core::EqualityReport{false, "different sizes: " + std::to_string(ps0.size()) + " vs "
                                         + std::to_string(ps1.size())};

    if (ps0().size() != ps1().size())
        return damn::core::EqualityReport{false, "different tile counts: " + std::to_string(ps0().size())
                                         + " vs " + std::to_string(ps1().size())};

    ParticlesComparator<AoSPC, CPU, AoSPC, CPU> comparator;
    for (std::size_t ti = 0; ti < ps0().size(); ++ti)
        if (auto eq = comparator(ps0()[ti](), ps1()[ti](), atol); !eq)
            return eq;

    return damn::core::EqualityReport{true};
}


template<>
template<typename PS0, typename PS1>
damn::core::EqualityReport ParticlesComparator<AoSPCTS, GPU_UNIFIED, AoSPCTS, GPU_UNIFIED>::operator()(
    PS0 const& ps0, PS1 const& ps1, double const atol)
{
    return ParticlesComparator<AoSPCTS, CPU, AoSPCTS, CPU>{}(ps0, ps1, atol);
}


} // namespace plamn::core


#endif /* PLAMN_CORE_DATA_PARTICLES_COMPARING_DETAIL_AOS_COMPARING */
