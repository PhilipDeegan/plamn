#ifndef PLAMN_CORE_DATA_PARTICLES_PARTITIONING_PARTICLES_PARTITIONING_CPU_HPP
#define PLAMN_CORE_DATA_PARTICLES_PARTITIONING_PARTICLES_PARTITIONING_CPU_HPP

#include "damn/core/vector.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/range.hpp"

#include "plamn/core/def.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/data/particles/partitioning/partitionner.hpp"

#include <stdexcept>

namespace plamn::core
{
template<typename Particles>
class ParticlesPartitioner<damn::AllocatorMode::CPU, Particles, /*impl = */ 0>
{
public:
    using box_t    = damn::core::Box<int, Particles::dimension>;
    using iterator = decltype(std::declval<Particles>().begin());
    using range_t  = damn::core::BoxRange<box_t, iterator>;

    auto operator()(box_t const& box)
    {
        auto constexpr static layout_mode = Particles::layout_mode;
        using enum LayoutMode;
        if constexpr (damn::core::any_in(layout_mode, SoA, SoAVX, SoAVXChunked))
        {
            DAMN_WITH_THRUST_ELSE_THROW(
                using Iterator_t = std::conditional_t < layout_mode == SoA, SoAIteratorAdaptor,
                SoAVXIteratorAdaptor > ;
                auto it = Iterator_t::make(array);
                auto pivot
                = thrust::partition(thrust::host, it + start, it + end,
                                    [=](auto& z) { return isIn(Iterator_t::iCell(z), box); });
                return range_t{box, array.begin() + start,
                               array.begin() + std::distance(it, pivot)}; //
            )
        }
        else if constexpr (damn::core::any_in(layout_mode, SoAVXTS))
            throw std::runtime_error("no impl");
        else if constexpr (damn::core::any_in(layout_mode, AoSMapped))
        {
            return array.partition(
                [&](auto& part) { return isIn(cellAsPoint(part.iCell()), box); });
        }
        else
        {
            static_assert(!damn::core::any_in(layout_mode, AoSPC, SoAPC, AoSPCTS),
                          "partitioning not implemented for this layout (no flat iteration)");
            return partitionner(array.begin() + start, array.begin() + end, box);
        }
    }

    // try dedupe
    auto notIn(box_t const& box)
    {
        auto constexpr static layout_mode = Particles::layout_mode;
        using enum LayoutMode;
        if constexpr (damn::core::any_in(layout_mode, SoA, SoAVX, SoAVXChunked))
        {
            DAMN_WITH_THRUST_ELSE_THROW( //
                using Iterator_t = std::conditional_t < layout_mode == SoA, SoAIteratorAdaptor,
                SoAVXIteratorAdaptor > ;
                auto it = Iterator_t::make(array);
                auto pivot
                = thrust::partition(thrust::host, it + start, it + end,
                                    [=](auto& z) { return !isIn(Iterator_t::iCell(z), box); });
                return range_t{box, array.begin() + start,
                               array.begin() + std::distance(it, pivot)}; //
            )
        }
        else if constexpr (damn::core::any_in(layout_mode, SoAVXTS))
            throw std::runtime_error("no impl");
        else if constexpr (damn::core::any_in(layout_mode, AoSMapped))
        {
            return array.partition(
                [&](auto& part) { return !isIn(cellAsPoint(part.iCell()), box); });
        }
        else
        {
            static_assert(!damn::core::any_in(layout_mode, AoSPC, SoAPC, AoSPCTS),
                          "partitioning not implemented for this layout (no flat iteration)");
            return partitionner(array.begin() + start, array.begin() + end, box,
                                [=](auto& part) { return !isIn(cellAsPoint(part.iCell()), box); });
        }
    }

    template<std::size_t S>
    auto operator()(std::array<box_t, S> const& boxes)
    {
        using enum LayoutMode;
        static_assert(S > 0);
        auto iterators = damn::core::generate_from(
            [&](auto const& box) { return range_t{box, array.begin(), array.begin()}; }, boxes);
        for (std::size_t i = 0; i < boxes.size(); ++i)
            start += (iterators[i] = (*this)(boxes[i])).size();
        return iterators;
    }

    auto operator()(std::vector<box_t> const& boxes)
    {
        std::vector<range_t> iterators;
        for (auto const& box : boxes)
            start += iterators.emplace_back((*this)(box)).size();
        return iterators;
    }

    Particles& array;
    std::size_t start = 0, end = array.size();
};


// template<typename Particles>
// class ParticlesPartitioner<AllocatorMode::CPU, Particles, /*impl = */ 1>
// {
// public:
//     using box_t = Box<int, Particles::dimension>;

//     auto operator()(box_t const& box)
//     {
//         static_assert(!Particles::is_mapped);

//         return partitionner(array.begin() + start, array.begin() + end, box,
//                             [&](auto const& part) { return part.norcell >= 0; });
//     }

//     Particles& array;
//     std::size_t start = 0, end = array.size();
// };



} // namespace plamn::core

#endif /*PLAMN_CORE_DATA_PARTICLES_PARTITIONING_PARTICLES_PARTITIONING_CPU_HPP*/
