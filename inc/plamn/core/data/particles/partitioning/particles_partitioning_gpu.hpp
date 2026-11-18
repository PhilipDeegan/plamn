#ifndef PLAMN_CORE_DATA_PARTICLES_PARTITIONING_PARTICLES_PARTITIONING_GPU_HPP
#define PLAMN_CORE_DATA_PARTICLES_PARTITIONING_PARTICLES_PARTITIONING_GPU_HPP

#include "damn/core/def/damn_config.hpp"

// #include "damn/core/def/damn_config.hpp"

#if DAMN_HAVE_THRUST

#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/range.hpp"

#include <thrust/partition.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>


namespace plamn::core
{

template<typename Particles>
class ParticlesPartitioner<damn::AllocatorMode::GPU_UNIFIED, Particles, /*impl = */ 0>
{
public:
    using box_t    = damn::core::Box<int, Particles::dimension>;
    using iterator = decltype(std::declval<Particles>().begin());
    using range_t  = damn::core::BoxRange<box_t, iterator>;

    auto operator()(box_t const& box)
    {
        using enum LayoutMode;
        if constexpr (Particles::layout_mode == AoS)
        {
            // static_assert(Particles::storage_mode == StorageMode::SPAN);
            auto view  = array.view();
            auto pivot = thrust::partition(
                thrust::device, view.begin() + start, view.begin() + end,
                [=] DAMN_ALL(auto& part) { return isIn(cellAsPoint(part.iCell()), box); });
            return range_t{box, array.begin() + start,
                           array.begin() + std::distance(view.begin(), pivot)};
        }
        else if constexpr (mkn::kul::any_in(Particles::layout_mode, SoAVX, SoAVXChunked, SoAVXTS))
            throw std::runtime_error("no impl");
        else if constexpr (Particles::layout_mode == SoA)
        {
            auto it    = SoAIteratorAdaptor::make(array);
            auto pivot = thrust::partition(
                thrust::device, it + start, it + end,
                [=] DAMN_ALL(auto& z) { return isIn(SoAIteratorAdaptor::iCell(z), box); });
            return range_t{box, array.begin() + start, array.begin() + std::distance(it, pivot)};
        }
    }

    auto notIn(box_t const& box)
    {
        using enum LayoutMode;
        if constexpr (Particles::layout_mode == AoS)
        {
            // auto view  = array.view();
            auto pivot = thrust::partition(
                thrust::device, array.begin() + start, array.begin() + end,
                [=] DAMN_ALL(auto& part) { return !isIn(cellAsPoint(part.iCell()), box); });
            return range_t{box, array.begin() + start,
                           array.begin() + std::distance(array.begin(), pivot)};
        }
        else if constexpr (mkn::kul::any_in(Particles::layout_mode, SoAVX, SoAVXChunked, SoAVXTS))
            throw std::runtime_error("no impl");
        else if constexpr (Particles::layout_mode == SoA)
        {
            auto it    = SoAIteratorAdaptor::make(array);
            auto pivot = thrust::partition(
                thrust::device, it + start, it + end,
                [=] DAMN_ALL(auto& z) { return !isIn(SoAIteratorAdaptor::iCell(z), box); });
            return range_t{box, array.begin() + start, array.begin() + std::distance(it, pivot)};
        }
    }


    template<std::size_t S>
    auto operator()(std::array<box_t, S> const& boxes)
    {
        static_assert(S > 0);
        auto iterators = mkn::kul::generate_from(
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



} // namespace plamn::core


#endif // DAMN_HAVE_THRUST


#endif /*PLAMN_CORE_DATA_PARTICLES_PARTITIONING_PARTICLES_PARTITIONING_GPU_HPP*/
