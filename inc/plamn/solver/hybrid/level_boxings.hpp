#ifndef PLAMN_SOLVER_HYBRID_LEVEL_BOXINGS_HPP
#define PLAMN_SOLVER_HYBRID_LEVEL_BOXINGS_HPP

// what a level's push needs of its decomposition, kept from one step to the next: every patch's
//  boxing by patch id (a PatchBoxings: .at(id)), and on unified memory the device lists of their
//  non level ghost and non owned boxes (in the level accessor's patch order, read by the device
//  copy push: mkn_xyz::MultiBoris borrows them instead of building its own each push). Rebuilt
//  when the level's generation (a new decomposition) differs from the one they were made for

#include "damn/core/def/damn_config.hpp"
#include "damn/core/geometry/box.hpp"

#if DAMN_HAVE_MKN_GPU
#include "damn/core/def/detail/mkn_gpu.hpp"
#include "damn/core/memory/span.hpp"
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace plamn::solver
{

template<typename Boxing_t>
struct LevelBoxingMap
{
    std::uint64_t generation = std::numeric_limits<std::uint64_t>::max(); // none yet
    std::unordered_map<std::string, Boxing_t> map;

    auto const& at(std::string const& id) const { return map.at(id); }

    // the accessor's patches' boxings (data: its level data, a patch's boxing at ->boxing)
    void rebuild_map(auto const& accessor, auto const& data)
    {
        map.clear();
        for (std::size_t i = 0; i < accessor.size(); ++i)
            map.try_emplace(accessor.id(i), *data.data[i]->boxing);
    }
};

template<typename Boxing_t, auto alloc_mode>
struct LevelBoxings : LevelBoxingMap<Boxing_t>
{
    void rebuild(std::uint64_t const generation, auto const& accessor, auto const& data)
    {
        this->rebuild_map(accessor, data);
        this->generation = generation;
    }
};

#if DAMN_HAVE_MKN_GPU
template<typename Boxing_t>
struct LevelBoxings<Boxing_t, damn::AllocatorMode::GPU_UNIFIED> : LevelBoxingMap<Boxing_t>
{
    using Box_t           = damn::core::Box<int, Boxing_t::Layout_t::dimension>;
    using GpuBoxSpanSet_t = damn::core::SpanSet<Box_t, damn::core::default_span_size_t,
                                                mkn::gpu::ManagedAllocator<Box_t>>;

    GpuBoxSpanSet_t gpu_nlgb;      // per patch, the accessor's order
    GpuBoxSpanSet_t gpu_non_owned; // per patch

    void rebuild(std::uint64_t const generation, auto const& accessor, auto const& data)
    {
        this->rebuild_map(accessor, data);
        auto const to_gpu = [&](auto const& boxes_of) {
            std::vector<damn::core::default_span_size_t> sizes;
            sizes.reserve(accessor.size());
            for (std::size_t i = 0; i < accessor.size(); ++i)
                sizes.push_back(boxes_of(this->at(accessor.id(i))).size());
            GpuBoxSpanSet_t set{std::move(sizes)};
            std::size_t offset = 0;
            for (std::size_t i = 0; i < accessor.size(); ++i)
            {
                auto const& boxes = boxes_of(this->at(accessor.id(i)));
                std::copy(boxes.begin(), boxes.end(), set.vec.begin() + offset);
                offset += boxes.size();
            }
            return set;
        };
        gpu_nlgb         = to_gpu([](auto const& b) -> auto& { return b.nonLevelGhostBox; });
        gpu_non_owned    = to_gpu([](auto const& b) -> auto& { return b.nonOwnedBoxes; });
        this->generation = generation;
    }
};
#endif

} // namespace plamn::solver

#endif /* PLAMN_SOLVER_HYBRID_LEVEL_BOXINGS_HPP */
