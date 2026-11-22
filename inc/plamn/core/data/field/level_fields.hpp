#ifndef PLAMN_CORE_DATA_FIELD_LEVEL_FIELDS_HPP
#define PLAMN_CORE_DATA_FIELD_LEVEL_FIELDS_HPP

// a level's fields as one: the contiguous buffers (for_each_buffer: a tiled field's tiles, an
//  untiled one's data) of the quantities chosen on every patch, an operation on all of them at
//  once. Unified memory: one kernel over every buffer (up to 128 a launch, passed by value: no
//  host access to the values, no allocation); host memory: per buffer.
//
//   level_of<alloc_mode>(level, quantities).zero()
//     quantities(patch, fields): fields.add(f) for each field f (or tensor field) of the patch,
//     e.g. domain_moments: every population's density and flux

#include "damn/core/def/damn_config.hpp"
#include "damn/core/exec/kernels.hpp"
#include "plamn/core/data/field/field_tiles.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

namespace plamn::core
{

template<auto alloc_mode, typename T = double>
class LevelFields
{
public:
    struct Buffer
    {
        T* data;
        std::size_t size;
    };

    // a field's buffers
    void add(auto& field)
        requires requires { field.data(); }
    {
        for_each_buffer(field, [&](T* const data, std::size_t const size) {
            if (size == 0)
                return;
            buffers_.push_back({data, size});
            largest_ = std::max(largest_, size);
        });
    }
    // a tensor field's, component by component
    void add(auto& tensor_field)
        requires(!requires { tensor_field.data(); })
    {
        for (auto& component : tensor_field)
            add(component);
    }

    LevelFields& zero()
        requires(alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
    {
        // the buffers passed by value, batch_size at a time (a kernel's parameters: no
        //  allocation, nothing written to unified memory by the host)
        for (std::size_t first = 0; first < buffers_.size(); first += batch_size)
        {
            Batch batch;
            batch.n = std::min(batch_size, buffers_.size() - first);
            for (std::size_t k = 0; k < batch.n; ++k)
            {
                batch.buffers[k] = buffers_[first + k];
                batch.largest    = std::max(batch.largest, batch.buffers[k].size);
            }
            damn::kernel::launch(batch.n * batch.largest, [=] DAMN_ALL() {
                auto const i = damn::kernel::idx();
                auto const b = batch.buffers[i / batch.largest];
                if (auto const j = i % batch.largest; j < b.size)
                    b.data[j] = T{0};
            });
        }
        return *this;
    }
    LevelFields& zero()
        requires(alloc_mode == damn::AllocatorMode::CPU)
    {
        for (auto const& b : buffers_)
            std::fill(b.data, b.data + b.size, T{0});
        return *this;
    }

private:
    static constexpr std::size_t batch_size = 128; // 2 KB of kernel parameters
    struct Batch
    {
        std::array<Buffer, batch_size> buffers{};
        std::size_t n = 0, largest = 0;
    };

    std::vector<Buffer> buffers_;
    std::size_t largest_ = 0;
};

template<auto alloc_mode>
NO_DISCARD auto level_of(auto&& level, auto&& quantities)
{
    LevelFields<alloc_mode> fields;
    level.per_patch([&](auto& patch) { quantities(patch, fields); });
    return fields;
}

// what the domain deposit sums into: every population's density and flux
inline constexpr auto domain_moments = [](auto& patch, auto& fields) {
    for (auto& pop : patch.ions)
    {
        fields.add(pop.particleDensity());
        fields.add(pop.flux());
    }
};

} // namespace plamn::core

#endif /* PLAMN_CORE_DATA_FIELD_LEVEL_FIELDS_HPP */
