#ifndef PLAMN_CORE_DATA_FIELD_LEVEL_FIELDS_HPP
#define PLAMN_CORE_DATA_FIELD_LEVEL_FIELDS_HPP

// a level's fields as one: the contiguous buffers (for_each_buffer: a tiled field's tiles, an
//  untiled one's data) of the quantities chosen on every patch, an operation on all of them at
//  once. Unified memory: one kernel over the concatenated buffers (up to 128 a launch, their
//  prefix offsets passed by value: no padding, no host access to the values, no allocation);
//  host memory: per buffer.
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
        for_each_batch([](T& value) DAMN_ALL { value = T{0}; });
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
    // a batch of buffers as one range of values: their prefix offsets (offsets[k]: buffer k's
    //  first value in the range, offsets[n]: the total), passed by value to the kernel
    static constexpr std::size_t batch_size = 128; // ~3 KB of kernel parameters
    struct Batch
    {
        std::array<T*, batch_size> data{};
        std::array<std::size_t, batch_size + 1> offsets{};
        std::size_t n = 0;

        // the buffer holding value i of the range: the last offset <= i
        std::size_t buffer_of(std::size_t const i) const DAMN_ALL
        {
            std::size_t lo = 0, hi = n; // offsets[lo] <= i < offsets[hi]
            while (hi - lo > 1)
                if (auto const mid = (lo + hi) / 2; offsets[mid] <= i)
                    lo = mid;
                else
                    hi = mid;
            return lo;
        }
    };

    // op(value) on every value of every buffer: a launch per batch_size buffers, one thread per
    //  value (no padding: a thread finds its buffer in the prefix offsets)
    void for_each_batch(auto const op)
    {
        for (std::size_t first = 0; first < buffers_.size(); first += batch_size)
        {
            Batch batch;
            batch.n = std::min(batch_size, buffers_.size() - first);
            for (std::size_t k = 0; k < batch.n; ++k)
            {
                batch.data[k]        = buffers_[first + k].data;
                batch.offsets[k + 1] = batch.offsets[k] + buffers_[first + k].size;
            }
            damn::kernel::launch(batch.offsets[batch.n], [=] DAMN_ALL() {
                auto const i = damn::kernel::idx();
                auto const k = batch.buffer_of(i);
                op(batch.data[k][i - batch.offsets[k]]);
            });
        }
    }

    std::vector<Buffer> buffers_;
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
