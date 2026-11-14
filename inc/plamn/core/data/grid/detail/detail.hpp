#ifndef PLAMN_CORE_GRID_DETAIL_DETAIL_HPP
#define PLAMN_CORE_GRID_DETAIL_DETAIL_HPP

#include "damn/core/def/damn_config.hpp"
#include "damn/core/def/detail/mkn_gpu.hpp"
#include "damn/core/logger.hpp"

// the kernel backend: mkn::gpu launches
#if DAMN_HAVE_MKN_GPU
#include "plamn/core/data/grid/detail/mkn_gpu.hpp"
#endif

#include "damn/core/geometry/box_spans.hpp"

#include <cstddef>

namespace damn::core
{
template<typename Type, std::size_t dim>
struct Box;
} // namespace damn::core

namespace plamn::core::gpu
{
// fn(index, args...) on device, one launch over box
template<typename T, std::size_t S, typename... Args>
void eval_on_box(auto& fn, damn::core::Box<T, S> const& box, Args&... args)
{
    DAMN_LOG_LINE_SS(box.size());
#if DAMN_HAVE_MKN_GPU
    impl::BoxKernel::evalOnBox(fn, box, args...);
#endif
}
} // namespace plamn::core::gpu

namespace plamn::core
{
// fn(index, args...) for each point of box: a host loop for CPU fields, else one launch
// field only selects where to run: pass it among args to use it in fn
template<typename Field, typename Fn, typename T, std::size_t S, typename... Args>
void eval_on_box(Field const& /*field*/, damn::core::Box<T, S> const& box, Fn&& fn, Args&&... args)
{
    if constexpr (Field::alloc_mode == damn::AllocatorMode::CPU)
        damn::core::for_each_point(box, [&](auto const& bix) { fn(bix, args...); });
    else
        gpu::eval_on_box(fn, box, args...);
}
} // namespace plamn::core

#endif /* PLAMN_CORE_GRID_DETAIL_DETAIL_HPP */
