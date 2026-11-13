#ifndef PLAMN_CORE_GRID_DETAIL_RAJA_H
#define PLAMN_CORE_GRID_DETAIL_RAJA_H

#include "plamn/core/def.hpp"
#include "damn/core/geometry/box_spans.hpp"

namespace plamn::core::gpu::impl
{

struct BoxKernel
{
    template<typename Fn, typename T, std::size_t S, typename... Args>
    static void evalOnBox(Fn& fn, damn::core::Box<T, S> const& box, Args&&... args)
    {
#if !defined(RAJA_ENABLE_CUDA)
        RAJA::forall<RAJA::seq_exec>(
            RAJA::RangeSegment(0, box.size()),
            [=] RAJA_HOST_DEVICE(std::size_t const i) mutable { fn(damn::core::point_at(box, i), args...); });

#else
        RAJA::resources::Cuda res;
        exec([=] RAJA_DEVICE(int i) mutable { fn(damn::core::point_at(box, i), args...); }, res, box.size());

#endif // defined(RAJA_ENABLE_CUDA)
    }
};


} // namespace plamn::core::gpu::impl

#endif /* PLAMN_CORE_GRID_DETAIL_RAJA_H */
