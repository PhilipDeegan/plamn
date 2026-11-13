#ifndef PLAMN_CORE_GRID_DETAIL_MKN_GPU_HPP
#define PLAMN_CORE_GRID_DETAIL_MKN_GPU_HPP

#include "plamn/core/def.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/exec/kernels.hpp"
#include "damn/core/geometry/box.hpp"

namespace plamn::core::gpu::impl
{

struct BoxKernel
{
    template<typename T, std::size_t S, typename... Args>
    static void evalOnBox(auto& fn, damn::core::Box<T, S> const& box, Args&... args)
    {
        damn::kernel::launch(box.size(),
                       [=] DAMN_ALL() mutable { fn(damn::core::point_at(box, damn::kernel::idx()), args...); });
    }
};


} // namespace plamn::core::gpu::impl


#endif /* PLAMN_CORE_GRID_DETAIL_MKN_GPU_HPP */
