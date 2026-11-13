#ifndef PLAMN_CORE_GRID_DETAIL_KOKKOS_HPP
#define PLAMN_CORE_GRID_DETAIL_KOKKOS_HPP

#include "plamn/core/def.hpp"
#include "damn/core/geometry/box_spans.hpp"

namespace plamn::core::gpu::impl
{

struct BoxKernel
{
    template<typename T, std::size_t S, typename... Args>
    static void evalOnBox(auto& fn, damn::core::Box<T, S> const& box, Args&&... args)
    {
        DAMN_LOG_LINE_SS(box.size());
        int size = box.size();
        Kokkos::parallel_for("evalOnBox", size, [&](int i) { // will fail with copy capture
            fn(damn::core::point_at(box, i), args...);
        });
    }
};


} // namespace plamn::core::gpu::impl


#endif /* PLAMN_CORE_GRID_DETAIL_KOKKOS_HPP */
