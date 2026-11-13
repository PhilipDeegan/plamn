#ifndef PLAMN_CORE_GRID_DETAIL_DETAIL_HPP
#define PLAMN_CORE_GRID_DETAIL_DETAIL_HPP

#include "plamn/core/def.hpp"

// one kernel backend: mkn::gpu launches when present (also on umpire memory), else raja, else kokkos
#if DAMN_HAVE_MKN_GPU
#include "plamn/core/data/grid/detail/mkn_gpu.hpp"
#elif DAMN_HAVE_RAJA
#include "plamn/core/data/grid/detail/raja.hpp"
#elif DAMN_HAVE_KOKKOS
#include "plamn/core/data/grid/detail/kokkos.hpp"
#endif

#include "damn/core/geometry/box.hpp"

#include <cstddef>

namespace plamn::core::gpu
{
struct GridLayout
{
    template<typename Fn, typename T, std::size_t S, typename... Args>
    static void evalOnBox(Fn& fn, damn::core::Box<T, S> const& box, Args&... args)
    {
        DAMN_LOG_LINE_SS(box.size());


#if DAMN_HAVE_MKN_GPU || DAMN_HAVE_RAJA || DAMN_HAVE_KOKKOS
        impl::GridLayout::evalOnBox(fn, box, args...);
#endif
    }
};


} // namespace plamn::core::gpu



#endif /* PLAMN_CORE_GRID_DETAIL_DETAIL_HPP */
