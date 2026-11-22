#ifndef PLAMN_CORE_GRID_DETAIL_MKN_GPU_HPP
#define PLAMN_CORE_GRID_DETAIL_MKN_GPU_HPP

#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/exec/kernels.hpp"
#include "damn/core/def/damn_config.hpp"
#include "plamn/core/data/field/field_tiles.hpp" // untiled_view

#include <cstddef>

namespace damn::core
{
template<typename Type, std::size_t dim>
struct Box;
} // namespace damn::core

namespace plamn::core::gpu::impl
{

struct BoxKernel
{
    // args as plain views (untiled_view: basic::, device copyable); fn must capture plain data
    //  too (damn's launchers check). Not mutable: fn and the views read in place, fn gets per
    //  thread copies of the views (written through)
    template<typename T, std::size_t S, typename... Args>
    static void evalOnBox(auto& fn, damn::core::Box<T, S> const& box, Args&... args)
    {
        launch_(fn, box, untiled_view(args)...);
    }

private:
    static void launch_(auto const& fn, auto const& box, auto const... views)
    {
        damn::kernel::launch(box.size(), [=] DAMN_ALL() {
            [](auto const& f, auto const& p, auto... vs) { f(p, vs...); }(
                fn, damn::core::point_at(box, damn::kernel::idx()), views...);
        });
    }
};


} // namespace plamn::core::gpu::impl


#endif /* PLAMN_CORE_GRID_DETAIL_MKN_GPU_HPP */
