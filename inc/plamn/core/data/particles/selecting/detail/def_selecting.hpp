// IWYU pragma: private, include "plamn/core/data/particles/selecting/particles_selecting.hpp"
#ifndef PLAMN_CORE_DATA_PARTICLES_SELECTING_DETAIL_DEF_SELECTING
#define PLAMN_CORE_DATA_PARTICLES_SELECTING_DETAIL_DEF_SELECTING

#include "mkn/kul/meta.hpp"

#include <cstdint>
#include <cstddef>

namespace damn
{
enum class AllocatorMode : std::uint16_t;
} // namespace damn
namespace damn::core
{
template<typename Type, std::size_t dim>
struct Box;
} // namespace damn::core

namespace plamn::core
{
enum class LayoutMode : std::uint16_t;

namespace detail
{
    // indexes given one at a time, merged into ranges of consecutive ones: fn(start, size) per
    //  range, the last on flush()
    template<typename Fn>
    struct IndexRuns
    {
        void operator()(std::size_t const i)
        {
            if (size and i == start + size)
            {
                ++size;
                return;
            }
            flush();
            start = i;
            size  = 1;
        }
        void flush()
        {
            if (size)
                fn(start, size);
            size = 0;
        }

        Fn& fn;
        std::size_t start = 0, size = 0;
    };
} // namespace detail

// the particles whose cell is in a box, visited as contiguous index ranges, each layout by what
//  it knows of its cells
template<auto layout_mde, auto alloc_mde>
struct ParticlesSelector
{
    static_assert(mkn::kul::all_are<LayoutMode>(layout_mde));
    static_assert(mkn::kul::all_are<damn::AllocatorMode>(alloc_mde));

    auto constexpr static layout_mode = layout_mde;
    auto constexpr static alloc_mode  = alloc_mde;

    // fn(start, size) for each range of src's particles whose cells are in box
    template<typename Src, std::size_t dim, typename Fn>
    void operator()(Src const& src, damn::core::Box<int, dim> const& box, Fn&& fn) const;
};

} // namespace plamn::core


#endif /* PLAMN_CORE_DATA_PARTICLES_SELECTING_DETAIL_DEF_SELECTING */
