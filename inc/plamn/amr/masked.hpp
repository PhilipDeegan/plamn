#ifndef PLAMN_AMR_MASKED_HPP
#define PLAMN_AMR_MASKED_HPP

// Masked<Op>: a refine operator (damn/amr/inter_level.hpp) applied only where none of the
// covered cells is, e.g. the new cells of a regridded level (the old ones kept their data)

#include "damn/core/geometry/box.hpp"
#include "damn/core/def.hpp"

#include <concepts>
#include <cstddef>
#include <utility>
#include <vector>
#include <format>

namespace plamn::amr
{
// op, only where none of the covered cells (an old level's, periodic images included) is:
//  new cells of a regridded level, the old ones kept their data
template<typename Op, std::size_t dim>
struct Masked
{
    using Box_t = damn::core::Box<int, dim>;
    Op op;
    std::vector<Box_t> const* covered;
    int stencil = Op{}.stencil;

    // box less every covered box, the covered points of a box given by points(box)
    NO_DISCARD std::vector<Box_t> uncovered(Box_t const& box, auto&& points) const
    {
        std::vector<Box_t> parts{box};
        for (auto const& c : *covered)
        {
            auto const p = points(c);
            std::vector<Box_t> rest;
            for (auto const& part : parts)
                for (auto const& r : part.remove(p))
                    rest.push_back(r);
            parts = std::move(rest);
        }
        return parts;
    }

    template<typename R, typename Coarse>
        requires std::invocable<Op const&, R&, Box_t const&, Box_t const&, Coarse const&>
    void operator()(R& dst, Box_t const& cells, Box_t const& box, Coarse const& coarse) const
    {
        for (auto const& part : uncovered(box, [&](auto const& b) { return domain_box(dst, b); }))
            op(dst, cells, part, coarse);
    }

    template<typename R, typename Items, typename Shift>
        requires std::invocable<Op const&, R&, Box_t const&, Box_t const&, Items, Shift const&>
    void operator()(R& dst, Box_t const& cells, Box_t const& box, Items const items,
                    Shift const& shift) const
    {
        for (auto const& part : uncovered(box, [](auto const& b) { return b; }))
            op(dst, cells, part, items, shift);
    }
};

} // namespace plamn::amr

#endif // PLAMN_AMR_MASKED_HPP
