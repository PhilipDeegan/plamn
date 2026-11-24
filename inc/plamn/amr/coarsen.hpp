#ifndef PLAMN_AMR_COARSEN_HPP
#define PLAMN_AMR_COARSEN_HPP

// coarsen operators for damn::amr::coarsen (and damn::mpi's), over plamn's field resources
// (resources.hpp), refinement ratio 2, as PHARE's:
//
//   DefaultFieldCoarsener:  per direction, primal: 1/4, 1/2, 1/4 of the fine nodes 2q - 1,
//                           2q, 2q + 1; dual: the mean of fine cells 2q and 2q + 1 (B)
//   ElectricFieldCoarsener: per direction, primal: fine node 2q; dual: the mean of fine cells
//                           2q and 2q + 1 (E; moments, all primal, are injected: PHARE's
//                           MomentsCoarsener; MHD face fluxes: PHARE's MHDFluxCoarsener)

#include "damn/core/geometry/box_spans.hpp"

#include "plamn/amr/resources.hpp"
#include "plamn/core/models/quantities/quantity.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace damn::amr
{
template<typename T, std::size_t dim>
struct BoxValues;
} // namespace damn::amr
namespace damn::core
{
template<typename Type, std::size_t dim>
struct Box;
} // namespace damn::core

namespace plamn::amr
{
namespace detail
{
    struct CoarsenRow
    {
        std::array<std::pair<int, double>, 3> points{}; // fine offsets from 2q, weights
        std::size_t size = 0;
    };

    template<bool smooth_primal, typename Field, std::uint32_t gw, typename T>
    void coarsen(FieldResource<Field, gw> const& fine,
                 damn::core::Box<int, Field::dimension> const& cells,
                 damn::amr::BoxValues<T, Field::dimension> const& out)
    {
        auto constexpr dim   = Field::dimension;
        auto const centering = core::centering<dim>(*fine.field);
        std::array<CoarsenRow, dim> rows;
        for (std::size_t d = 0; d < dim; ++d)
            if (centering[d] == core::QtyCentering::dual)
                rows[d] = {{{{0, .5}, {1, .5}, {0, 0.}}}, 2};
            else if (smooth_primal)
                rows[d] = {{{{-1, .25}, {0, .5}, {1, .25}}}, 3};
            else
                rows[d] = {{{{0, 1.}, {0, 0.}, {0, 0.}}}, 1};

        damn::core::for_each_point(out.box, [&](auto const& q) {
            T sum{0};
            std::array<std::size_t, dim> at{};
            while (true)
            {
                auto p      = q * 2;
                double coef = 1;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    p[d] += rows[d].points[at[d]].first;
                    coef *= rows[d].points[at[d]].second;
                }
                sum += coef * value(fine, cells, p);

                std::size_t d = 0;
                for (; d < dim and ++at[d] == rows[d].size; ++d)
                    at[d] = 0;
                if (d == dim)
                    break;
            }
            out(q) = sum;
        });
    }
} // namespace detail

struct DefaultFieldCoarsener
{
    template<typename Field, std::uint32_t gw, typename T>
    void operator()(FieldResource<Field, gw> const& fine,
                    damn::core::Box<int, Field::dimension> const& cells,
                    damn::amr::BoxValues<T, Field::dimension> const& out) const
    {
        detail::coarsen<true>(fine, cells, out);
    }
};

struct ElectricFieldCoarsener
{
    template<typename Field, std::uint32_t gw, typename T>
    void operator()(FieldResource<Field, gw> const& fine,
                    damn::core::Box<int, Field::dimension> const& cells,
                    damn::amr::BoxValues<T, Field::dimension> const& out) const
    {
        detail::coarsen<false>(fine, cells, out);
    }
};

using MomentsCoarsener = ElectricFieldCoarsener;
// PHARE's MHDFluxCoarsener: a coarse face's flux, the mean of the fine faces on it (primal:
//  injected, dual: the mean), the same per direction rule
using MHDFluxCoarsener = ElectricFieldCoarsener;

} // namespace plamn::amr

#endif // PLAMN_AMR_COARSEN_HPP
