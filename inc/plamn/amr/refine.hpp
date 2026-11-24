#ifndef PLAMN_AMR_REFINE_HPP
#define PLAMN_AMR_REFINE_HPP

// refine operators for damn::amr::refine (and damn::mpi's), over plamn's resources
// (resources.hpp), refinement ratio 2:
//
//   FieldRefiner:       PHARE's composite linear refinement, any centering. Per direction:
//                       primal, a fine node on a coarse node copies it, one between two coarse
//                       nodes is their mean; dual, the two fine cells of coarse cell I are
//                       u_I -/+ (u_{I+1} - u_{I-1}) / 8 (they mean back to u_I). The value is
//                       the tensor product of the per direction stencils
//   divergence_free_touch_up(B, cells, region, dx):
//                       stage 2 of PHARE's divergence free magnetic refinement (Balsara et al.
//                       2023, see PHARE's magnetic_refine_patch_strategy.hpp): after B's
//                       components were refined with FieldRefiner over whole coarse cells,
//                       the faces inside each coarse cell are corrected so the divergences of
//                       its fine cells are all the coarse cell's: a div free coarse B refines
//                       div free
//   ParticleSplitter<Splitter>:
//                       PHARE's particle refinement: each coarse particle split in
//                       Splitter::nbRefinedPart fine ones (split/split.hpp), those in the
//                       target box kept

#include "damn/core/def.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "plamn/amr/resources.hpp"
#include "plamn/amr/split/splitter.hpp"
#include "plamn/core/data/particles/particle.hpp"
#include "plamn/core/models/quantities/quantity.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace damn::amr
{
template<typename T, std::size_t dim>
struct BoxValues;
} // namespace damn::amr

namespace plamn::amr
{
namespace detail
{
    // one direction's stencil: coarse offsets from the anchor floor(fine / 2), and weights
    struct Row
    {
        std::array<std::pair<int, double>, 3> points{};
        std::size_t size = 0;
    };

    MKN_KUL_FN_SIG_RET inline Row refine_row(core::QtyCentering const centering, int const fine)
    {
        bool const odd = (fine % 2) != 0;
        if (centering == core::QtyCentering::primal)
            return odd ? Row{{{{0, .5}, {1, .5}, {0, 0.}}}, 2}
                       : Row{{{{0, 1.}, {0, 0.}, {0, 0.}}}, 1};
        double const sign = odd ? 1 : -1; // right, left child
        return Row{{{{-1, -sign / 8}, {0, 1.}, {1, sign / 8}}}, 3};
    }
} // namespace detail


struct FieldRefiner
{
    int stencil = 1; // coarse points around the coarsened target

    template<typename Field, std::uint32_t gw, typename T>
    void operator()(FieldResource<Field, gw>& dst,
                    damn::core::Box<int, Field::dimension> const& cells,
                    damn::core::Box<int, Field::dimension> const& box,
                    damn::amr::BoxValues<T const, Field::dimension> const& coarse) const
    {
        auto constexpr dim   = Field::dimension;
        auto const centering = core::centering<dim>(*dst.field);
        damn::core::for_each_point(box, [&](auto const& p) {
            std::array<detail::Row, dim> rows;
            for (std::size_t d = 0; d < dim; ++d)
                rows[d] = detail::refine_row(centering[d], p[d]);
            auto const anchor = damn::core::floor_div(p, 2);

            T sum{0};
            std::array<std::size_t, dim> at{}; // a point of the tensor product
            while (true)
            {
                auto q      = anchor;
                double coef = 1;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    q[d] += rows[d].points[at[d]].first;
                    coef *= rows[d].points[at[d]].second;
                }
                sum += coef * coarse(q);

                std::size_t d = 0;
                for (; d < dim and ++at[d] == rows[d].size; ++d)
                    at[d] = 0;
                if (d == dim)
                    break;
            }
            value(dst, cells, p) = sum;
        });
    }
};


// a fine point from the coarse point under it, piecewise constant (cell centred values: the
//  coarse cell's, as the MHD primitives a hybrid level over an MHD one loads its particles from)
struct CellInjector
{
    int stencil = 0;

    template<typename Field, std::uint32_t gw, typename T>
    void operator()(FieldResource<Field, gw>& dst,
                    damn::core::Box<int, Field::dimension> const& cells,
                    damn::core::Box<int, Field::dimension> const& box,
                    damn::amr::BoxValues<T const, Field::dimension> const& coarse) const
    {
        damn::core::for_each_point(box, [&](auto const& p) {
            value(dst, cells, p) = coarse(damn::core::floor_div(p, 2));
        });
    }
};


// B: the components' resources (x, y, z), cells: the fine patch's, region: fine cells, a union
// of whole coarse cells whose faces FieldRefiner wrote, dx: the fine mesh size
template<typename Resource, std::size_t dim>
void divergence_free_touch_up(std::array<Resource*, 3> const& B,
                              damn::core::Box<int, dim> const& cells,
                              damn::core::Box<int, dim> const& region,
                              std::array<double, dim> const& dx)
{
    using Point = damn::core::Point<int, dim>;
    for (std::size_t d = 0; d < dim; ++d)
        if (region.lower[d] % 2 != 0 or region.upper[d] % 2 == 0)
            throw std::runtime_error("plamn::amr::divergence_free_touch_up: the region is not "
                                     "whole coarse cells");

    auto const unit = [](std::size_t const d) {
        Point e{};
        e[d] = 1;
        return e;
    };

    // the fine cells' divergences before any correction
    auto const shape = region.shape();
    auto const index = [&](Point const& c) {
        std::size_t i = 0, stride = 1;
        for (std::size_t d = 0; d < dim; ++d)
        {
            i += static_cast<std::size_t>(c[d] - region.lower[d]) * stride;
            stride *= static_cast<std::size_t>(shape[d]);
        }
        return i;
    };
    std::vector<double> div(region.size());
    damn::core::for_each_point(region, [&](auto const& c) {
        double v = 0;
        for (std::size_t d = 0; d < dim; ++d)
            v += (value(*B[d], cells, c + unit(d)) - value(*B[d], cells, c)) / dx[d];
        div[index(c)] = v;
    });

    for (std::size_t normal = 0; normal < dim; ++normal)
    {
        auto& Bn = *B[normal];
        // the faces inside the region's coarse cells: odd along the normal
        damn::core::for_each_point(domain_box(Bn, region), [&](auto const& face) {
            if (face[normal] % 2 == 0)
                return;
            // divergence jump across the face, between the cells shifted by shift
            auto const jump = [&](Point const& shift) {
                auto const high = face + shift;
                return div[index(high)] - div[index(high - unit(normal))];
            };
            // one fine cell toward the sibling inside the same coarse cell
            auto const toward = [&](std::size_t const d) {
                Point s{};
                s[d] = face[d] % 2 != 0 ? -1 : 1;
                return s;
            };
            auto& b = value(Bn, cells, face);
            if constexpr (dim == 1)
                b += dx[normal] * jump(Point{}) / 2.0;
            else if constexpr (dim == 2)
                b += dx[normal] * (3.0 * jump(Point{}) + jump(toward(1 - normal))) / 8.0;
            else
            {
                std::array<std::size_t, 2> t{};
                for (std::size_t d = 0, k = 0; d < dim; ++d)
                    if (d != normal)
                        t[k++] = d;
                auto const t1 = toward(t[0]), t2 = toward(t[1]);
                b += dx[normal]
                     * (7.0 * jump(Point{}) + 2.0 * jump(t1) + 2.0 * jump(t2) + jump(t1 + t2))
                     / 24.0;
            }
        });
    }
}


// a coarse particle's position on the fine grid
template<std::size_t dim>
MKN_KUL_FN_SIG_RET core::ParticlePosition<dim> to_fine_grid(core::Particle<dim> const& p)
{
    core::ParticlePosition<dim> fine{p.iCell(), p.delta()};
    for (std::size_t d = 0; d < dim; ++d)
    {
        auto const fine_delta     = p.delta()[d] * refinementRatio;
        auto const fine_delta_int = static_cast<int>(fine_delta);
        fine._iCell[d]            = p.iCell()[d] * refinementRatio + fine_delta_int;
        fine._delta[d]            = fine_delta - fine_delta_int;
    }
    return fine;
}

template<typename Splitter>
struct ParticleSplitter
{
    // coarse cells around the coarsened target holding every parent with a child in it
    int stencil = (splitBoxGrowth<Splitter>() + 1) / 2;
    Splitter split{};

    template<typename Particles, std::uint32_t gw, typename Particle>
    void operator()(ParticleResource<Particles, gw>& dst,
                    damn::core::Box<int, Particles::dimension> const&,
                    damn::core::Box<int, Particles::dimension> const& box,
                    std::span<Particle const> const coarse,
                    damn::core::Point<int, Particles::dimension> const& shift) const
    {
        auto constexpr dim = Particles::dimension;
        struct Buffer // AoS, for the split patterns
        {
            std::array<Particle, Splitter::nbRefinedPart> particles{};
            auto begin() { return particles.begin(); }
            auto size() const { return particles.size(); }
        } children;

        for (auto parent : coarse)
        {
            for (std::size_t d = 0; d < dim; ++d)
                parent.iCell()[d] += shift[d];
            split(to_fine_grid(parent), parent, children);
            for (auto const& child : children.particles)
                if (damn::core::isIn(damn::core::Point<int, dim>{child.iCell()}, box))
                    dst.particles->emplace_back(child);
        }
    }
};

} // namespace plamn::amr

#endif // PLAMN_AMR_REFINE_HPP
