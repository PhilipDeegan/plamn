#ifndef PLAMN_AMR_TAGGING_HPP
#define PLAMN_AMR_TAGGING_HPP

// the cells of a level to refine, from its fields, after PHARE's tagging:
//
//   a quantity (B, or one component of it) is read at cell centres (cell_centred: each primal
//   direction averaged over the cell's two nodes), a criterion gives one indicator per cell
//   from the quantity's components around it, and a cell is tagged if any quantity's indicator
//   is above that quantity's threshold
//
//   criteria (indicator(components, cells, cell, dx), reach: cells read either side):
//     DefaultCriterion:  PHARE's default.
//                        1D: per component, |avg(c-1..c+3) - avg(c-2..c+2)| / (1 +
//                        |avg(c-2..c+2)|), the L2 norm over the components (reach 3) 2D/3D: the max
//                        over components and directions of |F(c+2e) - F(c)| / (1 + |F(c+e) - F(c)|)
//                        (reach 2)
//     LohnerCriterion:   Lohner's estimator as PLUTO's (Mignone et al. 2012, eq. 50-51), a
//                        vector as a whole (norms of the components): along each direction,
//                        n = |a+ - 2 a0 + a-|, D = |a+ - a0| + |a0 - a-|
//                        + reltol (|a-| + 2 |a0| + |a+|) + abstol, the indicator
//                        sqrt(sum n^2 / sum D^2) (reach 1)
//
//   tagging_from(dict, method_key): the input's simulation/AMR/refinement/tagging: a model's
//   method (hybrid_method, mhd_method) "default" or "lohner-per-direction", Q<i>/name and
//   Q<i>/threshold (or B at threshold), params/reltol and params/abstol

#include "damn/core/geometry/box_spans.hpp"

#include "plamn/amr/resources.hpp"
#include "plamn/core/def.hpp"
#include "plamn/core/models/quantities/quantity.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>
#include <cstdlib>
#include <format>
#include <type_traits>

namespace damn::core
{
template<typename Type, std::size_t dim>
class Point;
template<typename Type, std::size_t dim>
struct Box;
} // namespace damn::core

namespace plamn::amr
{
// a field resource's value at the centre of cell c: the mean over the cell's nodes in its
// primal directions
template<typename Field, std::uint32_t gw>
NO_DISCARD double cell_centred(FieldResource<Field, gw> const& r,
                               damn::core::Box<int, Field::dimension> const& cells,
                               damn::core::Point<int, Field::dimension> const& c)
{
    auto constexpr dim   = Field::dimension;
    auto const centering = core::centering<dim>(*r.field);
    double sum           = 0;
    int n                = 0;
    std::array<int, dim> at{}; // corner offsets
    while (true)
    {
        auto p = c;
        for (std::size_t d = 0; d < dim; ++d)
            p[d] += at[d];
        sum += value(r, cells, p);
        ++n;
        std::size_t d = 0;
        for (; d < dim; ++d)
        {
            if (centering[d] == core::QtyCentering::primal and at[d] == 0)
            {
                at[d] = 1;
                break;
            }
            at[d] = 0;
        }
        if (d == dim)
            break;
    }
    return sum / n;
}


struct DefaultCriterion
{
    template<std::size_t dim>
    static constexpr int reach = dim == 1 ? 3 : 2;

    template<typename Resource, std::size_t dim>
    NO_DISCARD double
    indicator(std::vector<Resource> const& components, damn::core::Box<int, dim> const& cells,
              damn::core::Point<int, dim> const& c, std::array<double, dim> const&) const
    {
        auto const at = [&](auto const& F, std::size_t const d, int const offset) {
            auto p = c;
            p[d] += offset;
            return cell_centred(F, cells, p);
        };
        if constexpr (dim == 1)
        {
            double sum = 0;
            for (auto const& F : components)
            {
                double avg = 0, avg1 = 0;
                for (int k = -2; k <= 2; ++k)
                {
                    avg += at(F, 0, k) / 5;
                    avg1 += at(F, 0, k + 1) / 5;
                }
                auto const v = std::abs(avg1 - avg) / (1 + std::abs(avg));
                sum += v * v;
            }
            return std::sqrt(sum);
        }
        else
        {
            double max = 0;
            for (auto const& F : components)
                for (std::size_t d = 0; d < dim; ++d)
                {
                    auto const f0 = at(F, d, 0);
                    auto const v  = std::abs(at(F, d, 2) - f0) / (1 + std::abs(at(F, d, 1) - f0));
                    if (!std::isnan(v))
                        max = std::max(max, v);
                }
            return max;
        }
    }
};

struct LohnerCriterion
{
    template<std::size_t>
    static constexpr int reach = 1;

    double reltol = 0.01;
    double abstol = 1e-8;

    template<typename Resource, std::size_t dim>
    NO_DISCARD double
    indicator(std::vector<Resource> const& components, damn::core::Box<int, dim> const& cells,
              damn::core::Point<int, dim> const& c, std::array<double, dim> const&) const
    {
        auto const norm = [](auto const& v) {
            double s = 0;
            for (auto const x : v)
                s += x * x;
            return std::sqrt(s);
        };
        double num = 0, den = 0;
        for (std::size_t d = 0; d < dim; ++d)
        {
            std::vector<double> m(components.size()), z(components.size()), p(components.size());
            for (std::size_t k = 0; k < components.size(); ++k)
            {
                auto cm = c, cp = c;
                --cm[d];
                ++cp[d];
                m[k] = cell_centred(components[k], cells, cm);
                z[k] = cell_centred(components[k], cells, c);
                p[k] = cell_centred(components[k], cells, cp);
            }
            std::vector<double> second(m.size()), up(m.size()), down(m.size());
            for (std::size_t k = 0; k < m.size(); ++k)
            {
                second[k] = p[k] - 2 * z[k] + m[k];
                up[k]     = p[k] - z[k];
                down[k]   = z[k] - m[k];
            }
            auto const n = norm(second);
            auto const D
                = norm(up) + norm(down) + reltol * (norm(m) + 2 * norm(z) + norm(p)) + abstol;
            num += n * n;
            den += D * D;
        }
        return den > 0 ? std::sqrt(num / den) : 0.;
    }
};


// what to tag on: a quantity's components (resource names) and its threshold
struct TaggedQuantity
{
    std::vector<std::string> components;
    double threshold = 0.1;
};

struct Tagging
{
    std::variant<DefaultCriterion, LohnerCriterion> criterion;
    std::vector<TaggedQuantity> quantities;
};

// B, Bx, By, Bz: the components' names, model free (B_x...: an MHD level's resources; a hybrid
//  level's are EM_B_x...)
NO_DISCARD inline std::vector<std::string> tagged_components(std::string const& name)
{
    if (name == "B")
        return {"B_x", "B_y", "B_z"};
    if (name == "Bx" or name == "By" or name == "Bz")
        return {std::string{"B_"} + name[1]};
    throw std::runtime_error("plamn tagging: " + name
                             + " is not a quantity to tag on (B, Bx, By, Bz)");
}

// the input's simulation/AMR/refinement/tagging node, the method of a model's levels by
//  method_key (PHARE's TaggerFactory: hybrid_method, mhd_method)
NO_DISCARD inline Tagging tagging_from(auto const& tagging,
                                       std::string const& method_key = "hybrid_method")
{
    Tagging t;
    auto const method = tagging.contains(method_key)
                            ? tagging[method_key].template to<std::string>()
                            : std::string{"default"};
    if (method == "default")
        t.criterion = DefaultCriterion{};
    else if (method == "lohner-per-direction")
    {
        LohnerCriterion l;
        if (tagging.contains("params"))
        {
            auto const& params = tagging["params"];
            if (params.contains("reltol"))
                l.reltol = params["reltol"].template to<double>();
            if (params.contains("abstol"))
                l.abstol = params["abstol"].template to<double>();
        }
        t.criterion = l;
    }
    else
        throw std::runtime_error("plamn tagging: no method " + method
                                 + " (default, lohner-per-direction)");

    auto const threshold
        = tagging.contains("threshold") ? tagging["threshold"].template to<double>() : 0.1;
    if (!tagging.contains("nbr_quantities"))
        t.quantities.push_back({tagged_components("B"), threshold});
    else
        for (int i = 0; i < tagging["nbr_quantities"].template to<int>(); ++i)
        {
            auto const& q = tagging["Q" + std::to_string(i)];
            t.quantities.push_back({tagged_components(q["name"].template to<std::string>()),
                                    q["threshold"].template to<double>()});
        }
    return t;
}

// the cells of a patch to refine (level index space): resource(name) gives the patch's field
// resource called name
template<std::size_t dim>
NO_DISCARD std::vector<damn::core::Point<int, dim>>
tagged_cells(Tagging const& tagging, damn::core::Box<int, dim> const& cells,
             std::array<double, dim> const& dx, auto&& resource)
{
    std::vector<damn::core::Point<int, dim>> out;
    std::visit(
        [&](auto const& criterion) {
            using Resource = std::decay_t<decltype(resource(std::string{}))>;
            std::vector<std::vector<Resource>> quantities;
            for (auto const& q : tagging.quantities)
            {
                auto& components = quantities.emplace_back();
                for (auto const& name : q.components)
                    components.push_back(resource(name));
            }
            damn::core::for_each_point(cells, [&](auto const& c) {
                for (std::size_t i = 0; i < quantities.size(); ++i)
                    if (criterion.indicator(quantities[i], cells, c, dx)
                        > tagging.quantities[i].threshold)
                    {
                        out.push_back(c);
                        return;
                    }
            });
        },
        tagging.criterion);
    return out;
}

} // namespace plamn::amr

#endif // PLAMN_AMR_TAGGING_HPP
