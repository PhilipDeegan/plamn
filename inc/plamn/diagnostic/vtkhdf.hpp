#ifndef PLAMN_DIAGNOSTIC_VTKHDF_HPP
#define PLAMN_DIAGNOSTIC_VTKHDF_HPP

// pharevtkhdf diagnostics (PHARE's format, see damn/hdf5/vtkhdf.hpp for the file), the pieces:
//
//   Diagnostic, diagnostics_from(dict)   what pyphare declared (simulation/diagnostics/<type>/
//                                        <type><N>: quantity, write_timestamps, flush_every,
//                                        attributes); electromag, fluid and particle (phareh5
//                                        only) ones (others: not written)
//   due(diagnostic, t, dt)               PHARE's schedule: a timestamp within dt of t (as
//                                        floats), past ones caught up
//   file_name(quantity)                  /ions/pop/protons/flux -> ions_pop_protons_flux.vtkhdf
//   hybrid_quantities, quantity_for      (diagnostic/quantities.hpp)
//   primal_into(PrimalInto{field, cells, out, n_components, component})
//                                        a patch's field on the primal nodes of its cells, x
//                                        fastest, written straight into its rows: dual directions
//                                        averaged (the neighbouring dual values: ghosts read),
//                                        each value at row * n_components + component, the 1D /
//                                        2D block repeated x_times (a 3D view)

#include "plamn/amr/untiled.hpp"
#include "damn/core/def.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/hdf5/vtkhdf.hpp"

#include "plamn/amr/resources.hpp"
#include "plamn/diagnostic/quantities.hpp" // IWYU pragma: export
#include "plamn/initializer/data_provider.hpp"
#include "plamn/core/data/grid/layout_defs.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"

#include "dict.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace plamn::diagnostic
{
struct Diagnostic
{
    std::string type, quantity;
    std::vector<double> timestamps{};
    std::size_t flush_every = 1, next = 0, dumps = 0;
    // phareh5's /py_attrs: strings, but pyphare's heat_capacity_ratio (a double, as PHARE's MHD
    //  writer reads it)
    std::vector<std::pair<std::string, std::variant<std::string, double>>> attributes{};
};

// pyphare's diagnostics plamn writes: electromag and fluid, particle (phareh5 only)
NO_DISCARD inline std::vector<Diagnostic> diagnostics_from(initializer::Dict const& dict)
{
    std::vector<Diagnostic> out;
    if (!dict["simulation"].contains("diagnostics"))
        return out;
    auto const& diags = dict["simulation"]["diagnostics"];
    for (auto const* type : {"electromag", "fluid", "mhd", "particle"})
    {
        if (!diags.contains(type))
            continue;
        for (std::size_t n = 0;; ++n)
        {
            auto const key = std::string{type} + std::to_string(n);
            if (!diags[type].contains(key))
                break;
            auto const& d = diags[type][key];
            Diagnostic diag{.type = type, .quantity = d["quantity"].template to<std::string>()};
            if (d.contains("write_timestamps"))
                diag.timestamps = d["write_timestamps"].template to<std::vector<double>>();
            if (d.contains("flush_every"))
                diag.flush_every = d["flush_every"].template to<std::size_t>();
            if (d.contains("n_attributes"))
                for (std::size_t i = 0; i < d["n_attributes"].template to<std::size_t>(); ++i)
                {
                    auto const at = "attribute_" + std::to_string(i);
                    auto const& value = d[at + "_value"];
                    auto const key    = d[at + "_key"].template to<std::string>();
                    if (auto const* number = std::get_if<double>(&value.data))
                        diag.attributes.emplace_back(key, *number);
                    else
                        diag.attributes.emplace_back(key, std::get<std::string>(value.data));
                }
            out.push_back(diag);
        }
    }
    return out;
}

// PHARE's DiagnosticsManager: due when a timestamp is within dt of t, compared as floats; the
// timestamps passed are caught up
NO_DISCARD inline bool due(Diagnostic& diag, double const t, double const dt)
{
    bool is_due = false;
    while (diag.next < diag.timestamps.size()
           and static_cast<float>(diag.timestamps[diag.next] - t) < static_cast<float>(dt))
    {
        ++diag.next;
        is_due = true;
    }
    return is_due;
}

NO_DISCARD inline std::string file_name(std::string const& quantity)
{
    auto name = quantity.substr(quantity.find_first_not_of('/'));
    std::replace(name.begin(), name.end(), '/', '_');
    return name + ".vtkhdf";
}

template<typename Field, typename Float>
struct PrimalInto
{
    Field const& field;
    damn::core::Box<int, Field::dimension> cells;
    std::span<Float> out;
    std::size_t n_components = 1, component = 0;
};

template<std::uint32_t gw, typename Field, typename Float>
void primal_into_untiled_(PrimalInto<Field, Float> const& in)
{
    auto constexpr dim                                       = Field::dimension;
    auto const& [field, cells, out, n_components, component] = in;
    auto const resource   = amr::field_resource<gw>(const_cast<Field&>(field));
    auto const centerings = core::centering<dim>(field);

    // the primal nodes of the cells, x fastest
    damn::core::Box<int, dim> nodes = cells;
    for (std::size_t i = 0; i < dim; ++i)
        nodes.upper[i] += 1;
    auto const points = damn::hdf5::vtkhdf::points_of(cells);
    if (out.size() < damn::hdf5::vtkhdf::rows_of(cells) * n_components)
        throw std::runtime_error("plamn diagnostics: rows too small for the patch");

    std::array<int, dim> shape{};
    for (std::size_t i = 0; i < dim; ++i)
        shape[i] = nodes.upper[i] - nodes.lower[i] + 1;

    std::size_t n_dual = 0;
    for (std::size_t i = 0; i < dim; ++i)
        n_dual += centerings[i] == core::QtyCentering::dual;
    double const weight = 1. / static_cast<double>(1u << n_dual);

    damn::core::for_each_point(nodes, [&](auto const& node) {
        // the node's row: Fortran order
        std::size_t row = 0;
        for (std::size_t i = dim; i-- > 0;)
            row = row * static_cast<std::size_t>(shape[i])
                  + static_cast<std::size_t>(node[i] - nodes.lower[i]);

        double v = 0;
        for (std::uint32_t corner = 0; corner < (1u << dim); ++corner)
        {
            damn::core::Point<int, dim> p = node;
            bool used                     = true;
            for (std::size_t i = 0; i < dim; ++i)
            {
                bool const high = corner >> i & 1u;
                if (centerings[i] == core::QtyCentering::dual)
                    p[i] -= high ? 0 : 1; // the duals either side: cells node - 1 and node
                else if (high)
                    used = false; // primal: the node itself, once
            }
            if (used)
                v += value(resource, cells, p);
        }
        auto const f = static_cast<Float>(v * weight);
        for (std::size_t copy = 0; copy < damn::hdf5::vtkhdf::x_times<dim>; ++copy)
            out[(copy * points + row) * n_components + component] = f;
    });
}

// tiled fields from an untiled copy of their tiles
template<std::uint32_t gw, typename Field, typename Float>
void primal_into(PrimalInto<Field, Float> const& in)
{
    amr::with_untiled(in.field, [&](auto const& f) {
        primal_into_untiled_<gw>(PrimalInto<std::decay_t<decltype(f)>, Float>{
            f, in.cells, in.out, in.n_components, in.component});
    });
}

// the same from PrimalInto's members as arguments
template<std::uint32_t gw, typename Field, typename Float>
void primal_into(Field const& field, damn::core::Box<int, Field::dimension> const& cells,
                 std::span<Float> const out, auto&&... args)
{
    primal_into<gw>(
        PrimalInto<Field, Float>{field, cells, out, std::forward<decltype(args)>(args)...});
}

} // namespace plamn::diagnostic

#endif // PLAMN_DIAGNOSTIC_VTKHDF_HPP
