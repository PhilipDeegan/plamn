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
//   hybrid_quantities(population_names)  quantity -> its fields on a patch (PHARE's VTKHDF diagnostics'
//                                        accessors), components in file order; an unknown
//                                        quantity throws (quantity_for)
//   primal_into(PrimalInto{field, cells, out, n_components, component})
//                                        a patch's field on the primal nodes of its cells, x
//                                        fastest, written straight into its rows: dual directions
//                                        averaged (the neighbouring dual values: ghosts read),
//                                        each value at row * n_components + component, the 1D /
//                                        2D block repeated x_times (a 3D view)

#include "damn/core/def.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/hdf5/vtkhdf.hpp"

#include "plamn/amr/resources.hpp"
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
#include <vector>

namespace plamn::diagnostic
{
struct Diagnostic
{
    std::string type, quantity;
    std::vector<double> timestamps;
    std::size_t flush_every = 1, next = 0, dumps = 0;
    std::vector<std::pair<std::string, std::string>> attributes; // phareh5's /py_attrs
};

// pyphare's diagnostics plamn writes: electromag and fluid, particle (phareh5 only)
NO_DISCARD inline std::vector<Diagnostic> diagnostics_from(initializer::Dict const& dict)
{
    std::vector<Diagnostic> out;
    if (!dict["simulation"].contains("diagnostics"))
        return out;
    auto const& diags = dict["simulation"]["diagnostics"];
    for (auto const* type : {"electromag", "fluid", "particle"})
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
                    diag.attributes.emplace_back(d[at + "_key"].template to<std::string>(),
                                                 d[at + "_value"].template to<std::string>());
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

// a quantity's fields in a patch's view (simulator::HybridPatchView), components in file order
//  (x y z; xx xy xz yy yz zz); the pointers into the view
template<typename View>
using Accessor = std::function<std::vector<typename View::field_type const*>(View const&)>;

template<typename View>
NO_DISCARD std::map<std::string, Accessor<View>>
hybrid_quantities(std::vector<std::string> const& population_names)
{
    using Field    = View::field_type;
    using Fields   = std::vector<Field const*>;
    auto const vec = [](auto const& vf) { return Fields{&vf[0], &vf[1], &vf[2]}; };
    std::map<std::string, Accessor<View>> q;
    q["/EM_B"]                = [=](View const& v) { return vec(v.em.B); };
    q["/EM_E"]                = [=](View const& v) { return vec(v.em.E); };
    q["/ions/charge_density"] = [](View const& v) { return Fields{&v.charge_density}; };
    q["/ions/mass_density"]   = [](View const& v) { return Fields{&v.mass_density}; };
    q["/ions/bulkVelocity"]   = [=](View const& v) { return vec(v.bulk_velocity); };
    q["/ions/charge_flux"]    = [=](View const& v) { return vec(v.charge_flux); };
    for (std::size_t i = 0; i < population_names.size(); ++i)
    {
        auto const pop      = "/ions/pop/" + population_names[i];
        q[pop + "/density"] = [i](View const& v) { return Fields{&v.populations[i].density}; };
        q[pop + "/charge_density"]
            = [i](View const& v) { return Fields{&v.populations[i].charge_density}; };
        q[pop + "/flux"] = [i, vec](View const& v) { return vec(v.populations[i].flux); };
    }
    return q;
}

template<typename View>
NO_DISCARD Accessor<View> const&
quantity_for(std::map<std::string, Accessor<View>> const& quantities, std::string const& quantity)
{
    auto const it = quantities.find(quantity);
    if (it == quantities.end())
        throw std::runtime_error("plamn diagnostics: unsupported quantity " + quantity);
    return it->second;
}


// a patch's field, its cells, where its rows go and which column it is
template<typename Field, typename Float>
struct PrimalInto
{
    Field const& field;
    damn::core::Box<int, Field::dimension> cells;
    std::span<Float> out;
    std::size_t n_components = 1, component = 0;
};

template<std::uint32_t gw, typename Field, typename Float>
void primal_into(PrimalInto<Field, Float> const& in)
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
