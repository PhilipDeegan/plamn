#ifndef PLAMN_CORE_DATA_ELECTRONS_CLOSURES_HPP
#define PLAMN_CORE_DATA_ELECTRONS_CLOSURES_HPP

// electron pressure closures: doc/core/data/electrons/closures.md

#include "plamn/core/data/field/field_descriptor.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"
#include "plamn/initializer/data_provider.hpp"
#include "dict.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace plamn::core::hybrid
{
using ScalarFieldDescriptor = core::ScalarFieldDescriptor<HybridQuantity>;
using VectorFieldDescriptor = core::VectorFieldDescriptor<HybridQuantity>;
using TensorFieldDescriptor = core::TensorFieldDescriptor<HybridQuantity>;
using FieldDescriptor       = core::FieldDescriptor<HybridQuantity>;

struct IsothermalClosure
{
    static constexpr char const* name = "isothermal";
    double Te                         = 0;
};

struct PolytropicClosure
{
    static constexpr char const* name = "polytropic";
    double gamma                      = 5. / 3.;

    // index in a patch's closure storage: doc/core/data/electrons/closures.md
    enum Field : std::uint8_t { step, subcycle, scratch, count };
    static constexpr std::array<char const*, count> names{"Se_old", "Se", "Se_tmp"};
    // most sub-steps a stage takes (beyond: the step is refused)
    static constexpr double max_subcycled_courant = 10;
    static constexpr double courant_tolerance     = 1e-12;
};

using ElectronClosure = std::variant<IsothermalClosure, PolytropicClosure>;

// the polytropic closure advects s with Ve on the moment nodes, from the moment nodes' ion
//  flux, which the Ohm energy fix stops depositing in the time loop: off until that settles
//  (its code still compiled, tests skipped)
inline constexpr bool polytropic_enabled = false;

// dict: the electrons' "pressure_closure" node
inline ElectronClosure make_electron_closure(initializer::Dict const& dict)
{
    auto const name = dict["name"].template to<std::string>();
    if (name == IsothermalClosure::name)
        return IsothermalClosure{dict["Te"].template to<double>()};
    if (name == PolytropicClosure::name) // disabled: see polytropic_enabled
    {
        if constexpr (polytropic_enabled)
            return PolytropicClosure{dict["gamma"].template to<double>()};
        throw std::runtime_error("plamn: the polytropic electron closure is disabled for now "
                                 "(#91: it waits on the Ohm energy fix's ion flux)");
    }
    throw std::runtime_error("plamn: unknown electron pressure closure \"" + name + "\"");
}

// the fields a closure keeps on every patch
inline std::vector<FieldDescriptor> electron_fields(IsothermalClosure const&)
{
    return {};
}
inline std::vector<FieldDescriptor> electron_fields(PolytropicClosure const&)
{ // s is primal, as the moments and Pe
    std::vector<FieldDescriptor> fields;
    fields.reserve(PolytropicClosure::count);
    for (auto const* name : PolytropicClosure::names)
        fields.emplace_back(ScalarFieldDescriptor{name, HybridQuantity::Scalar::P});
    return fields;
}
inline std::vector<FieldDescriptor> electron_fields(ElectronClosure const& closure)
{
    return std::visit([](auto const& c) { return electron_fields(c); }, closure);
}

// what its next step reads first (polytropic: Pe, from which the first stage takes Se_old; Se is
//  every stage's, the coarser level's redone before a finer level's ghosts read it)
inline std::vector<std::string> electron_restart_names(IsothermalClosure const&)
{
    return {};
}
inline std::vector<std::string> electron_restart_names(PolytropicClosure const&)
{
    return {"Pe"};
}
inline std::vector<std::string> electron_restart_names(ElectronClosure const& closure)
{
    return std::visit([](auto const& c) { return electron_restart_names(c); }, closure);
}

// the electron temperature splitting an MHD pressure into Pi + Pe at the model boundary
//  (Pi = max(P - Pe, 0), Pe = Te q n)
inline double coupling_Te(IsothermalClosure const& closure)
{
    return closure.Te;
}
inline double coupling_Te(PolytropicClosure const&)
{
    throw std::runtime_error("plamn coupling: polytropic electrons over MHD levels not yet (#91)");
}
inline double coupling_Te(ElectronClosure const& closure)
{
    return std::visit([](auto const& c) { return coupling_Te(c); }, closure);
}

} // namespace plamn::core::hybrid

#endif // PLAMN_CORE_DATA_ELECTRONS_CLOSURES_HPP
