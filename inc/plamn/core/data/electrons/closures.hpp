#ifndef PLAMN_CORE_DATA_ELECTRONS_CLOSURES_HPP
#define PLAMN_CORE_DATA_ELECTRONS_CLOSURES_HPP

// the electron pressure closures: one per model, chosen at run time by the input's
//  electrons/pressure_closure/name, operating on whole levels (solver/hybrid/closures.hpp: an
//  overload per closure); a patch holds the electrons' views (Ne, Ve, Pe) and the storage of
//  the closure's own fields (closure_fields: the same list on every patch of a run)
//   isothermal   Pe = Ne Te, Te constant in space and time
//   polytropic   Pe = s Ne^gamma, the entropy s = Pe / Ne^gamma carried by the electron flow,
//                ds/dt + Ve.grad s = 0 (first order upwind, sub-stepped past Courant 1);
//                gamma: the simulation's; the initial s: the input's Pe / n^gamma (from the
//                analytic densities, free of the deposit's noise)

#include "plamn/core/models/quantities/hybrid_quantities.hpp"
#include "plamn/initializer/data_provider.hpp"
#include "dict.hpp"

#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace plamn::core
{
struct IsothermalClosure
{
    static constexpr char const* name = "isothermal";
    double Te                         = 0;
};

struct PolytropicClosure
{
    static constexpr char const* name = "polytropic";
    double gamma                      = 5. / 3.;

    // s at the start of a step (from Pe / Ne^gamma on the domain, ghosts filled), and s within
    //  a stage (from Se_old; after a stage, at its end: what a finer level's ghosts interpolate
    //  to)
    static constexpr char const* step_state     = "Se_old";
    static constexpr char const* subcycle_state = "Se";
    // most sub-steps a stage takes (beyond: the step is refused)
    static constexpr double max_subcycled_courant = 10;
    static constexpr double courant_tolerance     = 1e-12;
};

using ElectronClosure = std::variant<IsothermalClosure, PolytropicClosure>;

// dict: the electrons' "pressure_closure" node
inline ElectronClosure make_closure(initializer::Dict const& dict)
{
    auto const name = dict["name"].template to<std::string>();
    if (name == IsothermalClosure::name)
        return IsothermalClosure{dict["Te"].template to<double>()};
    if (name == PolytropicClosure::name)
        return PolytropicClosure{dict["gamma"].template to<double>()};
    throw std::runtime_error("plamn: unknown electron pressure closure \"" + name + "\"");
}

// the fields a closure keeps on every patch: a scalar, a vector (name_x, _y, _z) or a symmetric
//  tensor (name_xx, _xy, _xz, _yy, _yz, _zz), each with its quantity (centering)
struct ClosureScalarField
{
    std::string name;
    HybridQuantity::Scalar qty;
};
struct ClosureVectorField
{
    std::string name;
    HybridQuantity::Vector qty;
};
struct ClosureTensorField
{
    std::string name;
    HybridQuantity::Tensor qty;
};
using ClosureField = std::variant<ClosureScalarField, ClosureVectorField, ClosureTensorField>;

inline std::vector<ClosureField> closure_fields(IsothermalClosure const&)
{
    return {};
}
inline std::vector<ClosureField> closure_fields(PolytropicClosure const&)
{ // s is primal, as the moments and Pe
    return {ClosureScalarField{PolytropicClosure::step_state, HybridQuantity::Scalar::P},
            ClosureScalarField{PolytropicClosure::subcycle_state, HybridQuantity::Scalar::P}};
}
inline std::vector<ClosureField> closure_fields(ElectronClosure const& closure)
{
    return std::visit([](auto const& c) { return closure_fields(c); }, closure);
}

// what a restart keeps of a closure: what its next step reads first (polytropic: Pe, from
//  which the first stage takes Se_old; Se is every stage's, the coarser level's redone before a
//  finer level's ghosts read it)
inline std::vector<std::string> closure_restart_names(IsothermalClosure const&)
{
    return {};
}
inline std::vector<std::string> closure_restart_names(PolytropicClosure const&)
{
    return {"Pe"};
}
inline std::vector<std::string> closure_restart_names(ElectronClosure const& closure)
{
    return std::visit([](auto const& c) { return closure_restart_names(c); }, closure);
}

// the electron temperature splitting an MHD pressure into Pi + Pe at the model boundary
//  (simulator/coupling.hpp: Pi = max(P - Pe, 0), Pe = Te q n): an overload per closure
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

} // namespace plamn::core

#endif // PLAMN_CORE_DATA_ELECTRONS_CLOSURES_HPP
