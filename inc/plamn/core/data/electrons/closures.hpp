#ifndef PLAMN_CORE_DATA_ELECTRONS_CLOSURES_HPP
#define PLAMN_CORE_DATA_ELECTRONS_CLOSURES_HPP

// the electron pressure closures: one per model, chosen at run time by the input's
//  electrons/pressure_closure/name, operating on whole levels (solver/hybrid/closures.hpp: an
//  overload per closure); a patch holds the electrons' views only (Ne, Ve, Pe)
//   isothermal   Pe = Ne Te, Te constant in space and time

#include "plamn/initializer/data_provider.hpp"
#include "dict.hpp"

#include <stdexcept>
#include <string>
#include <variant>

namespace plamn::core
{
struct IsothermalClosure
{
    static constexpr char const* name = "isothermal";
    double Te                         = 0;
};

using ElectronClosure = std::variant<IsothermalClosure>;

// dict: the electrons' "pressure_closure" node
inline ElectronClosure make_closure(initializer::Dict const& dict)
{
    auto const name = dict["name"].template to<std::string>();
    if (name == IsothermalClosure::name)
        return IsothermalClosure{dict["Te"].template to<double>()};
    throw std::runtime_error("plamn: unknown electron pressure closure \"" + name + "\"");
}

// the electron temperature splitting an MHD pressure into Pi + Pe at the model boundary
//  (simulator/coupling.hpp: Pi = max(P - Pe, 0), Pe = Te q n): an overload per closure, so a new
//  closure without its split does not build coupled
inline double coupling_Te(IsothermalClosure const& closure)
{
    return closure.Te;
}
inline double coupling_Te(ElectronClosure const& closure)
{
    return std::visit([](auto const& c) { return coupling_Te(c); }, closure);
}

} // namespace plamn::core

#endif // PLAMN_CORE_DATA_ELECTRONS_CLOSURES_HPP
