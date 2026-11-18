#ifndef PLAMN_DIAGNOSTIC_QUANTITIES_HPP
#define PLAMN_DIAGNOSTIC_QUANTITIES_HPP

// a diagnostic quantity's fields on a patch (PHARE's diagnostics' accessors), whatever the format
//   hybrid_quantities(population_names)  quantity -> its fields on a hybrid patch's view,
//                                        components in file order
//   quantity_for(quantities, quantity)   an unknown quantity throws

#include "damn/core/def.hpp"

#include <cstddef>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace plamn::diagnostic
{
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

} // namespace plamn::diagnostic

#endif // PLAMN_DIAGNOSTIC_QUANTITIES_HPP
