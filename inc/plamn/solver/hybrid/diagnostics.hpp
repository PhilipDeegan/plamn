#ifndef PLAMN_SOLVER_HYBRID_DIAGNOSTICS_HPP
#define PLAMN_SOLVER_HYBRID_DIAGNOSTICS_HPP

// what the diagnostics write of a hybrid level (the writers: simulator_diagnostics.hpp): its
//  quantities' fields on a patch (PHARE's names: /EM_B, /ions/pop/<name>/density, ...) and its
//  particle quantities' populations

#include "damn/core/def.hpp"

#include "plamn/diagnostic/quantities.hpp"
#include "plamn/solver/hybrid/model.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <initializer_list>

namespace plamn::solver::hybrid
{
// quantity -> its fields on a patch's view
template<auto opts>
NO_DISCARD auto diagnostic_quantities(HybridModel<opts> const& model)
{
    using View_t = HybridModel<opts>::View_t;
    return diagnostic::hybrid_quantities<View_t>(model.population_names);
}

// a particle diagnostic's population index (/ions/pop/<name>/domain or levelGhost)
template<auto opts>
NO_DISCARD std::size_t particle_population(HybridModel<opts> const& model,
                                           std::string const& quantity)
{
    for (std::size_t i = 0; i < model.population_names.size(); ++i)
        for (auto const* kind : {"/domain", "/levelGhost"})
            if (quantity == "/ions/pop/" + model.population_names[i] + kind)
                return i;
    throw std::runtime_error("plamn diagnostics: unsupported particle quantity " + quantity
                             + " (/ions/pop/<name>/domain or levelGhost)");
}

} // namespace plamn::solver::hybrid

#endif // PLAMN_SOLVER_HYBRID_DIAGNOSTICS_HPP
