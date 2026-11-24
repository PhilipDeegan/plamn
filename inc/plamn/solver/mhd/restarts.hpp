#ifndef PLAMN_SOLVER_MHD_RESTARTS_HPP
#define PLAMN_SOLVER_MHD_RESTARTS_HPP

// what a restart holds of an MHD level, and what the level rebuilds from it (the simulator
//  writes and reads the levels' resources by these names, simulator_restarts.hpp): the advanced
//  state's conservative variables. The scheme's stages, E and the primitive variables are made
//  again within a step; the state's ghosts are filled again after loading

#include "damn/core/def.hpp"
#include "plamn/solver/mhd/model.hpp"

#include <string>
#include <vector>
#include <initializer_list>

namespace plamn::solver::mhd
{
// the resources a restart stores
template<auto opts>
MKN_KUL_FN_SIG_RET std::vector<std::string> restart_names(MHDModel<opts> const&)
{
    std::vector<std::string> names{"rho", "Etot"};
    for (auto const* v : {"rhoV", "B"})
        for (auto const* c : {"_x", "_y", "_z"})
            names.push_back(std::string{v} + c);
    return names;
}

// what the running level holds between steps, from what was stored: the state's ghosts at t
template<auto opts>
void restart_loaded(MHDModel<opts> const&, auto&& level, double const t)
{
    level.fill_state_ghosts(0, t);
}

} // namespace plamn::solver::mhd

#endif // PLAMN_SOLVER_MHD_RESTARTS_HPP
