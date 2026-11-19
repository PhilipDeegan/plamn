#ifndef PLAMN_SOLVER_MHD_INITIALIZE_HPP
#define PLAMN_SOLVER_MHD_INITIALIZE_HPP

// an MHD patch's advanced state from the user's functions, as PHARE's MHDState::initialize:
//  simulation/mhd_state/{density, velocity, magnetic, pressure}/initializer (the vectors'
//  {x,y,z}_component) give the primitive state on the patch's ghost boxes, converted to the
//  conservative one (the model's heat capacity ratio). Its ghosts past a neighbour or a physical
//  boundary are the level's to fill after

#include "plamn/core/data/field/initializers/field_user_initializer.hpp"
#include "plamn/core/data/vecfield/vecfield_initializer.hpp"
#include "plamn/core/numerics/primitive_conservative/primitive_conservative_transformer.hpp"
#include "plamn/initializer/data_provider.hpp"
#include "plamn/solver/mhd/model.hpp"
#include "dict.hpp"

#include <string>

namespace plamn::solver::mhd
{
template<auto opts>
void initialize_state(MHDModel<opts> const& model, typename MHDModel<opts>::Patch_t& patch,
                      initializer::Dict const& sim)
{
    auto constexpr dim  = opts.dimension;
    auto const& state   = sim["mhd_state"];
    auto const& layout  = patch.layout;
    auto& u             = patch.states[0];
    auto const function = [&](std::string const& name) {
        return state[name]["initializer"].template to<initializer::InitFunction<dim>>();
    };
    core::FieldUserFunctionInitializer::initialize(u.rho, layout, function("density"));
    core::VecFieldInitializer<dim>{state["velocity"]["initializer"]}.initialize(u.V, layout);
    core::VecFieldInitializer<dim>{state["magnetic"]["initializer"]}.initialize(u.B, layout);
    core::FieldUserFunctionInitializer::initialize(u.P, layout, function("pressure"));
    core::PrimitiveConservativeTransformer{model.gamma}.to_conservative(layout, u.rho, u.V, u.B,
                                                                        u.P, u.rhoV, u.Etot);
}

} // namespace plamn::solver::mhd

#endif // PLAMN_SOLVER_MHD_INITIALIZE_HPP
