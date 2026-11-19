#ifndef PLAMN_SOLVER_ADVANCE_HPP
#define PLAMN_SOLVER_ADVANCE_HPP

// a level of one model from t0 to t1: one overload per model, each there only when the
//  simulation's options enable it
//
//   MHD    Godunov fluxes and constrained transport, integrated in time by the model's scheme
//          (core/numerics/time_integrator): the level adapter's patches hold the scheme's states
//          (states[0] advanced), and it fills their ghosts (fill_state_ghosts,
//          fill_electric_ghosts: neighbours, coarser levels, physical boundaries)

#include "plamn/core/models/options/mhd_options_def.hpp"
#include "plamn/core/numerics/time_integrator/mhd_time_integrator.hpp"
#include "plamn/solver/mhd/model.hpp"

#include <stdexcept>

namespace plamn::solver
{
template<auto opts>
void advance(MHDModel<opts> const& model, auto& level, double const t0, double const t1)
{
    using R  = MHDModel<opts>::Reconstruction;
    using RS = MHDModel<opts>::Riemann;
    core::MHDSteps<R, RS> const steps{model.gamma, model.resistivity};
    switch (model.scheme)
    {
        case MHDOpts::TimeIntegratorType::Euler:
            return core::EulerIntegrator<R, RS>{steps}(level, t0, t1);
        case MHDOpts::TimeIntegratorType::TVDRK2:
            return core::TVDRK2Integrator<R, RS>{steps}(level, t0, t1);
        case MHDOpts::TimeIntegratorType::TVDRK3:
            return core::TVDRK3Integrator<R, RS>{steps}(level, t0, t1);
        case MHDOpts::TimeIntegratorType::SSPRK4_5:
            return core::SSPRK4_5Integrator<R, RS>{steps}(level, t0, t1);
        default: throw std::runtime_error("plamn: no MHD time integration scheme");
    }
}

} // namespace plamn::solver

#endif // PLAMN_SOLVER_ADVANCE_HPP
