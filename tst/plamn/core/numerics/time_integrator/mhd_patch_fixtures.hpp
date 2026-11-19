#ifndef PLAMN_TST_CORE_NUMERICS_TIME_INTEGRATOR_MHD_PATCH_FIXTURES_HPP
#define PLAMN_TST_CORE_NUMERICS_TIME_INTEGRATOR_MHD_PATCH_FIXTURES_HPP

// the simulator's MHD patch, with the states any time integration scheme uses
//  (core/numerics/time_integrator), for tests of a scheme over one patch

#include "plamn/core/models/options/mhd_options_def.hpp"
#include "plamn/simulator/mhd_patch.hpp"
#include "plamn/simulator/simulator_def.hpp"

#include <cstddef>

namespace plamn::core
{
template<std::size_t dim_>
auto constexpr integrator_opts
    = SimOpts{.dimension           = dim_,
              .interp_order        = 1,
              .reconstruction_type = MHDOpts::ReconstructionType::Linear,
              .slope_limiter_type  = MHDOpts::SlopeLimiterType::VanLeer,
              .riemann_solver_type = MHDOpts::RiemannSolverType::Rusanov};

template<std::size_t dim_>
struct IntegratorPatch : simulator::MHDPatch<integrator_opts<dim_>>
{
    using Super                           = simulator::MHDPatch<integrator_opts<dim_>>;
    auto constexpr static dim             = dim_;
    auto constexpr static opts            = integrator_opts<dim_>;
    static constexpr std::size_t n_states = 5; // the most any integrator needs

    explicit IntegratorPatch(Super::Layout_t const& layout)
        : Super{layout, n_states}
    {
    }
};

} // namespace plamn::core

#endif // PLAMN_TST_CORE_NUMERICS_TIME_INTEGRATOR_MHD_PATCH_FIXTURES_HPP
