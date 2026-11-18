#ifndef PLAMN_TST_CORE_NUMERICS_TIME_INTEGRATOR_MHD_PATCH_FIXTURES_HPP
#define PLAMN_TST_CORE_NUMERICS_TIME_INTEGRATOR_MHD_PATCH_FIXTURES_HPP

// an MHD patch owning what the time integration schemes use (core/numerics/time_integrator):
//  the states (enough for any scheme), E, the fluxes, the constrained transport coefficients and
//  the butcher sums

#include "plamn/core/models/mhd_state.hpp"
#include "plamn/core/models/quantities/mhd_quantities.hpp"
#include "plamn/simulator/simulator_def.hpp"

#include "tst/plamn/core/data/vecfield/test_vecfield_fixtures.hpp"

#include <array>
#include <cstddef>
#include <string>

namespace plamn::core
{
template<std::size_t dim_>
struct IntegratorPatch
{
    auto constexpr static dim = dim_;
    auto constexpr static opts
        = SimOpts{.dimension           = dim,
                  .interp_order        = 1,
                  .reconstruction_type = MHDOpts::ReconstructionType::Linear,
                  .slope_limiter_type  = MHDOpts::SlopeLimiterType::VanLeer,
                  .riemann_solver_type = MHDOpts::RiemannSolverType::Rusanov};
    using Types                           = PLAMN_Types<opts>::MHD;
    using Layout_t                        = Types::Layout_t;
    using Grid_t                          = Types::Grid_t;
    auto constexpr static vf_opts         = TensorFieldOptions<Types>{};
    using VecField_t                      = UsableVecField<vf_opts>;
    using S                               = MHDQuantity::Scalar;
    using V_                              = MHDQuantity::Vector;
    using State_t                         = MHDState<Grid_t, VecField_t>;
    using Fluxes_t                        = MHDFaceFluxes<Grid_t, VecField_t>;
    using CT_t                            = MHDFaceCT<Grid_t, VecField_t>;
    static constexpr std::size_t n_states = 5; // the most any integrator needs

    explicit IntegratorPatch(Layout_t const& layout_)
        : layout{layout_}
    {
    }

    State_t state(std::size_t const i) const
    {
        auto const n = std::to_string(i);
        return State_t{Grid_t{"rho" + n, layout, S::rho},  VecField_t{"rhoV" + n, layout, V_::rhoV},
                       VecField_t{"B" + n, layout, V_::B}, Grid_t{"Etot" + n, layout, S::Etot},
                       VecField_t{"V" + n, layout, V_::V}, Grid_t{"P" + n, layout, S::P}};
    }
    Fluxes_t fluxes_(std::string const& n, S const sf, V_ const vf) const
    {
        return Fluxes_t{Grid_t{"F_rho" + n, layout, sf}, VecField_t{"F_rhoV" + n, layout, vf},
                        VecField_t{"F_B" + n, layout, vf}, Grid_t{"F_Etot" + n, layout, sf}};
    }
    CT_t ct_(std::string const& n, S const sf, V_ const vf) const
    {
        return CT_t{VecField_t{"vt" + n, layout, vf}, Grid_t{"aL" + n, layout, sf},
                    Grid_t{"aR" + n, layout, sf}, Grid_t{"dL" + n, layout, sf},
                    Grid_t{"dR" + n, layout, sf}};
    }

    Layout_t layout;
    std::array<State_t, n_states> states{state(0), state(1), state(2), state(3), state(4)};
    VecField_t E{"E", layout, V_::E}, butcherE{"butcherE", layout, V_::E};
    std::array<Fluxes_t, 3> fluxes{fluxes_("x", S::ScalarFlux_x, V_::VecFlux_x),
                                   fluxes_("y", S::ScalarFlux_y, V_::VecFlux_y),
                                   fluxes_("z", S::ScalarFlux_z, V_::VecFlux_z)};
    std::array<Fluxes_t, 3> butcher{fluxes_("bx", S::ScalarFlux_x, V_::VecFlux_x),
                                    fluxes_("by", S::ScalarFlux_y, V_::VecFlux_y),
                                    fluxes_("bz", S::ScalarFlux_z, V_::VecFlux_z)};
    std::array<CT_t, 3> ct{ct_("x", S::ScalarFlux_x, V_::VecFlux_x),
                           ct_("y", S::ScalarFlux_y, V_::VecFlux_y),
                           ct_("z", S::ScalarFlux_z, V_::VecFlux_z)};
};

} // namespace plamn::core

#endif // PLAMN_TST_CORE_NUMERICS_TIME_INTEGRATOR_MHD_PATCH_FIXTURES_HPP
