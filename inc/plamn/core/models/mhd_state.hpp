#ifndef PLAMN_CORE_MODELS_MHD_STATE_HPP
#define PLAMN_CORE_MODELS_MHD_STATE_HPP

// the MHD model's per patch data, as aggregates of (vec)fields owned by whoever builds them (an
//  MHD patch, a test): the time integrators (core/numerics/time_integrator/mhd_time_integrator.hpp)
//  only name the members
namespace plamn::core
{

// a state: conservative (rho, rhoV, B, Etot), what is evolved, and its primitive (V, P),
//  computed from it (to_primitive) where fluxes need it, as PHARE's MHDState
template<typename Grid, typename VecField>
struct MHDState
{
    Grid rho;
    VecField rhoV, B;
    Grid Etot;
    VecField V;
    Grid P;
};

// one face direction's fluxes of the conserved quantities (F_B: unused by the ideal update, B
//  is evolved by Faraday from the constrained transport E, as PHARE)
template<typename Grid, typename VecField>
struct MHDFaceFluxes
{
    Grid F_rho;
    VecField F_rhoV, F_B;
    Grid F_Etot;
};

// one face direction's upwind constrained transport inputs (GodunovFaceResult's vt, uct_coefs)
template<typename Grid, typename VecField>
struct MHDFaceCT
{
    VecField vt;
    Grid aL, aR, dL, dR;
};

} // namespace plamn::core

#endif /* PLAMN_CORE_MODELS_MHD_STATE_HPP */
