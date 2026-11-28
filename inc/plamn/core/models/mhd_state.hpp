#ifndef PLAMN_CORE_MODELS_MHD_STATE_HPP
#define PLAMN_CORE_MODELS_MHD_STATE_HPP

// the MHD model's per patch data, as aggregates of (vec)fields owned by whoever builds them (an
//  MHD patch, a test): the time integrators only name the members
namespace plamn::core
{

// a state: conservative (rho, rhoV, B, Etot), what is evolved, and its primitive (V, P),
//  computed from it (to_primitive) where fluxes need it
template<typename Grid, typename VecField>
struct MHDState
{
    Grid rho;
    VecField rhoV, B;
    Grid Etot;
    VecField V;
    Grid P;
};

// one face direction's fluxes of the evolved quantities but B (B is evolved by Faraday from the
//  constrained transport E: the Godunov faces' B fluxes are not stored)
template<typename Grid, typename VecField>
struct MHDFaceFluxes
{
    Grid F_rho;
    VecField F_rhoV;
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
