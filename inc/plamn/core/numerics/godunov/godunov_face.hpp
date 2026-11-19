#ifndef PLAMN_CORE_NUMERICS_GODUNOV_GODUNOV_FACE_HPP
#define PLAMN_CORE_NUMERICS_GODUNOV_GODUNOV_FACE_HPP

#include "damn/core/def/damn_config.hpp"
#include "plamn/core/models/quantities/mhd_quantities.hpp"
#include "plamn/core/numerics/godunov/mhd_point_state.hpp"
#include "plamn/core/numerics/godunov/hll.hpp"
#include "plamn/core/numerics/godunov/rusanov.hpp"
#include "damn/core/def.hpp"

#include <array>
#include <cstddef>

// one face of direction dir: the left / right states reconstructed from the primitive fields
//  (rho, V, P cell centred; B face centred: its normal component is on the face already, the
//  transverse ones are projected to the cell centre then reconstructed), then the Riemann flux,
//  PHARE's reconstructor.hpp + Godunov::operator() for one face. Where and how it is evaluated
//  (boxes, tiles, GPU) is up to the caller, see godunov_transformer.hpp
namespace plamn::core
{

template<typename Reconstruction, typename Riemann = RusanovRiemann>
struct GodunovFace
{
    double gamma;

    template<std::size_t dir>
    NO_DISCARD GodunovFaceResult operator()(auto const& ijk, auto const& rho, auto const& V,
                                            auto const& B, auto const& P) const DAMN_ALL
    {
        using R                 = Reconstruction;
        auto const [rhoL, rhoR] = R::template reconstruct<dir>(rho, ijk);
        auto const [VxL, VxR]   = R::template reconstruct<dir>(V[0], ijk);
        auto const [VyL, VyR]   = R::template reconstruct<dir>(V[1], ijk);
        auto const [VzL, VzR]   = R::template reconstruct<dir>(V[2], ijk);
        auto const [PL, PR]     = R::template reconstruct<dir>(P, ijk);

        MHDPrimitive uL{rhoL, {VxL, VyL, VzL}, {}, PL};
        MHDPrimitive uR{rhoR, {VxR, VyR, VzR}, {}, PR};
        uL.B[dir] = uR.B[dir] = B[dir](ijk);
        transverse_<dir, (dir + 1) % 3>(B, ijk, uL, uR);
        transverse_<dir, (dir + 2) % 3>(B, ijk, uL, uR);

        return Riemann::template solve<dir>(gamma, uL, uR);
    }

    // the face's result into the direction's flux and constrained transport fields
    static void store(auto const& ijk, GodunovFaceResult const& r, auto& F_rho, auto& F_rhoV,
                      auto& F_B, auto& F_Etot, auto& vt, auto& aL, auto& aR, auto& dL,
                      auto& dR) DAMN_ALL
    {
        F_rho(ijk) = r.F.rho;
        for (std::size_t c = 0; c < 3; ++c)
        {
            F_rhoV[c](ijk) = r.F.rhoV[c];
            F_B[c](ijk)    = r.F.B[c];
            vt[c](ijk)     = r.vt[c];
        }
        F_Etot(ijk) = r.F.Etot;
        aL(ijk)     = r.uct_coefs[0];
        aR(ijk)     = r.uct_coefs[1];
        dL(ijk)     = r.uct_coefs[2];
        dR(ijk)     = r.uct_coefs[3];
    }

private:
    template<std::size_t dir, std::size_t t>
    static void transverse_(auto const& B, auto const& ijk, MHDPrimitive& uL,
                            MHDPrimitive& uR) DAMN_ALL
    {
        using enum MHDQuantity::Scalar;
        auto constexpr static face_qty = std::array{Bx, By, Bz}[t];
        auto const [l, r]
            = Reconstruction::template center_reconstruct<dir, face_qty, rho>(B[t], ijk);
        uL.B[t] = l;
        uR.B[t] = r;
    }
};

} // namespace plamn::core

#endif /* PLAMN_CORE_NUMERICS_GODUNOV_GODUNOV_FACE_HPP */
