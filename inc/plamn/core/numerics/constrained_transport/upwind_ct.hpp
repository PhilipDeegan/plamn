#ifndef PLAMN_CORE_NUMERICS_CONSTRAINED_TRANSPORT_UPWIND_CT_HPP
#define PLAMN_CORE_NUMERICS_CONSTRAINED_TRANSPORT_UPWIND_CT_HPP

#include "damn/core/def.hpp"
#include "damn/core/def/damn_config.hpp"

#include <cstddef>

// upwind constrained transport: E on the cell edges from the faces' Riemann data (vt, aL, aR, dL,
//  dR of GodunovFaceResult) and the reconstructed B, PHARE's core/numerics/constrained_transport/
//  upwind_constrained_transport.hpp, ideal (Hall, resistive terms: not yet), its expressions term
//  for term (its results are the reference). One E component at one edge: where and how it is
//  evaluated (boxes, tiles, GPU) is up to the caller, see upwind_ct_transformer.hpp
namespace plamn::core
{

// one face direction's constrained transport fields, by reference
template<typename VecField, typename Field>
struct CTFace
{
    VecField const& vt;
    Field const &aL, &aR, &dL, &dR;
};

template<typename VecField, typename Field>
CTFace(VecField const&, Field const&, Field const&, Field const&, Field const&)
    -> CTFace<VecField, Field>;

template<typename Reconstruction>
struct UpwindCT
{
    using R = Reconstruction;

    // E component c at ijk; x, y, z: the CT fields of each face direction (only those of the
    //  directions < dim are read). In 1D only E_z has an equation (PHARE leaves E_x, E_y as
    //  they are), in 2D and 3D every component
    template<std::size_t c, std::size_t dim>
    static constexpr bool has_equation = dim > 1 or c == 2;

    template<std::size_t c, std::size_t dim>
    NO_DISCARD static double E(auto const& ijk, auto const& B, auto const& x, auto const& y,
                               auto const& z) DAMN_ALL
    {
        static_assert(has_equation<c, dim>);
        if constexpr (c == 0)
            return Ex_<dim>(ijk, B, y, z);
        else if constexpr (c == 1)
            return Ey_<dim>(ijk, B, x, z);
        else
            return Ez_<dim>(ijk, B, x, y);
    }

private:
    template<std::size_t dir>
    NO_DISCARD static auto prev(auto ijk) DAMN_ALL
    {
        ijk[dir] -= 1;
        return ijk;
    }

    // 0.5 * (f(ijk) + f(ijk - e_dir))
    template<std::size_t dir>
    NO_DISCARD static double avg(auto const& f, auto const& ijk) DAMN_ALL
    {
        return 0.5 * (f(ijk) + f(prev<dir>(ijk)));
    }

    template<std::size_t dim>
    NO_DISCARD static double Ex_(auto const& idx, auto const& B, auto const& y,
                                 auto const& z) DAMN_ALL
    {
        if constexpr (dim == 2)
        {
            auto const [BzL, BzR] = R::template reconstruct<1>(B[2], idx);
            auto const FL         = BzL * y.vt[1](idx) - B[1](idx) * y.vt[2](idx);
            auto const FR         = BzR * y.vt[1](idx) - B[1](idx) * y.vt[2](idx);
            return -(y.aL(idx) * FL + y.aR(idx) * FR) + (y.dR(idx) * BzR - y.dL(idx) * BzL);
        }
        else
        {
            auto const aS         = avg<2>(y.aL, idx);
            auto const aN         = avg<2>(y.aR, idx);
            auto const aB         = avg<1>(z.aL, idx);
            auto const aT         = avg<1>(z.aR, idx);
            auto const dS         = avg<2>(y.dL, idx);
            auto const dN         = avg<2>(y.dR, idx);
            auto const dB         = avg<1>(z.dL, idx);
            auto const dT         = avg<1>(z.dR, idx);
            auto const [vyS, vyN] = R::template reconstruct<1>(z.vt[1], idx);
            auto const [vzB, vzT] = R::template reconstruct<2>(y.vt[2], idx);
            auto const [BzS, BzN] = R::template reconstruct<1>(B[2], idx);
            auto const [ByB, ByT] = R::template reconstruct<2>(B[1], idx);
            return (aB * vzB * ByB + aT * vzT * ByT) - (aS * vyS * BzS + aN * vyN * BzN)
                   - (dT * ByT - dB * ByB) + (dN * BzN - dS * BzS);
        }
    }

    template<std::size_t dim>
    NO_DISCARD static double Ey_(auto const& idx, auto const& B, auto const& x,
                                 auto const& z) DAMN_ALL
    {
        if constexpr (dim <= 2)
        {
            auto const [BzL, BzR] = R::template reconstruct<0>(B[2], idx);
            auto const FL         = BzL * x.vt[0](idx) - B[0](idx) * x.vt[2](idx);
            auto const FR         = BzR * x.vt[0](idx) - B[0](idx) * x.vt[2](idx);
            return (x.aL(idx) * FL + x.aR(idx) * FR) - (x.dR(idx) * BzR - x.dL(idx) * BzL);
        }
        else
        {
            auto const aW         = avg<2>(x.aL, idx);
            auto const aE         = avg<2>(x.aR, idx);
            auto const aB         = avg<0>(z.aL, idx);
            auto const aT         = avg<0>(z.aR, idx);
            auto const dW         = avg<2>(x.dL, idx);
            auto const dE         = avg<2>(x.dR, idx);
            auto const dB         = avg<0>(z.dL, idx);
            auto const dT         = avg<0>(z.dR, idx);
            auto const [vxW, vxE] = R::template reconstruct<0>(z.vt[0], idx);
            auto const [vzB, vzT] = R::template reconstruct<2>(x.vt[2], idx);
            auto const [BzW, BzE] = R::template reconstruct<0>(B[2], idx);
            auto const [BxB, BxT] = R::template reconstruct<2>(B[0], idx);
            return (aW * vxW * BzW + aE * vxE * BzE) - (aB * vzB * BxB + aT * vzT * BxT)
                   - (dE * BzE - dW * BzW) + (dT * BxT - dB * BxB);
        }
    }

    template<std::size_t dim>
    NO_DISCARD static double Ez_(auto const& idx, auto const& B, auto const& x,
                                 auto const& y) DAMN_ALL
    {
        if constexpr (dim == 1)
        {
            auto const [ByL, ByR] = R::template reconstruct<0>(B[1], idx);
            auto const FL         = ByL * x.vt[0](idx) - B[0](idx) * x.vt[1](idx);
            auto const FR         = ByR * x.vt[0](idx) - B[0](idx) * x.vt[1](idx);
            return -(x.aL(idx) * FL + x.aR(idx) * FR) + (x.dR(idx) * ByR - x.dL(idx) * ByL);
        }
        else
        {
            auto const aW         = avg<1>(x.aL, idx);
            auto const aE         = avg<1>(x.aR, idx);
            auto const aS         = avg<0>(y.aL, idx);
            auto const aN         = avg<0>(y.aR, idx);
            auto const dW         = avg<1>(x.dL, idx);
            auto const dE         = avg<1>(x.dR, idx);
            auto const dS         = avg<0>(y.dL, idx);
            auto const dN         = avg<0>(y.dR, idx);
            auto const [vyS, vyN] = R::template reconstruct<1>(x.vt[1], idx);
            auto const [vxW, vxE] = R::template reconstruct<0>(y.vt[0], idx);
            auto const [BxS, BxN] = R::template reconstruct<1>(B[0], idx);
            auto const [ByW, ByE] = R::template reconstruct<0>(B[1], idx);
            return -(aW * vxW * ByW + aE * vxE * ByE) + (aS * vyS * BxS + aN * vyN * BxN)
                   + (dE * ByE - dW * ByW) - (dN * BxN - dS * BxS);
        }
    }
};

} // namespace plamn::core

#endif /* PLAMN_CORE_NUMERICS_CONSTRAINED_TRANSPORT_UPWIND_CT_HPP */
