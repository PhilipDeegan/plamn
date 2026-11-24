#ifndef PLAMN_CORE_NUMERICS_CONSTRAINED_TRANSPORT_UPWIND_CT_HPP
#define PLAMN_CORE_NUMERICS_CONSTRAINED_TRANSPORT_UPWIND_CT_HPP

#include "damn/core/def.hpp"
#include "damn/core/def/damn_config.hpp"

#include <cstddef>

// upwind constrained transport: E on the cell edges from the faces' Riemann data (vt, aL, aR, dL,
//  dR of GodunovFaceResult) and the reconstructed B, PHARE's core/numerics/constrained_transport/
//  upwind_constrained_transport.hpp: ideal (E), Hall (E_hall: the ideal E, then its Hall term
//  added, from the faces' current and rho averages jt, rhot), its expressions term for term (its
//  results are the reference; the resistive E is added by the caller). One E component at one edge:
//  where and how it is evaluated (boxes, tiles, GPU) is up to the caller, see
//  upwind_ct_transformer.hpp
namespace plamn::core
{

// one face direction's constrained transport fields, by reference
template<typename VecField, typename Field>
struct CTFace
{
    VecField const& vt;
    Field const &aL, &aR, &dL, &dR;
    VecField const* jt = nullptr; // Hall: the face's current and rho averages
    Field const* rhot  = nullptr;
};

template<typename VecField, typename Field>
CTFace(VecField const&, Field const&, Field const&, Field const&, Field const&)
    -> CTFace<VecField, Field>;
template<typename VecField, typename Field>
CTFace(VecField const&, Field const&, Field const&, Field const&, Field const&, VecField const*,
       Field const*) -> CTFace<VecField, Field>;

template<typename Reconstruction>
struct UpwindCT
{
    using R = Reconstruction;

    // E component c at ijk; x, y, z: the CT fields of each face direction (only those of the
    //  directions < dim are read). In 1D E_y and E_z have an equation (PHARE's
    //  UpwindConstrainedTransport: E_x is left as it is, B_x does not change), in 2D and 3D
    //  every component
    template<std::size_t c, std::size_t dim>
    static constexpr bool has_equation = dim > 1 or c != 0;

    template<std::size_t c, std::size_t dim>
    MKN_KUL_FN_SIG_RET static double E(auto const& ijk, auto const& B, auto const& x, auto const& y,
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

    // Hall: E, then its Hall term added (PHARE's "E(idx) += ..."), the faces' jt and rhot read
    template<std::size_t c, std::size_t dim>
    MKN_KUL_FN_SIG_RET static double E_hall(auto const& ijk, auto const& B, auto const& x,
                                            auto const& y, auto const& z) DAMN_ALL
    {
        auto e = E<c, dim>(ijk, B, x, y, z);
        e += hall_<c, dim>(ijk, B, x, y, z);
        return e;
    }

private:
    template<std::size_t c, std::size_t dim>
    MKN_KUL_FN_SIG_RET static double hall_(auto const& idx, auto const& B, auto const& x,
                                           auto const& y, auto const& z) DAMN_ALL
    {
        if constexpr (c == 0) // dim 2, 3
        {
            if constexpr (dim == 2)
            {
                auto const [BzL, BzR] = R::template reconstruct<1>(B[2], idx);
                auto const& jt        = *y.jt;
                auto const invRho     = 1.0 / (*y.rhot)(idx);
                auto const JxB_x_L    = jt[1](idx) * BzL - jt[2](idx) * B[1](idx);
                auto const JxB_x_R    = jt[1](idx) * BzR - jt[2](idx) * B[1](idx);
                auto const HallL      = -JxB_x_L * invRho;
                auto const HallR      = -JxB_x_R * invRho;
                auto const F_Bz_y     = y.aL(idx) * HallL + y.aR(idx) * HallR;
                return -F_Bz_y;
            }
            else
            {
                auto const aS           = avg<2>(y.aL, idx);
                auto const aN           = avg<2>(y.aR, idx);
                auto const aB           = avg<1>(z.aL, idx);
                auto const aT           = avg<1>(z.aR, idx);
                auto const [BzS, BzN]   = R::template reconstruct<1>(B[2], idx);
                auto const [ByB, ByT]   = R::template reconstruct<2>(B[1], idx);
                auto const [jyS, jyN]   = R::template reconstruct<1>((*z.jt)[1], idx);
                auto const [jzB, jzT]   = R::template reconstruct<2>((*y.jt)[2], idx);
                auto const [rhoS, rhoN] = R::template reconstruct<1>(*z.rhot, idx);
                auto const [rhoB, rhoT] = R::template reconstruct<2>(*y.rhot, idx);
                return -(aB * jzB * ByB / rhoB + aT * jzT * ByT / rhoT)
                       + (aS * jyS * BzS / rhoS + aN * jyN * BzN / rhoN);
            }
        }
        else if constexpr (c == 1)
        {
            if constexpr (dim <= 2)
            {
                auto const [BzL, BzR] = R::template reconstruct<0>(B[2], idx);
                auto const& jt        = *x.jt;
                auto const invRho     = 1.0 / (*x.rhot)(idx);
                auto const JxB_y_L    = jt[2](idx) * B[0](idx) - jt[0](idx) * BzL;
                auto const JxB_y_R    = jt[2](idx) * B[0](idx) - jt[0](idx) * BzR;
                auto const HallL      = JxB_y_L * invRho;
                auto const HallR      = JxB_y_R * invRho;
                return x.aL(idx) * HallL + x.aR(idx) * HallR; // F_Bz_x
            }
            else
            {
                auto const aW           = avg<2>(x.aL, idx);
                auto const aE           = avg<2>(x.aR, idx);
                auto const aB           = avg<0>(z.aL, idx);
                auto const aT           = avg<0>(z.aR, idx);
                auto const [BzW, BzE]   = R::template reconstruct<0>(B[2], idx);
                auto const [BxB, BxT]   = R::template reconstruct<2>(B[0], idx);
                auto const [jxW, jxE]   = R::template reconstruct<0>((*z.jt)[0], idx);
                auto const [jzB, jzT]   = R::template reconstruct<2>((*x.jt)[2], idx);
                auto const [rhoW, rhoE] = R::template reconstruct<0>(*z.rhot, idx);
                auto const [rhoB, rhoT] = R::template reconstruct<2>(*x.rhot, idx);
                return -(aW * jxW * BzW / rhoW + aE * jxE * BzE / rhoE)
                       + (aB * jzB * BxB / rhoB + aT * jzT * BxT / rhoT);
            }
        }
        else
        {
            if constexpr (dim == 1)
            {
                auto const [ByL, ByR] = R::template reconstruct<0>(B[1], idx);
                auto const& jt        = *x.jt;
                auto const invRho     = 1.0 / (*x.rhot)(idx);
                auto const JxB_z_L    = jt[0](idx) * ByL - jt[1](idx) * B[0](idx);
                auto const JxB_z_R    = jt[0](idx) * ByR - jt[1](idx) * B[0](idx);
                auto const HallL      = -JxB_z_L * invRho;
                auto const HallR      = -JxB_z_R * invRho;
                auto const F_By_x     = x.aL(idx) * HallL + x.aR(idx) * HallR;
                return -F_By_x;
            }
            else
            {
                auto const aW           = avg<1>(x.aL, idx);
                auto const aE           = avg<1>(x.aR, idx);
                auto const aS           = avg<0>(y.aL, idx);
                auto const aN           = avg<0>(y.aR, idx);
                auto const [BxS, BxN]   = R::template reconstruct<1>(B[0], idx);
                auto const [ByW, ByE]   = R::template reconstruct<0>(B[1], idx);
                auto const [jyS, jyN]   = R::template reconstruct<1>((*x.jt)[1], idx);
                auto const [jxW, jxE]   = R::template reconstruct<0>((*y.jt)[0], idx);
                auto const [rhoS, rhoN] = R::template reconstruct<1>(*x.rhot, idx);
                auto const [rhoW, rhoE] = R::template reconstruct<0>(*y.rhot, idx);
                return (aW * jxW * ByW / rhoW + aE * jxE * ByE / rhoE)
                       - (aS * jyS * BxS / rhoS + aN * jyN * BxN / rhoN);
            }
        }
    }

    template<std::size_t dir>
    MKN_KUL_FN_SIG_RET static auto prev(auto ijk) DAMN_ALL
    {
        ijk[dir] -= 1;
        return ijk;
    }

    // 0.5 * (f(ijk) + f(ijk - e_dir))
    template<std::size_t dir>
    MKN_KUL_FN_SIG_RET static double avg(auto const& f, auto const& ijk) DAMN_ALL
    {
        return 0.5 * (f(ijk) + f(prev<dir>(ijk)));
    }

    template<std::size_t dim>
    MKN_KUL_FN_SIG_RET static double Ex_(auto const& idx, auto const& B, auto const& y,
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
    MKN_KUL_FN_SIG_RET static double Ey_(auto const& idx, auto const& B, auto const& x,
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
    MKN_KUL_FN_SIG_RET static double Ez_(auto const& idx, auto const& B, auto const& x,
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
