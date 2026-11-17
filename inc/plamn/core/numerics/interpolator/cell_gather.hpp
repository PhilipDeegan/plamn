#ifndef PLAMN_CORE_NUMERICS_INTERPOLATOR_CELL_GATHER_HPP
#define PLAMN_CORE_NUMERICS_INTERPOLATOR_CELL_GATHER_HPP

#include "plamn/core/data/grid/patch_layout.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"
#include "plamn/core/models/quantities/quantity.hpp"

#include <array>
#include <cstddef>

#if defined(MKN_WITH_MKN_AVX)
#include "mkn/avx.hpp"
#endif

// m2p at interpolation order 1 for the particles of one cell, on lanes: the E and B values
//  every particle of the cell can need (per direction, a primal component's nodes cell and
//  cell + 1, a dual's cell - 1 to cell + 1: the stencils' union) loaded once per cell, then per
//  lane the weights of Interpolator's Weighter<1> and computeStartLeftShift, as dense rows over
//  the union (a dual direction's unused node weighted 0, first or last in its sum: the sums are
//  Interpolator::m2p's), summed as MeshToParticle<dim>::op: x outer, the last direction inner
//  A lane's particle must be in the cell (its delta in [0, 1) of it)
namespace plamn::core
{

template<std::size_t dim, std::size_t lanes>
struct CellGather1
{
    using Scalar                      = HybridQuantity::Scalar;
    static constexpr std::size_t span = 3; // nodes per direction of the union, primal uses 2
    static constexpr std::size_t size = dim == 1 ? 3 : dim == 2 ? 9 : 27;
    static constexpr std::array<Scalar, 6> qtys{Scalar::Ex, Scalar::Ey, Scalar::Ez,
                                                Scalar::Bx, Scalar::By, Scalar::Bz};

    template<std::size_t q>
    static constexpr auto cen = centering<dim>(qtys[q]);

    // the union's values of cell (AMR), the cell's local index for the weights
    void load(auto const& em, auto const& layout, auto const& cell)
    {
        auto const local = to_local(layout, cell);
        for (std::size_t k = 0; k < dim; ++k)
            lc[k] = static_cast<int>(local[k]);
        auto const& [Ex, Ey, Ez] = em.E();
        auto const& [Bx, By, Bz] = em.B();
        load_<0>(Ex);
        load_<1>(Ey);
        load_<2>(Ez);
        load_<3>(Bx);
        load_<4>(By);
        load_<5>(Bz);
    }

    // E, B of the lanes, from their deltas in the cell (all lanes: fixed trip counts)
    void gather(double const (&d)[dim][lanes], double (&E)[3][lanes], double (&B)[3][lanes]) const
    {
        // per direction: primal weights (nodes 0, 1 of the union from the cell), dual (0, 1, 2)
        double P[dim][2][lanes], D[dim][3][lanes];
        for (std::size_t k = 0; k < dim; ++k)
        {
            double const c = static_cast<double>(lc[k]);
            for (std::size_t l = 0; l < lanes; ++l)
            {
                double const p1 = (c + d[k][l]) - c; // Weighter<1>: pos - start
                P[k][1][l]      = p1;
                P[k][0][l]      = 1. - p1;

                // computeStartLeftShift<dual> as 1. or 0.: the selects as products by 1 or 0 (the
                //  same values, x * 0 + y == y), branch free: the loop vectorises
                double const sf = d[k][l] < .5 ? 1. : 0.;
                double const st = c - sf;
                double const d1 = ((c + d[k][l]) - .5) - st;
                double const d0 = 1. - d1;
                D[k][0][l]      = sf * d0;
                D[k][1][l]      = sf * d1 + (1. - sf) * d0;
                D[k][2][l]      = (1. - sf) * d1;
            }
        }
        value_<0>(P, D, E[0]);
        value_<1>(P, D, E[1]);
        value_<2>(P, D, E[2]);
        value_<3>(P, D, B[0]);
        value_<4>(P, D, B[1]);
        value_<5>(P, D, B[2]);
    }

    std::array<int, dim> lc;
    std::array<std::array<double, size>, 6> S; // [component][x][y][z] over the union, 3 a side

private:
    static constexpr std::size_t nodes(QtyCentering const c) { return c == QtyCentering::dual ? 3 : 2; }

    template<std::size_t q>
    void load_(auto const& field)
    {
        constexpr auto c = cen<q>;
        std::array<int, dim> base;
        for (std::size_t k = 0; k < dim; ++k)
            base[k] = lc[k] - (c[k] == QtyCentering::dual ? 1 : 0);
        auto& s = S[q];
        if constexpr (dim == 1)
            for (std::size_t i = 0; i < nodes(c[0]); ++i)
                s[i] = field(base[0] + i);
        else if constexpr (dim == 2)
            for (std::size_t i = 0; i < nodes(c[0]); ++i)
                for (std::size_t j = 0; j < nodes(c[1]); ++j)
                    s[i * span + j] = field(base[0] + i, base[1] + j);
        else
            for (std::size_t i = 0; i < nodes(c[0]); ++i)
                for (std::size_t j = 0; j < nodes(c[1]); ++j)
                    for (std::size_t k = 0; k < nodes(c[2]); ++k)
                        s[(i * span + j) * span + k] = field(base[0] + i, base[1] + j, base[2] + k);
    }

    // component q at the lanes: the direction's primal or dual weights over the union
    template<std::size_t q>
    void value_(double const (&P)[dim][2][lanes], double const (&D)[dim][3][lanes],
                double (&out)[lanes]) const
    {
        constexpr auto c = cen<q>;
        auto const& s    = S[q];
        auto const w = [&](std::size_t const k, std::size_t const i, std::size_t const l) {
            return c[k] == QtyCentering::dual ? D[k][i][l] : P[k][i][l];
        };
        // the lane loops innermost (vectorised), each lane's sums in MeshToParticle's order
        double x[lanes];
        for (std::size_t l = 0; l < lanes; ++l)
            x[l] = 0.;
        if constexpr (dim == 1)
            for (std::size_t i = 0; i < nodes(c[0]); ++i)
                for (std::size_t l = 0; l < lanes; ++l)
                    x[l] += s[i] * w(0, i, l);
        else if constexpr (dim == 2)
            for (std::size_t i = 0; i < nodes(c[0]); ++i)
            {
                double y[lanes];
                for (std::size_t l = 0; l < lanes; ++l)
                    y[l] = 0.;
                for (std::size_t j = 0; j < nodes(c[1]); ++j)
                    for (std::size_t l = 0; l < lanes; ++l)
                        y[l] += s[i * span + j] * w(1, j, l);
                for (std::size_t l = 0; l < lanes; ++l)
                    x[l] += y[l] * w(0, i, l);
            }
        else
            for (std::size_t i = 0; i < nodes(c[0]); ++i)
            {
                double y[lanes];
                for (std::size_t l = 0; l < lanes; ++l)
                    y[l] = 0.;
                for (std::size_t j = 0; j < nodes(c[1]); ++j)
                {
                    double z[lanes];
                    for (std::size_t l = 0; l < lanes; ++l)
                        z[l] = 0.;
                    for (std::size_t k = 0; k < nodes(c[2]); ++k)
                        for (std::size_t l = 0; l < lanes; ++l)
                            z[l] += s[(i * span + j) * span + k] * w(2, k, l);
                    for (std::size_t l = 0; l < lanes; ++l)
                        y[l] += z[l] * w(1, j, l);
                }
                for (std::size_t l = 0; l < lanes; ++l)
                    x[l] += y[l] * w(0, i, l);
            }
        for (std::size_t l = 0; l < lanes; ++l)
            out[l] = x[l];
    }
};

#if defined(MKN_WITH_MKN_AVX)

// CellGather1 on mkn::avx lanes (builds with -w mkn.avx): the same union and weights (the dual
//  ones by select, exact), the same sums per lane
template<std::size_t dim, std::size_t L>
struct CellGatherV : public CellGather1<dim, L>
{
    using Super = CellGather1<dim, L>;
    using V     = mkn::avx::Type<double, L>;
    using Super::lc;
    using Super::S;
    using Super::span;

    static V bc(double const x) { return V{mkn::avx::load<double, L>(x)}; }

    void gather(V const (&d)[dim], V (&E)[3], V (&B)[3]) const
    {
        V P[dim][2], D[dim][3];
        V const one = bc(1.), half = bc(.5), zero = bc(0.);
        for (std::size_t k = 0; k < dim; ++k)
        {
            V const c  = bc(static_cast<double>(lc[k]));
            V const p1 = (c + d[k]) - c;
            P[k][1]    = p1;
            P[k][0]    = one - p1;

            auto const s = mkn::avx::lt(d[k], half); // computeStartLeftShift<dual>
            V const st   = c - mkn::avx::select(s, one, zero);
            V const d1   = ((c + d[k]) - half) - st;
            V const d0   = one - d1;
            D[k][0]      = mkn::avx::select(s, d0, zero);
            D[k][1]      = mkn::avx::select(s, d1, d0);
            D[k][2]      = mkn::avx::select(s, zero, d1);
        }
        E[0] = value_<0>(P, D);
        E[1] = value_<1>(P, D);
        E[2] = value_<2>(P, D);
        B[0] = value_<3>(P, D);
        B[1] = value_<4>(P, D);
        B[2] = value_<5>(P, D);
    }

private:
    static constexpr std::size_t nodes(QtyCentering const c)
    {
        return c == QtyCentering::dual ? 3 : 2;
    }

    template<std::size_t q>
    V value_(V const (&P)[dim][2], V const (&D)[dim][3]) const
    {
        constexpr auto c = Super::template cen<q>;
        auto const& s    = S[q];
        auto const w     = [&](std::size_t const k, std::size_t const i) -> V const& {
            return c[k] == QtyCentering::dual ? D[k][i] : P[k][i];
        };
        V x = bc(0.);
        if constexpr (dim == 1)
            for (std::size_t i = 0; i < nodes(c[0]); ++i)
                x = x + bc(s[i]) * w(0, i);
        else if constexpr (dim == 2)
            for (std::size_t i = 0; i < nodes(c[0]); ++i)
            {
                V y = bc(0.);
                for (std::size_t j = 0; j < nodes(c[1]); ++j)
                    y = y + bc(s[i * span + j]) * w(1, j);
                x = x + y * w(0, i);
            }
        else
            for (std::size_t i = 0; i < nodes(c[0]); ++i)
            {
                V y = bc(0.);
                for (std::size_t j = 0; j < nodes(c[1]); ++j)
                {
                    V z = bc(0.);
                    for (std::size_t k = 0; k < nodes(c[2]); ++k)
                        z = z + bc(s[(i * span + j) * span + k]) * w(2, k);
                    y = y + z * w(1, j);
                }
                x = x + y * w(0, i);
            }
        return x;
    }
};

#endif // MKN_WITH_MKN_AVX

} // namespace plamn::core

#endif /* PLAMN_CORE_NUMERICS_INTERPOLATOR_CELL_GATHER_HPP */
