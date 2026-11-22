#ifndef PLAMN_CORE_NUMERICS_INTERPOLATOR_CELL_GATHER_HPP
#define PLAMN_CORE_NUMERICS_INTERPOLATOR_CELL_GATHER_HPP

#include "plamn/core/data/grid/patch_layout.hpp"
#include "plamn/core/data/particles/particles_def.hpp" // IWYU pragma: keep (is_soavx)
#include "plamn/core/models/quantities/hybrid_quantities.hpp"
#include "plamn/core/models/quantities/quantity.hpp"

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#if defined(MKN_WITH_MKN_AVX) // the lane kernels below
#include "mkn/avx.hpp"
#include "plamn/core/data/particles/aos_transpose.hpp"

#include <algorithm>
#include <type_traits>
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

    // the union's values of cell (AMR), from the cell's local index
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
            for (std::size_t l = 0; l < lanes; ++l)
            {
                double const p1 = d[k][l]; // Weighter<1>: the position relative to the start
                P[k][1][l]      = p1;
                P[k][0][l]      = 1. - p1;

                // computeStartLeftShift<dual> as 1. or 0.: the selects as products by 1 or 0 (the
                //  same values, x * 0 + y == y), branch free: the loop vectorises
                double const sf = d[k][l] < .5 ? 1. : 0.;
                double const d1 = d[k][l] + (sf - .5); // as indexAndWeights__: sf - .5 is exact
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
    static constexpr std::size_t nodes(QtyCentering const c)
    {
        return c == QtyCentering::dual ? 3 : 2;
    }

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

    // fn(i) for i in [0, n), unrolled at compile time
    template<std::size_t n>
    static void each(auto&& fn)
    {
        [&]<std::size_t... i>(std::index_sequence<i...>) {
            (fn(std::integral_constant<std::size_t, i>{}), ...);
        }(std::make_index_sequence<n>{});
    }

    // component q at the lanes: the direction's primal or dual weights over the union. The lane
    //  loop outermost, the nodes' unrolled in it (compile time trip counts): it vectorises over
    //  the lanes (an inner lane loop left gcc vectorising the 2 or 3 nodes, 2 wide). Each lane
    //  sums in MeshToParticle's order: x outer, the last direction inner
    template<std::size_t q>
    void value_(double const (&P)[dim][2][lanes], double const (&D)[dim][3][lanes],
                double (&out)[lanes]) const
    {
        constexpr auto c = cen<q>;
        auto const& s    = S[q];
        auto const w     = [&](std::size_t const k, std::size_t const i, std::size_t const l) {
            return c[k] == QtyCentering::dual ? D[k][i][l] : P[k][i][l];
        };
        for (std::size_t l = 0; l < lanes; ++l)
        {
            double x = 0.;
            if constexpr (dim == 1)
                each<nodes(c[0])>([&](auto i) { x += s[i] * w(0, i, l); });
            else if constexpr (dim == 2)
                each<nodes(c[0])>([&](auto i) {
                    double y = 0.;
                    each<nodes(c[1])>([&](auto j) { y += s[i * span + j] * w(1, j, l); });
                    x += y * w(0, i, l);
                });
            else
                each<nodes(c[0])>([&](auto i) {
                    double y = 0.;
                    each<nodes(c[1])>([&](auto j) {
                        double z = 0.;
                        each<nodes(c[2])>(
                            [&](auto k) { z += s[(i * span + j) * span + k] * w(2, k, l); });
                        y += z * w(1, j, l);
                    });
                    x += y * w(0, i, l);
                });
            out[l] = x;
        }
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
            P[k][1] = d[k];
            P[k][0] = one - d[k];

            auto const s = mkn::avx::lt(d[k], half); // computeStartLeftShift<dual>
            V const d1   = d[k] + (mkn::avx::select(s, one, zero) - half);
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

// the deposit of a cell's particles at interpolation order 1, on lanes: per node of the cell's
//  primal stencil (the cell and + 1 per direction) density and flux accumulate on lanes, summed
//  over the lanes and added to the moments once (flush). The weights from the deltas alone,
//  as Interpolator::particleToMesh (Weighter<1>, primal), its products in its
//  order (weight, x, y, z); the sums per node differ from its particle by particle order
template<std::size_t dim, std::size_t L>
struct CellDepositV
{
    using V                            = mkn::avx::Type<double, L>;
    using M                            = mkn::avx::Mask<double, L>;
    static constexpr std::size_t nodes = std::size_t{1} << dim;

    static V bc(double const x) { return V{mkn::avx::load<double, L>(x)}; }

    // the cell (AMR) and the moments' ghost grown box (Interpolator::particleToMesh's)
    void reset(auto const& cell, auto const& moments_box)
    {
        for (std::size_t n = 0; n < nodes; ++n)
        {
            rho[n] = bc(0.);
            for (std::size_t k = 0; k < 3; ++k)
                f[k][n] = bc(0.);
        }
        for (std::size_t k = 0; k < dim; ++k)
            lc[k] = cell[k] - moments_box.lower[k];
        touched = false;
    }

    // the lanes of m: their weight w, deltas d in the cell, velocities v
    void add(M const& m, V const& w, V const (&d)[dim], V const (&v)[3])
    {
        V const wt = mkn::avx::select(m, w, bc(0.));
        V P[dim][2];
        for (std::size_t k = 0; k < dim; ++k)
        {
            P[k][1] = d[k];
            P[k][0] = bc(1.) - d[k];
        }
        for (std::size_t n = 0; n < nodes; ++n)
        {
            V x = wt * P[0][(n >> (dim - 1)) & 1];
            for (std::size_t k = 1; k < dim; ++k)
                x = x * P[k][(n >> (dim - 1 - k)) & 1];
            rho[n] += x;
            for (std::size_t k = 0; k < 3; ++k)
                f[k][n] += x * v[k];
        }
        touched = true;
    }

    void flush(auto& density, auto& flux) const
    {
        if (!touched)
            return;
        auto& [fx, fy, fz] = flux();
        for (std::size_t n = 0; n < nodes; ++n)
        {
            auto idx = lc;
            for (std::size_t k = 0; k < dim; ++k)
                idx[k] += (n >> (dim - 1 - k)) & 1;
            at_(density, idx) += hsum_(rho[n]);
            at_(fx, idx) += hsum_(f[0][n]);
            at_(fy, idx) += hsum_(f[1][n]);
            at_(fz, idx) += hsum_(f[2][n]);
        }
    }

    V rho[nodes], f[3][nodes];
    std::array<int, dim> lc;
    bool touched = false;

private:
    static double hsum_(V const& x)
    {
        double t[L];
        mkn::avx::unaligned_store(t, x);
        double sum = 0;
        for (std::size_t l = 0; l < L; ++l)
            sum += t[l];
        return sum;
    }
    static auto& at_(auto& field, std::array<int, dim> const& i)
    {
        if constexpr (dim == 1)
            return field(i[0]);
        else if constexpr (dim == 2)
            return field(i[0], i[1]);
        else
            return field(i[0], i[1], i[2]);
    }
};

// the deposit of a sorted SoAVXChunked array at order 1: cell by cell, its chunk in blocks of
//  L slots (masked loads: past the chunk's end weight 0), every particle in its chunk's cell
template<std::size_t L, typename Particles_t>
    requires(is_soavx(Particles_t::layout_mode))
void cell_deposit_sorted(Particles_t const& particles, auto& density, auto& flux,
                         auto const& layout)
{
    using Layout_t                   = std::decay_t<decltype(layout)>;
    static constexpr std::size_t dim = Particles_t::dimension;
    using V                          = mkn::avx::Type<double, L>;
    auto const moments_box = damn::core::grow(layout.cells, Layout_t::options.field_ghost_width);

    auto const& W  = particles.weight();
    auto const& D  = particles.delta();
    auto const& Vc = particles.v();
    double idx[L];
    for (std::size_t l = 0; l < L; ++l)
        idx[l] = static_cast<double>(l);
    V const lane = mkn::avx::unaligned_load<double, L>(idx);

    CellDepositV<dim, L> win;
    damn::core::for_each_point(particles.shell(), [&](auto const& cell) {
        auto const [b, size] = particles.cell_range(cell);
        if (size == 0)
            return;
        win.reset(cell, moments_box);
        for (std::size_t i = b; i < b + size; i += L)
        {
            auto const n     = std::min(L, b + size - i);
            auto const first = mkn::avx::lt(lane, CellDepositV<dim, L>::bc(static_cast<double>(n)));
            V const w        = mkn::avx::mask_load(first, W.data() + i);
            V d[dim], v[3];
            for (std::size_t k = 0; k < dim; ++k)
                d[k] = mkn::avx::mask_load(first, D[k].data() + i);
            for (std::size_t k = 0; k < 3; ++k)
                v[k] = mkn::avx::mask_load(first, Vc[k].data() + i);
            win.add(mkn::avx::gt(w, CellDepositV<dim, L>::bc(0.)), w, d, v);
        }
        win.flush(density, flux);
    });
}

// the same of a sorted AoS4SoAVXChunked array: a block of a chunk's particle structures staged
//  into lanes (past the chunk's end weight 0: not live)
template<std::size_t L, typename Particles_t>
    requires(!is_soavx(Particles_t::layout_mode))
void cell_deposit_sorted(Particles_t const& particles, auto& density, auto& flux,
                         auto const& layout)
{
    using Layout_t                   = std::decay_t<decltype(layout)>;
    static constexpr std::size_t dim = Particles_t::dimension;
    using V                          = mkn::avx::Type<double, L>;
    auto const moments_box = damn::core::grow(layout.cells, Layout_t::options.field_ghost_width);

    CellDepositV<dim, L> win;
    damn::core::for_each_point(particles.shell(), [&](auto const& cell) {
        auto const [b, size] = particles.cell_range(cell);
        if (size == 0)
            return;
        win.reset(cell, moments_box);
        for (std::size_t i = b; i < b + size; i += L)
        {
            auto const n = std::min(L, b + size - i);
            alignas(64) double ws[L], ds[dim][L], vs[3][L];
            static_assert(L == 8, "a block is one transpose");
            aos_transpose::Cols<dim> cols{ws, {}, {}, {vs[0], vs[1], vs[2]}}; // no cells
            for (std::size_t k = 0; k < dim; ++k)
                cols.d[k] = ds[k];
            aos_transpose::to_columns<dim>(particles.data() + i, n, cols); // past n: weight 0
            V const w = mkn::avx::unaligned_load<double, L>(ws);
            V d[dim], v[3];
            for (std::size_t k = 0; k < dim; ++k)
                d[k] = mkn::avx::unaligned_load<double, L>(ds[k]);
            for (std::size_t k = 0; k < 3; ++k)
                v[k] = mkn::avx::unaligned_load<double, L>(vs[k]);
            win.add(mkn::avx::gt(w, CellDepositV<dim, L>::bc(0.)), w, d, v);
        }
        win.flush(density, flux);
    });
}

#endif // MKN_WITH_MKN_AVX

} // namespace plamn::core

#endif /* PLAMN_CORE_NUMERICS_INTERPOLATOR_CELL_GATHER_HPP */
