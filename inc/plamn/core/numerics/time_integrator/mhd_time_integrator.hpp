#ifndef PLAMN_CORE_NUMERICS_TIME_INTEGRATOR_MHD_TIME_INTEGRATOR_HPP
#define PLAMN_CORE_NUMERICS_TIME_INTEGRATOR_MHD_TIME_INTEGRATOR_HPP

#include "mkn/kul/for.hpp"
#include "plamn/core/data/field/field_tiles.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"
#include "plamn/core/models/quantities/mhd_quantities.hpp"
#include "plamn/core/numerics/ampere/ampere_transformer.hpp"
#include "plamn/core/numerics/constrained_transport/upwind_ct_transformer.hpp"
#include "plamn/core/numerics/faraday/faraday_transformer.hpp"
#include "plamn/core/numerics/finite_volume/finite_volume_euler_transformer.hpp"
#include "plamn/core/numerics/godunov/godunov_transformer.hpp"
#include "plamn/core/numerics/differential_operators.hpp"
#include "plamn/core/numerics/ohm/ohm.hpp"
#include "plamn/core/numerics/primitive_conservative/primitive_conservative_transformer.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/point.hpp"
#include "plamn/core/numerics/godunov/hll.hpp"
#include "plamn/core/numerics/constrained_transport/upwind_ct.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <concepts>
#include <iterator>

// MHD time integration on one level in PHARE's operation order, so results can match its own
//  bit for bit in the same floating point context: doc/core/numerics/time_integrator/
//  mhd_time_integrator.md
namespace plamn::core
{

namespace detail
{
    // fn(layout, views...) once per tile of tiled fields, or once with the patch layout
    void on_views(auto const& layout, auto&& fn, auto& field0, auto&... fields)
    {
        if constexpr (is_field_tile_set_v<std::remove_cvref_t<decltype(field0)>>)
            tile_exec_with_layout(
                [&](auto const& tile_layout, auto&&... views) { fn(tile_layout, views...); },
                field0, fields...);
        else
            fn(layout, field0, fields...);
    }

    // every point of the arrays, ghosts included: tiles one by one
    void on_arrays(auto&& fn, auto& field0, auto&... fields)
    {
        auto const one = [&](auto&& f0, auto&&... fs) {
            auto const n = f0.size();
            fn(n, f0.data(), fs.data()...);
        };
        if constexpr (is_field_tile_set_v<std::remove_cvref_t<decltype(field0)>>)
            for (std::size_t ti = 0; ti < tile_count(field0); ++ti)
                one(tile_at(field0, ti), tile_at(fields, ti)...);
        else
            one(field0, fields...);
    }

    // fn(field) for each (scalar) field of an MHDFaceFluxes
    void for_each_flux_field(auto&& fn, auto& f0, auto&... f)
    {
        fn(f0.F_rho, f.F_rho...);
        for (std::size_t c = 0; c < 3; ++c)
            fn(f0.F_rhoV[c], f.F_rhoV[c]...);
        fn(f0.F_Etot, f.F_Etot...);
    }

    // fn(field) for each evolved field of an MHDState, PHARE's RKUtils order
    void for_each_state_field(auto&& fn, auto& s0, auto&... s)
    {
        fn(s0.rho, s.rho...);
        for (std::size_t c = 0; c < 3; ++c)
            fn(s0.rhoV[c], s.rhoV[c]...);
        for (std::size_t c = 0; c < 3; ++c)
            fn(s0.B[c], s.B[c]...);
        fn(s0.Etot, s.Etot...);
    }
} // namespace detail


// the steps the integrators are made of (PHARE's ComputeFluxes, EulerUsingComputedFlux,
//  BaseMHDTimestepper's butcher sums, RKUtils), each over the level
template<typename Reconstruction, typename Riemann = RusanovRiemann>
class MHDSteps
{
public:
    // dissipation: the resistivity eta, the hyper resistivity nu and its mode (0, 0: ideal); with
    //  either, J and the dissipative E (eta J - nu' laplacian J) are the patch's (J, E_diss), as
    //  PHARE's ComputeFluxes (DissipativeElectricField)
    explicit MHDSteps(double const gamma,
                      OhmInfo const& dissipation = {0., 0., HyperMode::constant})
        : gamma_{gamma}
        , diss_{dissipation}
    {
    }

    // rhoV and Etot from V and P again, as PHARE; Hall: J every time (PHARE's needsCurrent)
    void compute_fluxes(auto& level, std::size_t const s, double const t) const
    {
        level.per_patch([&](auto& p) {
            auto constexpr static dim  = std::decay_t<decltype(p.layout)>::dimension;
            auto constexpr static hall = std::decay_t<decltype(p)>::hall;
            auto& u                    = p.states[s];
            PrimitiveConservativeTransformer const pc{gamma_};
            GodunovTransformer<Reconstruction, Riemann> const godunov{gamma_};

            pc.to_primitive(p.layout, u.rho, u.rhoV, u.B, u.Etot, u.V, u.P);
            if (hall or dissipative_()) // J = curl B
                AmpereTransformer{}(p.layout, u.B, p.J);
            if (dissipative_())
                dissipative_E_(p, u);
            mkn::kul::for_N<dim>([&](auto d) {
                auto& f = p.fluxes[d];
                auto& c = p.ct[d];
                if constexpr (hall)
                {
                    auto& h = p.hall_ct[d];
                    godunov.template hall<d>(p.layout, u.rho, u.V, u.B, u.P, p.J, f.F_rho, f.F_rhoV,
                                             f.F_Etot, c.vt, c.aL, c.aR, c.dL, c.dR, h.jt, h.rhot);
                }
                else
                    godunov.template operator()<d>(p.layout, u.rho, u.V, u.B, u.P, f.F_rho,
                                                   f.F_rhoV, f.F_Etot, c.vt, c.aL, c.aR, c.dL,
                                                   c.dR);
                if (dissipative_())
                    dissipative_fluxes_<d>(p, u.B);
            });
            pc.to_conservative(p.layout, u.rho, u.V, u.B, u.P, u.rhoV, u.Etot);
            auto& [x, y, z] = p.ct;
            if constexpr (hall)
            {
                auto const face = [&](auto const& c, auto const& h) { // Grid overloads operator&
                    return CTFace{
                        c.vt, c.aL, c.aR, c.dL, c.dR, std::addressof(h.jt), std::addressof(h.rhot)};
                };
                auto& [hx, hy, hz] = p.hall_ct;
                UpwindCTTransformer<Reconstruction>{}.hall(p.layout, u.B, face(x, hx), face(y, hy),
                                                           face(z, hz), p.E);
            }
            else
                UpwindCTTransformer<Reconstruction>{}(p.layout, u.B, x.vt, x.aL, x.aR, x.dL, x.dR,
                                                      y.vt, y.aL, y.aR, y.dL, y.dR, z.vt, z.aL,
                                                      z.aR, z.dL, z.dR, p.E);
            if (dissipative_()) // E += the dissipative E, every component on its domain box
                for (std::size_t c = 0; c < 3; ++c)
                    damn::core::for_each_point(domain_box(p.layout, p.E[c]), [&](auto const& ijk) {
                        p.E[c](ijk) += p.E_diss[c](ijk);
                    });
        });
        level.fill_electric_ghosts(t);
    }

    // states[so] = states[si] - dt div(fluxes), B by Faraday with E; then states[so]'s ghosts
    //  E_of(patch), fluxes_of(patch): the E and fluxes used (the last computed, or the butcher
    //  sums). so may be si
    void euler_using(auto& level, std::size_t const si, std::size_t const so, auto&& E_of,
                     auto&& fluxes_of, double const t, double const dt) const
    {
        level.per_patch([&](auto& p) {
            auto const& u = p.states[si];
            auto& un      = p.states[so];
            auto const& F = fluxes_of(p);
            FiniteVolumeEulerTransformer const fve{};

            fve(p.layout, dt, u.rho, F[0].F_rho, F[1].F_rho, F[2].F_rho, un.rho);
            for (std::size_t c = 0; c < 3; ++c)
                fve(p.layout, dt, u.rhoV[c], F[0].F_rhoV[c], F[1].F_rhoV[c], F[2].F_rhoV[c],
                    un.rhoV[c]);
            fve(p.layout, dt, u.Etot, F[0].F_Etot, F[1].F_Etot, F[2].F_Etot, un.Etot);
            FaradayTransformer{}(p.layout, u.B, E_of(p), un.B, dt);
        });
        level.fill_state_ghosts(so, t);
    }

    // compute_fluxes from states[si], then states[so] from them: PHARE's Euler
    void euler(auto& level, std::size_t const si, std::size_t const so, double const t,
               double const dt) const
    {
        compute_fluxes(level, si, t);
        euler_using(level, si, so, last_E, last_fluxes, t, dt);
    }

    // butcher = coef * (the last fluxes, E), every point: a step's first sum (PHARE's reset to 0
    //  then PlusEqualsProduct: 0 + x == x, the same bits but for the sign of a zero)
    void assign_butcher(auto& level, double const coef) const { butcher_<false>(level, coef); }

    // butcher += coef * (the last fluxes, E), every point (PHARE's PlusEqualsProduct)
    void accumulate_butcher(auto& level, double const coef) const { butcher_<true>(level, coef); }

    // states[dst] = sum_j w_j states[s_j] on the domain (PHARE's RKUtils), no ghosts
    void rk_blend(auto& level, std::size_t const dst,
                  std::same_as<std::pair<double, std::size_t>> auto const... ws) const
    {
        level.per_patch([&](auto& p) {
            detail::for_each_state_field(
                [&](auto& res, auto const&... src) {
                    detail::on_views(
                        p.layout,
                        [&](auto const& layout, auto& r, auto const&... s) {
                            damn::core::for_each_point(domain_box(layout, r), [&](auto const& ijk) {
                                auto sum = 0.0;
                                ((sum += ws.first * s(ijk)), ...);
                                r(ijk) = sum;
                            });
                        },
                        res, src...);
                },
                p.states[dst], p.states[ws.second]...);
        });
    }

private:
    template<bool accumulate>
    void butcher_(auto& level, double const coef) const
    {
        level.per_patch([&](auto& p) {
            auto const axpy = [&](auto& d, auto const& s) {
                detail::on_arrays(
                    [&](auto const n, auto* dp, auto const* sp) {
                        for (std::size_t i = 0; i < n; ++i)
                            if constexpr (accumulate)
                                dp[i] += sp[i] * coef;
                            else
                                dp[i] = sp[i] * coef;
                    },
                    d, s);
            };
            for (std::size_t d = 0; d < 3; ++d)
                detail::for_each_flux_field(axpy, p.butcher[d], p.fluxes[d]);
            for (std::size_t c = 0; c < 3; ++c)
                axpy(p.butcherE[c], p.E[c]);
        });
    }

public:
    // which E and fluxes euler_using takes
    static constexpr auto last_E         = [](auto& p) -> auto& { return p.E; };
    static constexpr auto last_fluxes    = [](auto& p) -> auto& { return p.fluxes; };
    static constexpr auto butcher_E      = [](auto& p) -> auto& { return p.butcherE; };
    static constexpr auto butcher_fluxes = [](auto& p) -> auto& { return p.butcher; };

private:
    double gamma_;
    OhmInfo diss_;

    bool dissipative_() const { return diss_.isResistive() or diss_.isHyperResistive(); }

    // fn.template operator()<resistive, hyper, mode>(): the dissipation flags as compile time tags
    //  (no branch on them per point), the mode only with hyper resistivity
    void with_dissipation_(auto&& fn) const
    {
        auto const hyper = [&]<bool resistive>() {
            if (not diss_.isHyperResistive())
                return fn.template operator()<resistive, false, HyperMode::constant>();
            if (diss_.hyper_mode == HyperMode::constant)
                return fn.template operator()<resistive, true, HyperMode::constant>();
            return fn.template operator()<resistive, true, HyperMode::spatial>();
        };
        if (diss_.isResistive())
            return hyper.template operator()<true>();
        return hyper.template operator()<false>();
    }

    // the dissipative E (PHARE's DissipativeElectricField) on each E component's domain box:
    //  e = eta J - coef laplacian(J), coef nu (constant) or nu dx^2 (|B| / rho + 1) (spatial: dx
    //  the smallest mesh size, B and rho projected onto the E component), in PHARE's order
    void dissipative_E_(auto& p, auto const& u) const
    {
        using S       = MHDQuantity::Scalar;
        using Layout  = std::decay_t<decltype(p.layout)>;
        auto const dx = *std::min_element(p.layout.dx.begin(), p.layout.dx.end());
        with_dissipation_([&]<bool resistive, bool hyper, HyperMode mode>() {
            mkn::kul::for_N<3>([&](auto ci) {
                auto constexpr static c  = ci();
                auto constexpr static Ec = c == 0 ? S::Ex : c == 1 ? S::Ey : S::Ez;
                auto& E                  = p.E_diss[c];
                auto const& J            = p.J[c];
                auto const coef          = [&](auto const& i) {
                    if constexpr (mode == HyperMode::constant)
                        return diss_.nu;
                    else
                    {
                        auto const bx = project<S::Bx, Ec>(u.B[0], i);
                        auto const by = project<S::By, Ec>(u.B[1], i);
                        auto const bz = project<S::Bz, Ec>(u.B[2], i);
                        auto const n  = project<S::rho, Ec>(u.rho, i);
                        auto const b  = std::sqrt(bx * bx + by * by + bz * bz);
                        return diss_.nu * dx * dx * (b / n + 1);
                    }
                };
                damn::core::for_each_point(domain_box(p.layout, E), [&](auto const& ijk) {
                    auto const i = index_<Layout::dimension>(ijk);
                    double e     = 0.;
                    if constexpr (resistive)
                        e += diss_.eta * J(ijk);
                    if constexpr (hyper)
                        e -= coef(i) * laplacian(p.layout, J, i);
                    E(ijk) = e;
                });
            });
        });
    }

    // a local point of a box as project's index
    template<std::size_t dim>
    static auto index_(auto const& ijk)
    {
        damn::core::Point<std::size_t, dim> index;
        for (std::size_t d = 0; d < dim; ++d)
            index[d] = static_cast<std::size_t>(ijk[d]);
        return index;
    }

    // the dissipative E's work on the face fluxes of direction d (PHARE's MHDEquations
    //  dissipative_contributions over the faces' domain box): F_Etot gets E x B along the face's
    //  normal, projected onto the face (B is evolved by Faraday: no F_B)
    template<std::size_t d>
    static void dissipative_fluxes_(auto& p, auto const& B)
    {
        using S       = MHDQuantity::Scalar;
        using Layout  = std::decay_t<decltype(p.layout)>;
        auto const& E = p.E_diss;
        auto& f       = p.fluxes[d];
        damn::core::for_each_point(domain_box(p.layout, f.F_rho), [&](auto const& ijk) {
            auto const i = index_<Layout::dimension>(ijk);
            if constexpr (d == 0)
            {
                auto const ey = project<S::Ey, S::Bx>(E[1], i), ez = project<S::Ez, S::Bx>(E[2], i);
                auto const by = project<S::By, S::Bx>(B[1], i), bz = project<S::Bz, S::Bx>(B[2], i);
                f.F_Etot(ijk) += ey * bz - ez * by;
            }
            else if constexpr (d == 1)
            {
                auto const ex = project<S::Ex, S::By>(E[0], i), ez = project<S::Ez, S::By>(E[2], i);
                auto const bx = project<S::Bx, S::By>(B[0], i), bz = project<S::Bz, S::By>(B[2], i);
                f.F_Etot(ijk) += ez * bx - ex * bz;
            }
            else
            {
                auto const ex = project<S::Ex, S::Bz>(E[0], i), ey = project<S::Ey, S::Bz>(E[1], i);
                auto const bx = project<S::Bx, S::Bz>(B[0], i), by = project<S::By, S::Bz>(B[1], i);
                f.F_Etot(ijk) += ex * by - ey * bx;
            }
        });
    }
};


// the integrators: operator()(level, t0, t1) advances every patch's states[0] from t0 to t1
template<typename Reconstruction, typename Riemann = RusanovRiemann>
struct EulerIntegrator
{
    static constexpr std::size_t n_states = 1;
    MHDSteps<Reconstruction, Riemann> steps;

    void operator()(auto& level, double const t0, double const t1) const
    {
        steps.euler(level, 0, 0, t1, t1 - t0);
        steps.assign_butcher(level, 1.0);
    }
};

template<typename Reconstruction, typename Riemann = RusanovRiemann>
struct TVDRK2Integrator
{
    static constexpr std::size_t n_states = 2;
    static constexpr auto w0              = 0.5;
    static constexpr auto w1              = 0.5;
    MHDSteps<Reconstruction, Riemann> steps;

    void operator()(auto& level, double const t0, double const t1) const
    {
        using S = MHDSteps<Reconstruction, Riemann>;
        steps.euler(level, 0, 1, t1, t1 - t0);
        steps.assign_butcher(level, w0);
        steps.compute_fluxes(level, 1, t1); // F(U1) only: the final update is from the sums
        steps.accumulate_butcher(level, w1);
        steps.euler_using(level, 0, 0, S::butcher_E, S::butcher_fluxes, t1, t1 - t0);
    }
};

template<typename Reconstruction, typename Riemann = RusanovRiemann>
struct TVDRK3Integrator
{
    static constexpr std::size_t n_states = 3;
    static constexpr auto w00             = 0.75;
    static constexpr auto w01             = 0.25;
    static constexpr auto w10             = 1. / 3.;
    static constexpr auto w11             = 2. / 3.;
    MHDSteps<Reconstruction, Riemann> steps;

    void operator()(auto& level, double const t0, double const t1) const
    {
        using S       = MHDSteps<Reconstruction, Riemann>;
        using P       = std::pair<double, std::size_t>;
        auto const dt = t1 - t0;
        steps.euler(level, 0, 1, t1, dt);
        steps.assign_butcher(level, w01 * w11);
        steps.euler(level, 1, 1, t1, dt);
        steps.accumulate_butcher(level, w01 * w11);
        steps.rk_blend(level, 2, P{w00, 0}, P{w01, 1});
        double const stage_t = t0 + 0.5 * dt;
        level.fill_state_ghosts(2, stage_t);
        steps.euler(level, 2, 2, t1, dt);
        steps.accumulate_butcher(level, w11);
        steps.euler_using(level, 0, 0, S::butcher_E, S::butcher_fluxes, t1, dt);
    }
};

template<typename Reconstruction, typename Riemann = RusanovRiemann>
struct SSPRK4_5Integrator
{
    static constexpr std::size_t n_states = 5;
    static constexpr auto w0              = 0.391752226571890;
    static constexpr auto w10             = 0.444370493651235;
    static constexpr auto w11             = 0.555629506348765;
    static constexpr auto w12             = 0.368410593050371;
    static constexpr auto w20             = 0.620101851488403;
    static constexpr auto w21             = 0.379898148511597;
    static constexpr auto w22             = 0.251891774271694;
    static constexpr auto w30             = 0.178079954393132;
    static constexpr auto w31             = 0.821920045606868;
    static constexpr auto w32             = 0.544974750228521;
    static constexpr auto w40             = 0.517231671970585;
    static constexpr auto w41             = 0.096059710526147;
    static constexpr auto w42             = 0.063692468666290;
    static constexpr auto w43             = 0.386708617503268;
    static constexpr auto w44             = 0.226007483236906;
    static constexpr double c1            = w0;
    static constexpr double c2            = w11 * c1 + w12;
    static constexpr double c3            = w21 * c2 + w22;
    static constexpr double c4            = w31 * c3 + w32;
    MHDSteps<Reconstruction, Riemann> steps;

    void operator()(auto& level, double const t0, double const t1) const
    {
        using S       = MHDSteps<Reconstruction, Riemann>;
        using P       = std::pair<double, std::size_t>;
        auto const dt = t1 - t0;
        steps.euler(level, 0, 1, t0 + c1 * dt, w0 * dt);
        steps.assign_butcher(level,
                             (w0 * w11 * w21 * w31 * w43 + w0 * w11 * w21 * w41 + w0 * w11 * w40));

        steps.rk_blend(level, 2, P{w10, 0}, P{w11, 1});
        steps.compute_fluxes(level, 1, t1);
        steps.euler_using(level, 2, 2, S::last_E, S::last_fluxes, t0 + c2 * dt, w12 * dt);
        steps.accumulate_butcher(level, (w12 * w21 * w31 * w43 + w12 * w21 * w41 + w12 * w40));

        steps.rk_blend(level, 3, P{w20, 0}, P{w21, 2});
        steps.compute_fluxes(level, 2, t1);
        steps.euler_using(level, 3, 3, S::last_E, S::last_fluxes, t0 + c3 * dt, w22 * dt);
        steps.accumulate_butcher(level, (w22 * w31 * w43 + w22 * w41));

        steps.rk_blend(level, 4, P{w30, 0}, P{w31, 3});
        steps.compute_fluxes(level, 3, t1);
        steps.euler_using(level, 4, 4, S::last_E, S::last_fluxes, t0 + c4 * dt, w32 * dt);
        steps.accumulate_butcher(level, (w32 * w43 + w42));

        steps.compute_fluxes(level, 4, t1);
        steps.accumulate_butcher(level, w44);
        steps.euler_using(level, 0, 0, S::butcher_E, S::butcher_fluxes, t1, dt);
    }
};

} // namespace plamn::core

#endif /* PLAMN_CORE_NUMERICS_TIME_INTEGRATOR_MHD_TIME_INTEGRATOR_HPP */
