#ifndef PLAMN_CORE_NUMERICS_TIME_INTEGRATOR_MHD_TIME_INTEGRATOR_HPP
#define PLAMN_CORE_NUMERICS_TIME_INTEGRATOR_MHD_TIME_INTEGRATOR_HPP

#include "damn/core/meta/for_N.hpp"
#include "plamn/core/data/field/field_tiles.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"
#include "plamn/core/numerics/constrained_transport/upwind_ct_transformer.hpp"
#include "plamn/core/numerics/faraday/faraday_transformer.hpp"
#include "plamn/core/numerics/finite_volume/finite_volume_euler_transformer.hpp"
#include "plamn/core/numerics/godunov/godunov_transformer.hpp"
#include "plamn/core/numerics/primitive_conservative/primitive_conservative_transformer.hpp"
#include "damn/core/geometry/box_spans.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>
#include <concepts>
#include <iterator>

// MHD time integration on one level, PHARE's amr/solvers/time_integrator (Euler, TVDRK2, TVDRK3,
//  SSPRK4_5) in its operation order, so results can match its own bit for bit (in the same
//  floating point context, see the tests)
//
// the level: an adapter giving
//   level.per_patch(fn)                  fn(patch) for every local patch
//   level.fill_state_ghosts(s, t)        patch and level ghosts of every patch's states[s]
//                                        (rho, rhoV, B, Etot) at time t
//   level.fill_electric_ghosts(t)        the same for every patch's E
// a patch: members
//   layout
//   states    indexable, MHDState (core/models/mhd_state.hpp): [0] the state advanced, the
//             others the integrator's stages (n_states of them in all)
//   E         the constrained transport E of the last fluxes computed (one: each E is used
//             right after it is computed, PHARE keeps one per state)
//   fluxes    [3] MHDFaceFluxes, ct [3] MHDFaceCT: of the last fluxes computed
//   butcher   [3] MHDFaceFluxes, butcherE: the step's weighted sums of fluxes and E (what a
//             coarser level is synchronised with)
// per patch work goes through the transformers: tiled patches are computed tile by tile
//  (todo: the level transformer's pool tasks; the butcher sums and RK blends run on the host)
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
        for (std::size_t c = 0; c < 3; ++c)
            fn(f0.F_B[c], f.F_B[c]...);
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
template<typename Reconstruction>
class MHDSteps
{
public:
    explicit MHDSteps(double const gamma)
        : gamma_{gamma}
    {
    }

    // fluxes and E from states[s]: to primitive, Godunov per direction, to conservative (rhoV
    //  and Etot from V and P again, as PHARE), constrained transport; then E's ghosts
    void compute_fluxes(auto& level, std::size_t const s, double const t) const
    {
        level.per_patch([&](auto& p) {
            auto constexpr static dim = std::decay_t<decltype(p.layout)>::dimension;
            auto& u                   = p.states[s];
            PrimitiveConservativeTransformer const pc{gamma_};
            GodunovTransformer<Reconstruction> const godunov{gamma_};

            pc.to_primitive(p.layout, u.rho, u.rhoV, u.B, u.Etot, u.V, u.P);
            damn::core::for_N<dim>([&](auto d) {
                auto& f = p.fluxes[d];
                auto& c = p.ct[d];
                godunov.template operator()<d>(p.layout, u.rho, u.V, u.B, u.P, f.F_rho, f.F_rhoV,
                                               f.F_B, f.F_Etot, c.vt, c.aL, c.aR, c.dL, c.dR);
            });
            pc.to_conservative(p.layout, u.rho, u.V, u.B, u.P, u.rhoV, u.Etot);
            auto& [x, y, z] = p.ct;
            UpwindCTTransformer<Reconstruction>{}(p.layout, u.B, x.vt, x.aL, x.aR, x.dL, x.dR, y.vt,
                                                  y.aL, y.aR, y.dL, y.dR, z.vt, z.aL, z.aR, z.dL,
                                                  z.dR, p.E);
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

    void reset_butcher(auto& level) const
    {
        level.per_patch([&](auto& p) {
            auto const zero = [](auto& f) {
                detail::on_arrays(
                    [](auto const n, auto* d) {
                        for (std::size_t i = 0; i < n; ++i)
                            d[i] = 0;
                    },
                    f);
            };
            for (auto& b : p.butcher)
                detail::for_each_flux_field(zero, b);
            for (std::size_t c = 0; c < 3; ++c)
                zero(p.butcherE[c]);
        });
    }

    // butcher += coef * (the last fluxes, E), every point (PHARE's PlusEqualsProduct)
    void accumulate_butcher(auto& level, double const coef) const
    {
        level.per_patch([&](auto& p) {
            auto const axpy = [&](auto& d, auto const& s) {
                detail::on_arrays(
                    [&](auto const n, auto* dp, auto const* sp) {
                        for (std::size_t i = 0; i < n; ++i)
                            dp[i] += sp[i] * coef;
                    },
                    d, s);
            };
            for (std::size_t d = 0; d < 3; ++d)
                detail::for_each_flux_field(axpy, p.butcher[d], p.fluxes[d]);
            for (std::size_t c = 0; c < 3; ++c)
                axpy(p.butcherE[c], p.E[c]);
        });
    }

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

    // which E and fluxes euler_using takes
    static constexpr auto last_E         = [](auto& p) -> auto& { return p.E; };
    static constexpr auto last_fluxes    = [](auto& p) -> auto& { return p.fluxes; };
    static constexpr auto butcher_E      = [](auto& p) -> auto& { return p.butcherE; };
    static constexpr auto butcher_fluxes = [](auto& p) -> auto& { return p.butcher; };

private:
    double gamma_;
};


// the integrators: operator()(level, t0, t1) advances every patch's states[0] from t0 to t1
template<typename Reconstruction>
struct EulerIntegrator
{
    static constexpr std::size_t n_states = 1;
    MHDSteps<Reconstruction> steps;

    void operator()(auto& level, double const t0, double const t1) const
    {
        steps.reset_butcher(level);
        steps.euler(level, 0, 0, t1, t1 - t0);
        steps.accumulate_butcher(level, 1.0);
    }
};

template<typename Reconstruction>
struct TVDRK2Integrator
{
    static constexpr std::size_t n_states = 2;
    static constexpr auto w0              = 0.5;
    static constexpr auto w1              = 0.5;
    MHDSteps<Reconstruction> steps;

    void operator()(auto& level, double const t0, double const t1) const
    {
        using S = MHDSteps<Reconstruction>;
        steps.reset_butcher(level);
        steps.euler(level, 0, 1, t1, t1 - t0);
        steps.accumulate_butcher(level, w0);
        steps.euler(level, 1, 1, t1, t1 - t0);
        steps.accumulate_butcher(level, w1);
        steps.euler_using(level, 0, 0, S::butcher_E, S::butcher_fluxes, t1, t1 - t0);
    }
};

template<typename Reconstruction>
struct TVDRK3Integrator
{
    static constexpr std::size_t n_states = 3;
    static constexpr auto w00             = 0.75;
    static constexpr auto w01             = 0.25;
    static constexpr auto w10             = 1. / 3.;
    static constexpr auto w11             = 2. / 3.;
    MHDSteps<Reconstruction> steps;

    void operator()(auto& level, double const t0, double const t1) const
    {
        using S       = MHDSteps<Reconstruction>;
        using P       = std::pair<double, std::size_t>;
        auto const dt = t1 - t0;
        steps.reset_butcher(level);
        steps.euler(level, 0, 1, t1, dt);
        steps.accumulate_butcher(level, w01 * w11);
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

template<typename Reconstruction>
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
    MHDSteps<Reconstruction> steps;

    void operator()(auto& level, double const t0, double const t1) const
    {
        using S       = MHDSteps<Reconstruction>;
        using P       = std::pair<double, std::size_t>;
        auto const dt = t1 - t0;
        steps.reset_butcher(level);

        steps.euler(level, 0, 1, t0 + c1 * dt, w0 * dt);
        steps.accumulate_butcher(
            level, (w0 * w11 * w21 * w31 * w43 + w0 * w11 * w21 * w41 + w0 * w11 * w40));

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
