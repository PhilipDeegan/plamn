#ifndef PLAMN_SOLVER_MHD_LEVEL_STEPS_HPP
#define PLAMN_SOLVER_MHD_LEVEL_STEPS_HPP

// an MHD level's steps with its coarser level (PHARE's SolverMHD and MultiPhysicsIntegrator for
//  an MHD pair): the reflux is all of the synchronization: doc/solver/mhd/level_steps.md

#include "damn/amr/copy.hpp"
#include "damn/amr/level_map.hpp"

#include "plamn/amr/coarsen.hpp"
#include "plamn/amr/level_pairs.hpp"
#include "plamn/core/numerics/time_integrator/mhd_time_integrator.hpp"
#include "plamn/solver/mhd/model.hpp"

#include <type_traits>
#include <cstddef>
#include <string>
#include <vector>

namespace plamn::solver::mhd
{
namespace detail
{
    // a patch whose states are {old, states[0]}: the reflux redoes its step from old
    template<typename Patch>
    struct RefluxPatch
    {
        struct States
        {
            Patch& p;
            auto& operator[](std::size_t const s) { return s == 0 ? p.old : p.states[0]; }
            auto const& operator[](std::size_t const s) const
            {
                return s == 0 ? p.old : p.states[0];
            }
        };

        Patch& patch;
        Patch::Layout_t const& layout;
        States states;
    };

    template<typename Level>
    struct RefluxLevel
    {
        Level& level;

        void per_patch(auto&& fn)
        {
            level.per_patch([&](auto& p) {
                RefluxPatch<std::remove_cvref_t<decltype(p)>> r{p, p.layout, {p}};
                fn(r);
            });
        }
        void fill_state_ghosts(std::size_t, double const t) { level.fill_state_ghosts(0, t); }
        void fill_electric_ghosts(double const t) { level.fill_electric_ghosts(t); }
    };

    // dst += coef src, every point (PHARE's PlusEqualsProduct)
    void axpy(auto& dst, auto const& src, double const coef)
    {
        auto* d       = dst.data();
        auto const* s = src.data();
        for (std::size_t i = 0; i < dst.size(); ++i)
            d[i] += s[i] * coef;
    }
} // namespace detail

template<auto opts>
void first_step(MHDModel<opts> const&, auto&& level)
{
    level.per_patch([](auto& patch) {
        auto const zero = [](auto& g) { g.fill(0.); };
        for (auto& f : patch.flux_sum)
            patch.for_each_flux_grid(f, zero);
        for (auto& g : patch.flux_sum_E.grids)
            zero(g);
    });
}

template<auto opts>
void accumulate_flux_sum(MHDModel<opts> const&, auto&& level, double const coef)
{
    level.per_patch([&](auto& patch) {
        for (std::size_t d = 0; d < 3; ++d)
        {
            auto sums = patch.flux_grids(patch.flux_sum[d]);
            auto step = patch.flux_grids(patch.butcher[d]);
            for (std::size_t i = 0; i < sums.size(); ++i)
                detail::axpy(*sums[i], *step[i], coef);
        }
        for (std::size_t c = 0; c < 3; ++c)
            detail::axpy(patch.flux_sum_E.grids[c], patch.butcherE.grids[c], coef);
    });
}

template<auto opts>
void synchronize(MHDModel<opts> const& model, auto&& level, double const t)
{
    auto coarse = level.coarser();
    std::vector<damn::amr::LevelPair> fluxes;
    for (std::size_t d = 0; d < opts.dimension; ++d)
    {
        std::string const x{"xyz"[d]};
        fluxes.push_back({"F_rho_butcher_" + x, "F_rho_sum_" + x, ""});
        fluxes.push_back({"F_Etot_butcher_" + x, "F_Etot_sum_" + x, ""});
        for (auto const& p : amr::vector_pairs("F_rhoV_butcher_" + x, "F_rhoV_sum_" + x))
            fluxes.push_back(p);
    }
    level.coarsen(amr::MHDFluxCoarsener{}, fluxes);
    level.coarsen(amr::ElectricFieldCoarsener{}, amr::vector_pairs("butcherE", "E_sum"));
    coarse.copy(damn::amr::fill_ghosts, "MHDReflux");

    using R = MHDModel<opts>::Reconstruction;
    detail::RefluxLevel<decltype(coarse)> reflux{coarse};
    core::MHDSteps<R>{model.gamma}.euler_using(
        reflux, 0, 1, [](auto& r) -> auto& { return r.patch.butcherE; },
        [](auto& r) -> auto& { return r.patch.butcher; }, t, t - coarse.old_time());

    if (coarse.number() > 0)
        accumulate_flux_sum(model, coarse, coarse.substep_weight());
}

} // namespace plamn::solver::mhd

#endif // PLAMN_SOLVER_MHD_LEVEL_STEPS_HPP
