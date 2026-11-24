#ifndef PLAMN_SOLVER_HYBRID_CLOSURES_HPP
#define PLAMN_SOLVER_HYBRID_CLOSURES_HPP

// the electron pressure closures (core/data/electrons/closures.hpp) on a whole level: an overload
//  per closure, std::visit on the model's closure picks it (pressure_level)
//
//   pressure(closure, level, stage)   every patch's Pe from its Ne (after the electrons' Ve), at
//                                     a stage of the step (ClosureStage)
//   seed(closure, level, sim)         a new level's closure state from the input (t = 0)
//
// isothermal: unified memory (untiled) one launch for the level, else per patch on the pool.
// polytropic (host memory, untiled fields; refused otherwise at setup): per stage, as
//  PHARE's polytropic closure,
//   first stage (and init): Se_old = Pe / Ne^gamma on the domain (Pe, Ne at t), its ghosts
//     from the level's patches and, at its border, the coarser level's Se_old (t_n) and Se
//     (t_n+1) interpolated at t
//   every stage: Se from Se_old, n sub-steps of dt / n (n: the level's largest electron Courant
//     number, rounded up, the same on every rank: the ghost fills between them are collective),
//     upwind ds/dt + Ve.grad s = 0, Se's ghosts filled between them as Se_old's at their times;
//     then Pe = Se Ne^gamma on the domain. A node with Ne <= 0 keeps its s and Pe
//   last stage: Se's ghosts at t + dt (what the finer level's ghosts interpolate to)

#include "plamn/amr/coarsen.hpp"
#include "plamn/amr/refine.hpp"
#include "plamn/core/data/electrons/closures.hpp"
#include "plamn/core/data/field/field_algorithm.hpp"
#include "plamn/core/data/field/initializers/field_user_initializer.hpp"
#include "plamn/core/data/field/level_fields.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"
#include "plamn/solver/hybrid/model.hpp"
#include "damn/amr/copy.hpp"
#include "damn/amr/level_map.hpp"
#include "damn/core/def/damn_config.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/mpi/mpi_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace plamn::solver::hybrid
{
// where in a level's step a closure is asked for Pe: init (a new level, dt = 0), the PPC
//  step's first stage (predictor 1: the state at t saved), a middle one, the last (corrector)
struct ClosureStage
{
    enum Kind { init, first, middle, last };
    Kind kind = init;
    double t = 0, dt = 0;
};

// isothermal: Pe = Ne Te
template<auto opts>
    requires(opts.alloc_mode == damn::AllocatorMode::GPU_UNIFIED
             and !core::is_tiled(opts.layout_mode))
void pressure(core::IsothermalClosure const& closure, auto& level, ClosureStage const&)
{
    auto constexpr am = opts.alloc_mode;
    auto const Ne     = core::level_of<am>(
        level, [](auto& patch, auto& fields) { fields.add(patch.electrons().density()); });
    core::level_of<am>(level, [](auto& patch, auto& fields) {
        fields.add(patch.electrons().pressure());
    }).transform([Te = closure.Te] DAMN_ALL(double& pe, double const n) { pe = n * Te; }, Ne);
}
template<auto opts>
    requires(opts.alloc_mode == damn::AllocatorMode::CPU or core::is_tiled(opts.layout_mode))
void pressure(core::IsothermalClosure const& closure, auto& level, ClosureStage const&)
{
    level.per_patch_threaded([Te = closure.Te](auto& patch) {
        auto& electrons = patch.electrons();
        core::transform(electrons.density(), electrons.pressure(),
                        [Te](auto const n) { return n * Te; });
    });
}
template<auto opts>
void seed(core::IsothermalClosure const&, auto&, auto const&)
{ // no state
}

namespace polytropic
{
    using C = core::PolytropicClosure;

    // can a build run the polytropic closure (host memory, untiled fields)
    template<auto opts>
    bool constexpr supported
        = opts.alloc_mode == damn::AllocatorMode::CPU and !core::is_tiled(opts.layout_mode);

    // patch ghosts, then (refined levels) level ghosts: the coarser level's Se_old and Se
    //  interpolated at t (its step's start and end; alpha 1 out of a coarser step)
    void fill_ghosts(auto& level, std::string const& key, double const t)
    {
        level.copy(damn::amr::fill_ghosts, key);
        if (level.number() > 0)
            level.refine(amr::FieldRefiner{}, damn::amr::LevelRegion::LevelGhosts,
                         {damn::amr::LevelPair{C::step_state, key, C::subcycle_state}},
                         level.alpha(t));
    }

    // Se_old = Pe / Ne^gamma on the domain, where Ne > 0
    void begin_step(C const& closure, auto& patch)
    {
        auto& electrons = patch.electrons();
        auto const& Ne  = electrons.density();
        auto const& Pe  = electrons.pressure();
        auto& s_old     = *patch.closure_grid(C::step_state);
        damn::core::for_each_point(domain_box(patch.layout, Pe), [&](auto const& p) {
            if (auto const n = Ne(p); n > 0)
                s_old(p) = Pe(p) / std::pow(n, closure.gamma);
        });
    }

    // sum over the directions of |Ve_d| dt / dx_d: the upwind update is a convex combination
    //  (stable, monotone, positive) only up to 1
    double courant(auto const& Ve, auto const& p, auto const& inv_dx, double const dt)
    {
        double c = 0;
        for (std::size_t d = 0; d < inv_dx.size(); ++d)
            c += std::abs(Ve[d](p)) * dt * inv_dx[d];
        return c;
    }

    // the patch's largest courant over its domain nodes with Ne > 0, and where
    auto max_courant(auto& patch, double const dt)
    {
        auto& electrons   = patch.electrons();
        auto const& Ne    = electrons.density();
        auto const& Ve    = electrons.velocity();
        auto const inv_dx = core::inverse_mesh_size(patch.layout);
        double worst      = 0;
        std::string where;
        damn::core::for_each_point(domain_box(patch.layout, Ne), [&](auto const& p) {
            if (Ne(p) <= 0)
                return;
            if (auto const c = courant(Ve, p, inv_dx, dt); c > worst)
            {
                worst = c;
                std::stringstream ss;
                ss << p << " (n = " << Ne(p) << ", Ve =";
                for (std::size_t d = 0; d < inv_dx.size(); ++d)
                    ss << " " << Ve[d](p);
                ss << ")";
                where = ss.str();
            }
        });
        return std::make_pair(worst, where);
    }

    // Se over dt on the domain nodes (from Se, its ghosts filled), returned in for_each_point
    //  order: the stencil reads Se's neighbours
    std::vector<double> advance(auto& patch, double const dt)
    {
        auto& electrons   = patch.electrons();
        auto const& Ne    = electrons.density();
        auto const& Ve    = electrons.velocity();
        auto const& s     = *patch.closure_grid(C::subcycle_state);
        auto const inv_dx = core::inverse_mesh_size(patch.layout);
        auto const box    = domain_box(patch.layout, Ne);
        std::vector<double> s1;
        s1.reserve(box.size());
        damn::core::for_each_point(box, [&](auto const& p) {
            if (Ne(p) <= 0)
            {
                s1.push_back(s(p));
                return;
            }
            double adv = 0; // Ve.grad s, each direction from the side the flow comes from
            for (std::size_t d = 0; d < inv_dx.size(); ++d)
            {
                auto const v = Ve[d](p);
                auto q       = p;
                if (v > 0)
                {
                    --q[d];
                    adv += v * inv_dx[d] * (s(p) - s(q));
                }
                else
                {
                    ++q[d];
                    adv += v * inv_dx[d] * (s(q) - s(p));
                }
            }
            s1.push_back(s(p) - dt * adv);
        });
        return s1;
    }

    // a sub-step: Se on the domain advanced over dt
    void subcycle(auto& patch, double const dt)
    {
        auto const s1 = advance(patch, dt);
        auto& s       = *patch.closure_grid(C::subcycle_state);
        std::size_t i = 0;
        damn::core::for_each_point(domain_box(patch.layout, s), [&](auto const& p) {
            s(p) = s1[i++];
        });
    }

    // the stage's last sub-step, then Pe = Se Ne^gamma on the domain (Ne > 0)
    void last_subcycle(C const& closure, auto& patch, double const dt)
    {
        auto const s1   = advance(patch, dt);
        auto& electrons = patch.electrons();
        auto const& Ne  = electrons.density();
        auto& Pe        = electrons.pressure();
        auto& s         = *patch.closure_grid(C::subcycle_state);
        std::size_t i   = 0;
        damn::core::for_each_point(domain_box(patch.layout, Pe), [&](auto const& p) {
            s(p) = s1[i++];
            if (auto const n = Ne(p); n > 0)
                Pe(p) = s(p) * std::pow(n, closure.gamma);
        });
    }

    // Pe = Se_old Ne^gamma on the domain (Ne > 0): a new level's, from its state (Se as
    //  Se_old, ghosts included: a stage's state before any)
    void pressure_from_state(C const& closure, auto& patch)
    {
        {
            auto const& from = *patch.closure_grid(C::step_state);
            auto& to         = *patch.closure_grid(C::subcycle_state);
            std::copy(from.data(), from.data() + from.size(), to.data());
        }
        auto& electrons   = patch.electrons();
        auto const& Ne    = electrons.density();
        auto& Pe          = electrons.pressure();
        auto const& s_old = *patch.closure_grid(C::step_state);
        damn::core::for_each_point(domain_box(patch.layout, Pe), [&](auto const& p) {
            if (auto const n = Ne(p); n > 0)
                Pe(p) = s_old(p) * std::pow(n, closure.gamma);
        });
    }
} // namespace polytropic

template<auto opts>
    requires(polytropic::supported<opts>)
void pressure(core::PolytropicClosure const& closure, auto& level, ClosureStage const& stage)
{
    namespace p = polytropic;
    using C     = core::PolytropicClosure;
    if (stage.kind == ClosureStage::init or stage.kind == ClosureStage::first)
    {
        level.per_patch_threaded([&](auto& patch) { p::begin_step(closure, patch); });
        p::fill_ghosts(level, C::step_state, stage.t);
    }

    // the level's sub-steps: its largest Courant number rounded up (a patch past the limit
    //  reports an out of range count, so every rank fails in the same reduction)
    std::size_t constexpr failed = std::numeric_limits<std::size_t>::max();
    std::size_t local            = 1;
    std::string why;
    level.per_patch([&](auto& patch) {
        auto const [c, where] = p::max_courant(patch, stage.dt);
        if (c > C::max_subcycled_courant)
        {
            std::stringstream ss;
            ss << "plamn polytropic closure: sum over directions of |Ve| dt / dx is " << c
               << " > " << C::max_subcycled_courant << " on level " << level.number() << " at "
               << where << ": too many sub-steps for the electron pressure, reduce the time step";
            why   = ss.str();
            local = failed;
        }
        else if (local != failed)
            local = std::max<std::size_t>(
                local, static_cast<std::size_t>(std::ceil(c * (1 - C::courant_tolerance))));
        auto& s_old = *patch.closure_grid(C::step_state);
        auto& s     = *patch.closure_grid(C::subcycle_state);
        std::copy(s_old.data(), s_old.data() + s_old.size(), s.data()); // ghosts included
    });
    auto const n = damn::mpi::max(local);
    if (n == failed)
        throw std::runtime_error(why.empty() ? "plamn polytropic closure: refused on another rank"
                                             : why);
    auto const sub_dt = stage.dt / static_cast<double>(n);

    for (std::size_t i = 1; i < n; ++i)
    {
        level.per_patch_threaded([&](auto& patch) { p::subcycle(patch, sub_dt); });
        p::fill_ghosts(level, C::subcycle_state, stage.t + static_cast<double>(i) * sub_dt);
    }
    level.per_patch_threaded([&](auto& patch) { p::last_subcycle(closure, patch, sub_dt); });
    if (stage.kind == ClosureStage::last or stage.kind == ClosureStage::init)
        p::fill_ghosts(level, C::subcycle_state, stage.t + stage.dt);
}
template<auto opts>
    requires(!polytropic::supported<opts>)
void pressure(core::PolytropicClosure const&, auto&, ClosureStage const&)
{
    throw std::runtime_error("plamn: the polytropic closure runs on host memory, untiled layouts");
}

// a new level at t = 0: Se_old from the input's initial entropy s (Pe / n^gamma of the analytic
//  densities), Pe = Se_old Ne^gamma; sim: the input's "simulation" node
template<auto opts>
void seed(core::PolytropicClosure const& closure, auto& level, auto const& sim)
{
    if constexpr (!polytropic::supported<opts>)
        throw std::runtime_error(
            "plamn: the polytropic closure runs on host memory, untiled layouts");
    else
    {
        auto const s_init = sim["electrons"]["pressure_closure"]["s"]
                                .template to<initializer::InitFunction<opts.dimension>>();
        level.per_patch([&](auto& patch) {
            core::FieldUserFunctionInitializer::initialize(
                *patch.closure_grid(core::PolytropicClosure::step_state), patch.layout, s_init);
            polytropic::pressure_from_state(closure, patch);
        });
    }
}

// a level made or regridded at t (every level synchronized) from its coarser level, op: where
//  (amr::FieldRefiner: everywhere, a Masked one: the new cells, the others moved from the old
//  level): Se_old from the coarser level's Pe and Ne now, the level's own where it was (its Pe
//  moved with it, its Ne deposited again), Pe from it (PHARE's regrid: the entropy refined, not
//  the pressure)
template<auto opts>
void refine_state(core::IsothermalClosure const&, auto&, auto const&, double)
{ // no state
}
template<auto opts>
void refine_state(core::PolytropicClosure const& closure, auto& level, auto const& op,
                  double const t)
{
    if constexpr (!polytropic::supported<opts>)
        throw std::runtime_error(
            "plamn: the polytropic closure runs on host memory, untiled layouts");
    else
    {
        namespace p = polytropic;
        using C     = core::PolytropicClosure;
        auto coarse = level.coarser();
        coarse.per_patch_threaded([&](auto& patch) { p::begin_step(closure, patch); });
        p::fill_ghosts(coarse, C::step_state, t);
        level.per_patch_threaded([&](auto& patch) { p::begin_step(closure, patch); });
        level.refine(op, damn::amr::LevelRegion::Interior,
                     {damn::amr::LevelPair{C::step_state, C::step_state, ""}});
        p::fill_ghosts(level, C::step_state, t);
        level.per_patch_threaded([&](auto& patch) { p::pressure_from_state(closure, patch); });
    }
}

// a restart loaded (Pe stored, closure_restart_names): the closure's state from it
template<auto opts>
void restart_loaded(core::IsothermalClosure const&, auto&)
{ // Pe from Ne at the first Ohm
}
template<auto opts>
void restart_loaded(core::PolytropicClosure const& closure, auto& level)
{ // Se_old and Se from Pe, as the first stage sets them (Pe as stored)
    if constexpr (polytropic::supported<opts>)
    {
        using C = core::PolytropicClosure;
        level.per_patch([&](auto& patch) {
            polytropic::begin_step(closure, patch);
            auto const& from = *patch.closure_grid(C::step_state);
            auto& to         = *patch.closure_grid(C::subcycle_state);
            std::copy(from.data(), from.data() + from.size(), to.data());
        });
    }
}

// a synchronization: what the coarser level under the level takes of its closure
template<auto opts>
void synchronize(core::IsothermalClosure const&, auto&)
{ // Pe from its own Ne
}
template<auto opts>
void synchronize(core::PolytropicClosure const&, auto& level)
{ // Pe, primal as the moments: the fine nodes' values (PHARE's electronPressureSynchronizers_)
    level.coarsen(amr::MomentsCoarsener{}, {damn::amr::LevelPair{"Pe", "Pe", ""}});
}

// the model's closure's Pe on every patch of the level, at a stage of its step
template<auto opts>
void pressure_level(HybridModel<opts> const& model, auto& level, ClosureStage const& stage)
{
    std::visit([&](auto const& closure) { pressure<opts>(closure, level, stage); },
               *model.closure);
}

// a new level's closure state from the input (sim: the "simulation" node), before its Pe
template<auto opts>
void seed_level(HybridModel<opts> const& model, auto& level, auto const& sim)
{
    std::visit([&](auto const& closure) { seed<opts>(closure, level, sim); }, *model.closure);
}

// a level made or regridded at t: its closure state from its coarser level (refine_state)
template<auto opts>
void refine_state_level(HybridModel<opts> const& model, auto& level, auto const& op,
                        double const t)
{
    std::visit([&](auto const& closure) { refine_state<opts>(closure, level, op, t); },
               *model.closure);
}

// a restart loaded: the closure's state from what was stored
template<auto opts>
void restart_loaded_level(HybridModel<opts> const& model, auto& level)
{
    std::visit([&](auto const& closure) { restart_loaded<opts>(closure, level); },
               *model.closure);
}

// a synchronization of the level onto its coarser one: the closure's part
template<auto opts>
void synchronize_level(HybridModel<opts> const& model, auto& level)
{
    std::visit([&](auto const& closure) { synchronize<opts>(closure, level); }, *model.closure);
}

} // namespace plamn::solver::hybrid

#endif // PLAMN_SOLVER_HYBRID_CLOSURES_HPP
