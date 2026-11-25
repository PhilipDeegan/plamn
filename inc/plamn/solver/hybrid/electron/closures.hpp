#ifndef PLAMN_SOLVER_HYBRID_ELECTRON_CLOSURES_HPP
#define PLAMN_SOLVER_HYBRID_ELECTRON_CLOSURES_HPP

// the electron pressure closures (core/data/electrons/closures.hpp) on a whole level: an overload
//  per closure, std::visit on the model's closure picks it (pressure_level)
//
//   pressure(closure, level, stage)   every patch's Pe from its Ne (after the electrons' Ve), at
//                                     a stage of the step (ClosureStage)
//   seed(closure, level, sim)         a new level's closure state from the input (t = 0)
//
// unified memory (untiled): one launch per operation for the level, else per patch on the pool
//  (tiled: on untiled copies). polytropic: per stage, as PHARE's polytropic closure,
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
#include "damn/core/exec/kernels/region_chunks.hpp"
#include "damn/core/operators.hpp"
#include "damn/mpi/mpi_utils.hpp"
#include "plamn/amr/untiled.hpp"
#include "damn/core/exec/kernels.hpp"
#include "mkn/gpu.hpp"
#include "plamn/core/data/field/field_tiles.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/initializer/data_provider.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <cstdlib>
#include <type_traits>
#include <utility>

namespace plamn::solver::hybrid
{
// where in a level's step a closure is asked for Pe: init (a new level, dt = 0), the PPC
//  step's first stage (predictor 1: the state at t saved), a middle one, the last (corrector)
struct ClosureStage
{
    enum Kind : std::uint8_t { init, first, middle, last };
    Kind kind = init;
    double t = 0, dt = 0;
};

// isothermal: Pe = Ne Te
template<auto opts>
    requires(opts.alloc_mode == damn::AllocatorMode::GPU_UNIFIED
             and !core::is_tiled(opts.layout_mode))
void pressure(core::hybrid::IsothermalClosure const& closure, auto& level, ClosureStage const&)
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
void pressure(core::hybrid::IsothermalClosure const& closure, auto& level, ClosureStage const&)
{
    level.per_patch_threaded([Te = closure.Te](auto& patch) {
        auto& electrons = patch.electrons();
        core::transform(electrons.density(), electrons.pressure(),
                        [Te](auto const n) { return n * Te; });
    });
}
template<auto opts>
void seed(core::hybrid::IsothermalClosure const&, auto&, auto const&)
{ // no state
}

namespace polytropic
{
    using C = core::hybrid::PolytropicClosure;

    // the plain views a polytropic kernel works on, one patch (device copyable: basic:: views,
    //  arrays), and the box its points run over
    template<typename F, typename Box, std::size_t dim>
    struct Item
    {
        F Ne, Pe, s_old, s, tmp;
        std::array<F, 3> Ve;
        std::array<double, dim> inv_dx;
        std::array<Box, 1> boxes; // what run_region_chunks iterates
    };

    // the item of untiled fields (a patch's, or untiled copies of a tiled patch's) over the
    //  patch's domain, or (ghosts) all its nodes
    auto make_item(auto const& layout, bool const ghosts, auto& Ne, auto& Pe, auto& s_old, auto& s,
                   auto& tmp, auto& vx, auto& vy, auto& vz)
    {
        auto constexpr dim = std::decay_t<decltype(layout)>::dimension;
        auto const box     = ghosts ? ghost_box(layout, Pe) : domain_box(layout, Pe);
        using F            = decltype(core::untiled_view(Ne));
        return Item<F, std::decay_t<decltype(box)>, dim>{
            core::untiled_view(Ne),
            core::untiled_view(Pe),
            core::untiled_view(s_old),
            core::untiled_view(s),
            core::untiled_view(tmp),
            {core::untiled_view(vx), core::untiled_view(vy), core::untiled_view(vz)},
            core::inverse_mesh_size(layout),
            {box}};
    }

    // fn(untiled...) with the fields untiled: a tiled one as a copy of its tiles (written back
    //  after), an untiled one itself
    void untiled_all(auto&& fn)
    {
        fn();
    }
    void untiled_all(auto&& fn, auto& field, auto&... fields)
    {
        amr::modify_untiled(
            field, [&](auto& u) { untiled_all([&](auto&... us) { fn(u, us...); }, fields...); });
    }

    // fn(field...) with the patch's fields a polytropic kernel reads and writes
    void with_fields(auto& patch, auto&& fn)
    {
        auto& electrons = patch.electrons();
        auto& Ve        = electrons.velocity();
        fn(electrons.density(), electrons.pressure(), *patch.electron_scalar(C::step),
           *patch.electron_scalar(C::subcycle), *patch.electron_scalar(C::scratch), Ve[0], Ve[1],
           Ve[2]);
    }

    // kernel(item, point) at every point of every patch of the level, over the domains (or, with
    //  ghosts, every node): unified memory (untiled) one launch for the level (damn's region
    //  chunks, the item list kept), else per patch on the pool (tiled: on untiled copies)
    template<auto opts>
        requires(opts.alloc_mode == damn::AllocatorMode::GPU_UNIFIED
                 and !core::is_tiled(opts.layout_mode))
    void run(auto& level, bool const ghosts, auto const kernel)
    {
        using Patch_t = HybridModel<opts>::Patch_t;
        using Item_t  = decltype(make_item(std::declval<Patch_t&>().layout, false,
                                           std::declval<Patch_t&>().electrons().density(),
                                           std::declval<Patch_t&>().electrons().pressure(),
                                           *std::declval<Patch_t&>().electron_scalar(0),
                                           *std::declval<Patch_t&>().electron_scalar(0),
                                           *std::declval<Patch_t&>().electron_scalar(0),
                                           std::declval<Patch_t&>().electrons().velocity()[0],
                                           std::declval<Patch_t&>().electrons().velocity()[1],
                                           std::declval<Patch_t&>().electrons().velocity()[2]));
        static thread_local damn::core::gpu::Vec_t<Item_t> items;
        items.clear();
        level.per_patch([&](auto& patch) {
            with_fields(
                patch, [&](auto&... f) { items.push_back(make_item(patch.layout, ghosts, f...)); });
        });
        damn::core::run_region_chunks(
            items, [=] DAMN_ALL(auto& it, auto const, auto const& p) { kernel(it, p); });
    }
    template<auto opts>
        requires(opts.alloc_mode == damn::AllocatorMode::CPU or core::is_tiled(opts.layout_mode))
    void run(auto& level, bool const ghosts, auto const kernel)
    {
        level.per_patch_threaded([&](auto& patch) {
            with_fields(patch, [&](auto&... fields) {
                untiled_all(
                    [&](auto&... u) {
                        auto item = make_item(patch.layout, ghosts, u...);
                        damn::core::for_each_point(item.boxes[0],
                                                   [&](auto const& p) { kernel(item, p); });
                    },
                    fields...);
            });
        });
    }

    // patch ghosts, then (refined levels) level ghosts: the coarser level's Se_old and Se
    //  interpolated at t (its step's start and end; alpha 1 out of a coarser step)
    void fill_ghosts(auto& level, C::Field const field, double const t)
    {
        std::string const key = C::names[field];
        level.copy(damn::amr::fill_ghosts, key);
        if (level.number() > 0)
            level.refine(amr::FieldRefiner{}, damn::amr::LevelRegion::LevelGhosts,
                         {damn::amr::LevelPair{C::names[C::step], key, C::names[C::subcycle]}},
                         level.alpha(t));
    }

    // the kernels, at a point p of a patch's item
    // Se_old = Pe / Ne^gamma, where Ne > 0
    auto begin_step(double const gamma)
    {
        return [=] DAMN_ALL(auto& it, auto const& p) {
            if (auto const n = it.Ne(p); n > 0)
                it.s_old(p) = it.Pe(p) / std::pow(n, gamma);
        };
    }
    // Se = Se_old (every node: a stage's start)
    auto start_stage()
    {
        return [] DAMN_ALL(auto& it, auto const& p) { it.s(p) = it.s_old(p); };
    }
    // the largest sum over the directions of |Ve_d| dt / dx_d (Ne > 0) into *word, its bits (a
    //  non negative double's order is its bits'): the upwind update is a convex combination
    //  (stable, monotone, positive) only up to 1
    template<bool gpu>
    auto max_courant(double const dt, unsigned long long* const word)
    {
        return [=] DAMN_ALL(auto& it, auto const& p) {
            if (it.Ne(p) <= 0)
                return;
            double c = 0;
            for (std::size_t d = 0; d < it.inv_dx.size(); ++d)
                c += std::abs(it.Ve[d](p)) * dt * it.inv_dx[d];
            damn::core::Operators<unsigned long long, true, gpu>{*word}.max_return_old(
                std::bit_cast<unsigned long long>(c));
        };
    }
    // Se_tmp = Se advanced over dt, upwind (each direction from the side the flow comes from);
    //  Ne <= 0: Se kept
    auto advance(double const dt)
    {
        return [=] DAMN_ALL(auto& it, auto const& p) {
            auto const s = it.s(p);
            if (it.Ne(p) <= 0)
            {
                it.tmp(p) = s;
                return;
            }
            double adv = 0;
            for (std::size_t d = 0; d < it.inv_dx.size(); ++d)
            {
                auto const v = it.Ve[d](p);
                auto q       = p;
                if (v > 0)
                {
                    --q[d];
                    adv += v * it.inv_dx[d] * (s - it.s(q));
                }
                else
                {
                    ++q[d];
                    adv += v * it.inv_dx[d] * (it.s(q) - s);
                }
            }
            it.tmp(p) = s - dt * adv;
        };
    }
    // Se = Se_tmp
    auto store()
    {
        return [] DAMN_ALL(auto& it, auto const& p) { it.s(p) = it.tmp(p); };
    }
    // Se = Se_tmp, Pe = Se Ne^gamma (Ne > 0)
    auto store_pressure(double const gamma)
    {
        return [=] DAMN_ALL(auto& it, auto const& p) {
            it.s(p) = it.tmp(p);
            if (auto const n = it.Ne(p); n > 0)
                it.Pe(p) = it.s(p) * std::pow(n, gamma);
        };
    }
    // Pe = Se_old Ne^gamma (Ne > 0)
    auto pressure_from_old(double const gamma)
    {
        return [=] DAMN_ALL(auto& it, auto const& p) {
            if (auto const n = it.Ne(p); n > 0)
                it.Pe(p) = it.s_old(p) * std::pow(n, gamma);
        };
    }

    // a new level's Pe from its Se_old (Se as Se_old: a stage's state before any)
    template<auto opts>
    void pressure_from_state(C const& closure, auto& level)
    {
        run<opts>(level, /*ghosts=*/true, start_stage());
        run<opts>(level, /*ghosts=*/false, pressure_from_old(closure.gamma));
    }
} // namespace polytropic

template<auto opts>
void pressure(core::hybrid::PolytropicClosure const& closure, auto& level,
              ClosureStage const& stage)
{
    namespace p        = polytropic;
    using C            = core::hybrid::PolytropicClosure;
    bool constexpr gpu = opts.alloc_mode == damn::AllocatorMode::GPU_UNIFIED;
    if (stage.kind == ClosureStage::init or stage.kind == ClosureStage::first)
    {
        p::run<opts>(level, false, p::begin_step(closure.gamma));
        p::fill_ghosts(level, C::step, stage.t);
    }

    // the level's sub-steps: its largest Courant number rounded up, the same on every rank (a
    //  rank past the limit reports an out of range count: every rank fails in the same reduction)
    static thread_local damn::core::gpu::Vec_t<unsigned long long> word(1);
    word[0] = 0;
    p::run<opts>(level, false, p::max_courant<gpu>(stage.dt, word.data()));
    auto const c                 = std::bit_cast<double>(word[0]);
    std::size_t constexpr failed = std::numeric_limits<std::size_t>::max();
    auto const local
        = c > C::max_subcycled_courant
              ? failed
              : std::max<std::size_t>(
                    1, static_cast<std::size_t>(std::ceil(c * (1 - C::courant_tolerance))));
    auto const n = damn::mpi::max(local);
    if (n == failed)
    {
        std::stringstream ss;
        ss << "plamn polytropic closure: sum over directions of |Ve| dt / dx is ";
        if (local == failed)
            ss << c;
        else
            ss << "(another rank's)";
        ss << " > " << C::max_subcycled_courant << " on level " << level.number()
           << ": too many sub-steps for the electron pressure, reduce the time step";
        throw std::runtime_error(ss.str());
    }
    auto const sub_dt = stage.dt / static_cast<double>(n);

    p::run<opts>(level, /*ghosts=*/true, p::start_stage());
    for (std::size_t i = 1; i < n; ++i)
    {
        p::run<opts>(level, false, p::advance(sub_dt));
        p::run<opts>(level, false, p::store());
        p::fill_ghosts(level, C::subcycle, stage.t + static_cast<double>(i) * sub_dt);
    }
    p::run<opts>(level, false, p::advance(sub_dt));
    p::run<opts>(level, false, p::store_pressure(closure.gamma));
    if (stage.kind == ClosureStage::last or stage.kind == ClosureStage::init)
        p::fill_ghosts(level, C::subcycle, stage.t + stage.dt);
}

// a new level at t = 0: Se_old from the input's initial entropy s (Pe / n^gamma of the analytic
//  densities), Pe = Se_old Ne^gamma; sim: the input's "simulation" node
template<auto opts>
void seed(core::hybrid::PolytropicClosure const& closure, auto& level, auto const& sim)
{
    using C           = core::hybrid::PolytropicClosure;
    auto const s_init = sim["electrons"]["pressure_closure"]["s"]
                            .template to<initializer::InitFunction<opts.dimension>>();
    level.per_patch([&](auto& patch) { // the input's functions on the host, as the root's
        core::FieldUserFunctionInitializer::initialize(*patch.electron_scalar(C::step),
                                                       patch.layout, s_init);
    });
    polytropic::pressure_from_state<opts>(closure, level);
}

// a level made or regridded at t (every level synchronized) from its coarser level, op: where
//  (amr::FieldRefiner: everywhere, a Masked one: the new cells, the others moved from the old
//  level): Se_old from the coarser level's Pe and Ne now, the level's own where it was (its Pe
//  moved with it, its Ne deposited again), Pe from it (PHARE's regrid: the entropy refined, not
//  the pressure)
template<auto opts>
void refine_state(core::hybrid::IsothermalClosure const&, auto&, auto const&, double)
{ // no state
}
template<auto opts>
void refine_state(core::hybrid::PolytropicClosure const& closure, auto& level, auto const& op,
                  double const t)
{
    namespace p = polytropic;
    using C     = core::hybrid::PolytropicClosure;
    auto coarse = level.coarser();
    p::run<opts>(coarse, false, p::begin_step(closure.gamma));
    p::fill_ghosts(coarse, C::step, t);
    p::run<opts>(level, false, p::begin_step(closure.gamma));
    level.refine(op, damn::amr::LevelRegion::Interior,
                 {damn::amr::LevelPair{C::names[C::step], C::names[C::step], ""}});
    p::fill_ghosts(level, C::step, t);
    p::pressure_from_state<opts>(closure, level);
}

// a restart loaded (Pe stored, electron_restart_names): the closure's state from it
template<auto opts>
void restart_loaded(core::hybrid::IsothermalClosure const&, auto&)
{ // Pe from Ne at the first Ohm
}
template<auto opts>
void restart_loaded(core::hybrid::PolytropicClosure const& closure, auto& level)
{ // Se_old and Se from Pe, as the first stage sets them (Pe as stored)
    namespace p = polytropic;
    p::run<opts>(level, false, p::begin_step(closure.gamma));
    p::run<opts>(level, true, p::start_stage());
}

// a synchronization: what the coarser level under the level takes of its closure
template<auto opts>
void synchronize(core::hybrid::IsothermalClosure const&, auto&)
{ // Pe from its own Ne
}
template<auto opts>
void synchronize(core::hybrid::PolytropicClosure const&, auto& level)
{ // Pe, primal as the moments: the fine nodes' values (PHARE's electronPressureSynchronizers_)
    level.coarsen(amr::MomentsCoarsener{}, {damn::amr::LevelPair{"Pe", "Pe", ""}});
}

// the model's closure's Pe on every patch of the level, at a stage of its step
template<auto opts>
void pressure_level(HybridModel<opts> const& model, auto& level, ClosureStage const& stage)
{
    std::visit([&](auto const& closure) { pressure<opts>(closure, level, stage); },
               *model.electron_closure);
}

// a new level's closure state from the input (sim: the "simulation" node), before its Pe
template<auto opts>
void seed_level(HybridModel<opts> const& model, auto& level, auto const& sim)
{
    std::visit([&](auto const& closure) { seed<opts>(closure, level, sim); },
               *model.electron_closure);
}

// a level made or regridded at t: its closure state from its coarser level (refine_state)
template<auto opts>
void refine_state_level(HybridModel<opts> const& model, auto& level, auto const& op, double const t)
{
    std::visit([&](auto const& closure) { refine_state<opts>(closure, level, op, t); },
               *model.electron_closure);
}

// a restart loaded: the closure's state from what was stored
template<auto opts>
void restart_loaded_level(HybridModel<opts> const& model, auto& level)
{
    std::visit([&](auto const& closure) { restart_loaded<opts>(closure, level); },
               *model.electron_closure);
}

// a synchronization of the level onto its coarser one: the closure's part
template<auto opts>
void synchronize_level(HybridModel<opts> const& model, auto& level)
{
    std::visit([&](auto const& closure) { synchronize<opts>(closure, level); },
               *model.electron_closure);
}

} // namespace plamn::solver::hybrid

#endif // PLAMN_SOLVER_HYBRID_ELECTRON_CLOSURES_HPP
