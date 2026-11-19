#ifndef PLAMN_SIMULATOR_SIMULATOR_ADVANCE_HPP
#define PLAMN_SIMULATOR_SIMULATOR_ADVANCE_HPP

// plamn::Simulator: time refined advance, SolverPPC per level, synchronization and reflux
// IWYU pragma: private, include "plamn/simulator/simulator.hpp"

#include "damn/amr/copy.hpp"
#include "damn/amr/level_map.hpp"
#include "damn/amr/time_refinement.hpp"
#include "damn/core/logger.hpp"

#include "plamn/simulator/simulator.hpp" // IWYU pragma: keep (the class)
#include "plamn/simulator/level_accessor.hpp"
#include "plamn/amr/coarsen.hpp"
#include "plamn/amr/level_pairs.hpp"
#include "plamn/amr/refine.hpp"
#include "plamn/amr/tagging.hpp"
#include "plamn/core/data/field/field_algorithm.hpp"
#include "plamn/core/numerics/ampere/ampere_transformer.hpp"
#include "plamn/core/numerics/faraday/faraday_transformer.hpp"
#include "plamn/core/numerics/ohm/ohm.hpp"
#include "plamn/core/numerics/ohm/ohm_transformer.hpp"
#include "plamn/core/numerics/stable_dt/whistler.hpp"
#include "plamn/core/numerics/particles_synchronised_mover/particles_synchronised_mover_def.hpp"
#include "plamn/solver/models.hpp"

#include "phlop/timing/scope.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
// plamn::Simulator: time refined advance, SolverPPC per level, synchronization and reflux
// IWYU pragma: private, include "plamn/simulator/simulator.hpp"

namespace plamn
{
template<auto opts>
void Simulator<opts>::step_model_(std::size_t const ln, auto&& hybrid)
{
    models_.visit(ln, [&](auto& model) {
        if constexpr (solver::is_hybrid_model_v<std::remove_cvref_t<decltype(model)>>)
            hybrid(model);
        else
            throw std::runtime_error("plamn: MHD levels are not in the simulator yet");
    });
}

// PHARE's MultiPhysicsIntegrator::computeStableDt: a level's dt (its model's bound) as a level 0
//  one, times the substeps of each coarser level (the time ratio per level)
template<auto opts>
double Simulator<opts>::stable_dt_(Timestepper::CFL const& cfl)
{
    double dt0  = std::numeric_limits<double>::max();
    double to_0 = 1;
    for (std::size_t ln = 0; ln < levels_.size(); ++ln)
    {
        if (ln > 0)
            to_0 *= time_ratio();
        double level = std::numeric_limits<double>::max();
        step_model_(ln, [&](auto&) {
            per_patch_(ln, [&](auto& patch) {
                level = std::min(
                    level, core::whistler_dt(patch.layout, patch.massDensity, patch.emB, cfl.wave));
            });
        });
        dt0 = std::min(dt0, level * to_0);
    }
    return n_ranks_ > 1 ? damn::mpi::min(dt0) : dt0;
}

template<auto opts>
double Simulator<opts>::advance()
{
    DAMN_LOG_SCOPE(1, "Simulator::advance");
    if (!isInitialized_)
        throw std::runtime_error("plamn Simulator: initialize() first");

    if constexpr (Models_t::hybrid_on) // load balancing: by the hybrid levels' particles
        if (should_rebalance_())
            rebalance_(currentTime_);

    dt_ = timestepper_.next([&](auto const& cfl) { return stable_dt_(cfl); });
    Hooks hooks{*this};
    damn::amr::advance(hierarchy_, hooks, currentTime_, dt_, time_ratio());
    currentTime_ = timestepper_.end_step();
    ++steps_;
    if constexpr (Models_t::hybrid_on) // regrids: of hybrid levels only yet
        if (regrid_mode_ == RegridMode::normal and tagging_ and steps_ % regrid_interval_ == 0)
            regrid_(currentTime_);
    return dt_;
}

// SAMRAI's TimeRefinementIntegrator::atRegridPoint / coarserLevelRegridsToo: level ln regrids its
//  finer levels after every regrid_interval_-th of its steps within the coarser level's step
//  (level 0: its steps), if it can have a finer level, unless that is its last substep and the
//  coarser level regrids then too (which regrids them all)
template<auto opts>
void Simulator<opts>::after_step_(int const ln, double const t, bool const last)
{
    if (regrid_mode_ != RegridMode::high)
        return;
    auto const steps = [&](int const l) { // level 0: this step not counted yet (advance())
        return l == 0 ? steps_ + 1 : level_steps_.at(static_cast<std::size_t>(l));
    };
    auto const at_regrid_point = [&](int const l) {
        return l + 1 < hierarchy_.max_levels and steps(l) > 0 and steps(l) % regrid_interval_ == 0;
    };
    if (!at_regrid_point(ln) or (ln > 0 and last and at_regrid_point(ln - 1)))
        return;
    if constexpr (Models_t::hybrid_on) // regrids: of hybrid levels only yet
        regrid_after_step_(ln, t);
}

template<auto opts>
void Simulator<opts>::regrid_after_step_(int const ln, double const t)
{
    if (tagging_)
        regrid_(t, ln, /*always=*/true);
    else if (steps_ == 0) // refinement boxes: level 0's first step only
        for (int fine = ln + 1; fine < static_cast<int>(levels_.size()); ++fine)
        {
            std::vector<Box_t> boxes;
            for (auto const& p : hierarchy_.levels[fine].patches)
                boxes.push_back(p.box);
            regrid_level_(fine, boxes, t, /*keep_finer=*/true);
        }
}

template<auto opts>
void Simulator<opts>::advance_level_(int const level_number, double const t, double const dt)
{
    auto const ln = static_cast<std::size_t>(level_number);
    level_steps_.resize(std::max(level_steps_.size(), ln + 1));
    ++level_steps_[ln];
    step_model_(ln, [&](auto& model) { solver::advance(model, level_services_(ln), t, dt); });
    if (ln + 1 == levels_.size()) // the finest; coarser ones after their synchronization
        fine_dump_(ln, t + dt);

    // the finest level's share of the coarser level's reflux, per substep: the time average of
    //  E over them (coarser finer levels accumulate theirs when synchronized)
    if (ln > 0 and ln + 1 == levels_.size())
        step_model_(ln, [&](auto&) {
            solver::hybrid::accumulate_flux_sum(level_services_(ln), 1. / time_ratio());
        });
}

template<auto opts>
void Simulator<opts>::prepare_step_(int const ln, double)
{
    auto const l = static_cast<std::size_t>(ln);
    step_model_(l, [&](auto& model) { solver::hybrid::prepare_step(model, level_services_(l)); });
}

// the coarser level is at coarse_t + coarse_dt
template<auto opts>
void Simulator<opts>::first_step_(int const level_number, double, double const coarse_t,
                                  double const coarse_dt)
{
    auto const ln = static_cast<std::size_t>(level_number);
    level_steps_.resize(std::max(level_steps_.size(), ln + 1));
    level_steps_[ln] = 0; // SAMRAI counts a level's steps within the coarser level's step
    auto& l          = levels_[ln];
    l.coarse_t       = coarse_t;
    l.coarse_dt      = coarse_dt;
    step_model_(ln, [&](auto& model) { solver::hybrid::first_step(model, level_services_(ln)); });
}

template<auto opts>
void Simulator<opts>::last_step_(int const ln)
{
    auto const l = static_cast<std::size_t>(ln);
    step_model_(l, [&](auto& model) { solver::hybrid::last_step(model, level_services_(l)); });
}

template<auto opts>
void Simulator<opts>::synchronize_(int const ln, double const t)
{
    auto const l = static_cast<std::size_t>(ln);
    step_model_(l, [&](auto& model) { solver::hybrid::synchronize(model, level_services_(l), t); });
    fine_dump_(static_cast<std::size_t>(ln) - 1, t);
}

} // namespace plamn

#endif // PLAMN_SIMULATOR_SIMULATOR_ADVANCE_HPP
