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
#include "plamn/core/numerics/stable_dt/mhd.hpp"
#include "plamn/core/numerics/stable_dt/particles.hpp"
#include "plamn/core/numerics/stable_dt/whistler.hpp"
#include "plamn/core/numerics/particles_synchronised_mover/particles_synchronised_mover_def.hpp"
#include "plamn/solver/advance.hpp"
#include "plamn/solver/models.hpp"

#include "phlop/timing/scope.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <iomanip>
#include <optional>
#include <sstream>
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
void Simulator<opts>::step_model_(std::size_t const ln, auto&& hybrid, auto&& mhd)
{
    models_.visit(ln, [&](auto& model) {
        if constexpr (solver::is_hybrid_model_v<std::remove_cvref_t<decltype(model)>>)
            hybrid(model);
        else
            mhd(model, MHDLevel{*this, ln});
    });
}

// this rank's bound of level ln, by its model: hybrid the whistler bound, MHD fast + Hall and the
//  diffusive one (-1: a non-finite MHD wave speed, logged)
template<auto opts>
double Simulator<opts>::level_bound_(std::size_t const ln, Timestepper::CFL const& cfl)
{
    double level = std::numeric_limits<double>::max();
    models_.visit(ln, [&](auto& model) {
        if constexpr (solver::is_mhd_model_v<std::remove_cvref_t<decltype(model)>>)
        {
            core::MHDStableDt const in{.gamma         = model.gamma,
                                       .resistivity   = model.resistivity,
                                       .hall          = model.hall,
                                       .cfl_wave      = cfl.wave,
                                       .cfl_diffusive = cfl.diffusive};
            for (auto& patch : model.levels[static_cast<int>(ln)].data)
            {
                auto const& u = patch->states[0];
                level
                    = std::min(level, core::mhd_dt(patch->layout, u.rho, u.rhoV, u.B, u.Etot, in));
            }
        }
        else
            per_patch_(ln, [&](auto& patch) {
                level = std::min(
                    level, core::whistler_dt(patch.layout, patch.massDensity, patch.emB, cfl.wave));
                if (cfl.particles > 0) // no domain particle further than cfl * the cell jump
                    for (auto const& pop : patch.populations)
                        level = std::min(
                            level, core::particles_dt(patch.layout.dx, pop->domain, cfl.particles));
            });
    });
    return level;
}

// a level's stable dt over the ranks, with the adaptive run stops: a non-finite wave speed, level
//  0's dt more than level0_dt_shrink_max times under the run's first one, a finer level's more
//  than fine_level_substeps_max times under its coarser level's last
template<auto opts>
double Simulator<opts>::level_stable_dt_(std::size_t const ln, double const t)
{
    auto const local = level_bound_(ln, timestepper_.cfl());
    auto const dt    = n_ranks_ > 1 ? damn::mpi::min(local) : local;
    auto const at    = [&] {
        std::ostringstream ss;
        ss << std::setprecision(17) << " on level " << ln << " at t = " << t;
        return ss.str();
    };
    if (!(dt > 0)) // every rank: the first such cell logged by the rank that has it
        throw std::runtime_error("plamn: non-finite wave speed" + at());
    if (level_dts_.size() <= ln)
        level_dts_.resize(ln + 1, -1);
    if (ln == 0)
    {
        if (first_level0_dt_ < 0)
            first_level0_dt_ = dt;
        else if (!(first_level0_dt_ <= timestepper_.level0_dt_shrink_max() * dt))
        {
            std::ostringstream ss;
            ss << std::setprecision(17) << "plamn: stable dt" << at() << " is " << dt
               << ", more than level0_dt_shrink_max = " << timestepper_.level0_dt_shrink_max()
               << " times smaller than the run's first (" << first_level0_dt_ << ")";
            throw std::runtime_error(ss.str());
        }
    }
    else if (auto const coarser = level_dts_[ln - 1];
             !(coarser <= timestepper_.fine_level_substeps_max() * dt))
    {
        std::ostringstream ss;
        ss << std::setprecision(17) << "plamn: stable dt" << at() << " is " << dt
           << ", more than fine_level_substeps_max = " << timestepper_.fine_level_substeps_max()
           << " times smaller than on level " << ln - 1 << " (" << coarser << ")";
        throw std::runtime_error(ss.str());
    }
    return level_dts_[ln] = dt;
}

// the adaptive dt of a step: level 0's stable dt (its finer levels step at their own, next_dt_)
template<auto opts>
double Simulator<opts>::stable_dt_(Timestepper::CFL const&)
{
    return level_stable_dt_(0, currentTime_);
}

template<auto opts>
double Simulator<opts>::next_dt_(int const level_number, damn::amr::Step const& step)
{
    auto const ln = static_cast<std::size_t>(level_number);
    if (!timestepper_.adaptive())
    {
        if (level_dts_.empty()) // the table, from the input's dt: dt / time_ratio per level
        {
            level_dts_.push_back(timestepper_.dt());
            for (int l = 1; l < hierarchy_.max_levels; ++l)
                level_dts_.push_back(level_dts_.back() / time_ratio());
        }
        return level_dts_.at(ln);
    }
    if (ln == 0)
        return step.dt;
    return std::min(step.coarse_dt, level_stable_dt_(ln, step.t + step.dt));
}

template<auto opts>
double Simulator<opts>::substep_weight_(std::size_t const ln) const
{
    auto const& step = level_step_.at(ln);
    return step.coarse_dt > 0 ? step.dt / step.coarse_dt : 0.;
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
    if (!(dt_ > 0))
    {
        std::ostringstream ss;
        ss << std::setprecision(17) << "plamn: zero-length step on level 0 at t = " << currentTime_;
        throw std::runtime_error(ss.str());
    }
    if (!stepping_) // the substeps' split (SAMRAI's, as with PHARE's integrator: grow_dt 1e3)
    {
        stepping_.emplace(damn::amr::TimeRefinementOptions{
            .time_ratio = time_ratio(), .regrid_interval = regrid_interval_, .grow_dt = 1e3});
        if (restored_stepping_) // a restart: the split continues as it was
        {
            stepping_->restore(*restored_stepping_);
            restored_stepping_.reset();
        }
    }
    substep_dts_.clear();
    Hooks hooks{*this};
    stepping_->advance(hierarchy_, hooks, currentTime_, dt_);
    currentTime_ = timestepper_.end_step();
    ++steps_;
    if (regrid_mode_ == RegridMode::normal and tagging_ and steps_ % regrid_interval_ == 0)
        regrid_(currentTime_);
    if (decomposition_.every > 0 and steps_ % decomposition_.every == 0)
        record_decomposition_("sample", currentTime_, /*always=*/true);
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
    DAMN_SUPER_DEBUG_DO(for (auto l = static_cast<std::size_t>(ln); l < levels_.size(); ++l)
                            sums_(l, "regridded", t);)
}

template<auto opts>
void Simulator<opts>::advance_level_(int const level_number, damn::amr::Step const& step)
{
    auto const ln = static_cast<std::size_t>(level_number);
    auto const t  = step.t;
    auto const dt = step.dt;
    level_step_.resize(std::max(level_step_.size(), ln + 1));
    level_step_[ln] = step;
    substep_dts_.resize(std::max(substep_dts_.size(), ln + 1));
    substep_dts_[ln].push_back(dt);
    level_steps_.resize(std::max(level_steps_.size(), ln + 1));
    ++level_steps_[ln];
    DAMN_SUPER_DEBUG_DO(sums_(ln, "start", t);)
    models_.visit(ln, [&](auto& model) {
        if constexpr (solver::is_mhd_model_v<std::remove_cvref_t<decltype(model)>>)
        {
            MHDLevel level{*this, ln};
            solver::advance(model, level, t, t + dt);
        }
        else
        {
            decomposition_.timing = !decomposition_.path.empty();
            solver::advance(model, level_services_(ln), t, dt);
            decomposition_.timing = false;
        }
    });
    DAMN_SUPER_DEBUG_DO(sums_(ln, "end", t);)
    if (ln + 1 == levels_.size()) // the finest; coarser ones after their synchronization
        fine_dump_(ln, t + dt);

    // the finest level's share of the coarser level's reflux, per substep, weighted by its share
    //  of the coarser level's step (coarser finer levels accumulate theirs when synchronized)
    if (ln > 0 and ln + 1 == levels_.size())
        step_model_(
            ln,
            [&](auto&) {
                solver::hybrid::accumulate_flux_sum(level_services_(ln), substep_weight_(ln));
            },
            [&](auto& model, auto level) {
                solver::mhd::accumulate_flux_sum(model, level, substep_weight_(ln));
            });
}

template<auto opts>
void Simulator<opts>::prepare_step_(int const ln, double const t)
{
    auto const l     = static_cast<std::size_t>(ln);
    levels_[l].old_t = t;
    models_.visit(l, [&](auto& model) {
        if constexpr (solver::is_hybrid_model_v<std::remove_cvref_t<decltype(model)>>)
            solver::hybrid::prepare_step(model, level_services_(l));
        else if (ln + 1 < hierarchy_.max_levels)
            // the state at the step's start (PHARE's MHDMessenger prepareStep), every point: the
            //  reflux redoes the step from it, so only a level that can have a finer one (one
            //  made by a regrid after this is still refluxed at the step's end)
            MHDLevel{*this, l}.per_patch([&](auto& patch) {
                auto& u       = patch.states[0];
                auto const cp = [](auto const& from, auto& to) { to.copyData(from); };
                cp(u.rho, patch.old.rho);
                cp(u.Etot, patch.old.Etot);
                for (std::size_t c = 0; c < 3; ++c)
                {
                    cp(u.rhoV.grids[c], patch.old.rhoV.grids[c]);
                    cp(u.B.grids[c], patch.old.B.grids[c]);
                }
                simulator::keep_E_old(patch); // under a hybrid level: its E ghosts in time
            });
    });
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
    step_model_(
        ln,
        [&](auto& model) {
            if (!models_.is_mhd(ln - 1))
                return solver::hybrid::first_step(model, level_services_(ln));
            // over MHD: the new level ghosts spawned from its state at the step's end
            level_services_(ln).per_patch([](auto& patch) {
                for (auto& pop : patch.ions)
                    pop.levelGhostParticlesNew().clear();
                for (auto& e : patch.fluxSumE)
                    e.zero();
            });
            mhd_prims_over_(ln, /*interior=*/false);
            spawn_level_ghosts_(ln, /*new_ghosts=*/true,
                                simulator::coupling::Salt::ghosts_new(coarse_t + coarse_dt));
        },
        [](auto& model, auto level) { solver::mhd::first_step(model, level); });
}

template<auto opts>
void Simulator<opts>::last_step_(int const ln)
{
    auto const l = static_cast<std::size_t>(ln);
    step_model_(
        l, [&](auto& model) { solver::hybrid::last_step(model, level_services_(l)); },
        [](auto&, auto) {}); // MHD: nothing (PHARE's MHDMessenger lastStep)
}

template<auto opts>
void Simulator<opts>::synchronize_(int const ln, double const t)
{
    auto const l = static_cast<std::size_t>(ln);
    step_model_(
        l,
        [&](auto& model) {
            if (models_.is_mhd(l - 1))
                return; // hybrid -> MHD (sync and reflux at the model boundary): not yet
            solver::hybrid::synchronize(model, level_services_(l), t);
        },
        [&](auto& model, auto level) { solver::mhd::synchronize(model, level, t); });
    DAMN_SUPER_DEBUG_DO(sums_(l - 1, "synchronized", t); sums_(l, "synchronized", t);)
    fine_dump_(static_cast<std::size_t>(ln) - 1, t);
}

} // namespace plamn

#endif // PLAMN_SIMULATOR_SIMULATOR_ADVANCE_HPP
