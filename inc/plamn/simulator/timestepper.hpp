#ifndef PLAMN_SIMULATOR_TIMESTEPPER_HPP
#define PLAMN_SIMULATOR_TIMESTEPPER_HPP

// the simulation's time steps, from what pyphare's timestepper.py writes in the dict:
// simulation/time_step/{mode, value, cfl_wave, cfl_diffusive, cfl_particles,
// fine_level_substeps_max, level0_dt_shrink_max}, simulation/final_time (cfl_particles: plamn's,
// see core/numerics/stable_dt/particles.hpp; plamn/__init__.py lets pyphare pass it)
//
//   constant  the dict's dt; the time after n steps is start + n * dt (PHARE's
//             ConstantTimeStamper: no rounding accumulated over the steps)
//   adaptive  each step's dt from a stable bound (the caller's, given the cfl numbers), no
//             further than final_time; the time a Kahan sum of the steps (PHARE's
//             KahanTimeStamper)
//
// start is 0, or a restart's time: the steps are counted from it, so a restart may change dt
// (or the mode). A restart of the same mode (and, constant, dt) resumes the stepper as it was
// (state / resume: its start, steps and Kahan compensation): the times those of the run that kept
// going, bit for bit

#include "damn/core/def.hpp"
#include "plamn/initializer/data_provider.hpp"
#include "plamn/simulator/simulator_def.hpp"
#include "dict.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace plamn
{
class Timestepper
{
public:
    struct CFL
    {
        double wave      = 0;
        double diffusive = 0;
        double particles = 0.5; // a hybrid level's domain particles: this share of a cell jump
    };

    // dict: the whole dict (its "simulation" node)
    MKN_KUL_FN_SIG_RET static Timestepper FROM(initializer::Dict const& dict)
    {
        Timestepper ts;
        ts.mode_ = cppdict::get_value(dict, "simulation/time_step/mode", TimeStepType::constant);
        ts.dt_   = cppdict::get_value(dict, "simulation/time_step/value", 0.);
        ts.final_time_    = dict["simulation"]["final_time"].template to<double>();
        ts.cfl_.wave      = cppdict::get_value(dict, "simulation/time_step/cfl_wave", 0.);
        ts.cfl_.diffusive = cppdict::get_value(dict, "simulation/time_step/cfl_diffusive", 0.);
        ts.cfl_.particles = cppdict::get_value(dict, "simulation/time_step/cfl_particles", 0.5);
        ts.fine_level_substeps_max_
            = cppdict::get_value(dict, "simulation/time_step/fine_level_substeps_max", 1000);
        ts.level0_dt_shrink_max_
            = cppdict::get_value(dict, "simulation/time_step/level0_dt_shrink_max", 1000);
        if (ts.adaptive() and ts.cfl_.wave <= 0)
            throw std::runtime_error("plamn Timestepper: adaptive needs cfl_wave > 0");
        return ts;
    }

    // what a restart keeps: the mode and dt it was for, its start, steps, Kahan compensation
    struct State
    {
        int mode            = 0;
        double dt           = 0;
        double start        = 0;
        std::uint64_t steps = 0;
        double compensation = 0;
    };
    MKN_KUL_FN_SIG_RET State state() const
    {
        return {.mode         = static_cast<int>(mode_),
                .dt           = dt_,
                .start        = start_,
                .steps        = static_cast<std::uint64_t>(steps_),
                .compensation = compensation_};
    }
    // a restart at t: the steps resumed if the mode is the same (constant: and dt), else from t
    void resume(State const& s, double const t)
    {
        if (s.mode != static_cast<int>(mode_) or (!adaptive() and s.dt != dt_))
            return start_at(t);
        start_        = s.start;
        time_         = t;
        steps_        = static_cast<std::size_t>(s.steps);
        compensation_ = s.compensation;
    }

    // steps counted from t (the start, or a restart's time)
    void start_at(double const t)
    {
        start_        = t;
        time_         = t;
        steps_        = 0;
        compensation_ = 0;
    }

    MKN_KUL_FN_SIG_RET bool adaptive() const { return mode_ == TimeStepType::adaptive; }
    MKN_KUL_FN_SIG_RET double dt() const { return dt_; }
    MKN_KUL_FN_SIG_RET double time() const { return time_; }
    MKN_KUL_FN_SIG_RET double final_time() const { return final_time_; }
    MKN_KUL_FN_SIG_RET CFL const& cfl() const { return cfl_; }
    // adaptive run stops: a finer level's stable dt more than this many times smaller than its
    //  coarser level's; level 0's more than that many times smaller than the run's first
    MKN_KUL_FN_SIG_RET int fine_level_substeps_max() const { return fine_level_substeps_max_; }
    MKN_KUL_FN_SIG_RET int level0_dt_shrink_max() const { return level0_dt_shrink_max_; }

    // the next step's dt: constant, the dict's; adaptive, stable(cfl) up to final_time
    double next(auto&& stable)
    {
        if (adaptive())
            dt_ = std::min(stable(cfl_), final_time_ - time_);
        return dt_;
    }

    // a step of dt() done: the time after it
    double end_step() { return adaptive() ? adaptive_() : constant_(); }

private:
    double constant_() { return time_ = start_ + static_cast<double>(++steps_) * dt_; }

    double adaptive_()
    {
        ++steps_;
        double const y    = dt_ - compensation_;
        double const next = time_ + y;
        compensation_     = (next - time_) - y;
        return time_      = next;
    }

    TimeStepType mode_   = TimeStepType::constant;
    double dt_           = 0;
    double final_time_   = 0;
    CFL cfl_             = {};
    int fine_level_substeps_max_ = 1000;
    int level0_dt_shrink_max_    = 1000;
    double start_        = 0;
    double time_         = 0;
    std::size_t steps_   = 0;
    double compensation_ = 0; // Kahan: the low order bits the sum lost
};

} // namespace plamn

#endif // PLAMN_SIMULATOR_TIMESTEPPER_HPP
