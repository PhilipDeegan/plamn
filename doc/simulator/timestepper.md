# Time stepper

`inc/plamn/simulator/timestepper.hpp` gives the simulation's time steps, from what pyphare's
`timestepper.py` writes in the dict: `simulation/time_step/{mode, value, cfl_wave,
cfl_diffusive, cfl_particles, fine_level_substeps_max, level0_dt_shrink_max}` and
`simulation/final_time`.

`cfl_particles` is plamn's own (`core/numerics/stable_dt/particles.hpp`); `plamn/__init__.py`
lets pyphare pass it.

## Modes

| Mode | dt | Time after n steps |
|---|---|---|
| `constant` | the dict's | `start + n * dt`, as PHARE's `ConstantTimeStamper`: no rounding accumulated over the steps |
| `adaptive` | from a stable bound (the caller's, given the cfl numbers), never past `final_time` | a Kahan sum of the steps, as PHARE's `KahanTimeStamper` |

## Adaptive run stops

The adaptive run throws when:

- a wave speed is not finite;
- a finer level's stable dt is more than `fine_level_substeps_max` times smaller than its coarser
  level's;
- level 0's dt is more than `level0_dt_shrink_max` times smaller than the run's first.

## Restarts

`start` is 0, or a restart's time: the steps are counted from it, so a restart may change dt or
the mode.

A restart of the same mode (and, for `constant`, the same dt) resumes the stepper as it was
(`state` / `resume`: its start, steps and Kahan compensation), so its times are those of the run
that kept going, bit for bit.
