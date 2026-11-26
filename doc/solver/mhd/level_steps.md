# An MHD level and its coarser level

`inc/plamn/solver/mhd/level_steps.hpp` holds an MHD level's steps with its coarser level, as
PHARE's `SolverMHD` and `MultiPhysicsIntegrator` for an MHD pair. PHARE's `MHDMessenger`
synchronize does nothing there: the reflux is all of it.

| Step | What |
|---|---|
| `first_step(model, level)` | at the level's first substep in the coarser level's step: its flux sums zeroed (PHARE's `resetFluxSum`) |
| `accumulate_flux_sum(model, level, c)` | flux sums `+= c *` the step's fluxes and E (the scheme's Butcher sums), at every point (`accumulateFluxSum`) |
| `synchronize(model, level, t)` | the reflux, below |

## Reflux

1. The coarser level's step fluxes and E under the level are replaced by the level's sums
   (`MHDFluxCoarsener`, `ElectricFieldCoarsener`), and their patch ghosts refilled.
2. The coarser level's step is redone from its old state with them, to t: its patches are seen
   with the states `{old, states[0]}`, so the step runs from `old` into `states[0]`.
3. If the coarser level has a coarser level of its own, its sum is accumulated.

## Level services

`level` is the simulator's MHD level (`Simulator::MHDLevel`): `per_patch`, `fill_state_ghosts`,
`fill_electric_ghosts`, `number`, `coarser`, `coarsen`, `copy`, `old_time`, `time_ratio`, and
`substep_weight` (its last step's share of its coarser level's step).
