# Electron pressure closures on a level

`inc/plamn/solver/hybrid/electron/closures.hpp` runs the electron pressure closures
(`core/data/electrons/closures.hpp`) on a whole level. There is an overload per closure; the
`*_level` functions `std::visit` the model's closure to pick it.

| Operation | When | What |
|---|---|---|
| `pressure(closure, level, stage)` | after the electrons' Ve, at a stage of the step (`ClosureStage`) | every patch's Pe from its Ne |
| `seed(closure, level, sim)` | a new level at t = 0, before its Pe | the closure's state from the input |
| `refine_state(closure, level, op, t)` | a level made or regridded at t | the state from the coarser level |
| `restart_loaded(closure, level)` | after a restart load (Pe stored) | the state from Pe |
| `synchronize(closure, level)` | a synchronization onto the coarser level | what the coarser level takes |

## Stages

`ClosureStage` says where in a level's step a closure is asked for Pe:

| Kind | When |
|---|---|
| `init` | a new level, dt = 0 |
| `first` | the PPC step's predictor 1: the state at t is saved |
| `middle` | predictor 2 |
| `last` | the corrector |

## Execution

On unified memory with untiled fields, each operation is one launch for the level (damn's
region chunks, the item list kept). Otherwise it runs per patch on the pool, and tiled fields are
worked on through untiled copies, written back after.

## Isothermal

`Pe = Ne Te`. The closure has no state: seeding, refining, restarts and synchronizations do
nothing (Pe comes from the level's own Ne at the next Ohm).

## Polytropic

The polytropic closure advects the electron entropy `s = Pe / Ne^gamma`
with the electrons:

1. **first stage (and init)**: `Se_old = Pe / Ne^gamma` on the domain (Pe, Ne at t). Its ghosts
   come from the level's patches and, at the level border, from the coarser level's `Se_old`
   (at `t_n`) and `Se` (at `t_n+1`) interpolated at t (alpha 1 outside a coarser step).
2. **every stage**: `Se` from `Se_old` in `n` sub-steps of `dt / n`, each solving
   `ds/dt + Ve . grad s = 0` upwind (each direction from the side the flow comes from); Se's
   ghosts are filled between sub-steps as Se_old's, at their times. Then `Pe = Se Ne^gamma` on
   the domain.
3. **last stage**: Se's ghosts at `t + dt`, which the finer level's ghosts interpolate to.

A node with `Ne <= 0` keeps its s and its Pe.

### Sub-steps

`n` is the level's largest electron Courant number `sum_d |Ve_d| dt / dx_d` (over nodes with
`Ne > 0`), rounded up. The upwind update is a convex combination (stable, monotone, positive)
only up to a Courant number of 1.

- `n` is the same on every rank (an MPI max): the ghost fills between sub-steps are collective.
- The max is taken over a double's bits as an unsigned integer: a non negative double's order
  is its bits' order, so an integer atomic max works on the device.
- Past `C::max_subcycled_courant`, a rank reports an out of range count, so every rank fails in
  the same reduction, with an error asking for a smaller time step.

### Regrids and new refined levels

A level made or regridded at t (every level synchronized) gets its state from its coarser level:
the entropy is refined, not the pressure.

- `Se_old` is computed from the coarser level's Pe and Ne at t, and from the level's own where
  it was (its Pe moved with it, its Ne deposited again).
- `op` says where it is refined: `amr::FieldRefiner` everywhere, or a `Masked` one for the new
  cells only (the others moved from the old level).
- Pe then follows from `Se_old`.

### Restarts

Pe is stored; `Se_old` and `Se` are rebuilt from it, as the first stage sets them.

### Synchronization

Pe is primal, as the moments: the coarse nodes take the fine nodes' values
(`amr::MomentsCoarsener`).
