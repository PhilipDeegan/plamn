# Electron pressure closures

`inc/plamn/core/data/electrons/closures.hpp` defines the electron pressure closures of the
hybrid model. One closure per model is chosen at run time by the input's
`electrons/pressure_closure/name`. Closures operate on whole levels: the solver has an overload
per closure (`solver/hybrid/electron/closures.hpp`). A patch holds the electrons' views (`Ne`,
`Ve`, `Pe`) and the storage of the closure's own fields (`electron_fields`: the same list on
every patch).

## Closures

| Closure | Pressure | Notes |
|---|---|---|
| `isothermal` | `Pe = Ne Te` | `Te` constant in space and time |
| `polytropic` | `Pe = s Ne^gamma` | see below |

### Polytropic

The entropy `s = Pe / Ne^gamma` is carried by the electron flow:

    ds/dt + Ve.grad s = 0

solved first order upwind, sub-stepped past Courant 1. `gamma` is the simulation's. The initial
`s` is the input's `Pe / n^gamma`, from the analytic densities, so it is free of the deposit's
noise.

Its fields, in `electron_fields` order (the index of each in a patch's closure storage, all
scalars, primal as the moments and `Pe`):

| Index | Name | Holds |
|---|---|---|
| `step` | `Se_old` | `s` at the start of a step: from `Pe / Ne^gamma` on the domain, ghosts filled |
| `subcycle` | `Se` | `s` within a stage, from `Se_old`; after a stage, `s` at its end: what a finer level's ghosts interpolate to |
| `scratch` | `Se_tmp` | a sub-step's output: the upwind stencil reads `s`'s neighbours |

`max_subcycled_courant` caps the sub-steps a stage takes; beyond it the step is refused.

The closure is disabled for now (`polytropic_enabled`): it advects `s` with `Ve` on the moment
nodes, from the moment nodes' ion flux, which the Ohm energy fix stops depositing in the time
loop. Its code is still compiled and its tests are skipped until that settles.

## Restarts

A restart keeps what a closure's next step reads first: nothing for isothermal, `Pe` for
polytropic. The first stage takes `Se_old` from `Pe`; `Se` is written by every stage, and the
coarser level's is redone before a finer level's ghosts read it.

## Coupling to MHD

`coupling_Te` gives the electron temperature that splits an MHD pressure into `Pi + Pe` at the
model boundary (`Pi = max(P - Pe, 0)`, `Pe = Te q n`, see `doc/simulator/coupling.md`). Only the
isothermal closure supports it so far.
