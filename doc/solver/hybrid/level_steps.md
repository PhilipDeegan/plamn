# A hybrid level and its coarser hybrid level

`inc/plamn/solver/hybrid/level_steps.hpp` holds what happens between a hybrid level and its
coarser hybrid level around the finer level's substeps, as PHARE's
`HybridHybridMessengerStrategy`. damn's `damn::amr::advance` hooks call them.

## Hooks

| Hook | When | What |
|---|---|---|
| `prepare_step` | a step's start | old E and B = the current ones (the finer level's time interpolation, reflux), in one level copy |
| `first_step` | the coarser level is at the end of its step | the finer level's new level ghost particles, split from the coarser level's domain particles; the reflux sums zeroed |
| `last_step` | the finer level reached the coarser one | the new level ghost particles become the old ones |
| `synchronize` | both levels at t | the finer level onto the coarser one (below) |

Level ghost particles live in one patch, their cell's owner: after a split, the other patches'
copies are dropped (`own_level_ghosts`).

## Synchronization

1. E and the total ion charge density and flux are coarsened onto the coarser level (as PHARE's
   synchronize: `chargeDensitySynchronizers_`, `ionChargeFluxSynchronizers_`). The populations'
   moments, the mass density and the bulk velocity stay the coarser level's own.
2. The closure's part (polytropic: Pe).
3. Reflux: the coarser `Eavg` under the fine level becomes the fine level's time average of E
   (`EM_fluxSum_E`), and the coarser level's B is recomputed from its old B by Faraday over its
   step; then its B ghosts are refilled.
4. A coarser level that is itself refined accumulates its share of its own coarser level's
   reflux.
5. The coarser level's E and moment ghosts are refilled after the coarsening.

The finest level accumulates its share of the coarser level's reflux per substep
(`accumulate_flux_sum`): the time average of E over its substeps. Coarser finer levels
accumulate theirs when synchronized.

## Level services

Besides those listed in [advance.md](advance.md):

| Call | What |
|---|---|
| `refine(op, region, pairs)` | from the coarser level into this one |
| `coarsen(op, pairs)` | from this level onto the coarser one |
| `coarser()` | the coarser level's services |
| `coarse_dt()` | the coarser level's step this level subcycles in |
| `time_ratio()` | the coarser level's dt over a new level's first substep |
| `substep_weight()` | this level's last step's share of its coarser level's step |

Initialization (`initialize.hpp`) also uses `check_set(key, where)` (debug: throws if a ghost of
`key` was not filled) and `mesh_size()`.
