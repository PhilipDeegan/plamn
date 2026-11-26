# A hybrid level's step

`inc/plamn/solver/hybrid/advance.hpp` advances a level of the hybrid model over one step, as
PHARE's `SolverPPC`. It also holds the steps the simulator uses to initialize and regrid a level:
the moments, the ion border sums, Ampere and Ohm.

## The step

`plamn::solver::advance(model, level, t, dt)`:

| Stage | What |
|---|---|
| predictor 1 | Faraday with `E` into `emPred.B`, its ghosts at `t + dt`; Ampere, the electrons, Ohm (closure stage `first`) |
| average | `emAvg = (em + emPred) / 2`, its E ghosts at `t + dt / 2` |
| ions moved, domain only | the pushed copies deposited in the updater; level ghosts, border sums, moments |
| predictor 2 | Faraday with `emAvg.E` into `emPred.B`; Ampere, the electrons, Ohm (closure stage `middle`) |
| average | as above |
| ions moved, all | the particles that left their patch gathered, domains deposited; level ghosts, border sums, moments |
| corrector | Faraday with `emAvg.E` into `em.B`; Ampere, the electrons, Ohm (closure stage `last`); `em.E` ghosts at `t + dt` |

### Ampere, the electrons, Ohm

`ampere_ohm` computes J from B (Ampere) on the physical box + 1, which is what Ohm reads, so J
is never exchanged. Then the electrons from the ions and J (Ve, then Pe through the model's
closure: `pressure_level`), then E from Ohm's law. The `electromag` template argument is the
`HybridPatch` member holding the B read and the E written: the model's (`em`) or the predicted
one (`emPred`).

### Moving the ions and depositing

As PHARE's deposit after the exchange (and the level initialization):

1. the particles are pushed: per patch with a thread local copy of the mover (it keeps push
   state: mesh, dt, a buffer), or the level's patches at once, each with its boxing by patch id
   (the boxings, and their device box lists, kept per decomposition);
2. except for the domain only predictor (which deposits its pushed copies in the updater): the
   particles that left their patch go to the patch owning their cell (`gather_leaving`), then
   each patch deposits its domain particles, so every particle is deposited once;
3. the level border's moments: the level ghost particles split from the coarser level at both
   ends of its step are deposited, weighted linearly in time (`1 - alpha` and `alpha`, as PHARE's
   `fillIonPopMomentGhosts`). Each level ghost particle is held by one patch and deposited once;
4. the border sums complete every patch's copy of a shared node (`IonPopBorderSum`);
5. the ions' totals from the populations' moments.

The border sums are the first exchange after the push: time waiting there is mostly other ranks'
longer pushes.

## Level services

`level` is the simulator's services for one level (`Simulator::LevelServices`):

| Call | What |
|---|---|
| `number()` | the level number |
| `data()` | its patch data (`LevelData` of `HybridPatch`) |
| `per_patch_threaded(fn)` | `fn(patch)` for every local patch, on the pool's threads (`fn` touches its patch only, never python) |
| `copy(how, key)` | damn's copy of a resource key between its patches |
| `fill_magnetic_ghosts(key, t)`, `fill_electric_ghosts(key, t)` | patch ghosts, then level ghosts from the coarser level at `t` |
| `alpha(t)` | `t` within the coarser level's step, in `[0, 1]` |

## Host and device

Each level operation has an overload per memory kind:

| Operation | Unified memory (GPU) | Host memory, or tiled fields |
|---|---|---|
| ion totals (`compute_moments`) | a few kernels for the whole level: zero, each population, then the velocity's division (`Ions::update`'s sums, in its order) | `Ions::update` per patch |
| Ampere | one launch for the level, its item list kept (an item per patch, or per tile) | per patch (host memory only) |
| electrons' Ve | one launch for the level (untiled) | per patch |
| Faraday | one launch for the level (untiled) | per patch |
| Ohm | one launch for the level (untiled) | per patch (tiled: the transformer's tiles path) |
| average | one kernel for the whole level | per patch |
| domain deposits | the moments zeroed in one kernel, then the device deposits | per patch |

The item lists are `static thread_local` and cleared per call, so nothing is allocated once warm.
