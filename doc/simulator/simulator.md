# Simulator

`inc/plamn/simulator/simulator.hpp`: `plamn::Simulator` is PHARE's simulator shell (dict
parsing, times, `initialize` / `advance`) over damn. Its definitions are split by concern into
`simulator_*.hpp`, included at the end of `simulator.hpp`.

## Models and levels

The models a build enables (`solver/models.hpp`) own the levels: MHD below `max_mhd_level`,
hybrid above. The simulator's own types and members are model free; a model's paths compile only
where the build enables it.

- Level 0 is the domain chopped into patches, partitioned over MPI's ranks by the particles they
  will hold (`damn::amr::partition`).
- Finer levels come from the input's refinement boxes (pyphare's `refinement="boxes"`), fixed for
  the run, or from tagging (`refinement="tagging"`), regridded as the run goes (below).
- Each level's data is a `LevelData` of the model's patches (`HybridPatch`, `MHDPatch`).
  Communication is damn::mpi's, or damn::amr's on one rank.

The `Level` struct holds what is model free: how a level's data moves (neighbours, the map to the
coarser level, the level ghost cells) and the coarser step it subcycles in.

## Initialize

`initialize()` follows PHARE's `HybridLevelInitializer`:

1. the root level from the user's functions;
2. each finer level refined from the coarser one: fields, B made divergence free, particles
   split.

## Advance

`advance()` is `damn::amr::advance` (time refinement, `doc/amr/time_refinement.md` in damn). Each
level takes its model's step, `solver::advance` (hybrid: PHARE's `SolverPPC`,
`solver/hybrid/advance.hpp`). A finer level takes `ratio^2` steps per coarser one, since whistler
waves need `dt ~ dx^2` (PHARE's `getMaxFinerLevelDt`); `PLAMN_TIME_RATIO` overrides that for
experiments.

The coupling between levels follows PHARE's `MultiPhysicsIntegrator` and
`HybridHybridMessengerStrategy`.

### Level ghosts

Level ghosts are the fine ghosts no fine patch covers.

- E and B are refined from the coarser level, linear in time between its start of step
  (`EM_*_old`) and its end (`EM_*`). B is then made divergence free.
- The moments come from the deposit of level ghost particles, split from the coarser level at
  both ends of its step, weighted linearly in time.
- A level ghost particle lives in one patch, the owner of its cell: the other patches' copies of a
  split are dropped.

### Synchronize

After a finer level's substeps:

1. the fine level's E and moments are coarsened onto the coarser level;
2. the coarser level recomputes B from the time average of the fine E over its substeps (reflux).

## Regrid modes

When levels are regridded, by `simulation/AMR/regrid_mode` or `PLAMN_REGRID_MODE`:

| Mode | Tagging | Refinement boxes |
|---|---|---|
| `normal` (plamn's) | every `regrid_interval_` level 0 steps, a level only where its boxes changed | never |
| `high` (PHARE's) | re-clustered and re-made even with unchanged boxes | re-made with the same boxes, during level 0's first step only |

`high` follows SAMRAI's `TimeRefinementIntegrator` and `GriddingAlgorithm`: after every
`regrid_interval_`-th substep of a level within its coarser level's step, its finer levels are
re-made, unless the coarser level regrids then too. SAMRAI marks the user's refinement boxes
changed for its first cycle, hence the first step regrids.

`high` is for comparisons with PHARE. There a regrid is not a no-op: the level ghost fields are
refilled after the synchronization.

## Load balancing: partitions

How a level's patches get their ranks (`level_of_`). These are plamn's; the PHARE comparisons
keep `levels`.

| Partition | Ranks |
|---|---|
| `levels` | each level along the Hilbert curve on its own |
| `keep` | where the data is (`damn::amr::keep`): the old level's owners, else the coarser patch's under it, else the curve |
| `finest` / `aligned` | every level together (`damn::amr::decompose`) where all are known: initialization with pretag or tag_boxes, a rebalance. A level alone goes to the owners of the coarser cells under it |

| Setting | Input | Environment |
|---|---|---|
| partition | `simulation/AMR/loadbalancing/partition` | `PLAMN_PARTITION` |
| spill | `simulation/AMR/loadbalancing/spill` | `PLAMN_SPILL` |
| near band | `simulation/AMR/loadbalancing/near_band` | `PLAMN_NEAR_BAND` |

- `spill >= 0`: after partitioning, no rank holds more than `(1 + spill)` of its share
  (`damn::amr::spill`).
- `near_band`, with `finest`: the level 0 cells about level 1 that its step reads go with level
  1's owners. The default is a particle ghost width, where level 1's level ghost particles split
  from.

## Threading over patches

`per_patch_threaded_` hands patches to damn's thread pools:

- untiled layouts: a patch to the next idle pool, round robin; a pool works one patch at a time;
- tiled layouts: the threads work over tiles inside the kernels.

The function must touch only its patch and never call python. It runs serially with one pool,
and while scope timing is on: phlop's timer is serial, scopes on one thread a rank.

## Tiled fields and leases

On tiled fields, damn's schedules (copies, refine, coarsen, regrid) work on untiled temporaries
(`amr/untiled.hpp`), leased on every patch of a level for the names an operation touches:

| Lease | For |
|---|---|
| `read` | names an operation reads |
| `fill` | names it writes |
| `sum` | names it sums borders into |

`fill` and `sum` leases are written back into every tile's ghost box when released, inner tile
ghosts included.

A `Leases` releases its leases when the operation ends, an exception included. The destination
lease is taken first, so where a name is both read and written, the write wins.

## Rank logs and timing

`std::cout` goes to a file per rank as in PHARE's simulator, chosen by `PHARE_LOG`:

| `PHARE_LOG` | Output |
|---|---|
| `RANK_FILES` | `.log/<rank>.out`; pyphare sets it (`log_to_file`, its default) after making `.log/` |
| `NULL` | discarded |
| `CLI`, or unset | the terminal |

With `-DDAMN_LOG_SCOPE_PRINT` (`mkn.py --scope-print`, a build without `NDEBUG`) every scope's
name is printed: a rank stuck in a collective is the one whose file stops.

`PHARE_SCOPE_TIMING=1` at setup turns on phlop's `DAMN_LOG_SCOPE` timers, a file per rank under
`.phare/timings/`, written when the simulator goes (pyphare's reset; collective). Read them with
`python3 -m phlop.timing.scope_timer`.
