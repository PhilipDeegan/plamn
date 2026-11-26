# Particles synchronised mover

`inc/plamn/core/numerics/particles_synchronised_mover/particles_synchronised_mover.hpp` (with its
definitions in `particles_synchronised_mover_def.hpp`) pushes a level's ion populations, places
the pushed particles in their arrays (domain, patch ghosts, level ghosts) and deposits their
moments.

## What a mover call does: MoveSpec

A mover call is `update<spec>`, with a `MoveSpec`:

| Field | Meaning |
|---|---|
| `copy` | push copies: the particles stay at time n, the copies are deposited, nothing to sync |
| `deposit` | deposit the pushed particles in the mover |
| `sync` | after an in place push (the particles to n+1), place them with `post_move_sync` |

- `UpdaterMode::domain_only` is `copy_spec`.
- `UpdaterMode::all` is each mover's `all_spec`. On the CPU it pushes and syncs, and the domain
  deposit comes after the particle exchange (`deposit_domain`). The device mover deposits too.
- The other combinations time the steps on their own (`etc/bench/particles/bench_mover.cpp`).
  Without the sync, the arrays stay as the push left them.

`copy` requires `deposit` and no `sync` (`check_move_spec`).

## Particle fates after a push

After a push, a particle's fate depends only on its cell (`PostMoveSyncCells`):

| Particle | Cell | Fate |
|---|---|---|
| domain | owned (in the domain, not in a non-owned box) | stays |
| domain | a non-owned domain cell, or the patch ghost layer | patch ghost |
| domain | elsewhere | deleted |
| level ghost | owned | domain |
| level ghost | still in the level ghost layer (no patch of the level covers it) | stays |
| level ghost | another patch's cell | patch ghost |
| level ghost | outside the ghost box | deleted |

A level ghost lives in one patch only (the simulator's `own_level_ghosts_`). One that lands in a
cell another patch of the level covers is handed to that patch as a patch ghost, as a domain
particle would be: no other patch holds a copy of it to keep instead. Kept as a level ghost, it
would come back as a duplicate.

## Synchronisation per layout

- **Generic** (`post_move_sync`): whole array passes: partitions, deletions, moves by box.
- **Chunked layouts, host**: the domain's particles that changed cell go back into their
  chunks' gaps (resettle), level ghosts entering the domain into its gaps, with no sort while
  there is room. The fates are `post_move_sync`'s. An array not sorted yet, or without slack
  (which would mean a sort every push), takes the generic way.
- **SoAVXChunked, device** (`synced_on_move`): no host access to the particles. One kernel per
  array classifies its slots by cell, appends the movers to their destination array (in no
  particular order) and sets the weight of moved or deleted slots to 0, leaving gaps the
  following sort (`on_moved`) drops.
- **Tiled layouts**: a tile only holds particles whose cell it owns (clamped), and
  `emplace_back` routes by that same owner, so tile `ti` of each array is only ever written
  from tile `ti`. Distinct tiles can be synced concurrently, then `post_move_sync_finalize`
  runs once per patch. For AoSCMTS, only the cells of a tile's ghost layer outside the patch
  domain and its non-owned domain cells are visited, through the tile's cellmap; level ghosts
  only live in patch border tiles.

Layouts that place every particle themselves after a push (`is_synced_on_move`: the push
records the moves and the layout's own post push step applies them) skip the generic
`post_move_sync` when the spec syncs. The per cell arrays do not: AoSPCTS's push moves particles
between cells but not out to the patch ghosts (`test_multi_updater`'s border moments fail
without the generic step), and AoSPC is untested.

## Deposits

`deposit_domain` resets the moments of a patch and deposits its domain particles, to be
completed by the border sums. As at level initialization, after mode all's push and the
particle exchange, every particle is on the patch owning its cell, like PHARE's `depositDomain`
after the exchange: a particle that left is deposited once, by its new patch.

The device deposit after an in place push (`Interpolating::on_tiles`) runs a block per tile of
`warps` warps (`DepositSpec`), each a contiguous share of the tile's particles: sorted, so each
warp works on its own cells. The default comes from `PLAMN_DEPOSIT_WARPS`, for a bootstrap probe
of the local GPU to set.

## domain_only: patch ghosts entering the domain (to revise)

After the domain_only update, domain particles overlapping the ghost layers of neighbour
patches are sent to those neighbours' patch ghost arrays. After being pushed, some patch ghost
particles may enter the domain: these need to be copied into the domain array, so that they are
transferred to the neighbour patch's ghost array and contribute to the moments there too.

Level ghost particles entering the domain do not need to be copied: they contribute to nodes not
shared with neighbour patches, and level border nodes receive contributions from the old and new
level ghost particles.

## Movers

| Mover | For |
|---|---|
| `ParticlesSynchronisedMover` | one patch's ions and fields, given directly (AoSMapped) |
| `ParallelParticlesSynchronisedMover` | every patch of a level in parallel through a model accessor; tiled layouts, a task per tile |
| `FlatParallelParticlesSynchronisedMover` | the same for untiled layouts: a patch is the unit of work |
| `mkn_xyz::ParallelParticlesSynchronisedMover` | unified memory layouts, tiled or not: the steps dispatched with mkn.gpu's `ThreadedStreamLauncher` (one host thread and stream per patch), GPU deposit kernels |

`ParticlesSynchronisedMoverFor_t` picks one per particle layout.
