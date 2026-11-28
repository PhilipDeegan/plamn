# Restarts

`inc/plamn/simulator/simulator_restarts.hpp` writes and loads restarts, as pyphare's
`restart_options` declare them (`simulation/restarts` in the dict).

## When

- written by `dump_restarts` at `write_timestamps`, or once every `elapsed_timestamps` seconds of
  wall time after setup (rank 0's clock, so every rank decides the same);
- only between steps, so every level is at the same time;
- loaded by `initialize()` when `loadPath` is set.

## The file

`<filePath>/<time {:021.10f}>/restart.h5`: one file, every rank writing its own patches.

| Group | What |
|---|---|
| `/restart` | time, step, dt, levels, ranks, the serialized simulation, the rebalance state, the timestepper's and the substeps' split state |
| `/level<ln>` | a `damn::hdf5` checkpoint of `EM_B_*`, `EM_E_*`, each population's domain particles and moments |
| `/clustered<ln>` | the boxes the level was clustered to |

A restart without the timestepper or split state (an older one, or one dumped before any step)
starts those afresh from its time.

## What is stored

Only what the next step needs and cannot rebuild bit for bit:

- B and E;
- the populations' moments with their ghosts: a refined level's level ghosts were filled from the
  coarser level before its synchronization, and the moments' border sums and coarsenings come in
  an order a fresh deposit does not repeat;
- the domain particles.

The total moments, J, the old copies and the level ghost particles are rebuilt after loading.

Loading needs the ranks that wrote the restart: the same decomposition.
