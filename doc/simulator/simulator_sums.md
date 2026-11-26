# Debug sums

`inc/plamn/simulator/simulator_sums.hpp` is heavy debugging, compiled only with
`-DDAMN_SUPER_DEBUG` (the calls sit in `DAMN_SUPER_DEBUG_DO`).

## `sums_(ln, stage, t)`

Per level and stage, every resource of the level's model summed, whatever the patches:

- fields over the level's nodes, each once (`dom`: the patch owning a node, its cells' range plus
  the level's last node row), and over the level ghost nodes, each once (`gh`: the first patch in
  box order holding it);
- particle arrays as counts.

One line each:

```
SUMS L<ln> t=<t> <stage> <name> dom <sum> gh <sum>
```

The hybrid model's resources carry PHARE's names (`Bx`, `PredBx`, `AvgEx`, `rhoC`, ...), as the
PHARE debug branch's `SolverPPC::debug_sums_` prints them, so the two runs' lines diff.

The sums are bit equal only on one rank, where they run in node order. With more ranks, each rank
prints its own (`R<rank>`).

## `debug_regrid_(ln, t)`

`PLAMN_DEBUG_REGRID_TIMES="t0,t1,..."`: the finest level, just synchronized at one of those
times, is regridded onto its own boxes, as SAMRAI does with refinement boxes. This isolates regrid
bugs.

## `rebalance_state_()`, `check_rebalance_(before, t)`

A rebalance moves data. These check that, after it, every level has the same:

- E, B and moments on its domain nodes;
- domain and old level ghost particles: count and sums of weights, velocities and positions
  (order independent, over every rank).

The moments are deposited again, so they and the sums agree to round off. The check throws naming
what changed.
