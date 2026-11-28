# Decomposition log

`inc/plamn/simulator/simulator_decomposition.hpp` records what the running hierarchy's
decomposition costs, for load balancing work.

`simulation/AMR/decomposition/log` (or `PLAMN_DECOMPOSITION`) names a file, truncated at setup,
to which rank 0 writes one JSON line per event.

Writing a line is collective: the weights are summed over the ranks, so every rank calls it at
the same points.

## Events

| Event | When |
|---|---|
| `init` | after initialize (regrids while initializing make no line of their own) |
| `regrid` | after a regrid that changed some level's boxes or owners |
| `rebalance` | after a rebalance that changed them |
| `sample` | every `simulation/AMR/decomposition/every` (`PLAMN_DECOMPOSITION_EVERY`) steps, changed or not, since the particles move |

## A line

The event, `t`, the step, the hierarchy's frame (domain, periodic, ratio, dl, interp, reach) and
each rank's node; `busy_steps`, the steps since the last line (0 for a line within a step); then
per level:

| Key | What |
|---|---|
| `patches` | `[lower, upper, rank]` each: the decomposition itself |
| `weights` | each patch's particles (cells for an MHD level, or with load balancing `homogeneous`) |
| `metrics` | `damn::amr::metrics` with those weights (below) |
| `busy` | hybrid levels: seconds of work per patch (below), summed over the ranks; `busy_rank`: each rank's |
| `migration` | `damn::amr::migration` from the previous line's decomposition (moved and refined cells by Distance), when there is one |

### metrics

Weight per rank, imbalance, and the cells exchanged by Distance within `reach` (the field ghost
width):

- `ghost_cells`: same level, patch ghosts;
- `coarse_cells`: the coarser cells under each grown patch;
- `ghost_box_cells`: every patch's grown box less the patch. Level ghosts and physical boundary
  cells are `ghost_box_cells` less `ghost_cells`.

### busy

Seconds of work in the level's advance since the last line, timed in `per_patch_`, so outside
the exchanges; the particle push is included only with a per patch mover (`busy_push`). The time
is attributed to the patches of the previous line's level, the decomposition that did the work;
there is none if the level is new.

## Costs

The costs (critical path, distance weighted exchanges, migration) are derived from the log in
python (`plamn.amr.read_decomposition`), so their per cell costs can change without a build.
