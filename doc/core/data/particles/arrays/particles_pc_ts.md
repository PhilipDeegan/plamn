# Per cell tile sets

`inc/plamn/core/data/particles/arrays/particles_pc_ts.hpp` splits a patch's particles into
threading tiles (`PCTileSetVector`, viewed by `PCTileSetSpan`), each tile holding per cell
particles ([particles_pc.md](particles_pc.md)).

## Cells a tile reaches

A tile reaches its domain box and, for `LevelGhost`, its own ghost halo. Adjacent tiles' halos
overlap, so a ghost cell is held by one tile only: its clamp owner (`TileSet::tag_cells_`).
Level ghost particles live only in their cell's clamp owner, and register their cross-tile
traffic against the cells it owns.

Only the cells within `ghost_cells_` of a tile's wall can be entered or left from another tile in
one step. `on_appended()` sizes the tile-set gap vectors of those cells, and of the clamp-owned
ghost cells, so `move_check` has room. This is regardless of the particle type: the first sync
after a fill may run as `Domain` on a level ghost array. A tile too small for an interior has
every cell as a wall cell.

## Moves

`move_check` registers departures only; the copies and removals happen during the sync:

- `Domain`: a particle whose new cell is in the patch but outside its old tile is registered
  against the tile-set cell (`gaps_`, `add_into_`). Otherwise the tile's own per cell
  `move_check` registers it, including a particle leaving the domain into the tile's ghost layer.
- `LevelGhost`: a particle whose new cell has another clamp owner is handed off to it, even into
  another tile's domain cell, and even when the new cell is still within this tile's ghost box.

`on_moved()`:

1. `sync_moved`: each tile folds the cross-tile traffic of its owned cells into its per cell
   reallocation. The bookkeeping is shared, so exactly one tile (the owner) folds each cell in.
2. per tile, the within-tile movers added, then the cross-tile leavers copied into the tile owning
   their new cell (capacity is reserved exactly by `sync_moved`). A level ghost leaving the level
   ghost box has no destination.
3. per tile, one merged removal pass over both leaver lists of a cell (see the per cell doc for
   why they cannot be two passes).
4. `on_appended()`.

On `GPU_UNIFIED`, steps 2 and 3 are two kernel launches on the stream: every tile's appends must
land before any tile removes its leavers. `__syncthreads` between the passes would only order one
block, and the tiles span several.

## Copies

Copy and move constructors call `on_appended()`; the assignments destroy and reconstruct in place
through them, as for `TileSetVector`: a defaulted memberwise copy leaves the views stale.

There is no top-level `begin()` / `end()`: the particles are tiled, iterate with `enumerate()` /
`per_particle()`.
