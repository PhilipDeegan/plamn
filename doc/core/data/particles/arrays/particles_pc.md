# Per cell particles

`inc/plamn/core/data/particles/arrays/particles_pc.hpp` stores particles in one container per
cell (`PerCellVector`, viewed by `PerCellSpan`). Iteration and `size()` go through `cell_size_`,
not each cell container's own `size()`, so every bit of per cell bookkeeping must be kept in step
with the containers (`clear()` resets it as the constructor does).

## Moving particles between cells

A push changes particles' cells in place. `move_check(pt, idx, particle)` only registers the move:
the particle already carries its new cell (set by the caller), `pt` carries the old one, so the
departure is registered against the old cell. Per cell:

| Counter | Meaning |
|---|---|
| `gaps_(c)`, `gap_idx_(c)` | indexes of the particles leaving `c`, and their count |
| `add_into_(c)` | arrivals expected in `c` |
| `left_(c)` | departures handled, and the cursor for arrivals assigned into `c` |

`on_moved()` then applies the registered moves:

1. `sync_moved`: each cell reserved and resized for its arrivals, ahead of the copies. Copies
   transiently exceed the final size, before the removals. The pre-move size is recorded so freshly
   reset views start the copy step from it.
2. `sync_add_new`: the movers assigned into their new cells, into the slots opened by the resize,
   cursored by `left_`. Never emplaced, so the per cell sizes stay stable for the removal pass. A
   mover whose new cell is outside `ghost_box()` (a level ghost leaving) has no destination: the
   removal pass deletes it from its old cell.
3. `sync_rm_left`: the leavers removed from their old cell by swap-pop (the last particle into the
   hole), in descending index order, so the indexes still to remove stay valid.
4. `on_appended()`: recount, gap sizing, views reset. It is also what a raw append (without
   `move_check`) calls.

## `add_into_` as the arrival signal

`move_check` increments `add_into_` at a destination only when it intends the particle to land
there; a departure handed off to another tile is counted at that tile instead.

A geometric check (`isIn(ghost_box())`) cannot tell "stays local" from "crossed into a sibling
tile's domain that overlaps this tile's ghost reach": both are within `ghost_box()`. So
`sync_add_new` trusts `add_into_` over geometry, decrementing it once per accepted arrival; a mover
with no expected arrival is only counted as left. `sync_moved` leaves `add_into_` as it is for
that reason, and `sync_rm_left` resets it to 0 for the next cycle.

## Merged removals

`sync_rm_left` can also remove an externally registered, ascending sorted list of leavers (the
tile set's cross-tile ones, see [particles_pc_ts.md](particles_pc_ts.md)) in the same descending
swap-pop. Two separate passes would move particles the other list still indexes.

## Device

`sync()` walks every cell on the host for `CPU`; for `GPU_UNIFIED` it launches one kernel thread
per cell, on the given stream if one is passed. The per cell `sync_add_new` / `sync_rm_left`
overloads are the kernels' entry points, the cell from `damn::kernel::idx()`.
