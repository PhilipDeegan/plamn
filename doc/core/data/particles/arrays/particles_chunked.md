# Chunked particles

`inc/plamn/core/data/particles/arrays/particles_chunked.hpp` keeps particles sorted by cell, in
per cell chunks. It is the bookkeeping, whatever the storage: columns (`SoAVXChunked`) or particle
structures (`AoS4SoAVXChunked`).

## Layout

`starts_[c]` is the first particle of cell `c` and `sizes_[c]` its count. Cells are flattened in
C order (last direction fastest) over `shell() = grow(box, 1)`, the safe outer shell, as for
`AoSMapped`'s `CellMap`.

A span of cells along the last direction is then one contiguous range of particles, so selecting
or exporting a box is a list of ranges (`Chunks::for_each_range`).

Chunks are in cell order with gaps between them. A gap's slots hold fillers: weight 0, with the
cell of the chunk before them.

## Sorting

`sort()` lays the chunks out with a counting sort:

1. a histogram, giving each particle a slot in its cell;
2. an exclusive scan over each cell's particles and gap, giving the chunks' starts;
3. one scatter of all particles into a second storage, which is swapped in.

The gaps are the slack: a fraction of the particles, spread over the patch's cells. Border cells
are weighted (`border_weight_`, 3 by default): their particles leave to and arrive from other
patches, while inside a cell about as many particles enter as leave. The shell outside the box
gets no slack. Without slack (the default) the chunks are contiguous.

`compact()` sorts without gaps, e.g. before writing the particles out; the next `sort()` lays the
gaps out again.

## Changing cells without a sort

Between sorts, particles change cells through `resettle()`, `insert()` and `remove()`:

- a chunk grows into the gap at its tail or its head, whichever is larger;
- it shrinks from the end with the smaller gap, so the gaps either side even out;
- an empty chunk takes the middle of the gap it is in;
- a chunk with no room on either side has a neighbour slide over by one into its other gap. If
  that neighbour has no room either, its own neighbour slides first, up to `max_slides` chunks
  each way.

Only when none of them has room is a full sort needed.

A push of a sorted array records the particles that crossed cells (`begin_crossers`,
`add_crosser`): `resettle()` then takes those instead of scanning every chunk.

## Staleness

Any other mutation (push, append, erase, resize) leaves the chunks stale: `sorted()` is false until
the next `sort()`, and the range accessors assert `sorted()`.

## Device

`GPU_UNIFIED` sorts on the device, without gaps (the slack serves the host's `resettle()`):
kernels on the default stream (zero the sizes, count, `mkn::gpu::exclusive_scan` for the starts,
scatter), the host reads only the total (the scan's one wait), and a sync at the end leaves the
array ready for the host. thrust's algorithms each synchronized the stream, and its scan alone
took ~45-60 us whatever the size (RTX 5080; mkn.gpu's: 12-18 us). The order of particles within a
cell is not deterministic (slots come from atomics); the CPU sort is stable.

## Storage interface

`ChunkedOps<Derived>` holds the algorithms. `Derived` (the particles), with a `ChunkState` in its
storage, provides the storage's part:

| Member | |
|---|---|
| `size()`, `copy(i)`, `push_back(p)` | as any particles |
| `weight_at_(i)`, `set_weight_(i, w)` | a slot's weight (0: a gap) |
| `cell_at_(i)` | a slot's cell (`std::array<int, dim>`) |
| `move_slot_(to, from)`, `write_slot_(i, p)`, `write_filler_(i, cell)` | slot writes |
| `slot_weights_()`, `slot_cells_()` | `sort()`'s per slot functors (`DAMN_ALL`: device sorts) |
| `scatter_sorted_(to, n, slots_total)` | every slot `i` with `to(i) != none` into slot `to(i)` of a scratch storage of `slots_total` slots, swapped in |
