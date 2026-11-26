# Device appends of particles

`inc/plamn/amr/device_append.hpp` implements damn's device `Append` hooks
(`damn/amr/device_copy.hpp`) for `SoAVXChunked` particles in unified memory (GPU builds with
thrust). A copy's pieces, all of a resource pair's (every patch of the level), are appended in a
few launches, and the host never touches the particles.

## Steps

| Step | What |
|---|---|
| 1 | a thread per (piece, cell) of the pieces' boxes, in their sources' cells: that cell's particle count |
| 2 | an exclusive scan of the counts (thrust); each piece's total is read back (one value per piece) |
| 3 | the destinations are resized; their columns are copied on the device (`mkn::gpu::resize`) |
| 4 | a thread per particle: its cell is found by binary search over the scan, and it is copied to its piece's place in its destination (pieces in plan order), with `iCell + shift` |

## Requirements

- Sources must be sorted: their chunks give a cell's particles. Otherwise
  (`can_append_items_on_device` returns false) the host path runs.
- The destinations come out unsorted; damn sorts each one after the copy (`items_appended`).

## Buffers

The counts, the scan and the per piece totals live in thread local buffers kept between copies,
so nothing is allocated once they are large enough.
