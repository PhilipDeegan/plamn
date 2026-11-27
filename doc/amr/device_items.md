# Device items of particles

`inc/plamn/amr/device_items.hpp` gives damn's device item hooks (damn's
`doc/amr/device_items.md`) for `SoAVXChunked` particles in unified memory (GPU builds with
thrust). damn moves a copy's leaving particles on the device as columns, locally and between
ranks, the host never touching them.

| Hook | Gives |
|---|---|
| `device_items(src)` | the chunks' cells (`shell()`), their starts and sizes, and the columns weight, delta.., v.. (8 bytes each) |
| `device_items_appended(dst, n)` | `dst` grown by `n` particles on the device, the new particles' columns and their `iCell` columns |

`iCell` is not one of the columns: a particle's cell is its chunk's, so damn's messages leave it
out and write the destination's from the cell a particle arrives in.

## Requirements

- Sources must be sorted: their chunks give a cell's particles. `device_items` throws
  otherwise.
- The destinations come out unsorted; damn sorts each one after the copy (`items_appended`).
