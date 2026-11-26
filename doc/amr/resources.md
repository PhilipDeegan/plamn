# plamn data as damn resources

`inc/plamn/amr/resources.hpp` presents plamn's fields and particle arrays as `damn::amr`
resources, for damn's schedules (copies, refines, coarsens, regrids, checkpoints). Both are small
handles over the data, made on the fly in a `for_each_resource`.

| Handle | damn kind | What |
|---|---|---|
| `field_resource<gw>(field)` | `BoxData` | values on the field's nodes (its centering), `gw` ghost nodes each side |
| `particle_resource<gw>(array)` | `CellItemData` | particles in cells, `gw` ghost cells |

## Fields

- A C ordered field (not tiled) also gives `damn::hdf5`'s checkpoints its buffer (`box_span`):
  its values over its ghost box, written and read in place.
- In unified memory, damn's same rank copies run on the device over that `box_span`.
- Checkpoints hold a field's ghosts too (`checkpoint_box`): a refined level's level ghosts were
  filled from the coarser level before its synchronization, so they cannot be rebuilt bit for
  bit on load.

## Particles

A layout with `components()` also gives checkpoints its columns (`for_each_column`,
`resize_columns`; the caller rebuilds the layout's maps after a resize).

Between patches, particles move through damn's cell_items hooks (`damn/amr/cell_items.hpp`):
`pack_items`, `unpack_items`, `append_items_from` (the same without the bytes) and `read_items`
(refine's coarse particles, to be split). Two layout families have their own hooks.

### Columns: `SoAVX`, `SoAVXChunked` (host memory)

Moved column by column. The bytes:

| Field | Size |
|---|---|
| `n`, the number of particles | `std::uint64_t` |
| each component's values, contiguous, in `components()`' order: `weight`, `iCell_x..`, `delta_x..`, `v_x`, `v_y`, `v_z` | `n` values each |

- A selection reads the `iCell` columns only, or `SoAVXChunked`'s per cell ranges when sorted.
- Appends are column wise; a `SoAVXChunked` array that is sorted with slack takes arrivals into
  its chunks' gaps, one at a time, and the rest are appended and sorted once after all of them
  (`items_appended`).
- A resource over `Particles const` (a read only source, see `particle_access.hpp`) packs too.
- `append_items_from` may have `src` be `dst`: its first particles stay, and the columns are read
  after the resize.

### Structures: `AoS4SoAVXChunked` (host memory)

Moved as whole particle structures. The bytes: `n` (`std::uint64_t`), then the `n` structures.

- A selection reads the per cell ranges when sorted.
- Arrivals go into their chunks' gaps when the destination is sorted with slack; otherwise they
  are appended, and sorted once after all of them (`items_appended`).
- `append_items_from` may have `src` be `dst`: the selection is copied first.

### Chunked layouts

`SoAVXChunked` leaves gaps in its chunks, marked by a zero weight; iteration and selection skip
them. `items_appended` sorts an array once after a copy or regrid appended to it (only if it was
appended to, not only inserted into its gaps).

## GPU

damn's device `Append` hooks for unified memory `SoAVXChunked` live in `device_append.hpp`
(see [device_append.md](device_append.md)), included at the end of `resources.hpp` since they use
the resources.
