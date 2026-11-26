# Interpolating: deposits on tiles and on the device

`inc/plamn/core/numerics/interpolator/interpolating.hpp` launches `Interpolator`'s deposits over
particle arrays: on the host, per tile, or in GPU kernels.

## GPU tile deposits: TileDeposit

A per tile GPU deposit kernel accumulates rhoP and the 3 flux components, all with atomics, in
one of three modes:

| Mode | Staging | Passes over the particles |
|---|---|---|
| `shared_all` | all four tile fields in dynamic shared memory | one |
| `shared_one` | one field at a time | four (a pushing kernel pushes again per pass) |
| `global` | none: straight into the tile's fields in global memory | one |

`tile_deposit_mode` picks the fewest passes that fit in the dynamic shared memory available. For
benchmarking, `PLAMN_GPU_TILE_DEPOSIT=shared_all|shared_one|global` forces a mode; it throws if
the forced mode does not fit.

With several blocks per tile (`gridDim.y > 1`), the staged sums are added to the tile's fields
atomically.

## on_tiles: warps per tile

A block deposits one tile. Each group of `blockDim.x / warps` threads (a warp each, launched as
`warp_size() * warps`, see `DepositSpec`) takes a contiguous share of the tile's particles.
Sorted particles keep each group on its own cells, so the groups do not contend for the same
staged nodes. The group width comes from the launch rather than `warpSize`, so it is correct
whatever width the host launched with.

## on_tiles_push_deposit: push and deposit in one kernel

The Boris push and the deposit run in a single kernel. The originals are untouched: each thread
pushes a register copy.

The grid is `x` = tile, `y` = blocks per tile; each block takes a strided share of the tile's
particles, in batches over all the blocks of the tile.

Particles that qualify after the push (in `filter_box` and in none of `exclude_boxes`) are
placed atomically into a block buffer in shared memory. Once the buffer holds a full block, the
whole block deposits together and the buffer is compacted; the remainder is deposited at the
end. With `shared_one`, the push runs once per staged field.

### Shared memory layout

`launcher.ds = tile_deposit_bytes(deposit_v, ziz, push_deposit_extra_bytes<Particle_t>())`:

| Bytes | Content |
|---|---|
| `tile_deposit_fields(deposit_v) * ziz * sizeof(double)` | staged fields |
| `2 * blockDim.x * sizeof(Particle_t)` | block buffer |
| `sizeof(int)` | atomic fill counter |

`2 * blockDim.x` slots are the minimum needed: before a batch the fill count can be
`blockDim.x - 1`, and if every thread of the batch qualifies it reaches `2 * blockDim.x - 1`,
still within bounds.
