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

## Sorted SoAVXChunked on the device: a warp per cell

`warp_cell_deposit` (interp 1, GPU_UNIFIED). A thread per particle issues 4 x 2^dim global
atomics per particle. Particles sorted by cell put threads adding into the same nodes next to
each other, and the kernel waits on the queue of same address atomics. It is slower on sorted
particles than on unsorted ones (1.0 ms against 0.77 ms, 20^3 cells, 100 per cell).

At interp 1 every particle of a cell deposits on the same 2^dim nodes, the cell's corners. So a
warp takes a cell of the chunks' shell:

1. Each lane sums its share of the cell's particles into V = 4 fields x 2^dim nodes values (8, 16,
   32 in 1D, 2D, 3D), in registers. Every index is known at compile time: a run time index into
   a register array would put it in local memory.
2. The warp reduce-scatters the V values (recursive halving). At the step of h (16, 8, 4, 2, 1),
   lane L and lane L xor h each keep the half of their values whose index has L's bit h, and add
   in what the partner sends of it. After V - 1 shuffles lane L holds value L's total. A
   butterfly all-reduce would take 5 V shuffles and leave the atomics to one lane.
3. Lane L adds value L to its field and node: V atomics a cell, not V a particle.

Shared memory staging would not help: a `double` atomicAdd into shared memory is a compare and
swap loop on sm_80 to sm_120 (CUDA 13.5), which is slower than the global atomic it replaces.

A particle whose cell is not its chunk's (pushed in place since the sort, no sync yet) is
deposited on its own, as the thread per particle kernel does.

The warps of empty cells still launch, so sparse arrays (patch ghosts, a few particles per cell)
keep the thread per particle kernel. `warp_cell_min_ppc`, 16 particles per shell cell on
average (`PLAMN_GPU_WARP_CELL_MIN_PPC` overrides it; test_layouts forces 0), is set from the
crossover (~25 per cell, 3D, RTX 5080):

| particles a cell (20^3) | 4 | 10 | 25 | 50 | 100 |
|---|---|---|---|---|---|
| a warp per cell, ms | 0.15 | 0.13 | 0.27 | 0.14 | 0.28 |
| a thread per particle, ms | 0.04 | 0.05 | 0.19 | 0.60 | 1.06 |

bench_mover, 3D: push+sync+dep minus push+sync, the same build for both.
