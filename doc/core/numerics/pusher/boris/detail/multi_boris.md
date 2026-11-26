# MultiBoris: the Boris push over every patch of a level

`inc/plamn/core/numerics/pusher/boris/detail/multi_boris.hpp` pushes the particles of every
patch of a level, given a model accessor. It has two modes:

- **REF**: the particles are pushed in place.
- **COPY**: copies are pushed and deposited; the particles stay as they were.

## Dispatch

| Type | Dispatch |
|---|---|
| `core::MultiBoris` | damn's thread pools. With `-DDAMN_THREAD_POOL_SYNC=1` they are synchronous, on the calling thread |
| `core::mkn_xyz::MultiBoris` | mkn.gpu's `ThreadedStreamLauncher`: a host thread and a stream per patch, with the same per patch steps and functors, plus the GPU kernel paths. `PLAMN_STREAM_THREADS` sets the host threads (default 1; none when the pools are synchronous) |

The per patch steps take `in`, either MultiBoris; the GPU paths need `mkn_xyz::MultiBoris`
(`in.streamer`).

### Thread pools: two levels of parallelism

`move_pooled` assigns each patch one pool (round-robin over whichever is free next), and
within that pool every tile of the patch is its own task. With N pools of M threads, up to N
patches and N * M tiles are in flight at once.

All the dispatch happens from the calling thread: `move_rest_pooled` and `move_cpu_copy_pooled`
must never be called from inside a task already running on the pool they submit to, since
`detach_task` followed by a wait on one's own pool deadlocks.

In COPY mode, the Domain and LevelGhost passes of one tile write the same tile's rhoP and flux,
so they run in one task; different tiles are independent.

## Backends per layout

| Layout | Backend |
|---|---|
| AoSPCTS | tiled: an AoS per cell per tile. The GPU COPY goes through `move_rest`: only flat tiles have a push and deposit GPU kernel |
| AoSCMTS | tiled: flat, cell-mapped |
| AoSTS, AoSMapped | tiled: flat |
| AoS | untiled and untracked: leavers are partitioned out after the push (`post_move_sync`). On GPU_UNIFIED SoAVX, REF and COPY run on the device, a thread per particle, with global atomics |
| SoAVX | as AoS: leavers are sorted or partitioned out after the push; the AoS backend branches where SoAVX cannot be pushed in place |
| SoAVXChunked | as SoAVX, the offsets rebuilt after the post push sync (`on_moved`); on the CPU its own batched patch loop (below), on the GPU the flat backend's |
| AoS4SoAVXChunked | staged through SoAVX arrays (below) |

The per tile functors (`MultiBorisFunctors`) are shared by the CPU and the GPU and by every
dispatch. The per tile containers differ per layout, so each is handled by its own
`requires(layout)` overload.

## SoAVXChunked on the CPU: batched in phases

The CPU patch loop works in phases over a block of particles (load and half step; m2p,
accelerate and half step; deposit or store back), rather than the flat backend's load, push and
deposit per particle, which is built for a GPU thread per particle. It handles the same
particles in the same order, so the results are the same.

### batched: lanes in plain loops

Blocks of `lanes` (8) consecutive slots, each component a `[lane]` array loaded from and stored
back to its column, the arithmetic in lane loops.

- A block always has `lanes` lanes, so the trip counts are fixed and the loops vectorise alike;
  a partial block's extra lanes copy its last slot and are not live.
- **REF**: pushed in place. A level ghost leaving the ghost box on the first half step is stored
  half pushed and not accelerated, as `push_particle` does; `post_move_sync` drops it.
- **COPY**: the copies are pushed and deposited, in slot order, if they end in the non level
  ghost box.
- Gap slots (weight 0) are computed with the others, and neither stored nor deposited.
- m2p uses the cell's stencils for the lanes still in the cell at mid step, otherwise per lane:
  the stencils differ.

### batched_avx: mkn::avx lanes

With `-w mkn.avx`, `batched_avx` uses the widest double lanes the build has:

- the columns are loaded and stored masked: lanes past the block's end and gap slots load
  weight 0, are never live, and their memory is untouched;
- advance and accelerate run on the lanes;
- sorted domain arrays use the per cell gather (`CellGatherV`);
- per lane: the cells (int columns), the level ghost box test, m2p off the cell, and the
  deposit.

`boris::avx::Push` pushes one block of `L` slots:

- **REF**: stored back, the cell crossings recorded (`on_crossed(slot, old cell)`) for the
  chunks' resettle.
- **COPY**: deposited through the cell's window (`CellDepositV`: the lanes ending the push in the
  cell, when the cell is in the non level ghost box), else per lane.
- The gather `g` is the cell's `CellGatherV`, or null for m2p per lane.

Everything a block works on is its own (lane state, constants, the columns' pointers), so it can
stay in registers. The block is inlined into each call site, where `g` and the window are known
to be null or not; otherwise gcc keeps one copy that tests them, about 9% slower on the coupled
case.

## AoS4SoAVXChunked: staged pushes

An AoS sorted by cell (`particles_aos4soavx_chunked.hpp`), pushed as SoAVXChunked's
`batched_avx`:

- a cell's run of particles (contiguous) is staged into a SoAVXArray, `stage_size` slots at a
  time, transposed 8 at a time;
- pushed there (`boris::avx::Push`: the cell's gather loaded once, its deposit window);
- written back: in REF the live slots, with the crossings recorded against the array's slots;
  the gaps' slots stay unchanged.

Unsorted arrays, or interpolation orders above 1, stage the whole array without the gather.
Without mkn.avx, or outside host memory, the AoS backend's per particle push is used.
`PLAMN_AVX_STAGE` sets `stage_size`: a multiple of the lanes, 256 by default.

## GPU COPY

Every patch's boxings are copied into managed memory as span sets (`gpu_nlgb`,
`gpu_non_owned`), read by the GPU COPY kernels per patch index, unless the boxings are a
level's kept `LevelBoxings`, which are borrowed: then nothing is built or allocated per push.

Flat tiles launch `Interpolating::on_tiles_push_deposit` with grid `x` = tile and `y` = blocks
per tile, enough for the fullest tile at `gpu_copy_threads` (128) threads per block. Several
blocks per tile beat one, even with the staged modes' reduction into the tile (measured: 6.8 ms
against 60 ms for 10 tiles of 100k particles).

Level ghosts are deposited as domain particles: each is held by one patch only.
