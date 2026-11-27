# Device deposit bench

`etc/bench/particles/bench_tile_deposit.cpp` times the deposit of one SoAVXChunked GPU_UNIFIED
patch's domain particles (density and the 3 flux components) in several ways, each checked
against the simulator's deposit (max relative difference).

```
mkn build -p bench_tile_deposit -dKO -g 0 -x res/mkn/clang_cuda [-a -DPLAMN_BENCH_INTERP=2]
PLAMN_CELLS=20 ./bin/bench_tile_deposit/plamn
```

## Variants

"Global atomics" add into the patch fields in device memory; "block atomics" add into the
block's dynamic shared memory (`extern __shared__`, sized by the launch's `ds`).

| Variant | Work | A particle's contribution summed in | Atomics |
|---|---|---|---|
| `untiled` (the simulator's) | a thread per particle | nothing: straight into the fields | global, 4 fields x stencil nodes per particle |
| tiled `global` | a block per (tile, split), threads over each cell's particles | nothing, in tile order | global, as untiled |
| `smem_all` / `smem_one` | as tiled `global` | the tile's footprint in shared memory, all fields / one per pass | block per particle, then global per staged node |
| `warp_win` | a warp per window of w^dim cells, its cells in turn | interp 1: registers, then the warp's own slice of shared memory (w > 1) or the fields (w = 1); interp > 1: the warp's slice | interp 1: plain adds into the slice, global per node; interp > 1: block among the warp's lanes, then global per node |
| `window_rs` | a block per window, a warp per cell | registers, then the window's nodes in shared memory | block per (cell, node, field), then global per node |

SoAVXChunked is sorted by cell (per cell `starts` / `sizes`), so a window or tile is a set of
cell ranges of the untiled array: no tiled layout needed. The simulator sorts after every push
(the mover's sync, `on_moved`), so the deposit after it sees valid chunks; a deposit without the
sync must check `sorted()`.

## Why block atomics lose

A `double` (also `float`, `std::uint64_t`) `atomicAdd` into shared memory compiles to a compare
and swap loop on sm_80 to sm_120 (CUDA 13.5: `ATOMS.CAST.SPIN.64` in `cuobjdump -sass`); into
global memory it is a native add done by the L2 (`ATOM.E.ADD.F64`). Threads adding into the same
shared node retry until their swap wins: with a block's threads on one cell's nodes, the
`smem_*` kernels executed 27 to 32 times the untiled kernel's instructions (ncu, 20^3), the SMs
busy spinning, memory idle. The untiled kernel instead waits on the queue of global atomics
(`lg_throttle` stalls): one per particle, field and node.

So what pays is fewer atomics, not cheaper ones.

## Fewer atomics: a cell's sum in registers

At interp 1 every particle of a cell deposits on the same 2^dim nodes (the cell's corners), so a
warp can take a cell: each lane sums its share of the cell's particles into V = 4 x 2^dim values
(fields x nodes, 32 in 3D) in registers, then the warp combines the 32 lanes' values and issues
V atomics for the whole cell, not V per particle (100 times fewer at 100 particles per cell). Two
ways to combine:

### Butterfly (all-reduce)

For each of the V values, 5 shuffle steps (`__shfl_down_sync` or `__shfl_xor_sync` with offsets
16, 8, 4, 2, 1) add the 32 lanes' copies together:

```
4 lanes, 1 value (a, b, c, d):   step 2: a+c  b+d  c+a  d+b
                                 step 1: a+b+c+d on every lane (xor) / on lane 0 (down)
```

Cost: V x 5 shuffles (160 in 3D) and, with every sum on lane 0, V atomics issued one after
another by that lane.

### Reduce-scatter (recursive halving)

The V values are split between lanes as they are added: at the step of h (16, 8, 4, 2, 1 for
V = 32), lane L and lane L xor h each keep one half of the values they hold (the half whose index
has L's bit h) and send the other half to the partner, adding what they receive:

```
4 lanes, 4 values (lane l holds a_l b_l c_l d_l):
  step h=2: lanes 0,1 keep a b, lanes 2,3 keep c d, each adding its partner's (l xor 2) copy
            lane 0: a0+a2  b0+b2     lane 2: c2+c0  d2+d0
            lane 1: a1+a3  b1+b3     lane 3: c3+c1  d3+d1
  step h=1: lanes 0,2 keep the first, lanes 1,3 the second, adding their partner's (l xor 1)
            lane 0: a (all 4 lanes)  lane 1: b   lane 2: c   lane 3: d
```

Each step sends half of what is left: V/2 + V/4 + ... + 1 = V - 1 shuffles (31 in 3D), and lane
L ends with value L's total, so the V lanes issue their atomics at once. With fewer values than
lanes (V = 8 in 1D, 16 in 2D) the halving stops after log2(V) steps and the lanes holding partial
sums of the same value finish with an all-reduce (the remaining 5 - log2(V) butterfly steps on one
value).

### Which

| | Butterfly | Reduce-scatter |
|---|---|---|
| shuffles for V values over 32 lanes | 5 V | V - 1 |
| result | every value on every lane (xor) or lane 0 (down) | value L on lane L |
| atomics after | V by one lane, in turn | V lanes, one each, at once |
| needs | nothing | V a power of 2; the halving's indices known at compile time (an unrolled loop: dynamic indices into a register array spill it to local memory) |

Reduce-scatter when each value is written out on its own (an atomic per value): less shuffling
and the writes in parallel. Butterfly when every lane needs every total (a value broadcast to the
warp), or for a single value (V = 1: both are the same 5 steps). With more values than lanes
(V > 32), the halving runs out of lane bits after 5 steps: each lane then holds V / 32 totals,
after V - V / 32 shuffles, and issues V / 32 atomics.

## Results

RTX 5080 laptop (sm_120, 60 SMs, 99 KB shared memory a block opted in), 3D, 100 particles per
cell, the fastest of 5 runs, 256 threads a block (each kernel clamped to its register limit,
`mkn::gpu::getDeviceKernelMaxThreadsPerBlock`). Ratios are against the untiled deposit of the
same run.

Interp 1:

| | 12^3 | 20^3 |
|---|---|---|
| untiled | 0.21 ms | 0.96 ms |
| tiled `global`, tile 4, 8 splits | 1.02 | 0.73 |
| `smem_all` / `smem_one`, best | 4.1 / 4.5 | 4.1 / 4.1 |
| butterfly, a warp a cell (replaced by `warp_win` 1) | 0.49 | 0.42 |
| `warp_win` 1 (reduce-scatter) | **0.28** | **0.24** |
| `warp_win` 2, 3, 4 | 0.54, 1.71, 3.99 | 0.34, 0.36, 0.88 |
| `window_rs` 2, 4 | 0.29, 0.54 | 0.24, 0.34 |

A warp's window is done cell after cell: larger windows have fewer warps working (27 blocks at
12^3, w = 2), and at interp 1 they save nothing the registers did not already.

Interp 2 (`warp_win` stages in the warp's slice, see below for the register version):

| | 12^3 | 20^3 |
|---|---|---|
| untiled | 0.73 ms | 5.5 ms |
| `warp_win` 1, 2 | 1.48, 2.80 | 0.82, 1.09 |
| `smem_one`, tile 4 | 1.49 | 0.91 |
| tiled `global`, tile 4 | **0.75** | **0.63** |

Staged in shared memory, the warp's own slice included, interp 2 is back to compare and swap
atomics; the best so far is the deposit in tile order with global atomics.

### Register sums at higher orders

A cell's particles still share a fixed set of (primal) nodes at every order
(`computeStartLeftShift`, interpolator.hpp), per direction:

| interp | start | a particle's nodes | the cell's nodes |
|---|---|---|---|
| 1 | iCell | 2 | iCell, iCell + 1: 2 |
| 2 | iCell - 1 if delta < .5, else iCell | 3 | iCell - 1 to iCell + 2: 4 (a particle adds 0 on the others) |
| 3 | iCell - 1 | 4 | iCell - 1 to iCell + 2: 4 |

so the reduce-scatter applies, over V = 4 fields x nodes values a cell:

| | 1D | 2D | 3D |
|---|---|---|---|
| interp 1 | 8 | 16 | 32 |
| interp 2, 3 | 16 | 64 | 256 |

256 doubles are 512 registers a lane, past the 255 limit: in passes (8 of 32 values, a field
and half its nodes each), each pass reading the cell's particles again (L1 resident) and
recomputing their weights, for 256 global atomics a cell instead of 4 x 27 (interp 2) or
4 x 64 (interp 3) a particle.
