# Live data from python: the DataWrangler

`src/python3/plamn_data_wrangler.hpp` gives python live, zero-copy access to a running
simulator's data, with the API of pyphare's live `DataWrangler`, so that pyphare's
`hierarchy_from_sim` works unchanged.

## API

### `DataWrangler(simulator, hierarchy)`

`hierarchy` is unused: plamn's hierarchy is the simulator's own.

| Method | What |
|---|---|
| `getNumberOfLevels()` | |
| `getHybridPatchLevel(lvl)` | a `HybridPatchLevel` |
| `sync(patch_datas)` | every rank's patch datas on rank 0, as owned copies in rank order (to plot the whole domain, for example); other ranks get none. On one rank: the input |

### `HybridPatchLevel`

Each getter returns one `PatchData` per local patch; every quantity written as a diagnostic has
one.

| Getter | Quantity |
|---|---|
| `getB(c)`, `getE(c)` | the electromagnetic field's component `c` |
| `getNi()` | the ions' charge density |
| `getN(pop)` | a population's charge density |
| `getDensity(pop)` | a population's particle density |
| `getVi(c)` | the ions' bulk velocity |
| `getFlux(c, pop)` | a population's flux |
| `getMassDensity()`, `getChargeFlux(c)` | the ions' totals |
| `getMomentumTensor(c, pop = "")` | computed on the call, so collective: every rank calls it. `c` is one of `xx xy xz yy yz zz`; `pop` `""` means all ions |
| `getParticles(pop)` | a population's domain particles |

### `PatchData`

`patchID`, `origin`, `lower`, `upper` (AMR cells), `nGhosts` (field or particle ghost width),
`data`.

## Lifetime

`data` is a writable numpy view on the patch's grid, and particles are the patch's domain
particle array, by reference: both are only valid while the simulator and its patches exist.

## Tiled layouts

Tiles count as patches: a tiled field gives a patch data per tile, each with its own contiguous
data, layout and ghosts (their domains do not overlap); a tiled particle array gives one per
tile, each its own array, live.

## Particle arrays

A particle array's python class has `size`, `__len__`, `__getitem__` and `components()`:

- `__getitem__` gives a reference into the array, or a proxy by value where the layout holds no
  `Particle` to refer to (SoAVX's zip particle, thrust builds);
- `components()` gives `{name: numpy array}` for every component (`weight`, `iCell_x..`,
  `delta_x..`, `v_x..`): live, writable views in the array's own memory where it has columns;
  copies, tile by tile, for a tiled array (it has no contiguous columns).
