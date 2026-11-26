# Cell gather and deposit on lanes

`inc/plamn/core/numerics/interpolator/cell_gather.hpp` interpolates fields to particles and
deposits particles' moments at interpolation order 1, cell by cell, with the particles of one cell
on SIMD lanes. Results follow `Interpolator`'s: the gather gives the same sums, the deposit the
same products.

## Gather: CellGather1

For the particles of one cell, the E and B values that any particle of the cell can need are
loaded once per cell: per direction, the union of the stencils.

| Centering along a direction | Nodes, relative to the cell |
|---|---|
| primal | `cell`, `cell + 1` |
| dual | `cell - 1`, `cell`, `cell + 1` |

The union is 3 nodes a side; a primal component uses 2 of them.

Per lane, the weights are those of `Interpolator`'s `Weighter<1>` and `computeStartLeftShift`,
written as dense rows over the union. A dual direction's unused node gets weight 0, first or
last in its sum, so the sums are `Interpolator::m2p`'s. They are summed as
`MeshToParticle<dim>::op` does: x outer, the last direction inner.

A lane's particle must be in the cell: its delta in `[0, 1)` of it.

### Branch free dual weights

`computeStartLeftShift<dual>` becomes a factor `sf` of 1 or 0, and the selects become products
by 1 or 0. The values are the same (`x * 0 + y == y`) and the loop has no branch, so it
vectorises. `sf - .5` is exact, as in `indexAndWeights__`.

### Vectorisation

In `value_` the lane loop is outermost and the nodes' loops are unrolled inside it (compile time
trip counts), so the compiler vectorises over the lanes. With the lane loop innermost, gcc
vectorised the 2 or 3 nodes instead, 2 wide.

`CellGatherV` is the same on `mkn::avx` lanes (built with `-w mkn.avx`): the same union and
weights (the dual ones by select, exact) and the same sums per lane.

## Deposit: CellDepositV

Per node of the cell's primal stencil (the cell and `+ 1` per direction), density and flux
accumulate on lanes; they are summed over the lanes and added to the moments once (`flush`).

The weights come from the deltas alone, as in `Interpolator::particleToMesh` (`Weighter<1>`,
primal), with its products in its order (weight, x, y, z). The sums per node differ from its
particle by particle order.

Sorted arrays are deposited cell by cell:

- **SoAVXChunked**: each chunk in blocks of `L` slots, with masked loads (past the chunk's end:
  weight 0). Every particle is in its chunk's cell.
- **AoS4SoAVXChunked**: a block of a chunk's particle structures is staged into lanes (past the
  chunk's end: weight 0, not live).
