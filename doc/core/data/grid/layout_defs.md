# Recentering and prolongation stencils

`inc/plamn/core/data/grid/layout_defs.hpp` builds compile time stencils (`WeightPoint` lists)
per direction, combined across directions by `tensorProduct`.

## Dual-direction prolongation

`directionalProlongation<dim, dir, sign, order>` refines, coarse to fine with ratio 2, a quantity
that is dual in `dir`.

Such a quantity is an average over its dual extent in `dir`. Refining splits coarse cell `I` into
two fine children sitting at +-1/4 of the coarse spacing. With the child's parity `p` in {0, 1},
its sign is `s = 2p - 1` (`s = +1` the right child, `s = -1` the left). Offsets are relative to
coarse cell `I`, and `u_I` is the coarse cell's mean.

| Order | Stencil | Children |
|---|---|---|
| 2 (linear, 3 points) | `I - 1`, `I`, `I + 1` | `u_I + s (u_{I+1} - u_{I-1}) / 8` |

### Why only odd orders

Subtracting the conservation constraint cancels every even reconstruction coefficient, so the
children depend only on the odd coefficients: a parabolic reconstruction gives the same children
as the linear one, and degree 2 is skipped.

The children are antisymmetric in `s` about `u_I`: the two neighbours enter with the same
magnitude and opposite signs. They always average back to `u_I`, so the prolongation is
conservative at every order.

## Primal directions

Primal-direction prolongation needs no table of its own: the coincident fine node is an exact
copy of the coarse one, and the half-way node is
`directionalInterp<dim, dir, PrimalToDual, order>` (the same midpoint weights).
`directionalProlongation` is its dual counterpart, kept beside it.
