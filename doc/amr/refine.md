# Refinement operators

`inc/plamn/amr/refine.hpp` holds the refine operators that `damn::amr::refine` (and
`damn::mpi`'s) apply to plamn's resources (`resources.hpp`), for a refinement ratio of 2.

| Operator | What |
|---|---|
| `FieldRefiner` | PHARE's composite linear refinement, any centering |
| `CellInjector` | piecewise constant: a fine point takes the coarse point under it |
| `divergence_free_touch_up` | stage 2 of PHARE's divergence free magnetic refinement |
| `ParticleSplitter<Splitter>` | PHARE's particle refinement |

## `FieldRefiner`

The rule is chosen per direction, from the field's centering in that direction; the value is
the tensor product of the per direction stencils.

| Centering | Fine point | Value |
|---|---|---|
| primal | on a coarse node | that node's value |
| primal | between two coarse nodes | their mean |
| dual | the two fine cells of coarse cell `I` | `u_I -/+ (u_{I+1} - u_{I-1}) / 8` (left / right) |

The two dual children mean back to `u_I`.

## `divergence_free_touch_up(B, cells, region, dx)`

Stage 2 of PHARE's divergence free magnetic refinement (Balsara et al. 2023, see PHARE's
`magnetic_refine_patch_strategy.hpp`).

After B's components were refined with `FieldRefiner` over whole coarse cells, the faces inside
each coarse cell are corrected so that the divergences of its fine cells all equal the coarse
cell's. A divergence free coarse B then refines divergence free.

- `region` must be a union of whole coarse cells (even lower, odd upper bounds), else it throws.
- The fine cells' divergences are computed once, before any correction; each interior face
  (odd along its normal) is then corrected from the divergence jumps across it and across its
  siblings in the same coarse cell, with weights per dimension (1D `1/2`; 2D `3, 1` over 8; 3D
  `7, 2, 2, 1` over 24).

## `ParticleSplitter<Splitter>`

Each coarse particle is split into `Splitter::nbRefinedPart` fine ones (`split/split.hpp`), and
those in the target box are kept. Its stencil covers the coarse cells holding every parent that
has a child in the target.
