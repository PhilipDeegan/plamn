# Coarsening operators

`inc/plamn/amr/coarsen.hpp` holds the coarsen operators that `damn::amr::coarsen` (and
`damn::mpi`'s) apply to plamn's field resources (`resources.hpp`), for a refinement ratio of 2.

The rule is chosen per direction, from the field's centering in that direction. Coarse index `q`
covers the fine nodes / cells around `2q`.

| Operator | Primal | Dual | Used for |
|---|---|---|---|
| `DefaultFieldCoarsener` | `1/4, 1/2, 1/4` of fine nodes `2q - 1`, `2q`, `2q + 1` | mean of fine cells `2q` and `2q + 1` | B |
| `ElectricFieldCoarsener` | fine node `2q` (injection) | mean of fine cells `2q` and `2q + 1` | E |
| `MomentsCoarsener` | as `ElectricFieldCoarsener` | | moments (all primal: injected) |
| `MHDFluxCoarsener` | as `ElectricFieldCoarsener` | | MHD face fluxes |

In more than one direction the weights multiply: a point's coefficient is the product of its
per direction weights.

`MHDFluxCoarsener`: a coarse face's flux is the mean of the fine faces on it, which is the same
per direction rule (primal: injected, dual: the mean).
