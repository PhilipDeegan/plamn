# Ghost widths

`inc/plamn/core/data/grid/ghost_width_calculator.hpp` derives the field and particle ghost widths
of both models at compile time.

## The allowed cell jump

`particle_allowed_cell_jump` (`PLAMN_PARTICLE_ALLOWED_CELL_JUMP`, default 2) is the most cells a
domain particle may move in one advance (two half pushes): a CFL for particles.

- A domain particle moving further throws (the cell jump limit).
- Level ghost particles leaving the particle ghost box are dropped.
- The particle and field ghost widths follow from it.

`PLAMN_PARTICLE_ALLOWED_CELL_JUMP=1` gives PHARE master's widths, for comparisons with PHARE.

`particle_half_push_limit` is how far past its cell's start a particle may be after a half push
(`|delta|`) before the Boris pusher throws: PHARE's 2, or the allowed cell jump if larger. It
equals the jump by default; a narrower jump (PHARE's ghost widths) keeps PHARE's limit.

## Hybrid PIC

| Width | Interp 1 | Interp 2 | Interp 3 | From |
|---|---|---|---|---|
| `stencilReach` | 1 | 1 | 2 | the interpolation stencil outside a particle's cell, the worst of primal and dual, either side |
| `levelGhostParticleWidth` | 1 | 2 | 2 | level ghost particles' deposits reach every domain node of the level |
| `particleGhostWidth` | | | | `max(jump, levelGhostParticleWidth)`: level ghost particles, and domain particles that moved out by up to the jump |
| `nbrGhostsFromInterpOrder`, jump 1 | 2 | 4 | 4 | PHARE's widths |
| `nbrGhostsFromInterpOrder`, jump 2 | 4 | 4 | 4 | the default |

A particle in cell `c` touches nodes `c - shift ... c - shift + order`, with `shift` from
`Interpolator::computeStartLeftShift`; hence the stencil reach.

The field ghost width is `particleGhostWidth + stencilReach`: particles in the particle ghost box
interpolate the mesh. It is rounded up to even, see [Even widths](#even-widths).

## MHD

`nbrGhostsFromReconstruction` is the reconstruction stencil width plus one, rounded up to even:

- the extra layer: Ampere computes `J` on the ghost box shrunk by one, so `J` is only valid on
  `ghost_width - 1` layers, while the reconstruction reads it out to the stencil width;
- there is no hyper-resistivity layer: the dissipative electric field takes its Laplacian on `J`
  at its native edge location, and the flux contributions project it to the face, so they stay
  within the reconstruction reach.

## Even widths

Both models round their ghost width up to even, so that the ghost box stays a union of whole
coarse cells (lower even, upper odd) at refinement ratio 2. The magnetic refinement's fill-box
round-out relies on it (PHARE's `MagneticRefinePatchStrategy::reconstructionRegion` and
`coarse_cell_round_out.hpp`).
