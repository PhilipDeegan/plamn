# MHD time integration

`inc/plamn/core/numerics/time_integrator/mhd_time_integrator.hpp` integrates the MHD equations on
one level, as PHARE's `amr/solvers/time_integrator` does: Euler, TVDRK2, TVDRK3 and SSPRK4_5. It
follows PHARE's operation order, so results can match PHARE's bit for bit in the same floating
point context (see the tests).

## The level

The integrators take the level through an adapter giving:

| Call | Effect |
|---|---|
| `level.per_patch(fn)` | `fn(patch)` for every local patch |
| `level.fill_state_ghosts(s, t)` | the patch and level ghosts of every patch's `states[s]` (rho, rhoV, B, Etot) at time `t` |
| `level.fill_electric_ghosts(t)` | the same for every patch's `E` |

## A patch

A patch has the members:

| Member | Content |
|---|---|
| `layout` | the patch's layout |
| `states` | indexable, of `MHDState` (`core/models/mhd_state.hpp`): `[0]` is the state advanced, the others the integrator's stages (`n_states` of them in all) |
| `E` | the constrained transport E of the last fluxes computed. There is one: each E is used right after it is computed, where PHARE keeps one per state |
| `fluxes`, `ct` | `[3]` `MHDFaceFluxes` and `[3]` `MHDFaceCT`, of the last fluxes computed |
| `butcher`, `butcherE` | `[3]` `MHDFaceFluxes` and an E: the step's weighted sums of fluxes and E, what a coarser level is synchronised with |

Per patch work goes through the transformers: tiled patches are computed tile by tile. The
butcher sums and the RK blends run on the host; running them as the level transformer's pool
tasks is still to do.

## Steps

`MHDSteps` holds the steps the integrators are made of, each over the whole level. They follow
PHARE's `ComputeFluxes`, `EulerUsingComputedFlux`, `BaseMHDTimestepper`'s butcher sums and
`RKUtils`.

- `compute_fluxes(level, s, t)`: from `states[s]`, to primitive, Godunov per direction, to
  conservative (rhoV and Etot are recomputed from V and P, as PHARE does), constrained
  transport; then E's ghosts. A Hall patch (`Patch::hall`, fixed by the build) computes J every
  time (PHARE's `needsCurrent`), and the Godunov faces and the constrained transport include
  the Hall term (`GodunovTransformer::hall`, `UpwindCTTransformer::hall`).
- `euler_using(level, si, so, E_of, fluxes_of, t, dt)`: `states[so] = states[si] - dt div(F)`,
  B by Faraday with E, then `states[so]`'s ghosts. `E_of` and `fluxes_of` select the E and the
  fluxes used: the last computed, or the butcher sums. `so` may be `si`.
- `euler`: `compute_fluxes`, then `euler_using` with them (PHARE's Euler).
- `assign_butcher(level, coef)`: `butcher = coef * (fluxes, E)` on every point, a step's first
  sum. PHARE resets to 0 then adds the product: `0 + x == x`, the same bits except for the sign
  of a zero.
- `accumulate_butcher(level, coef)`: `butcher += coef * (fluxes, E)` (PHARE's
  `PlusEqualsProduct`).
- `rk_blend(level, dst, {w_j, s_j}...)`: `states[dst] = sum_j w_j states[s_j]` on the domain
  (PHARE's `RKUtils`), no ghosts.

## Dissipation

With a resistivity `eta` or a hyper resistivity `nu`, J and the dissipative E are the patch's
`J` and `E_diss`, as PHARE's `ComputeFluxes` (`DissipativeElectricField`):

$$
E_{diss} = \eta J - c \nabla^2 J, \qquad
c = \begin{cases} \nu & \text{constant} \\ \nu\, \Delta x^2 \left(\frac{|B|}{\rho} + 1\right) & \text{spatial} \end{cases}
$$

with `dx` the smallest mesh size and B and rho projected onto the E component, in PHARE's order.
The dissipative E also works on the face fluxes (PHARE's `MHDEquations`
`dissipative_contributions`): `F_Etot` gets E x B along the face's normal, projected onto the
face. There is no `F_B`: B is evolved by Faraday.

## Integrators

Each integrator's `operator()(level, t0, t1)` advances every patch's `states[0]` from `t0` to
`t1`: `EulerIntegrator`, `TVDRK2Integrator`, `TVDRK3Integrator` and `SSPRK4_5Integrator`.
