# MHD patch

`inc/plamn/simulator/mhd_patch.hpp`: `MHDPatch` is one patch of an MHD level, holding what the
time integration schemes (`core/numerics/time_integrator`) use, under the member names they
expect.

## Members

| Member | What |
|---|---|
| `states` | the scheme's states: `states[0]` the advanced one, the others its stages; their number is the scheme's, a run time choice |
| `old` | `states[0]` at the start of the level's step (`prepare_step`): a finer level's ghosts interpolate in time between it and `states[0]` (PHARE's `MHDMessenger` `rhoOld_`...), and a reflux redoes the step from it (PHARE's `SolverMHD` `stateOld_`) |
| `E` | the constrained transport electric field of the last flux computation |
| `fluxes` | per face direction, of the last flux computation |
| `ct` | the upwind constrained transport inputs of those fluxes |
| `butcher`, `butcherE` | the scheme's weighted sums of the fluxes and E: the step's fluxes and E, what a reflux replaces under a finer level |
| `J`, `E_diss` | resistive MHD: `J = curl B` and the dissipative E (`eta J`), added to the constrained transport E and to the face fluxes (PHARE's `DissipativeElectricField`); Hall MHD needs `J` too |
| `flux_sum`, `flux_sum_E` | the step fluxes and E summed over the level's substeps in its coarser level's step (PHARE's `SolverMHD` `fluxSum_`, `fluxSumE_`): what its reflux coarsens |
| `coupling` | for a hybrid level over this one, in builds with both models: `E_old`, E at the step's start (`keep_E_old`); the hybrid level's E ghosts interpolate in time between it and `E`. Only on the finest MHD level (`make_under_hybrid`) |

## Storage

The grids are named as their views. The states are not filled (NaN in debug builds): the
simulator writes them before reading them. The other grids start at 0, as PHARE's fields do,
which matters in two places:

- E's component without an equation (`E_x` in 1D) stays 0;
- a direction a lower dimension does not have has 0 fluxes.

A patch is not movable: the vector fields are views over their own grids. `LevelData` keeps
patches behind `unique_ptr`.

## Resources

`for_each_resource` names what damn's schedules may move. The advanced state uses PHARE's MHD
names.

| Resources | What |
|---|---|
| `rho`, `rhoV_x`..., `B_x`..., `Etot`, `V_x`..., `P` | the advanced state |
| `rho_s<s>`, `rhoV_s<s>_x`... | stage `s > 0` |
| `rho_old`, `rhoV_old_x`..., `B_old_x`..., `Etot_old` | the old state |
| `E_x`... | the constrained transport E |
| `F_rho_butcher_<d>`, `F_rhoV_butcher_<d>_x`..., `F_Etot_butcher_<d>` | per direction `d < dim`, the refluxed step fluxes |
| `F_rho_sum_<d>`... | their sums over substeps |
| `butcherE_x`..., `E_sum_x`... | the step's E and its sum |
| `E_old_x`... | under a hybrid level only, last, so the other grids' indices do not change |

The last fluxes and the constrained transport inputs never move.
