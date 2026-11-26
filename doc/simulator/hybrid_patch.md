# Hybrid patch

`inc/plamn/simulator/hybrid_patch.hpp`: `HybridPatch` is one patch of a hybrid level, its data
and the core views over it, as a SAMRAI patch's data would be.

| Part | What |
|---|---|
| views | `em`, `J`, `ions` (a runtime list of populations), the electrons; the PPC solver's predicted and averaged electromag (`emPred`, `emAvg`) |
| storage | the grids and particle arrays the views point at, named as the views |

The storage is not filled at construction: the simulator writes everything before reading it.
In debug builds every value starts NaN, so a point nothing filled is found (`check_set_`) rather
than read.

A patch is not movable: the views, and the electrons' copies of the ions and J views, point at
its members. `LevelData` keeps patches behind `unique_ptr`.

## Resources

`for_each_resource` names what damn's schedules may move:

| Resources | What |
|---|---|
| `EM_B_x`..., `EM_E_x`..., `EMPred_B_x`..., `EMAvg_E_x`... | electromag, predicted and averaged |
| `<pop>_density`, `<pop>_charge_density`, `<pop>_flux_x`... | per population moments |
| `ions_charge_density`, `ions_charge_flux_x`... | the totals, what a synchronization coarsens |
| `EM_E_old_x`..., `EM_B_old_x`..., `EM_fluxSum_E_x`... | AMR: E and B at the start of the level's step, the time average of E over a coarser step |
| `Pe` | the electrons' pressure, a synchronization's |
| `Se_old`, `Se`, ... | the closure's own fields |
| `<pop>_domain`, `<pop>_patch_ghost`, `<pop>_level_ghost[_old, _new]` | particles |

The field grids come in a fixed order, the same on every patch of the model (`grid_table_`),
with the computed quantities' scratch grids after them.
