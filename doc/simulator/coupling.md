# MHD to hybrid coupling

`inc/plamn/simulator/coupling.hpp` holds what crosses the model boundary (`AMR/max_mhd_level`)
from an MHD level to the hybrid level over it. Coupled runs are plamn's own: there is no PHARE
reference for them.

## Pieces

| Function | What |
|---|---|
| `to_mhd_coarse(pairs)` | a hybrid level's pairs as read from the MHD level under it: the coarse names as MHD holds them |
| `prim_pairs()` | MHD's primitives (`rho`, `V`, `P`) into the hybrid patch's cell centred scratch grids (`mhd_rho`, `mhd_V_x`, ..., `mhd_P`) |
| `make_prim_scratch(patch)` | those scratch grids, made only on the hybrid level over MHD |
| `spawn(patch, cells, ...)` | each population's particles in `cells`, from the scratch values |

## Names across the boundary

B and E sit on the same staggered grid in both models, so only their names change:

| Hybrid | MHD |
|---|---|
| `EM_B_x` | `B_x` |
| `EM_B_old_x` | `B_old_x` |
| `EM_E_x` | `E_x` |
| `EM_E_old_x` | `E_old_x` |

## Spawning particles

From the MHD primitives on the scratch grids, per cell:

- density `n = rho / m`, bulk velocity `V`;
- an isotropic Maxwellian of the ion pressure `Pi = max(P - Pe, 0)`, with `Pe = Te q n`
  (isothermal electrons).

There is one ion population for now: MHD's `rho` is that species' mass density.

The random streams of a spawn start from the population's seed, salted
so they differ from the root loader's and from other spawns of the same cells (a coarse step's
new level ghosts are salted by its end time's bits).
