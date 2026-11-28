# The MHD model

`inc/plamn/solver/mhd/model.hpp` holds the MHD model of a simulation: its patch data per level
number, for the levels it is active on, and what its levels' advance needs that is not the
levels' data. It is an empty specialization when the build does not enable MHD.

## From the input

`MHDModel::from(sim)` reads pyphare's `MHDModel` input:

| Member | Input | Default |
|---|---|---|
| `gamma`, the heat capacity ratio | `simulation/algo/fv_method/heat_capacity_ratio` | required |
| `resistivity` | `.../fv_method/resistivity` | 0 |
| `hyper_resistivity`, `hyper_mode` | `.../fv_method/hyper_resistivity`, `hyper_mode` | 0, constant |
| `hall` | `.../fv_method/hall` | false |
| `scheme`, the time integration | `simulation/algo/time_integrator_type`: Euler, TVDRK2, TVDRK3, SSPRK4_5 | required |
| `boundaries`, the physical boundaries' conditions | `simulation/grid/domain_boundaries` | none |

- The time integration scheme is a run time choice; the reconstruction (and its slope limiter)
  and the Riemann solver are the simulation's compile time options.
- Hall is a build's choice too (a template flag in its module's name): the input's
  `hall` must match the module's, else it throws (`./mkn.py --mhd --hall true|false`).
- Hyper resistivity is only allowed alongside the Hall term.
