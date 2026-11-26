# Simulator modules

`src/python3/plamn_sim.cpp` builds a `plamn_cpp_<id>` module: one simulator build permutation,
as pyphare's `cpp_<id>` modules.

`make_simulator(dict)` returns a `Simulator`. `dict` is the `plamn_etc.Dict` that pyphare filled,
handed over by pyphare as the "hierarchy" (`plamn.etc.make_hierarchy` returns it).

## The permutation

It comes from compile definitions, set by `./mkn.py`, which also names the module:

| Definition | What | Default |
|---|---|---|
| `PLAMN_SIM_DIM` | dimension | 1 |
| `PLAMN_SIM_INTERP` | interpolation order; 0: no hybrid model | 1 |
| `PLAMN_SIM_REFINED` | refined particle number | pyphare's default |
| `PLAMN_SIM_LAYOUT` | particle layout | `AoSMapped` |
| `PLAMN_SIM_ALLOCATOR` | allocator | `CPU` |
| `PLAMN_SIM_RECONSTRUCTION` | MHD reconstruction | |
| `PLAMN_SIM_LIMITER` | MHD slope limiter | |
| `PLAMN_SIM_RIEMANN` | MHD Riemann solver | `MHDOff`: no MHD model |

A model's bindings exist only where the build enables it: the hybrid levels' particles, for
example, are a template, so a build without the hybrid model never instantiates them.
