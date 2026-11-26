# plamn overview

plamn is plasma physics on top of [damn](https://github.com/PhilipDeegan/damn): grid layouts,
fields, particles, ions, electrons, electromagnetics, their numerics, and a simulator over damn's
patch hierarchy. It follows PHARE's algorithms (hybrid PIC, MHD, AMR with time refinement) and
takes PHARE's python input (pyphare) unchanged, so the same input script can run on either code.

This is a work in progress: a map of the headers and of how to run things. Longer explanations
live in `doc/`, beside this page.

## Layout

| Path | What |
|------|------|
| `inc/plamn/core/data` | grids and layouts, fields, vector/tensor fields, particles (several memory layouts), ions, electrons, electromag |
| `inc/plamn/core/numerics` | Faraday, Ampere, Ohm, the pusher, interpolation/deposit, moments, boundary conditions, MHD (reconstructions, Riemann solvers, constrained transport, time integrators) |
| `inc/plamn/solver` | the models (hybrid, MHD): per level patch data, initialization, a level's step |
| `inc/plamn/amr` | refine/coarsen operators, particle splitting, tagging, load balancing schedule |
| `inc/plamn/simulator` | the simulator: input, setup, advance, regrid, ghosts, diagnostics, restarts, MHD/hybrid coupling |
| `inc/plamn/diagnostic` | phareh5 and pharevtkhdf writers |
| `src/python3` | the nanobind modules: `plamn_etc` (dict, enums, MPI) and the simulator modules |
| `plamn/` | the python package |
| `etc/compare/phare` | PHARE vs plamn comparisons |
| `etc/bench` | benchmarks (harris, the mover, Ohm, AMR cost models, diagnostics I/O) |
| `etc/view3d` | a standalone 3D viewer fed by `python3 -m plamn --view3d` |

## Models and levels

A simulation's levels belong to models: MHD below `max_mhd_level`, hybrid above (either can be
the only one). Each model owns its patch data per level; the hierarchy's geometry, neighbours
and level maps are the simulator's and shared. Time refinement is damn's; each level takes its
model's step, and finer levels are synchronised onto coarser ones (coarsening, reflux), as
PHARE's multi physics integrator.

Refinement is either fixed boxes from the input or tagging, with regrids and level 0 load
balancing during the run.

## Build permutations

The simulator is compiled per permutation, as PHARE's: dimension, interpolation order, refined
particle number, particle layout and allocator (and for MHD the reconstruction, limiter, Riemann
solver and Hall). Each permutation is its own python module, named after pyphare's simulator id:

```
plamn/plamn_cpp_<dim>_<interp>_<refined>_<layout>_<allocator>.so
```

`./mkn.py` builds them (see its docstring for every option), e.g.

```sh
./mkn.py --dim 2 3                         # 2D and 3D, interp 1, AoSMapped, CPU
./mkn.py --dim 3 --layout SoAVXChunked     # another particle layout
./mkn.py --dim 2 --diag-doubles --cell-jump 1   # as a PHARE_DIAG_DOUBLES=1 build, PHARE's ghost widths
./mkn.py --mhd                             # MHD only
```

Modules are built with MPI (`res/mkn/mpi`) and `-O` (native) by default; `--optimise 3` for
modules run on another machine. `-- <mkn args>` passes anything else to mkn (e.g. `-g 0` for
`NDEBUG`).

## Running

plamn runs pyphare's input with plamn's C++: `plamn.patch()` swaps pyphare's C++ modules for
plamn's. pyphare itself comes from a PHARE checkout; no PHARE build is needed.

```python
import plamn
plamn.patch()                       # before using pyphare
import pyphare.pharein as ph
# ... the usual pyphare input ...
```

or, without naming pyphare, `import plamn.backend.phare as ph`. With

```sh
PYTHONPATH=<plamn root>:<PHARE checkout>/pyphare mpirun -n 4 python3 input.py
```

the module loaded is the one of the simulation's permutation; plamn's extra `Simulation`
keywords `model_layout` and `allocator` choose the layout and allocator (default AoSMapped,
CPU). A missing module raises with the `./mkn.py` line that builds it.

`python3 -m plamn <input>` runs an input with live plots (`plamn/watch.py`).

## Comparing with PHARE

`etc/compare/phare` runs the same input under a PHARE build and under plamn, one process each
(each with its own pyphare on `PYTHONPATH`), and compares the diagnostics:

- `cases.py`: the cases, from small ladders (one mechanism added per case: loading, a refined
  level, a step, level ghosts, a third level, tagging) to longer physics runs (harris 2D/3D, ion
  beams, MHD shocks), each with a memory estimate and the tier it runs on (GitHub, local, by hand)
- `run_case.py`: one case under one code
- `test_ladder.py`: the ladder as unittests, cell by cell where hierarchies overlap
  (`hierarchy_overlap.py`); environment variables pick the codes, ranks, layout and format
- `compare_physics.py`: longer runs compared on the physics rather than bit for bit: each code's
  B (and E) as one composite on the finest grid, max and rms differences, optionally at every
  dump time to see how the codes drift apart

PHARE references are built with `PHARE_DIAG_DOUBLES=1`; the plamn module to compare against
them with `--diag-doubles --cell-jump 1`.

## Timing

`etc/bench/harris/harris.py` times setup and coarse steps of the harris tests on PHARE builds and
plamn (per plamn layout), on one rank or under mpirun. Both codes write phlop scope timers with
`PHARE_SCOPE_TIMING=1` (`.phare/timings/rank.<r>.txt`); damn's phlop reads them:

```sh
PYTHONPATH=tpp/damn/tpp/phlop python3 -m phlop.timing.scope_timer times .phare/timings/rank.0.txt
```

What plamn's scopes record is chosen when building the module (`./mkn.py --scope-flags`,
`--log-level`); `--sync-pools` gives serial numbers with no pool threads.

## Testing

| Profile / script | What |
|------------------|------|
| `test_plamn_core` | the C++ unit tests |
| `test_plamn_core_slow` | particle partitioning at scale |
| `test_plamn_diagnostics` | diagnostics through embedded python, 1 and 3 ranks |
| `tst/plamn/python` | python tests against built modules (phlop's test runner) |
| `res/ci/local.sh` | what CI runs, locally |

```sh
mkn clean build test -p test_plamn_core -dKO -t 4 -x res/mkn/thrust
```

damn is found at `tpp/damn`, cloned by mkn, or a symlink to a damn checkout.
