# The harris benchmark

`etc/bench/harris/harris.py` times one coarse step of PHARE tiling_gpu's harris
(`tests/simulator/layouts/test_harris_{2,3}d_layouts.py`) on each code: the wall clock of the
setup + initialization and of the step, and the scope timers (`PHARE_SCOPE_TIMING=1`,
`.phare/timings/rank.<r>.txt` in each code's directory).

```
etc/bench/harris/harris.py [--codes master pr_1601 plamn] [--ndim 2] [--levels 3]
                           [--steps 1] [--ppc N] [--out DIR] [--tiling-gpu DIR]
                           [--plamn-layouts AoSMapped SoAVXChunked] [--np N]
```

## The runs

One rank (or `--np N` under mpirun, a rank per core, rank 0 reports; no particle counts then),
the pyphare threading defaults. PHARE codes: the AoSMapped layout. plamn: once per
`--plamn-layouts` layout (default AoSMapped), as the code `plamn_<layout>` (needs that layout's
module of the dimension, `./mkn.py --dim 2 --layout SoAVXChunked`).

The input is tiling_gpu's `config()` itself, loaded from its file: its module settings
(time_step, ppc) are kept but final_time (steps), max_nbr_levels (levels) and nppc (`--ppc`) are
overridden and there are no diagnostics. Keywords a code's pyphare does not know (model_layout,
threading on PHARE master) are dropped, and said so.

One process per code: each has its own pyphare on PYTHONPATH (a PHARE checkout's `build/` and
`pyphare/`; plamn: its root (the plamn package) and PHARE master's pyphare, patched).

## Reading the timers

The timers are read with phlop's own command line, from damn's phlop (`tpp/phlop`, phlop's next:
it reads damn's `#clock` lines, PHARE's pinned phlop does not), as printed at the end, e.g.

```
PYTHONPATH=tpp/damn/tpp/phlop python3 -m phlop.timing.scope_timer times <out>/<code>/.phare/timings/rank.0.txt
PYTHONPATH=tpp/damn/tpp/phlop python3 -m phlop.timing.scope_compare <base file> <file>
```
