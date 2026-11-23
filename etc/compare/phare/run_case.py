#!/usr/bin/env python3
"""Runs one case of cases.py under one code, diagnostics into OUT.

  run_case.py phare|plamn CASE OUT [--ndim 2] [--largest-patch-size N] [--fine-dump]
              [--format phareh5|pharevtkhdf] [--steps N]

A case with steps advances them one by one; an init case only initializes (dumps t=0). After
initializing and after each step, each level's domain particles (from the data wrangler, as
both codes provide it) go to OUT/particles/<step>.npz, for hierarchy_overlap.compare_particles
(plamn's wrangler has no level ghost particles yet). One process per code:
PHARE's and plamn's python modules both register as pyphare's simulator, so the caller sets
PYTHONPATH (see test_ladder.py's environment()). One rank (the wrangler's view is rank 0's).
Each rank writes its peak resident memory and wall time to OUT.mem.<rank>.json at the end
(test_ladder.run sums them: what a case's "mem" estimate is checked against).
"""

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np


def _flat(a, n):
    return np.asarray(a).reshape(n, -1)


def domain_particles(dw, ilvl, pop):
    """(icell, delta, v, weight) of level ilvl's domain particles, from the data wrangler:
    PHARE's getParticles is {pop: {"domain": [...], "levelGhost": [...]}} of packed arrays,
    plamn's a list (one per patch) of live arrays read through components()."""
    cpp = dw.cpp  # PHARE's C++ wrangler: getPatchLevel; plamn's: getHybridPatchLevel
    level = getattr(cpp, "getPatchLevel", None) or cpp.getHybridPatchLevel
    got = level(ilvl).getParticles(pop)
    rows = []
    if isinstance(got, dict):
        for pd in got.get(pop, {}).get("domain", []):
            d = pd.data
            n = len(np.asarray(d.weight))
            rows.append((_flat(d.iCell, n), _flat(d.delta, n), _flat(d.v, n),
                         np.asarray(d.weight)))
    else:
        for pd in got:
            c = pd.data.components()
            dims = sorted(k[-1] for k in c if k.startswith("iCell_"))
            rows.append((np.stack([np.asarray(c[f"iCell_{x}"]) for x in dims], axis=1),
                         np.stack([np.asarray(c[f"delta_{x}"]) for x in dims], axis=1),
                         np.stack([np.asarray(c[f"v_{x}"]) for x in "xyz"], axis=1),
                         np.array(c["weight"])))
    if not rows:
        return None
    return [np.concatenate([r[i] for r in rows]) for i in range(4)]


def particle_arrays(simulator, pops):
    """{key: array} of every level's domain particles, keys "<pop>/L<n>/<what>"."""
    dw = simulator.data_wrangler()
    arrays = {"time": np.array([simulator.currentTime()])}
    for pop in pops:
        for ilvl in range(dw.getNumberOfLevels()):
            got = domain_particles(dw, ilvl, pop)
            if got is None:
                continue
            for name, a in zip(("icell", "delta", "v", "weight"), got):
                arrays[f"{pop}/L{ilvl}/{name}"] = np.array(a)  # copies: live views in plamn
    return arrays


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("code", choices=["phare", "plamn"])
    p.add_argument("case")
    p.add_argument("out")
    p.add_argument("--ndim", type=int, default=2)
    p.add_argument("--largest-patch-size", type=int, default=None)
    p.add_argument("--fine-dump", action="store_true", help="fine_dump_lvl_max (see cases.py)")
    p.add_argument("--format", default="phareh5", choices=["phareh5", "pharevtkhdf", "none"])
    p.add_argument("--live-out", help="live.py snapshots at the dump times to this npz (no "
                   "diagnostics): a reference")
    p.add_argument("--live-key", default="{}", help="--live-out: what made it (json)")
    p.add_argument("--live-ref", help="live.py: compare against this npz's snapshots at the "
                   "dump times (no diagnostics); the rows to --live-report")
    p.add_argument("--live-report", help="--live-ref: the rows as json")
    p.add_argument("--steps", type=int, default=None, help="the case's steps, else cases.py's")
    a = p.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).parent))
    import cases

    start = time.monotonic()
    if a.code == "plamn":
        # PHARE's regrid points (SAMRAI's): a regrid refills the level ghost fields after the
        #  synchronization, plamn's default skips it where the boxes do not change
        os.environ.setdefault("PLAMN_REGRID_MODE", "high")
        import plamn

        plamn.patch()
    from pyphare.simulator.simulator import Simulator, startMPI

    startMPI()
    if a.steps is not None:
        cases.CASES[a.case] = dict(cases.CASES[a.case], steps=a.steps)
    live = a.live_out or a.live_ref
    fmt = "none" if live else a.format
    sim = cases.simulation(a.case, a.out, a.ndim, a.largest_patch_size, fmt, a.fine_dump)
    if a.code == "plamn":  # its layout and allocator (test_ladder.py: LADDER_LAYOUT, ...)
        sim.model_layout = os.environ.get("LADDER_LAYOUT", "AoSMapped")
        sim.allocator = os.environ.get("LADDER_ALLOCATOR", "CPU")
    simulator = Simulator(sim, log_to_file=False).setup().initialize()
    every = cases.dump_every(a.case)
    if live:
        import live as lv

        snaps = lv.Snapshots(simulator, a.ndim)
        comparer = lv.Comparer(lv.load(a.live_ref)[1]) if a.live_ref else None
        saved, rows = [], []

        def on_dump():
            snap = snaps.take()
            if comparer:
                rows.extend(comparer.against(snap))
            else:
                saved.append(snap)

        on_dump()
    keep = os.environ.get("LADDER_PARTICLES", "1") == "1"  # 0: no particles (large runs)
    parts = Path(a.out) / "particles"
    if keep:
        parts.mkdir(parents=True, exist_ok=True)
        np.savez(parts / "0.npz", **particle_arrays(simulator, cases.populations(a.case)))
    for step in range(1, cases.CASES[a.case]["steps"] + 1):
        simulator.advance()
        if keep:
            np.savez(parts / f"{step}.npz", **particle_arrays(simulator, cases.populations(a.case)))
        if live and step % every == 0:
            on_dump()
    if a.live_out:
        lv.save(a.live_out, json.loads(a.live_key), saved)
    if a.live_ref:
        Path(a.live_report).write_text(json.dumps(
            [dict(quantity=c, t=ta, levels=[la, lb], scale=float(sc), max_diff=float(dm),
                  rms_diff=float(dr), at=[int(i) for i in at])
             for c, ta, tb, la, lb, sc, dm, dr, at in rows], indent=1))
    simulator.reset()
    write_memory(a.out, time.monotonic() - start)
    return 0


def write_memory(out, seconds):
    """this rank's peak resident memory (Linux ru_maxrss: KiB) and wall time, OUT.mem.<rank>.json"""
    from pyphare import cpp

    rank = cpp.mpi_rank()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    Path(f"{out}.mem.{rank}.json").write_text(
        json.dumps(dict(rank=rank, peak_rss=peak, seconds=seconds)))


if __name__ == "__main__":
    sys.exit(main())
