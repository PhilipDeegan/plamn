#!/usr/bin/env python3
"""Runs one case of cases.py under one code, diagnostics into OUT.

  run_case.py phare|plamn CASE OUT [--ndim 2] [--largest-patch-size N] [--fine-dump]

A case with steps advances them one by one; an init case only initializes (dumps t=0). After
initializing and after each step, each level's domain particles (from the data wrangler, as
both codes provide it) go to OUT/particles/<step>.npz, for hierarchy_overlap.compare_particles
(plamn's wrangler has no level ghost particles yet). One process per code:
PHARE's and plamn's python modules both register as pyphare's simulator, so the caller sets
PYTHONPATH (see test_ladder.py's environment()). One rank (the wrangler's view is rank 0's).
"""

import argparse
import sys
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
    a = p.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).parent))
    import cases

    if a.code == "plamn":
        import pyplamn

        pyplamn.patch()
    from pyphare.simulator.simulator import Simulator, startMPI

    startMPI()
    fmt = "phareh5" if a.code == "phare" and a.fine_dump else "pharevtkhdf"  # cases._diag_options
    sim = cases.simulation(a.case, a.out, a.ndim, a.largest_patch_size, fmt, a.fine_dump)
    simulator = Simulator(sim, log_to_file=False).setup().initialize()
    parts = Path(a.out) / "particles"
    parts.mkdir(parents=True, exist_ok=True)
    np.savez(parts / "0.npz", **particle_arrays(simulator, cases.POPULATIONS))
    for step in range(1, cases.CASES[a.case]["steps"] + 1):
        simulator.advance()
        np.savez(parts / f"{step}.npz", **particle_arrays(simulator, cases.POPULATIONS))
    simulator.reset()
    return 0


if __name__ == "__main__":
    sys.exit(main())
