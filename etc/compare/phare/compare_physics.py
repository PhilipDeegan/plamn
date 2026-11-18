#!/usr/bin/env python3
"""PHARE vs plamn on the physics, not bit for bit: a tagged 3 level case runs a number of steps
under both codes (run_case.py, phareh5); at the final time each code's B becomes one composite on
the finest level's grid, and the two composites are compared.

  python3 etc/compare/phare/compare_physics.py [--case harris_2d] [--ndim 2] [--steps 10]
                                               [--ref next] [--cmp plamn] [--np 1]

The composite: each level's B at cell centres (a primal direction averaged over the cell's two
nodes), the coarsest level everywhere, each finer level linearly interpolated from the one under
it (periodic) and replaced by its own values where it has them. Where the two codes regridded
differently, a region holds one code's fine data and the other's interpolated coarse data.
Reported per component: max |B|, the largest and the root mean square difference, both relative
to max |B|, and where the largest is.

Environment as test_ladder.py's (PHARE_DIR, LADDER_PPC, LADDER_OUT, ...).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def composite(path, component, time=None):
    """(the finest grid's B component at cell centres, time, number of levels)"""
    import h5py
    from scipy.ndimage import map_coordinates

    f = h5py.File(Path(path) / "EM_B.h5", "r")
    times = sorted(f["/t"].keys(), key=float)
    t = times[-1] if time is None else time
    g = f[f"/t/{t}"]
    levels = sorted((k for k in g.keys() if k.startswith("pl")), key=lambda k: int(k[2:]))
    ds = f"EM_B_{component}"
    ndim = len(g[levels[0]][next(iter(g[levels[0]]))].attrs["lower"])
    primal = ["xyz".index(component)] if "xyz".index(component) < ndim else []

    def level_cells(ln, shape):
        a = np.full(shape, np.nan)
        for p in g[levels[ln]].values():
            if ds not in p:
                continue
            G = int(p[ds].attrs["ghosts"])
            lo = np.array(p.attrs["lower"])
            hi = np.array(p.attrs["upper"])
            n = hi - lo + 1
            v = p[ds][:]
            sl = tuple(slice(G, G + n[d]) for d in range(ndim))
            cells = v[sl]
            for d in primal:  # the cell's two nodes along d
                nxt = tuple(slice(G + 1, G + 1 + n[d]) if e == d else sl[e] for e in range(ndim))
                cells = 0.5 * (cells + v[nxt])
            a[tuple(slice(lo[d], hi[d] + 1) for d in range(ndim))] = cells
        return a

    n0 = np.zeros(ndim, dtype=int)
    for p in g[levels[0]].values():
        n0 = np.maximum(n0, np.array(p.attrs["upper"]) + 1)
    comp = level_cells(0, tuple(n0))
    for ln in range(1, len(levels)):
        shape = tuple(n0 * 2**ln)
        # the coarser composite at this level's cell centres (coarse cell units)
        idx = np.meshgrid(*[(np.arange(s) + 0.5) / 2 - 0.5 for s in shape], indexing="ij")
        up = map_coordinates(comp, idx, order=1, mode="grid-wrap")
        fine = level_cells(ln, shape)
        comp = np.where(np.isnan(fine), up, fine)
    return comp, float(t), len(levels)


def compare(ref, cmp, ndim):
    rows = []
    for c in "xyz":
        a, ta, la = composite(ref, c)
        b, tb, lb = composite(cmp, c)
        if a.shape != b.shape:  # a code with fewer levels: up to the finer one
            from scipy.ndimage import zoom

            if a.size < b.size:
                a = zoom(a, b.shape[0] / a.shape[0], order=1, mode="grid-wrap")
            else:
                b = zoom(b, a.shape[0] / b.shape[0], order=1, mode="grid-wrap")
        scale = max(np.max(np.abs(a)), 1e-300)
        d = np.abs(a - b)
        at = np.unravel_index(np.argmax(d), d.shape)
        rows.append((c, ta, tb, la, lb, scale, d.max() / scale, np.sqrt(np.mean(d**2)) / scale, at))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--case", default="harris_2d")
    p.add_argument("--ndim", type=int, default=2)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--ref", default="master")
    p.add_argument("--cmp", default="plamn")
    p.add_argument("--np", type=int, default=1, help="MPI ranks per run")
    p.add_argument("--no-run", action="store_true", help="compare the runs already there")
    a = p.parse_args(argv)

    import os

    import test_ladder

    os.environ["LADDER_NP"] = str(a.np)
    os.environ["LADDER_FINE_DUMP"] = "0"
    os.environ["LADDER_PARTICLES"] = "0"  # B only is compared: no particle dumps
    base = Path(test_ladder.setting("LADDER_OUT", test_ladder.PLAMN / "phare_outputs" / "compare"))
    import cases

    base = base / f"physics_{a.case}_{a.steps}_ppc{cases.PPC}"
    out = {code: base / Path(code).name for code in (a.ref, a.cmp)}
    if not a.no_run:
        os.environ["LADDER_NDIM"] = str(a.ndim)
        for code in (a.ref, a.cmp):
            test_ladder.run(code, a.case, out[code], None, steps=a.steps)

    print(f"{a.case} {a.ndim}D, {a.steps} steps, {cases.PPC} ppc: {a.ref} vs {a.cmp}, final B on the"
          " finest grid")
    for c, ta, tb, la, lb, scale, dmax, drms, at in compare(out[a.ref], out[a.cmp], a.ndim):
        print(f"  B{c}: t {ta:g}/{tb:g}, levels {la}/{lb}, max|B| {scale:.3g}, "
              f"max diff {dmax:.2e}, rms diff {drms:.2e} (relative), largest at {tuple(int(i) for i in at)}")


if __name__ == "__main__":
    sys.exit(main())
