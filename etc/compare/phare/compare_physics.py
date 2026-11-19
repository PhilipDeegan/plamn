#!/usr/bin/env python3
"""PHARE vs plamn on the physics, not bit for bit: a tagged 3 level case runs a number of steps
under both codes (run_case.py, phareh5); at the final time each code's B becomes one composite on
the finest level's grid, and the two composites are compared.

  python3 etc/compare/phare/compare_physics.py [--case harris_2d] [--ndim 2] [--steps 10]
                                               [--ref next] [--cmp plamn] [--np 1]
                                               [--all-times] [--json FILE]

--all-times: B and E at every dump time (a long run: how the two codes drift apart), --json: the
rows written to FILE too.

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


# the MHD quantities (PHARE's MHDDiagnostics /mhd/<qty>), cell centred: scalars, or V (vector)
MHD_SCALARS = ("rho", "P")
MHD_VECTORS = ("V",)


def components(qty):
    return [""] if qty in MHD_SCALARS else list("xyz")


def field(qty, component, ndim):
    """(file, dataset, primal directions) of a quantity's component"""
    if qty in MHD_SCALARS:
        return f"mhd_{qty}.h5", qty, []
    if qty in MHD_VECTORS:
        return f"mhd_{qty}.h5", f"{qty}_{component}", []
    # Yee: B_c primal along c, E_c dual along c and primal across
    d_c = "xyz".index(component)
    if qty == "B":
        primal = [d_c] if d_c < ndim else []
    else:
        primal = [d for d in range(ndim) if d != d_c]
    return f"EM_{qty}.h5", f"EM_{qty}_{component}", primal


def dump_times(path, qty="B"):
    import h5py

    with h5py.File(Path(path) / field(qty, components(qty)[0], 1)[0], "r") as f:
        return sorted(f["/t"].keys(), key=float)


def composite(path, component, time=None, qty="B"):
    """(the finest grid's quantity component at cell centres, time, number of levels)"""
    import h5py
    from scipy.ndimage import map_coordinates

    f = h5py.File(Path(path) / field(qty, component, 1)[0], "r")
    times = sorted(f["/t"].keys(), key=float)
    t = times[-1] if time is None else time
    g = f[f"/t/{t}"]
    levels = sorted((k for k in g.keys() if k.startswith("pl")), key=lambda k: int(k[2:]))
    ndim = len(g[levels[0]][next(iter(g[levels[0]]))].attrs["lower"])
    _, ds, primal = field(qty, component, ndim)

    def level_cells(ln, shape):
        a = np.full(shape, np.nan)
        for p in g[levels[ln]].values():
            if ds not in p:
                continue
            G = int(p[ds].attrs["ghosts"])
            lo = np.array(p.attrs["lower"])
            hi = np.array(p.attrs["upper"])
            n = hi - lo + 1
            if p[ds].dtype != np.float64:  # float diagnostics: 1e-8 "differences" from storage
                raise SystemExit(
                    f"{path}: {ds} is {p[ds].dtype}: comparing needs double precision diagnostics "
                    "(plamn: ./mkn.py --diag-doubles, PHARE: -DPHARE_DIAG_DOUBLES=1)")
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


def compare(ref, cmp, ndim, qty="B", time=None):
    rows = []
    for c in components(qty):
        a, ta, la = composite(ref, c, time, qty)
        b, tb, lb = composite(cmp, c, time, qty)
        if a.shape != b.shape:  # a code with fewer levels: up to the finer one
            from scipy.ndimage import zoom

            if a.size < b.size:
                a = zoom(a, b.shape[0] / a.shape[0], order=1, mode="grid-wrap")
            else:
                b = zoom(b, a.shape[0] / b.shape[0], order=1, mode="grid-wrap")
        scale = max(np.max(np.abs(a)), 1e-300)
        d = np.abs(a - b)
        at = np.unravel_index(np.argmax(d), d.shape)
        rows.append((qty + c, ta, tb, la, lb, scale, d.max() / scale,
                     np.sqrt(np.mean(d**2)) / scale, at))
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
    p.add_argument("--all-times", action="store_true", help="B and E at every dump time")
    p.add_argument("--json", default=None, help="the rows written to this file too")
    p.add_argument(
        "--tolerance", nargs=3, type=float, default=None, metavar=("START", "GROWTH", "CAP"),
        help="fail where a relative difference at the k-th time compared exceeds "
        "min(CAP, START * GROWTH**k) (default: the case's tolerance, else none: print only)",
    )
    p.add_argument(
        "--quantities", nargs="+", default=None,
        help="B, E, rho, V, P (default: B, and E with --all-times; an MHD case: B rho V P)",
    )
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

    quantities = a.quantities
    if quantities is None:
        if cases.CASES[a.case].get("mhd_shock") or cases.CASES[a.case].get("mhd_advance"):
            quantities = ["B", "rho", "V", "P"]
        else:
            quantities = ["B", "E"] if a.all_times else ["B"]
    times = dump_times(out[a.ref], quantities[0]) if a.all_times else [None]
    what = " ".join(quantities) + (" at every dump" if a.all_times else ", final")
    print(f"{a.case} {a.ndim}D, {a.steps} steps, {cases.PPC} ppc: {a.ref} vs {a.cmp}, {what} on the"
          " finest grid (differences relative to the largest |value|)")
    report = []
    for t in times:
        for qty in quantities:
            for c, ta, tb, la, lb, scale, dmax, drms, at in compare(
                out[a.ref], out[a.cmp], a.ndim, qty, t
            ):
                print(f"  {c}: t {ta:g}/{tb:g}, levels {la}/{lb}, max|{c}| {scale:.3g}, "
                      f"max diff {dmax:.2e}, rms diff {drms:.2e}, largest at "
                      f"{tuple(int(i) for i in at)}")
                report.append(dict(quantity=c, t=ta, levels=[la, lb], scale=scale,
                                   max_diff=dmax, rms_diff=drms))
    if a.json:
        import json

        Path(a.json).write_text(json.dumps(report, indent=1))
    if not all(np.isfinite(r["max_diff"]) for r in report):
        print("a difference is not finite: a run blew up or wrote NaN")
        return 1
    tolerance = a.tolerance or cases.CASES[a.case].get("tolerance")
    return 1 if tolerance and not within(report, *tolerance) else 0


def within(report, start, growth, cap):
    """every row's relative difference within the staggered tolerance: at the k-th time compared
    min(cap, start * growth**k), tight first (a bug shows at once), looser as round off grows"""
    times = sorted({r["t"] for r in report})
    ok = True
    for r in report:
        k = times.index(r["t"])
        allowed = min(cap, start * growth**k)
        if r["max_diff"] > allowed:
            print(f"FAIL {r['quantity']} t {r['t']:g} (time {k}): {r['max_diff']:.2e} > "
                  f"{allowed:.2e} allowed")
            ok = False
    worst = max(report, key=lambda r: r["max_diff"] / min(cap, start * growth ** times.index(r["t"])))
    k = times.index(worst["t"])
    print(f"tolerance {start:g} x {growth:g}^k (cap {cap:g}): "
          f"{'pass' if ok else 'FAIL'}, closest {worst['quantity']} t {worst['t']:g}: "
          f"{worst['max_diff']:.2e} of {min(cap, start * growth**k):.2e}")
    return ok


if __name__ == "__main__":
    sys.exit(main())
