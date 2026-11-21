"""Live comparisons, no diagnostics: at each dump time of a case (cases.dump_every), each
quantity component's composite on the finest grid (compare_physics.composite_of) straight from the
running simulator's data wrangler (PHARE's or plamn's, the same PatchData: lower, upper, nGhosts,
data with ghosts).

  Snapshots(simulator, ndim, quantities)   take(): the composites at the current time
  save(path, key, snapshots)               the reference: one compressed npz (and its key, what
                                           made it: a stale one is never compared against)
  load(path)                               (key, {(time, name): array, (time, "levels"): n})
  Comparer(reference)                      against(snapshot): compare_physics.compare_pairs'
                                           rows at the snapshot's time

One rank: the composites are of the local patches (comparisons run on one rank).
"""

import json

import numpy as np

from compare_physics import compare_pairs, components, composite_of, field


def _level(dw, ln):
    cpp = dw.cpp  # PHARE's C++ wrangler: getPatchLevel; plamn's: getHybridPatchLevel
    level = getattr(cpp, "getPatchLevel", None) or cpp.getHybridPatchLevel
    return level(ln)


def level_patches(dw, ln, qty, comp, ndim):
    """level ln's patches of qty's component as (lower, upper, ghosts, values with ghosts)"""
    _, _, primal = field(qty, comp, ndim)
    get = getattr(_level(dw, ln), f"get{qty}")
    out = []
    for p in get(comp):
        lo = np.asarray(p.lower, dtype=int)
        up = np.asarray(p.upper, dtype=int)
        g = int(p.nGhosts)
        shape = tuple(up[d] - lo[d] + 1 + 2 * g + (d in primal) for d in range(ndim))
        v = np.asarray(p.data, dtype=np.float64)
        if v.size != np.prod(shape):
            raise RuntimeError(f"live: {qty}{comp} patch {lo}..{up} has {v.size} values, "
                               f"{shape} expected")
        out.append((lo, up, g, np.array(v.reshape(shape))))  # a copy: the views are live
    return out


class Snapshots:
    def __init__(self, simulator, ndim, quantities=("B", "E")):
        self.simulator, self.ndim, self.quantities = simulator, ndim, quantities

    def take(self):
        """(time, number of levels, {qty + component: composite})"""
        dw = self.simulator.data_wrangler()
        n = dw.getNumberOfLevels()
        t = float(self.simulator.currentTime())
        out = {}
        for qty in self.quantities:
            for c in components(qty):
                _, _, primal = field(qty, c, self.ndim)
                levels = [level_patches(dw, ln, qty, c, self.ndim) for ln in range(n)]
                out[qty + c] = composite_of(levels, primal, self.ndim)
        return t, n, out


def _key(t, name):
    return f"{t!r}/{name}"


def save(path, key, snapshots):
    """snapshots: [(time, levels, {name: composite})]"""
    arrays = {"key": np.array(json.dumps(key, sort_keys=True))}
    for t, n, comps in snapshots:
        arrays[_key(t, "levels")] = np.array(n)
        for name, a in comps.items():
            arrays[_key(t, name)] = a
    np.savez_compressed(path, **arrays)


def load(path):
    """(key, the npz: read lazily) of a saved reference"""
    data = np.load(path)
    return json.loads(str(data["key"])), data


class Comparer:
    def __init__(self, reference):
        self.ref = reference
        self.times = sorted({float(k.split("/")[0]) for k in reference.files if "/" in k})

    def against(self, snapshot, quantities=("B", "E")):
        """rows (compare_physics.compare_pairs') of the snapshot against the reference at the
        closest reference time (dumps are at the same steps: equal to roundoff)"""
        t, n, comps = snapshot
        rt = min(self.times, key=lambda x: abs(x - t))
        if abs(rt - t) > 1e-9 * max(1.0, abs(t)):
            raise RuntimeError(f"live: no reference at t {t} (closest {rt})")
        rn = int(self.ref[_key(rt, "levels")])
        rows = []
        for qty in quantities:
            pairs = [(c, self.ref[_key(rt, qty + c)], comps[qty + c], rt, t, rn, n)
                     for c in components(qty)]
            rows += compare_pairs(qty, pairs)
        return rows
