#!/usr/bin/env python3
"""A hybrid run's step modelled from its levels' boxes alone, no particles, no MPI: each level's
boxes chopped into patches (as plamn's level_of_), each patch's cost from the measured cost terms
(cost_terms.py: a particle, a cell, a ghost cell, a patch, seconds a substep), the patches
partitioned over N ranks (damn's partition along the Hilbert curve, plamn's "levels"), and a
coarse step's time: the sum over levels of their substeps times the busiest rank's cost (levels
one after another). Particles per patch: the level 0 particles per cell (ppc above the cut-off,
uniform: harris) times the refined particles per particle over ratio^ndim a level.

  python3 etc/bench/amr/model_hybrid.py [--cells 400 800 200] [--fractions 0.4 0.8]
      [--ppc 100] [--patch 20] [--ranks 1 2 10 100] [--ranks-per-node 64]

Levels: level 1 the two harris sheets' bands across y (the fraction of level 0), level 2 the middle
of each band (the fraction of level 1), every band across x and z. Exchanges: per level, the patch
faces between patches of different ranks (cells: the face times the reach, both ways), by node.

Costs default to 3D's (cost_terms.py --ndim 3, one rank, -g 0, this machine): a coarse step's
modelled time, not a prediction for another machine; exchanges are counted, not timed.
"""

import argparse
import time

import numpy as np

# 3D, interp 1, AoSMapped, one rank, -g 0 (cost_terms.py --ndim 3 --cells 40 --sizes 5 10 20 40)
COSTS_3D = dict(particles=1.259e-07, cells=2.472e-07, ghost=1.703e-07, patch=2.087e-04)
REACH = 4


def bands(n_y, fraction, centres=(0.3, 0.7)):
    """y ranges (lower, upper) of len(centres) bands around those fractions of n_y, together the
    fraction of n_y"""
    width = int(round(fraction * n_y / len(centres)))
    out = []
    for c in centres:
        lo = int(round(c * n_y - width / 2))
        out.append((lo, lo + width - 1))
    return out


def level_boxes(cells, ratio, fractions):
    """per level its boxes [(lower, upper)] in its index space: level 0 the domain, level l the
    middle fraction of each of level l-1's bands (refined), across x and z"""
    nd = len(cells)
    levels = [[([0] * nd, [c - 1 for c in cells])]]
    y_ranges = None
    for ln, f in enumerate(fractions, start=1):
        n = [c * ratio**ln for c in cells]
        if y_ranges is None:  # level 1: the sheets' bands of level 0, refined
            y_ranges = [(lo * ratio, (up + 1) * ratio - 1) for lo, up in bands(cells[1], f)]
        else:  # the middle of each coarser band, refined
            shrunk = []
            for lo, up in y_ranges:
                w = up - lo + 1
                keep = int(round(f * w)) // ratio * ratio
                start = lo + (w - keep) // 2 // ratio * ratio
                shrunk.append((start * ratio, (start + keep) * ratio - 1))
            y_ranges = shrunk
        boxes = []
        for lo, up in y_ranges:
            lower, upper = [0] * nd, [x - 1 for x in n]
            lower[1], upper[1] = lo, up
            boxes.append((lower, upper))
        levels.append(boxes)
    return levels


def chop(plamn_etc, boxes, size, ratio, ln):
    """plamn's level_of_ chop: refined levels in whole coarser cells, the size a multiple"""
    nd = len(size)
    s = list(size) if ln == 0 else [max(ratio, x // ratio * ratio) for x in size]
    fn = getattr(plamn_etc, f"chop_{nd}")
    out = []
    for lo, up in boxes:
        out += fn(lo, up, s, ratio if ln > 0 else 1, "even")
    return out


def costs_of(patches, ppc_l, costs, reach):
    lo = np.array([p[0] for p in patches])
    up = np.array([p[1] for p in patches])
    n = up - lo + 1
    cells = n.prod(axis=1).astype(float)
    ghost = (n + 2 * reach).prod(axis=1) - cells
    particles = cells * ppc_l
    return (costs["particles"] * particles + costs["cells"] * cells + costs["ghost"] * ghost
            + costs["patch"]), cells, particles


CLASSES = ("same rank", "same NUMA", "other NUMA", "other node")


def where(a, b, node_of_rank, numa_of_rank):
    """the distance class from rank a to rank b (CLASSES' index)"""
    if a == b:
        return 0
    if node_of_rank[a] != node_of_rank[b]:
        return 3
    return 1 if numa_of_rank[a] == numa_of_rank[b] else 2


def crossing(patches, ranks, node_of_rank, extent, reach, numa_of_rank=None):
    """ghost cells exchanged both ways between face neighbours of different ranks, per distance
    class (CLASSES, same rank: 0); neighbours along each axis through the chop's grid (a
    neighbour's lower is this patch's upper + 1, the other lowers the same), periodic within
    extent"""
    numa_of_rank = numa_of_rank or [0] * len(node_of_rank)
    index = {tuple(p[0]): i for i, p in enumerate(patches)}
    out = np.zeros(len(CLASSES))
    for i, (lo, up) in enumerate(patches):
        for d in range(len(lo)):
            key = list(lo)
            key[d] = (up[d] + 1) % extent[d]
            j = index.get(tuple(key))
            if j is None or ranks[j] == ranks[i]:
                continue
            area = np.prod([up[e] - lo[e] + 1 for e in range(len(lo)) if e != d])
            out[where(ranks[i], ranks[j], node_of_rank, numa_of_rank)] += 2 * area * reach
    return out


class OwnerGrid:
    """a level's patch owners on the grid of all its patches' edges (a level's patches: each box
    chopped into a grid, so a coarse cell's owner is one slice): any box's overlap with each
    owner from a slice of it, the overlap per grid cell separable along the axes"""

    def __init__(self, patches, ranks):
        lo = np.array([q[0] for q in patches])
        up = np.array([q[1] for q in patches]) + 1  # exclusive
        self.edges = [np.unique(np.concatenate([lo[:, d], up[:, d]])) for d in range(lo.shape[1])]
        self.owner = np.full([len(e) - 1 for e in self.edges], -1, dtype=np.int64)
        i0 = [np.searchsorted(e, lo[:, d]) for d, e in enumerate(self.edges)]
        i1 = [np.searchsorted(e, up[:, d]) for d, e in enumerate(self.edges)]
        for k, rk in enumerate(ranks):
            self.owner[tuple(slice(i0[d][k], i1[d][k]) for d in range(len(self.edges)))] = rk

    def overlap(self, lower, upper):
        """(owners, cells) of the box lower..upper (inclusive) with the patches, per grid cell"""
        sl, lengths = [], []
        for d, e in enumerate(self.edges):
            lo, hi = lower[d], upper[d] + 1
            a = max(int(np.searchsorted(e, lo, side="right")) - 1, 0)
            b = min(int(np.searchsorted(e, hi, side="left")), len(e) - 1)
            if b <= a:
                return np.zeros(0, dtype=np.int64), np.zeros(0)
            sl.append(slice(a, b))
            lengths.append(np.clip(np.minimum(e[a + 1:b + 1], hi) - np.maximum(e[a:b], lo), 0,
                                   None))
        cells = lengths[0]
        for length in lengths[1:]:
            cells = np.multiply.outer(cells, length)
        owners = self.owner[tuple(sl)]
        keep = owners >= 0
        return owners[keep], cells[keep].astype(float)


def under_by_distance(fine, fine_ranks, coarse, coarse_ranks, node_of_rank, ratio,
                      numa_of_rank=None):
    """the coarse cells under the fine patches (each coarsened) by the distance class from the
    fine patch's rank to their owner (CLASSES' shares; all same rank: every core's levels in the
    same place)"""
    grid = OwnerGrid(coarse, coarse_ranks)
    node = np.asarray(node_of_rank)
    numa = np.asarray(numa_of_rank if numa_of_rank is not None else [0] * len(node_of_rank))
    out = np.zeros(len(CLASSES))
    for (lo, up), r in zip(fine, fine_ranks):
        owners, cells = grid.overlap(np.asarray(lo) // ratio, np.asarray(up) // ratio)
        same_node = node[owners] == node[r]
        out[0] += cells[owners == r].sum()
        out[1] += cells[(owners != r) & same_node & (numa[owners] == numa[r])].sum()
        out[2] += cells[same_node & (numa[owners] != numa[r])].sum()
        out[3] += cells[~same_node].sum()
    return out / out.sum() if out.sum() else out


def majority_owner(fine, coarse, coarse_ranks, ratio, n_ranks):
    """each fine patch's rank: the owner of most of the coarse cells under it (coarsened)"""
    grid = OwnerGrid(coarse, coarse_ranks)
    out = []
    for lo, up in fine:
        owners, cells = grid.overlap(np.asarray(lo) // ratio, np.asarray(up) // ratio)
        out.append(int(np.argmax(np.bincount(owners, weights=cells, minlength=n_ranks))))
    return out


def owner_above(coarse, fine, fine_ranks, ratio, n_ranks):
    """each coarse patch's rank: the owner of most of the fine cells above it (its box refined),
    -1 where no fine patch is above"""
    grid = OwnerGrid(fine, fine_ranks)
    out = []
    for lo, up in coarse:
        owners, cells = grid.overlap(np.asarray(lo) * ratio, (np.asarray(up) + 1) * ratio - 1)
        out.append(int(np.argmax(np.bincount(owners, weights=cells, minlength=n_ranks)))
                   if len(owners) else -1)
    return out


def spill(ranks, weights, node_of_rank, numa_of_rank, tol):
    """no rank above (1 + tol) of the mean: an overloaded rank, the most loaded first, hands
    patches (the heaviest that fits in its excess, else its lightest) to the least loaded rank
    with room, the closest first (its NUMA zone, its node, any: damn::amr::spill's order); it
    keeps its last patch"""
    ranks = list(ranks)
    w = np.asarray(weights, dtype=float)
    n = len(node_of_rank)
    load = np.bincount(ranks, weights=w, minlength=n)
    limit = (1 + tol) * load.sum() / n
    mine = [[] for _ in range(n)]
    for i, rk in enumerate(ranks):
        mine[rk].append(i)
    node, numa = np.asarray(node_of_rank), np.asarray(numa_of_rank)
    for rk in np.argsort(-load, kind="stable"):
        while load[rk] > limit and len(mine[rk]) > 1:
            excess = load[rk] - limit
            own = mine[rk]
            fits = [i for i in own if w[i] <= excess]
            i = max(fits, key=lambda j: w[j]) if fits else min(own, key=lambda j: w[j])
            room = np.nonzero(load + w[i] <= limit)[0]
            room = room[room != rk]
            best = None
            for tier in (room[(node[room] == node[rk]) & (numa[room] == numa[rk])],
                         room[node[room] == node[rk]], room):
                if len(tier):
                    best = int(tier[np.argmin(load[tier])])
                    break
            if best is None:
                break
            own.remove(i)
            mine[best].append(i)
            load[rk] -= w[i]
            load[best] += w[i]
            ranks[i] = best
    return ranks


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cells", type=int, nargs="+", default=[400, 800, 200])
    p.add_argument("--fractions", type=float, nargs="+", default=[0.4, 0.8])
    p.add_argument("--ratio", type=int, default=2)
    p.add_argument("--ppc", type=float, default=100)
    p.add_argument("--refined-nbr", type=int, default=6, help="refined particles a particle")
    p.add_argument("--patch", type=int, nargs="+", default=[20], help="largest patch size")
    p.add_argument("--topologies", nargs="+", default=["4:2", "100:10", "1000:10"],
                   help="ranks:ranks per node[:NUMA zones per node, default --numa]")
    p.add_argument("--numa", type=int, default=2, help="NUMA zones a node (dual socket)")
    p.add_argument("--partition", nargs="+", default=["levels"],
                   choices=["levels", "aligned", "finest"],
                   help="levels: each level along the curve (plamn's); aligned: level 0 along "
                   "the curve, a finer patch to the owner of most of the coarse cells under it, "
                   "then spilled (--spill); finest: the finest level along the curve, a coarser "
                   "patch to the owner of most of the fine cells above it (the rest along the "
                   "curve), then spilled")
    p.add_argument("--spill", type=float, nargs="+", default=[0.1],
                   help="aligned: a rank's tolerance over the mean before it hands patches on")
    p.add_argument("--no-exchanges", action="store_true", help="skip the face counts (big runs)")
    p.add_argument("--locality", action="store_true",
                   help="each level's coarse cells under its patches on the same rank")
    a = p.parse_args()

    from plamn import plamn_etc

    nd, r = len(a.cells), a.ratio
    costs = COSTS_3D
    boxes = level_boxes(a.cells, r, a.fractions)
    for size in a.patch:
        t0 = time.monotonic()
        levels = [chop(plamn_etc, b, [size] * nd, r, ln) for ln, b in enumerate(boxes)]
        per_level = []
        for ln, patches in enumerate(levels):
            ppc_l = a.ppc * (a.refined_nbr / r**nd) ** ln
            w, cells, particles = costs_of(patches, ppc_l, costs, REACH)
            per_level.append((patches, w, cells.sum(), particles.sum()))
        print(f"\npatch {size}^{nd}: " + ", ".join(
            f"L{ln} {len(pt)} patches {c / 1e6:.0f}M cells {q / 1e9:.1f}G particles"
            for ln, (pt, _, c, q) in enumerate(per_level)) + f" ({time.monotonic() - t0:.1f} s)")
        serial = sum(r**ln * w.sum() for ln, (_, w, _, _) in enumerate(per_level))
        print(f"  1 core: {serial:.0f} s a coarse step (L0 1, L1 {r}, L2 {r * r} substeps)")
        print(f"  {'topology':>9} {'partition':>12} {'step s':>9} {'speedup':>8} {'eff':>6} "
              + " ".join(f"{'imb L' + str(ln):>8}" for ln in range(len(levels))))
        strategies = [(m, t) for m in a.partition
                      for t in (a.spill if m in ("aligned", "finest") else [None])]
        for (mode, tol), topology in ((x, t) for t in a.topologies for x in strategies):
            fields = [int(x) for x in topology.split(":")]
            n, per_node = fields[0], fields[1]
            zones = fields[2] if len(fields) > 2 else a.numa
            per_zone = max(1, -(-per_node // zones))  # a node's ranks in halves (dual socket)
            node_of_rank = [k // per_node for k in range(n)]
            numa_of_rank = [(k % per_node) // per_zone for k in range(n)]
            partition = getattr(plamn_etc, f"partition_{nd}")
            step, imbs, xc = 0.0, [], []
            curve = lambda patches, w: partition(  # noqa: E731
                [(lo, up) for lo, up in patches], list(map(float, w)), node_of_rank,
                numa_of_rank)
            owners = [None] * len(per_level)
            if mode == "finest":  # the finest level first, each coarser one under it
                for ln in reversed(range(len(per_level))):
                    patches, w = per_level[ln][0], per_level[ln][1]
                    if ln == len(per_level) - 1:
                        owners[ln] = curve(patches, w)
                        continue
                    ranks = owner_above(patches, per_level[ln + 1][0], owners[ln + 1], r, n)
                    rest = [i for i, rk in enumerate(ranks) if rk < 0]
                    if rest:
                        for i, rk in zip(rest, curve([patches[i] for i in rest], w[rest])):
                            ranks[i] = rk
                    owners[ln] = spill(ranks, w, node_of_rank, numa_of_rank, tol)
            for ln, (patches, w, _, _) in enumerate(per_level):
                if mode == "finest":
                    ranks = owners[ln]
                elif mode == "aligned" and ln > 0:
                    ranks = majority_owner(patches, per_level[ln - 1][0], owners[ln - 1], r, n)
                    ranks = spill(ranks, w, node_of_rank, numa_of_rank, tol)
                else:
                    ranks = curve(patches, w)
                owners[ln] = ranks
                load = np.bincount(ranks, weights=w, minlength=n)
                step += r**ln * load.max()
                imbs.append(load.max() / load.mean())
                if not a.no_exchanges:
                    extent = [c * r**ln for c in a.cells]
                    xc.append(crossing(patches, ranks, node_of_rank, extent, REACH,
                                       numa_of_rank))
            name = mode if tol is None else f"{mode} {tol:g}"
            print(f"  {topology:>9} {name:>12} {step:>9.1f} {serial / step:>8.1f} {serial / step / n:>6.2f} "
                  + " ".join(f"{i:>8.3f}" for i in imbs))
            if not a.no_exchanges:
                for ln, (by, cells) in enumerate(zip(xc, [c for _, _, c, _ in per_level])):
                    print(f"{'':14}L{ln} ghosts to other ranks a substep: "
                          + ", ".join(f"{name} {v / 1e6:.1f}M" for name, v in
                                      zip(CLASSES[1:], by[1:]))
                          + f" ({by[1:].sum() / cells:.1%} of the level's cells)")
            if a.locality:
                for ln in range(1, len(per_level)):
                    d = under_by_distance(per_level[ln][0], owners[ln], per_level[ln - 1][0],
                                          owners[ln - 1], node_of_rank, r, numa_of_rank)
                    print(f"{'':14}L{ln} over L{ln - 1}: coarse cells "
                          + ", ".join(f"{name} {v:.0%}" for name, v in zip(CLASSES, d)))


if __name__ == "__main__":
    main()
