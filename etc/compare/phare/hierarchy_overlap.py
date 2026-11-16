#!/usr/bin/env python3
"""Compare two diagnostics directories (PHARE vs plamn, or any two runs of one input) whose
patches differ: per level, how much the two hierarchies overlap, and the values where they do.

pyphare's hierarchy_compare pairs patches by index, so it needs identical hierarchies. Here
each level is compared as a set of cells:

  overlap   |A n B| / |A u B| over the level's cells (patches of a level are disjoint, so
            |A n B| is the sum of the pairwise patch intersections). Level 0 must be 1, finer
            levels at least --min-overlap (tagging decides them; they need not be identical).
  values    every quantity / component on the nodes of each patch intersection, except nodes
            within --margin cells of either level's border: a refined level's border nodes
            are filled from the coarser level, so they differ wherever the levels differ.
            Also skipped: nodes of a level within --margin cells of where the two runs' next
            finer levels disagree (one refines there, the other does not), as a refined
            region's values are coarsened onto the level below it.
            Pass: max |ref - cmp| <= atol + rtol * max |ref| (over the compared nodes).

Files are read with the pharesee reader of PHAREHUB/PHARE#1595 (pyphare.pharesee.amr):
.vtkhdf and .h5 alike, all primal. Put that pyphare on PYTHONPATH.

  python3 etc/compare/phare/hierarchy_overlap.py REF_DIR CMP_DIR [--min-overlap 0.9] ...
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EXTENSIONS = (".vtkhdf", ".h5")
TIME_ATOL = 1e-10


@dataclass
class LevelOverlap:
    quantity: str
    time: float
    level: int
    ref_cells: int
    cmp_cells: int
    shared_cells: int
    ref_patches: int
    cmp_patches: int

    @property
    def overlap(self):
        union = self.ref_cells + self.cmp_cells - self.shared_cells
        return self.shared_cells / union if union else 1.0


@dataclass
class ValueDiff:
    quantity: str
    component: str
    time: float
    level: int
    nodes: int
    max_abs: float
    scale: float
    where: tuple

    @property
    def max_rel(self):
        return self.max_abs / self.scale if self.scale else self.max_abs


@dataclass
class Report:
    """What was compared and what failed; truthy when nothing failed (as EqualityReport)."""

    overlaps: list = field(default_factory=list)
    values: list = field(default_factory=list)
    particles: list = field(default_factory=list)
    failed: list = field(default_factory=list)

    def __bool__(self):
        return not self.failed

    def fail(self, reason):
        self.failed.append(reason)


def diagnostic_files(directory):
    """{quantity stem: path} of the diagnostic files in ``directory``."""
    files = {}
    for path in sorted(Path(directory).iterdir()):
        if path.suffix in EXTENSIONS and path.stem not in files:
            files[path.stem] = path
    return files


def open_field(run, path):
    """Lazy AMR field of every time in ``path`` (PHARE#1595's Run opener)."""
    return run._open_path(str(path), None, None)


def common_times(ref_times, cmp_times, atol=TIME_ATOL):
    """(times in both, times in only one) for two time arrays."""
    both = [t for t in ref_times if np.isclose(cmp_times, t, rtol=0, atol=atol).any()]
    only = [t for t in ref_times if t not in both]
    only += [t for t in cmp_times if not np.isclose(both, t, rtol=0, atol=atol).any()]
    return both, only


def cell_count(lower, upper):
    return int(np.prod(upper - lower + 1, axis=1).sum()) if len(lower) else 0


def covered_cells(geom):
    """Boolean array over the level's cells: True where a patch of the level is."""
    cover = np.zeros(tuple(geom.domain_cells), dtype=bool)
    for lo, hi in zip(geom.lower, geom.upper):
        cover[tuple(slice(l, h + 1) for l, h in zip(lo, hi))] = True
    return cover


def covered_from_finer(geom, fine):
    """Cells of level ``geom`` under the finer level ``fine`` (None: no finer level)."""
    cover = np.zeros(tuple(geom.domain_cells), dtype=bool)
    if fine is None:
        return cover
    ratio = fine.domain_cells // geom.domain_cells
    for lo, hi in zip(fine.lower // ratio, fine.upper // ratio):
        cover[tuple(slice(l, h + 1) for l, h in zip(lo, hi))] = True
    return cover


def cells_to_nodes(cells, periodic):
    """Nodes (cells + 1 per direction) whose adjacent cells are all True."""
    padded = cells
    for axis, p in enumerate(periodic):  # one cell each side: a node touches cells i-1 and i
        pad = [(0, 0)] * cells.ndim
        pad[axis] = (1, 1)
        padded = np.pad(padded, pad, mode="wrap" if p else "edge")
    n = np.array(cells.shape)
    nodes = np.ones(tuple(n + 1), dtype=bool)
    for offset in np.ndindex(*([2] * cells.ndim)):
        nodes &= padded[tuple(slice(o, o + c + 1) for o, c in zip(offset, n))].astype(bool)
    return nodes


def finer_agreement_nodes(geom, ref_fine, cmp_fine, margin):
    """Nodes of level ``geom`` at least ``margin`` cells from any cell that only one run
    refines (its values come from different finer data in the two runs)."""
    from scipy.ndimage import maximum_filter

    differ = covered_from_finer(geom, ref_fine) ^ covered_from_finer(geom, cmp_fine)
    if margin > 0 and differ.any():
        modes = ["wrap" if p else "nearest" for p in geom.periodic]
        differ = maximum_filter(differ.astype(np.uint8), size=2 * margin + 1, mode=modes)
    return cells_to_nodes(~differ.astype(bool), geom.periodic)


def interior_nodes(geom, margin):
    """Boolean array over the level's nodes (cells + 1 per direction): True for nodes whose
    cells within ``margin`` cells all belong to the level. Periodic directions wrap; past a
    non periodic domain edge counts as inside (a physical boundary is no level border)."""
    from scipy.ndimage import minimum_filter

    modes = ["wrap" if p else "nearest" for p in geom.periodic]
    cells = covered_cells(geom).astype(np.uint8)
    if margin > 0:
        cells = minimum_filter(cells, size=2 * margin + 1, mode=modes)
    return cells_to_nodes(cells.astype(bool), geom.periodic)


def compare_level(quantity, time, ref_ld, cmp_ld, ref_fine, cmp_fine, margin, report):
    """One level of one quantity at one time; ``*_fine``: geometry of the next finer level
    of each run, or None."""
    from pyphare.pharesee.amr.boxes import intersect_tables

    rg, cg = ref_ld.geom, cmp_ld.geom
    pairs = intersect_tables(rg.lower, rg.upper, cg.lower, cg.upper)
    shared = int(np.prod(pairs.hi - pairs.lo + 1, axis=1).sum()) if len(pairs.a) else 0
    report.overlaps.append(
        LevelOverlap(quantity, time, ref_ld.level, cell_count(rg.lower, rg.upper),
                     cell_count(cg.lower, cg.upper), shared, rg.npatch, cg.npatch)
    )
    mask = interior_nodes(rg, margin) & interior_nodes(cg, margin)
    mask &= finer_agreement_nodes(rg, ref_fine, cmp_fine, margin)

    for comp in ref_ld.names:
        if comp not in cmp_ld.names:
            report.fail(f"{quantity} t={time} L{ref_ld.level}: no component {comp} in cmp")
            continue
        worst = ValueDiff(quantity, comp, time, ref_ld.level, 0, 0.0, 0.0, ())
        for a, b, lo, hi in zip(pairs.a, pairs.b, pairs.lo, pairs.hi):
            nodes = tuple(slice(l, h + 2) for l, h in zip(lo, hi))
            m = mask[nodes]
            if not m.any():
                continue
            ra = ref_ld.view(a, comp)[tuple(slice(l - o, h + 2 - o)
                                            for l, h, o in zip(lo, hi, rg.lower[a]))]
            cb = cmp_ld.view(b, comp)[tuple(slice(l - o, h + 2 - o)
                                            for l, h, o in zip(lo, hi, cg.lower[b]))]
            diff = np.where(m, np.abs(ra - cb), 0.0)
            worst.nodes += int(m.sum())
            worst.scale = max(worst.scale, float(np.abs(ra[m]).max()))
            i = np.unravel_index(np.argmax(diff), diff.shape)
            if not worst.where or diff[i] > worst.max_abs:
                worst.max_abs = float(diff[i])
                worst.where = tuple(int(l + k) for l, k in zip(lo, i))
        report.values.append(worst)


def compare_runs(ref_dir, cmp_dir, min_overlap=0.9, atol=0.0, rtol=1e-12, margin=2,
                 quantities=None, tolerance=None):
    """Report of every quantity (file stem) present in both directories. ``tolerance``:
    optional ``stem -> (atol, rtol)``, e.g. per kind (B, E, moments) as test_layouts."""
    tolerance = tolerance or (lambda stem: (atol, rtol))
    from pyphare.pharesee.run import Run

    report = Report()
    ref_files, cmp_files = diagnostic_files(ref_dir), diagnostic_files(cmp_dir)
    stems = sorted(set(ref_files) & set(cmp_files))
    if quantities:
        stems = [s for s in stems if s in quantities]
    for stem in sorted(set(ref_files) ^ set(cmp_files)):
        if not quantities or stem in quantities:
            report.fail(f"{stem}: only in {'ref' if stem in ref_files else 'cmp'}")
    ref_run, cmp_run = Run(str(ref_dir)), Run(str(cmp_dir))

    from pyphare.pharesee.amr.errors import FormatError

    for stem in stems:
        try:  # e.g. PHARE master's vtkhdf momentum_tensor: the file but no data
            rf = open_field(ref_run, ref_files[stem])
            cf = open_field(cmp_run, cmp_files[stem])
        except FormatError as e:
            report.fail(f"{stem}: unreadable: {str(e).split(': ', 1)[-1][:120]}")
            continue
        times, only = common_times(rf.times(), cf.times())
        for t in only:
            report.fail(f"{stem}: time {t} in one run only")
        for t in times:
            rl = {ld.level: ld for ld in rf.levels_data(t)}
            cl = {ld.level: ld for ld in cf.levels_data(t)}
            for lvl in sorted(set(rl) ^ set(cl)):
                report.fail(f"{stem} t={t}: level {lvl} in {'ref' if lvl in rl else 'cmp'} only")
            for lvl in sorted(set(rl) & set(cl)):
                rfine, cfine = rl.get(lvl + 1), cl.get(lvl + 1)
                compare_level(stem, t, rl[lvl], cl[lvl], rfine and rfine.geom,
                              cfine and cfine.geom, margin, report)

    for o in report.overlaps:
        need = 1.0 if o.level == 0 else min_overlap
        if o.overlap < need:
            report.fail(f"{o.quantity} t={o.time} L{o.level}: overlap {o.overlap:.3f} < {need}")
    for v in report.values:
        v_atol, v_rtol = tolerance(v.quantity)
        if v.max_abs > v_atol + v_rtol * v.scale:
            report.fail(f"{v.quantity}.{v.component} t={v.time} L{v.level}: max |diff| "
                        f"{v.max_abs:.3e} (rel {v.max_rel:.3e}) at node {v.where}")
    return report


@dataclass
class ParticleDiff:
    pop: str
    kind: str  # domain (cells in the level) or ghost (level ghost particles, cells outside it)
    step: int
    time: float
    level: int
    ref_count: int
    cmp_count: int
    cells: int  # compared cells
    cells_count_differs: int  # cells whose number of particles differs
    max_delta: float
    max_v: float
    max_weight: float


def _in_boxes(icell, lower, upper):
    inside = np.zeros(len(icell), dtype=bool)
    for lo, hi in zip(lower, upper):
        inside |= np.all((icell >= lo) & (icell <= hi), axis=1)
    return inside


def _sorted(parts, mask):
    p = {k: v[mask] for k, v in parts.items()}
    keys = [p["weight"]] + [p["v"][:, i] for i in range(p["v"].shape[1])[::-1]]
    keys += [p["delta"][:, i] for i in range(p["delta"].shape[1])[::-1]]
    keys += [p["icell"][:, i] for i in range(p["icell"].shape[1])[::-1]]
    order = np.lexsort(keys)  # icell first, then delta, v, weight
    return {k: v[order] for k, v in p.items()}


def compare_particle_sets(r, c):
    """(cells, cells whose count differs, max |delta|, |v|, |weight| diffs over the cells with
    equal counts) of two sorted particle sets."""
    rc, ri, rn = np.unique(r["icell"], axis=0, return_index=True, return_counts=True)
    cc, ci, cn = np.unique(c["icell"], axis=0, return_index=True, return_counts=True)
    cells = {tuple(x): (i, n) for x, i, n in zip(rc, ri, rn)}
    other = {tuple(x): (i, n) for x, i, n in zip(cc, ci, cn)}
    differs, md, mv, mw = 0, 0.0, 0.0, 0.0
    for cell in set(cells) | set(other):
        if cell not in cells or cell not in other or cells[cell][1] != other[cell][1]:
            differs += 1
            continue
        (i, n), (j, _) = cells[cell], other[cell]
        md = max(md, float(np.abs(r["delta"][i:i + n] - c["delta"][j:j + n]).max()))
        mv = max(mv, float(np.abs(r["v"][i:i + n] - c["v"][j:j + n]).max()))
        mw = max(mw, float(np.abs(r["weight"][i:i + n] - c["weight"][j:j + n]).max()))
    return len(set(cells) | set(other)), differs, md, mv, mw


def _level_boxes(run, path, time):
    """{level: (lower, upper)} of the diagnostics file at ``path`` at ``time``."""
    f = open_field(run, path)
    return {ld.level: (np.asarray(ld.geom.lower), np.asarray(ld.geom.upper))
            for ld in f.levels_data(time)}


def compare_particles(ref_dir, cmp_dir, report, atol=1e-12, boxes_from="EM_B"):
    """Domain particles saved by run_case.py (<dir>/particles/<step>.npz): per step,
    population and level, on the cells both levels hold (level boxes from each run's
    ``boxes_from`` diagnostics at that time), matched by cell (sorted by cell, delta, v,
    weight). Appends to report."""
    from pyphare.pharesee.run import Run

    ref_files = {int(p.stem): p for p in (Path(ref_dir) / "particles").glob("*.npz")}
    cmp_files = {int(p.stem): p for p in (Path(cmp_dir) / "particles").glob("*.npz")}
    runs = (Run(str(ref_dir)), Run(str(cmp_dir)))
    paths = (diagnostic_files(ref_dir)[boxes_from], diagnostic_files(cmp_dir)[boxes_from])
    for step in sorted(set(ref_files) & set(cmp_files)):
        r, c = np.load(ref_files[step]), np.load(cmp_files[step])
        time = float(r["time"][0])
        ref_boxes, cmp_boxes = (_level_boxes(run, path, time) for run, path in zip(runs, paths))
        keys = sorted({k.rsplit("/", 1)[0] for k in r.files if k != "time"}
                      | {k.rsplit("/", 1)[0] for k in c.files if k != "time"})
        for key in keys:
            pop, lvl = key.split("/")
            level = int(lvl[1:])
            if f"{key}/icell" not in r.files or f"{key}/icell" not in c.files:
                report.fail(f"particles {key} step {step}: in one run only")
                continue
            rp = {n: r[f"{key}/{n}"] for n in ("icell", "delta", "v", "weight")}
            cp = {n: c[f"{key}/{n}"] for n in ("icell", "delta", "v", "weight")}
            both = [ref_boxes[level], cmp_boxes[level]]
            rm = _in_boxes(rp["icell"], *both[0]) & _in_boxes(rp["icell"], *both[1])
            cm = _in_boxes(cp["icell"], *both[0]) & _in_boxes(cp["icell"], *both[1])
            rs, cs = _sorted(rp, rm), _sorted(cp, cm)
            cells, differs, md, mv, mw = compare_particle_sets(rs, cs)
            d = ParticleDiff(pop, "domain", step, time, level, int(rm.sum()), int(cm.sum()),
                             cells, differs, md, mv, mw)
            report.particles.append(d)
            if differs or max(md, mv, mw) > atol:
                report.fail(f"particles {pop} step {step} L{level}: {d.ref_count} / "
                            f"{d.cmp_count}, {differs} of {cells} cells differ in count, max "
                            f"|delta| {md:.2e} |v| {mv:.2e} |weight| {mw:.2e}")
    return report


def print_report(report, out=sys.stdout, values=True):
    """Overlaps, then (``values``) every value comparison, then the failures."""
    seen = set()
    print("overlap per level (first quantity per time / level):", file=out)
    for o in report.overlaps:
        if (o.time, o.level) in seen:
            continue
        seen.add((o.time, o.level))
        print(f"  t={o.time:<10g} L{o.level}: {o.overlap:6.1%}  (ref {o.ref_cells}, cmp "
              f"{o.cmp_cells}, shared {o.shared_cells} cells; patches {o.ref_patches} / "
              f"{o.cmp_patches})", file=out)
    if values:
        print("values (max |ref - cmp| over the compared nodes):", file=out)
    for v in report.values if values else []:
        print(f"  {v.quantity + '.' + v.component:<40} t={v.time:<10g} L{v.level} "
              f"nodes {v.nodes:<7} abs {v.max_abs:.3e} rel {v.max_rel:.3e}", file=out)
    if report.particles:
        print("particles (domain, cells both levels hold):", file=out)
    for d in report.particles:
        print(f"  {d.pop} step {d.step} L{d.level}: {d.ref_count} / {d.cmp_count} particles, "
              f"{d.cells_count_differs} of {d.cells} cells differ in count, max |delta| "
              f"{d.max_delta:.1e} |v| {d.max_v:.1e} |weight| {d.max_weight:.1e}", file=out)
    print(f"{len(report.failed)} failure(s)" if report.failed else "all within tolerance",
          file=out)
    for f in report.failed:
        print("  " + f, file=out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("ref", type=Path)
    p.add_argument("cmp", type=Path)
    p.add_argument("--min-overlap", type=float, default=0.9)
    p.add_argument("--atol", type=float, default=0.0)
    p.add_argument("--rtol", type=float, default=1e-12)
    p.add_argument("--margin", type=int, default=2, help="cells skipped at level borders")
    p.add_argument("--quantities", nargs="*", help="file stems (default: all in both)")
    a = p.parse_args(argv)
    report = compare_runs(a.ref, a.cmp, a.min_overlap, a.atol, a.rtol, a.margin, a.quantities)
    print_report(report)
    return 0 if report else 1


if __name__ == "__main__":
    sys.exit(main())
