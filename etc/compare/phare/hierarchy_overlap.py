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
                 quantities=None):
    """Report of every quantity (file stem) present in both directories."""
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

    for stem in stems:
        rf, cf = open_field(ref_run, ref_files[stem]), open_field(cmp_run, cmp_files[stem])
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
        if v.max_abs > atol + rtol * v.scale:
            report.fail(f"{v.quantity}.{v.component} t={v.time} L{v.level}: max |diff| "
                        f"{v.max_abs:.3e} (rel {v.max_rel:.3e}) at node {v.where}")
    return report


def print_report(report, out=sys.stdout):
    seen = set()
    print("overlap per level (first quantity per time / level):", file=out)
    for o in report.overlaps:
        if (o.time, o.level) in seen:
            continue
        seen.add((o.time, o.level))
        print(f"  t={o.time:<10g} L{o.level}: {o.overlap:6.1%}  (ref {o.ref_cells}, cmp "
              f"{o.cmp_cells}, shared {o.shared_cells} cells; patches {o.ref_patches} / "
              f"{o.cmp_patches})", file=out)
    print("values (max |ref - cmp| over the compared nodes):", file=out)
    for v in report.values:
        print(f"  {v.quantity + '.' + v.component:<40} t={v.time:<10g} L{v.level} "
              f"nodes {v.nodes:<7} abs {v.max_abs:.3e} rel {v.max_rel:.3e}", file=out)
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
