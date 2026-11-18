"""Hierarchies without data: chop and partition a simulation's levels from python, before (and
without) running it, to compare decompositions.

    tag_boxes = {0: [domain], 1: [...], ...}           # e.g. pyphare's Tagger(sim).tag()
    h = Chopper(sim, ranks=6, ranks_per_node=6, mode="even").chop(tag_boxes, weights)
    h.levels[1]                                         # [(lower, upper, rank)]
    h.metrics()                                         # per level: imbalance, cells exchanged
    h.hierarchy()                                       # pharesee PatchHierarchy, "rank" field

chop: each level's boxes cut into patches as the simulator does (largest_patch_size; refined
levels in whole coarser cells), mode "even" (SAMRAI's on one rank, the simulator's) or "chop"
(from the lower corner); each level partitioned along the Hilbert curve by patch weight, each
node an equal share split over its ranks. A patch weighs its cells times the particles per cell,
or on level 0, given a work map per level 0 cell (Tagger.weights(): the work above it too), its
cells' sum of the map, so level 0 is cut by the work the levels above will bring.

metrics (damn::amr::metrics): per level the patches, cells, weight per rank and imbalance (max
over mean), and the cells exchanged by distance: same level ghosts within reach, and for levels
above 0 the coarser cells under each patch grown by reach (what refining its ghosts reads):
Self (own periodic image), Local (same rank), Numa, Node (other rank, same node), Remote.
"""

from dataclasses import dataclass, field

import numpy as np

from . import plamn_etc

# field ghost width per interpolation order (PHARE's, plamn's): the default reach
_ghosts = {1: 2, 2: 4, 3: 4}


def _box(b):
    """(lower, upper) of a pyphare Box or a (lower, upper) pair, as int lists"""
    if hasattr(b, "lower"):
        return [int(v) for v in b.lower], [int(v) for v in b.upper]
    lower, upper = b
    return [int(v) for v in np.atleast_1d(lower)], [
        int(v) for v in np.atleast_1d(upper)
    ]


def _per_dim(v, ndim):
    return [int(x) for x in np.broadcast_to(np.asarray(v), (ndim,))]


@dataclass
class NoDataHierarchy:
    ndim: int
    domain: tuple  # level 0 (lower, upper)
    periodic: list
    ratio: int
    dl: list
    origin: list
    interp_order: int
    node_of_rank: list  # each rank's node
    levels: list = field(default_factory=list)  # [[(lower, upper, rank)]]
    weights: list = field(default_factory=list)  # [[weight per patch]]

    @property
    def ranks(self):
        return len(self.node_of_rank)

    def metrics(self, reach=None):
        """damn::amr::metrics per level (a list of dicts)"""
        reach = _ghosts[self.interp_order] if reach is None else reach
        fn = getattr(plamn_etc, f"metrics_{self.ndim}")
        return fn(
            *self.domain,
            self.periodic,
            self.ratio,
            self.levels,
            self.node_of_rank,
            reach,
            self.weights,
        )

    def total_work_imbalance(self):
        """the busiest rank's work over all levels (each level's weight times r^l) over the mean:
        the step time's bound when a rank's levels overlap perfectly (level 0's work far from
        the finer levels filling the waits), where critical_path is the bound without overlap"""
        per_rank = np.zeros(self.ranks)
        for ln, (patches, weights) in enumerate(zip(self.levels, self.weights)):
            for (_, _, rank), w in zip(patches, weights):
                per_rank[rank] += self.ratio**ln * w
        return per_rank.max() / per_rank.mean() if per_rank.mean() else 1.0

    def comm_cost(self, cost=None, reach=None):
        """the cells exchanged (same level ghosts and coarse cells under each patch) weighed by
        distance: cost {distance: per cell}, placeholders until measured (Self and Local free,
        Numa 0.3, Node 1, Remote 10)"""
        cost = cost or {
            "Self": 0.0,
            "Local": 0.0,
            "Numa": 0.3,
            "Node": 1.0,
            "Remote": 10.0,
        }
        total = 0.0
        for m in self.metrics(reach):
            for key in ("ghost_cells", "coarse_cells"):
                total += sum(cost[d] * n for d, n in m[key].items())
        return total

    def migration(self, previous):
        """moving from `previous` to this hierarchy, per level: the cells by distance from their
        old owner, held by the old level (moved) or refined from its coarser one (refined)"""
        fn = getattr(plamn_etc, f"migration_{self.ndim}")
        return fn(
            *self.domain,
            self.periodic,
            self.ratio,
            previous.levels,
            self.levels,
            self.node_of_rank,
        )

    def migration_cost(self, previous, cost=None):
        """migration's cells weighed by distance (comm_cost's placeholders)"""
        cost = cost or {
            "Self": 0.0,
            "Local": 0.0,
            "Numa": 0.3,
            "Node": 1.0,
            "Remote": 10.0,
        }
        return sum(
            cost[d] * n
            for m in self.migration(previous)
            for key in ("moved", "refined")
            for d, n in m[key].items()
        )

    def critical_path(self):
        """sum over levels of r^l (substeps) times the busiest rank's weight, over the same with
        the mean: 1 when every level is balanced (levels advance one after another)"""
        busiest = ideal = 0.0
        for ln, (patches, weights) in enumerate(zip(self.levels, self.weights)):
            per_rank = np.zeros(self.ranks)
            for (_, _, rank), w in zip(patches, weights):
                per_rank[rank] += w
            busiest += self.ratio**ln * per_rank.max()
            ideal += self.ratio**ln * per_rank.mean()
        return busiest / ideal if ideal else 1.0

    def summary(self, reach=None):
        lines = []
        for m in self.metrics(reach):
            exchanged = ", ".join(
                f"{d} {m['ghost_cells'][d]}/{m['coarse_cells'][d]}"
                for d in ("Self", "Local", "Numa", "Node", "Remote")
                if m["ghost_cells"][d] or m["coarse_cells"][d]
            )
            lines.append(
                f"  L{m['level']}: {m['patches']:5d} patches {m['cells']:9d} cells"
                f" smallest {m['smallest_patch']:6d} imbalance {m['imbalance']:.3f}"
                f"  ghost/coarse cells: {exchanged}"
            )
        return "\n".join(lines)

    def hierarchy(self):
        """the patches as a pharesee PatchHierarchy (time 0), a cell centred "rank" field"""
        from pyphare.core.box import Box
        from pyphare.core.gridlayout import GridLayout
        from pyphare.pharesee.hierarchy.hierarchy import PatchHierarchy
        from pyphare.pharesee.hierarchy.patch import Patch
        from pyphare.pharesee.hierarchy.patchdata import FieldData
        from pyphare.pharesee.hierarchy.patchlevel import PatchLevel

        levels = {}
        for ln, patches in enumerate(self.levels):
            dx = np.asarray(self.dl) / self.ratio**ln
            out = []
            for i, (lower, upper, rank) in enumerate(patches):
                box = Box(np.asarray(lower), np.asarray(upper))
                origin = (np.asarray(self.origin) + box.lower * dx).tolist()
                layout = GridLayout(box, origin, dx.tolist(), self.interp_order)
                data = FieldData(
                    layout,
                    "rank",
                    np.full(box.shape, float(rank)),
                    centering=["dual"] * self.ndim,
                    ghosts_nbr=[0] * self.ndim,
                )
                out.append(Patch({"rank": data}, patch_id=f"p{ln}#{i}"))
            levels[ln] = PatchLevel(ln, out)
        lower, upper = self.domain
        return PatchHierarchy({0.0: levels}, Box(np.asarray(lower), np.asarray(upper)))


def coalesce(boxes):
    """disjoint (lower, upper) boxes merged while two are adjacent along a direction with the
    same extent in every other: passes over the directions until none merge (sorted: n log n a
    pass, for the many pieces of large domains)"""
    boxes = [(list(lo), list(up)) for lo, up in boxes]
    if not boxes:
        return boxes
    ndim = len(boxes[0][0])
    while True:
        before = len(boxes)
        for d in range(ndim):
            others = [i for i in range(ndim) if i != d]
            boxes.sort(
                key=lambda b: (
                    tuple(b[0][i] for i in others)
                    + tuple(b[1][i] for i in others)
                    + (b[0][d],)
                )
            )
            merged = []
            for lo, up in boxes:
                if merged:
                    mlo, mup = merged[-1]
                    if (
                        all(mlo[i] == lo[i] and mup[i] == up[i] for i in others)
                        and mup[d] + 1 == lo[d]
                    ):
                        mup[d] = up[d]
                        continue
                merged.append((lo, up))
            boxes = merged
        if len(boxes) == before:
            return boxes


def _minus(a, b):
    """box a without box b: up to 2 x dim disjoint boxes"""
    if _intersect(a, b) is None:
        return [a]
    lower, upper = list(a[0]), list(a[1])
    out = []
    for d in range(len(lower)):
        if lower[d] < b[0][d]:  # the part of a below b along d
            out.append((lower.copy(), upper[:d] + [b[0][d] - 1] + upper[d + 1 :]))
            lower[d] = b[0][d]
        if upper[d] > b[1][d]:  # above
            out.append((lower[:d] + [b[1][d] + 1] + lower[d + 1 :], upper.copy()))
            upper[d] = b[1][d]
    return out


def _intersect(a, b):
    lower = [max(x, y) for x, y in zip(a[0], b[0])]
    upper = [min(x, y) for x, y in zip(a[1], b[1])]
    return (lower, upper) if all(lo <= up for lo, up in zip(lower, upper)) else None


@dataclass
class Chopper:
    """partition "levels": each level cut on its own along the Hilbert curve (the simulator's);
    "curve": the same, every level ordered on level 0's curve (its patches' boxes coarsened to
    level 0): balanced per level, fine patches near the ranks of the coarse cells under them;
    "columns": level 0 cut in blocks of `block` cells, each weighed by the work of every level
    over it (a level l cell: cell_weight(l) times its r^l substeps), the blocks partitioned along
    the curve (per node, then over its ranks), every finer level split along its owner's blocks:
    a rank owns all the levels over its level 0 region. cell_weight(l): a level l cell's work
    (default the particles per cell: uniform densities above the cut-off; the simulator weighs
    from the densities, see box_weights_)"""

    sim: object
    ranks: int = 1
    ranks_per_node: int = None  # default: all on one node
    mode: str = "even"
    partition: str = "levels"
    block: int = 8
    cell_weight: object = None
    spill: float = (
        None  # columns: a rank above (1 + spill) x a level's mean sheds patches
    )

    def __post_init__(self):
        sim = self.sim
        self.ndim = sim.ndim
        self.cells = _per_dim(sim.cells, self.ndim)
        self.dl = [float(x) for x in np.broadcast_to(np.asarray(sim.dl), (self.ndim,))]
        self.origin = [
            float(x)
            for x in np.broadcast_to(np.asarray(sim.origin or 0.0), (self.ndim,))
        ]
        boundary = getattr(sim, "boundary_types", None) or "periodic"
        if isinstance(boundary, str):
            boundary = [boundary] * self.ndim
        self.periodic = [b == "periodic" for b in boundary]
        self.ratio = int(getattr(sim, "refinement_ratio", 2))
        size = getattr(sim, "largest_patch_size", None)
        self.patch_size = _per_dim(size if size is not None else 2**30, self.ndim)
        model = sim.model
        self.ppc = sum(
            model.model_dict[p]["nbrParticlesPerCell"] for p in model.populations
        )
        per_node = self.ranks_per_node or self.ranks
        self.node_of_rank = [r // per_node for r in range(self.ranks)]

    def _patch_boxes(self, ln, boxes):
        size, gran = self.patch_size, 1
        if (
            ln > 0
        ):  # refined: patches of whole coarser cells, the size a multiple of the ratio
            gran = self.ratio
            size = [max(gran, s // gran * gran) for s in self.patch_size]
        chop = getattr(plamn_etc, f"chop_{self.ndim}")
        out = []
        for b in boxes:
            lower, upper = _box(b)
            out += chop(lower, upper, size, gran, self.mode)
        return out

    def _cell_weight(self, ln):
        return float(self.cell_weight(ln) if self.cell_weight else self.ppc)

    def _weight(self, ln, lower, upper, work):
        if ln == 0 and work is not None:
            sl = tuple(slice(lo, up + 1) for lo, up in zip(lower, upper))
            return float(np.sum(work[sl]))
        cells = np.prod(np.asarray(upper) - np.asarray(lower) + 1)
        return float(cells * self._cell_weight(ln))

    def _chop_owned(self, ln, pieces):
        """pieces [(lower, upper, rank)]: each rank's coalesced, then chopped as the level's"""
        by_rank = {}
        for lo, up, r in pieces:
            by_rank.setdefault(r, []).append((lo, up))
        out = []
        for r in sorted(by_rank):
            for lo, up in self._patch_boxes(ln, coalesce(by_rank[r])):
                out.append((lo, up, r))
        return out

    def _columns(self, tag_boxes):
        """every level's (lower, upper, rank), ranks from level 0 blocks weighed by the work
        over them (per axis counts, no per cell map: large domains)"""
        r, b = self.ratio, self.block
        partition = getattr(plamn_etc, f"partition_{self.ndim}")
        grid = [-(-c // b) for c in self.cells]

        work = np.zeros(grid)  # per block: every level's work over it
        for ln, boxes in tag_boxes.items():
            f = r**ln
            scale = self._cell_weight(ln) * f
            for box in boxes:
                lower, upper = _box(box)
                counts = [
                    np.bincount(np.arange(lo, up + 1) // (f * b), minlength=n)[:n]
                    for lo, up, n in zip(lower, upper, grid)
                ]
                per_block = counts[0]
                for c in counts[1:]:
                    per_block = np.multiply.outer(per_block, c)
                work += scale * per_block

        index = list(np.ndindex(*grid))
        blocks = [
            (
                [k * b for k in idx],
                [min((k + 1) * b, c) - 1 for k, c in zip(idx, self.cells)],
            )
            for idx in index
        ]
        ranks = partition(
            blocks, [float(work[idx]) for idx in index], self.node_of_rank
        )
        owner = np.asarray(ranks).reshape(grid)

        levels = {}
        for ln in sorted(tag_boxes):
            f = r**ln
            pieces = []
            for box in tag_boxes[ln]:
                lower, upper = _box(box)
                first = [lo // (f * b) for lo in lower]
                last = [up // (f * b) for up in upper]
                for off in np.ndindex(*[hi - lo + 1 for lo, hi in zip(first, last)]):
                    k = tuple(lo + o for lo, o in zip(first, off))
                    block = (
                        [x * b * f for x in k],
                        [min((x + 1) * b, c) * f - 1 for x, c in zip(k, self.cells)],
                    )
                    if (piece := _intersect((lower, upper), block)) is not None:
                        pieces.append((*piece, int(owner[k])))
            levels[ln] = self._chop_owned(ln, pieces)
        return levels

    def _spill(self, patches, weights):
        """a level's patches with no rank above (1 + spill) x the mean: an overloaded rank hands
        patches (the largest within its excess first, else its smallest) to the least loaded
        rank of its node with room, else of any node: closest first, as cross level overlaps
        cost by distance"""
        load = np.zeros(self.ranks)
        mine = [[] for _ in range(self.ranks)]
        for i, ((_, _, r), w) in enumerate(zip(patches, weights)):
            load[r] += w
            mine[r].append(i)
        cap = (1 + self.spill) * load.mean()
        owner = [r for _, _, r in patches]
        node = np.asarray(self.node_of_rank)
        for r in np.argsort(-load):
            while load[r] > cap and len(mine[r]) > 1:
                excess = load[r] - cap
                ws = sorted(mine[r], key=lambda i: -weights[i])
                fits = [i for i in ws if weights[i] <= excess]
                i = fits[0] if fits else ws[-1]
                w = weights[i]
                room = np.where(load + w <= cap)[0]
                same = room[node[room] == node[r]]
                pool = same if len(same) else room
                if not len(pool):
                    break
                t = pool[np.argmin(load[pool])]
                if t == r:
                    break
                mine[r].remove(i)
                mine[t].append(i)
                load[r] -= w
                load[t] += w
                owner[i] = int(t)
        return [(lo, up, o) for (lo, up, _), o in zip(patches, owner)]

    def _keep(self, ln, tag_boxes, previous):
        """level ln's (lower, upper, rank): its boxes cut along the previous hierarchy's owners,
        where its level ln held the cells (their data stays), else where its coarser level
        did (the cells refine from there), else partitioned afresh; each rank's pieces then
        coalesced and chopped (as columns)"""
        r = self.ratio
        old = previous.levels[ln] if ln < len(previous.levels) else []
        coarse = previous.levels[ln - 1] if 0 < ln <= len(previous.levels) else []
        pieces, fresh = [], []
        for b in tag_boxes:
            box = _box(b)
            for lo, up, owner in old:
                if (piece := _intersect(box, (lo, up))) is not None:
                    pieces.append((*piece, owner))
            rest = [box]  # what the old level did not hold
            for lo, up, _ in old:
                rest = [p for q in rest for p in _minus(q, (lo, up))]
            for part in rest:
                left = [part]
                for lo, up, owner in coarse:
                    fine = ([x * r for x in lo], [(x + 1) * r - 1 for x in up])
                    if (piece := _intersect(part, fine)) is not None:
                        pieces.append((*piece, owner))
                        left = [p for q in left for p in _minus(q, piece)]
                fresh += left
        if fresh:
            partition = getattr(plamn_etc, f"partition_{self.ndim}")
            weights = [self._weight(ln, lo, up, None) for lo, up in fresh]
            ranks = partition(fresh, weights, self.node_of_rank)
            pieces += [(lo, up, rank) for (lo, up), rank in zip(fresh, ranks)]
        return self._chop_owned(ln, pieces)

    def chop(self, tag_boxes, work=None, previous=None):
        """a NoDataHierarchy of tag_boxes ({level: boxes}, level 0 the domain if missing);
        work: optional per level 0 cell weights (level 0's shape); previous: the hierarchy
        being replaced (partition "keep": its owners kept where its data is)"""
        if work is not None:
            work = np.asarray(work, dtype=float)
            if list(work.shape) != self.cells:
                raise ValueError(
                    f"Chopper: work map shape {work.shape}, level 0 {self.cells}"
                )
        domain = ([0] * self.ndim, [c - 1 for c in self.cells])
        tag_boxes = dict(tag_boxes)
        tag_boxes.setdefault(0, [domain])
        partition = getattr(plamn_etc, f"partition_{self.ndim}")
        if self.partition not in ("levels", "curve", "columns", "keep"):
            raise ValueError(
                f"Chopper: partition {self.partition}: levels, curve, columns or keep"
            )
        if self.partition == "keep" and previous is None:
            raise ValueError("Chopper: partition keep needs the previous hierarchy")
        columns = self._columns(tag_boxes) if self.partition == "columns" else None

        h = NoDataHierarchy(
            self.ndim,
            domain,
            self.periodic,
            self.ratio,
            self.dl,
            self.origin,
            self.sim.interp_order,
            self.node_of_rank,
        )
        for ln in sorted(tag_boxes):
            if ln != len(h.levels):
                raise ValueError(
                    f"Chopper: levels must be 0..n, got {sorted(tag_boxes)}"
                )
            if columns is not None:
                patches = columns[ln]
                weights = [self._weight(ln, lo, up, None) for lo, up, _ in patches]
                if self.spill is not None:
                    patches = self._spill(patches, weights)
                h.levels.append(patches)
                h.weights.append(weights)
                continue
            if self.partition == "keep":
                patches = self._keep(ln, tag_boxes[ln], previous)
                weights = [self._weight(ln, lo, up, None) for lo, up, _ in patches]
                if self.spill is not None:
                    patches = self._spill(patches, weights)
                h.levels.append(patches)
                h.weights.append(weights)
                continue
            boxes = self._patch_boxes(ln, tag_boxes[ln])
            weights = [self._weight(ln, lo, up, work) for lo, up in boxes]
            if self.partition == "curve":
                # every level on level 0's curve: boxes coarsened to level 0, and a weightless
                #  box over the domain so every level's curve has the domain's frame
                #  (damn::amr::curve_order fits the curve to the boxes given)
                f = self.ratio**ln
                ordered = [
                    ([x // f for x in lo], [x // f for x in up]) for lo, up in boxes
                ]
                ranks = partition(
                    [*ordered, domain], [*weights, 0.0], self.node_of_rank
                )[:-1]
            else:
                ranks = partition(boxes, weights, self.node_of_rank)
            h.levels.append([(lo, up, r) for (lo, up), r in zip(boxes, ranks)])
            h.weights.append(weights)
        return h
