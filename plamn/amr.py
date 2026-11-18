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


@dataclass
class Chopper:
    sim: object
    ranks: int = 1
    ranks_per_node: int = None  # default: all on one node
    mode: str = "even"

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

    def _weight(self, ln, lower, upper, work):
        if ln == 0 and work is not None:
            sl = tuple(slice(lo, up + 1) for lo, up in zip(lower, upper))
            return float(np.sum(work[sl]))
        return float(np.prod(np.asarray(upper) - np.asarray(lower) + 1) * self.ppc)

    def chop(self, tag_boxes, work=None):
        """a NoDataHierarchy of tag_boxes ({level: boxes}, level 0 the domain if missing);
        work: optional per level 0 cell weights (level 0's shape)"""
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
            boxes = self._patch_boxes(ln, tag_boxes[ln])
            weights = [self._weight(ln, lo, up, work) for lo, up in boxes]
            ranks = partition(boxes, weights, self.node_of_rank)
            h.levels.append([(lo, up, r) for (lo, up), r in zip(boxes, ranks)])
            h.weights.append(weights)
        return h
