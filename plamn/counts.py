"""Particles as one count per cell: cheap stand ins for a run's particles, to assess regrids and
partitions over many configurations (memory: a number per cell, not ~250 B per particle).

    c = Counts.from_density(cells, dl, density, ppc)   # pyphare's loading: ppc where the
                                                       #  density is at or above the cut-off
    c.advance(dt, vbulk, vth)                          # the particles move: counts follow
    c.box_sum(lower, upper)                            # the particles in a box (O(1))
    c.refined(boxes, ratio, refined_nbr)               # a refined level's counts
    tags = c.tags(threshold)                           # a density gradient tagging

advance: a cell's count dealt to the cells its particles can reach, the total exact; no
velocities (advance once per regrid interval), no fields: doc/plamn/counts.md

Nothing runs on import.
"""

from dataclasses import dataclass, field

import numpy as np


def _H(y, s):
    """the integral of Phi(y / s) over y: y Phi(y / s) + s phi(y / s); s = 0: max(y, 0)"""
    from scipy.special import ndtr

    y, s = np.broadcast_arrays(np.asarray(y, dtype=float), np.asarray(s, dtype=float))
    shape = y.shape
    y, s = np.atleast_1d(y).ravel(), np.atleast_1d(s).ravel()
    out = np.maximum(y, 0.0)
    warm = s > 0
    z = y[warm] / s[warm]
    out[warm] = y[warm] * ndtr(z) + s[warm] * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    return out.reshape(shape)


def landing(shift, sigma, offsets):
    """per offset k (cells), the probability that a particle uniform in its cell, displaced by
    shift plus a Gaussian of sigma (both in cells, per cell or scalars), lands k cells away:
    H(k + 1 - m) - 2 H(k - m) + H(k - 1 - m) (the cell's hat convolved with the Gaussian).
    Returns [len(offsets), *shape]"""
    shift = np.asarray(shift, dtype=float)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), shift.shape)
    out = np.empty((len(offsets), *shift.shape))
    for i, k in enumerate(offsets):
        y = k - shift
        out[i] = _H(y + 1, sigma) - 2 * _H(y, sigma) + _H(y - 1, sigma)
    return np.clip(out, 0.0, 1.0)


@dataclass
class Counts:
    counts: np.ndarray  # integer, one per cell of level 0 (cells, x first)
    dl: tuple
    periodic: tuple = None  # per direction, default all
    origin: tuple = None
    seed: int = 12345
    outflow: int = 0  # particles that left through a non periodic boundary
    rng: object = field(default=None, repr=False)

    def __post_init__(self):
        self.counts = np.asarray(self.counts, dtype=np.int64)
        nd = self.counts.ndim
        self.dl = tuple(float(d) for d in np.broadcast_to(self.dl, (nd,)))
        self.periodic = tuple(np.broadcast_to(True if self.periodic is None else self.periodic,
                                              (nd,)))
        self.origin = tuple(np.broadcast_to(0.0 if self.origin is None else self.origin, (nd,)))
        self.rng = self.rng or np.random.default_rng(self.seed)

    @property
    def ndim(self):
        return self.counts.ndim

    @property
    def total(self):
        return int(self.counts.sum())

    def centres(self):
        """the cell centres' coordinates, one array per direction (cells' shape)"""
        axes = [o + (np.arange(n) + 0.5) * d
                for o, n, d in zip(self.origin, self.counts.shape, self.dl)]
        return np.meshgrid(*axes, indexing="ij")

    @classmethod
    def from_density(cls, cells, dl, density, ppc, cutoff=1e-16, **kwargs):
        """pyphare's loading: ppc particles in every cell whose density (at its centre) is at or
        above the cut-off; density: one function, or a list (populations: summed)"""
        cells = tuple(int(c) for c in np.atleast_1d(cells))
        c = cls(np.zeros(cells, dtype=np.int64), dl, **kwargs)
        X = c.centres()
        for d, n in zip(density if isinstance(density, (list, tuple)) else [density],
                        ppc if isinstance(ppc, (list, tuple)) else [ppc]):
            c.counts += np.where(np.asarray(d(*X)) >= cutoff, int(n), 0)
        return c

    def _field(self, f, d):
        """a velocity component: a function of the centres, a number, or an array"""
        if callable(f):
            return np.broadcast_to(np.asarray(f(*self.centres()), dtype=float),
                                   self.counts.shape)
        f = np.asarray(f, dtype=float)
        if f.ndim == 1 and f.size == self.ndim:  # one value per direction
            return np.full(self.counts.shape, f[d])
        return np.broadcast_to(f, self.counts.shape)

    def advance(self, dt, vbulk=0.0, vth=0.0):
        """the particles move for dt: vbulk, vth per direction (a list of ndim functions of the
        cell centres / numbers / arrays) or one for every direction"""
        for d in range(self.ndim):
            vb = vbulk[d] if isinstance(vbulk, (list, tuple)) else vbulk
            vt = vth[d] if isinstance(vth, (list, tuple)) else vth
            shift = self._field(vb, d) * dt / self.dl[d]
            sigma = self._field(vt, d) * dt / self.dl[d]
            self._move(d, shift, sigma)
        return self

    def _move(self, d, shift, sigma):
        reach = 6 * float(np.max(sigma)) + 1  # the tails past 6 sigma (~1e-9): the last offset
        lo = int(np.floor(np.min(shift) - reach))
        hi = int(np.ceil(np.max(shift) + reach))
        offsets = list(range(lo, hi + 1))
        left = self.counts.copy()
        mass = np.ones(self.counts.shape)
        new = np.zeros_like(self.counts)
        n = self.counts.shape[d]
        for i, k in enumerate(offsets):  # one offset at a time: a few arrays of the cells' size
            p = landing(shift, sigma, [k])[0]
            q = np.clip(np.divide(p, mass, out=np.zeros_like(mass), where=mass > 1e-300),
                        0.0, 1.0)
            if i == len(offsets) - 1:
                q = np.ones_like(q)  # the rest: the total exact
            moved = self.rng.binomial(left, q)
            left -= moved
            mass -= p
            if self.periodic[d]:
                new += np.roll(moved, k, axis=d)
            else:
                shifted = np.zeros_like(moved)
                src = [slice(None)] * self.ndim
                dst = [slice(None)] * self.ndim
                if k >= 0:
                    src[d], dst[d] = slice(0, n - k), slice(k, n)
                else:
                    src[d], dst[d] = slice(-k, n), slice(0, n + k)
                if abs(k) < n:
                    shifted[tuple(dst)] = moved[tuple(src)]
                self.outflow += int(moved.sum() - shifted.sum())
                new += shifted
        self.counts = new

    def _prefix(self):
        P = self.counts
        for d in range(self.ndim):
            P = np.cumsum(P, axis=d)
        return np.pad(P, [(1, 0)] * self.ndim)

    def box_sum(self, lower, upper, prefix=None):
        """the particles in the cells lower..upper (inclusive, level 0's index space), from the
        prefix sums (pass prefix=c._prefix() for many boxes at the same time)"""
        P = self._prefix() if prefix is None else prefix
        lower, upper = np.atleast_1d(lower), np.atleast_1d(upper) + 1
        total = 0
        for corner in np.ndindex(*([2] * self.ndim)):
            idx = tuple(int(upper[d]) if c else int(lower[d]) for d, c in enumerate(corner))
            sign = (-1) ** (self.ndim - sum(corner))
            total += sign * int(P[idx])
        return total

    def box_sums(self, boxes):
        """box_sum of each (lower, upper)"""
        P = self._prefix()
        return [self.box_sum(lo, up, P) for lo, up in boxes]

    def refined(self, ln, ratio, refined_nbr):
        """level ln's particles per cell (floats) on the whole level ln grid: each level 0 cell's
        count split as plamn splits particles onto a refined level, refined_nbr per particle per
        refinement, spread evenly over the ratio^ndim finer cells under it"""
        c = self.counts.astype(float)
        for _ in range(ln):
            for d in range(self.ndim):
                c = np.repeat(c, ratio, axis=d)
            c *= refined_nbr / ratio**self.ndim
        return c

    def tags(self, threshold, counts=None):
        """cells to refine: the counts' relative gradient |c[i+1] - c[i-1]| / (c[i+1] + c[i-1])
        above threshold in any direction (as PHARE's default tagger on a quantity)"""
        c = (self.counts if counts is None else counts).astype(float)
        tagged = np.zeros(c.shape, dtype=bool)
        for d in range(c.ndim):
            up, down = np.roll(c, -1, axis=d), np.roll(c, 1, axis=d)
            den = up + down
            rel = np.divide(np.abs(up - down), den, out=np.zeros_like(den), where=den > 0)
            tagged |= rel > threshold
        return tagged
