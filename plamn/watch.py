"""plots of a simulation while it runs (as PHARE's dw branch's pyphare/__main__.py --watch)

    python3 -m plamn [--watch N] [--save DIR] [--quantities B_y rho ...] [--config config]
                     [--input overrides.yaml] <module or path.py>

The module's config() (--config: another function) declares the simulation (pyphare's input,
through plamn); --input's YAML overrides its attributes. It runs to its final time and every N
level 0 steps (--watch, 0: never; the initial state too) each quantity is drawn, a subplot per
quantity, on the finest level's mesh (--interp nearest, the default, or linear): each level's
points where no finer level has any, interpolated onto the finest cell centres over the whole
domain (as pyphare's merged fields: flat_finest_field, then make_interpolator). --interp none:
each level as it is: 1D, a line per level (finer levels marked); 2D, an image per level, finer
levels over coarser ones. 2D colour map: --cmap (Spectral_r, as pyphare's hierarchy plots).
--save DIR: a PNG per drawing (DIR/<step>.png), no window. Quantities are resource
names (Simulator.level_field): by default the hybrid model's B and E (EM_B_x, ...), the MHD
model's rho, P, rhoV_x, B_y, B_z. Under MPI every rank gathers, rank 0 draws.
"""

import argparse
import importlib
import math
import os
import sys
from pathlib import Path

HYBRID_QUANTITIES = ["EM_B_x", "EM_B_y", "EM_B_z", "EM_E_x", "EM_E_y", "EM_E_z"]
MHD_QUANTITIES = ["rho", "P", "rhoV_x", "B_y", "B_z"]


def cli_args_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("module", help="the module (dotted, or a path.py) declaring the input")
    parser.add_argument("--config", default="config", help="the function declaring it")
    parser.add_argument("--input", default="", help="YAML overrides of the simulation's attributes")
    parser.add_argument("--watch", type=int, default=1, help="draw every N level 0 steps (0: never)")
    parser.add_argument("--save", default="", help="a PNG per drawing in this directory, no window")
    parser.add_argument("--quantities", nargs="+", default=None, help="resource names to draw")
    parser.add_argument("--interp", default="nearest", choices=["nearest", "linear", "none"],
                        help="onto the finest mesh (none: each level as it is)")
    parser.add_argument("--cmap", default="Spectral_r",
                        help="2D colour map (pyphare's hierarchy plots: Spectral_r)")
    return parser


def config_function(module, name):
    if module.endswith(".py"):
        module = module[:-3].replace(os.path.sep, ".")
    sys.path.insert(0, os.getcwd())
    function = getattr(importlib.import_module(module), name, None)
    if function is None:
        raise ValueError(f"{module} has no {name}()")
    return function


def is_mhd(sim):
    options = getattr(sim, "model_options", None) or ["HybridModel"]
    return "MHDModel" in options and not getattr(sim, "interp_order", None)


class Plotter:
    """draws the quantities of a running simulation (rank 0)"""

    def __init__(self, cpp, quantities, save, interp="nearest", cmap="Spectral_r"):
        import matplotlib
        import matplotlib.pyplot as plt

        if save or not os.environ.get("DISPLAY"):
            plt.switch_backend("Agg")
        elif matplotlib.get_backend().lower() == "agg":
            # an input may have chosen Agg (pyphare's NO_GUI(), e.g. PHARE's harris_2d.py)
            plt.switch_backend("TkAgg")

        self.plt, self.cpp, self.quantities, self.save = plt, cpp, quantities, save
        self.interp, self.cmap = interp, cmap
        self.draw = cpp.rank() == 0
        self.dx = cpp.cell_width()
        self.ndim = len(self.dx)
        self.colorbars = {}  # per axes, removed at each redraw (else they stack up)
        if self.draw:
            cols = min(3, len(quantities))
            rows = math.ceil(len(quantities) / cols)
            self.fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
            self.axes = axes.ravel()
            for ax in self.axes[len(quantities) :]:
                ax.set_visible(False)
            if save:
                Path(save).mkdir(parents=True, exist_ok=True)
            else:
                plt.ion()
                plt.show()

    def fields(self, name):
        """per level: (coordinates (n, ndim), values), every rank (collective)"""
        import numpy as np

        out = []
        for ln in range(self.cpp.n_levels()):
            points, values, dual = self.cpp.level_field(name, ln)
            points = np.asarray(points, dtype=float).reshape(-1, self.ndim)
            dx = np.asarray(self.dx) / 2**ln  # refinement ratio 2
            out.append(((points + 0.5 * np.asarray(dual)) * dx, np.asarray(values)))
        return out

    def __call__(self, step):
        """every rank: the levels' data gathered, rank 0 draws"""
        data = [self.fields(q) for q in self.quantities]
        if not self.draw:
            return
        for ax, name, levels in zip(self.axes, self.quantities, data):
            if ax in self.colorbars:
                self.colorbars.pop(ax).remove()
            ax.clear()
            if self.interp != "none":
                x, v = finest(levels, self.dx, self.cpp.domain_box(), self.interp)
                mappable = self.draw_finest(ax, x, v, len(levels), self.cmap)
                if mappable is not None:
                    self.colorbars[ax] = ax.figure.colorbar(mappable, ax=ax)
            elif self.ndim == 1:
                self.draw_1d(ax, levels)
            else:
                mappable = self.draw_2d(ax, levels, self.cmap)
                if mappable is not None:
                    self.colorbars[ax] = ax.figure.colorbar(mappable, ax=ax)
            ax.set_title(name)
        self.fig.suptitle(f"t = {self.cpp.currentTime():.6g}, step {step}")
        self.fig.tight_layout()
        if self.save:
            self.fig.savefig(Path(self.save) / f"{step:06d}.png")
        else:
            self.fig.canvas.draw_idle()
            self.plt.pause(0.05)

    @staticmethod
    def draw_finest(ax, x, v, n_levels, cmap=None):
        """the finest mesh's values: 1D a line, 2D an image; the image, for a colourbar"""
        label = f"{n_levels} level{'s' * (n_levels > 1)} on the finest mesh"
        if len(x) == 1:
            ax.plot(x[0], v, "-", lw=1)
            ax.set_xlabel(label, fontsize="small")
            return None
        dx, dy = (c[1] - c[0] if len(c) > 1 else 1.0 for c in x)
        extent = [x[0][0] - dx / 2, x[0][-1] + dx / 2, x[1][0] - dy / 2, x[1][-1] + dy / 2]
        mappable = ax.imshow(v.T, origin="lower", extent=extent, aspect="auto",
                             interpolation="nearest", cmap=cmap)
        ax.set_xlabel(label, fontsize="small")
        return mappable

    @staticmethod
    def draw_1d(ax, levels):
        for ln, (x, v) in enumerate(levels):
            ax.plot(x[:, 0], v, "-" if ln == 0 else ".", ms=3, lw=1, label=f"L{ln}")
        if len(levels) > 1:
            ax.legend(fontsize="small")

    @staticmethod
    def draw_2d(ax, levels, cmap=None):
        """an image per level over its points' bounding box (NaN where it has none), the axes
        over all of them (each image would set them to its own); the last image, for a
        colourbar"""
        import numpy as np

        values = np.concatenate([v for _, v in levels])
        vmin, vmax = (values.min(), values.max()) if values.size else (0, 1)
        mappable, extents = None, []
        for x, v in levels:
            if not v.size:
                continue
            spacing = [np.diff(np.unique(x[:, d])).min() if len(np.unique(x[:, d])) > 1 else 1.0
                       for d in range(2)]
            lo = x[:, :2].min(axis=0)
            index = np.rint((x[:, :2] - lo) / spacing).astype(int)
            image = np.full(index.max(axis=0) + 1, np.nan)
            image[index[:, 0], index[:, 1]] = v
            hi = lo + (np.asarray(image.shape) - 1) * spacing
            extent = [lo[0] - spacing[0] / 2, hi[0] + spacing[0] / 2,
                      lo[1] - spacing[1] / 2, hi[1] + spacing[1] / 2]
            mappable = ax.imshow(image.T, origin="lower", extent=extent, vmin=vmin, vmax=vmax,
                                 aspect="auto", interpolation="nearest", cmap=cmap)
            extents.append(extent)
        if extents:
            e = np.asarray(extents)
            ax.set_xlim(e[:, 0].min(), e[:, 1].max())
            ax.set_ylim(e[:, 2].min(), e[:, 3].max())
        return mappable


def flat_finest(levels, dx):
    """(points (n, ndim), values): each level's points where the next finer level has none
    within half its cell (in every direction), the finest level's all (pyphare's
    flat_finest_field, from the points: no boxes needed)"""
    import numpy as np
    from scipy.spatial import cKDTree

    points, values = [levels[-1][0]], [levels[-1][1]]
    for ln in range(len(levels) - 2, -1, -1):
        x, v = levels[ln]
        finer = levels[ln + 1][0]
        if len(x) and len(finer):
            half = 0.5 * np.asarray(dx, dtype=float) / 2**ln
            # in units of half a cell: covered when within it in every direction
            d, _ = cKDTree(finer / half).query(x / half, p=np.inf, distance_upper_bound=1.0)
            keep = ~np.isfinite(d) | (d >= 1.0 - 1e-9)
            x, v = x[keep], v[keep]
        points.append(x)
        values.append(v)
    return np.concatenate(points), np.concatenate(values)


def finest(levels, dx, domain_box, interp="nearest"):
    """(the finest mesh's cell centres per direction, values on them (shape: their lengths)):
    the flat finest points interpolated there (nearest, or linear: nearest outside their hull)"""
    import numpy as np

    points, values = flat_finest(levels, dx)
    ratio = 2 ** (len(levels) - 1)  # refinement ratio 2
    cells = np.ravel(np.asarray(domain_box))[-len(dx):] + 1  # the domain's upper cell + 1
    fdx = np.asarray(dx, dtype=float) / ratio
    x = [(np.arange(n * ratio) + 0.5) * h for n, h in zip(cells, fdx)]
    if len(dx) == 1:
        order = np.argsort(points[:, 0])
        px, pv = points[order, 0], values[order]
        if interp == "linear":
            return x, np.interp(x[0], px, pv)
        idx = np.clip(np.searchsorted(px, x[0]), 1, len(px) - 1)
        idx -= (x[0] - px[idx - 1]) < (px[idx] - x[0])
        return x, pv[idx]
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    grid = np.stack(np.meshgrid(*x, indexing="ij"), axis=-1)
    nearest = NearestNDInterpolator(points, values)(grid)
    if interp == "nearest":
        return x, nearest
    linear = LinearNDInterpolator(points, values)(grid)
    return x, np.where(np.isfinite(linear), linear, nearest)


def main(argv=None):
    args = cli_args_parser().parse_args(argv)
    import plamn.backend.phare as ph

    ph.global_vars.sim = None
    ph.clearDict()
    sim = config_function(args.module, args.config)() or ph.global_vars.sim
    if sim is None:
        raise ValueError(f"{args.module}.{args.config}() declared no simulation")
    if args.input:
        import yaml

        sim.__dict__.update(yaml.safe_load(Path(args.input).read_text()))

    simulator = ph.Simulator(sim, log_to_file=False).setup().initialize()
    cpp = simulator.cpp_sim
    quantities = args.quantities or (MHD_QUANTITIES if is_mhd(sim) else HYBRID_QUANTITIES)
    plot = Plotter(cpp, quantities, args.save, args.interp, args.cmap) if args.watch else None
    step = 0
    if plot:
        plot(step)
    end = cpp.endTime()
    while cpp.currentTime() < end - 1e-10 * max(cpp.timeStep(), 1e-300):
        simulator.advance()
        step += 1
        if plot and step % args.watch == 0:
            plot(step)
    simulator.reset()
    return 0


if __name__ == "__main__":
    sys.exit(main())
