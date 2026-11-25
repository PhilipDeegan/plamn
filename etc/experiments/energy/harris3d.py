"""A small 3D 2 level harris for energy budgets (#81): the harris of etc/compare/phare/cases.py's
_harris_3d (two periodic current sheets, warm ions in pressure balance, Te 0), with one fixed
level 1 box around the lower sheet's X point (no tagging), constant hyper resistivity (the
spatial mode's nu varies in space: not a single eta_h for the budget) and a fixed dt. The
budget: etc/experiments/energy/yee_budget.py run --case harris3d ...

    PYTHONPATH=.:$PHARE/pyphare python3 etc/experiments/energy/harris3d.py --final-time 0.2
"""

import argparse
import sys
import time

import numpy as np


def config(a):
    import plamn.backend.phare as ph

    L = 0.5
    cells = (a.nx, a.ny, a.nz)
    Lx, Ly = a.nx * a.dl, a.ny * a.dl
    # level 1: the lower sheet's X point (Lx / 2, 0.3 Ly), +- 1 in x, +- 0.8 in y, z but the end cells (nesting buffer)
    lo = (int((Lx / 2 - 1) / a.dl), int((0.3 * Ly - 0.8) / a.dl), 1)
    hi = (int((Lx / 2 + 1) / a.dl) - 1, int((0.3 * Ly + 0.8) / a.dl) - 1, a.nz - 2)
    boxes = {"L0": {"B0": [lo, hi]}} if a.levels > 1 else {}
    times = np.arange(0, a.final_time + 0.5 * a.diag_every, a.diag_every)
    sim = ph.Simulation(
        time_step=a.time_step,
        final_time=a.final_time,
        cells=cells,
        dl=(a.dl,) * 3,
        interp_order=1,
        refinement_boxes=boxes,
        resistivity=a.resistivity,
        hyper_resistivity=a.hyper_resistivity,
        hyper_mode="constant",
        model_layout=a.layout,
        diag_options={
            "format": "pharevtkhdf",
            "options": {"dir": a.out, "mode": "overwrite"},
        },
        strict=False,
    )

    def density(x, y, z):
        return (
            0.4
            + 1.0 / np.cosh((y - Ly * 0.3) / L) ** 2
            + 1.0 / np.cosh((y - Ly * 0.7) / L) ** 2
        )

    def S(y, y0, l):
        return 0.5 * (1.0 + np.tanh((y - y0) / l))

    def by(x, y, z):
        x0, y1, y2 = x - 0.5 * Lx, y - 0.3 * Ly, y - 0.7 * Ly
        return 2 * 0.1 * x0 * (np.exp(-(x0**2 + y1**2)) - np.exp(-(x0**2 + y2**2)))

    def bx(x, y, z):
        x0, y1, y2 = x - 0.5 * Lx, y - 0.3 * Ly, y - 0.7 * Ly
        dbx = 2 * 0.1 * (-y1 * np.exp(-(x0**2 + y1**2)) + y2 * np.exp(-(x0**2 + y2**2)))
        return -1 + 2.0 * (S(y, Ly * 0.3, L) - S(y, Ly * 0.7, L)) + dbx

    def zero(x, y, z):
        return 0.0 * x

    def vth(x, y, z):
        T = 1.0 / density(x, y, z) * (0.7 - (bx(x, y, z) ** 2 + by(x, y, z) ** 2) * 0.5)
        assert np.all(T > 0)
        return np.sqrt(T)

    ph.MaxwellianFluidModel(
        bx=bx,
        by=by,
        bz=zero,
        protons={
            "charge": 1,
            "mass": 1,
            "density": density,
            "nbr_part_per_cell": a.ppc,
            "init": {"seed": 12},
            **{f"vbulk{c}": zero for c in "xyz"},
            **{f"vth{c}": vth for c in "xyz"},
        },
    )
    ph.ElectronModel(closure="isothermal", Te=0.0)
    for q in ["E", "B"]:
        ph.ElectromagDiagnostics(quantity=q, write_timestamps=times)
    for q in ["mass_density", "bulkVelocity"]:
        ph.FluidDiagnostics(quantity=q, write_timestamps=times)
    return ph, sim, boxes


def parse(argv):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--nx", type=int, default=40)
    p.add_argument("--ny", type=int, default=40)
    p.add_argument("--nz", type=int, default=12)
    p.add_argument("--dl", type=float, default=0.2)
    p.add_argument("--levels", type=int, default=2)
    p.add_argument("--ppc", type=int, default=100)
    p.add_argument("--final-time", type=float, default=0.2)
    p.add_argument("--diag-every", type=float, default=0.025)
    p.add_argument("--time-step", type=float, default=1e-3)
    p.add_argument("--layout", default="SoAVXChunked")
    p.add_argument("--resistivity", type=float, default=1e-3)
    p.add_argument("--hyper-resistivity", type=float, default=1e-3)
    p.add_argument("--out", default="phare_outputs/harris3d_energy")
    return p.parse_args(argv)


def main(argv):
    a = parse(argv)
    ph, sim, boxes = config(a)
    sys.argv = [sys.argv[0]]
    print(f"harris 3D {a.nx}x{a.ny}x{a.nz} dl {a.dl}, level 1 {boxes or 'none'}")
    t0 = time.time()
    s = ph.Simulator(sim, log_to_file=False).setup().initialize()
    print(f"initialized in {time.time() - t0:.1f} s")
    t0, steps = time.time(), 0
    while s.currentTime() < a.final_time - 1e-12:
        s.advance()
        steps += 1
        if steps % 25 == 0:
            print(
                f"t {s.currentTime():.4f} step {steps} {(time.time() - t0) / steps:.2f} s/step"
            )
            sys.stdout.flush()
    print(f"t {s.currentTime():.4f} after {steps} steps, {time.time() - t0:.1f} s")
    s.reset()


if __name__ == "__main__":
    main(sys.argv[1:])
