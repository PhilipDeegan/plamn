"""MHD level 0, hybrid level 1 (#41, #67): a 2D harris (two periodic current sheets), the
hybrid level a fixed box around the lower sheet's X point. MHD: the harris density and B, the total pressure from the
balance p = 0.7 - B^2 / 2 (as the hybrid harris: n T_i = 0.7 - B^2 / 2 with Te 0); electrons
isothermal Te so that the coupling's Pi = max(p - Te n, 0) stays positive (Te 0.1: Pi >= 0.16).

    PYTHONPATH=.:$PHARE/pyphare python3 etc/experiments/coupling/harris2d.py [--steps N]
        [--hybrid-only]
needs the 2D coupled module (./mkn.py --coupled --dim 2 --layout AoSMapped --reconstruction Linear
--limiter VanLeer --riemann Rusanov --hall false), and the 2D hybrid one for --hybrid-only
--hybrid-only: the same harris, hybrid on both levels (the reference to compare level 1 with)
"""

import argparse
import sys
import time

import numpy as np

import plamn.backend.phare as ph


def config(a):
    L = 0.5
    cells = (a.nx, a.ny)
    Lx, Ly = a.nx * a.dl, a.ny * a.dl
    # level 1: the lower sheet's X point (Lx / 2, 0.3 Ly), +- 2 in x, +- 1 in y
    lo = (int((Lx / 2 - 2) / a.dl), int((0.3 * Ly - 1) / a.dl))
    hi = (int((Lx / 2 + 2) / a.dl) - 1, int((0.3 * Ly + 1) / a.dl) - 1)

    def density(x, y):
        return (
            0.4
            + 1.0 / np.cosh((y - Ly * 0.3) / L) ** 2
            + 1.0 / np.cosh((y - Ly * 0.7) / L) ** 2
        )

    def S(y, y0, l):
        return 0.5 * (1.0 + np.tanh((y - y0) / l))

    def by(x, y):
        x0, y1, y2 = x - 0.5 * Lx, y - 0.3 * Ly, y - 0.7 * Ly
        return 2 * 0.1 * x0 * (np.exp(-(x0**2 + y1**2)) - np.exp(-(x0**2 + y2**2)))

    def bx(x, y):
        x0, y1, y2 = x - 0.5 * Lx, y - 0.3 * Ly, y - 0.7 * Ly
        dbx = 2 * 0.1 * (-y1 * np.exp(-(x0**2 + y1**2)) + y2 * np.exp(-(x0**2 + y2**2)))
        return -1 + 2.0 * (S(y, Ly * 0.3, L) - S(y, Ly * 0.7, L)) + dbx

    def zero(x, y):
        return 0.0 * x

    def pressure(x, y):
        return 0.7 - (bx(x, y) ** 2 + by(x, y) ** 2) * 0.5

    common = {
        "time_step": a.dt,
        "time_step_nbr": a.steps,
        "cells": cells,
        "dl": (a.dl, a.dl),
        "refinement_boxes": {"L0": {"B0": [lo, hi]}},
        "largest_patch_size": a.patch,
        "resistivity": a.eta,
        "hyper_resistivity": a.nu,
        "hyper_mode": "constant",
        "diag_options": {
            "format": "pharevtkhdf",
            "options": {"dir": a.out, "mode": "overwrite"},
        },
    }
    species = {"charge": 1, "mass": 1, "nbr_part_per_cell": a.ppc, "init": {"seed": 12}}
    if a.hybrid_only:
        ph.Simulation(**common)

        def vth(x, y):
            T = (pressure(x, y) - a.Te * density(x, y)) / density(x, y)
            assert np.all(T > 0)
            return np.sqrt(T)

        ph.MaxwellianFluidModel(
            bx=bx,
            by=by,
            bz=zero,
            protons={
                **species,
                "density": density,
                **{f"vbulk{c}": zero for c in "xyz"},
                **{f"vth{c}": vth for c in "xyz"},
            },
        )
    else:
        ph.Simulation(
            **common,
            max_mhd_level=1,
            eta=a.eta,
            nu=0.0,  # MHD hyper resistivity needs the Hall term (ideal MHD here)
            gamma=5.0 / 3.0,
            hall=False,
            reconstruction="Linear",
            limiter="VanLeer",
            riemann="Rusanov",
            mhd_timestepper="TVDRK2",
            model_options=["MHDModel", "HybridModel"],
        )
        ph.MHDModel(
            density=density,
            vx=zero,
            vy=zero,
            vz=zero,
            bx=bx,
            by=by,
            bz=zero,
            p=pressure,
            protons=species,
        )
    ph.ElectronModel(closure="isothermal", Te=a.Te)
    times = a.dt * np.arange(0, a.steps + 1, a.diag_every)
    for q in ["E", "B"]:
        ph.ElectromagDiagnostics(quantity=q, write_timestamps=times)
    for q in ["mass_density", "bulkVelocity"]:
        ph.FluidDiagnostics(quantity=q, write_timestamps=times)
    return lo, hi


def main(argv):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=32)
    p.add_argument("--dl", type=float, default=0.2)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--diag-every", type=int, default=5)
    p.add_argument("--ppc", type=int, default=50)
    p.add_argument("--patch", type=int, default=32)
    p.add_argument("--Te", type=float, default=0.1)
    p.add_argument("--eta", type=float, default=1e-3)
    p.add_argument("--nu", type=float, default=1e-3)
    p.add_argument("--hybrid-only", action="store_true")
    p.add_argument("--out", default="phare_outputs/harris2d_coupled")
    a = p.parse_args(argv)
    lo, hi = config(a)
    sys.argv = [sys.argv[0]]
    print(
        f"harris 2D {a.nx}x{a.ny} dl {a.dl}, level 1 box {lo} {hi}, "
        f"{'hybrid only' if a.hybrid_only else 'MHD level 0, hybrid level 1'}"
    )
    t0 = time.time()
    s = ph.Simulator(ph.global_vars.sim, log_to_file=False).setup().initialize()
    print(f"initialized in {time.time() - t0:.1f} s")
    t0 = time.time()
    for step in range(1, a.steps + 1):
        s.advance()
        if step % 5 == 0:
            print(
                f"step {step} t {s.currentTime():.4f} {(time.time() - t0) / step:.2f} s/step"
            )
            sys.stdout.flush()
    print(f"done: {a.steps} steps, {time.time() - t0:.1f} s")
    s.reset()


if __name__ == "__main__":
    main(sys.argv[1:])
