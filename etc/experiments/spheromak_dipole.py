"""A Taylor spheromak (radius R, force free inside, no field outside) at the centre of a cube,
relaxing into a dipole (plamn #68). Pure hybrid: the sphere is at ion scale (R ~ d_i), below what
MHD models. Two levels by default: level 0 over the whole domain, a fixed level 1 box around the
sphere. The plasma: uniform density, cold ions (vth 1e-5), Te 0.

    PYTHONPATH=.:$PHARE/pyphare python3 etc/experiments/spheromak_dipole.py
    ... spheromak_dipole.py --cells 60 --dl 0.1 --margin -1       # one level, 60^3 at dl 0.1
    ... spheromak_dipole.py --final-time 0.5 --diag-every 0.05 --out phare_outputs/sph

Options: --cells (level 0, per side), --dl (level 0), --margin (level 1: the box around the sphere,
this far beyond R, in units of length; < 0: no level 1), --ppc, --final-time, --diag-every,
--layout, --allocator, --cfl-wave, --cfl-particles, --time-step (fixed, else adaptive), --out, --R, --B0, --density, --vth,
--resistivity, --hyper-resistivity.
Diagnostics (pharevtkhdf, --out): B, E, the ions' mass density and bulk velocity, at 0 and every
--diag-every; compare runs with spheromak_compare.py.
"""

import argparse
import sys
import time

import numpy as np
from scipy.special import spherical_jn


def taylor_spheromak(x, y, z, centre, R, B0):
    """B of a Taylor spheromak of radius R centred at centre: k R = 4.493 (j1's first zero over
    R: B_r = 0 on the sphere), 0 outside"""
    dx, dy, dz = x - centre[0], y - centre[1], z - centre[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    inside = (r <= R) & (r > 0)
    bx, by, bz = np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)
    k = 4.493 / R
    rr = r[inside]
    f = B0 * spherical_jn(1, k * rr) / rr
    bx[inside] = f * (dz[inside] * dx[inside] / rr - dy[inside])
    by[inside] = f * (dz[inside] * dy[inside] / rr + dx[inside])
    bz[inside] = f * (-(dx[inside] ** 2 + dy[inside] ** 2) / rr)
    return bx, by, bz


def refinement_boxes(cells, dl, R, margin):
    """level 1: the level 0 cells covering the sphere and margin beyond it (inclusive bounds)"""
    if margin < 0:
        return {}
    mid = cells * dl / 2
    lo = int(np.floor((mid - R - margin) / dl))
    hi = int(np.ceil((mid + R + margin) / dl)) - 1
    lo, hi = max(lo, 0), min(hi, cells - 1)
    return {"L0": {"B0": [(lo,) * 3, (hi,) * 3]}}


def config(a):
    import plamn.backend.phare as ph

    cells, dl = a.cells, a.dl
    centre = (cells * dl / 2,) * 3
    boxes = refinement_boxes(cells, dl, a.R, a.margin)
    times = np.arange(0, a.final_time + 0.5 * a.diag_every, a.diag_every)
    sim = ph.Simulation(
        time_step=(
            a.time_step
            if a.time_step
            else dict(
                mode="adaptive", cfl_wave=a.cfl_wave, cfl_particles=a.cfl_particles
            )
        ),
        final_time=a.final_time,
        cells=(cells,) * 3,
        dl=(dl,) * 3,
        interp_order=1,
        refinement_boxes=boxes,
        resistivity=a.resistivity,
        hyper_resistivity=a.hyper_resistivity,
        model_layout=a.layout,
        allocator=a.allocator,
        diag_options={
            "format": "pharevtkhdf",
            "options": {"dir": a.out, "mode": "overwrite"},
        },
        strict=False,
    )

    def b(i):
        return lambda x, y, z: taylor_spheromak(x, y, z, centre, a.R, a.B0)[i]

    def const(v):
        return lambda x, y, z: v + 0.0 * x

    ph.MaxwellianFluidModel(
        bx=b(0),
        by=b(1),
        bz=b(2),
        protons={
            "charge": 1,
            "mass": 1,
            "density": const(a.density),
            "nbr_part_per_cell": a.ppc,
            "init": {"seed": 12334},
            **{f"vbulk{c}": const(0.0) for c in "xyz"},
            **{f"vth{c}": const(a.vth) for c in "xyz"},
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
    p.add_argument("--cells", type=int, default=30)
    p.add_argument("--dl", type=float, default=0.2)
    p.add_argument("--margin", type=float, default=0.4)
    p.add_argument("--ppc", type=int, default=100)
    p.add_argument("--final-time", type=float, default=1.0)
    p.add_argument("--diag-every", type=float, default=0.1)
    p.add_argument("--layout", default="SoAVXChunked")
    p.add_argument("--allocator", default="CPU")
    p.add_argument("--cfl-wave", type=float, default=0.5)
    p.add_argument("--cfl-particles", type=float, default=0.5)
    p.add_argument("--time-step", type=float, help="a fixed dt (default: adaptive)")
    p.add_argument("--R", type=float, default=1.0)
    p.add_argument("--B0", type=float, default=1.0)
    p.add_argument("--density", type=float, default=0.5)
    p.add_argument("--vth", type=float, default=1e-5)
    p.add_argument("--resistivity", type=float, default=1e-3)
    p.add_argument("--hyper-resistivity", type=float, default=1e-3)
    p.add_argument("--out", default="phare_outputs/spheromak_dipole")
    return p.parse_args(argv)


def main(argv):
    a = parse(argv)
    ph, sim, boxes = config(a)
    sys.argv = [sys.argv[0]]
    print(
        f"spheromak R {a.R} B0 {a.B0}: level 0 {a.cells}^3 dl {a.dl}, level 1 {boxes or 'none'}"
    )
    t0 = time.time()
    s = ph.Simulator(sim, log_to_file=False).setup().initialize()
    print(f"initialized in {time.time() - t0:.1f} s")
    t0, steps = time.time(), 0
    while s.currentTime() < a.final_time - 1e-12:
        s.advance()
        steps += 1
        if steps % 50 == 0:
            el = time.time() - t0
            print(
                f"t {s.currentTime():.4f} step {steps} dt {s.timeStep():.3e} {el / steps:.2f} s/step"
            )
            sys.stdout.flush()
    print(f"t {s.currentTime():.4f} after {steps} steps, {time.time() - t0:.1f} s")
    s.reset()


if __name__ == "__main__":
    main(sys.argv[1:])
