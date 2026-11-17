"""The PHARE / plamn comparison ladder: one input per case, each case adding one mechanism to the
previous one, so the first case that differs says where (as tiling_gpu's tests/simulator/layouts).

  init_L0      particle loading, moments, initial fields              no refinement, no step
  init_L1      + a refined level made from level 0                    one box
  advance_L0   + push, deposit, field solve                           1 step
  advance_L1   + level ghosts, substeps, sync (coarsening), reflux    one box, 1 step
  init_L2      + a level 2 whose ghosts need level 1's ghosts         two boxes
               (2 level 1 cells inside level 1: less than the 3 its ghosts reach, 2 + stencil)
  init_L2_nested   the same, 4 cells inside: level 1's domains only
  advance_L2   + a third level, several syncs                         two boxes, 4 steps
  advance_L2_nested  the same, 4 cells inside
  harris_2d    + tagging, regrids, load balancing                     3 levels, 2 steps

The model is PHARE's tests/simulator/advance/test_advance_hybrid.py (getHierarchy), the sizes,
seeds and boxes those of tiling_gpu's tests/simulator/layouts (10 particles per cell); harris_2d is
tests/simulator/layouts/test_harris_2d_layouts.py. Diagnostics every step (fields, particles in
phareh5), and every fine level substep with fine_dump (_diag_options).
Nothing runs on import: ``simulation(case, out, ...)`` declares the input (pyphare.pharein) of
whichever code imported pyphare.
"""

import numpy as np

NDIM = 2
INTERP = 1
CELLS = 14
DL = 0.2
TIME_STEP = 0.001
import os

PPC = int(os.environ.get("LADDER_PPC", "10"))  # per cell: few, so particles can be compared too
SEED = 1337
POPULATIONS = ["protons"]


def _boxes(ndim, *levels):
    """refinement_boxes: one box per level, (lo, hi) as cell indices in that level's space."""
    from pyphare.core.box import nDBox

    return {f"L{i}": {"B0": nDBox(ndim, lo, hi)} for i, (lo, hi) in enumerate(levels)}


CASES = {
    "init_L0": dict(boxes=(), steps=0),
    "init_L1": dict(boxes=((1, 12),), steps=0),
    "advance_L0": dict(boxes=(), steps=1),
    "advance_L1": dict(boxes=((1, 12),), steps=1),
    "init_L2": dict(boxes=((1, 12), (4, 21)), steps=0),
    "init_L2_nested": dict(boxes=((1, 12), (6, 19)), steps=0),
    "advance_L2": dict(boxes=((1, 12), (4, 21)), steps=4),
    "advance_L2_nested": dict(boxes=((1, 12), (6, 19)), steps=4),
    "harris_2d": dict(harris=True, steps=2),
    # compare_physics.py's (not a ladder step): 3 levels, tagging, final B composites
    "harris_3d": dict(harris3d=True, steps=2),
}


def _diag_options(out, fmt, fine_dump):
    """the regular dumps; with fine_dump, also each fine level at the end of its substeps
    (fine_dump_lvl_max: phareh5, PHARE master's pharevtkhdf writes them with level tables
    shorter than the steps, fixed in a later PHARE)."""
    options = {"dir": out, "mode": "overwrite"}
    if fine_dump:
        options["fine_dump_lvl_max"] = 10
    return {"format": fmt, "options": options}


def _diagnostics(ph, timestamps, pops, fmt="phareh5"):
    particles = os.environ.get("LADDER_PARTICLES", "1") == "1"  # 0: fields only
    if fmt == "phareh5" and particles:  # pharevtkhdf has no particles
        for pop in pops:
            for q in ["domain", "levelGhost"]:
                ph.ParticleDiagnostics(quantity=q, write_timestamps=timestamps, population_name=pop)
    for q in ["E", "B"]:
        ph.ElectromagDiagnostics(quantity=q, write_timestamps=timestamps)
    for q in ["charge_density", "mass_density", "bulkVelocity", "charge_flux"]:
        ph.FluidDiagnostics(quantity=q, write_timestamps=timestamps)
    for pop in pops:
        for q in ["density", "flux"]:
            ph.FluidDiagnostics(quantity=q, write_timestamps=timestamps, population_name=pop)


def _advance_hybrid(ph, out, case, ndim, largest_patch_size, fmt, fine_dump):
    """PHARE's test_advance_hybrid model (bump density, cosine B and V, cold vth)."""
    sim = ph.Simulation(
        time_step_nbr=max(case["steps"], 1),
        time_step=TIME_STEP,
        cells=[CELLS] * ndim,
        dl=[DL] * ndim,
        interp_order=INTERP,
        refinement_boxes=_boxes(ndim, *case["boxes"]) if case["boxes"] else {},
        largest_patch_size=largest_patch_size,
        diag_options=_diag_options(out, fmt, fine_dump),
        strict=True,
    )
    L = np.array(sim.simulation_domain())

    def density(*xyz):  # a gaussian bump made periodic
        hL = L / 2
        return 0.3 + np.exp(
            sum(-((L[i] / np.pi * np.sin(np.pi * (xyz[i] - hL[i]) / L[i])) ** 2)
                for i in range(len(xyz)))
        )

    def cosines(*xyz):
        return np.asarray(
            [0.1 * np.cos(2 * np.pi * xyz[i] / L[i]) for i in range(len(xyz))]
        ).prod(axis=0)

    def one(*xyz):
        return 1.0

    def vth(*xyz):
        return 0.01 + np.zeros_like(xyz[0])

    ph.MaxwellianFluidModel(
        bx=one,
        by=cosines,
        bz=cosines,
        protons={
            "charge": 1,
            "density": density,
            "vbulkx": cosines,
            "vbulky": cosines,
            "vbulkz": cosines,
            "vthx": vth,
            "vthy": vth,
            "vthz": vth,
            "nbr_part_per_cell": PPC,
            "init": {"seed": SEED},
        },
    )
    ph.ElectronModel(closure="isothermal", Te=0.12)
    _diagnostics(ph, TIME_STEP * np.arange(case["steps"] + 1), POPULATIONS, fmt)
    return sim


def _harris_2d(ph, out, case, largest_patch_size, fmt, fine_dump):
    """tiling_gpu's tests/simulator/layouts/test_harris_2d_layouts.py, vtkhdf diagnostics."""
    L, time_step = 0.5, 0.005
    sim = ph.Simulation(
        time_step=time_step,
        final_time=time_step * case["steps"],
        cells=(100, 50),
        dl=(0.40, 0.40),
        refinement="tagging",
        max_nbr_levels=3,
        hyper_resistivity=0.002,
        resistivity=0.001,
        largest_patch_size=largest_patch_size,
        diag_options=_diag_options(out, fmt, fine_dump),
        nesting_buffer=1,
    )
    Lx, Ly = sim.simulation_domain()

    def density(x, y):
        return 0.4 + 1.0 / np.cosh((y - Ly * 0.3) / L) ** 2 + 1.0 / np.cosh((y - Ly * 0.7) / L) ** 2

    def S(y, y0, l):
        return 0.5 * (1.0 + np.tanh((y - y0) / l))

    def by(x, y):
        x0, y1, y2 = x - 0.5 * Lx, y - 0.3 * Ly, y - 0.7 * Ly
        return 2 * 0.1 * x0 * (np.exp(-(x0**2 + y1**2)) - np.exp(-(x0**2 + y2**2)))

    def bx(x, y):
        x0, y1, y2 = x - 0.5 * Lx, y - 0.3 * Ly, y - 0.7 * Ly
        dbx = 2 * 0.1 * (-y1 * np.exp(-(x0**2 + y1**2)) + y2 * np.exp(-(x0**2 + y2**2)))
        return -1 + 2.0 * (S(y, Ly * 0.3, L) - S(y, Ly * 0.7, L)) + dbx

    def bz(x, y):
        return 0.0

    def T(x, y):
        temp = 1.0 / density(x, y) * (0.7 - (bx(x, y) ** 2 + by(x, y) ** 2) * 0.5)
        assert np.all(temp > 0)
        return temp

    def zero(x, y):
        return 0.0

    def vth(x, y):
        return np.sqrt(T(x, y))

    ph.MaxwellianFluidModel(
        bx=bx,
        by=by,
        bz=bz,
        protons={
            "charge": 1,
            "density": density,
            "vbulkx": zero,
            "vbulky": zero,
            "vbulkz": zero,
            "vthx": vth,
            "vthy": vth,
            "vthz": vth,
            "nbr_part_per_cell": PPC,
            "init": {"seed": int(os.environ.get("LADDER_SEED", "12334"))},
        },
    )
    ph.ElectronModel(closure="isothermal", Te=0.0)
    ph.LoadBalancer(every=1, mode="nppc", tol=0.001)
    _diagnostics(ph, time_step * np.arange(case["steps"] + 1), POPULATIONS, fmt)
    return sim



def _harris_3d(ph, out, case, largest_patch_size, fmt, fine_dump):
    """tiling_gpu's tests/simulator/layouts/test_harris_3d_layouts.py, 3 levels (2 there)."""
    L, time_step = 0.5, 0.001
    sim = ph.Simulation(
        time_step=time_step,
        final_time=time_step * case["steps"],
        cells=(20, 20, 20),
        dl=(0.4, 0.4, 0.4),
        refinement="tagging",
        max_nbr_levels=3,
        nesting_buffer=1,
        tagging_threshold=0.5,
        hyper_resistivity=0.008,
        hyper_mode="spatial",
        resistivity=0.001,
        tag_buffer=3,
        largest_patch_size=largest_patch_size,
        diag_options=_diag_options(out, fmt, fine_dump),
    )
    Lx, Ly, _ = sim.simulation_domain()

    def density(x, y, z):
        return 0.4 + 1.0 / np.cosh((y - Ly * 0.3) / L) ** 2 + 1.0 / np.cosh((y - Ly * 0.7) / L) ** 2

    def S(y, y0, l):
        return 0.5 * (1.0 + np.tanh((y - y0) / l))

    def by(x, y, z):
        x0, y1, y2 = x - 0.5 * Lx, y - 0.3 * Ly, y - 0.7 * Ly
        return 2 * 0.1 * x0 * (np.exp(-(x0**2 + y1**2)) - np.exp(-(x0**2 + y2**2)))

    def bx(x, y, z):
        x0, y1, y2 = x - 0.5 * Lx, y - 0.3 * Ly, y - 0.7 * Ly
        dbx = 2 * 0.1 * (-y1 * np.exp(-(x0**2 + y1**2)) + y2 * np.exp(-(x0**2 + y2**2)))
        return -1 + 2.0 * (S(y, Ly * 0.3, L) - S(y, Ly * 0.7, L)) + dbx

    def bz(x, y, z):
        return 0.0 * x

    def T(x, y, z):
        temp = 1.0 / density(x, y, z) * (0.7 - (bx(x, y, z) ** 2 + by(x, y, z) ** 2) * 0.5)
        assert np.all(temp > 0)
        return temp

    def zero(x, y, z):
        return 0.0 * x

    def vth(x, y, z):
        return np.sqrt(T(x, y, z))

    ph.MaxwellianFluidModel(
        bx=bx,
        by=by,
        bz=bz,
        protons={
            "charge": 1,
            "density": density,
            "vbulkx": zero,
            "vbulky": zero,
            "vbulkz": zero,
            "vthx": vth,
            "vthy": vth,
            "vthz": vth,
            "nbr_part_per_cell": PPC,
            "init": {"seed": int(os.environ.get("LADDER_SEED", "12"))},
        },
    )
    ph.ElectronModel(closure="isothermal", Te=0.0)
    ph.LoadBalancer(every=1, mode="nppc", tol=0.001)
    _diagnostics(ph, time_step * np.arange(case["steps"] + 1), POPULATIONS, fmt)
    return sim


def simulation(name, out, ndim=NDIM, largest_patch_size=None, fmt="pharevtkhdf", fine_dump=False):
    """Declares case ``name``'s input (pyphare.pharein) with diagnostics in ``out``, in
    format ``fmt``."""
    import pyphare.pharein as ph

    ph.global_vars.sim = None
    ph.clearDict()
    case = CASES[name]
    if case.get("harris"):
        return _harris_2d(ph, out, case, largest_patch_size, fmt, fine_dump)
    if case.get("harris3d"):
        return _harris_3d(ph, out, case, largest_patch_size, fmt, fine_dump)
    return _advance_hybrid(ph, out, case, ndim, largest_patch_size, fmt, fine_dump)
