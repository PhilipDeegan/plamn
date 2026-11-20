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
  ion_ion_beam_1d  PHARE's tests/functional/ionIonBeam/ion_ion_beam1d.py to t=0.001 (1 step):
               two populations, its refinement boxes (3 levels), hyper resistivity

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


# memory: each case's "mem" estimate (GB, one code's run, all its ranks), checked against the
#  measured peak after every run (run_case.py, test_ladder.memory): update it from the numbers.
#  Without one, an estimate from the level 0 particles. The tier says where a case runs:
#    gh     GitHub's runners (public repo: 4 cores, 16 GB, 14 GB disk), both codes one after
#           the other: slow.yml, on merges to master
#    local  this machine's (8 ranks or threads, the thermal governor)
#    big    by hand
GB = 1 << 30
TIERS = {"gh": 6 * GB, "local": 40 * GB}
BYTES_PER_PARTICLE = 250  # AoS ~80 B, times domain + ghosts + sort buffers + levels (rough)
BYTES_BASE = GB // 2  # python, SAMRAI, the modules


def ppc(name):
    return CASES[name].get("ppc", PPC)


def memory(name):
    """a case's estimated peak memory in bytes, one code's run"""
    case = CASES[name]
    if "mem" in case:
        return int(case["mem"] * GB)
    cells = np.prod(case.get("cells", [CELLS] * NDIM))
    return int(BYTES_BASE + cells * ppc(name) * BYTES_PER_PARTICLE)


def tier(name):
    m = memory(name)
    return next((t for t, budget in TIERS.items() if m <= budget), "big")


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
    "ion_ion_beam_1d": dict(ion_ion_beam=True, steps=1, pops=["main", "beam"]),
    # compare_physics.py --all-times (the slow CI's PHARE comparison): the beam on 2 levels, long
    "beam_1d_long": dict(
        ion_ion_beam=True,
        steps=2000,
        dump_every=100,
        pops=["main", "beam"],
        refinement_boxes={"L0": {"B0": [(50,), (110,)]}},
    ),
    # compare_physics.py's (not a ladder step): 3 levels, tagging, final B composites
    "harris_3d": dict(harris3d=True, cells=(20, 20, 20), steps=2),
    # harris 3D sized by memory (tier()): 10^3 at 100 ppc vs PHARE on GitHub (slow.yml), 80^3 at
    #  100 ppc locally (51M level 0 particles; 8 ranks or 8 threads)
    "harris_3d_10": dict(harris3d=True, cells=(10, 10, 10), ppc=100, steps=4, mem=1.0,
                         tolerance=(1e-12, 1.5, 1e-10)),  # roundoff: 5.5e-14 locally
    "harris_3d_80": dict(harris3d=True, cells=(80, 80, 80), ppc=100, steps=4, mem=25.0),
    # compare_physics.py --all-times: PHARE's tests/functional/mhd_shock/shock_periodic.py (MHD
    #  only, one level), dumps every 50 of its 400 steps
    "mhd_shock_1d": dict(mhd_shock=True, steps=400, dump_every=50, pops=[]),
    # PHARE's shock_with_boundaries.py: inflow (super magnetofast, the left state) at x lower,
    #  open at x upper, the shock moving in at 1.05 the fastest wave; dumps every 100 of 1000 steps
    "mhd_shock_walls_1d": dict(mhd_shock=True, walls=True, steps=1000, dump_every=100, pops=[]),
    # the periodic shock on 2 levels, the finer where PHARE's default tagger (B) puts it, regrids
    #  following the waves; dumps every 10 of 100 steps
    "mhd_shock_tagged_1d": dict(mhd_shock=True, levels=2, steps=100, dump_every=10, pops=[]),
    # PHARE's tests/simulator/advance/test_advance_mhd.py input (bump density, cosine B and V),
    #  1D, the ported scheme (Linear, VanLeer, Rusanov, TVDRK2), no resistivity; one refinement
    #  box: the refined level made from level 0 (init), then advanced
    "mhd_advance_L0": dict(mhd_advance=True, boxes=(), steps=8, pops=[]),
    # resistive (eta: the dissipative E in the constrained transport and the face fluxes)
    "mhd_resistive_L0": dict(mhd_advance=True, eta=0.05, boxes=(), steps=8, pops=[]),
    "mhd_resistive_L1": dict(mhd_advance=True, eta=0.05, boxes=((10, 29),), steps=8, pops=[]),
    # the other reconstructions (PHARE's test_advance_mhd's own: WENOZ, no limiter)
    "mhd_wenoz_L1": dict(mhd_advance=True, reconstruction="WENOZ", limiter="None",
                         boxes=((10, 29),), steps=8, pops=[]),
    "mhd_hll_L1": dict(mhd_advance=True, riemann="HLL", boxes=((10, 29),), steps=8, pops=[]),
    "mhd_hlld_L1": dict(mhd_advance=True, riemann="HLLD", boxes=((10, 29),), steps=8, pops=[]),
    # Hall MHD (the Hall term in the fluxes and the constrained transport; PHARE's whistler cw: 0)
    "mhd_hall_L0": dict(mhd_advance=True, hall=True, boxes=(), steps=8, pops=[]),
    "mhd_hall_L1": dict(mhd_advance=True, hall=True, boxes=((10, 29),), steps=8, pops=[]),
    # hyper resistivity (with Hall only, as PHARE): the constant mode alone (nu small: explicit
    #  4th order diffusion, stable for dt nu / dx^4 small: 0.016 at L1), the spatial mode
    #  (nu dx^2 (|B| / rho + 1)) with resistivity
    "mhd_hall_hyper_L1": dict(mhd_advance=True, hall=True, nu=1e-4, boxes=((10, 29),), steps=8,
                              pops=[]),
    "mhd_hall_hyper_spatial_L1": dict(mhd_advance=True, hall=True, eta=0.05, nu=0.01,
                                      hyper_mode="spatial", boxes=((10, 29),), steps=8, pops=[]),
    # HLLD on the cells' values (no reconstruction: PHARE's res/sim/all.txt has this one)
    "mhd_hlld_constant_L1": dict(mhd_advance=True, reconstruction="Constant", limiter="None",
                                 riemann="HLLD", boxes=((10, 29),), steps=8, pops=[]),
    "mhd_init_L1": dict(mhd_advance=True, boxes=((10, 29),), steps=0, pops=[]),
    "mhd_advance_L1": dict(mhd_advance=True, boxes=((10, 29),), steps=8, pops=[]),
    # the refined level against the periodic boundary (its ghosts wrap around)
    "mhd_advance_L1_edge": dict(mhd_advance=True, boxes=((1, 10),), steps=8, pops=[]),
    # tagged (PHARE's default tagger on B, a low threshold: the cosine B's steeper parts), 2
    #  levels, regrids at PHARE's points
    "mhd_tagged_1d": dict(mhd_advance=True, tagging=0.008, boxes=(), steps=40, pops=[]),
}

# the expected agreement with PHARE (compare_physics.py --tolerance: fails above it): the
#  difference relative to the largest |value| at the k-th time compared, min(cap, start *
#  growth**k): tight first (a bug shows at once), looser as round off grows. The MHD cases agree
#  to round off (2026-11-19, double diagnostics: identical at t0, at most 3.4e-14 after 1000
#  steps): 10x that and more. Not the hybrid ones (particle noise) nor mhd_shock_tagged_1d
#  (NaN in PHARE itself)
MHD_TOLERANCE = (1e-13, 1.25, 1e-11)
# the long shocks (hundreds of steps) grow their round off faster once the shock interacts (in CI:
#  the walls' reflections, ~1e-14 for 5 dumps then 2e-12 by the 10th): 2.6x and more over that
SHOCK_TOLERANCE = (1e-13, 1.6, 1e-11)
for _name in ("mhd_shock_1d", "mhd_shock_walls_1d"):
    CASES[_name]["tolerance"] = SHOCK_TOLERANCE
for _name, _case in CASES.items():
    if (_case.get("mhd_advance") or _case.get("mhd_shock")) and _name != "mhd_shock_tagged_1d":
        _case.setdefault("tolerance", MHD_TOLERANCE)


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


def populations(name):
    return CASES[name].get("pops", POPULATIONS)


def _ion_ion_beam_1d(ph, out, case, largest_patch_size, fmt, fine_dump):
    """PHARE's tests/functional/ionIonBeam/ion_ion_beam1d.py, its time step to t=steps * 0.001
    (its diagnostics every 0.1: here every step), the populations seeded, 500 particles per cell
    (100 there)"""
    sim = ph.Simulation(
        smallest_patch_size=20,
        largest_patch_size=largest_patch_size or 60,
        time_step_nbr=case["steps"],
        time_step=0.001,
        cells=165,
        dl=0.2,
        hyper_resistivity=0.01,
        # the script's commented boxes: its uniform plasma tags nothing, one level otherwise
        refinement_boxes=case.get(
            "refinement_boxes", {"L0": {"B0": [(50,), (110,)]}, "L1": {"B0": [(140,), (180,)]}}
        ),
        diag_options=_diag_options(out, fmt, fine_dump),
    )
    def const(v):
        return lambda x: v + np.zeros_like(x)

    vth = const(np.sqrt(0.1))
    ppc = int(os.environ.get("LADDER_PPC", "500"))  # many: the deposits' sums differ if anything does
    v = {"vbulky": const(0.0), "vbulkz": const(0.0), "vthx": vth, "vthy": vth, "vthz": vth,
         "nbr_part_per_cell": ppc}
    ph.MaxwellianFluidModel(
        bx=const(1.0),
        by=const(0.0),
        bz=const(0.0),
        main={"charge": 1, "density": const(1.0), "vbulkx": const(0.0), **v,
              "init": {"seed": SEED}},
        beam={"charge": 1, "density": const(0.01), "vbulkx": const(5.0), **v,
              "init": {"seed": SEED + 1}},
    )
    ph.ElectronModel(closure="isothermal", Te=0.0)
    # no LoadBalancer (the script's auto nppc): PHARE refuses one with refinement boxes
    every = case.get("dump_every", 1)
    _diagnostics(ph, 0.001 * np.arange(0, case["steps"] + 1, every), case["pops"], fmt)
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
        cells=case.get("cells", (20, 20, 20)),
        dl=(0.4, 0.4, 0.4),
        refinement="tagging",
        max_nbr_levels=3,
        nesting_buffer=1,
        tagging_threshold=float(os.environ.get("LADDER_TAGGING", case.get("tagging", 0.5))),
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
            "nbr_part_per_cell": case.get("ppc", PPC),
            "init": {"seed": int(os.environ.get("LADDER_SEED", "12"))},
        },
    )
    ph.ElectronModel(closure="isothermal", Te=0.0)
    if os.environ.get("LADDER_LB", "1") == "1":  # 0: none (PHARE re-initializes level 0 at
        #  each rebalance, even on one rank: its moments from its particles, not the synchronized)
        ph.LoadBalancer(every=1, mode="nppc", tol=0.001)
    _diagnostics(ph, time_step * np.arange(case["steps"] + 1), POPULATIONS, fmt)
    return sim


def _advance_mhd_1d(ph, out, case, largest_patch_size, fmt, fine_dump):
    """PHARE's test_advance_mhd model in 1D: 60 cells of 0.1, dt 0.001, every step dumped"""
    cells, dl, dt = 60, 0.1, 0.001
    sim = ph.Simulation(
        time_step_nbr=max(case["steps"], 1),
        time_step=dt,
        cells=(cells,),
        dl=(dl,),
        **(
            dict(refinement="tagging", max_nbr_levels=2, tagging_threshold=case["tagging"])
            if case.get("tagging")
            else dict(refinement_boxes=_boxes(1, *case["boxes"]) if case["boxes"] else {})
        ),
        largest_patch_size=largest_patch_size or 10,
        diag_options=_diag_options(out, fmt, fine_dump),
        strict=True,
        nesting_buffer=1,
        eta=case.get("eta", 0.0),
        nu=case.get("nu", 0.0),
        hyper_mode=case.get("hyper_mode", "constant"),
        gamma=5.0 / 3.0,
        reconstruction=case.get("reconstruction", "Linear"),
        limiter=case.get("limiter", "VanLeer"),
        riemann=case.get("riemann", "Rusanov"),
        mhd_timestepper="TVDRK2",
        hall=case.get("hall", False),
        model_options=["MHDModel"],
    )
    L = sim.simulation_domain()[0]

    def density(x):
        return 0.3 + np.exp(-((x - L / 2) ** 2))

    def cosine(x):
        return 0.1 * np.cos(2 * np.pi * x / L)

    def const(v):
        return lambda x: v + 0.0 * x

    ph.MHDModel(density=density, vx=cosine, vy=cosine, vz=cosine, bx=const(1.0), by=cosine,
                bz=cosine, p=const(1.0))
    timestamps = dt * np.arange(case["steps"] + 1)
    ph.ElectromagDiagnostics(quantity="B", write_timestamps=timestamps)
    for q in ["rho", "V", "P"]:
        ph.MHDDiagnostics(quantity=q, write_timestamps=timestamps)
    return sim


def _fast_speed(gamma, rho, p, bx, by):
    """the fast magnetosonic speed along x (PHARE's mhd_shock common.State)"""
    cs2, va2 = gamma * p / rho, (bx**2 + by**2) / rho
    common = np.sqrt((cs2 + va2) ** 2 - 4.0 * cs2 * bx**2 / rho)
    return np.sqrt((cs2 + va2 + common) / 2)


def _mhd_shock_1d(ph, out, case, largest_patch_size, fmt):
    """PHARE's tests/functional/mhd_shock (common.py config): 800 cells, B, rho, V, P every
    dump_every steps. shock_periodic.py: dt 0.2, the discontinuity at the middle, at rest;
    shock_with_boundaries.py (walls): inflow of the left state at x lower, open at x upper, the
    discontinuity 10 cells in, moving at 1.05 the fastest wave, dt half the CFL limit"""
    gamma, bx, ncells, dx = 5.0 / 3.0, 0.75, 800, 1.0
    left = dict(rho=1.0, p=1.0, by=1.0)
    right = dict(rho=0.125, p=0.1, by=-1.0)
    extra = {}
    if case.get("walls"):
        cf_max = max(_fast_speed(gamma, s["rho"], s["p"], bx, s["by"]) for s in (left, right))
        u = 1.05 * cf_max
        dt = 0.5 * dx / (u + cf_max)
        x0 = 10 * dx
        extra["domain_boundaries"] = {
            "xlower": {
                "type": "super-magnetofast-inflow",
                "velocity": u,
                "density": left["rho"],
                "pressure": left["p"],
                "B": [bx, left["by"], 0.0],
            },
            "xupper": {"type": "open"},
        }
    else:
        u, dt, x0 = 0.0, 0.2, ncells * dx / 2
    sim = ph.Simulation(
        smallest_patch_size=15,
        largest_patch_size=largest_patch_size,
        time_step=dt,
        final_time=case["steps"] * dt,
        cells=(ncells,),
        dl=(dx,),
        refinement="tagging",
        max_mhd_level=case.get("levels", 1),
        max_nbr_levels=case.get("levels", 1),
        hyper_resistivity=0.0,
        resistivity=0.0,
        diag_options=_diag_options(out, fmt, False),
        strict=True,
        eta=0.0,
        nu=0.0,
        gamma=gamma,
        reconstruction="Linear",
        limiter="VanLeer",
        riemann="Rusanov",
        mhd_timestepper="TVDRK2",
        hall=False,
        model_options=["MHDModel"],
        **extra,
    )

    def side(name):
        return lambda x: np.where(x < x0, left[name], right[name])

    def const(v):
        return lambda x: v + 0.0 * x

    ph.MHDModel(
        density=side("rho"), vx=const(u), vy=const(0.0), vz=const(0.0),
        bx=const(bx), by=side("by"), bz=const(0.0), p=side("p"),
    )
    timestamps = dt * np.arange(0, case["steps"] + 1, case["dump_every"])
    ph.ElectromagDiagnostics(quantity="B", write_timestamps=timestamps)
    for q in ["rho", "V", "P"]:
        ph.MHDDiagnostics(quantity=q, write_timestamps=timestamps)
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
    if case.get("ion_ion_beam"):
        return _ion_ion_beam_1d(ph, out, case, largest_patch_size, fmt, fine_dump)
    if case.get("harris3d"):
        return _harris_3d(ph, out, case, largest_patch_size, fmt, fine_dump)
    if case.get("mhd_shock"):
        return _mhd_shock_1d(ph, out, case, largest_patch_size, fmt)
    if case.get("mhd_advance"):
        return _advance_mhd_1d(ph, out, case, largest_patch_size, fmt, fine_dump)
    return _advance_hybrid(ph, out, case, ndim, largest_patch_size, fmt, fine_dump)
