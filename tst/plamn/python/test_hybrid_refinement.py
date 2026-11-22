"""two levels with a static refinement box (pyphare's refinement="boxes"), on plamn

The fine level takes ratio^2 = 4 steps per coarse step (dt ~ dx^2, PHARE's getMaxFinerLevelDt),
its level ghosts are refined from the coarse level in time between the coarse step's start and
end, and it is synchronized onto the coarse level after its substeps, the coarse B under it
recomputed from its time averaged E (reflux).

1D: the same Alfven wave, from the same particles, on one level and on two. Level 0 B beside the
box differs between the two by no more than inside it: a wrongly weighted reflux kicks the coarse
cells sharing the coarse-fine interface (see PHARE's tests/simulator/refinement). Far from the box
nothing changes in a step. The fine level holds the coarse particles under it, split.

2D: a small double Harris sheet with one sheet refined stays finite, its fine level holding the
coarse particles under it, split. With refinement="tagging", level 1 starts on the sheets and is
regridded every 2 steps, divergence B kept at its initial (discretization) value.

Load balancing (pyphare's LoadBalancer, with tagging, on more than one rank): a beam drifting
across the patches unbalances the ranks; rebalancing brings them back without changing B, also
when whole patches are too coarse to balance (level 0 cut on blocks, damn::amr::cut_partition).
"""

import os
import shutil
import unittest
from pathlib import Path

import numpy as np

import plamn
import plamn.backend.phare as ph

PPC = 100


def alfven(refined, steps):
    ph.global_vars.sim = None
    ph.clearDict()
    sim = ph.Simulation(
        time_step=0.01,
        time_step_nbr=steps,
        cells=100,
        dl=1.0,
        largest_patch_size=25,
        hyper_resistivity=0.001,
        **(
            dict(refinement="boxes", refinement_boxes={"L0": {"B0": [(40,), (59,)]}})
            if refined
            else {}
        ),
    )
    k = 2 * np.pi / sim.simulation_domain()[0]

    def const(v):
        return lambda x: v + 0 * x

    ph.MaxwellianFluidModel(
        bx=const(1.0),
        by=lambda x: 0.01 * np.cos(k * x),
        bz=lambda x: 0.01 * np.sin(k * x),
        protons={
            "charge": 1,
            "density": const(1.0),
            "vbulkx": const(0.0),
            "vbulky": lambda x: 0.01 * np.cos(k * x),
            "vbulkz": lambda x: 0.01 * np.sin(k * x),
            "vthx": const(0.01),
            "vthy": const(0.01),
            "vthz": const(0.01),
            "nbr_part_per_cell": PPC,
            "init": {"seed": 1337},
        },
    )
    ph.ElectronModel(closure="isothermal", Te=0.0)
    simulator = ph.Simulator(sim, log_to_file=False).setup().initialize()
    for _ in range(
        steps
    ):  # not run(): newer pyphares reset the simulator once it is done
        simulator.advance()
    cpp = simulator.cpp_sim
    out = {
        "B": {q: np.asarray(cpp.all_domain_values(f"EM_B_{q}")) for q in "yz"},
        "particles": [cpp.total_particle_count(ln) for ln in range(cpp.n_levels())],
    }
    simulator.reset()
    return out


def harris(
    steps,
    tagging=False,
    measure=None,
    load_balancer=None,
    levels=2,
    cells=(100, 100),
    patch=(25, 25),
    tag_boxes=None,
    by_node=False,  # out["final"]: b_by_node at the end
):
    ph.global_vars.sim = None
    ph.clearDict()
    L = 0.5
    refinement = (
        dict(refinement="tagging", max_nbr_levels=levels)
        if tagging
        else dict(
            refinement="boxes", refinement_boxes={"L0": {"B0": [(30, 22), (69, 37)]}}
        )
    )
    sim = ph.Simulation(
        time_step=0.005,
        time_step_nbr=steps,
        cells=cells,
        dl=(0.4, 0.4),
        largest_patch_size=patch,
        hyper_resistivity=0.002,
        resistivity=0.001,
        **refinement,  # boxes: around the lower sheet
        strict=True,
    )
    Lx, Ly = sim.simulation_domain()

    def density(x, y):
        return (
            0.4
            + 1 / np.cosh((y - Ly * 0.3) / L) ** 2
            + 1 / np.cosh((y - Ly * 0.7) / L) ** 2
        )

    def S(y, y0):
        return 0.5 * (1 + np.tanh((y - y0) / L))

    def perturbation(x, y):
        x0, y1, y2 = x - 0.5 * Lx, y - 0.3 * Ly, y - 0.7 * Ly
        return x0, y1, y2, np.exp(-(x0**2 + y1**2)), np.exp(-(x0**2 + y2**2))

    def bx(x, y):
        _, y1, y2, e1, e2 = perturbation(x, y)
        return (
            -1 + 2 * (S(y, Ly * 0.3) - S(y, Ly * 0.7)) - 0.2 * y1 * e1 + 0.2 * y2 * e2
        )

    def by(x, y):
        x0, _, _, e1, e2 = perturbation(x, y)
        return 0.2 * x0 * e1 - 0.2 * x0 * e2

    def bz(x, y):
        return 0.0 * x

    def vth(x, y):
        return np.sqrt((0.7 - 0.5 * (bx(x, y) ** 2 + by(x, y) ** 2)) / density(x, y))

    def zero(x, y):
        return 0.0 * x

    ph.MaxwellianFluidModel(
        bx=bx,
        by=by,
        bz=bz,
        protons={
            "charge": 1,
            "density": density,
            **{f"vbulk{c}": zero for c in "xyz"},
            **{f"vth{c}": vth for c in "xyz"},
            "nbr_part_per_cell": 20,
            "init": {"seed": 12334},
        },
    )
    ph.ElectronModel(closure="isothermal", Te=0.0)
    if load_balancer:
        ph.LoadBalancer(**load_balancer)
    if tag_boxes:
        plamn.set_tag_boxes(sim, tag_boxes)
    simulator = ph.Simulator(sim, log_to_file=False).setup().initialize()
    cpp = simulator.cpp_sim
    initial = [cpp.total_particle_count(ln) for ln in range(cpp.n_levels())]
    measured = [measure(simulator)] if measure else []
    for _ in range(steps):
        simulator.advance()
        if measure:
            measured.append(measure(simulator))
    final = b_by_node(simulator) if by_node else None
    out = {
        "final": final,
        "levels": cpp.n_levels(),
        "initial": initial,
        "particles": [cpp.total_particle_count(ln) for ln in range(cpp.n_levels())],
        "B": [
            np.concatenate(
                [np.asarray(cpp.all_domain_values(f"EM_B_{q}", ln)) for q in "xyz"]
            )
            for ln in range(cpp.n_levels())
        ],
        "measured": measured,
        "boxes": {
            ln: cpp.all_patch_boxes(ln) for ln in range(1, cpp.n_levels())
        },
    }
    simulator.reset()
    return out


def drift(load_balancer, steps=120, patch=(10, 10), diag_dir=None, before_step=None):
    """level 0's rank imbalance (max / mean - 1 of the particles per rank) after each step, and
    B at the end: uniform protons and a drifting beam blob, refinement="tagging" (2 levels).
    diag_dir: the beam's pressure tensor (a computed quantity: scratch grids on the patches it
    is computed on) dumped there at t0, before the rebalances move patches between ranks.
    before_step(simulator, k): called before step k
    """
    ph.global_vars.sim = None
    ph.clearDict()
    sim = ph.Simulation(
        time_step=0.01,
        time_step_nbr=steps,
        cells=(60, 60),
        dl=(0.2, 0.2),
        largest_patch_size=patch,
        hyper_resistivity=0.001,
        refinement="tagging",
        max_nbr_levels=2,
        strict=True,
        **(
            dict(diag_options={"format": "phareh5", "options": {"dir": diag_dir, "mode": "overwrite"}})
            if diag_dir
            else {}
        ),
    )
    L = sim.simulation_domain()[0]

    def blob(x, y):
        dx = (x + L / 2) % L - L / 2  # centred at x = 0, periodic
        rho = 0.5 * np.exp(-(dx**2 + (y - L / 2) ** 2))
        rho[rho < 1e-3] = 0  # no beam particles there
        return rho

    def const(v):
        return lambda x, y: v + 0 * x

    def pop(density, vx, ppc, seed):
        return {
            "charge": 1,
            "density": density,
            "vbulkx": const(vx),
            "vbulky": const(0.0),
            "vbulkz": const(0.0),
            **{f"vth{c}": const(0.3) for c in "xyz"},
            "nbr_part_per_cell": ppc,
            "init": {"seed": seed},
        }

    ph.MaxwellianFluidModel(
        bx=const(1.0),
        by=const(0.0),
        bz=const(0.0),
        protons=pop(const(1.0), 0.0, 10, 1),
        beam=pop(blob, 3.0, 100, 2),
    )
    ph.ElectronModel(closure="isothermal", Te=0.0)
    if diag_dir:
        ph.FluidDiagnostics(quantity="pressure_tensor", write_timestamps=[0.0],
                            population_name="beam")
    if load_balancer:
        ph.LoadBalancer(**load_balancer)
    simulator = ph.Simulator(sim, log_to_file=False).setup().initialize()
    cpp = simulator.cpp_sim
    imbalance = []
    for k in range(steps + 1):
        per_rank = np.asarray(cpp.particle_counts_per_rank(0), dtype=float)
        imbalance.append(per_rank.max() / per_rank.mean() - 1)
        if k < steps:
            if before_step:
                before_step(simulator, k)
            simulator.advance()
    B = np.concatenate([np.asarray(cpp.level_values(f"EM_B_{q}", 0)) for q in "xyz"])
    simulator.reset()
    return {"imbalance": imbalance, "B": B}


def init_fields(simulator):
    """every level's E, B and protons density on its patches' domains, at initialization"""
    cpp = simulator.cpp_sim
    names = [f"EM_{v}_{q}" for v in "EB" for q in "xyz"] + ["protons_density"]
    return [
        np.concatenate([np.asarray(cpp.all_domain_values(n, ln)) for n in names])
        for ln in range(cpp.n_levels())
    ]


def level_fields(simulator):
    """every level's E, B and protons density at each of its points once, in point order: the
    same for any patches"""
    cpp = simulator.cpp_sim
    names = [f"EM_{v}_{q}" for v in "EB" for q in "xyz"] + ["protons_density"]
    return [
        np.concatenate([np.asarray(cpp.level_values(n, ln)) for n in names])
        for ln in range(cpp.n_levels())
    ]


def b_by_node(simulator):
    """{(level, component, AMR node): B} of every patch's domain nodes, every rank's on rank 0
    (empty on the others; collective): comparable whatever the patches (all_domain_values is
    in patch order)"""
    dw = simulator.data_wrangler().cpp
    out = {}
    for ln in range(dw.getNumberOfLevels()):
        for c in "xyz":
            for p in dw.sync(dw.getHybridPatchLevel(ln).getB(c)):
                lo, up, g = np.asarray(p.lower), np.asarray(p.upper), int(p.nGhosts)
                v = np.asarray(p.data)
                n = [min(v.shape[d] - 2 * g, up[d] - lo[d] + 2) for d in range(v.ndim)]
                for idx in np.ndindex(*n):  # the patch's nodes (a dual direction: its cells)
                    out[(ln, c, *(lo[d] + idx[d] for d in range(v.ndim)))] = v[
                        tuple(g + i for i in idx)]
    return out


def max_divb(simulator):
    """max |div B| per level, over every patch's cells, over the ranks (collective)"""
    from pyphare.pharesee.hierarchy.fromsim import hierarchy_from_sim

    hx = hierarchy_from_sim(simulator, qty="EM_B_x")
    hy = hierarchy_from_sim(simulator, qty="EM_B_y")
    out = []
    for ln in range(
        simulator.cpp_sim.n_levels()
    ):  # every rank: a level may have none here
        dx = 0.4 / 2**ln
        m = 0.0
        here = ln in hx.levels()
        for px, py in zip(*(h.levels()[ln].patches if here else [] for h in (hx, hy))):
            bx = px.patch_datas["EM_B_x"][px.box]
            by = py.patch_datas["EM_B_y"][py.box]
            div = (bx[1:, :] - bx[:-1, :]) / dx + (by[:, 1:] - by[:, :-1]) / dx
            m = max(m, float(np.abs(div).max()))
        out.append(plamn.etc.mpi_max(m))
    return out


class HybridRefinementTest(unittest.TestCase):
    def test_alfven_reflux_does_not_kick_the_coarse_cells_beside_the_box(self):
        one, two = alfven(False, 1), alfven(True, 1)
        lo, up = 40, 59
        for q in "yz":
            d = np.abs(two["B"][q] - one["B"][q])
            beside = max(d[lo - 1], d[up + 1])
            inside = d[lo + 1 : up].max()
            far = np.concatenate([d[: lo - 4], d[up + 5 :]]).max()
            self.assertEqual(far, 0)
            self.assertLess(
                beside, 3 * inside, f"B{q}: beside {beside}, inside {inside}"
            )

        # 20 coarse cells refined, each coarse particle split in 2 (1D, interp 1)
        fine = two["particles"][1]
        self.assertLess(abs(fine - 20 * PPC * 2) / (20 * PPC * 2), 0.01)
        self.assertEqual(two["particles"][0], one["particles"][0])

    def test_alfven_two_levels_follow_one(self):
        one, two = alfven(False, 100), alfven(True, 100)
        for q in "yz":
            # the fine level's particles are split: its noise is its own
            self.assertLess(np.abs(two["B"][q] - one["B"][q]).max(), 0.1 * 0.01)

    def test_harris_two_levels_stay_finite(self):
        out = harris(200)
        self.assertEqual(out["levels"], 2)
        # 40 x 16 coarse cells refined, each coarse particle split in 4 (2D, interp 1)
        expected = 40 * 16 * 20 * 4
        self.assertLess(abs(out["initial"][1] - expected) / expected, 0.01)
        for B in out["B"]:
            self.assertTrue(np.all(np.isfinite(B)))
            self.assertLess(np.abs(B).max(), 2)

    def test_harris_tagging_regrids_keep_div_b(self):
        out = harris(40, tagging=True, measure=max_divb)
        self.assertEqual(out["levels"], 2)
        divb = out["measured"]
        # the initial condition's discrete divergence is the most there is, on every level,
        #  through every regrid
        initial = max(divb[0])
        for d in divb:
            for ln in range(2):
                self.assertLess(d[ln], initial * (1 + 1e-9) + 1e-12, divb)
        for B in out["B"]:
            self.assertTrue(np.all(np.isfinite(B)))
            self.assertLess(np.abs(B).max(), 2)

    def test_load_balancing_moves_patches_not_physics(self):
        """a beam population, only in a blob, drifting along B across the patches: without
        load balancing the ranks' particles drift apart, every 30 steps it brings them back.
        The physics is the same: B at the end as without it, to round off"""
        ph.startMPI()  # before asking how many ranks: pyphare starts it with the first simulator
        if plamn.etc.mpi_size() < 2:
            self.skipTest("one rank: nothing to balance (--tags mpi runs it on 3)")
        no, lb = drift(None), drift(dict(every=30, tol=0.05))
        for step in (40, 100):  # after a rebalance (at the start of steps 30, 60, 90)
            self.assertLess(lb["imbalance"][step], no["imbalance"][step], step)
        self.assertGreater(max(no["imbalance"]), 0.4)
        self.assertLess(lb["imbalance"][100], 0.15)
        self.assertLess(np.abs(lb["B"] - no["B"]).max(), 1e-12)

    def test_threads_take_patches_not_physics(self):
        """the pool's threads taking a level's patches (untiled layouts): B at the end as on one
        thread, digit for digit (each patch's work its own)"""
        def run(threads):
            os.environ["PHARE_THREADS_PER_POOL"] = str(threads)
            try:
                return drift(None, steps=20)["B"]
            finally:
                os.environ.pop("PHARE_THREADS_PER_POOL")

        np.testing.assert_array_equal(run(3), run(1))

    def test_load_balancing_moves_computed_quantities(self):
        """a computed quantity (the beam's pressure tensor) dumped before the rebalances: its
        scratch grids on every patch, so a patch moved to another rank unpacks what was packed
        (they were only on the patches it was computed on: an unpack out of step, a crash)"""
        ph.startMPI()
        if plamn.etc.mpi_size() < 2:
            self.skipTest("one rank: nothing to balance (--tags mpi runs it on 3)")
        # one directory for every rank (a parallel HDF5 file), one per rank count (phlop runs the
        #  1 and 3 rank cases at the same time)
        d = Path("phare_outputs") / f"plamn_lb_computed_np{plamn.etc.mpi_size()}"
        lb = drift(dict(every=30, tol=0.05), steps=40, diag_dir=str(d))
        plamn.etc.mpi_barrier()
        if plamn.etc.mpi_rank() == 0:
            shutil.rmtree(d, ignore_errors=True)
        self.assertLess(lb["imbalance"][40], max(lb["imbalance"]))
        self.assertTrue(np.isfinite(lb["B"]).all())

    def test_load_balancing_moves_quantities_computed_mid_run(self):
        """a computed quantity first asked for mid run (the DataWrangler's momentum tensor of
        the beam, at step 5): its scratch grids made then on every patch of every rank, so the
        rebalance at step 30 moves patches that all hold them"""
        ph.startMPI()
        if plamn.etc.mpi_size() < 2:
            self.skipTest("one rank: nothing to balance (--tags mpi runs it on 3)")

        def momentum_tensor(simulator, k):
            if k == 5:
                level = simulator.data_wrangler().getPatchLevel(0)
                for pd in level.getMomentumTensor("xx", "beam"):
                    self.assertTrue(np.isfinite(np.asarray(pd.data)).all())

        lb = drift(dict(every=30, tol=0.05), steps=40, before_step=momentum_tensor)
        self.assertLess(lb["imbalance"][40], max(lb["imbalance"]))
        self.assertTrue(np.isfinite(lb["B"]).all())

    def test_load_balancing_cuts_patches(self):
        """6 level 0 patches (20 x 30) for 3 ranks, the beam blob in one: initialization cuts
        them on blocks, the ranks within 0.15; the blob drifts, unbalanced past 0.25 without
        rebalancing by step 60; each rebalance (start of steps 30, 60) cuts level 0 on blocks
        and brings them within 0.15 again, B the same as without the rebalances"""
        ph.startMPI()
        if (
            plamn.etc.mpi_size() != 3
        ):  # 6 patches split evenly on 2 ranks: nothing to cut
            self.skipTest("built for 3 ranks (--tags mpi runs it on 3)")
        patch, steps = (20, 30), 61
        no = drift(None, steps, patch)
        lb = drift(dict(every=30, tol=0.05), steps, patch)
        self.assertLess(lb["imbalance"][0], 0.15)
        self.assertGreater(no["imbalance"][60], 0.25)
        for step in (1, 31, 61):  # just after initialization and each rebalance
            self.assertLess(lb["imbalance"][step], 0.15, step)
        self.assertLess(lb["imbalance"][61], no["imbalance"][61])
        self.assertLess(np.abs(lb["B"] - no["B"]).max(), 1e-12)

    def test_pretag_initializes_the_same_levels(self):
        # pretag: every level's boxes from B alone before any particle exists, then the levels
        # made knowing all of them: the same levels, fields and particles as tagging level by
        # level with the particles
        runs = {}
        for pretag in ("0", "1"):
            os.environ["PLAMN_PRETAG"] = pretag
            try:
                runs[pretag] = harris(0, tagging=True, levels=3, measure=init_fields)
            finally:
                del os.environ["PLAMN_PRETAG"]
        off, on = runs["0"], runs["1"]
        self.assertEqual(on["levels"], 3)
        self.assertEqual(on["levels"], off["levels"])
        self.assertEqual(on["initial"], off["initial"])
        for ln, (a, b) in enumerate(zip(off["measured"][0], on["measured"][0])):
            self.assertEqual(a.shape, b.shape, f"level {ln}")
            np.testing.assert_array_equal(a, b, f"level {ln}")

    def test_supplied_tag_boxes_make_the_same_levels(self):
        # the levels of a tagged initialization, supplied as tag boxes (as from Tagger(sim).tag()):
        # the same values at every point of every level, the levels made before any particle
        tagged = harris(0, tagging=True, levels=3, measure=level_fields)
        supplied = harris(
            0, tagging=True, levels=3, measure=level_fields, tag_boxes=tagged["boxes"]
        )
        self.assertEqual(supplied["levels"], 3)
        self.assertEqual(supplied["initial"], tagged["initial"])
        for ln, (a, b) in enumerate(zip(tagged["measured"][0], supplied["measured"][0])):
            np.testing.assert_array_equal(a, b, f"level {ln}")

    def test_supplied_tag_boxes_must_nest(self):
        # pyphare's setup raises a ValueError from the simulator's error
        with self.assertRaises(ValueError) as raised:
            harris(
                0,
                tagging=True,
                levels=3,
                tag_boxes={1: [((0, 0), (9, 9))], 2: [((100, 100), (109, 109))]},
            )
        self.assertIn("not nested", str(raised.exception.__context__))

    def test_load_balancing_keeps_the_finer_levels(self):
        """three levels from the start (the sheets are tagged at initialization): a rebalance
        moves every level onto its new owners, coarser to finer, none dropped (a dropped level
        would come back at the next regrid, split anew from its coarser level: other physics).
        The same levels at every step and B as without load balancing"""
        ph.startMPI()
        if plamn.etc.mpi_size() < 2:
            self.skipTest("one rank: nothing to balance (--tags mpi runs it on 3)")
        steps = 4
        run = lambda lb: harris(
            steps,
            tagging=True,
            measure=lambda sim: sim.cpp_sim.n_levels(),
            load_balancer=lb,
            levels=3,
            cells=(
                40,
                100,
            ),  # y as harris()'s default: its sheets periodic to round off
            patch=(10, 10),
        )
        no, lb = run(None), run(dict(every=2, tol=0.05))
        self.assertEqual(no["measured"], [3] * (steps + 1))  # three levels throughout
        self.assertEqual(lb["measured"], no["measured"])
        for b_lb, b_no in zip(lb["B"], no["B"]):
            self.assertLess(np.abs(b_lb - b_no).max(), 1e-12)

    def test_load_balancing_finest(self):
        """PLAMN_PARTITION=finest with rebalances (every 2 steps): each rebalance decomposes every
        level together (their boxes known), the regrids between them level by level (a level to
        the owners of the coarser cells under it). Three levels throughout, B as without load
        balancing"""
        ph.startMPI()
        if plamn.etc.mpi_size() < 2:
            self.skipTest("one rank: nothing to balance (--tags mpi runs it on 3)")
        steps = 4

        def run(lb):
            os.environ.update(PLAMN_PARTITION="finest", PLAMN_SPILL="0.2")
            try:
                return harris(steps, tagging=True, measure=lambda sim: sim.cpp_sim.n_levels(),
                              load_balancer=lb, levels=3, cells=(40, 100), patch=(10, 10),
                              by_node=True)
            finally:
                for key in ("PLAMN_PARTITION", "PLAMN_SPILL"):
                    os.environ.pop(key)

        no, lb = run(None), run(dict(every=2, tol=0.05))
        self.assertEqual(no["measured"], [3] * (steps + 1))
        self.assertEqual(lb["measured"], no["measured"])
        plamn.etc.mpi_barrier()
        if plamn.etc.mpi_rank() != 0:
            return
        # by node: the rebalanced run's patches differ
        a, b = lb["final"], no["final"]
        self.assertEqual(a.keys(), b.keys())
        self.assertLess(max(abs(a[k] - b[k]) for k in a), 1e-12)

    def test_keep_partition(self):
        """PLAMN_PARTITION=keep (with PLAMN_SPILL): a regridded level's patches stay with their
        data's owner, a new one with the coarser patch's under it, then spilled to within the
        tolerance. The same physics as the curve ("levels"); on more than one rank, fewer of the
        coarse cells under a fine patch on another rank, every level within its tolerance"""
        from plamn.amr import read_decomposition

        ph.startMPI()
        size, rank = plamn.etc.mpi_size(), plamn.etc.mpi_rank()
        steps, spill = 4, 0.2
        out, logs = {}, {}
        for mode in ("levels", "keep"):
            path = Path("phare_outputs") / f"partition_{mode}_n{size}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            logs[mode] = path
            os.environ.update(PLAMN_PARTITION=mode, PLAMN_SPILL=str(spill),
                              PLAMN_DECOMPOSITION=str(path))
            try:
                out[mode] = harris(steps, tagging=True, levels=3, cells=(40, 100),
                                   patch=(10, 10), by_node=True)
                out[mode]["nodes"] = out[mode]["final"]
            finally:
                for key in ("PLAMN_PARTITION", "PLAMN_SPILL", "PLAMN_DECOMPOSITION"):
                    os.environ.pop(key)
        self.assertEqual(out["keep"]["levels"], out["levels"]["levels"])
        plamn.etc.mpi_barrier()
        if rank != 0:
            return
        a, b = out["keep"]["nodes"], out["levels"]["nodes"]
        self.assertEqual(a.keys(), b.keys())  # the same levels' nodes
        self.assertLess(max(abs(a[k] - b[k]) for k in a), 1e-12)
        if size == 1:
            return
        last = {m: read_decomposition(p)[-1] for m, p in logs.items()}

        def crossing(rec):  # coarse cells under fine patches owned by another rank
            return sum(m["coarse_cells"]["Node"] + m["coarse_cells"]["Remote"]
                       for m in rec["levels"])

        self.assertLess(crossing(last["keep"]), crossing(last["levels"]))
        for m in last["keep"]["levels"]:
            share = sum(m["weight"]) / size
            heaviest = max(last["keep"]["hierarchy"].weights[m["level"]])
            self.assertLessEqual(max(m["weight"]), (1 + spill) * share + heaviest)

    def test_finest_partition(self):
        """PLAMN_PARTITION=finest with pretag (every level's boxes known at initialization):
        every level decomposed together, the finest along the curve, each coarser patch with the
        owner of the fine cells above it, then spilled. The same levels and physics as "levels";
        on more than one rank fewer of the coarse cells under a fine patch on another rank at
        initialization"""
        from plamn.amr import read_decomposition

        ph.startMPI()
        size, rank = plamn.etc.mpi_size(), plamn.etc.mpi_rank()
        out, logs = {}, {}
        for mode in ("levels", "finest"):
            path = Path("phare_outputs") / f"partition_pretag_{mode}_n{size}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            logs[mode] = path
            os.environ.update(PLAMN_PARTITION=mode, PLAMN_SPILL="0.2", PLAMN_PRETAG="1",
                              PLAMN_DECOMPOSITION=str(path))
            try:
                out[mode] = harris(4, tagging=True, levels=3, cells=(40, 100),
                                   patch=(10, 10), by_node=True)
            finally:
                for key in ("PLAMN_PARTITION", "PLAMN_SPILL", "PLAMN_PRETAG",
                            "PLAMN_DECOMPOSITION"):
                    os.environ.pop(key)
        self.assertEqual(out["finest"]["levels"], 3)
        self.assertEqual(out["finest"]["levels"], out["levels"]["levels"])
        plamn.etc.mpi_barrier()
        if rank != 0:
            return
        a, b = out["finest"]["final"], out["levels"]["final"]
        self.assertEqual(a.keys(), b.keys())
        self.assertLess(max(abs(a[k] - b[k]) for k in a), 1e-12)
        if size == 1:
            return
        first = {m: read_decomposition(p)[0] for m, p in logs.items()}

        def crossing(rec):  # coarse cells under fine patches owned by another rank
            return sum(m["coarse_cells"]["Node"] + m["coarse_cells"]["Remote"]
                       for m in rec["levels"])

        self.assertLess(crossing(first["finest"]), crossing(first["levels"]))

    def test_decomposition_log(self):
        """PLAMN_DECOMPOSITION: a line at initialization, after each regrid or rebalance that
        changes the decomposition, and every PLAMN_DECOMPOSITION_EVERY steps; each level's
        weights are its particles (every rank's), the metrics consistent with its patches"""
        from plamn.amr import read_decomposition

        ph.startMPI()
        size, rank = plamn.etc.mpi_size(), plamn.etc.mpi_rank()
        path = Path("phare_outputs") / f"decomposition_n{size}.jsonl"  # one per rank count:
        path.parent.mkdir(parents=True, exist_ok=True)  #  1 and 3 ranks run at once
        steps = 4
        os.environ.update(PLAMN_DECOMPOSITION=str(path), PLAMN_DECOMPOSITION_EVERY="2")
        try:
            out = harris(
                steps,
                tagging=True,
                levels=3,
                cells=(40, 100),
                patch=(10, 10),
                load_balancer=dict(every=2, tol=0.05) if size > 1 else None,
            )
        finally:
            for key in ("PLAMN_DECOMPOSITION", "PLAMN_DECOMPOSITION_EVERY"):
                os.environ.pop(key)
        plamn.etc.mpi_barrier()
        if rank != 0:
            return
        lines = read_decomposition(path)
        events = [r["event"] for r in lines]
        self.assertEqual(events[0], "init")
        self.assertEqual(events.count("sample"), steps // 2)
        self.assertIsNone(lines[0]["migration"])
        last = lines[-1]
        h = last["hierarchy"]
        self.assertEqual(len(h.levels), out["levels"])
        self.assertEqual(h.ranks, size)
        for ln, m in enumerate(last["levels"]):
            self.assertEqual(sum(h.weights[ln]), out["particles"][ln])
            self.assertAlmostEqual(sum(m["weight"]), out["particles"][ln])
            self.assertEqual(m["patches"], len(h.levels[ln]))
            self.assertEqual(
                m["cells"],
                sum(np.prod(np.asarray(up) - lo + 1) for lo, up, _ in h.levels[ln]),
            )
            self.assertGreaterEqual(m["level_ghost_cells"], 0)
            self.assertEqual(m["ghost_cells"]["Remote"], 0)  # one node
            if size == 1:
                self.assertEqual(m["ghost_cells"]["Node"], 0)
            if ln > 0:
                self.assertGreater(sum(m["coarse_cells"].values()), 0)
        for r in lines[1:]:
            self.assertEqual(len(r["migration"]), len(r["levels"]))
        if size > 1:  # three ranks: ghosts cross ranks, and every rank has peers
            self.assertGreater(last["levels"][0]["ghost_cells"]["Node"], 0)
            self.assertGreater(last["remote_ranks"][0][0], 0)
        self.assertGreaterEqual(last["critical_path"], 1.0)
        # busy time: every second of a level's patch work is a patch's of the decomposition it
        #  ran on (the cost fit: test_load_balancing, where particles do not follow cells)
        for prev, r in zip(lines, lines[1:]):
            for ln, m in enumerate(r["levels"]):
                if m["busy"]:
                    self.assertEqual(len(m["busy"]), len(prev["hierarchy"].levels[ln]))
                    self.assertAlmostEqual(sum(m["busy"]), sum(m["busy_rank"]), places=9)


if __name__ == "__main__":
    unittest.main()
