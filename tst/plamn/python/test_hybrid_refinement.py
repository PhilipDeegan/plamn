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
coarse particles under it, split.
"""

import unittest

import numpy as np

import pyplamn

pyplamn.patch()

import pyphare.pharein as ph  # noqa: E402
from pyphare.simulator.simulator import Simulator  # noqa: E402

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
    simulator = Simulator(sim, log_to_file=False).setup().initialize()
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


def harris(steps):
    ph.global_vars.sim = None
    ph.clearDict()
    L = 0.5
    sim = ph.Simulation(
        time_step=0.005,
        time_step_nbr=steps,
        cells=(100, 100),
        dl=(0.4, 0.4),
        largest_patch_size=(25, 25),
        hyper_resistivity=0.002,
        resistivity=0.001,
        refinement="boxes",
        refinement_boxes={"L0": {"B0": [(30, 22), (69, 37)]}},  # around the lower sheet
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
    simulator = Simulator(sim, log_to_file=False).setup().initialize()
    cpp = simulator.cpp_sim
    initial = [cpp.total_particle_count(ln) for ln in range(cpp.n_levels())]
    for _ in range(steps):
        simulator.advance()
    out = {
        "levels": cpp.n_levels(),
        "initial": initial,
        "particles": [cpp.total_particle_count(ln) for ln in range(cpp.n_levels())],
        "B": [
            np.concatenate(
                [np.asarray(cpp.all_domain_values(f"EM_B_{q}", ln)) for q in "xyz"]
            )
            for ln in range(cpp.n_levels())
        ],
    }
    simulator.reset()
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


if __name__ == "__main__":
    unittest.main()
