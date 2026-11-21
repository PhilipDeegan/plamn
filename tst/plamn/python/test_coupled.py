"""MHD under hybrid (AMR/max_mhd_level): a uniform plasma at rest, level 0 MHD, level 1 hybrid
over part of it, against the same with hybrid on both levels. Nothing should happen: level 1's
particles (spawned from MHD's state, or split from level 0's particles) keep the uniform density
as they cross its border. The particles differ, so the comparison is statistical, over seeds.
Coupled runs are plamn's own (no pyphare has coupling): plamn/__init__.py's coupled input.

needs py_etc, the 1D hybrid module and the coupled one built
(./mkn.py --dim 1, ./mkn.py --coupled --dim 1), then from the plamn root:
    PYTHONPATH=.:$PHARE/pyphare:tst/plamn/python python3 -m phlop.run.test_cases -i tst
"""

import unittest

import numpy as np

import plamn.backend.phare as ph

CELLS = 64
DX = 0.2
DT = 0.001
STEPS = 300
PPC = 50
TE = 0.5  # isothermal electrons: Pi = P - Te n = 0.5, vth = sqrt(0.5)
SEEDS = (1, 2, 3, 4)
FINE_CELLS = 40  # level 1 over level 0's cells 20 to 39


def const(v):
    return lambda x: v + 0.0 * x


def run(coupled, seed):
    ph.global_vars.sim = None
    ph.clearDict()
    common = dict(
        time_step=DT,
        time_step_nbr=STEPS,
        cells=(CELLS,),
        dl=(DX,),
        largest_patch_size=32,
        refinement_boxes={"L0": {"B0": [(20,), (39,)]}},
        hyper_resistivity=0.0,
        resistivity=0.0,
        eta=0.0,
        nu=0.0,
        gamma=5.0 / 3.0,
        hall=False,
    )
    species = {"charge": 1, "mass": 1, "nbr_part_per_cell": PPC, "init": {"seed": seed}}
    if coupled:
        ph.Simulation(
            **common,
            max_mhd_level=1,
            reconstruction="Linear",
            limiter="VanLeer",
            riemann="Rusanov",
            mhd_timestepper="TVDRK2",
            model_options=["MHDModel", "HybridModel"],
        )
        ph.MHDModel(
            density=const(1.0), vx=const(0.0), vy=const(0.0), vz=const(0.0),
            bx=const(1.0), by=const(0.0), bz=const(0.0), p=const(1.0),
            protons=species,
        )
    else:
        ph.Simulation(**common)
        vth = const((1.0 - TE) ** 0.5)
        ph.MaxwellianFluidModel(
            bx=const(1.0), by=const(0.0), bz=const(0.0),
            protons={
                **species,
                "density": const(1.0),
                **{f"vbulk{k}": const(0.0) for k in "xyz"},
                **{f"vth{k}": vth for k in "xyz"},
            },
        )
    ph.ElectronModel(closure="isothermal", Te=TE)

    simulator = ph.Simulator(ph.global_vars.sim, log_to_file=False).setup().initialize()
    cpp = simulator.cpp_sim
    for _ in range(STEPS):
        simulator.advance()
    out = {
        "count": cpp.particle_count(1),
        "density": np.asarray(cpp.level_values("protons_density", 1)),
        "bx": np.asarray(cpp.level_values("EM_B_x", 1)),
        "time": cpp.currentTime(),
    }
    if coupled:
        out["rho0"] = np.asarray(cpp.level_values("rho", 0))
    simulator.reset()
    return out


class CoupledUniformTest(unittest.TestCase):
    def test_hybrid_over_mhd_matches_hybrid_over_hybrid(self):
        coupled = [run(True, s) for s in SEEDS]
        hybrid = [run(False, s) for s in SEEDS]

        for c in coupled:
            self.assertAlmostEqual(c["time"], STEPS * DT)
            # B stays the uniform field; MHD is not fed back yet (no hybrid -> MHD sync)
            np.testing.assert_array_equal(c["bx"], 1.0)
            np.testing.assert_array_equal(c["rho0"], 1.0)
            self.assertTrue(np.isfinite(c["density"]).all())

        # particles and mean density: the same within the seeds' spread (each run's count
        #  random walks with the particles crossing level 1's border)
        counts = {k: np.array([r["count"] for r in v]) for k, v in (("c", coupled), ("h", hybrid))}
        means = {k: np.array([r["density"].mean() for r in v])
                 for k, v in (("c", coupled), ("h", hybrid))}
        expected = FINE_CELLS * PPC
        for name, values, target in (("count", counts, expected), ("density", means, 1.0)):
            spread = np.sqrt((values["c"].var(ddof=1) + values["h"].var(ddof=1)) / 2)
            sigma = spread / np.sqrt(len(SEEDS))  # of a mean over the seeds
            for k in "ch":
                self.assertLess(abs(values[k].mean() - target), 3 * sigma + 1e-12 * target,
                                f"{name} {k}: {values[k]} vs {target}")
            self.assertLess(abs(values["c"].mean() - values["h"].mean()),
                            3 * np.sqrt(2) * sigma + 1e-12 * target,
                            f"{name}: coupled {values['c']} vs hybrid {values['h']}")

    def tearDown(self):
        ph.clearDict()


if __name__ == "__main__":
    unittest.main()
