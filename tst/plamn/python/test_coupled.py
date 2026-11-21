"""MHD under hybrid (AMR/max_mhd_level): a uniform plasma at rest, level 0 MHD, level 1 hybrid
over part of it. Nothing should happen: level 1's particles (spawned from MHD's state) keep the
uniform density as they cross its border. Against high ppc reference samples of it and of the same
with hybrid on both levels (etc/bench/coupling/ref/uniform.json, made by
etc/bench/coupling/uniform.py): statistical, over seeds, the edge nodes and the interior apart.
Coupled runs are plamn's own (no pyphare has coupling): plamn/__init__.py's coupled input.

needs py_etc, the 1D hybrid module and the coupled one built
(./mkn.py --dim 1, ./mkn.py --coupled --dim 1), then from the plamn root:
    PYTHONPATH=.:$PHARE/pyphare:tst/plamn/python python3 -m phlop.run.test_cases -i tst
"""

import json
import unittest
from pathlib import Path

import numpy as np

import plamn.backend.phare as ph

CELLS = 64
DX = 0.2
DT = 0.001
STEPS = 300
PPC = 50
TE = 0.5  # isothermal electrons: Pi = P - Te n = 0.5, vth = sqrt(0.5)
SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
EDGE = 4  # level 1's nodes at each end, next to the MHD level
SAMPLES = (100, 200, 300)  # the steps level 1 is sampled at (etc/bench/coupling/uniform.py)
FINE_CELLS = 40  # level 1 over level 0's cells 20 to 39
# high ppc samples of both setups (etc/bench/coupling/uniform.py)
REF = Path(__file__).resolve().parents[3] / "etc" / "bench" / "coupling" / "ref" / "uniform.json"


def const(v):
    return lambda x: v + 0.0 * x


def run(coupled, seed, ppc=PPC):
    """level 1 at the end and at SAMPLES: its particle count, density per node"""
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
    species = {"charge": 1, "mass": 1, "nbr_part_per_cell": ppc, "init": {"seed": seed}}
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
    samples = {}
    for step in range(1, STEPS + 1):
        simulator.advance()
        if step in SAMPLES:
            samples[step] = {
                "count": cpp.particle_count(1),
                "density": np.asarray(cpp.level_values("protons_density", 1)),
            }
    out = {
        "samples": samples,
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
    def test_hybrid_over_mhd_matches_the_reference(self):
        ref = json.loads(REF.read_text())
        coupled = [run(True, s) for s in SEEDS]

        for c in coupled:
            self.assertAlmostEqual(c["time"], STEPS * DT)
            # B stays the uniform field; MHD is not fed back yet (no hybrid -> MHD sync)
            np.testing.assert_array_equal(c["bx"], 1.0)
            np.testing.assert_array_equal(c["rho0"], 1.0)
            self.assertTrue(np.isfinite(c["density"]).all())

        for step in SAMPLES:
            href, cref = ref["hybrid"][str(step)], ref["coupled"][str(step)]
            truth = np.asarray(href["density"])  # hybrid on both levels, high ppc
            # the reference itself: coupled and hybrid within its (small) noise
            self.assertLess(np.sqrt(np.mean((np.asarray(cref["density"]) - truth) ** 2)), 0.005)
            self.assertLess(abs(cref["count"] - href["count"]) / href["count"], 1e-4)

            # this build's coupled runs: the count, the edge nodes (next to MHD) and the interior
            #  against the reference, within the seeds' spread
            n = len(SEEDS)
            per_seed = {
                "count": [c["samples"][step]["count"] for c in coupled],
                "edges": [np.r_[c["samples"][step]["density"][:EDGE],
                                c["samples"][step]["density"][-EDGE:]].mean() for c in coupled],
                "interior": [c["samples"][step]["density"][EDGE:-EDGE].mean() for c in coupled],
            }
            expected = {
                "count": href["count"] * PPC / ref["ppc"],
                "edges": np.r_[truth[:EDGE], truth[-EDGE:]].mean(),
                "interior": truth[EDGE:-EDGE].mean(),
            }
            for name, values in per_seed.items():
                values = np.asarray(values, dtype=float)
                sigma = values.std(ddof=1) / np.sqrt(n)
                self.assertLess(abs(values.mean() - expected[name]), 3 * sigma,
                                f"step {step} {name}: {values} vs {expected[name]}")

    def tearDown(self):
        ph.clearDict()


if __name__ == "__main__":
    unittest.main()
