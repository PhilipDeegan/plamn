"""a serial 1 level MHD run of a 1D shock tube (Brio-Wu like states, small, periodic) through
pyphare on plamn's MHD only module

needs py_etc and the MHD module built (./mkn.py --mhd), then from the plamn root:
    PYTHONPATH=.:$PHARE/pyphare:tst/plamn/python python3 -m phlop.run.test_cases -i tst
"""

import os
import tempfile
import unittest

import numpy as np

import plamn.backend.phare as ph

CELLS = 200
STEPS = 40
DT = 0.1
GAMMA = 5.0 / 3.0


def config(largest_patch_size, diag_dir):
    ph.global_vars.sim = None
    ph.clearDict()
    sim = ph.Simulation(
        time_step=DT,
        time_step_nbr=STEPS,
        cells=(CELLS,),
        dl=(1.0,),
        largest_patch_size=largest_patch_size,
        hyper_resistivity=0.0,
        resistivity=0.0,
        diag_options={
            "format": "phareh5",
            "options": {"dir": diag_dir, "mode": "overwrite"},
        },
        eta=0.0,
        nu=0.0,
        gamma=GAMMA,
        reconstruction="Linear",
        limiter="VanLeer",
        riemann="Rusanov",
        mhd_timestepper="TVDRK2",
        hall=False,
        model_options=["MHDModel"],
    )

    # left and right states of the shock, a tube each way (periodic)
    def left(x):
        return (x > CELLS / 4) & (x < 3 * CELLS / 4)

    ph.MHDModel(
        density=lambda x: np.where(left(x), 1.0, 0.125),
        vx=lambda x: 0.0 * x,
        vy=lambda x: 0.0 * x,
        vz=lambda x: 0.0 * x,
        bx=lambda x: 0.75 + 0.0 * x,
        by=lambda x: np.where(left(x), 1.0, -1.0),
        bz=lambda x: 0.0 * x,
        p=lambda x: np.where(left(x), 1.0, 0.1),
    )
    ph.MHDDiagnostics(quantity="rho", write_timestamps=[0.0, STEPS * DT])
    return sim


def run(largest_patch_size, diag_dir):
    sim = config(largest_patch_size, diag_dir)
    simulator = ph.Simulator(sim, log_to_file=False).setup().initialize()
    cpp = simulator.cpp_sim
    mass = np.sum(cpp.all_domain_values("rho"))
    for _ in range(STEPS):
        simulator.advance()
    rho = np.asarray(cpp.level_values("rho"))
    finite = all(
        np.isfinite(cpp.level_values(q)).all() for q in ("rho", "Etot", "rhoV_x", "B_y")
    )
    return {"rho": rho, "mass": (mass, np.sum(rho)), "finite": finite, "time": cpp.currentTime()}


class MHDRunTest(unittest.TestCase):
    def test_shock_tube(self):
        with tempfile.TemporaryDirectory() as one_dir, tempfile.TemporaryDirectory() as four_dir:
            one = run(CELLS, one_dir)
            four = run(CELLS // 4, four_dir)
            self.assertTrue(one["finite"])
            self.assertAlmostEqual(one["time"], STEPS * DT)
            # periodic: the mass stays
            self.assertAlmostEqual(one["mass"][1], one["mass"][0], delta=1e-12 * one["mass"][0])
            # the waves moved (the profile is no longer the two states), the density positive
            self.assertGreater(np.unique(np.round(one["rho"], 6)).size, 2)
            self.assertGreater(one["rho"].min(), 0)
            # the patches change nothing but rounding
            np.testing.assert_allclose(one["rho"], four["rho"], rtol=1e-14, atol=1e-14)
            self.assertTrue(os.path.exists(os.path.join(one_dir, "mhd_rho.h5")))

    def tearDown(self):
        ph.clearDict()


if __name__ == "__main__":
    unittest.main()
