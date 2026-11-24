"""A coupled build (MHD and hybrid models, ./mkn.py --coupled) runs either model alone: every level
MHD (an input without ions or Ohm options) or every level hybrid (an input without MHD's finite
volume options). A uniform plasma at rest stays uniform.

needs py_etc and the coupled 1D module (./mkn.py --coupled --dim 1), then from the plamn root:
    PYTHONPATH=.:$PHARE/pyphare:tst/plamn/python python3 -m phlop.run.test_cases -i tst
"""

import unittest

import numpy as np

import plamn.backend.phare as ph

CELLS = 32
DX = 0.2
DT = 0.001
STEPS = 5
# the coupled module: interp_order and an MHD time stepper (plamn.cpp_lib)
COUPLED = dict(
    interp_order=1,
    reconstruction="Linear",
    limiter="VanLeer",
    riemann="Rusanov",
    mhd_timestepper="TVDRK2",
    gamma=5.0 / 3.0,
    eta=0.0,
    nu=0.0,
    hall=False,
)


def const(v):
    return lambda x: v + 0.0 * x


def run(model):
    ph.global_vars.sim = None
    ph.clearDict()
    ph.Simulation(
        time_step=DT,
        time_step_nbr=STEPS,
        cells=(CELLS,),
        dl=(DX,),
        hyper_resistivity=0.0,
        resistivity=0.0,
        model_options=[model],
        **COUPLED,
    )
    if model == "MHDModel":
        ph.MHDModel(
            density=const(1.0), vx=const(0.0), vy=const(0.0), vz=const(0.0),
            bx=const(1.0), by=const(0.0), bz=const(0.0), p=const(1.0),
        )
    else:
        ph.MaxwellianFluidModel(
            bx=const(1.0), by=const(0.0), bz=const(0.0),
            protons={
                "charge": 1, "mass": 1, "nbr_part_per_cell": 100, "init": {"seed": 1},
                "density": const(1.0),
                **{f"vbulk{c}": const(0.0) for c in "xyz"},
                **{f"vth{c}": const(0.5) for c in "xyz"},
            },
        )
        ph.ElectronModel(closure="isothermal", Te=0.5)
    simulator = ph.Simulator(ph.global_vars.sim, log_to_file=False).setup().initialize()
    import plamn

    module = plamn.cpp_lib(ph.global_vars.sim).__name__
    assert module.startswith("plamn.plamn_cpp_1_1_") and module.endswith(
        "_Linear_VanLeer_Rusanov_false"
    ), module  # the coupled build
    for _ in range(STEPS):
        simulator.advance()
    cpp = simulator.cpp_sim
    out = {"time": cpp.currentTime()}
    if model == "MHDModel":
        out["rho"] = np.asarray(cpp.level_values("rho", 0))
    else:
        out["count"] = cpp.particle_count(0)
    out["bx"] = np.asarray(cpp.level_values("B_x" if model == "MHDModel" else "EM_B_x", 0))
    simulator.reset()
    return out


class CoupledBuildsTest(unittest.TestCase):
    def test_mhd_alone(self):
        out = run("MHDModel")
        self.assertAlmostEqual(out["time"], STEPS * DT)
        np.testing.assert_array_equal(out["rho"], 1.0)
        np.testing.assert_array_equal(out["bx"], 1.0)

    def test_hybrid_alone(self):
        out = run("HybridModel")
        self.assertAlmostEqual(out["time"], STEPS * DT)
        self.assertEqual(out["count"], CELLS * 100)
        np.testing.assert_allclose(out["bx"], 1.0)


if __name__ == "__main__":
    unittest.main()
