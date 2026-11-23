"""Adaptive time steps with a refined level: level 0 steps at its own stable dt, level 1 substeps
at its own (damn::amr::TimeRefinement: SAMRAI's split). A uniform 1D hybrid plasma, one refined
box (level 1 not refinable: no regrid interval rounding):

  the level's first substep (new): level 0's dt / time_ratio (4), then the rest split into equal
  substeps no larger than level 1's stable dt; later steps: equal substeps no larger than it

The whistler bound (core/numerics/stable_dt/whistler.hpp) of each level is computed here from its
mass density and the uniform B, at the start of each step: the counts are checked where the ratio
of the bounds (as measured: each level's density minimum) keeps 1 / ratio and 0.75 / ratio away
from integers (dx 25: 0.45, 3 substeps, the first step 1 + 2; dx 2: 0.28, 4, then 1 + 3).
needs py_etc and the 1D hybrid module built (./mkn.py --dim 1), then from the plamn root:
    PYTHONPATH=.:$PHARE/pyphare:tst/plamn/python python3 -m phlop.run.test_cases -i tst
"""

import math
import shutil
import unittest
from pathlib import Path

import numpy as np

import plamn
import plamn.backend.phare as ph

B = (0.3, 0.4, 1.2)
CFL_WAVE = 0.5
TIME_RATIO = 4
STEPS = 4


def whistler_bound(cpp, ln, dx):
    """level ln's own stable dt (its density's minimum, uniform B)"""
    rho = np.min(np.asarray(cpp.level_values("protons_density", ln)))
    inv_dx = 1.0 / dx
    sqrt_rho = math.sqrt(rho)
    v_a = math.sqrt(sum(b * b for b in B)) / sqrt_rho
    x = math.pi * inv_dx / (2.0 * sqrt_rho)
    return CFL_WAVE / (v_a * (math.sqrt(1.0 + x * x) + x) / dx)


def split(remaining, bound):
    """SAMRAI's: the fewest equal substeps no larger than bound"""
    n = int(remaining / bound)
    if remaining - bound * n > math.sqrt(np.finfo(float).eps) * remaining:
        n += 1
    return n


def simulator(dl, **kwargs):
    ph.global_vars.sim = None
    ph.clearDict()
    sim = ph.Simulation(
        **kwargs,
        time_step=dict(mode="adaptive", cfl_wave=CFL_WAVE),
        final_time=1e6,
        cells=(64,),
        dl=(dl,),
        refinement_boxes={"L0": {"B0": [(16,), (47,)]}},
        hyper_resistivity=0.001,
    )

    def const(v):
        return lambda x: v + 0 * x

    ph.MaxwellianFluidModel(
        bx=const(B[0]),
        by=const(B[1]),
        bz=const(B[2]),
        protons={
            "mass": 1,
            "charge": 1,
            "density": const(1.0),
            "vbulkx": const(0.0),
            "vbulky": const(0.0),
            "vbulkz": const(0.0),
            "vthx": const(0.3),
            "vthy": const(0.3),
            "vthz": const(0.3),
            "nbr_part_per_cell": 100,
            "init": {"seed": 1337},
        },
    )
    ph.ElectronModel(closure="isothermal", Te=0.12)
    return ph.Simulator(sim, log_to_file=False).setup().initialize()


class AdaptiveLevelsTest(unittest.TestCase):
    def setUp(self):
        ph.startMPI()
        self.s = None

    def tearDown(self):
        if self.s is not None:
            self.s.reset()
        ph.global_vars.sim = None

    def check(self, dl):
        self.s = simulator(dl)
        cpp = self.s.cpp_sim
        self.assertEqual(cpp.n_levels(), 2)
        for step in range(STEPS):
            b0 = whistler_bound(cpp, 0, dl)  # at the step's start, from its state
            b1 = whistler_bound(cpp, 1, dl / 2)
            self.s.advance()
            dt0 = cpp.timeStep()  # the step's
            # level 0: its own bound only (to round off while B is the initial, uniform one: the
            #  C++ may contract to FMAs)
            if step == 0:
                self.assertAlmostEqual(dt0, b0, delta=1e-14 * b0)
            else:
                self.assertAlmostEqual(dt0, b0, delta=1e-2 * b0)
            dts = cpp.substep_dts(1)
            self.assertAlmostEqual(sum(dts), dt0, delta=1e-14 * dt0)
            if step == 0:  # new: the first substep dt0 / time_ratio, then the rest
                self.assertEqual(dts[0], dt0 / TIME_RATIO)
                expected = 1 + split(dt0 - dts[0], min(dt0, b1))
            else:
                expected = split(dt0, min(dt0, b1))
            self.assertEqual(len(dts), expected, f"step {step}: {dts}, bound {b1}")
            for dt in dts[1:] if step == 0 else dts:  # the rest: no larger than the bound
                self.assertLessEqual(dt, min(dt0, b1) * 1.01)
        return cpp

    def test_three_substeps(self):  # level 1's bound 0.45 of level 0's
        self.check(dl=25.0)

    def test_four_substeps(self):  # 0.28
        self.check(dl=2.0)

    def test_a_restart_continues_the_split(self):
        """after a restart, level 1 is not new: its substeps those of the run that kept going (a
        fresh split would take dt0 / 4 first: [0.25, 0.375, 0.375] x dt0, not thirds)"""
        dl, steps, at = 25.0, 4, 2
        # per rank count: the 1 and 3 rank cases run at the same time (phlop)
        name = f"{self.id().split('.')[-1]}_np{plamn.etc.mpi_size()}"
        out = Path("phare_outputs/plamn_test_adaptive_levels") / name
        shutil.rmtree(out, ignore_errors=True)

        def substeps(s, n):
            cpp, dts = s.cpp_sim, []
            for _ in range(n):
                s.advance()
                dts.append(cpp.substep_dts(1))
            return dts

        self.s = simulator(dl)
        continued = substeps(self.s, steps)
        self.s.reset()
        self.s = simulator(dl)
        substeps(self.s, at)
        t_restart = self.s.cpp_sim.currentTime()
        self.s.reset()

        options = dict(dir=str(out), mode="overwrite", timestamps=[t_restart])
        self.s = simulator(dl, restart_options=options)
        substeps(self.s, at)  # writes the restart at t_restart
        self.s.reset()
        self.s = simulator(
            dl, restart_options=dict(dir=str(out), mode="overwrite", restart_time=t_restart)
        )
        self.assertEqual(self.s.cpp_sim.currentTime(), t_restart)
        restarted = substeps(self.s, steps - at)
        self.assertEqual(restarted, continued[at:])
        self.assertEqual(len(restarted[0]), 3)


if __name__ == "__main__":
    unittest.main()
