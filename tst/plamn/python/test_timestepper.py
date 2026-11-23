"""plamn's Timestepper (inc/plamn/simulator/timestepper.hpp) through pyphare's time_step options:

  constant  the time after n steps is start + n * dt, from 0; a restart with the same dt resumes
            those steps (its times the run's that kept going), one with another dt (a job
            restarted at a smaller time step) counts from the restart's time
  adaptive  each step's dt is level 0's whistler bound (core/numerics/stable_dt/whistler.hpp)
            before it, at most what is left to final_time, which the run ends on; a refined
            level substeps at its own (test_adaptive_levels.py)

The bound is checked against the same formula over the level's mass density (one population of
mass 1: its density) and a uniform B, so B projected onto the density nodes is B itself: exactly
while B is the initial one (the first dt and the first step's), within 1% after (B evolves).

    mpirun -n 3 python3 -m unittest tst.plamn.python.test_timestepper   (phlop: --tags mpi)
"""

import math
import shutil
import unittest
from pathlib import Path

import numpy as np

import plamn
import plamn.backend.phare as ph

dl = 0.3
cfl_wave = 0.5
B = (0.3, 0.4, 1.2)


def setup_model():
    """one population of mass 1, uniform B"""

    def const(v):
        return lambda x: v + 0 * x

    ph.MaxwellianFluidModel(
        bx=const(B[0]),
        by=const(B[1]),
        bz=const(B[2]),
        protons={
            "mass": 1,
            "charge": 1,
            "density": lambda x: 1.0 + 0.5 * np.sin(2 * np.pi * x / (200 * dl)),
            "vbulkx": const(0.0),
            "vbulky": const(0.0),
            "vbulkz": const(0.0),
            "vthx": const(0.3),
            "vthy": const(0.3),
            "vthz": const(0.3),
            "nbr_part_per_cell": 50,
            "init": {"seed": 1337},
        },
    )
    ph.ElectronModel(closure="isothermal", Te=0.12)


def simulation(**kwargs):
    """a Simulator, set up and initialized"""
    ph.global_vars.sim = None
    ph.clearDict()
    args = dict(cells=200, dl=dl, hyper_resistivity=0.001)
    args.update(kwargs)
    sim = ph.Simulation(**args)
    setup_model()
    return ph.Simulator(sim, log_to_file=False).setup().initialize()


def whistler_dt(cpp, ln):
    """a level's bound, as a level 0 dt"""
    dx = dl / 2**ln
    rho = np.min(np.asarray(cpp.level_values("protons_density", ln)))
    B2 = B[0] * B[0] + B[1] * B[1] + B[2] * B[2]
    # as whistler_speed (stable_dt/whistler.hpp), operation for operation: the dt is compared
    #  exactly
    inv_dx = 1.0 / dx
    sqrt_rho = math.sqrt(rho)
    vA = math.sqrt(B2) / sqrt_rho
    x = math.pi * inv_dx / (2.0 * sqrt_rho)
    inv_dt = vA * (math.sqrt(1.0 + x * x) + x) / dx
    return cfl_wave / inv_dt * 4**ln


def stable_dt(cpp):
    """level 0's own bound: each finer level steps at its own (damn::amr::TimeRefinement's
    substeps, from its stable dt)"""
    return whistler_dt(cpp, 0)


class TimestepperTest(unittest.TestCase):
    def setUp(self):
        ph.startMPI()
        name = f"{self.id().split('.')[-1]}_np{plamn.etc.mpi_size()}"
        self.dir = str(Path("phare_outputs/plamn_test_timestepper") / name)
        self.simulators = []

    def tearDown(self):
        for s in self.simulators:
            s.reset()
        plamn.etc.mpi_barrier()
        if plamn.etc.mpi_rank() == 0:
            shutil.rmtree(self.dir, ignore_errors=True)

    def keep(self, simulator):
        self.simulators.append(simulator)
        return simulator

    def test_constant_times_are_start_plus_n_dt(self):
        dt = 0.001
        s = self.keep(simulation(time_step=dt, time_step_nbr=10))
        for n in range(1, 11):
            s.advance()
            self.assertEqual(s.cpp_sim.currentTime(), n * dt)
            self.assertEqual(s.cpp_sim.timeStep(), dt)

    def test_a_restart_resumes_the_steps(self):
        """the same dt after a restart: the times those of the run that kept going, n * dt from
        its start (at 7 steps of 0.01: 0.07 + k * 0.01 != (7 + k) * 0.01 for k 2, 4, 5)"""
        dt, at, steps = 0.01, 7, 13
        first = self.keep(
            simulation(
                time_step=dt,
                time_step_nbr=steps,
                restart_options=dict(
                    dir=self.dir, mode="overwrite", timestamps=[at * dt]
                ),
            )
        )
        for _ in range(at):
            first.advance()
        second = self.keep(
            simulation(
                time_step=dt,
                time_step_nbr=steps,
                restart_options=dict(
                    dir=self.dir, mode="overwrite", restart_time=at * dt
                ),
            )
        )
        cpp = second.cpp_sim
        self.assertEqual(cpp.currentTime(), at * dt)
        for n in range(at + 1, steps + 1):
            second.advance()
            self.assertEqual(cpp.currentTime(), n * dt, f"step {n}")

    def test_restart_with_a_smaller_dt(self):
        dt, at = 0.001, 4
        first = self.keep(
            simulation(
                time_step=dt,
                time_step_nbr=10,
                restart_options=dict(
                    dir=self.dir, mode="overwrite", timestamps=[at * dt]
                ),
            )
        )
        for _ in range(at):
            first.advance()
        restart_time = at * dt

        small = dt / 2
        second = self.keep(
            simulation(
                time_step=small,
                time_step_nbr=6,
                restart_options=dict(
                    dir=self.dir, mode="overwrite", restart_time=restart_time
                ),
            )
        )
        cpp = second.cpp_sim
        self.assertEqual(cpp.currentTime(), restart_time)
        self.assertEqual(cpp.timeStep(), small)
        np.testing.assert_array_equal(
            cpp.level_values("EM_B_y", 0), first.cpp_sim.level_values("EM_B_y", 0)
        )
        for n in range(1, 7):
            second.advance()
            self.assertEqual(cpp.currentTime(), restart_time + n * small)
        self.assertEqual(cpp.currentTime(), cpp.endTime())

    def check_adaptive(self, final_time, **kwargs):
        s = self.keep(
            simulation(
                time_step=dict(mode="adaptive", cfl_wave=cfl_wave),
                final_time=final_time,
                **kwargs,
            )
        )
        cpp = s.cpp_sim
        self.assertEqual(cpp.endTime(), final_time)
        self.assertEqual(
            cpp.timeStep(), stable_dt(cpp)
        )  # the first, for the first dump

        steps, total, compensation = 0, 0.0, 0.0
        while cpp.currentTime() < final_time:
            left = final_time - cpp.currentTime()
            expected = min(stable_dt(cpp), left)
            s.advance()
            steps += 1
            dt = cpp.timeStep()
            if steps == 1:
                self.assertEqual(dt, expected)
            else:
                self.assertLessEqual(dt, left)
                self.assertAlmostEqual(
                    dt, expected, delta=1e-2 * expected, msg=f"step {steps}"
                )
            y = dt - compensation  # Kahan, as the Timestepper
            t = total + y
            compensation = (t - total) - y
            total = t
            self.assertEqual(cpp.currentTime(), total, f"step {steps}")
            self.assertLess(steps, 1000)
        self.assertAlmostEqual(cpp.currentTime(), final_time, delta=1e-12)
        return s, steps

    def test_adaptive_dt_is_the_whistler_bound(self):
        s, steps = self.check_adaptive(final_time=0.05)
        self.assertGreater(steps, 2)

    def test_adaptive_dt_with_a_refined_level(self):
        s, steps = self.check_adaptive(
            final_time=0.02, refinement_boxes={"L0": {"B0": [[80], [119]]}}
        )
        self.assertEqual(s.cpp_sim.n_levels(), 2)
        self.assertGreater(steps, 1)


if __name__ == "__main__":
    unittest.main()
