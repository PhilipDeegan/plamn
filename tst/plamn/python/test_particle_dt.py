"""The hybrid adaptive dt's particle bound (core/numerics/stable_dt/particles.hpp): no domain
particle moves more than cfl_particles of the allowed cell jump in a step,
dt <= cfl_particles * cell_jump * dx / max |v|. A cold beam (vth ~ 0) at vbulk_x in a weak B: every
particle at about vbulk_x, the whistler bound far larger, so the beam sets the dt; with
cfl_particles 0 the whistler bound alone, and the beam breaks the cell jump limit. plamn's time_step keys pass through pyphare's check
(plamn/__init__.py _patch_time_step_keys).
needs py_etc and the 1D hybrid module built (./mkn.py --dim 1), then from the plamn root:
    PYTHONPATH=.:$PHARE/pyphare:tst/plamn/python python3 -m phlop.run.test_cases -i tst
"""

import math
import unittest

import numpy as np

import plamn
import plamn.backend.phare as ph

DL = 0.2
V_BEAM = 5.0
B = (0.01, 0.0, 0.0)
CFL_WAVE = 0.5


def whistler_bound(cpp):
    rho = np.min(np.asarray(cpp.level_values("protons_density", 0)))
    sqrt_rho = math.sqrt(rho)
    v_a = math.sqrt(sum(b * b for b in B)) / sqrt_rho
    x = math.pi / (DL * 2.0 * sqrt_rho)
    return CFL_WAVE / (v_a * (math.sqrt(1.0 + x * x) + x) / DL)


def simulator(**time_step):
    ph.global_vars.sim = None
    ph.clearDict()
    sim = ph.Simulation(
        time_step=dict(mode="adaptive", cfl_wave=CFL_WAVE, **time_step),
        final_time=10.0,
        cells=(100,),
        dl=(DL,),
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
            "vbulkx": const(V_BEAM),
            "vbulky": const(0.0),
            "vbulkz": const(0.0),
            "vthx": const(1e-6),
            "vthy": const(1e-6),
            "vthz": const(1e-6),
            "nbr_part_per_cell": 20,
            "init": {"seed": 1337},
        },
    )
    ph.ElectronModel(closure="isothermal", Te=0.0)
    return ph.Simulator(sim, log_to_file=False).setup().initialize()


class ParticleDtTest(unittest.TestCase):
    def setUp(self):
        ph.startMPI()
        self.s = None

    def tearDown(self):
        if self.s is not None:
            self.s.reset()
        ph.global_vars.sim = None

    def test_a_fast_beam_sets_the_dt(self):
        self.s = simulator()
        cpp = self.s.cpp_sim
        cell_jump = 2  # PLAMN_PARTICLE_ALLOWED_CELL_JUMP's default
        expected = 0.5 * cell_jump * DL / V_BEAM  # cfl_particles' default 0.5
        self.assertLess(expected, whistler_bound(cpp))
        for _ in range(3):
            self.s.advance()
            self.assertAlmostEqual(cpp.timeStep(), expected, delta=1e-4 * expected)

    def test_cfl_particles_scales_it(self):
        self.s = simulator(cfl_particles=0.25)
        self.s.advance()
        expected = 0.25 * 2 * DL / V_BEAM
        self.assertAlmostEqual(self.s.cpp_sim.timeStep(), expected, delta=1e-4 * expected)

    def test_cfl_particles_0_breaks_the_cell_jump_limit(self):
        """the whistler bound alone: a dt far above the beam's, a particle moves further than the
        allowed cell jump and the push throws (what the particle bound is for)"""
        self.s = simulator(cfl_particles=0)
        cpp = self.s.cpp_sim
        expected = whistler_bound(cpp)
        self.assertAlmostEqual(cpp.timeStep(), expected, delta=1e-12 * expected)
        self.assertGreater(cpp.timeStep(), 2 * DL / V_BEAM)  # more than a cell jump
        if plamn.etc.mpi_size() > 1:  # a rank throwing alone would leave others in a collective
            return
        with self.assertRaises(RuntimeError):  # the C++'s (pyphare's advance exits the process)
            cpp.advance()

    def test_keys_are_checked(self):
        with self.assertRaises(ValueError):
            ph.global_vars.sim = None
            ph.Simulation(
                time_step=dict(mode="constant", value=0.01, cfl_particles=0.5),
                final_time=1.0,
                cells=(10,),
                dl=(DL,),
            )
        with self.assertRaises(ValueError):
            ph.global_vars.sim = None
            ph.Simulation(
                time_step=dict(mode="adaptive", cfl_wave=0.5, cfl_particles=-1),
                final_time=1.0,
                cells=(10,),
                dl=(DL,),
            )
        ph.global_vars.sim = None


if __name__ == "__main__":
    unittest.main()
