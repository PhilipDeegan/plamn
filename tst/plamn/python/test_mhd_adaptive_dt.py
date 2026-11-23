"""MHD adaptive time steps (core/numerics/stable_dt/mhd.hpp, PHARE's SolverMHD::computeStableDt):
a uniform 1D state, ideal and Hall MHD, the dt against the formula (PHARE's
tests/simulator/test_adaptive_stable_dt.py, in 1D: the MHD modules built locally are)
needs py_etc and the 1D MHD modules built (./mkn.py --mhd --dim 1, ideal and Hall), then from the
plamn root:
    PYTHONPATH=.:$PHARE/pyphare:tst/plamn/python python3 -m phlop.run.test_cases -i tst
"""

import math
import unittest

import plamn.backend.phare as ph

V = (0.3, -0.2, 0.1)
B = (0.5, 0.4, 0.3)
P = 0.2
GAMMA = 5.0 / 3.0
CFL_WAVE = 0.5
CELLS = 20


def fast_magnetosonic(gamma, rho, b_dir, b2, p):
    sound = math.sqrt(gamma * p / rho)
    alfven_dir = math.sqrt(b_dir * b_dir / rho)
    alfven = math.sqrt(b2 / rho)
    c02, ca2, cadir2 = sound * sound, alfven * alfven, alfven_dir * alfven_dir
    disc = (c02 + ca2) * (c02 + ca2) - 4.0 * c02 * cadir2
    return math.sqrt((c02 + ca2) * 0.5 + math.sqrt(max(disc, 0.0)) * 0.5)


def whistler(inv_dx, rho, b2):
    v_a = math.sqrt(b2) / math.sqrt(rho)
    x = math.pi * inv_dx / (2.0 * math.sqrt(rho))
    return v_a * (math.sqrt(1.0 + x * x) + x)


def expected_dt(hall, dx, rho0):
    """1D: the x direction only"""
    b2 = sum(b * b for b in B)
    cf = fast_magnetosonic(GAMMA, rho0, B[0], b2, P)
    hall_term = whistler(1.0 / dx, rho0, b2) - math.sqrt(b2 / rho0) if hall else 0.0
    return CFL_WAVE / ((abs(V[0]) + cf + hall_term) / dx)


def simulator(hall, dx, rho0):
    ph.global_vars.sim = None
    ph.clearDict()
    sim = ph.Simulation(
        time_step=dict(mode="adaptive", cfl_wave=CFL_WAVE),
        final_time=100.0,
        cells=(CELLS,),
        dl=(dx,),
        hyper_resistivity=0.0,
        resistivity=0.0,
        eta=0.0,
        nu=0.0,
        gamma=GAMMA,
        reconstruction="Linear",
        limiter="VanLeer",
        riemann="Rusanov",
        mhd_timestepper="TVDRK2",
        hall=hall,
        model_options=["MHDModel"],
    )

    def const(v):
        return lambda x: v + 0.0 * x

    ph.MHDModel(
        density=const(rho0),
        vx=const(V[0]),
        vy=const(V[1]),
        vz=const(V[2]),
        bx=const(B[0]),
        by=const(B[1]),
        bz=const(B[2]),
        p=const(P),
    )
    return ph.Simulator(sim, log_to_file=False).setup().initialize()


class MHDAdaptiveDtTest(unittest.TestCase):
    def setUp(self):
        ph.startMPI()
        self.s = None

    def tearDown(self):
        if self.s is not None:
            self.s.reset()
        ph.global_vars.sim = None

    def check(self, hall, dx, rho0):
        self.s = simulator(hall, dx, rho0)
        cpp = self.s.cpp_sim
        expected = expected_dt(hall, dx, rho0)
        self.assertAlmostEqual(cpp.timeStep() / expected, 1.0, delta=1e-12)
        for _ in range(3):  # a uniform state stays uniform: the same dt each step
            self.s.advance()
            self.assertAlmostEqual(cpp.timeStep() / expected, 1.0, delta=1e-12)

    def test_ideal(self):
        self.check(hall=False, dx=0.3, rho0=1.0)

    def test_hall_coarse(self):  # the whistler term near the Alfven speed
        self.check(hall=True, dx=10.0, rho0=1.0)

    def test_hall_fine(self):  # Hall dominated
        self.check(hall=True, dx=0.05, rho0=1.0)

    def test_hall_light(self):
        self.check(hall=True, dx=0.05, rho0=0.25)


if __name__ == "__main__":
    unittest.main()
