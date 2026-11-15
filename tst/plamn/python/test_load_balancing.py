"""PHARE's tests/simulator/test_load_balancing.py set up on plamn, one level: two populations,
one without particles where its density is below the cutoff. The patches are
partitioned at initialization by the particles they will hold (damn::amr::partition), so every
rank starts within one patch of the others. Meaningful on more than one rank:

    mpirun -n 3 python3 -m tst.plamn.python.test_load_balancing   (phlop: --tags mpi)
"""

import unittest

import numpy as np

import pyplamn


import pyphare.pharein as ph
from pyphare.simulator.simulator import Simulator

cells = (100, 100)
dl = (0.2, 0.2)
patch = (10, 10)
ppc_high, ppc_low = 90, 120
density_cut_off = 1e-16  # pyphare's default


def density_low(x, y):
    """PHARE's: no particles in the rows around y = 10 (the cell centres 9.9 and 10.1)"""
    rho = y.copy()
    rho[:] = 0.1
    rho[np.where(np.isclose(y, 10, atol=0.1))] = 0
    return rho


def density_empty_middle(x, y):
    """no particles in the middle half of y (periodic, as pyphare checks): a split by cells alone
    leaves the ranks holding it with far fewer particles"""
    rho = y.copy()
    rho[:] = 0.1
    rho[(y > 5) & (y < 15)] = 0
    return rho


def config(protons_density):
    ph.global_vars.sim = None
    ph.clearDict()
    sim = ph.Simulation(
        time_step_nbr=1,
        time_step=0.001,
        cells=cells,
        dl=dl,
        largest_patch_size=patch,
        hyper_resistivity=0.001,
        resistivity=0.001,
        strict=True,
    )

    def density_high(x, y):
        rho = y.copy()
        rho[:] = 1
        return rho

    def const(v):
        return lambda x, y: v + 0.0 * x

    def pop(density, ppc, seed):
        return {
            "charge": 1,
            "density": density,
            "vbulkx": const(0.0),
            "vbulky": const(0.0),
            "vbulkz": const(0.0),
            "vthx": const(0.3),
            "vthy": const(0.3),
            "vthz": const(0.3),
            "nbr_part_per_cell": ppc,
            "init": {"seed": seed},
        }

    ph.MaxwellianFluidModel(
        bx=const(1.0),
        by=const(0.0),
        bz=const(0.0),
        brotons=pop(density_high, ppc_high, 12334),
        protons=pop(protons_density, ppc_low, 43210),
    )
    ph.ElectronModel(closure="isothermal", Te=0.0)
    return sim


def expected_particles(protons_density):
    """the cells at or above the cutoff, at the cell centres the loader uses"""
    x = (np.arange(cells[0]) + 0.5) * dl[0]
    y = (np.arange(cells[1]) + 0.5) * dl[1]
    X, Y = np.meshgrid(x, y, indexing="ij")
    low = np.count_nonzero(protons_density(X.ravel(), Y.ravel()) >= density_cut_off)
    return ppc_high * cells[0] * cells[1] + ppc_low * low


class LoadBalancingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pyplamn.patch()

    def init(self, protons_density):
        self.simulator = (
            Simulator(config(protons_density), log_to_file=False).setup().initialize()
        )
        return self.simulator.cpp_sim

    def tearDown(self):
        if hasattr(self, "simulator"):
            self.simulator.reset()

    def test_rows_without_particles(self):
        cpp = self.init(density_low)
        total = cpp.total_particle_count()
        self.assertEqual(total, expected_particles(density_low))
        self.assertLess(total, (ppc_high + ppc_low) * cells[0] * cells[1])

    def test_ranks_start_balanced(self):
        cpp = self.init(density_empty_middle)
        per_rank = cpp.particle_counts_per_rank()
        self.assertEqual(len(per_rank), cpp.ranks())
        self.assertEqual(sum(per_rank), expected_particles(density_empty_middle))
        # every rank within one patch's weight of the share (its particles, and a particle's
        #  worth per cell): two ranks within two
        heaviest = np.prod(patch) * (ppc_high + ppc_low + 1)
        self.assertLessEqual(max(per_rank) - min(per_rank), 2 * heaviest)


if __name__ == "__main__":
    unittest.main()
