"""plamn.amr: chopping, partitioning and decomposition metrics without a simulation (plamn_etc's
chop_<dim>, partition_<dim>, metrics_<dim> on damn's data free hierarchy code).

    python3 -m unittest tst.plamn.python.test_amr
"""

import unittest

import numpy as np

import plamn.backend.phare as ph
from plamn import plamn_etc
from plamn.amr import Chopper


def simulation(cells=(40, 20), largest_patch_size=10, ppc=10):
    """a 2D input, never run: what Chopper reads of it"""
    ph.global_vars.sim = None
    sim = ph.Simulation(
        cells=cells,
        dl=(0.2, 0.2),
        time_step=0.001,
        time_step_nbr=1,
        largest_patch_size=largest_patch_size,
        hyper_resistivity=0.001,
    )

    def const(v):
        return lambda x, y: v + 0 * x

    ph.MaxwellianFluidModel(
        bx=const(1.0),
        by=const(0.0),
        bz=const(0.0),
        protons={
            "charge": 1,
            "density": const(1.0),
            "vbulkx": const(0.0),
            "vbulky": const(0.0),
            "vbulkz": const(0.0),
            "vthx": const(0.1),
            "vthy": const(0.1),
            "vthz": const(0.1),
            "nbr_part_per_cell": ppc,
        },
    )
    ph.ElectronModel(closure="isothermal", Te=0.1)
    return sim


def simulation_1d(cells=40, largest_patch_size=10, ppc=10):
    """a 1D input, never run: one population, all its density a background above the cut-off,
    so every cell of a level weighs the same (ppc)"""
    ph.global_vars.sim = None
    sim = ph.Simulation(
        cells=cells,
        dl=0.2,
        time_step=0.001,
        time_step_nbr=1,
        largest_patch_size=largest_patch_size,
        hyper_resistivity=0.001,
    )

    def const(v):
        return lambda x: v + 0 * x

    ph.MaxwellianFluidModel(
        protons={
            "charge": 1,
            "density": const(0.0),
            "density_background": 1.0,
            "vthx": const(0.1),
            "vthy": const(0.1),
            "vthz": const(0.1),
            "nbr_part_per_cell": ppc,
        },
    )
    ph.ElectronModel(closure="isothermal", Te=0.1)
    return sim


class AmrTest(unittest.TestCase):
    def tearDown(self):
        ph.global_vars.sim = None

    def test_chop_modes(self):
        # 122 cells by 60, in units of 2: SAMRAI's 42 40 40, chop's 60 60 2
        even = plamn_etc.chop_1([100], [221], [60], 2, "even")
        self.assertEqual(even, [([100], [141]), ([142], [181]), ([182], [221])])
        chop = plamn_etc.chop_1([100], [221], [60], 1, "chop")
        self.assertEqual([up[0] - lo[0] + 1 for lo, up in chop], [60, 60, 2])
        with self.assertRaises(RuntimeError):
            plamn_etc.chop_1([0], [9], [5], 1, "aligned")

    def test_metrics_count_by_distance(self):
        # the C++ test's hierarchy (tst/damn/amr/test_metrics.cpp): 40 periodic cells, 4 patches
        # on ranks 0 0 1 1 of one node, level 1 the coarse cells 10..19 on rank 0
        levels = [
            [([0], [9], 0), ([10], [19], 0), ([20], [29], 1), ([30], [39], 1)],
            [([20], [39], 0)],
        ]
        m = plamn_etc.metrics_1([0], [39], [True], 2, levels, [0, 0], 2, [])
        self.assertEqual(m[0]["ghost_cells"]["Local"], 8)
        self.assertEqual(m[0]["ghost_cells"]["Node"], 8)
        self.assertEqual(m[1]["coarse_cells"]["Local"], 11)
        self.assertEqual(m[1]["coarse_cells"]["Node"], 1)
        self.assertAlmostEqual(m[1]["imbalance"], 2.0)

    def test_chopper(self):
        sim = simulation()
        refined = {1: [([20, 10], [59, 29])]}  # level 1 boxes (fine index space)
        h = Chopper(sim, ranks=4, ranks_per_node=2).chop(refined)
        self.assertEqual(len(h.levels), 2)
        self.assertEqual(len(h.levels[0]), 8)  # 40 x 20 cut in 10 x 10
        self.assertEqual(len(h.levels[1]), 8)  # 40 x 20 fine cells cut in 10 x 10
        self.assertEqual(sorted({r for _, _, r in h.levels[0]}), [0, 1, 2, 3])
        m = h.metrics()
        for lm in m:
            self.assertEqual(sum(lm["weight"]), lm["cells"] * 10)  # cells x ppc
        self.assertGreater(m[0]["ghost_cells"]["Remote"], 0)  # two nodes
        self.assertEqual(m[0]["coarse_cells"]["Local"], 0)

        # a work map: level 0 weights from it, the busy half spread over more ranks' patches
        work = np.ones(sim.cells)
        work[:20, :] = 3.0
        hw = Chopper(sim, ranks=4, ranks_per_node=2).chop(refined, work)
        self.assertEqual(sum(hw.weights[0]), work.sum())

        # pharesee's view: one patch a box, a "rank" field
        ph_h = h.hierarchy()
        self.assertEqual(len(ph_h.level(1).patches), 8)

    def test_columns_own_every_level_over_their_cells(self):
        # 40 periodic cells, level 1 over the coarse cells 10..29 (fine 20..59), 2 ranks, blocks
        # of 10: block works (ppc x cells, a level 1 cell twice for its 2 substeps) 10 50 50 10
        # times ppc: each rank 60, blocks 0 1 on one, 2 3 on the other
        sim = simulation_1d()
        refined = {1: [([20], [59])]}
        h = Chopper(sim, ranks=2, partition="columns", block=10).chop(refined)
        self.assertEqual(h.total_work_imbalance(), 1.0)

        owner = {}  # level 0 cell -> rank
        for (lo,), (up,), rank in h.levels[0]:
            for c in range(lo, up + 1):
                owner[c] = rank
        self.assertEqual(len({owner[c] for c in range(20)}), 1)
        self.assertNotEqual(owner[0], owner[39])
        for (lo,), (up,), rank in h.levels[
            1
        ]:  # each level 1 patch over its owner's cells
            self.assertEqual({owner[c // 2] for c in range(lo, up + 1)}, {rank})


if __name__ == "__main__":
    unittest.main()
