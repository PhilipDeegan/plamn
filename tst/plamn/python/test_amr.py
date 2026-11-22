"""plamn.amr: chopping, partitioning and decomposition metrics without a simulation (plamn_etc's
chop_<dim>, partition_<dim>, metrics_<dim> on damn's data free hierarchy code).

    python3 -m unittest tst.plamn.python.test_amr
"""

import math
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

    def test_spill_sheds_to_the_same_node_first(self):
        # 80 cells, level 1 over the coarse cells 0..9 (fine 0..19, patches of 4 fine cells),
        # 4 ranks 2 per node, blocks of 10: level 1 is all under block 0 (work 10 + 2 x 20 = 50
        # of 120: rank 0 alone), so columns put all of level 1 on rank 0. spill 0.5: cap
        # 1.5 x 20 / 4 = 7.5 (x ppc) for patches of 4: rank 0 sheds to rank 1 (its node) first,
        # then to the other node, down to 8 4 4 4
        sim = simulation_1d(cells=80, largest_patch_size=4)
        refined = {1: [([0], [19])]}
        columns = Chopper(sim, ranks=4, ranks_per_node=2, partition="columns", block=10)
        h0 = columns.chop(refined)
        self.assertEqual({rank for _, _, rank in h0.levels[1]}, {0})

        columns.spill = 0.5
        h = columns.chop(refined)
        level1 = np.zeros(4)
        for (lo,), (up,), rank in h.levels[1]:
            level1[rank] += (up - lo + 1) * 10  # cells x ppc
        self.assertEqual(level1.tolist(), [80.0, 40.0, 40.0, 40.0])
        self.assertLess(h.critical_path(), h0.critical_path())

        # the first patch rank 0 sheds goes to rank 1, on its node, before the other node
        first = [rank for _, _, rank in h.levels[1] if rank != 0]
        self.assertIn(1, first)

    def test_comm_cost_weighs_the_exchanged_cells_by_distance(self):
        sim = simulation_1d()
        h = Chopper(sim, ranks=2, partition="columns", block=10).chop(
            {1: [([20], [59])]}
        )
        cost = {"Self": 0.0, "Local": 0.0, "Numa": 1.0, "Node": 1.0, "Socket": 1.0, "Remote": 1.0}
        crossing = sum(
            m[key][d]
            for m in h.metrics()
            for key in ("ghost_cells", "coarse_cells")
            for d in ("Numa", "Node", "Remote")
        )
        self.assertGreater(crossing, 0)
        self.assertEqual(h.comm_cost(cost), crossing)
        self.assertEqual(h.comm_cost({**cost, "Node": 0.0}), 0.0)  # one node: all Node

    def test_keep_moves_nothing_for_the_same_levels(self):
        sim = simulation_1d(cells=80, largest_patch_size=8)
        refined = {1: [([20], [79])]}
        before = Chopper(
            sim, ranks=4, ranks_per_node=2, partition="columns", block=10
        ).chop(refined)
        keep = Chopper(sim, ranks=4, ranks_per_node=2, partition="keep")
        after = keep.chop(refined, previous=before)
        self.assertEqual(after.levels, before.levels)
        for m in after.migration(before):
            for key in ("moved", "refined"):
                self.assertEqual(
                    m[key]["Node"] + m[key]["Remote"], 0, (m["level"], key)
                )
        self.assertEqual(after.migration_cost(before), 0.0)

    def test_keep_moves_less_than_a_fresh_partition(self):
        # a regrid: level 1 shifted by 4 coarse cells (8 fine)
        sim = simulation_1d(cells=80, largest_patch_size=8)
        before = Chopper(
            sim, ranks=4, ranks_per_node=2, partition="columns", block=10
        ).chop({1: [([20], [79])]})
        shifted = {1: [([28], [87])]}
        fresh = Chopper(
            sim, ranks=4, ranks_per_node=2, partition="columns", block=10
        ).chop(shifted)
        kept = Chopper(sim, ranks=4, ranks_per_node=2, partition="keep").chop(
            shifted, previous=before
        )
        self.assertLessEqual(kept.migration_cost(before), fresh.migration_cost(before))
        # the new fine cells (80..87) refine from the old level 0, all of level 1 accounted
        cells = sum(
            sum(m[k].values())
            for m in kept.migration(before)[1:]
            for k in ("moved", "refined")
        )
        self.assertEqual(cells, 60)

    def test_decompose_weighs_from_python_or_its_cost_model(self):
        # damn::amr::decompose through plamn_etc: a python weigh(ln, boxes) and the C++ cost
        # model over cells times ppc give the same levels; finest puts each coarse patch under
        # the fine level with the owner of the fine cells above it (no spill)
        costs = {"particle": 2.0, "cell": 1.0, "ghost_cell": 0.5, "patch": 3.0, "reach": 2}
        ppc = [10.0, 5.0]

        def weigh(ln, boxes):
            out = []
            for lo, up in boxes:
                n = np.asarray(up) - lo + 1
                cells = float(np.prod(n))
                ghost = float(np.prod(n + 4)) - cells
                out.append(math.fma(2.0, cells * ppc[ln], math.fma(1.0, cells, math.fma(0.5, ghost, 3.0))))
            return out

        args = ([0, 0], [63, 63], [True, True], 2, [], 0,
                [[([0, 0], [63, 63])], [([32, 32], [95, 95])]], [0, 0, 1, 1])
        for partition in ("levels", "finest", "aligned"):
            a = plamn_etc.decompose_2(*args, patch_size=[8, 8], partition=partition,
                                      weigh=weigh)
            b = plamn_etc.decompose_2(*args, patch_size=[8, 8], partition=partition,
                                      costs=costs, ppc=ppc)
            self.assertEqual(a, b, partition)
        fine = plamn_etc.decompose_2(*args, patch_size=[8, 8], partition="finest",
                                     costs=costs, ppc=ppc)
        for lo, up, rk in fine[0]:
            owners = {}
            for flo, fup, frk in fine[1]:
                ilo = np.maximum(np.asarray(lo) * 2, flo)
                iup = np.minimum((np.asarray(up) + 1) * 2 - 1, fup)
                if (ilo <= iup).all():
                    owners[frk] = owners.get(frk, 0) + np.prod(iup - ilo + 1)
            if owners:
                self.assertEqual(rk, max(sorted(owners), key=lambda k: owners[k]))
        with self.assertRaises(RuntimeError):
            plamn_etc.decompose_2(*args, patch_size=[8, 8], partition="nope", ppc=ppc)


if __name__ == "__main__":
    unittest.main()
