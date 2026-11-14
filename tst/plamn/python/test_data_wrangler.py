"""live, zero-copy access to plamn's simulation data through pyphare's hierarchy_from_sim
(PHAREHUB/PHARE#1400): needs a pyphare with the DataWrangler of PHAREHUB/PHARE#1171

    PYTHONPATH=python:$PHARE_WRANGLER/pyphare:tst/plamn/python \
        python3 -m unittest tst/plamn/python/test_data_wrangler.py
"""

import unittest

import numpy as np

import pyplamn

pyplamn.patch()

import alfven1d  # noqa: E402
import pyphare.pharein as ph  # noqa: E402
from pyphare.simulator.simulator import Simulator  # noqa: E402

try:
    from pyphare.pharesee.hierarchy.fromsim import hierarchy_from_sim  # noqa: E402
    from pyphare.pharesee.hierarchy.patchdata import LiveParticleData  # noqa: E402, F401
except ImportError:  # a pyphare without PHARE#1171
    hierarchy_from_sim = None

CELLS = 100
PATCH = 20


def domain(hier, qty):
    """qty on the domain nodes of every patch, flattened in C order, in the level's patch
    order: as the simulator's domain_values"""
    patches = hier.level(0).patches
    return np.concatenate([np.asarray(p.patch_datas[qty][p.box]).ravel() for p in patches])


@unittest.skipIf(hierarchy_from_sim is None, "pyphare has no live DataWrangler (PHARE#1171)")
class DataWranglerTest(unittest.TestCase):
    def setUp(self):
        ph.global_vars.sim = None
        ph.clearDict()
        sim = alfven1d.config(cells=CELLS, time_step_nbr=10, final_time=0.1)
        sim.largest_patch_size = (PATCH,)
        self.simulator = Simulator(sim, log_to_file=False).setup().initialize()
        self.cpp = self.simulator.cpp_sim

    def tearDown(self):
        self.simulator.reset()

    def test_fields_match_the_simulation(self):
        hier = hierarchy_from_sim(self.simulator, qty="EM_B_y")
        self.assertEqual(len(hier.level(0).patches), CELLS // PATCH)
        np.testing.assert_array_equal(
            domain(hier, "EM_B_y"), self.cpp.domain_values("EM_B_y")
        )
        for qty in ["EM_B_x", "EM_E_y", "density"]:  # every getter used by pharesee works
            hier = hierarchy_from_sim(self.simulator, qty=qty)
            self.assertTrue(np.isfinite(domain(hier, qty)).all(), qty)

    def test_views_are_live(self):
        dw = self.simulator.data_wrangler()
        views = [pd.data for pd in dw.getPatchLevel(0).getB("y")]

        before = np.asarray(self.cpp.domain_values("EM_B_y"))
        for v in views:  # write through python: C++ sees it, no copy
            v *= 2
        np.testing.assert_array_equal(self.cpp.domain_values("EM_B_y"), 2 * before)
        for v in views:
            v /= 2

        self.simulator.advance()  # C++ writes: the same views see it
        after = hierarchy_from_sim(self.simulator, qty="EM_B_y")
        np.testing.assert_array_equal(domain(after, "EM_B_y"), self.cpp.domain_values("EM_B_y"))
        self.assertFalse(np.array_equal(self.cpp.domain_values("EM_B_y"), before))
        patch0 = dw.getPatchLevel(0).getB("y")[0]
        self.assertEqual(views[0].__array_interface__["data"][0],
                         patch0.data.__array_interface__["data"][0])

    def test_particles(self):
        hier = hierarchy_from_sim(self.simulator, qty="particles", pop="protons")
        patches = hier.level(0).patches
        total = sum(p.patch_datas["particles"].size() for p in patches)
        self.assertEqual(total, self.cpp.particle_count())

        live = patches[0].patch_datas["particles"].dataset._cpp
        p = live[0]
        box = patches[0].box
        self.assertTrue(box.lower[0] <= p.iCell[0] <= box.upper[0])
        self.assertTrue(0 <= p.delta[0] < 1)
        self.assertEqual(len(p.v), 3)


CELLS_2D = 20
PATCH_2D = 10
PPC_2D = 100


def config_2d():
    """PHARE's tests/simulator populate_simulation(2, 1) model, on one level"""
    ph.global_vars.sim = None
    ph.clearDict()
    sim = ph.Simulation(
        interp_order=1,
        largest_patch_size=[PATCH_2D] * 2,
        time_step=0.001,
        time_step_nbr=5,
        cells=[CELLS_2D] * 2,
        dl=[1.0 / CELLS_2D] * 2,
        refined_particle_nbr=4,
        diag_options={},
        strict=True,
    )
    L = sim.simulation_domain()

    def density(x, y):
        return 1.0 + 0.2 * np.cos(2 * np.pi * x / L[0]) * np.cos(2 * np.pi * y / L[1])

    def by(x, y):
        return 0.1 * np.cos(2 * np.pi * x / L[0])

    def bz(x, y):
        return 0.1 * np.sin(2 * np.pi * y / L[1])

    def const(v):
        return lambda x, y: v + 0.0 * x

    ph.MaxwellianFluidModel(
        bx=const(1.0),
        by=by,
        bz=bz,
        protons={
            "charge": 1,
            "density": density,
            "vbulkx": const(0.0),
            "vbulky": const(0.0),
            "vbulkz": const(0.0),
            "vthx": const(0.1),
            "vthy": const(0.1),
            "vthz": const(0.1),
            "nbr_part_per_cell": PPC_2D,
            "init": {"seed": 1},
        },
    )
    ph.ElectronModel(closure="isothermal", Te=0.12)
    return sim


@unittest.skipIf(hierarchy_from_sim is None, "pyphare has no live DataWrangler (PHARE#1171)")
class DataWrangler2DTest(unittest.TestCase):
    """the checks of PHARE's test_data_wrangler.py test_hybrid (2D, interp 1), on one level"""

    def setUp(self):
        self.simulator = Simulator(config_2d(), log_to_file=False).setup().initialize()
        self.cpp = self.simulator.cpp_sim

    def tearDown(self):
        self.simulator.reset()

    def test_fields(self):
        hier = None
        for qty in ["EM_B_x", "EM_B_y", "density"]:
            hier = hierarchy_from_sim(self.simulator, qty=qty, hier=hier, sync=True)
        self.assertEqual(len(hier.level(0).patches), (CELLS_2D // PATCH_2D) ** 2)
        for qty, name in [("EM_B_y", "EM_B_y"), ("EM_B_x", "EM_B_x")]:
            np.testing.assert_array_equal(domain(hier, qty), self.cpp.domain_values(name))
        self.assertTrue(np.isfinite(domain(hier, "density")).all())

    def test_particles(self):
        hier = hierarchy_from_sim(self.simulator, qty="particles", pop="protons")
        total = sum(p.patch_datas["particles"].size() for p in hier.level(0).patches)
        self.assertEqual(total, self.cpp.particle_count())
        self.assertGreaterEqual(total, 0.95 * PPC_2D * CELLS_2D**2)

    def test_views_are_live_across_advances(self):
        dw = self.simulator.data_wrangler()
        views = [pd.data for pd in dw.getPatchLevel(0).getE("y")]
        particles = self.cpp.particle_count()
        for _ in range(5):
            self.simulator.advance()
        hier = hierarchy_from_sim(self.simulator, qty="EM_E_y")
        np.testing.assert_array_equal(domain(hier, "EM_E_y"), self.cpp.domain_values("EM_E_y"))
        self.assertTrue(all(np.isfinite(v).all() for v in views))
        self.assertEqual(self.cpp.particle_count(), particles)  # periodic


if __name__ == "__main__":
    unittest.main()
