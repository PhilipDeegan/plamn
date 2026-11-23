"""a 1 level hybrid run of pyphare's 1D Alfven wave input, on plamn (one or several ranks)

needs the py_etc and py_sim modules built (mkn build -p py_etc / -p py_sim), then from
the plamn root:
    PYTHONPATH=.:$PHARE/pyphare:tst/plamn/python python3 -m phlop.run.test_cases -i tst
"""

import unittest

import alfven1d
import numpy as np

import plamn
import plamn.backend.phare as ph

CELLS = 100
STEPS = 150


def run(largest_patch_size):
    ph.global_vars.sim = None
    ph.clearDict()
    sim = alfven1d.config(cells=CELLS, time_step_nbr=STEPS, final_time=STEPS * 0.01)
    sim.largest_patch_size = (largest_patch_size,)
    simulator = ph.Simulator(sim, log_to_file=False).setup().initialize()
    cpp = simulator.cpp_sim

    k = 2 * np.pi / CELLS
    x = np.arange(CELLS) + 0.5  # By is dual in x

    def fit():  # By = a cos(k x + phase)
        by = np.asarray(cpp.all_domain_values("EM_B_y"))
        c = (by * np.cos(k * x)).sum() * 2 / CELLS
        s = (by * np.sin(k * x)).sum() * 2 / CELLS
        return np.hypot(c, s), np.arctan2(-s, c)

    result = {
        "particles": [cpp.total_particle_count()],
        "amplitude": [],
        "time": [],
        "phase": [],
    }
    for step in range(STEPS + 1):
        if step % (STEPS // 5) == 0:
            amplitude, phase = fit()
            result["amplitude"].append(amplitude)
            result["phase"].append(phase)
            result["time"].append(cpp.currentTime())
        if step < STEPS:
            simulator.advance()
    result["particles"].append(cpp.total_particle_count())
    result["patches"] = cpp.all_patch_count()
    result["finite"] = all(
        np.isfinite(cpp.all_domain_values(f"EM_{q}_{c}")).all()
        for q in "BE"
        for c in "xyz"
    )
    # By = a cos(k (x + v t)): travels in -x for this polarisation (V in phase with B)
    result["speed"] = np.polyfit(result["time"], np.unwrap(result["phase"]), 1)[0] / k
    return result


class HybridAlfven1DTest(unittest.TestCase):
    def check(self, result, patches):
        # on several ranks the patches are cut where a rank's share ends inside one: at least
        #  one a rank, at most a cut per rank boundary more
        ranks = plamn.etc.mpi_size()
        if ranks == 1:
            self.assertEqual(result["patches"], patches)
        else:
            self.assertGreaterEqual(result["patches"], max(patches, ranks))
            self.assertLessEqual(result["patches"], patches + ranks - 1)
        self.assertTrue(result["finite"])
        self.assertEqual(result["particles"][0], result["particles"][-1])
        np.testing.assert_allclose(result["amplitude"], 0.01, rtol=0.05)
        # parallel waves: v/vA = sqrt(1 + (kd/2)^2) + kd/2 ~ 1.03 here, d = 1
        self.assertGreater(result["speed"], 0.95)
        self.assertLess(result["speed"], 1.12)

    def test_one_patch(self):
        self.check(run(CELLS), 1)

    def test_five_patches(self):
        self.check(run(CELLS // 5), 5)

    def tearDown(self):
        ph.clearDict()


if __name__ == "__main__":
    unittest.main()
