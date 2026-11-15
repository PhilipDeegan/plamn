"""PHARE's tests/simulator/test_restarts.py (PHAREHUB/PHARE#1410's), in memory: a simulation
writes a restart at step 4 and keeps running; a second one, restart_time at step 4, loads it.
Both live in the same process (plamn has no process wide state), so they are compared directly,
without diagnostics: at the restart time and after every step after it, the same levels and
patches, B, E and the moments on every level, and every patch's particles, bit for bit.

The restart is written in memory order (no sort): the restarted run continues identically on
the same ranks. Run with any number of ranks (the same for both), e.g.

    mpirun -n 3 python3 -m unittest tst.plamn.python.test_restarts   (phlop: --tags mpi)
"""

import shutil
import unittest
from pathlib import Path

import numpy as np

import pyplamn

pyplamn.patch()

import pyphare.pharein as ph  # noqa: E402
from pyphare.simulator.simulator import Simulator, startMPI  # noqa: E402

timestep = 0.001
restart_step = 4


def setup_model(ppc=100):
    """PHARE's test_restarts model: protons and alpha, a tanh by"""
    sim = ph.global_vars.sim
    L = sim.simulation_domain()[0]

    def S(x, x0, l):
        return 0.5 * (1 + np.tanh((x - x0) / l))

    def by(x):
        v1, v2 = -1, 1.0
        return v1 + (v2 - v1) * (S(x, L * 0.25, 1) - S(x, L * 0.75, 1))

    def b2(x):
        return by(x) ** 2 + 0.5**2

    def T(x):
        return 1 - b2(x) * 0.5

    def const(v):
        return lambda x: v + 0 * x

    def vvv(vx):
        return {
            "vbulkx": const(vx),
            "vbulky": const(0.0),
            "vbulkz": const(0.0),
            "vthx": T,
            "vthy": T,
            "vthz": T,
        }

    ph.MaxwellianFluidModel(
        bx=const(0.0),
        by=by,
        bz=const(0.5),
        protons={
            "mass": 1,
            "charge": 1,
            "density": const(1.0),
            **vvv(2.0),
            "nbr_part_per_cell": ppc,
            "init": {"seed": 1337},
        },
        alpha={
            "mass": 4.0,
            "charge": 1,
            "density": const(1.0),
            **vvv(3.0),
            "nbr_part_per_cell": ppc,
            "init": {"seed": 2334},
        },
    )
    ph.ElectronModel(closure="isothermal", Te=0.12)


def simulation(restart_dir, restart_time=None, **kwargs):
    ph.global_vars.sim = None
    ph.clearDict()
    restart_options = dict(dir=restart_dir, mode="overwrite")
    if restart_time is None:
        restart_options["timestamps"] = [timestep * restart_step]
    else:
        restart_options["restart_time"] = restart_time
    args = dict(
        time_step_nbr=10,
        time_step=timestep,
        cells=200,
        dl=0.3,
        hyper_resistivity=0.001,
        restart_options=restart_options,
    )
    args.update(kwargs)
    sim = ph.Simulation(**args)
    setup_model()
    return Simulator(sim, log_to_file=False).setup().initialize()


def state(simulator):
    """everything the restarted run must match: per level B, E, moments, particles"""
    cpp = simulator.cpp_sim
    pops = ["protons", "alpha"]
    out = {"summary": cpp.summary(), "time": cpp.currentTime()}
    dw = simulator.data_wrangler()
    for ln in range(cpp.n_levels()):
        for q in ["EM_B", "EM_E"]:
            for c in "xyz":
                out[f"{q}_{c}@{ln}"] = np.asarray(cpp.level_values(f"{q}_{c}", ln))
        for pop in pops:
            for q in ["density", "charge_density"]:
                out[f"{pop}_{q}@{ln}"] = np.asarray(cpp.level_values(f"{pop}_{q}", ln))
            for c in "xyz":
                out[f"{pop}_flux_{c}@{ln}"] = np.asarray(
                    cpp.level_values(f"{pop}_flux_{c}", ln)
                )
            for pd in dw.getPatchLevel(ln).getParticles(pop):  # this rank's patches
                key = f"{pop}@{ln}:{tuple(pd.lower)}"
                for name, view in pd.data.components().items():
                    out[f"{key}:{name}"] = np.array(view)
    return out


class RestartsTest(unittest.TestCase):
    def setUp(self):
        startMPI()
        # the same on every rank of a run, its own per test and rank count: phlop runs a test
        #  on 1 and on 3 ranks at the same time
        name = f"{self.id().split('.')[-1]}_np{pyplamn.etc.mpi_size()}"
        self.dir = str(Path("phare_outputs/plamn_test_restarts") / name)
        self.simulators = []

    def tearDown(self):
        for s in self.simulators:
            s.reset()
        pyplamn.etc.mpi_barrier()
        if self.rank0():
            shutil.rmtree(self.dir, ignore_errors=True)

    def rank0(self):
        return pyplamn.etc.mpi_rank() == 0

    def assertSameState(self, a, b, when):
        self.assertEqual(sorted(a), sorted(b), when)
        for key in a:
            if isinstance(a[key], np.ndarray):
                np.testing.assert_array_equal(a[key], b[key], err_msg=f"{key} {when}")
            else:
                self.assertEqual(a[key], b[key], f"{key} {when}")

    def run_both(self, steps, **kwargs):
        first = simulation(self.dir, **kwargs)
        self.simulators.append(first)
        for _ in range(restart_step):
            first.advance()  # the restart is written after step 4
        restart_time = timestep * restart_step
        written = Path(pyplamn.etc.restart_path_for_time(self.dir, restart_time))
        self.assertTrue((written / "restart.h5").exists(), written)

        second = simulation(self.dir, restart_time=restart_time, **kwargs)
        self.simulators.append(second)
        self.assertEqual(second.cpp_sim.currentTime(), first.cpp_sim.currentTime())
        self.assertSameState(state(first), state(second), "at the restart")
        for step in range(restart_step, steps):
            first.advance()
            second.advance()
            self.assertSameState(state(first), state(second), f"after step {step + 1}")
        return first

    def test_one_level(self):
        self.run_both(steps=8, refinement_boxes={})

    def test_refinement_boxes(self):
        first = self.run_both(steps=5, refinement_boxes={"L0": {"B0": [[10], [19]]}})
        self.assertEqual(first.cpp_sim.n_levels(), 2)

    def test_tagging(self):
        first = self.run_both(steps=10, refinement="tagging", max_nbr_levels=3)
        self.assertGreater(first.cpp_sim.n_levels(), 1)

    def test_directory_names(self):
        """10 digits each side of the point: steps far below pyphare's 5 decimals apart"""
        etc = pyplamn.etc
        self.assertEqual(
            Path(etc.restart_path_for_time("d", 0.004)).name, "0000000000.0040000000"
        )
        self.assertNotEqual(
            etc.restart_path_for_time("d", 1e-6), etc.restart_path_for_time("d", 2e-6)
        )
        import pyphare.pharein.restarts as restarts

        self.assertEqual(restarts.format_time(0.004), "0000000000.0040000000")


if __name__ == "__main__":
    unittest.main()
