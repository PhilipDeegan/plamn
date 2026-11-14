"""PHARE_SCOPE_TIMING=1: phlop's scope timers (DAMN_LOG_SCOPE, PHAREHUB/PHARE#1585), one file per
rank, .phare/timings/rank.<r>.txt, written when the simulator goes; and Simulator.summary(), as
PHARE's. On any number of ranks (phlop: --tags mpi runs it on 3):

    mpirun -n 3 python3 -m tst.plamn.python.test_scope_timing

read with phlop (python3 -m phlop.timing.scope_timer times|perf|variance 'rank.*.txt')
"""

import os
import re
import unittest
from pathlib import Path

import pyplamn

pyplamn.patch()

import alfven1d  # noqa: E402
import pyphare.pharein as ph  # noqa: E402
from pyphare.simulator.simulator import Simulator  # noqa: E402

CELLS = 100
PATCH = 10
STEPS = 3
DIR = Path("plamn_test_scope_timing")  # the run's working directory: .phare/timings in it

SCOPES = [
    "Simulator::initialize",
    "Simulator::advance",
    "Simulator::predictor1_",
    "Simulator::faraday",
    "Simulator::ohm",
    "Simulator::move_ions::push",
    "Simulator::fill_ion_borders_",
    "damn::amr::make_neighbours",
]


def config():
    ph.global_vars.sim = None
    ph.clearDict()
    sim = alfven1d.config(cells=CELLS, time_step_nbr=STEPS, final_time=STEPS * 0.01)
    sim.largest_patch_size = (PATCH,)
    return sim


class ScopeTimingTest(unittest.TestCase):
    def test_a_file_per_rank_and_summary(self):
        cwd = Path.cwd()
        DIR.mkdir(exist_ok=True)
        os.chdir(DIR)
        os.environ["PHARE_SCOPE_TIMING"] = "1"
        try:
            simulator = Simulator(config(), log_to_file=False).setup().initialize()
            cpp = simulator.cpp_sim
            for _ in range(STEPS):
                simulator.advance()
            summary = cpp.summary()  # collective
            rank, ranks = cpp.rank(), cpp.ranks()
            del cpp  # the last reference goes with reset: the C++ simulator writes the files
            simulator.reset()
        finally:
            del os.environ["PHARE_SCOPE_TIMING"]
            os.chdir(cwd)

        content = (DIR / f".phare/timings/rank.{rank}.txt").read_bytes()
        for scope in SCOPES:
            self.assertIn(scope.encode(), content, scope)
        copy = b"damn::mpi::copy<" if ranks > 1 else b"damn::amr::copy<"
        self.assertIn(copy + b"fill_ghosts>", content)

        if rank != 0:
            self.assertEqual(summary, "")
            return
        # periodic 1D, patches of 10 cells: every ghost cell is another patch's (or an image)
        match = re.match(r"lvl:0 \((\d+),(\d+),(\d+),(\d+)\) parts\((\d+)\), \ntot:(\d+)", summary)
        self.assertIsNotNone(match, summary)
        patches, cells, patch_ghosts, level_ghosts, parts, total = map(int, match.groups())
        self.assertEqual((patches, cells, level_ghosts), (CELLS // PATCH, CELLS, 0))
        self.assertGreater(patch_ghosts, 0)
        self.assertEqual(parts, total)
        self.assertGreater(parts, 0)


if __name__ == "__main__":
    unittest.main()
