"""python3 -m plamn --watch N --save DIR: a run drawn every N level 0 steps (the initial state
too), a PNG each, for an MHD and a hybrid input (1D)

needs py_etc, the 1D hybrid module and the 1D MHD module (./mkn.py --mhd), from the plamn root:
    PYTHONPATH=.:$PHARE/pyphare:tst/plamn/python python3 -m phlop.run.test_cases -i tst
"""

import tempfile
import unittest
from pathlib import Path

import plamn.backend.phare as ph
from plamn import watch


class WatchTest(unittest.TestCase):
    def check(self, module, steps, every, quantities=None):
        with tempfile.TemporaryDirectory() as d:
            argv = [module, "--watch", str(every), "--save", d]
            if quantities:
                argv += ["--quantities", *quantities]
            self.assertEqual(watch.main(argv), 0)
            pngs = sorted(p.name for p in Path(d).glob("*.png"))
        self.assertEqual(pngs, [f"{s:06d}.png" for s in range(0, steps + 1, every)])

    def test_mhd(self):
        self.check("mhd_tube", steps=20, every=10)

    def test_hybrid(self):
        self.check("alfven1d", steps=10, every=5, quantities=["EM_B_y", "EM_B_z"])

    def tearDown(self):
        ph.clearDict()


if __name__ == "__main__":
    unittest.main()
