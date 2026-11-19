"""python3 -m plamn --watch N --save DIR: a run drawn every N level 0 steps (the initial state
too), a PNG each, for an MHD and a hybrid input (1D); the levels onto the finest mesh

needs py_etc, the 1D hybrid module and the 1D MHD module (./mkn.py --mhd), from the plamn root:
    PYTHONPATH=.:$PHARE/pyphare:tst/plamn/python python3 -m phlop.run.test_cases -i tst
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import plamn.backend.phare as ph
from plamn import watch


class WatchTest(unittest.TestCase):
    def check(self, module, steps, every, quantities=None, interp=None):
        with tempfile.TemporaryDirectory() as d:
            argv = [module, "--watch", str(every), "--save", d]
            if quantities:
                argv += ["--quantities", *quantities]
            if interp:
                argv += ["--interp", interp]
            self.assertEqual(watch.main(argv), 0)
            pngs = sorted(p.name for p in Path(d).glob("*.png"))
        self.assertEqual(pngs, [f"{s:06d}.png" for s in range(0, steps + 1, every)])

    def test_mhd(self):
        self.check("mhd_tube", steps=20, every=10)

    def test_hybrid(self):
        self.check("alfven1d", steps=10, every=5, quantities=["EM_B_y", "EM_B_z"])

    def test_view3d_missing_says_how_to_build(self):
        with self.assertRaises(FileNotFoundError) as e:
            watch.main(["mhd_tube", "--watch", "1", "--view3d", "/no/view3d"])
        self.assertIn("res/build_3d_view.py", str(e.exception))

    def test_levels_as_they_are(self):
        self.check("alfven1d", steps=10, every=10, quantities=["EM_B_y"], interp="none")


def two_levels(fn, cells, box, dual=True):
    """level 0 over cells (dx 1), level 1 over the coarse cells box (lo, hi per direction,
    inclusive): their cell centres (dual) or nodes, and fn there (level 1: fn + 1, to tell
    them apart)"""
    s = 0.5 if dual else 0.0
    coarse = np.stack(np.meshgrid(*[np.arange(n) + s for n in cells], indexing="ij"), -1)
    fine = np.stack(
        np.meshgrid(*[np.arange(2 * lo, 2 * hi + 2 + (not dual)) / 2 + s / 2 for lo, hi in box],
                    indexing="ij"),
        -1,
    )
    coarse, fine = coarse.reshape(-1, len(cells)), fine.reshape(-1, len(cells))
    return [(coarse, fn(coarse)), (fine, fn(fine) + 1)]


class FinestTest(unittest.TestCase):
    """the levels on the finest mesh (watch.finest): the finer level wherever it is"""

    def test_flat_finest_drops_the_covered_points(self):
        for dual in (True, False):
            levels = two_levels(lambda x: x[:, 0], [10], [(3, 5)], dual)
            x, v = watch.flat_finest(levels, [1.0])
            coarse = x[v == x[:, 0]][:, 0]
            # coarse cells 3..5 covered (nodes 3..6 too: the fine level has them)
            self.assertTrue(((coarse < 3) | (coarse > 6)).all(), (dual, coarse))
            self.assertEqual((v != x[:, 0]).sum(), len(levels[1][0]))

    def test_flat_finest_levels(self):
        """with_level: each kept point's level (view3d sizes its balls by it)"""
        levels = two_levels(lambda x: x[:, 0], [10], [(3, 5)])
        x, v, lv = watch.flat_finest(levels, [1.0], with_level=True)
        np.testing.assert_array_equal(lv == 1, v != x[:, 0])  # level 1: fn + 1
        self.assertEqual((lv == 1).sum(), len(levels[1][0]))

    def test_arrows_block_means(self):
        """view3d's arrows: B over blocks of W cells, at the blocks' centres (a PLMV record)"""
        import struct

        class Cpp:
            def domain_box(self):
                return [[0, 0, 0], [19, 19, 19]]

        x = np.stack(np.meshgrid(*[np.arange(20) + 0.5] * 3, indexing="ij"), -1).reshape(-1, 3)
        components = [[(x, x[:, 0])], [(x, 2 * x[:, 1])], [(x, 0 * x[:, 2])]]
        view = watch.View3D.__new__(watch.View3D)  # no viewer started
        view.cpp, view.dx, view.arrows = Cpp(), np.ones(3), 10
        record = view.arrows_record(components)
        self.assertEqual(record[:4], b"PLMV")
        (n,) = struct.unpack("<Q", record[4:12])
        self.assertEqual(n, 8)
        data = np.frombuffer(record[12:], "<f4")
        xyz, vec, half = data[: 3 * n].reshape(n, 3), data[3 * n : 6 * n].reshape(n, 3), data[6 * n :]
        np.testing.assert_allclose(vec[:, 0], xyz[:, 0])
        np.testing.assert_allclose(vec[:, 1], 2 * xyz[:, 1])
        np.testing.assert_allclose(vec[:, 2], 0)
        np.testing.assert_allclose(half, 5)
        self.assertEqual(sorted(set(xyz[:, 0])), [5.0, 15.0])

    def test_1d(self):
        levels = two_levels(lambda x: x[:, 0], [10], [(3, 5)])
        for interp in ("nearest", "linear"):
            (x,), v = watch.finest(levels, [1.0], [9], interp)
            self.assertEqual(len(x), 20)
            inside = (x > 3) & (x < 6)
            np.testing.assert_allclose(v[inside], x[inside] + 1)
            # away from the level's edge, within the coarse centres (beyond them: the nearest)
            far = ((x < 2.5) | (x > 6.5)) & (x > 0.5) & (x < 9.5)
            np.testing.assert_allclose(v[far], x[far], atol=0.5 if interp == "nearest" else 1e-12)

    def test_2d(self):
        levels = two_levels(lambda x: x[:, 0] + 2 * x[:, 1], [8, 6], [(2, 4), (1, 3)])
        for interp in ("nearest", "linear"):
            (x, y), v = watch.finest(levels, [1.0, 1.0], [[0, 0], [7, 5]], interp)
            self.assertEqual(v.shape, (16, 12))
            X, Y = np.meshgrid(x, y, indexing="ij")
            inside = (X > 2) & (X < 5) & (Y > 1) & (Y < 4)
            np.testing.assert_allclose(v[inside], (X + 2 * Y + 1)[inside])
            far = ((X < 1.5) | (X > 5.5) | (Y < 0.5) | (Y > 4.5)) & (X > 0.5) & (X < 7.5)
            far &= (Y > 0.5) & (Y < 5.5)
            np.testing.assert_allclose(
                v[far], (X + 2 * Y)[far], atol=1.5 if interp == "nearest" else 1e-12
            )

    def tearDown(self):
        ph.clearDict()


if __name__ == "__main__":
    unittest.main()
