"""plamn.counts: particles as one count per cell, advanced as if they moved: the total exact
(periodic, or with the outflow of open boundaries), a cold drift a plain shift, a blob's centre
moving at the bulk velocity, its variance growing by (vth dt / dx)^2 + 1/6 cells^2 a step; box
sums, the loading's cut-off, refined levels' totals, the gradient tagging"""

import unittest

import numpy as np

from plamn.counts import Counts, landing


class CountsTest(unittest.TestCase):
    def test_landing_is_the_cell_hat_convolved_with_the_gaussian(self):
        np.testing.assert_allclose(landing(0.3, 0.0, range(-2, 3)), [0, 0, 0.7, 0.3, 0],
                                   atol=1e-15)
        k = np.arange(-12, 13)
        p = landing(0.0, 2.0, k)
        self.assertAlmostEqual(p.sum(), 1.0, places=8)  # the tails past 6 sigma
        self.assertAlmostEqual((p * k * k).sum(), 4 + 1 / 6, places=6)

    def test_total_exact(self):
        c = Counts(np.full((40, 30), 50), dl=(0.2, 0.3), seed=1)
        total = c.total
        for _ in range(50):
            c.advance(0.01, vbulk=[lambda x, y: np.sin(y), 2.0], vth=[3.0, lambda x, y: 1 + x])
        self.assertEqual(c.total, total)

    def test_open_boundaries_count_the_outflow(self):
        c = Counts(np.full((20,), 100), dl=0.1, periodic=False, seed=2)
        total = c.total
        for _ in range(10):
            c.advance(0.01, vbulk=4.0, vth=2.0)
        self.assertGreater(c.outflow, 0)
        self.assertEqual(c.total + c.outflow, total)

    def test_cold_drift_of_whole_cells_is_a_shift(self):
        rng = np.random.default_rng(3)
        start = rng.integers(0, 100, size=(16, 12))
        c = Counts(start.copy(), dl=(0.5, 0.25))
        c.advance(0.1, vbulk=[10.0, -5.0])  # +2 cells in x, -2 in y
        np.testing.assert_array_equal(c.counts, np.roll(start, (2, -2), axis=(0, 1)))

    def test_a_blob_moves_at_the_bulk_velocity_and_spreads(self):
        n, steps, shift, sigma = 400, 20, 0.3, 1.0  # cells a step
        start = np.zeros(n, dtype=np.int64)
        start[100] = 10_000_000
        c = Counts(start, dl=1.0, seed=4)
        for _ in range(steps):
            c.advance(1.0, vbulk=shift, vth=sigma)
        x = np.arange(n)
        mean = (c.counts * x).sum() / c.total
        var = (c.counts * (x - mean) ** 2).sum() / c.total
        self.assertAlmostEqual(mean, 100 + steps * shift, delta=0.01)
        self.assertAlmostEqual(var, steps * (sigma**2 + 1 / 6), delta=0.02 * var)

    def test_box_sums(self):
        rng = np.random.default_rng(5)
        c = Counts(rng.integers(0, 50, size=(9, 7, 5)), dl=1.0)
        boxes = [([0, 0, 0], [8, 6, 4]), ([2, 1, 0], [4, 5, 3]), ([3, 3, 3], [3, 3, 3])]
        for (lo, up), s in zip(boxes, c.box_sums(boxes)):
            direct = c.counts[lo[0]:up[0] + 1, lo[1]:up[1] + 1, lo[2]:up[2] + 1].sum()
            self.assertEqual(s, direct)

    def test_loading_cut_off(self):
        c = Counts.from_density((10, 4), (0.5, 0.5),
                                [lambda x, y: np.where(x < 2.5, 1.0, 0.0), lambda x, y: 1 + 0 * x],
                                [100, 7])
        self.assertTrue(np.all(c.counts[:5] == 107))
        self.assertTrue(np.all(c.counts[5:] == 7))

    def test_refined_levels_carry_the_split(self):
        c = Counts(np.full((6, 4), 20), dl=1.0)
        for ln in (1, 2):
            r = c.refined(ln, ratio=2, refined_nbr=4)
            self.assertEqual(r.shape, (6 * 2**ln, 4 * 2**ln))
            self.assertAlmostEqual(r.sum(), c.total * 4**ln)

    def test_tags_where_the_counts_change(self):
        counts = np.zeros((20,), dtype=np.int64)
        counts[8:12] = 100
        tagged = Counts(counts, dl=1.0).tags(0.2)
        self.assertEqual(list(np.nonzero(tagged)[0]), [7, 8, 11, 12])


if __name__ == "__main__":
    unittest.main()
