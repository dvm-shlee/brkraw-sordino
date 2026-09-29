"""Parent-owned tests for WI-0058-LEE-1 (Park): tools/peakfit.py. Standard library only."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import peakfit  # noqa: E402


class RefinePeak(unittest.TestCase):
    def test_exact_parabola(self):
        xs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        ys = [-(x - 1.2) ** 2 + 3.0 for x in xs]
        x, y = peakfit.refine_peak(xs, ys)
        self.assertAlmostEqual(x, 1.2, places=12)
        self.assertAlmostEqual(y, 3.0, places=12)

    def test_negative_grid(self):
        xs = [-3.0 + 0.1 * i for i in range(61)]
        ys = [1.0 - 2.0 * (x + 0.73) ** 2 for x in xs]
        x, y = peakfit.refine_peak(xs, ys)
        self.assertAlmostEqual(x, -0.73, places=9)
        self.assertAlmostEqual(y, 1.0, places=9)

    def test_edge_returns_grid_point(self):
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [5.0, 4.0, 3.0, 2.0]
        self.assertEqual(peakfit.refine_peak(xs, ys), (0.0, 5.0))
        self.assertEqual(peakfit.refine_peak(xs, list(reversed(ys))), (3.0, 5.0))

    def test_flat_neighbours(self):
        xs = [0.0, 1.0, 2.0]
        ys = [1.0, 1.0, 1.0]
        self.assertEqual(peakfit.refine_peak(xs, ys), (0.0, 1.0))

    def test_first_max_on_tie(self):
        xs = [0.0, 1.0, 2.0, 3.0, 4.0]
        ys = [0.0, 2.0, 0.0, 2.0, 0.0]
        x, y = peakfit.refine_peak(xs, ys)
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 2.0)

    def test_errors(self):
        with self.assertRaises(ValueError):
            peakfit.refine_peak([0.0, 1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            peakfit.refine_peak([0.0, 1.0, 2.0], [1.0, 2.0])

    def test_returns_floats(self):
        x, y = peakfit.refine_peak([0, 1, 2], [0, 1, 0])
        self.assertIsInstance(x, float)
        self.assertIsInstance(y, float)


class WeightedStats(unittest.TestCase):
    def test_equal_weights(self):
        m, s = peakfit.weighted_mean_std([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(m, 2.5)
        self.assertAlmostEqual(s, math.sqrt(1.25))

    def test_weights(self):
        m, s = peakfit.weighted_mean_std([0.0, 10.0], [3.0, 1.0])
        self.assertAlmostEqual(m, 2.5)
        self.assertAlmostEqual(s, math.sqrt((3 * 6.25 + 1 * 56.25) / 4.0))

    def test_zero_weight_ignored(self):
        m, s = peakfit.weighted_mean_std([5.0, 100.0], [1.0, 0.0])
        self.assertAlmostEqual(m, 5.0)
        self.assertAlmostEqual(s, 0.0)

    def test_errors(self):
        with self.assertRaises(ValueError):
            peakfit.weighted_mean_std([], [])
        with self.assertRaises(ValueError):
            peakfit.weighted_mean_std([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            peakfit.weighted_mean_std([1.0, 2.0], [0.0, 0.0])
        with self.assertRaises(ValueError):
            peakfit.weighted_mean_std([1.0, 2.0], [1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
