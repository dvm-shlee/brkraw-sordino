"""Parent-owned tests for WI-0058-LEE-2 (Park): tools/curvefit.py. Standard library only."""
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import curvefit as cf  # noqa: E402


class PolyfitTest(unittest.TestCase):
    def test_line(self):
        c = cf.polyfit([0.0, 1.0, 2.0], [1.0, 3.0, 5.0], 1)
        self.assertIsInstance(c, list)
        self.assertEqual(len(c), 2)
        self.assertAlmostEqual(c[0], 1.0, places=9)
        self.assertAlmostEqual(c[1], 2.0, places=9)

    def test_quadratic(self):
        c = cf.polyfit([-1.0, 0.0, 1.0, 2.0], [2.0, 1.0, 2.0, 5.0], 2)
        for got, want in zip(c, [1.0, 0.0, 1.0]):
            self.assertAlmostEqual(got, want, places=9)

    def test_degree_zero_is_weighted_mean(self):
        c = cf.polyfit([0.0, 1.0, 2.0], [1.0, 2.0, 6.0], 0, w=[1.0, 1.0, 2.0])
        self.assertAlmostEqual(c[0], (1 + 2 + 12) / 4.0, places=9)

    def test_weights_select_points(self):
        # zero weight on an outlier: the line through the other points
        c = cf.polyfit([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 100.0], 1, w=[1.0, 1.0, 1.0, 0.0])
        self.assertAlmostEqual(c[0], 0.0, places=9)
        self.assertAlmostEqual(c[1], 1.0, places=9)

    def test_least_squares_matches_closed_form(self):
        rng = random.Random(4)
        x = [i * 0.37 for i in range(12)]
        y = [2.0 - 0.5 * xi + 0.1 * xi * xi + rng.uniform(-0.05, 0.05) for xi in x]
        c = cf.polyfit(x, y, 2)
        # residual is orthogonal to 1, x, x^2 at the optimum
        r = [yi - cf.polyval(c, xi) for xi, yi in zip(x, y)]
        for p in range(3):
            self.assertAlmostEqual(sum(ri * xi ** p for ri, xi in zip(r, x)), 0.0, places=8)

    def test_errors(self):
        with self.assertRaises(ValueError):
            cf.polyfit([0.0, 1.0], [1.0], 1)
        with self.assertRaises(ValueError):
            cf.polyfit([0.0, 1.0], [1.0, 2.0], -1)
        with self.assertRaises(ValueError):
            cf.polyfit([0.0, 1.0], [1.0, 2.0], 2)
        with self.assertRaises(ValueError):
            cf.polyfit([0.0, 1.0], [1.0, 2.0], 1, w=[1.0])
        with self.assertRaises(ValueError):
            cf.polyfit([0.0, 1.0], [1.0, 2.0], 1, w=[1.0, -1.0])
        with self.assertRaises(ValueError):
            cf.polyfit([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], 1)   # singular


class PolyvalTest(unittest.TestCase):
    def test_values(self):
        self.assertEqual(cf.polyval([1.0, 0.0, 1.0], 3.0), 10.0)
        self.assertEqual(cf.polyval([], 2.0), 0.0)
        self.assertAlmostEqual(cf.polyval([0.5, -2.0], 0.25), 0.0, places=12)


class UnwrapTest(unittest.TestCase):
    def test_example(self):
        u = cf.unwrap([3.0, -3.0])
        self.assertAlmostEqual(u[0], 3.0, places=12)
        self.assertAlmostEqual(u[1], 3.0 + (2 * math.pi - 6.0), places=12)

    def test_empty_and_smooth(self):
        self.assertEqual(cf.unwrap([]), [])
        # cumulative form out[i] = out[i-1] + d (as specified): equal within rounding
        for a, b in zip(cf.unwrap([0.1, 0.5, -0.2]), [0.1, 0.5, -0.2]):
            self.assertAlmostEqual(a, b, places=12)

    def test_ramp(self):
        true = [0.9 * i for i in range(20)]
        wrapped = [math.atan2(math.sin(t), math.cos(t)) for t in true]
        u = cf.unwrap(wrapped)
        for a, b in zip(u, true):
            self.assertAlmostEqual(a, b, places=9)


class ExtrapolateTest(unittest.TestCase):
    def test_log_line(self):
        r = cf.extrapolate_log_magnitude([1.0, 2.0, 3.0], [math.exp(-1.0), math.exp(-2.0), math.exp(-3.0)], [0.0])
        self.assertEqual(len(r), 1)
        self.assertAlmostEqual(r[0], 1.0, places=9)

    def test_log_quadratic_gaussian(self):
        x = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        mag = [3.0 * math.exp(-0.4 * xi * xi) for xi in x]
        r = cf.extrapolate_log_magnitude([xi * xi for xi in x], mag, [0.0, 0.01], degree=1)
        self.assertAlmostEqual(r[0], 3.0, places=9)
        self.assertAlmostEqual(r[1], 3.0 * math.exp(-0.004), places=9)

    def test_log_errors(self):
        with self.assertRaises(ValueError):
            cf.extrapolate_log_magnitude([1.0, 2.0], [1.0, 0.0], [0.0])
        with self.assertRaises(ValueError):
            cf.extrapolate_log_magnitude([1.0, 2.0], [1.0, -1.0], [0.0])
        with self.assertRaises(ValueError):
            cf.extrapolate_log_magnitude([1.0], [1.0], [0.0], degree=1)

    def test_phase(self):
        r = cf.extrapolate_phase([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], [0.0], 1)
        self.assertAlmostEqual(r[0], 0.0, places=9)

    def test_phase_across_wrap_not_rewrapped(self):
        x = [1.0, 2.0, 3.0, 4.0]
        true = [2.5 + 0.4 * xi for xi in x]          # crosses pi
        wrapped = [math.atan2(math.sin(t), math.cos(t)) for t in true]
        r = cf.extrapolate_phase(x, wrapped, [5.0, 0.0], 1)
        self.assertAlmostEqual(r[0], 2.5 + 2.0, places=9)   # 4.5, above pi: not wrapped
        self.assertAlmostEqual(r[1], 2.5, places=9)


if __name__ == "__main__":
    unittest.main()
