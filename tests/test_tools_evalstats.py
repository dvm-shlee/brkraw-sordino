"""Tests for tools/evalstats.py (summary statistics of the evaluation tool)."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import evalstats as es  # noqa: E402


class CV(unittest.TestCase):
    def test_value(self):
        self.assertAlmostEqual(es.cv([1.0, 2.0, 3.0]), math.sqrt(2 / 3) / 2, 12)

    def test_constant(self):
        self.assertEqual(es.cv([5.0, 5.0]), 0.0)

    def test_errors(self):
        with self.assertRaises(ValueError):
            es.cv([])
        with self.assertRaises(ValueError):
            es.cv([-1.0, 1.0])

    def test_no_mutation(self):
        v = [3.0, 1.0, 2.0]
        es.cv(v)
        self.assertEqual(v, [3.0, 1.0, 2.0])


class Pearson(unittest.TestCase):
    def test_values(self):
        self.assertAlmostEqual(es.pearson([1, 2, 3], [2, 4, 6]), 1.0, 12)
        self.assertAlmostEqual(es.pearson([1, 2, 3], [3, 2, 1]), -1.0, 12)
        self.assertAlmostEqual(es.pearson([1, 2, 3, 4], [1, 3, 2, 4]), 0.8, 12)

    def test_errors(self):
        for a, b in (([1, 2], [1, 2, 3]), ([1], [1]), ([1, 1, 1], [1, 2, 3]),
                     ([1, 2, 3], [4, 4, 4])):
            with self.assertRaises(ValueError):
                es.pearson(a, b)


class Detrend(unittest.TestCase):
    def test_line(self):
        for x, y in zip(es.detrend_linear([1.0, 3.0, 5.0]), [0.0, 0.0, 0.0]):
            self.assertAlmostEqual(x, y, 12)

    def test_bump(self):
        out = es.detrend_linear([0.0, 2.0, 0.0])
        for x, y in zip(out, [-2 / 3, 4 / 3, -2 / 3]):
            self.assertAlmostEqual(x, y, 12)
        self.assertEqual(len(out), 3)

    def test_new_list(self):
        v = [0.0, 2.0, 0.0]
        out = es.detrend_linear(v)
        self.assertIsNot(out, v)
        self.assertEqual(v, [0.0, 2.0, 0.0])

    def test_error(self):
        with self.assertRaises(ValueError):
            es.detrend_linear([1.0])


class Oscillation(unittest.TestCase):
    def test_example(self):
        r = es.oscillation([100.0, 0.0, 2.0, 0.0, 2.0], exclude=1)
        self.assertEqual(set(r), {"n", "mean", "std", "rel_std", "peak_to_peak"})
        self.assertEqual(r["n"], 4)
        self.assertIsInstance(r["n"], int)
        self.assertAlmostEqual(r["mean"], 1.0, 12)
        self.assertAlmostEqual(r["std"], math.sqrt(0.8), 12)
        self.assertAlmostEqual(r["rel_std"], math.sqrt(0.8), 12)
        self.assertAlmostEqual(r["peak_to_peak"], 2.4, 12)

    def test_negative_mean_uses_abs(self):
        r = es.oscillation([-1.0, -3.0, -1.0, -3.0])
        self.assertGreater(r["rel_std"], 0.0)
        self.assertAlmostEqual(r["rel_std"], r["std"] / 2.0, 12)

    def test_errors(self):
        with self.assertRaises(ValueError):
            es.oscillation([1.0, 2.0, 3.0], exclude=-1)
        with self.assertRaises(ValueError):
            es.oscillation([1.0, 2.0, 3.0], exclude=1)
        with self.assertRaises(ValueError):
            es.oscillation([1.0, -1.0, 0.0])


class Pattern(unittest.TestCase):
    def test_example(self):
        r = es.pattern_stability([[9, 9, 9], [1, 2, 3], [2, 4, 6]], exclude=1)
        self.assertEqual(len(r), 2)
        for x in r:
            self.assertAlmostEqual(x, 1.0, 12)

    def test_order_and_value(self):
        rows = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 2.0, 3.0]]
        r = es.pattern_stability(rows)
        ref = [5 / 3, 2.0, 7 / 3]
        self.assertAlmostEqual(r[0], es.pearson(rows[0], ref), 12)
        self.assertAlmostEqual(r[1], es.pearson(rows[1], ref), 12)
        self.assertLess(r[1], 0)

    def test_errors(self):
        with self.assertRaises(ValueError):
            es.pattern_stability([[1, 2]], exclude=0)
        with self.assertRaises(ValueError):
            es.pattern_stability([[1, 2], [1, 2, 3]])
        with self.assertRaises(ValueError):
            es.pattern_stability([[1, 2], [2, 3], [3, 4]], exclude=-1)

    def test_no_mutation(self):
        rows = [[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]]
        es.pattern_stability(rows)
        self.assertEqual(rows, [[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]])


if __name__ == "__main__":
    unittest.main()
