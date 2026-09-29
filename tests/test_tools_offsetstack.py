"""Parent-owned tests for offsetstack.py (WI-0056-LEE-5, Park). Standard library only."""
import math
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import offsetstack as os_  # noqa: E402


class StackLayout(unittest.TestCase):
    def test_example(self):
        r = os_.stack_layout([0.0, -1.0, 0.5], [2.0, 1.0, 3.0], gap_frac=0.25)
        self.assertEqual(r["lo"], -1.0)
        self.assertEqual(r["hi"], 3.0)
        self.assertAlmostEqual(r["step"], 5.0)
        self.assertEqual(len(r["offsets"]), 3)
        for got, want in zip(r["offsets"], [11.0, 6.0, 1.0]):
            self.assertAlmostEqual(got, want)
        for got, want in zip(r["centres"], [12.0, 7.0, 2.0]):
            self.assertAlmostEqual(got, want)

    def test_default_gap(self):
        r = os_.stack_layout([0.0, 0.0], [1.0, 1.0])
        self.assertAlmostEqual(r["step"], 1.15)
        self.assertAlmostEqual(r["offsets"][0], 1.15)
        self.assertAlmostEqual(r["offsets"][1], 0.0)

    def test_rows_do_not_overlap(self):
        mins = [-0.3, 0.1, -2.0, 0.0]
        maxs = [1.2, 0.4, -1.5, 3.0]
        r = os_.stack_layout(mins, maxs, gap_frac=0.0)
        bands = [(mn + o, mx + o) for mn, mx, o in zip(mins, maxs, r["offsets"])]
        for upper, lower in zip(bands, bands[1:]):
            self.assertGreaterEqual(upper[0] + 1e-12, lower[1])

    def test_flat_range_uses_unit_band(self):
        r = os_.stack_layout([2.0, 2.0], [2.0, 2.0], gap_frac=0.0)
        self.assertEqual(r["lo"], 2.0)
        self.assertEqual(r["hi"], 3.0)
        self.assertAlmostEqual(r["step"], 1.0)
        self.assertAlmostEqual(r["offsets"][0], -1.0)
        self.assertAlmostEqual(r["offsets"][1], -2.0)

    def test_errors(self):
        with self.assertRaises(ValueError):
            os_.stack_layout([], [])
        with self.assertRaises(ValueError):
            os_.stack_layout([0.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            os_.stack_layout([1.0], [0.0])
        with self.assertRaises(ValueError):
            os_.stack_layout([0.0], [1.0], gap_frac=-0.1)

    def test_returns_lists_of_float(self):
        r = os_.stack_layout([0, 1], [2, 3])
        self.assertIsInstance(r["offsets"], list)
        self.assertIsInstance(r["centres"], list)
        self.assertTrue(all(isinstance(x, float) for x in r["offsets"] + r["centres"]))
        self.assertEqual(set(r), {"lo", "hi", "step", "offsets", "centres"})


class WrapToPi(unittest.TestCase):
    def test_values(self):
        self.assertAlmostEqual(os_.wrap_to_pi(0.0), 0.0)
        self.assertAlmostEqual(os_.wrap_to_pi(math.pi), math.pi)
        self.assertAlmostEqual(os_.wrap_to_pi(-math.pi), math.pi)
        self.assertAlmostEqual(os_.wrap_to_pi(3 * math.pi / 2), -math.pi / 2)
        self.assertAlmostEqual(os_.wrap_to_pi(-3 * math.pi / 2), math.pi / 2)
        self.assertAlmostEqual(os_.wrap_to_pi(7.0), 7.0 - 2 * math.pi)
        self.assertAlmostEqual(os_.wrap_to_pi(-20.0), -20.0 + 6 * math.pi)

    def test_range(self):
        for k in range(-50, 51):
            v = os_.wrap_to_pi(k * 0.37)
            self.assertGreater(v, -math.pi)
            self.assertLessEqual(v, math.pi)


class NeighbourDiff(unittest.TestCase):
    def test_cyclic(self):
        d = os_.neighbour_diff([0.0, 0.5, 3.0, -3.0])
        want = [0.0 - (-3.0), 0.5, 2.5, -6.0 + 2 * math.pi]
        self.assertEqual(len(d), 4)
        for got, w in zip(d, want):
            self.assertAlmostEqual(got, w)

    def test_not_cyclic(self):
        d = os_.neighbour_diff([1.0, 1.25, 1.0], cyclic=False)
        for got, w in zip(d, [0.0, 0.25, -0.25]):
            self.assertAlmostEqual(got, w)

    def test_single_and_empty(self):
        self.assertEqual(os_.neighbour_diff([]), [])
        self.assertEqual(os_.neighbour_diff([2.0]), [0.0])
        self.assertEqual(os_.neighbour_diff([2.0], cyclic=False), [0.0])

    def test_wrapped(self):
        d = os_.neighbour_diff([3.0, -3.0], cyclic=False)
        self.assertAlmostEqual(d[1], -6.0 + 2 * math.pi)
        self.assertIsInstance(d, list)


if __name__ == "__main__":
    unittest.main()
