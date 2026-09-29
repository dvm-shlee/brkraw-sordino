"""Parent-owned tests for WI-0056-LEE-3 (Park). Standard library only."""
import cmath
import math
import random
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import circstats  # noqa: E402


def ref_stats(angles):
    z = sum(cmath.exp(1j * a) for a in angles) / len(angles)
    r = abs(z)
    std = 0.0 if r >= 1.0 else (math.inf if r == 0.0 else math.sqrt(-2.0 * math.log(r)))
    return cmath.phase(z), r, std


class CircularStats(unittest.TestCase):
    def test_identical(self):
        s = circstats.circular_stats([0.0, 0.0, 0.0])
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["mean"], 0.0, 12)
        self.assertAlmostEqual(s["resultant_length"], 1.0, 12)
        self.assertEqual(s["std"], 0.0)

    def test_symmetric_pair(self):
        s = circstats.circular_stats([0.1, -0.1])
        self.assertAlmostEqual(s["mean"], 0.0, 12)
        self.assertAlmostEqual(s["resultant_length"], math.cos(0.1), 12)
        self.assertAlmostEqual(s["std"], math.sqrt(-2 * math.log(math.cos(0.1))), 12)

    def test_opposite(self):
        s = circstats.circular_stats([0.0, math.pi])
        self.assertLess(s["resultant_length"], 1e-12)
        self.assertTrue(math.isinf(s["std"]) or s["std"] > 5.0)

    def test_wrap_around_mean(self):
        s = circstats.circular_stats([3.0, -3.0])
        self.assertAlmostEqual(abs(s["mean"]), math.pi, 9)

    def test_random_against_reference(self):
        rng = random.Random(7)
        for _ in range(20):
            angles = [rng.uniform(-4, 4) for _ in range(rng.randint(1, 50))]
            s = circstats.circular_stats(angles)
            mean, r, std = ref_stats(angles)
            self.assertEqual(s["n"], len(angles))
            self.assertAlmostEqual(s["resultant_length"], r, 12)
            self.assertAlmostEqual(math.cos(s["mean"] - mean), 1.0, 9)
            if math.isinf(std):
                self.assertTrue(math.isinf(s["std"]))
            else:
                self.assertAlmostEqual(s["std"], std, 9)

    def test_empty(self):
        with self.assertRaises(ValueError):
            circstats.circular_stats([])


class Unwrap(unittest.TestCase):
    def test_examples(self):
        out = circstats.unwrap([0.0, 3.0, -3.0, 0.0])
        self.assertEqual(len(out), 4)
        for a, b in zip(out, [0.0, 3.0, 3.0 + (2 * math.pi - 6.0), 2 * math.pi]):
            self.assertAlmostEqual(a, b, 12)
        self.assertEqual(circstats.unwrap([1.0, 1.5, 2.0]), [1.0, 1.5, 2.0])
        out = circstats.unwrap([-3.0, 3.0])
        self.assertAlmostEqual(out[1], 3.0 - 2 * math.pi, 12)
        self.assertEqual(circstats.unwrap([]), [])
        self.assertEqual(circstats.unwrap([0.4]), [0.4])

    def test_smooth_after_unwrap_and_congruent(self):
        rng = random.Random(3)
        truth = [0.0]
        for _ in range(300):
            truth.append(truth[-1] + rng.uniform(-2.5, 2.5))
        wrapped = [math.atan2(math.sin(a), math.cos(a)) for a in truth]
        out = circstats.unwrap(wrapped)
        for i in range(1, len(out)):
            self.assertLessEqual(abs(out[i] - out[i - 1]), math.pi + 1e-9)
            k = (out[i] - wrapped[i]) / (2 * math.pi)
            self.assertAlmostEqual(k, round(k), 9)
        # same as the true sequence up to one constant multiple of 2 pi
        k0 = (out[0] - truth[0]) / (2 * math.pi)
        for a, b in zip(out, truth):
            self.assertAlmostEqual((a - b) / (2 * math.pi), k0, 9)

    def test_input_not_modified(self):
        data = [0.0, 3.0, -3.0]
        copy = list(data)
        circstats.unwrap(data)
        self.assertEqual(data, copy)


if __name__ == "__main__":
    unittest.main()
