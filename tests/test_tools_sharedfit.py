"""Parent-owned tests for WI-0058-LEE-3 (Park): tools/sharedfit.py. Standard library only.

The reference for noisy data is an independent route: the full normal
equations over all G + 2 unknowns, solved by Gauss-Jordan elimination here."""
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import sharedfit as sf  # noqa: E402


def _reference(xs, ys, s, w=None, drift=True):
    g_n = len(xs)
    n_u = g_n + (2 if drift else 1)
    m = [[0.0] * n_u for _ in range(n_u)]
    r = [0.0] * n_u
    for g in range(g_n):
        for j, (x, y) in enumerate(zip(xs[g], ys[g])):
            wt = 1.0 if w is None else w[g][j]
            row = [0.0] * n_u
            row[0] = 1.0
            if drift:
                row[1] = s[g]
            row[(2 if drift else 1) + g] = x
            for p in range(n_u):
                r[p] += wt * row[p] * y
                for q in range(n_u):
                    m[p][q] += wt * row[p] * row[q]
    a = [m[i][:] + [r[i]] for i in range(n_u)]
    for c in range(n_u):
        piv = max(range(c, n_u), key=lambda i: abs(a[i][c]))
        a[c], a[piv] = a[piv], a[c]
        for i in range(n_u):
            if i != c:
                f = a[i][c] / a[c][c]
                for k in range(c, n_u + 1):
                    a[i][k] -= f * a[c][k]
    sol = [a[i][n_u] / a[i][i] for i in range(n_u)]
    if drift:
        return sol[0], sol[1], sol[2:]
    return sol[0], 0.0, sol[1:]


def _make(rng, g_n, n, a0, a1, noise):
    xs, ys, s, b = [], [], [], []
    for g in range(g_n):
        sg = -0.5 + g / max(g_n - 1, 1)
        bg = rng.uniform(-3.0, -0.5)
        x = [0.5 + j + rng.uniform(-0.1, 0.1) for j in range(n)]
        y = [a0 + a1 * sg + bg * xj + rng.gauss(0.0, noise) for xj in x]
        xs.append(x); ys.append(y); s.append(sg); b.append(bg)
    return xs, ys, s, b


class FitTest(unittest.TestCase):
    def test_example(self):
        xs = [[1.0, 2.0], [1.0, 2.0]]
        ys = [[2.0 - 1.0 - 1.0, 2.0 - 1.0 - 2.0], [2.0 + 1.0 + 3.0, 2.0 + 1.0 + 6.0]]
        a0, a1, b = sf.shared_intercept_fit(xs, ys, [-1.0, 1.0])
        self.assertIsInstance(b, list)
        self.assertAlmostEqual(a0, 2.0, places=9)
        self.assertAlmostEqual(a1, 1.0, places=9)
        self.assertAlmostEqual(b[0], -1.0, places=9)
        self.assertAlmostEqual(b[1], 3.0, places=9)

    def test_exact_recovery_many_groups(self):
        rng = random.Random(1)
        xs, ys, s, b_true = _make(rng, 40, 6, 1.7, -0.3, 0.0)
        a0, a1, b = sf.shared_intercept_fit(xs, ys, s)
        self.assertAlmostEqual(a0, 1.7, places=9)
        self.assertAlmostEqual(a1, -0.3, places=9)
        for got, want in zip(b, b_true):
            self.assertAlmostEqual(got, want, places=9)

    def test_noisy_matches_full_least_squares(self):
        rng = random.Random(7)
        xs, ys, s, _ = _make(rng, 12, 5, 0.4, 0.8, 0.05)
        got = sf.shared_intercept_fit(xs, ys, s)
        want = _reference(xs, ys, s)
        self.assertAlmostEqual(got[0], want[0], places=9)
        self.assertAlmostEqual(got[1], want[1], places=9)
        for u, v in zip(got[2], want[2]):
            self.assertAlmostEqual(u, v, places=9)

    def test_weighted_matches_full_least_squares(self):
        rng = random.Random(11)
        xs, ys, s, _ = _make(rng, 9, 4, -1.0, 0.2, 0.1)
        w = [[rng.uniform(0.1, 2.0) for _ in x] for x in xs]
        got = sf.shared_intercept_fit(xs, ys, s, w=w)
        want = _reference(xs, ys, s, w=w)
        self.assertAlmostEqual(got[0], want[0], places=9)
        self.assertAlmostEqual(got[1], want[1], places=9)
        for u, v in zip(got[2], want[2]):
            self.assertAlmostEqual(u, v, places=9)

    def test_no_drift(self):
        rng = random.Random(3)
        xs, ys, s, _ = _make(rng, 10, 4, 0.9, 0.5, 0.05)
        got = sf.shared_intercept_fit(xs, ys, s, drift=False)
        want = _reference(xs, ys, s, drift=False)
        self.assertEqual(got[1], 0.0)
        self.assertAlmostEqual(got[0], want[0], places=9)
        for u, v in zip(got[2], want[2]):
            self.assertAlmostEqual(u, v, places=9)

    def test_zero_weight_point_ignored(self):
        xs = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
        ys = [[1.0 - 1.0, 1.0 - 2.0, 50.0], [1.0 - 2.0, 1.0 - 4.0, 1.0 - 6.0]]
        w = [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
        a0, a1, b = sf.shared_intercept_fit(xs, ys, [0.0, 0.0], w=w, drift=False)
        self.assertAlmostEqual(a0, 1.0, places=9)
        self.assertAlmostEqual(b[0], -1.0, places=9)
        self.assertAlmostEqual(b[1], -2.0, places=9)


class ErrorTest(unittest.TestCase):
    def test_empty(self):
        with self.assertRaises(ValueError):
            sf.shared_intercept_fit([], [], [])

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            sf.shared_intercept_fit([[1.0, 2.0]], [[1.0, 2.0]], [0.0, 1.0])
        with self.assertRaises(ValueError):
            sf.shared_intercept_fit([[1.0, 2.0]], [[1.0]], [0.0])

    def test_too_few_points(self):
        with self.assertRaises(ValueError):
            sf.shared_intercept_fit([[1.0], [2.0, 3.0]], [[1.0], [2.0, 3.0]], [0.0, 1.0])

    def test_bad_weights(self):
        xs = [[1.0, 2.0], [1.0, 2.0]]
        ys = [[1.0, 2.0], [1.0, 2.0]]
        with self.assertRaises(ValueError):
            sf.shared_intercept_fit(xs, ys, [0.0, 1.0], w=[[1.0, 1.0]])
        with self.assertRaises(ValueError):
            sf.shared_intercept_fit(xs, ys, [0.0, 1.0], w=[[1.0, -1.0], [1.0, 1.0]])

    def test_no_slope_information(self):
        with self.assertRaises(ValueError):
            sf.shared_intercept_fit([[0.0, 0.0], [1.0, 2.0]], [[1.0, 1.0], [1.0, 2.0]], [0.0, 1.0])

    def test_singular_drift(self):
        # every group at the same s: a0 and a1 cannot be separated
        with self.assertRaises(ValueError):
            sf.shared_intercept_fit([[1.0, 2.0], [1.0, 3.0]], [[1.0, 2.0], [2.0, 1.0]], [0.5, 0.5])


class ResidualTest(unittest.TestCase):
    def test_exact_zero(self):
        xs = [[1.0, 2.0], [1.0, 2.0]]
        ys = [[0.0, -1.0], [6.0, 9.0]]
        self.assertAlmostEqual(sf.shared_residual_rms(xs, ys, [-1.0, 1.0], 2.0, 1.0, [-1.0, 3.0]), 0.0,
                               places=12)

    def test_value(self):
        xs = [[0.0, 1.0]]
        ys = [[1.0, 3.0]]
        # residuals 1 and 2 with weights 1 and 3: sqrt((1 + 12) / 4)
        got = sf.shared_residual_rms(xs, ys, [0.0], 0.0, 0.0, [1.0], w=[[1.0, 3.0]])
        self.assertAlmostEqual(got, math.sqrt(13.0 / 4.0), places=12)

    def test_errors(self):
        with self.assertRaises(ValueError):
            sf.shared_residual_rms([[0.0, 1.0]], [[1.0, 3.0]], [0.0], 0.0, 0.0, [1.0, 2.0])
        with self.assertRaises(ValueError):
            sf.shared_residual_rms([[0.0, 1.0]], [[1.0, 3.0]], [0.0], 0.0, 0.0, [1.0], w=[[0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
