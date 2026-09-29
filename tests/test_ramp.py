"""Tests for brkraw_sordino.ramp (closed form of the gradient ramp)."""
import unittest

from brkraw_sordino import ramp


def numeric_integral(t, start, length, n=200000):
    if t == 0:
        return 0.0
    step = t / n
    acc = 0.0
    for k in range(n):
        s = (k + 0.5) * step
        acc += ramp.ramp_fraction(s, start, length)
    return acc * step


class Fraction(unittest.TestCase):
    def test_values(self):
        self.assertEqual(ramp.ramp_fraction(5.0, 2.0, 4.0), 0.75)
        self.assertEqual(ramp.ramp_fraction(1.0, 2.0, 4.0), 0.0)
        self.assertEqual(ramp.ramp_fraction(2.0, 2.0, 4.0), 0.0)
        self.assertEqual(ramp.ramp_fraction(6.0, 2.0, 4.0), 1.0)
        self.assertEqual(ramp.ramp_fraction(9.0, 2.0, 4.0), 1.0)

    def test_errors(self):
        for length in (0.0, -1.0):
            with self.assertRaises(ValueError):
                ramp.ramp_fraction(1.0, 0.0, length)


class Integral(unittest.TestCase):
    def test_examples(self):
        self.assertAlmostEqual(ramp.ramp_integral(1.0, 2.0, 4.0), 0.0, 12)
        self.assertAlmostEqual(ramp.ramp_integral(4.0, 2.0, 4.0), 0.5, 12)
        self.assertAlmostEqual(ramp.ramp_integral(10.0, 2.0, 4.0), 6.0, 12)
        self.assertAlmostEqual(ramp.ramp_integral(2.0, -2.0, 8.0), 0.75, 12)
        self.assertAlmostEqual(ramp.ramp_integral(10.0, -2.0, 8.0), 7.75, 12)
        self.assertAlmostEqual(ramp.ramp_integral(5.0, -10.0, 3.0), 5.0, 12)

    def test_matches_numeric(self):
        cases = [(3.0, 2.0, 4.0), (7.5, 2.0, 4.0), (1.0, -2.0, 8.0),
                 (9.0, -2.0, 8.0), (4.0, -10.0, 3.0), (0.4, 0.3, 0.05),
                 (615.0, -53.6, 615.0), (430.0, 2.0, 436.4)]
        for t, a, T in cases:
            self.assertAlmostEqual(ramp.ramp_integral(t, a, T),
                                   numeric_integral(t, a, T), 4, msg=(t, a, T))

    def test_zero_and_negative_t(self):
        self.assertEqual(ramp.ramp_integral(0.0, -2.0, 8.0), 0.0)
        self.assertAlmostEqual(ramp.ramp_integral(-1.0, -2.0, 8.0),
                               (1 ** 2 / 16) - 0.25, 12)

    def test_errors(self):
        with self.assertRaises(ValueError):
            ramp.ramp_integral(1.0, 0.0, 0.0)


class Delay(unittest.TestCase):
    def test_values(self):
        self.assertAlmostEqual(ramp.phase_delay(10.0, 2.0, 4.0), 4.0, 12)
        self.assertAlmostEqual(ramp.phase_delay(5.0, -10.0, 3.0), 0.0, 12)
        self.assertAlmostEqual(ramp.phase_delay(1.5, 2.0, 4.0), 1.5, 12)

    def test_errors(self):
        with self.assertRaises(ValueError):
            ramp.phase_delay(1.0, 0.0, -2.0)

    def test_returns_float(self):
        self.assertIsInstance(ramp.ramp_integral(1.0, 2.0, 4.0), float)
        self.assertIsInstance(ramp.phase_delay(10.0, 2.0, 4.0), float)


if __name__ == "__main__":
    unittest.main()
