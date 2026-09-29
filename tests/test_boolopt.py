"""Tests for boolopt.parse_bool (drafted by Lee Minjun, WI-0060; BRK-0066)."""
import unittest

from brkraw_sordino import boolopt


class ParseBoolTests(unittest.TestCase):
    def test_bool_unchanged(self):
        self.assertIs(boolopt.parse_bool("x", True), True)
        self.assertIs(boolopt.parse_bool("x", False), False)

    def test_int(self):
        self.assertIs(boolopt.parse_bool("x", 1), True)
        self.assertIs(boolopt.parse_bool("x", 0), False)
        for bad in (2, -1, 10):
            with self.assertRaises(ValueError):
                boolopt.parse_bool("x", bad)

    def test_strings_case_insensitive(self):
        for s in ("true", "True", "TRUE", " tRuE ", "1", "yes", "YES", "on", "On"):
            self.assertIs(boolopt.parse_bool("x", s), True, s)
        for s in ("false", "False", "FALSE", "\tfalse\n", "0", "no", "No", "off", "OFF"):
            self.assertIs(boolopt.parse_bool("x", s), False, s)

    def test_bad_values(self):
        for bad in ("", " ", "maybe", "truee", None, 1.0, 0.0, [], [True], (1,), {}):
            with self.assertRaises(ValueError, msg=repr(bad)):
                boolopt.parse_bool("x", bad)

    def test_message(self):
        with self.assertRaises(ValueError) as cm:
            boolopt.parse_bool("estimate_k0", "maybe")
        self.assertEqual(str(cm.exception),
                         "estimate_k0 must be true or false (case-insensitive), got 'maybe'")
        with self.assertRaises(ValueError) as cm:
            boolopt.parse_bool("x", None)
        self.assertEqual(str(cm.exception), "x must be true or false (case-insensitive), got None")


if __name__ == "__main__":
    unittest.main()
