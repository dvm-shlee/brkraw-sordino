"""Parent-owned tests for WI-0056-LEE-4 (Park). Standard library only."""
import math
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import stagetable  # noqa: E402


class FormatValue(unittest.TestCase):
    def test_values(self):
        self.assertEqual(stagetable.format_value(None, ".4f"), "-")
        self.assertEqual(stagetable.format_value("n/a", ".4f"), "n/a")
        self.assertEqual(stagetable.format_value(0.00449, ".5f"), "0.00449")
        self.assertEqual(stagetable.format_value(0.1234, ".1%"), "12.3%")
        self.assertEqual(stagetable.format_value(200, "d"), "200")
        self.assertEqual(stagetable.format_value(float("nan"), ".4f"), "nan")
        self.assertEqual(stagetable.format_value(1.5, ".2f"), "1.50")


class StageTable(unittest.TestCase):
    def test_example(self):
        out = stagetable.format_stage_table(
            ["oscillation", "diff"], ["S0", "S1"],
            {("oscillation", "S0"): 0.00449, ("oscillation", "S1"): 0.00445,
             ("diff", "S1"): 0.096},
            fmt={"oscillation": ".5f", "diff": ".1%"}, first_header="metric")
        self.assertEqual(out, "\n".join([
            "| metric | S0 | S1 |",
            "| --- | --- | --- |",
            "| oscillation | 0.00449 | 0.00445 |",
            "| diff | - | 9.6% |",
        ]))

    def test_defaults_and_string_fmt(self):
        out = stagetable.format_stage_table(["a"], ["x", "y", "z"], {("a", "y"): 2.0}, fmt=".1f")
        self.assertEqual(out, "| 지표 | x | y | z |\n| --- | --- | --- | --- |\n| a | - | 2.0 | - |")
        out = stagetable.format_stage_table(["a"], ["x"], {("a", "x"): 0.5})
        self.assertEqual(out.splitlines()[-1], "| a | 0.5000 |")

    def test_dict_fmt_falls_back(self):
        out = stagetable.format_stage_table(["a", "b"], ["x"], {("a", "x"): 0.5, ("b", "x"): 3},
                                            fmt={"b": "d"})
        self.assertEqual(out.splitlines()[2], "| a | 0.5000 |")
        self.assertEqual(out.splitlines()[3], "| b | 3 |")

    def test_order_and_no_trailing_newline(self):
        out = stagetable.format_stage_table(["b", "a"], ["y", "x"], {("a", "x"): "v", ("b", "y"): "w"})
        lines = out.split("\n")
        self.assertEqual(lines[0], "| 지표 | y | x |")
        self.assertEqual(lines[2], "| b | w | - |")
        self.assertEqual(lines[3], "| a | - | v |")
        self.assertFalse(out.endswith("\n"))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            stagetable.format_stage_table([], ["x"], {})
        with self.assertRaises(ValueError):
            stagetable.format_stage_table(["a"], [], {})


if __name__ == "__main__":
    unittest.main()
