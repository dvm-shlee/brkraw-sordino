"""Tests for tools/goldencmp.py, the comparison of the golden-trajectory success criteria (WI-0113 CP2).

One calculation for every comparison (WI-0112 plan): magnitude images; mask =
Otsu threshold of each image's magnitude, largest 26-connected component;
Dice of the two masks; centroid shift; NRMSE in the reference mask after one
least-squares scale a = sum|T||R| / sum|T|^2; boundary width (10-90 %) of the
central profiles.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import goldencmp as gc  # noqa: E402


# ------------------------------------------------------------------ worked examples
def test_otsu_worked_example_and_tie_rule():
    assert gc.otsu_threshold([0, 0, 0, 10, 10, 10], nbins=10) == 1.0


def test_otsu_separates_two_groups():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1.0, 0.1, 5000), rng.normal(5.0, 0.3, 2000)
    t = gc.otsu_threshold(np.concatenate([a, b]))
    # every bin of the empty gap scores the same; the first one is taken
    assert a.max() <= t < b.min()


@pytest.mark.parametrize("values,word", [([], "empty"), ([2.0, 2.0, 2.0], "constant")])
def test_otsu_errors(values, word):
    with pytest.raises(ValueError, match=word):
        gc.otsu_threshold(values)


def test_largest_component_worked_example():
    out = gc.largest_component([1, 1, 0, 1, 1, 1, 0, 1])
    assert out.tolist() == [False, False, False, True, True, True, False, False]


def test_largest_component_counts_corner_neighbours_in_3d():
    m = np.zeros((5, 5, 5), dtype=bool)
    m[0, 0, 0] = m[1, 1, 1] = m[2, 2, 2] = True       # corner-connected chain of 3
    m[4, 4, 0] = m[4, 3, 0] = True                     # face-connected pair
    out = gc.largest_component(m)
    assert out.sum() == 3 and out[0, 0, 0] and out[2, 2, 2] and not out[4, 4, 0]
    assert not gc.largest_component(np.zeros((3, 3, 3))).any()


def test_dice_centroid_nrmse_worked_examples():
    assert gc.dice([1, 1, 0, 0], [1, 0, 1, 0]) == 0.5
    m = np.zeros((5, 6), dtype=bool)
    m[1, 2] = m[3, 4] = True
    assert gc.centroid(m).tolist() == [2.0, 3.0]
    assert gc.scaled_nrmse([2, 4], [1, 2], [True, True]) == (0.0, 0.5)


def test_scaled_nrmse_is_the_formula():
    rng = np.random.default_rng(1)
    t = rng.normal(size=(6, 7)) + 1j * rng.normal(size=(6, 7))
    r = rng.normal(size=(6, 7))
    m = np.abs(r) > 0.3
    tv, rv = np.abs(t)[m], np.abs(r)[m]
    a = (tv * rv).sum() / (tv * tv).sum()
    want = np.sqrt(np.mean((a * tv - rv) ** 2)) / np.mean(rv)
    got, ga = gc.scaled_nrmse(t, r, m)
    assert got == pytest.approx(want, rel=1e-12) and ga == pytest.approx(a, rel=1e-12)
    # one overall scale of the test image does not change the error
    assert gc.scaled_nrmse(7.5 * t, r, m)[0] == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("call,word", [
    (lambda: gc.dice([1, 0], [1, 0, 0]), "shape"),
    (lambda: gc.dice([0, 0], [0, 0]), "empty"),
    (lambda: gc.centroid([0, 0]), "empty"),
    (lambda: gc.scaled_nrmse([1, 2], [1, 2], [True]), "shape"),
    (lambda: gc.scaled_nrmse([1, 2], [1, 2], [False, False]), "empty"),
    (lambda: gc.scaled_nrmse([0, 0], [1, 2], [True, True]), "zero"),
])
def test_errors_name_the_problem(call, word):
    with pytest.raises(ValueError, match=word):
        call()


# ------------------------------------------------------------------ objects
def _ball(n=32, radius=9.0, centre=(15.5, 16.0, 16.5), noise=0.02, seed=0):
    z, y, x = np.indices((n, n, n), dtype=float)
    d = np.sqrt((z - centre[0]) ** 2 + (y - centre[1]) ** 2 + (x - centre[2]) ** 2)
    img = (d <= radius).astype(float)
    rng = np.random.default_rng(seed)
    return img + noise * rng.normal(size=img.shape), d <= radius


def test_object_mask_finds_the_ball():
    img, truth = _ball()
    m = gc.object_mask(img)
    assert gc.dice(m, truth) > 0.99


def _ramp_box(n=40, lo=10, hi=30, ramp=10.0):
    """1 inside [lo, hi), falling linearly to 0 over ``ramp`` voxels outside, on every axis."""
    x = np.arange(n, dtype=float)
    dist = np.maximum(np.maximum(lo - x, x - (hi - 1)), 0.0)
    prof = np.clip(1.0 - dist / ramp, 0.0, 1.0)
    return prof[:, None, None] * prof[None, :, None] * prof[None, None, :]


def test_edge_width_of_a_linear_ramp():
    img = _ramp_box(ramp=10.0)
    mask = gc.object_mask(img)
    for axis in range(3):
        # 90 % to 10 % of a linear 10-voxel ramp is 8 voxels on both sides
        assert gc.edge_width(img, mask, axis) == pytest.approx(8.0, abs=1e-9)
    sharper = _ramp_box(ramp=4.0)
    assert gc.edge_width(sharper, mask, 0) == pytest.approx(3.2, abs=1e-9)


def test_edge_width_needs_the_edge_inside_the_image():
    img = np.ones((8, 8, 8))
    img[0, 0, 0] = 0.0
    with pytest.raises(ValueError, match="edge"):
        gc.edge_width(img, np.ones((8, 8, 8), dtype=bool), 0)


def test_edge_sides_report_each_side_and_a_missing_edge():
    img = _ramp_box(n=40, lo=10, hi=40, ramp=10.0)        # the + side runs out of the image
    mask = gc.object_mask(_ramp_box(ramp=10.0))
    sides = gc.edge_sides(img, mask, 0)
    assert sides["-"]["width"] == pytest.approx(8.0, abs=1e-9)
    assert sides["+"]["width"] is None and sides["+"]["lowest_fraction"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="edge"):
        gc.edge_width(img, mask, 0)
    both = gc.edge_sides(_ramp_box(ramp=4.0), mask, 2)
    assert both["+"]["width"] == pytest.approx(3.2, abs=1e-9)
    assert both["-"]["width"] == pytest.approx(3.2, abs=1e-9)


def test_compare_reports_the_criteria_numbers():
    ref, _ = _ball(seed=1)
    test, _ = _ball(seed=2)
    out = gc.compare(3.0 * test, ref)
    assert set(out) >= {"dice", "centroid_shift_vox", "nrmse", "scale"}
    assert out["dice"] > 0.98 and out["centroid_shift_vox"] < 0.1
    assert out["scale"] == pytest.approx(1 / 3, rel=0.05)
    shifted = np.roll(test, 3, axis=0)
    worse = gc.compare(shifted, ref)
    assert worse["dice"] < out["dice"] and worse["nrmse"] > out["nrmse"]
    assert worse["centroid_shift_vox"] == pytest.approx(3.0, abs=0.1)
