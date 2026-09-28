"""Saving as uint16 with slope/intercept must keep the brightest voxel
(a slope of range / 2**16 maps the maximum to 65536, which wraps to 0)."""

import numpy as np

from brkraw_sordino.hook import _calc_slope_inter


def _roundtrip(data):
    stored, slope, inter = _calc_slope_inter(np.asarray(data, dtype=float))
    return stored, np.asarray(stored, dtype=float) * slope + inter


def test_maximum_is_kept():
    data = np.array([[[0.0, 1.0], [2.0, 4.0]]])
    stored, back = _roundtrip(data)
    assert stored.max() == 65535
    assert np.isclose(back.max(), 4.0)
    assert int(np.argmax(back)) == int(np.argmax(data))


def test_values_within_one_step():
    rng = np.random.default_rng(0)
    data = rng.normal(100.0, 30.0, size=(4, 4, 3, 2))
    stored, back = _roundtrip(data)
    step = (data.max() - data.min()) / 65535
    assert np.all(np.abs(back - data.squeeze()) <= step / 2 + 1e-9)


def test_constant_data():
    stored, back = _roundtrip(np.full((2, 2, 2), 7.0))
    assert np.allclose(back, 7.0)
