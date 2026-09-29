"""Tests for tools/legacytraj.py (the legacy trajectory moved out of src, BRK-0066)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import eval_stages as es  # noqa: E402
import legacytraj  # noqa: E402
from brkraw_sordino import traj as product_traj  # noqa: E402


def _recon_info(matrix=16, os_=2.0):
    npro = 2 * product_traj.calc_npro(matrix, 1.0)
    return {
        "Matrix": [matrix] * 3, "NPro": npro, "HalfAcquisition": False,
        "UseOrigin": False, "Reorder": False, "OverSampling": os_,
        "AcqDelayTotal_us": 6.75, "EffBandwidth_Hz": 75000.0,
    }


def test_legacy_equals_the_independent_stage_implementation():
    # eval_stages.legacy_trajectory was written apart from this module (and was
    # checked equal to the former product function, atol 1e-15)
    info = _recon_info()
    np.testing.assert_allclose(legacytraj.legacy_from_recon_info(info), es.legacy_trajectory(info, 1.0),
                               rtol=0, atol=1e-15)


def test_legacy_formula_spot_values():
    m, os_ = 16, 2.0
    g = np.array([[1.0, 0.0, 0.6], [0.0, 1.0, 0.0], [0.0, 0.0, 0.8]])   # (3, 3 spokes)
    off = 1.5
    t = legacytraj.calc_radial_traj3d_legacy(g, m, os_, off)
    n = 16
    assert t.shape == (3, n, 3)
    s = lambda j: ((j + off) / (n - 1)) / 2
    j = 5
    expect = s(j) * (g[:, 0] + (g[:, 1] - g[:, 0]) * j / n)          # spoke 1 starts from spoke 0
    np.testing.assert_allclose(t[1, j], expect, atol=1e-15)
    np.testing.assert_allclose(t[0, j], s(j) * (g[:, -1] + (g[:, 0] - g[:, -1]) * j / n), atol=1e-15)
    np.testing.assert_allclose(t[-1, j], s(j) * g[:, -1], atol=1e-15)   # last projection uncorrected


def test_origin_scans_are_refused():
    info = _recon_info()
    info["UseOrigin"] = True
    with pytest.raises(ValueError):
        legacytraj.legacy_from_recon_info(info)
