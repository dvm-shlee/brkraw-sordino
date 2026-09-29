"""Tests for tools/eval_ramp.py helpers (no dataset needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import eval_ramp as er  # noqa: E402
from brkraw_sordino import traj as product_traj  # noqa: E402


def _recon_info(matrix=16, os_=2.0):
    npro = 2 * product_traj.calc_npro(matrix, 1.0)
    return {
        "Matrix": [matrix] * 3, "NPro": npro, "HalfAcquisition": False,
        "UseOrigin": False, "Reorder": False, "OverSampling": os_,
        "AcqDelayTotal_us": 6.75, "EffBandwidth_Hz": 75000.0,
    }


def test_integral_differs_from_product_only_in_ramp_term():
    info = _recon_info()
    matrix, npro, os_ = 16, info["NPro"], info["OverSampling"]
    g = product_traj.calc_radial_grad3d(matrix, npro, False, False, False)
    off = 6.75e-6 * 75000.0 * os_
    code = product_traj.calc_radial_traj3d(g, matrix, False, os_, True, off)
    integ = er.integral_trajectory(info)
    n = int(matrix / 2 * os_)
    j = np.arange(n, dtype=float)
    unit = 1.0 / (n - 1) / 2
    d = (g - np.roll(g, 1, axis=1)).T  # [pro, 3]
    expected_gap = unit * ((j + off) * j / n - j ** 2 / (2 * n))
    gap = code - integ
    np.testing.assert_allclose(gap[:-1], expected_gap[None, :, None] * d[:-1, None, :],
                               atol=1e-12)
    np.testing.assert_allclose(gap[-1], 0.0, atol=1e-12)  # last projection: no ramp


def test_integral_rejects_use_origin():
    info = _recon_info()
    info["UseOrigin"] = True
    with pytest.raises(ValueError):
        er.integral_trajectory(info)


def test_detect_version():
    assert er.detect_version({"MaximizeRampTime": "Yes"}) == "v3"
    assert er.detect_version({"RFWait": 0.01, "RampDelay": None}) == "v3"
    assert er.detect_version({"TrigSegmentMode": "Off", "RampDelay": 0.1}) == "v2"
    assert er.detect_version({"RampDelay": 0.04}) == "v1"
    assert er.detect_version({}) is None


def test_tau_models():
    meta = {"AcqDelayTotal": 6.75, "PVM_EffSWh": 75000.0, "OverSampling": 8.0,
            "NPoints": 256.0, "RampTime": 0.43641666666666667,
            "ExcPulLength_ms": 0.004, "RampDelay": 0.1645833333333333}
    dwell = 1e6 / 600000.0
    code = er.current_code_tau_us(meta, 4)
    assert code == pytest.approx([(6.75 + k * dwell) * (1 - k / 256) for k in range(4)])
    # v2/v3: ramp starts at RF end (2 us after RF centre): tau = t - (t-2)^2/(2 T)
    integ = er.model_tau_us(meta, "v2", 3)
    t = np.array([6.75 + k * dwell for k in range(3)])
    assert integ == pytest.approx(list(t - (t - 2.0) ** 2 / (2 * 436.41666666666667)), abs=1e-3)
    # v1: the ramp is already running at the RF, so tau grows slower than t
    v1 = er.model_tau_us(meta, "v1", 3)
    assert v1[2] - v1[0] < t[2] - t[0]


def test_config_check():
    with pytest.raises(ValueError):
        er.EvalConfig(dataset="x", scan_id=1, out_dir="y", sample_index=9, n_keep=8).check()
