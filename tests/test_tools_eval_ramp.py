"""Tests for tools/eval_ramp.py helpers (no dataset needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import eval_ramp as er  # noqa: E402
import legacytraj  # noqa: E402
from brkraw_sordino import traj as product_traj  # noqa: E402


def _recon_info(matrix=16, os_=2.0):
    npro = 2 * product_traj.calc_npro(matrix, 1.0)
    return {
        "Matrix": [matrix] * 3, "NPro": npro, "HalfAcquisition": False,
        "UseOrigin": False, "Reorder": False, "OverSampling": os_,
        "AcqDelayTotal_us": 6.75, "EffBandwidth_Hz": 75000.0,
        "ExcPulLength_ms": 0.004, "RampTime_ms": 0.1164, "RampDelay_ms": 0.16,
        "TrigSegmentMode": "Off", "MaximizeRampTime": None, "RFWait_ms": None,
        "O1List_Hz": [0.0],
    }


def test_mode_options_map_to_product_settings(tmp_path):
    for mode, ramp_ok in {"off": False, "pre": True, "post_traj": True, "post": True}.items():
        o = er.mode_options(mode, tmp_path)
        assert o.correct_ramptime is ramp_ok
        assert not hasattr(o, "ramp_model") and not hasattr(o, "correct_phase")
    with pytest.raises(ValueError):
        er.mode_options("code", tmp_path)


def test_pre_mode_is_the_previous_trajectory(tmp_path):
    info = _recon_info()
    matrix, npro, os_ = 16, info["NPro"], info["OverSampling"]
    g = product_traj.calc_radial_grad3d(matrix, npro, False, False, False)
    off = 6.75e-6 * 75000.0 * os_
    ref = legacytraj.calc_radial_traj3d_legacy(g, matrix, os_, off)
    traj, phase = er.trajectory(info, "pre", tmp_path)
    np.testing.assert_array_equal(traj, ref)
    assert phase is None
    post, post_phase = er.trajectory(info, "post", tmp_path)
    assert post.shape == ref.shape and not np.allclose(post, ref)
    assert post_phase is None          # O1 list of length 1: no FOV offset
    off_traj, off_phase = er.trajectory(info, "off", tmp_path)
    np.testing.assert_array_equal(off_traj, product_traj.calc_radial_traj3d(g, matrix, os_, off))
    assert off_phase is None


def test_post_traj_has_no_phase_factor_but_post_has(tmp_path):
    info = _recon_info()
    g = product_traj.calc_radial_grad3d(16, info["NPro"], False, False, False)
    info["O1List_Hz"] = (1000.0 * g[0] - 500.0 * g[1] + 250.0 * g[2]).tolist()
    info["TrigSegmentMode"] = "Off"
    t_traj, p_traj = er.trajectory(info, "post_traj", tmp_path)
    t_post, p_post = er.trajectory(info, "post", tmp_path)
    np.testing.assert_array_equal(t_traj, t_post)
    assert p_traj is None and p_post is not None and p_post.shape[0] == info["NPro"]


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
