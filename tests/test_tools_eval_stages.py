"""Tests for tools/eval_stages.py (staged comparison; no dataset needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import eval_ramp as er  # noqa: E402
import eval_stages as es  # noqa: E402
from brkraw_sordino import timing, traj as product_traj  # noqa: E402
from brkraw_sordino.recon import phase_correction_factor  # noqa: E402


def _recon_info(matrix=16, os_=2.0, n_o1=True):
    npro = 2 * product_traj.calc_npro(matrix, 1.0)
    rng = np.random.default_rng(1)
    return {
        "Matrix": [matrix] * 3, "NPro": npro, "HalfAcquisition": False,
        "UseOrigin": False, "Reorder": False, "OverSampling": os_,
        "AcqDelayTotal_us": 6.75, "EffBandwidth_Hz": 75000.0,
        "ExcPulLength_ms": 0.004, "RampTime_ms": 0.1164, "RampDelay_ms": 0.16,
        "TrigSegmentMode": "Off", "MaximizeRampTime": None, "RFWait_ms": None,
        "O1List_Hz": (rng.normal(0, 300, npro).tolist() if n_o1 else [0.0]),
        "NRepetitions": 1, "EncNReceivers": 1, "NPoints": int(matrix / 2 * os_),
    }


def test_seven_stages_and_lookup():
    assert es.STAGE_NAMES == ("S0", "S1", "S1p", "S1h", "S1hp", "S2", "S3")
    assert es.stage("S3").traj == "integral" and es.stage("S3").phase == "integral"
    assert es.stage("S0").phase == "none" and es.stage("S1h").traj == "legacy_half"
    with pytest.raises(ValueError):
        es.stage("S9")


def test_legacy_curvature_one_equals_product(tmp_path):
    info = _recon_info()
    ref = er.trajectory(info, "pre", tmp_path)[0]
    np.testing.assert_allclose(es.legacy_trajectory(info, 1.0), ref, atol=1e-15)
    np.testing.assert_allclose(es.stage_trajectory(info, "legacy", tmp_path), ref, atol=1e-15)


def test_legacy_half_is_midway_between_legacy_and_previous_vector(tmp_path):
    info = _recon_info()
    full = es.legacy_trajectory(info, 1.0)
    half = es.legacy_trajectory(info, 0.5)
    g = es.gradient_vectors(info).T
    n = full.shape[1]
    j = np.arange(n)
    s = ((j + es._offset_samples(info)) / (n - 1)) / 2
    prev_const = s[None, :, None] * np.roll(g, 1, axis=0)[:, None, :]
    # 2 * half - full == s_j g(i-1) for every projection but the last (uncorrected)
    np.testing.assert_allclose((2 * half - full)[:-1], prev_const[:-1], atol=1e-15)
    np.testing.assert_allclose(half[-1], full[-1])
    assert not np.allclose(half, full)
    # S0 and S2 come from the product
    off = es.stage_trajectory(info, "off", tmp_path)
    np.testing.assert_allclose(off, er.trajectory(info, "off", tmp_path)[0])
    np.testing.assert_allclose(es.stage_trajectory(info, "integral", tmp_path),
                               er.trajectory(info, "post_traj", tmp_path)[0])


def test_implied_phase_models(tmp_path):
    info = _recon_info()
    n = int(info["Matrix"][0] / 2 * info["OverSampling"])
    seq = timing.read_timing(info)
    t, _, tau_int = timing.ramp_terms(seq, timing.tuning_for(seq.version), n)
    t = np.asarray(t)
    assert es.implied_tau_us(info, "none", n) is None
    np.testing.assert_allclose(es.implied_tau_us(info, "integral", n), tau_int)
    np.testing.assert_allclose(es.implied_tau_us(info, "legacy", n), t * (1 - np.arange(n) / n))
    np.testing.assert_allclose(es.implied_tau_us(info, "legacy_half", n), t * (1 - np.arange(n) / (2 * n)))
    # factors: integral == product; legacy sign and last projection
    prod = phase_correction_factor(info, er.mode_options("post", tmp_path), n)
    np.testing.assert_allclose(es.stage_phase_factor(info, "integral", n, tmp_path), prod)
    o1 = np.asarray(info["O1List_Hz"])
    d = np.roll(o1, 1) - o1
    leg = es.stage_phase_factor(info, "legacy", n, tmp_path)
    assert leg.shape == (info["NPro"], n)
    np.testing.assert_allclose(leg[:-1, 0], np.exp(-2j * np.pi * d[:-1] * t[0] * 1e-6), rtol=1e-5)
    np.testing.assert_allclose(leg[-1], 1.0)
    assert es.stage_phase_factor(info, "none", n, tmp_path) is None
    # no FOV offset: nothing to correct
    assert es.stage_phase_factor(_recon_info(n_o1=False), "legacy", n, tmp_path) is None


def test_first_peak_k_position(tmp_path):
    info = _recon_info()
    g = es.gradient_vectors(info)
    off = es.stage_trajectory(info, "off", tmp_path)
    n = off.shape[1]
    kp = es.first_peak_k(off, g, 1, 16)
    expect = ((1 + es._offset_samples(info)) / (n - 1)) / 2 * 16
    np.testing.assert_allclose(kp["radius_kgrid"], expect)
    assert kp["angle_max_deg"] < 1e-3   # arccos rounding near 1
    kp2 = es.first_peak_k(es.stage_trajectory(info, "integral", tmp_path), g, 1, 16)
    assert kp2["angle_max_deg"] > 1.0 and kp2["radius_max"] > kp2["radius_min"]


def test_circular_helpers_match_lee_module():
    import circstats

    rng = np.random.default_rng(3)
    z = rng.normal(size=(40, 3)) + 1j * rng.normal(size=(40, 3))
    ours = es._circular_std(z)
    for col in range(3):
        st = circstats.circular_stats(np.angle(z[:, col]).tolist())
        assert ours[col] == pytest.approx(st["std"], rel=1e-9)
    coh = es._coherence(z)
    assert np.all((coh >= 0) & (coh <= 1))


def test_stage_config_check():
    with pytest.raises(ValueError):
        es.StageConfig(dataset="x", scan_id=1, out_dir="y", stages=("S0", "S9")).check()
    with pytest.raises(ValueError):
        es.StageConfig(dataset="x", scan_id=1, out_dir="y", sample_index=40, phase_samples=32).check()


def test_simulation_small(tmp_path):
    info = _recon_info(matrix=16, os_=2.0)
    res = es.simulate(info, tmp_path, stages=("S0", "S1", "S2", "S3"))
    st = res["stages"]
    assert st["S3"]["rel_error_vs_ideal"] < 1e-3
    assert st["S0"]["rel_error_vs_ideal"] > st["S3"]["rel_error_vs_ideal"]
    assert st["S2"]["rel_error_vs_ideal"] > st["S3"]["rel_error_vs_ideal"]
    assert res["images"]["S3"].shape == (16, 16, 16)


def test_stage_table_uses_every_stage():
    summary = {
        "stages": {n: {} for n in ("S0", "S3")},
        "a_magnitude_cv_median": 0.1,
        "a_phase": {"none": {"coherence_median": 0.9, "circular_std_median": 0.5, "coherence_vs_sample": [0.9, 0.8],
                             "coherence_gain_median": 0.0, "coherence_gain_vs_sample": [0.0, 0.0],
                             "correction_phase_first_peak_abs_max_rad": 0.0},
                    "integral": {"coherence_median": 0.95, "circular_std_median": 0.4, "coherence_vs_sample": [0.95, 0.9],
                                 "coherence_gain_median": 5e-5, "coherence_gain_vs_sample": [5e-5, 1e-4],
                                 "correction_phase_first_peak_abs_max_rad": 0.03}},
        "k_first_peak": {"off": {"radius_median": 0.6, "angle_median_deg": 0.0, "displacement_median": 0.0},
                         "integral": {"radius_median": 0.61, "angle_median_deg": 3.0, "displacement_median": 0.04}},
        "b_magnitude": {"pattern_stability_r": {"min": 0.99}, "timecourse_oscillation": {"rel_std": 0.002}},
        "b_phase": {"none": {"coherence_timecourse_oscillation": {"rel_std": 0.001}},
                    "integral": {"coherence_timecourse_oscillation": {"rel_std": 0.001}}},
        "c": {"S0": {"roi_oscillation": {"rel_std": 0.0045}, "sharpness": 0.28, "rel_diff_vs_S3": 0.1},
              "S3": {"roi_oscillation": {"rel_std": 0.0044}, "sharpness": 0.27}},
    }
    table = es.stage_table(summary)
    lines = table.splitlines()
    assert lines[0] == "| metric | S0 | S3 |"
    assert "| (c) mean image rel. diff vs S3 | 10.0% | - |" in lines
    assert "| (a) |first peak| CV across spokes | 0.1000 | 0.1000 |" in lines
