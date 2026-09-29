"""Tests for tools/eval_timing_centre.py and the WI-0058 stages (no dataset needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import eval_ramp as er  # noqa: E402
import eval_stages as es  # noqa: E402
import eval_timing_centre as etc  # noqa: E402
from brkraw_sordino import traj as product_traj  # noqa: E402
from brkraw_sordino.recon import phase_correction_factor  # noqa: E402


def _recon_info(matrix=16, os_=2.0, o1_sd=300.0, version="v2"):
    npro = 2 * product_traj.calc_npro(matrix, 1.0)
    rng = np.random.default_rng(1)
    info = {
        "Matrix": [matrix] * 3, "NPro": npro, "HalfAcquisition": False,
        "UseOrigin": False, "Reorder": False, "OverSampling": os_,
        "AcqDelayTotal_us": 6.75, "EffBandwidth_Hz": 75000.0,
        "ExcPulLength_ms": 0.004, "RampTime_ms": 0.1164, "RampDelay_ms": 0.16,
        "TrigSegmentMode": "Off", "MaximizeRampTime": None, "RFWait_ms": None,
        "O1List_Hz": rng.normal(0, o1_sd, npro).tolist(),
        "NRepetitions": 1, "EncNReceivers": 1, "NPoints": int(matrix / 2 * os_),
    }
    if version == "v1":
        info.pop("TrigSegmentMode")
    return info


def _physical_o1(info, amp_hz=20000.0, r0=(0.6, -0.3, 0.74)):
    """O1_i = K g_i . r0, the relation the real scans follow (WI-0056 run 2)."""
    g = es.gradient_vectors(info)
    info = dict(info)
    info["O1List_Hz"] = (amp_hz * (g.T @ np.asarray(r0))).tolist()
    return info


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_zero_delay_equals_product(tmp_path, version):
    info = _recon_info(version=version)
    n = int(info["Matrix"][0] / 2 * info["OverSampling"])
    np.testing.assert_allclose(etc.delayed_trajectory(info, 0.0),
                               er.trajectory(info, "post_traj", tmp_path)[0], atol=1e-15)
    np.testing.assert_allclose(etc.delayed_phase_factor(info, 0.0, n),
                               phase_correction_factor(info, er.mode_options("post", tmp_path), n))


def test_delay_shifts_sample_times_and_first_radius():
    info = _recon_info()
    t0, _, _ = etc.tuned_terms(info, 0.0)
    t1, _, _ = etc.tuned_terms(info, 2.5)
    np.testing.assert_allclose(np.asarray(t1) - np.asarray(t0), 2.5)
    r0 = np.linalg.norm(etc.delayed_trajectory(info, 0.0)[:, 1], axis=1)
    r1 = np.linalg.norm(etc.delayed_trajectory(info, 2.5)[:, 1], axis=1)
    assert np.all(r1 > r0)


def test_estimate_recovers_known_delay():
    info = _recon_info(matrix=24, os_=4.0, o1_sd=2000.0)
    n = 13
    d = etc.o1_steps(info)
    rng = np.random.default_rng(3)
    amp = 1.0 + 0.2 * rng.random(int(info["NPro"]))
    for true in (-2.5, 0.0, 2.4):
        _, _, tau = etc.tuned_terms(info, true, n)
        z = amp[:, None] * np.exp(2j * np.pi * np.outer(d, np.asarray(tau) * 1e-6))
        est = etc.estimate_delay(z, info, range(1, 13), span_us=6.0, step_us=0.1)
        assert abs(est["delta_us"] - true) < 0.02, (true, est["delta_us"])
        assert est["objective"] > est["objective_at_0"] - 1e-12
        assert abs(est["per_sample_weighted_mean_us"] - true) < 0.05


def test_estimate_needs_o1_list():
    info = _recon_info()
    info["O1List_Hz"] = [0.0]
    with pytest.raises(ValueError):
        etc.estimate_delay(np.ones((int(info["NPro"]), 4), complex), info, [1, 2])


def test_centre_points_end_at_first_kept_sample():
    info = _recon_info()
    cp = etc.centre_points(info, 1.5, fractions=(0.0, 1.0))
    np.testing.assert_allclose(cp[:, 0], 0.0, atol=1e-15)
    np.testing.assert_allclose(cp[:, 1], etc.delayed_trajectory(info, 1.5)[:, 1], atol=1e-14)


def test_cg_beats_adjoint_on_model_data():
    info = _recon_info(matrix=16, os_=4.0)
    shape = [16] * 3
    img, _ = es.phantom(shape)
    tr = etc.delayed_trajectory(info, 0.0)
    y = etc._operator(tr, shape).op(img).reshape(tr.shape[:2])
    x0, info0 = etc.cg_reconstruct(y, tr, shape, 1, None, n_iter=0, return_info=True)
    assert not np.any(x0)
    x, inf = etc.cg_reconstruct(y, tr, shape, 1, None, n_iter=15, return_info=True)
    assert inf["residual_norms"][-1] < 0.1 * inf["residual_norms"][0]
    adj = er.reconstruct(y, tr, shape, 1)
    c = inf["adjoint_scale"]

    def err(a):
        return np.linalg.norm(a - img) / np.linalg.norm(img)

    assert err(x * c) < err(adj * c)


def test_wi0058_stages_and_config():
    assert es.STAGE_NAMES == ("S0", "S1", "S1p", "S1h", "S1hp", "S2", "S3")
    assert es.ALL_STAGE_NAMES[-4:] == ("S4p", "S4", "S4z", "S3z")
    assert es.stage("S3z").recon == "fill" and es.stage("S3z").traj == "integral"
    es.StageConfig(dataset="x", scan_id=1, out_dir="y", stages=("S3", "S3z")).check()   # no delay needed
    assert es.stage("S4z").recon == "fill" and es.stage("S4").traj == "integral_delay"
    assert es.stage("S4p").traj == "integral" and es.stage("S4p").phase == "integral_delay"
    with pytest.raises(ValueError):
        es.StageConfig(dataset="x", scan_id=1, out_dir="y", stages=("S3", "S4")).check()
    with pytest.raises(ValueError):
        es.StageConfig(dataset="x", scan_id=1, out_dir="y", stages=("S4",), delay_phase_us=1.0).check()
    es.StageConfig(dataset="x", scan_id=1, out_dir="y", stages=("S4p",), delay_phase_us=1.0).check()
    es.StageConfig(dataset="x", scan_id=1, out_dir="y", stages=("S3", "S4", "S4z"),
                   delay_traj_us=1.0, delay_phase_us=-1.0).check()


def test_stage_builders_with_delay(tmp_path):
    info = _recon_info()
    n = int(info["Matrix"][0] / 2 * info["OverSampling"])
    np.testing.assert_allclose(es.stage_trajectory(info, "integral_delay", tmp_path, 2.0),
                               etc.delayed_trajectory(info, 2.0))
    np.testing.assert_allclose(es.stage_phase_factor(info, "integral_delay", n, tmp_path, 2.0),
                               etc.delayed_phase_factor(info, 2.0, n))


def test_coherence_estimate_is_biased_by_object_position():
    """The premise of the consistency estimator: an off-centre object's own
    spoke phase shifts the O1-step coherence estimate (true delay 0)."""
    info = _physical_o1(_recon_info(matrix=16, os_=4.0))
    shape = [16] * 3
    tr = etc.delayed_trajectory(info, 0.0)
    ph = etc.delayed_phase_factor(info, 0.0, tr.shape[1])
    y = etc._operator(tr, shape).op(etc.smooth_phantom(shape)).reshape(tr.shape[:2]) / ph
    est = etc.estimate_delay(y[:, :13], info, range(1, 13))
    assert abs(est["delta_us"]) > 1.0


def test_reduce_problem():
    info = _recon_info(matrix=16, os_=4.0)
    tr = etc.delayed_trajectory(info, 0.0)
    y = np.ones(tr.shape[:2], complex)
    y2, t2, p2, shape2 = etc.reduce_problem(y, tr, None, [16] * 3, factor=2, spoke_stride=3)
    assert shape2 == [8, 8, 8] and p2 is None
    assert t2.shape[0] == len(range(0, tr.shape[0], 3)) and y2.shape == t2.shape[:2]
    assert np.linalg.norm(t2, axis=-1).max() <= 0.5 + 1e-12
    assert t2.shape[1] < tr.shape[1]


def test_consistency_separates_trajectory_and_phase_delays():
    info = _physical_o1(_recon_info(matrix=16, os_=4.0))
    shape = [16] * 3
    tr = etc.delayed_trajectory(info, 2.4)
    ph = etc.delayed_phase_factor(info, 0.0, tr.shape[1])
    y = etc._operator(tr, shape).op(etc.smooth_phantom(shape)).reshape(tr.shape[:2]) / ph
    est = etc.estimate_timing(y.astype(np.complex64), info, shape, np.arange(-4.0, 4.01, 1.0), n_iter=20)
    assert abs(est["delta_traj_us"] - 2.4) < 0.2, est["delta_traj_us"]
    assert abs(est["delta_phase_us"]) < 0.3, est["delta_phase_us"]
    assert est["residual_at_estimate"] < est["residual_at_0"]


def test_cg_ext_returns_central_fov():
    info = _recon_info(matrix=16, os_=4.0)
    tr = etc.delayed_trajectory(info, 0.0)
    y = etc._operator(tr, [16] * 3).op(etc.smooth_phantom([16] * 3)).reshape(tr.shape[:2])
    x = etc.cg_reconstruct(y, tr, [16] * 3, 1, None, n_iter=3, ext=2)
    assert x.shape == (16, 16, 16)


def test_out_of_fov_signal_biases_1x_grid_only():
    """Signal outside the nominal FOV (radial ZTE receives it) pulls the
    trajectory estimate on a 1x grid; a 2x grid removes the bias."""
    info = _physical_o1(_recon_info(matrix=16, os_=4.0))
    big = np.zeros((32, 32, 32), np.complex64)
    big[8:24, 8:24, 8:24] = etc.smooth_phantom([16] * 3)
    big[2:8, 12:20, 12:20] += 1.0          # outside the central 16^3 FOV
    tr = etc.delayed_trajectory(info, 0.0)
    ph = etc.delayed_phase_factor(info, 0.0, tr.shape[1])
    y = (etc._operator(tr, [32] * 3).op(big).reshape(tr.shape[:2]) / ph).astype(np.complex64)
    grid = np.arange(-4.0, 4.01, 1.0)
    r1 = etc.estimate_delay_consistency(y, info, [16] * 3, grid, "traj", 20, ext=1)
    r2 = etc.estimate_delay_consistency(y, info, [16] * 3, grid, "traj", 20, ext=2)
    assert abs(r2["delta_us"]) < 0.3, r2["delta_us"]
    assert abs(r1["delta_us"]) > abs(r2["delta_us"])
    assert r2["residual_min"] < r1["residual_min"]


def test_cross_validation_resists_noise_fitting():
    """The spoke cross-validated residual stays unbiased with noise. (The
    in-sample bias seen on the real 3200-spoke geometries in WI-0058 does not
    appear at this size, so it is not asserted here.)"""
    info = _physical_o1(_recon_info(matrix=16, os_=4.0))
    shape = [16] * 3
    tr = etc.delayed_trajectory(info, 0.0)
    ph = etc.delayed_phase_factor(info, 0.0, tr.shape[1])
    y = etc._operator(tr, shape).op(etc.smooth_phantom(shape)).reshape(tr.shape[:2]) / ph
    rng = np.random.default_rng(5)
    s = 0.002 * float(np.abs(y).max())
    y = (y + s * (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape)) / np.sqrt(2)).astype(np.complex64)
    grid = np.arange(-4.0, 4.01, 1.0)
    r_cv = etc.estimate_delay_consistency(y, info, shape, grid, "traj", 20, cv=True)
    assert abs(r_cv["delta_us"]) < 0.5, r_cv["delta_us"]


def test_simulation_small():
    info = _physical_o1(_recon_info(matrix=16, os_=4.0))
    res = etc.simulate(info, 2.4, cg_iters=(10,), cons_grid_us=np.arange(-4.0, 4.01, 1.0))
    assert abs(res["estimated_traj_us"] - 2.4) < 0.2, res["estimated_traj_us"]
    assert abs(res["estimated_phase_us"] - 2.4) < 0.3, res["estimated_phase_us"]
    st = res["stages"]
    # timing: S4 reproduces the adjoint with the true timing, S3 does not
    assert st["S4"]["rel_error_vs_true_timing_adjoint"] < 0.1 * st["S3"]["rel_error_vs_true_timing_adjoint"]
    # centre: the algebraic image is close to the phantom and predicts the gap
    assert st["S4z_cg10"]["rel_error_vs_phantom"] < 0.1 * st["true_timing"]["rel_error_vs_phantom"]
    assert st["S4z_cg10"]["gap_rel_error"] < 0.1 * st["S4"]["gap_rel_error"]
    # S4z as used on real data: the adjoint with estimated leading samples is
    # much closer to the gap-free adjoint than S4 (the adjoint with the gap)
    assert (st["S4z_fill_cg10"]["rel_error_vs_no_gap_reference"]
            < 0.2 * st["S4"]["rel_error_vs_no_gap_reference"])


def test_leading_points_reach_the_centre():
    info = _recon_info(matrix=16, os_=4.0)
    vp = etc.leading_points(info, 0.0, 1)
    tr = etc.delayed_trajectory(info, 0.0)
    r_v = np.linalg.norm(vp, axis=-1)
    r_1 = np.linalg.norm(tr[:, 1], axis=-1)
    assert vp.shape[0] == tr.shape[0] and vp.shape[1] >= 1
    assert np.all(r_v.max(axis=1) < r_1) and np.all(r_v.min(axis=1) < r_1 / vp.shape[1] + 1e-12)
    # the latest virtual sample is exactly sample 0 (same time t1 - dwell)
    np.testing.assert_allclose(vp[:, -1], tr[:, 0], atol=1e-14)
