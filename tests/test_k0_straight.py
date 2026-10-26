"""estimate_k0 on the straight trajectory (WI-0137, D-0234).

``correct_ramptime=false`` and a general ZTE (no RampTime) put every sample of a spoke on
one straight line, ``k = (j + offset) / (2 (N - 1)) g`` (``traj.calc_radial_traj3d``). The
virtual leading samples of ``estimate_k0`` continue that line back to the RF centre
(``kcentre.leading_points(straight=True)``); the solve is the unchanged BRK-0066 one.
"""
import io
import logging
from dataclasses import asdict

import numpy as np
import pytest

from brkraw_sordino import hook, kcentre
from brkraw_sordino.hook import _build_options, _resolve_k0
from brkraw_sordino.recon import build_recon_cache_path, nufft_adjoint, recon_dataobj
from brkraw_sordino.traj import calc_npro, calc_radial_traj3d, get_trajectory, spoke_directions

MATRIX, OS = 16, 4.0
SHAPE = [MATRIX] * 3
N = int(MATRIX / 2 * OS)


def _info(version="v3", gap_samples=3.0, bw=75000.0):
    """Synthetic recon_info; ``gap_samples`` = RF centre to sample 0 in samples."""
    info = {
        "Matrix": SHAPE, "NPro": 2 * calc_npro(MATRIX, 1.0), "HalfAcquisition": False,
        "UseOrigin": False, "Reorder": False, "OverSampling": OS,
        "AcqDelayTotal_us": gap_samples / (bw * OS) * 1e6, "EffBandwidth_Hz": bw,
        "ExcPulLength_ms": 0.004, "RampTime_ms": 0.3, "RampDelay_ms": None,
        "TrigSegmentMode": None, "MaximizeRampTime": "No", "RFWait_ms": 0.01,
        "O1List_Hz": [0.0], "NRepetitions": 1, "EncNReceivers": 1, "NPoints": N,
    }
    if version == "zte":
        info["RampTime_ms"], info["MaximizeRampTime"], info["RFWait_ms"] = None, None, None
    return info


def _opts(tmp_path, **kw):
    return _build_options({"cache_dir": str(tmp_path), **kw})


def _phantom():
    n = np.asarray(SHAPE, dtype=float)
    grid = np.meshgrid(*[np.arange(int(s)) - s / 2 for s in SHAPE], indexing="ij")
    r = np.sqrt(sum(((g - c * s) / (a * s)) ** 2 for g, c, a, s in
                    zip(grid, (0.04, -0.03, 0.02), (0.32, 0.27, 0.22), n)))
    img = 0.5 * (1 - np.tanh((r - 1.0) * 0.22 * n.min()))
    r2 = np.sqrt(sum(((g - c * s) / (0.06 * s)) ** 2 for g, c, s in zip(grid, (-0.12, 0.1, 0.05), n)))
    return (img + 1.5 * 0.5 * (1 - np.tanh((r2 - 1.0) * 0.06 * n.min()))).astype(np.complex64)


# ---- the virtual samples -------------------------------------------------------------------


def test_straight_points_continue_the_straight_trajectory(tmp_path):
    info = _info("v3", gap_samples=3.4)
    tr = get_trajectory(info, _opts(tmp_path, correct_ramptime=False))
    vt = kcentre.leading_points(info, 1, straight=True)
    grad, _ = spoke_directions(info)
    # samples j = 1 - m + 3.4 >= 0 for m = 1..4 -> indices 0.4, 1.4, 2.4, 3.4 (sample 0 dropped)
    assert vt.shape == (info["NPro"], 4, 3)
    line = np.concatenate([vt, tr[:, 1:]], axis=1)                   # virtual + kept, one line per spoke
    step = np.diff(line, axis=1)
    np.testing.assert_allclose(step, np.broadcast_to(step[:, :1], step.shape), atol=1e-15)
    # every point is a multiple of the spoke's own vector, the first one 0.4 samples from the RF centre
    unit = 1.0 / (N - 1) / 2.0
    np.testing.assert_allclose(vt[:, 0], 0.4 * unit * grad.T, atol=1e-15)
    np.testing.assert_allclose(vt[:, 3], tr[:, 0], atol=1e-15)       # the dropped sample 0


def test_general_zte_gets_the_same_points_either_way():
    info = _info("zte", gap_samples=5.33)
    # the ramp-corrected formula of a general ZTE is the constant gradient: the same line
    np.testing.assert_allclose(kcentre.leading_points(info, 1), kcentre.leading_points(info, 1, straight=True),
                               rtol=0, atol=1e-15)


def test_straight_points_differ_from_the_ramp_points_on_sordino():
    info = _info("v3", gap_samples=3.4)
    assert not np.allclose(kcentre.leading_points(info, 1), kcentre.leading_points(info, 1, straight=True))


def test_hook_chooses_the_points_from_correct_ramptime(tmp_path):
    info = _info("v3", gap_samples=3.4)
    np.testing.assert_array_equal(hook._leading_points(info, _opts(tmp_path)), kcentre.leading_points(info, 1))
    np.testing.assert_array_equal(hook._leading_points(info, _opts(tmp_path, correct_ramptime=False)),
                                  kcentre.leading_points(info, 1, straight=True))


def test_straight_points_need_only_the_straight_values(tmp_path):
    info = _info("v3", gap_samples=3.4)
    for key in ("ExcPulLength_ms", "RampTime_ms", "MaximizeRampTime", "RFWait_ms"):
        info.pop(key)
    o = _resolve_k0(_opts(tmp_path, correct_ramptime=False), info)
    assert o.estimate_k0 is True and kcentre.leading_points(info, 1, straight=True).shape[1] == 4
    info.pop("AcqDelayTotal_us")
    with pytest.raises(hook.K0TimingError, match="AcqDelayTotal"):
        _resolve_k0(_opts(tmp_path, correct_ramptime=False), info)


def test_gap_radius_matches_the_timing_metadata():
    from brkraw_sordino import timing

    for version, gap in (("v3", 3.4), ("zte", 5.33)):
        info = _info(version, gap_samples=gap)
        seq = timing.read_timing(info)
        for ignore in (0, 1, 2):
            assert kcentre.gap_used_kgrid(info, ignore) == pytest.approx(
                timing.kspace_gap(seq, OS, ignore)["gap_used_kgrid"], abs=1e-12)
            assert kcentre.gap_used_kgrid(info, ignore) == pytest.approx((gap + ignore) / OS, abs=1e-12)


# ---- the fill on synthetic data -------------------------------------------------------------


@pytest.mark.parametrize("version, gap_samples", [("v3", 3.0), ("zte", 3.0), ("zte", 4.0)])
def test_fill_restores_the_centre_on_straight_data(tmp_path, version, gap_samples):
    """Data made on the straight line: the filled image is close to the no-dead-time image."""
    info = _info(version, gap_samples=gap_samples)
    options = _opts(tmp_path, correct_ramptime=False)
    assert options.estimate_k0 is True
    tr = get_trajectory(info, options)
    img = _phantom()
    y = np.asarray(kcentre._operator(tr, SHAPE).op(img)).reshape(tr.shape[:2])
    vt = hook._leading_points(info, options)
    true_v = np.asarray(kcentre._operator(vt, SHAPE).op(img)).reshape(vt.shape[:2])
    ref = nufft_adjoint(np.concatenate([true_v, y[:, 1:]], 1), np.concatenate([vt, tr[:, 1:]], 1), SHAPE, 1)
    plain = nufft_adjoint(y[:, 1:], tr[:, 1:], SHAPE, 1)
    filled, meta = kcentre.fill_centre(y[:, 1:], tr[:, 1:], vt, SHAPE)
    err_plain = np.linalg.norm(plain - ref) / np.linalg.norm(ref)
    err_fill = np.linalg.norm(filled - ref) / np.linalg.norm(ref)
    true_k0 = complex(np.asarray(kcentre._operator(np.zeros((1, 1, 3)), SHAPE).op(img)).reshape(-1)[0])
    # gap 1.0 / 1.25 k-grid units: the plain adjoint misses 40-60 % of the image, the fill < 1 %
    assert err_plain > 0.3 and err_fill < 0.01, (err_plain, err_fill)
    assert abs(meta["k0"] - true_k0) < 0.01 * abs(true_k0)


def test_product_loop_uses_the_straight_points(tmp_path):
    """recon_dataobj with the hook's straight points equals fill_centre on the same arrays."""
    info = _info("zte", gap_samples=3.0)
    options = _opts(tmp_path, correct_ramptime=False)
    tr = get_trajectory(info, options)
    y = np.asarray(kcentre._operator(tr, SHAPE).op(_phantom())).reshape(tr.shape[:2])
    raw = np.stack([y.real.T, y.imag.T])[:, :, None, :].astype("<f4").tobytes(order="F")
    info["FIDDataType"] = np.dtype("<f4")
    vt = hook._leading_points(info, options)
    k0s: list = []
    out = io.BytesIO()
    dtype = recon_dataobj(io.BytesIO(raw), tr, info, out, options, virtual_traj=vt, k0_out=k0s)
    got = np.frombuffer(out.getvalue(), dtype=dtype).reshape(SHAPE[::-1]).T
    ref, meta = kcentre.fill_centre(y[:, 1:].astype(np.complex64), tr[:, 1:], vt, SHAPE)
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-4
    assert abs(k0s[0][0] - meta["k0"]) < 1e-4 * abs(meta["k0"])


# ---- gap warning ------------------------------------------------------------------------------


def test_wide_gap_warns_once_and_narrow_gap_does_not(tmp_path, caplog):
    assert kcentre.GAP_WARN_KGRID == 2.0
    options = _opts(tmp_path, correct_ramptime=False)
    info = _info("zte", gap_samples=8.0)                                   # (8 + 1) / 4 = 2.25
    hook._set_k0_iterations(info, options)
    with caplog.at_level(logging.DEBUG, logger="brkraw_sordino.hook"):
        hook._warn_centre_gap(info, options)
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1 and "2.25 k-grid units" in warns[0].getMessage()
    assert "160 iterations" in warns[0].getMessage()
    for gap_samples, n_iter in ((5.33, 40), (3.0, None)):                  # 1.58: info line; 1.0: debug only
        caplog.clear()
        info = _info("zte", gap_samples=gap_samples)
        hook._set_k0_iterations(info, options)
        with caplog.at_level(logging.DEBUG, logger="brkraw_sordino.hook"):
            hook._warn_centre_gap(info, options)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert infos == ([] if n_iter is None else [
            f"estimate_k0: unsampled centre radius 1.58 k-grid units, {n_iter} iterations (10 up to 1.3)."])


# ---- iterations by gap (WI-0137 run 2) --------------------------------------------------------


def test_iterations_by_gap_rule():
    assert kcentre.N_ITER == 10
    for gap, n_iter in ((0.5, 10), (1.0, 10), (1.3, 10), (1.31, 40), (1.58, 40), (1.8, 40), (1.81, 160),
                        (2.0, 160), (3.0, 160)):
        assert kcentre.iterations_for_gap(gap) == n_iter, gap


def test_iterations_are_keyed_only_for_a_wide_gap(tmp_path):
    """recon_info is in the recon cache key: K0Iterations is written only when it is not N_ITER."""
    for version, correct in (("v3", True), ("v3", False), ("zte", True)):
        narrow = _info(version, gap_samples=3.0)                            # 1.0 k-grid units
        before = dict(narrow)
        hook._set_k0_iterations(narrow, _opts(tmp_path, correct_ramptime=correct))
        assert narrow == before and kcentre.k0_iterations(narrow) == 10
        assert (build_recon_cache_path(tmp_path, {"recon_info": narrow})
                == build_recon_cache_path(tmp_path, {"recon_info": before}))
    wide = _info("zte", gap_samples=5.33)
    plain = dict(wide)
    hook._set_k0_iterations(wide, _opts(tmp_path, correct_ramptime=False))
    assert wide["K0Iterations"] == 40 and kcentre.k0_iterations(wide) == 40
    assert (build_recon_cache_path(tmp_path, {"recon_info": wide})
            != build_recon_cache_path(tmp_path, {"recon_info": plain}))
    # estimate_k0 off: nothing written (the key of the plain adjoint is unchanged)
    off = _info("zte", gap_samples=5.33)
    hook._set_k0_iterations(off, _opts(tmp_path, estimate_k0=False))
    assert "K0Iterations" not in off
    # a second plan on the same dict follows the options again
    hook._set_k0_iterations(wide, _opts(tmp_path, estimate_k0=False))
    assert "K0Iterations" not in wide


def test_wide_gap_fill_needs_the_extra_iterations(tmp_path):
    """Gap 1.58 (scan 23): 10 iterations leave a K0 error, the chosen 40 restore it."""
    info = _info("zte", gap_samples=5.33)
    options = _opts(tmp_path, correct_ramptime=False)
    hook._set_k0_iterations(info, options)
    tr = get_trajectory(info, options)
    img = _phantom()
    y = np.asarray(kcentre._operator(tr, SHAPE).op(img)).reshape(tr.shape[:2])
    vt = hook._leading_points(info, options)
    true_k0 = complex(np.asarray(kcentre._operator(np.zeros((1, 1, 3)), SHAPE).op(img)).reshape(-1)[0])
    errs = {}
    for n_iter in (kcentre.N_ITER, kcentre.k0_iterations(info)):
        _, meta = kcentre.fill_centre(y[:, 1:], tr[:, 1:], vt, SHAPE, n_iter)
        errs[n_iter] = abs(meta["k0"] - true_k0) / abs(true_k0)
    assert errs[40] < 0.001 and errs[10] > 10 * errs[40], errs


def test_product_loop_uses_the_chosen_iterations(tmp_path):
    """recon_dataobj reads recon_info["K0Iterations"]: equal to fill_centre with 40 iterations."""
    info = _info("zte", gap_samples=5.33)
    options = _opts(tmp_path, correct_ramptime=False)
    hook._set_k0_iterations(info, options)
    assert info["K0Iterations"] == 40
    tr = get_trajectory(info, options)
    y = np.asarray(kcentre._operator(tr, SHAPE).op(_phantom())).reshape(tr.shape[:2])
    raw = np.stack([y.real.T, y.imag.T])[:, :, None, :].astype("<f4").tobytes(order="F")
    info["FIDDataType"] = np.dtype("<f4")
    vt = hook._leading_points(info, options)
    k0s: list = []
    out = io.BytesIO()
    dtype = recon_dataobj(io.BytesIO(raw), tr, info, out, options, virtual_traj=vt, k0_out=k0s)
    got = np.frombuffer(out.getvalue(), dtype=dtype).reshape(SHAPE[::-1]).T
    ref, meta = kcentre.fill_centre(y[:, 1:].astype(np.complex64), tr[:, 1:], vt, SHAPE, 40)
    ref10, _ = kcentre.fill_centre(y[:, 1:].astype(np.complex64), tr[:, 1:], vt, SHAPE, 10)
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-4
    assert np.linalg.norm(got - ref10) / np.linalg.norm(ref10) > 1e-3
    assert abs(k0s[0][0] - meta["k0"]) < 1e-4 * abs(meta["k0"])


def test_zte_trajectory_log_says_the_centre_is_filled(tmp_path, caplog):
    info = _info("zte", gap_samples=5.33)
    with caplog.at_level(logging.INFO, logger="brkraw_sordino.traj"):
        get_trajectory(info, _opts(tmp_path))
    assert [r for r in caplog.records if "estimate_k0 fills the centre" in r.getMessage()]
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="brkraw_sordino.traj"):
        get_trajectory(info, _opts(tmp_path / "off", estimate_k0=False))
    assert [r for r in caplog.records if "the centre is not filled" in r.getMessage()]


# ---- cache keys -------------------------------------------------------------------------------


def _key(tmp_path, options):
    keyed = {k: v for k, v in asdict(options).items()
             if k not in hook.RECON_KEY_EXCLUDED and k not in ("golden_weights", "recon_matrix")
             and k not in ("frame_spokes", "frame_step", "frame_accumulate")}
    return build_recon_cache_path(tmp_path, {"options": keyed})


def test_new_defaults_never_read_the_old_no_k0_caches(tmp_path):
    """Before WI-0137 the ramp-off default and a general ZTE were keyed with estimate_k0 false."""
    new_straight = _opts(tmp_path, correct_ramptime=False)
    old_straight = _opts(tmp_path, correct_ramptime=False, estimate_k0=False)
    assert new_straight.estimate_k0 is True
    assert _key(tmp_path, new_straight) != _key(tmp_path, old_straight)
    new_zte = _resolve_k0(_opts(tmp_path), _info("zte"))
    old_zte = _opts(tmp_path, estimate_k0=False)                          # what the switch-off keyed
    assert _key(tmp_path, new_zte) != _key(tmp_path, old_zte)
    # the straight K0 result and the ramp-corrected K0 result do not share a key either
    assert _key(tmp_path, new_straight) != _key(tmp_path, _opts(tmp_path))


def test_options_fields_are_unchanged(tmp_path, monkeypatch):
    """No new option field: every other recon cache key is the one written before WI-0137."""
    monkeypatch.setattr(hook, "_get_cache_dir", lambda path: tmp_path)
    assert sorted(asdict(_build_options({}))) == sorted([
        "ext_factors", "ignore_samples", "offset", "num_frames", "correct_spoketiming", "correct_ramptime",
        "offreso_freqs", "mem_limit", "clear_cache", "split_ch", "cache_dir", "as_complex", "estimate_k0",
        "frame_spokes", "frame_step", "frame_accumulate", "golden_weights", "recon_matrix",
        "peri_event", "peri_censor", "subtract_rep_mean", "motion_file",
        "motion_spokes"])
