"""Sample-based density weights for golden trajectories (WI-0113 run 5, D-0197 decision 2).

The product's weight |k|^2 / max assumes straight spokes spread evenly over the
sphere. The golden lists with the ramp model bend each spoke from the previous
direction, and the Golden Grid order bends them unevenly; scan 17's streaks came
from this weight (WI-0113 run 3: synthetic data on 17's trajectory, Dice 0.75,
NRMSE 0.30 with |k|^2; 0.93 / 0.12 with Pipe-Menon weights of the actual samples).

Rule (golden modes only; Default keeps |k|^2 and its code path):

* Pipe-Menon weights of the samples of one repetition: w <- w / |interp(spread(w))|,
  10 iterations, spread/interpolate only with ``upsampfac`` 2 (mrinufft's
  ``MRIfinufft.pipe``), computed chunk by chunk so no whole trajectory is built (D-0133);
* scaled so that their sum equals the sum of the |k|^2 / max weights of the same
  samples (the brightness of the |k|^2 image);
* with ``estimate_k0`` the virtual leading samples join the set (the final adjoint is
  over virtual + measured samples) and the K0 solve uses the same weights;
* one repetition's weights serve every repetition and frame, so the frames still add
  up to the repetition image.
"""
import io

import numpy as np
import pytest

from brkraw_sordino import hook, kcentre, recon, serial
from brkraw_sordino.hook import _build_options
from brkraw_sordino.recon import phase_correction_rows, recon_dataobj
from brkraw_sordino.traj import get_trajectory, trajectory_rows

from test_golden_recon import _Entry, _Scan, _golden_info, _hook_image
from test_serial_recon import N_POINTS, SHAPE, _fid_frames, _info as _serial_info, _rel

TOL = 1e-9


def _pipe_reference(traj, shape, virtual=None):
    """mrinufft's own Pipe-Menon on the whole sample set, scaled by the product's rule."""
    from mrinufft.operators.interfaces.finufft import MRIfinufft

    meas = traj.reshape(-1, 3)
    pts = meas if virtual is None else np.concatenate([virtual, traj], axis=1).reshape(-1, 3)
    omega = np.ascontiguousarray(pts / 0.5 * np.pi, dtype=np.float64)
    w = np.abs(np.asarray(MRIfinufft.pipe(omega, tuple(shape), max_iter=10, osf=2, normalize=False)))
    w = w.reshape(traj.shape[0], -1)
    n_v = 0 if virtual is None else virtual.shape[1]
    k2 = np.square(traj).sum(-1)
    dmax = k2.max() if virtual is None else max(k2.max(), np.square(virtual).sum(-1).max())
    scale = (k2.sum() / dmax) / w[:, n_v:].sum()
    w = w * scale
    # run 5 rule: never above the |k|^2 / max weight of the same sample, then the same sum again
    k2_all = k2 if virtual is None else np.concatenate([np.square(virtual).sum(-1), k2], axis=1)
    w = np.minimum(w, k2_all.reshape(w.shape) / dmax)
    w = w * ((k2.sum() / dmax) / w[:, n_v:].sum())
    return w[:, n_v:].reshape(-1), (None if virtual is None else w[:, :n_v].reshape(-1))


def _weighted_adjoint(k, traj, shape, w):
    """``recon.nufft_adjoint`` with the weights ``w`` instead of |k|^2 / max."""
    op = recon.make_nufft_operator(traj.reshape(-1, 3) / 0.5 * np.pi, shape, np.asarray(w).reshape(-1))
    return op.adj_op(np.asarray(k).reshape(-1))


def _opts(tmp_path, **kw):
    return _build_options({"cache_dir": str(tmp_path), **kw})


# ------------------------------------------------------------------ the weights
@pytest.mark.parametrize("mode", ["GoldenSampling", "GoldenGridSampling"])
@pytest.mark.parametrize("chunk", [None, 7, 200], ids=["one", "spokes7", "spokes200"])
def test_pipe_weights_equal_mrinufft_pipe_chunk_by_chunk(tmp_path, mode, chunk):
    from brkraw_sordino import dcf

    info = _golden_info(1, mode)
    rows = trajectory_rows(info, _opts(tmp_path))
    n = rows.n_pro
    sw = dcf.pipe_weights(rows, n, 1, SHAPE, chunk or n)
    ref, _ = _pipe_reference(rows.rows(0, n)[:, 1:], SHAPE)
    assert sw.w.shape == (n, N_POINTS - 1) and sw.w_virtual is None
    assert _rel(sw.w.reshape(-1), ref) < TOL


def test_pipe_weights_with_the_virtual_samples(tmp_path):
    from brkraw_sordino import dcf

    info = _golden_info(1, "GoldenGridSampling")
    rows = trajectory_rows(info, _opts(tmp_path))
    vt = kcentre.leading_points(info, 1)
    sw = dcf.pipe_weights(rows, rows.n_pro, 1, SHAPE, 50, virtual_traj=vt)
    ref, ref_v = _pipe_reference(rows.rows(0, rows.n_pro)[:, 1:], SHAPE, vt)
    assert sw.w_virtual.shape == vt.shape[:2]
    assert _rel(sw.w.reshape(-1), ref) < TOL and _rel(sw.w_virtual.reshape(-1), ref_v) < TOL


def test_weights_keep_the_brightness_of_the_k2_weights(tmp_path):
    from brkraw_sordino import dcf

    info = _golden_info(1, "GoldenGridSampling")
    rows = trajectory_rows(info, _opts(tmp_path))
    tr = rows.rows(0, rows.n_pro)[:, 1:]
    sw = dcf.pipe_weights(rows, rows.n_pro, 1, SHAPE, 64)
    k2 = serial.density(tr)
    assert sw.w.sum() == pytest.approx((k2 / k2.max()).sum(), rel=1e-12)
    assert np.all(sw.w > 0)


def test_weights_are_capped_at_the_k2_rule(tmp_path):
    """Pipe boosts sparsely covered samples (noise; scan 17's head got 15x the weight, run 5): the weight of a
    sample is never above its |k|^2 / max weight; after the cap the sum is restored, so the largest ratio is the
    restoring factor and it is reached by the capped samples."""
    from brkraw_sordino import dcf

    info = _golden_info(1, "GoldenGridSampling")
    rows = trajectory_rows(info, _opts(tmp_path))
    tr = rows.rows(0, rows.n_pro)[:, 1:]
    k2 = serial.density(tr)
    k2 = (k2 / k2.max()).reshape(rows.n_pro, -1)
    sw = dcf.pipe_weights(rows, rows.n_pro, 1, SHAPE, 64)
    ratio = sw.w / k2
    top = ratio.max()                                   # the restoring factor, shared by every capped sample
    assert 1.0 <= top < 1.5
    assert np.isclose(ratio, top, rtol=1e-9, atol=0).sum() > 10
    assert sw.w.sum() == pytest.approx(k2.sum(), rel=1e-12)
    assert dcf.RULE["cap"] == "min(Pipe, |k|^2/max), then the same sum"


def test_uncapped_weights_for_diagnostics_are_mrinufft_pipe(tmp_path):
    """``cap=False`` (diagnostics and figures only) is mrinufft's pipe scaled to the |k|^2 sum."""
    from brkraw_sordino import dcf
    from mrinufft.operators.interfaces.finufft import MRIfinufft

    info = _golden_info(1, "GoldenGridSampling")
    rows = trajectory_rows(info, _opts(tmp_path))
    tr = rows.rows(0, rows.n_pro)[:, 1:]
    sw = dcf.pipe_weights(rows, rows.n_pro, 1, SHAPE, 50, cap=False)
    w = np.abs(np.asarray(MRIfinufft.pipe(np.ascontiguousarray(tr.reshape(-1, 3) / 0.5 * np.pi), tuple(SHAPE),
                                         max_iter=10, osf=2, normalize=False)))
    k2 = serial.density(tr)
    w *= (k2 / k2.max()).sum() / w.sum()
    assert _rel(sw.w.reshape(-1), w) < TOL


def test_only_golden_modes_use_sample_weights():
    from brkraw_sordino import dcf

    assert dcf.uses_sample_weights(_golden_info(1, "GoldenSampling"))
    assert dcf.uses_sample_weights(_golden_info(1, "GoldenGridSampling"))
    assert not dcf.uses_sample_weights(_serial_info(1, 1))


# ------------------------------------------------------------------ the weights reach every path
def _k2_weights(rows, n_pro, virtual=None):
    from brkraw_sordino import dcf

    tr = rows.rows(0, n_pro)[:, 1:]
    k2 = np.square(tr).sum(-1)
    dmax = k2.max() if virtual is None else max(k2.max(), np.square(virtual).sum(-1).max())
    return dcf.SampleWeights(serial.density(tr).reshape(n_pro, -1) / dmax,
                             None if virtual is None else serial.density(virtual).reshape(n_pro, -1) / dmax)


@pytest.mark.parametrize("k0,method", [(False, None), (True, "toeplitz"), (True, "samples")],
                         ids=["plain", "k0-toeplitz", "k0-samples"])
def test_k2_values_given_as_weights_reproduce_the_product(tmp_path, k0, method):
    """The weights path is the product path: |k|^2 / max handed in as weights gives the same image."""
    info = _golden_info(2, "GoldenGridSampling")
    opts = _opts(tmp_path, estimate_k0=k0)
    rows = trajectory_rows(info, opts)
    factor = phase_correction_rows(info, opts, N_POINTS)
    fids = _fid_frames(info, get_trajectory(info, opts), 2, 1, factor[:])
    vt = kcentre.leading_points(info, 1) if k0 else None
    outs = []
    for weights in (None, _k2_weights(rows, rows.n_pro, vt)):
        out = io.BytesIO()
        k0s = []
        recon_dataobj(io.BytesIO(b"".join(fids)), rows, info, out, opts, phase_factor=factor,
                      virtual_traj=vt, k0_out=k0s, chunk_spokes=97, k0_method=method, weights=weights)
        outs.append((np.frombuffer(out.getvalue(), dtype=np.complex128), k0s))
    assert _rel(outs[1][0], outs[0][0]) < 1e-12
    if k0:
        assert np.allclose(np.asarray(outs[1][1]), np.asarray(outs[0][1]), rtol=1e-12, atol=0)


@pytest.mark.parametrize("k0", [False, True])
def test_k2_values_given_as_weights_reproduce_the_frames(tmp_path, k0):
    from brkraw_sordino import frames

    info = dict(_golden_info(2, "GoldenGridSampling"))
    opts = _opts(tmp_path, estimate_k0=k0, frame_spokes=120, frame_step=60)
    rows = trajectory_rows(info, opts)
    factor = phase_correction_rows(info, opts, N_POINTS)
    fids = _fid_frames(info, get_trajectory(info, opts), 2, 1, factor[:])
    vt = kcentre.leading_points(info, 1) if k0 else None
    plan = frames.make_plan(info, opts)
    outs = []
    for weights in (None, _k2_weights(rows, rows.n_pro, vt)):
        out = io.BytesIO()
        frames.recon_frames(io.BytesIO(b"".join(fids)), rows, info, out, opts, plan, phase_factor=factor,
                            virtual_traj=vt, chunk_spokes=80, weights=weights)
        outs.append(np.frombuffer(out.getvalue(), dtype=np.complex128))
    assert _rel(outs[1], outs[0]) < 1e-12


@pytest.mark.parametrize("mode", ["GoldenSampling", "GoldenGridSampling"])
def test_golden_hook_uses_the_sample_weights(tmp_path, monkeypatch, mode):
    from brkraw_sordino.orientation import correct as correct_orientation

    info = _golden_info(1, mode)
    options = _opts(tmp_path / "t")
    traj = get_trajectory(info, options)
    factor = phase_correction_rows(info, options, N_POINTS)[:]
    (fid,) = _fid_frames(info, traj, 1, 1, factor)
    img = _hook_image(tmp_path / "h", monkeypatch, info, fid)[..., 0]
    vol = np.frombuffer(fid, dtype="<i4").reshape((2, N_POINTS, 1, info["NPro"]), order="F")
    k = ((vol[0] + 1j * vol[1]).T[:, 0, :]) * factor
    w, _ = _pipe_reference(traj[:, 1:, :], SHAPE)
    ref = correct_orientation(_weighted_adjoint(k[:, 1:], traj[:, 1:, :], SHAPE, w), info)
    assert _rel(img, ref) < 1e-8
    k2 = correct_orientation(recon.nufft_adjoint(k[:, 1:], traj[:, 1:, :], SHAPE, 1), info)
    assert _rel(img, k2) > 1e-3


def test_golden_frames_still_add_up_to_the_repetition(tmp_path, monkeypatch):
    info = _golden_info(1, "GoldenSampling")                       # 20 subsets of 40
    options = _opts(tmp_path / "t")
    traj = get_trajectory(info, options)
    factor = phase_correction_rows(info, options, N_POINTS)[:]
    (fid,) = _fid_frames(info, traj, 1, 1, factor)
    rep = _hook_image(tmp_path / "r", monkeypatch, info, fid)[..., 0]
    frs = _hook_image(tmp_path / "f", monkeypatch, info, fid, frame_spokes="subset")
    assert frs.shape[-1] == 20
    assert _rel(frs.sum(-1) / 20.0, rep) < 1e-9


# ------------------------------------------------------------------ cache key and memory
def test_cache_key_names_the_weight_rule_for_golden_scans_only(tmp_path):
    from brkraw_sordino import dcf

    opts = _opts(tmp_path)
    g = hook._build_cache_params(_Scan(), None, _Entry(b""), opts, _golden_info())
    d = hook._build_cache_params(_Scan(), None, _Entry(b""), opts, _serial_info(1, 1))
    assert g["density"] == dcf.RULE
    assert "density" not in d


def test_memory_estimate_counts_the_weights(tmp_path, monkeypatch):
    from brkraw_sordino import memguard

    info = _golden_info(1, "GoldenGridSampling")
    monkeypatch.setattr(hook, "_parse_recon_info", lambda scan: dict(info))
    monkeypatch.setattr(hook, "_get_fid_entry", lambda scan: _Entry(b"\0" * (8 * N_POINTS * info["NPro"])))
    gi = hook.get_dataobj_info(_Scan(), None, cache_dir=str(tmp_path), frame_spokes="repetition")
    expected = memguard.dcf_nbytes(info["NPro"], N_POINTS - 1, SHAPE)
    assert expected == 8 * info["NPro"] * (N_POINTS - 1) + 16 * int(np.prod(SHAPE))
    assert gi["recon_dcf_nbytes"] == expected and gi["recon_nbytes"] >= expected
    dinfo = _serial_info(1, 1)
    monkeypatch.setattr(hook, "_parse_recon_info", lambda scan: dict(dinfo))
    monkeypatch.setattr(hook, "_get_fid_entry", lambda scan: _Entry(b"\0" * (8 * N_POINTS * dinfo["NPro"])))
    di = hook.get_dataobj_info(_Scan(), None, cache_dir=str(tmp_path))
    assert di["recon_dcf_nbytes"] == 0


# ------------------------------------------------------------------ spoke mask entry (WI-0118)
def _mask_case(tmp_path):
    from brkraw_sordino import frames

    info = dict(_golden_info(2, "GoldenSampling"))
    opts = _opts(tmp_path, frame_spokes=200, frame_step=100)
    rows = trajectory_rows(info, opts)
    factor = phase_correction_rows(info, opts, N_POINTS)
    fids = _fid_frames(info, get_trajectory(info, opts), 2, 1, factor[:])
    return frames, info, opts, rows, factor, fids, frames.make_plan(info, opts)


def _run_frames(fr, fid, rows, info, opts, plan, factor, **kw):
    out = io.BytesIO()
    fr.recon_frames(io.BytesIO(fid), rows, info, out, opts, plan, phase_factor=factor, chunk_spokes=90, **kw)
    return np.frombuffer(out.getvalue(), dtype=np.complex128)


def test_an_all_true_spoke_mask_changes_nothing(tmp_path):
    fr, info, opts, rows, factor, fids, plan = _mask_case(tmp_path)
    fid = b"".join(fids)
    a = _run_frames(fr, fid, rows, info, opts, plan, factor)
    b = _run_frames(fr, fid, rows, info, opts, plan, factor, spoke_mask=np.ones(2 * info["NPro"], bool))
    # finufft's multithreaded spreading is not bit-reproducible between two runs, so "the same"
    # is held to rounding (two identical calls differ by about 1e-16 relative)
    assert _rel(b, a) < 1e-12


def test_masked_spokes_add_nothing(tmp_path):
    fr, info, opts, rows, factor, fids, plan = _mask_case(tmp_path)
    npro = info["NPro"]
    mask = np.ones(2 * npro, bool)
    mask[350:530] = False                                   # crosses frames, inside repetition 0
    mask[npro + 10:npro + 40] = False                       # inside repetition 1
    zeroed = []
    for r, raw in enumerate(fids):
        vol = np.frombuffer(raw, dtype="<i4").reshape((2, N_POINTS, 1, npro), order="F").copy()
        vol[..., ~mask[r * npro:(r + 1) * npro]] = 0
        zeroed.append(vol.tobytes(order="F"))
    a = _run_frames(fr, b"".join(fids), rows, info, opts, plan, factor, spoke_mask=mask)
    b = _run_frames(fr, b"".join(zeroed), rows, info, opts, plan, factor)
    assert _rel(a, b) < 1e-12
    assert _rel(a, _run_frames(fr, b"".join(fids), rows, info, opts, plan, factor)) > 1e-3


def test_spoke_mask_is_checked(tmp_path):
    fr, info, opts, rows, factor, fids, plan = _mask_case(tmp_path)
    with pytest.raises(ValueError, match="spoke_mask"):
        _run_frames(fr, b"".join(fids), rows, info, opts, plan, factor, spoke_mask=np.ones(5, bool))
    k0 = _opts(tmp_path, frame_spokes=200, frame_step=100, estimate_k0=True)
    vt = kcentre.leading_points(info, 1)
    with pytest.raises(ValueError, match="WI-0118"):
        _run_frames(fr, b"".join(fids), rows, info, k0, fr.make_plan(info, k0), factor, virtual_traj=vt,
                    spoke_mask=np.ones(2 * info["NPro"], bool))
