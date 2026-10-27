"""frame_static='cg': x_hat as the static image of rep_pattern='model' frames (frames.py).

Small synthetic scans of test_rep_pattern (16^3 grid, 800 spokes per repetition), data made by the
forward model of a smooth phantom and a small time signal. Frames are the golden method subsets (40
spokes, 20 positions per repetition). No real data.
"""
import io

import numpy as np
import pytest

from brkraw_sordino import frames, hook, kcentre, memguard
from brkraw_sordino.hook import _build_options
from brkraw_sordino.recon import phase_correction_rows
from brkraw_sordino.traj import trajectory_rows

from test_hook_read import _Fid, _Scan, _recon_info
from test_peri_event_corrections import _encode
from test_rep_pattern import _info, _opts, _phantom, _rel, _roi, _samples
from test_serial_recon import N_POINTS, SHAPE

POSITIONS = 20           # 800 spokes per repetition / the 40-spoke method subset
SUBSET = None            # frame_spokes at its default: the method subset
FS = SUBSET
EXACT_ITER = 200         # noise-free solve long enough for the residual to vanish


def _frames(tmp_path, info, samples, out=None, **kw):
    """The complex frames (n_frames, *SHAPE) of the golden subset frames; ``out`` gets the record of the solve."""
    kw.setdefault("frame_spokes", SUBSET)
    options = _opts(tmp_path, **kw)
    plan = frames.make_plan(info, options)
    buf = io.BytesIO()
    frames.recon_frames(io.BytesIO(_encode(samples)), trajectory_rows(info, options), info, buf, options, plan,
                        phase_factor=phase_correction_rows(info, options, N_POINTS), chunk_spokes=200,
                        anat_out=out)
    data = np.frombuffer(buf.getvalue(), dtype=np.complex128).reshape((plan.n_frames, *SHAPE[::-1]))
    return plan, data.transpose(0, 3, 2, 1)


def _err(img, truth):
    """Relative error of |img| to the truth after the best scalar gain (the gain is not under test)."""
    m = np.abs(img)
    alpha = float(np.vdot(truth, m).real / np.vdot(truth, truth).real)
    return float(np.linalg.norm(m - alpha * truth) / np.linalg.norm(alpha * truth))


def _xhat(tmp_path, info, samples, n_iter=10):
    """x_hat: the CG image of every spoke (anat_solver='cg'), the solve the frames start from."""
    opts = dict(frame_spokes="all", anat_solver="cg", anat_cg_iter=n_iter)
    return _frames(tmp_path, info, samples, **opts)[1][0]


def test_static_noise_free_object_gives_x_hat_exactly_in_every_frame(tmp_path):
    # converged CG on consistent data: A x_hat = d, the residual term vanishes and y_k = x_hat
    info, x = _info(2), _phantom()
    samples = _samples(tmp_path, info, [x, x])
    rec = {}
    plan, cg = _frames(tmp_path / "cg", info, samples, out=rec, rep_pattern="model", frame_static="cg",
                       frame_static_iter=EXACT_ITER)
    xhat = _xhat(tmp_path / "xhat", info, samples, n_iter=EXACT_ITER)
    assert plan.n_frames == 2 * POSITIONS
    assert rec["solver"] == "cg" and rec["n_iter"] == EXACT_ITER and rec["lambda"] == 0.0
    assert rec["relative_residual"] < 1e-5
    for f in cg:
        assert _rel(f, xhat) < 1e-3


def test_cg_frames_are_closer_to_the_truth_than_the_adjoint_frames(tmp_path):
    info, x = _info(2), _phantom()
    samples = _samples(tmp_path, info, [x, x])
    rec = {}
    plan, cg = _frames(tmp_path / "cg", info, samples, out=rec, rep_pattern="model", frame_static="cg")
    _, adj = _frames(tmp_path / "adj", info, samples, rep_pattern="model")
    assert plan.n_frames == 2 * POSITIONS and cg.shape == adj.shape
    assert _err(cg.mean(axis=0), x) < _err(adj.mean(axis=0), x)
    assert rec["solver"] == "cg" and rec["n_iter"] == 10 and rec["lambda"] == 0.0
    assert 0 < rec["relative_residual"] < 1.0 and rec["iterations_run"] >= 1


def test_time_signal_in_a_region_is_kept_in_the_residual_term(tmp_path):
    info = _info(3)
    x, roi = _phantom(), _roi()
    amp = np.array([0.0, 0.2, -0.2])
    samples = _samples(tmp_path, info, [x + a * roi for a in amp])
    _, cg = _frames(tmp_path / "cg", info, samples, rep_pattern="model", frame_static="cg")
    _, adj = _frames(tmp_path / "adj", info, samples, rep_pattern="model")
    mask = roi > 0.5

    def corr(images):
        per_rep = images.reshape(3, POSITIONS, *SHAPE).real.mean(axis=1)
        return float(np.corrcoef([per_rep[r][mask].mean() for r in range(3)], amp)[0, 1])

    assert corr(adj) > 0.9
    assert corr(cg) > 0.9
    assert corr(cg) > corr(adj) - 0.05


def test_default_frames_are_unchanged(tmp_path):
    info, x = _info(2), _phantom()
    samples = _samples(tmp_path, info, [x, x + 0.1 * _roi()])
    _, plain = _frames(tmp_path / "a", info, samples, rep_pattern="model")
    _, explicit = _frames(tmp_path / "b", info, samples, rep_pattern="model", frame_static="adjoint")
    assert np.array_equal(plain, explicit)


def test_frame_static_with_estimate_k0_runs(tmp_path):
    info, x = _info(2), _phantom()
    samples = _samples(tmp_path, info, [x, x])
    options = _opts(tmp_path, estimate_k0=True, frame_spokes=FS, rep_pattern="model", frame_static="cg")
    plan = frames.make_plan(info, options)
    buf, rec = io.BytesIO(), {}
    frames.recon_frames(io.BytesIO(_encode(samples)), trajectory_rows(info, options), info, buf, options, plan,
                        phase_factor=phase_correction_rows(info, options, N_POINTS),
                        virtual_traj=kcentre.leading_points(info, 1), chunk_spokes=200, anat_out=rec)
    data = np.frombuffer(buf.getvalue(), dtype=np.complex128)
    assert data.size == plan.n_frames * int(np.prod(SHAPE)) and np.isfinite(data).all()
    assert rec["solver"] == "cg"


def _k0_frames(tmp_path, info, samples, **kw):
    """Complex frames with estimate_k0 on (the virtual-centre term in the operator)."""
    options = _opts(tmp_path, estimate_k0=True, **kw)
    plan = frames.make_plan(info, options)
    buf = io.BytesIO()
    frames.recon_frames(io.BytesIO(_encode(samples)), trajectory_rows(info, options), info, buf, options, plan,
                        phase_factor=phase_correction_rows(info, options, N_POINTS),
                        virtual_traj=kcentre.leading_points(info, 1), chunk_spokes=200)
    data = np.frombuffer(buf.getvalue(), dtype=np.complex128).reshape((plan.n_frames, *SHAPE[::-1]))
    return plan, data.transpose(0, 3, 2, 1)


def test_static_object_with_estimate_k0_gives_x_hat_in_every_frame(tmp_path):
    # the residual gain uses the same normal operator as the solve (virtual-centre term included),
    # so a static object leaves a zero residual term and each frame is x_hat
    info, x = _info(2), _phantom()
    samples = _samples(tmp_path, info, [x, x])
    plan, cg = _k0_frames(tmp_path / "cg", info, samples, frame_spokes=FS, rep_pattern="model",
                          frame_static="cg", frame_static_iter=EXACT_ITER)
    _, xhat = _k0_frames(tmp_path / "xhat", info, samples, frame_spokes="all", anat_solver="cg",
                         anat_cg_iter=EXACT_ITER)
    assert plan.n_frames == 2 * POSITIONS
    for f in cg:
        assert _rel(f, xhat[0]) < 1e-2


def test_option_values_and_guards(tmp_path):
    def opts(**kw):
        return _build_options({"cache_dir": str(tmp_path), **kw})

    default = opts()
    assert (default.frame_static, default.frame_static_iter, default.frame_static_lambda) == ("adjoint", 10, 0.0)
    assert opts(frame_static=" CG ", rep_pattern="model", frame_spokes=FS).frame_static == "cg"
    for bad in ({"frame_static": "lsqr"}, {"frame_static_iter": 0}, {"frame_static_iter": 2.5},
                {"frame_static_iter": True}, {"frame_static_lambda": -1}, {"frame_static_lambda": "x"},
                {"frame_static_lambda": float("nan")}):
        with pytest.raises(ValueError):
            opts(**bad)
    info = _info(2)
    for rep in ("none", "measured"):
        with pytest.raises(ValueError, match="needs rep_pattern='model'"):
            frames.make_plan(info, opts(frame_static="cg", frame_spokes=FS, rep_pattern=rep))
    with pytest.raises(ValueError, match="need frame_static='cg'"):
        frames.make_plan(info, opts(frame_static_lambda=0.1, frame_spokes=FS, rep_pattern="model"))
    ok = opts(frame_static="cg", frame_spokes=FS, rep_pattern="model", frame_static_iter=4)
    assert frames.make_plan(info, ok).n_frames == 2 * POSITIONS


def test_key_entry_only_when_not_adjoint_and_memory_estimate(tmp_path):
    base = {"cache_dir": str(tmp_path), "frame_spokes": FS, "rep_pattern": "model"}
    plain = hook._build_cache_params(_Scan(), None, _Fid(), _build_options(base), _recon_info(1))
    assert "frame_static" not in plain
    assert not {"frame_static", "frame_static_iter", "frame_static_lambda"} & set(plain["options"])
    cg = _build_options({**base, "frame_static": "cg", "frame_static_iter": 7, "frame_static_lambda": 0.1})
    key = hook._build_cache_params(_Scan(), None, _Fid(), cg, _recon_info(1))["frame_static"]
    assert key == {"static": "cg", "n_iter": 7, "lambda": 0.1}
    plain_n = memguard.frames_nbytes(1, 3, (4, 4, 4), rep_pattern=True)
    assert memguard.frames_nbytes(1, 3, (4, 4, 4), rep_pattern=True, static_cg=True) - plain_n == 5 * 3 * 64 * 16
