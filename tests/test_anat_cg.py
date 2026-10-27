"""anat_solver='cg': the anatomical reference of a whole scan by CG on all spokes (frames.py).

Small synthetic scans of test_rep_pattern (16^3 grid, 800 spokes per repetition), data made by the
forward model of a smooth phantom, one frame of every spoke (frame_spokes='all'). No real data.
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
from test_rep_pattern import NOISE, _info, _opts, _phantom, _run, _samples
from test_serial_recon import N_POINTS, SHAPE


def _anat(tmp_path, info, samples, k0=False, **kw):
    """(plan, the one image, the record of the solve) of frame_spokes='all'."""
    out = {}
    opts = _opts(tmp_path, estimate_k0=k0, frame_spokes="all", **kw)
    plan = frames.make_plan(info, opts)
    vt = kcentre.leading_points(info, 1) if k0 else None
    buf = io.BytesIO()
    frames.recon_frames(io.BytesIO(_encode(samples)), trajectory_rows(info, opts), info, buf, opts, plan,
                        phase_factor=phase_correction_rows(info, opts, N_POINTS), virtual_traj=vt,
                        chunk_spokes=200, anat_out=out)
    img = np.frombuffer(buf.getvalue(), dtype=np.complex128).reshape((plan.n_frames, *SHAPE[::-1]))
    return plan, img.transpose(0, 3, 2, 1)[0], out


def _err(img, truth):
    """Relative error of |img| to the truth after the best scalar gain (the gain is not under test)."""
    m = np.abs(img)
    alpha = float(np.vdot(truth, m).real / np.vdot(truth, truth).real)
    return float(np.linalg.norm(m - alpha * truth) / np.linalg.norm(alpha * truth))


def test_cg_beats_the_adjoint_on_a_noise_free_phantom_and_improves_with_iterations(tmp_path):
    info, x = _info(2), _phantom()
    samples = _samples(tmp_path, info, [x, x])
    plan, adj, _ = _anat(tmp_path / "adj", info, samples)
    _, cg2, rec2 = _anat(tmp_path / "cg2", info, samples, anat_solver="cg", anat_cg_iter=2)
    _, cg10, rec10 = _anat(tmp_path / "cg10", info, samples, anat_solver="cg", anat_cg_iter=10)
    assert plan.n_frames == 1 and cg10.shape == adj.shape
    assert _err(cg10, x) < _err(cg2, x) < _err(adj, x)
    assert rec10["relative_residual"] < rec2["relative_residual"] < 1.0
    assert rec10["solver"] == "cg" and rec10["n_iter"] == 10 and rec10["lambda"] == 0.0


def test_cg_with_estimate_k0_runs_and_beats_the_adjoint(tmp_path):
    info, x = _info(1), _phantom()
    samples = _samples(tmp_path, info, [x])
    _, adj, _ = _anat(tmp_path / "adj", info, samples, k0=True)
    _, cg, rec = _anat(tmp_path / "cg", info, samples, k0=True, anat_solver="cg")
    assert _err(cg, x) < _err(adj, x)
    assert rec["relative_residual"] < 1.0


def test_lambda_reduces_noise_amplification(tmp_path):
    info, x = _info(1), _phantom()
    clean = _samples(tmp_path, info, [x])
    noisy = _samples(tmp_path, info, [x], noise=NOISE)

    def response(lam):
        kw = dict(anat_solver="cg", anat_lambda=lam)
        _, a, _ = _anat(tmp_path / f"c{lam}", info, clean, **kw)
        _, b, rec = _anat(tmp_path / f"n{lam}", info, noisy, **kw)
        assert rec["lambda"] == lam
        return float(np.linalg.norm(b - a))

    assert response(0.5) < response(0.0)


def test_default_is_the_adjoint_and_the_key_is_unchanged(tmp_path):
    info, x = _info(1), _phantom()
    samples = _samples(tmp_path, info, [x])
    _, whole, _ = _anat(tmp_path / "a", info, samples)
    _, spokes = _run(tmp_path / "b", info, samples, frame_spokes=int(info["NPro"]))
    assert np.allclose(whole, spokes[0])
    plain = hook._build_cache_params(_Scan(), None, _Fid(), _build_options({"cache_dir": str(tmp_path)}),
                                     _recon_info(1))
    assert "anat" not in plain and not {"anat_solver", "anat_cg_iter", "anat_lambda"} & set(plain["options"])
    cg = _build_options({"cache_dir": str(tmp_path), "frame_spokes": "all", "anat_solver": "cg",
                         "anat_cg_iter": 7, "anat_lambda": 0.1})
    key = hook._build_cache_params(_Scan(), None, _Fid(), cg, _recon_info(1))["anat"]
    assert key == {"solver": "cg", "n_iter": 7, "lambda": 0.1}


def test_option_values_and_guards(tmp_path):
    def opts(**kw):
        return _build_options({"cache_dir": str(tmp_path), **kw})

    default = opts()
    assert (default.anat_solver, default.anat_cg_iter, default.anat_lambda) == ("adjoint", 10, 0.0)
    assert opts(anat_solver=" CG ", frame_spokes="all").anat_solver == "cg"
    for bad in ({"anat_solver": "lsqr"}, {"anat_cg_iter": 0}, {"anat_cg_iter": 2.5}, {"anat_cg_iter": True},
                {"anat_lambda": -1}, {"anat_lambda": "x"}, {"anat_lambda": float("nan")}):
        with pytest.raises(ValueError):
            opts(**bad)
    info = _info(1)
    with pytest.raises(ValueError, match="frame_spokes='all'"):
        frames.make_plan(info, opts(anat_solver="cg"))
    with pytest.raises(ValueError, match="need anat_solver='cg'"):
        frames.make_plan(info, opts(anat_lambda=0.1, frame_spokes="all"))
    with pytest.raises(ValueError, match="rep_pattern"):
        frames.make_plan(info, opts(anat_solver="cg", frame_spokes="all", rep_pattern="model"))
    assert frames.make_plan(info, opts(frame_spokes="all")).n_frames == 1


def test_memory_estimate_counts_the_cg_volumes():
    plain = memguard.frames_nbytes(1, 3, (4, 4, 4))
    assert memguard.frames_nbytes(1, 3, (4, 4, 4), anat_cg=True) - plain == 5 * 3 * 64 * 16
