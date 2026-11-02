"""frame_export=complex: complex frames, pure-noise frames and the k-space noise level (frameexport.py).

Small synthetic scans (16^3 grid, 800 spokes per repetition, frames of 100 spokes) with a static object and
known complex Gaussian k-space noise. No real data.

Noise convention of the tests: ``_samples(noise=s)`` adds ``s * (randn + 1j randn)`` to every sample, so
E|n|^2 = 2 s^2, and sigma below is that E|n|^2 (the k-space level of ``ksigma``).
"""
import io
import json

import nibabel
import numpy as np
import pytest

from brkraw_sordino import frameexport, frames, hook, kcentre, recon, serial
from brkraw_sordino.recon import phase_correction_rows
from brkraw_sordino.traj import trajectory_rows

from test_golden_frames import _opts
from test_golden_recon import _Entry, _Scan as _GScan
from test_hook_read import _Fid, _Scan, _recon_info
from test_peri_event_corrections import SPEC, _encode
from test_rep_pattern import FS, NOISE, _info, _phantom, _samples
from test_serial_recon import N_POINTS, SHAPE

SIGMA2 = 2.0 * NOISE ** 2           # E|n|^2 of the injected noise
N_REP = 4
#: stated tolerances: the mean voxel variance of the noise frames against the variance predicted through
#: the reconstruction (about 1500 independent samples per frame, 24 or more frames), the k-space level
#: against the injected level (many thousand samples per band, 3 degrees of freedom each)
VARIANCE_TOL = 0.05
SIGMA_TOL = 0.03
BAND_TOL = 0.05


def _noise_run(tmp_path, info, samples, k0=False, **kw):
    kw.setdefault("frame_spokes", FS)
    options = _opts(tmp_path, estimate_k0=k0, **kw)
    plan = frames.make_plan(info, options)
    vt = kcentre.leading_points(info, 1) if k0 else None
    stats = {}
    out = io.BytesIO()
    frames.recon_frames(io.BytesIO(_encode(samples)), trajectory_rows(info, options), info, out, options, plan,
                        phase_factor=phase_correction_rows(info, options, N_POINTS), virtual_traj=vt,
                        chunk_spokes=200, noise_out=stats)
    data = np.frombuffer(out.getvalue(), dtype=np.complex128).reshape((plan.n_frames, *SHAPE[::-1]))
    return options, plan, data.transpose(0, 3, 2, 1), stats, vt


def _data_run(tmp_path, info, samples, **kw):
    kw.setdefault("frame_spokes", FS)
    options = _opts(tmp_path, **kw)
    plan = frames.make_plan(info, options)
    out = io.BytesIO()
    frames.recon_frames(io.BytesIO(_encode(samples)), trajectory_rows(info, options), info, out, options, plan,
                        phase_factor=phase_correction_rows(info, options, N_POINTS), chunk_spokes=200)
    return plan, out.getvalue()


def _predicted_variance(info, options, plan, sigma2, vt=None):
    """E|y|^2 per voxel of every frame through the reconstruction: (s / nf)^2 sigma^2 sum_j w_j^2."""
    tr = trajectory_rows(info, options).rows(0, plan.npro)[:, 1:]
    dens = serial.density(tr)
    dmax = float(dens.max())
    if vt is not None:
        dmax = max(dmax, float(serial.density(vt).max()))
    w2 = ((dens / dmax) ** 2).reshape(plan.npro, -1).sum(axis=1)       # per spoke position
    nf = serial.norm_factor(SHAPE)
    return np.array([(s / nf) ** 2 * sigma2 * w2[np.arange(lo, hi) % plan.npro].sum()
                     for (lo, hi), s in zip(plan.ranges, plan.scales)])


def _noisy(tmp_path, n_rep=N_REP, seed=7):
    info = _info(n_rep)
    x = _phantom()
    return info, _samples(tmp_path, info, [x] * n_rep, noise=NOISE, seed=seed)


# ------------------------------------------------------------------ the noise frames
@pytest.mark.parametrize("k0,extra", [(False, {}), (True, {}), (False, {"frame_step": 50})],
                         ids=["plain", "estimate_k0", "sliding"])
def test_noise_frames_have_the_variance_predicted_through_the_reconstruction(tmp_path, k0, extra):
    info, samples = _noisy(tmp_path / "s")
    options, plan, noise, _, vt = _noise_run(tmp_path / "n", info, samples, k0=k0, **extra)
    pred = _predicted_variance(info, options, plan, SIGMA2, vt)
    meas = np.mean(np.abs(noise) ** 2, axis=(1, 2, 3))
    assert noise.shape[0] == plan.n_frames == len(pred)
    assert abs(meas.sum() / pred.sum() - 1.0) < VARIANCE_TOL
    assert np.max(np.abs(meas / pred - 1.0)) < 0.2                      # frame by frame (1 / sqrt(1500) each)


def test_noise_frames_hold_no_object_and_sum_to_zero_over_repetitions(tmp_path):
    info, samples = _noisy(tmp_path / "s")
    _, plan, noise, _, _ = _noise_run(tmp_path / "n", info, samples)
    pos = int(plan.npro // plan.window)
    assert plan.n_frames == N_REP * pos
    by_pos = noise.reshape(N_REP, pos, *SHAPE)
    # the residuals of one spoke position sum to zero over the repetitions (linearity of the frame)
    assert np.abs(by_pos.sum(axis=0)).max() < 1e-3 * np.abs(noise).std()
    # a noise-free scan of a static object has no noise frame at all
    _, clean = _noise_run(tmp_path / "c", info, _samples(tmp_path / "s0", info, [_phantom()] * N_REP, noise=0.0))[1:3]
    assert np.abs(clean).max() < 1e-3 * np.abs(noise).std()


def test_noise_frames_use_the_data_frames_plan_and_weights(tmp_path):
    # the data frames of the same run: same plan, same shape; the noise frames are not the data frames
    info, samples = _noisy(tmp_path / "s")
    plan, raw = _data_run(tmp_path / "d", info, samples)
    _, nplan, noise, _, _ = _noise_run(tmp_path / "n", info, samples)
    assert nplan == plan
    assert len(raw) == noise.size * 16
    data = np.frombuffer(raw, dtype=np.complex128)
    assert np.abs(data).mean() > np.abs(noise).mean()                     # the object is in the data only


def test_the_object_cancels_in_the_noise_frames(tmp_path):
    info = _info(N_REP)
    x = _phantom()
    s1 = _samples(tmp_path / "a", info, [x] * N_REP, noise=NOISE, seed=3)
    s2 = _samples(tmp_path / "b", info, [3.0 * x] * N_REP, noise=0.0)
    noisy2 = s1 + (s2 - _samples(tmp_path / "c", info, [x] * N_REP, noise=0.0))   # same noise, another object
    n1 = _noise_run(tmp_path / "n1", info, s1)[2]
    n2 = _noise_run(tmp_path / "n2", info, noisy2)[2]
    assert np.abs(n1 - n2).max() < 1e-3 * np.abs(n1).std()               # the object cancels in the residual


# ------------------------------------------------------------------ k-space sigma
def test_ksigma_recovers_the_injected_level(tmp_path):
    info, samples = _noisy(tmp_path / "s")
    _, plan, _, stats, _ = _noise_run(tmp_path / "n", info, samples)
    s = float(np.sqrt(SIGMA2))
    assert stats["n_repetitions"] == N_REP and stats["n_positions"] == plan.npro
    assert stats["bessel_factor"] == pytest.approx(np.sqrt(N_REP / (N_REP - 1.0)))
    assert abs(stats["sigma_0"] / s - 1.0) < SIGMA_TOL
    assert abs(stats["sigma_outer_pooled"] / s - 1.0) < SIGMA_TOL
    bands = [v for v in stats["sigma_per_band"] if v is not None]
    assert len(bands) >= 6 and len(stats["band_edges"]) == frameexport.N_BANDS + 1
    assert all(abs(v / s - 1.0) < BAND_TOL for v in bands), bands
    pos = np.array([v for v in stats["sigma_per_position"] if v is not None])
    assert len(pos) == plan.npro and abs(np.mean(pos ** 2) / s ** 2 - 1.0) < SIGMA_TOL
    json.dumps(frameexport.ksigma_document(stats))                         # plain JSON


def test_ksigma_follows_a_changed_level(tmp_path):
    info = _info(N_REP)
    samples = _samples(tmp_path / "s", info, [_phantom()] * N_REP, noise=2.0 * NOISE)
    stats = _noise_run(tmp_path / "n", info, samples)[3]
    assert abs(stats["sigma_0"] / (2.0 * np.sqrt(SIGMA2)) - 1.0) < SIGMA_TOL


def test_needs_two_repetitions(tmp_path):
    info = _info(1)
    options = _opts(tmp_path, frame_spokes=FS)
    plan = frames.make_plan(info, options)
    with pytest.raises(ValueError, match="at least 2 repetitions"):
        frameexport.check(hook._build_options({"cache_dir": str(tmp_path), "frame_export": "complex"}), plan, 1)
    samples = _samples(tmp_path / "s", info, [_phantom()], noise=NOISE)
    with pytest.raises(ValueError, match="at least 2 repetitions"):
        _noise_run(tmp_path / "n", info, samples)
    with pytest.raises(ValueError, match="Bessel"):
        frameexport.bessel_factor(1)


# ------------------------------------------------------------------ options and refusals
@pytest.mark.parametrize("value,expected", [(None, "off"), (False, "off"), ("off", "off"), (" Complex ", "complex"),
                                            ("", "off")])
def test_option_values(value, expected):
    assert frameexport.parse_export(value) == expected


@pytest.mark.parametrize("value", ["real", "on", True, 1, "magnitude"])
def test_bad_option_values_are_refused_by_name(tmp_path, value):
    with pytest.raises(ValueError, match="frame_export"):
        hook._build_options({"cache_dir": str(tmp_path), "frame_export": value})


def _plan_with(tmp_path, monkeypatch, info, **kw):
    monkeypatch.setattr(hook, "_parse_recon_info", lambda scan: dict(info))
    size = 2 * N_POINTS * int(info["NPro"]) * 4 * int(info["NRepetitions"])      # a FID of the right size
    monkeypatch.setattr(hook, "_get_fid_entry", lambda scan: _Entry(bytes(size)))
    return hook._plan(_GScan(), None, dict(cache_dir=str(tmp_path), estimate_k0=False, **kw))


def test_refusals_name_the_reason(tmp_path, monkeypatch):
    info = _info(N_REP)
    ok = _plan_with(tmp_path, monkeypatch, info, frame_export="complex", frame_spokes=FS)
    assert ok["frame_plan"].n_frames == N_REP * 8
    with pytest.raises(ValueError, match="needs a frame plan"):
        _plan_with(tmp_path, monkeypatch, info, frame_export="complex", frame_spokes="repetition")
    with pytest.raises(ValueError, match="at least 2 repetitions"):
        _plan_with(tmp_path, monkeypatch, _info(1), frame_export="complex", frame_spokes=FS)
    with pytest.raises(ValueError, match="peri_event"):
        _plan_with(tmp_path, monkeypatch, info, frame_export="complex", peri_event=SPEC)
    with pytest.raises(ValueError, match="subtract_rep_mean"):
        _plan_with(tmp_path, monkeypatch, info, frame_export="complex", frame_spokes=FS, subtract_rep_mean=True)
    with pytest.raises(ValueError, match="frame_export_prefix needs"):
        _plan_with(tmp_path, monkeypatch, info, frame_spokes=FS, frame_export_prefix=str(tmp_path / "x"))
    with pytest.raises(ValueError, match="single receive channel"):
        frameexport.check(hook._build_options({"cache_dir": str(tmp_path), "frame_export": "complex"}),
                          ok["frame_plan"], 2)


# ------------------------------------------------------------------ off: nothing changes
def test_export_keys_are_not_in_the_recon_cache_key(tmp_path):
    def params(**kw):
        options = hook._build_options(dict(cache_dir=str(tmp_path), **kw))
        return hook._build_cache_params(_Scan(), None, _Fid(), options, _recon_info(1))

    plain = params()
    on = params(frame_export="complex", frame_export_prefix=str(tmp_path / "x"))
    assert on["options"] == plain["options"]
    assert not [k for k in plain["options"] if k.startswith("frame_export")]
    assert set(frameexport.EXPORT_KEYS) <= set(frames.FRAME_KEYS)
    assert recon.build_recon_cache_path(tmp_path, on) == recon.build_recon_cache_path(tmp_path, plain)


def _base_module(name):
    """A module of the code before this option, loaded from git next to the current package."""
    import importlib.util
    import pathlib
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parents[1]
    base = subprocess.run(["git", "merge-base", "HEAD", "integrate/golden-sordino"], cwd=root,
                          capture_output=True, text=True)
    rev = base.stdout.strip()
    if base.returncode or not rev:
        pytest.skip("no integration branch to take the pre-change code from")
    shown = subprocess.run(["git", "show", f"{rev}:src/brkraw_sordino/{name}.py"], cwd=root,
                           capture_output=True, text=True)
    if shown.returncode or "frameexport" in shown.stdout:
        pytest.skip("the integration branch already holds the frame_export change")
    spec = importlib.util.spec_from_loader(f"brkraw_sordino._base_fe_{name}", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "brkraw_sordino"
    sys.modules[spec.name] = mod
    exec(compile(shown.stdout, f"<base {name}>", "exec"), mod.__dict__)
    return mod


def test_off_mode_images_match_the_pre_change_engine(tmp_path):
    base_frames = _base_module("frames")
    info, samples = _noisy(tmp_path / "s")
    options = _opts(tmp_path, frame_spokes=FS)
    plan = base_frames.make_plan(info, options)
    out = io.BytesIO()
    base_frames.recon_frames(io.BytesIO(_encode(samples)), trajectory_rows(info, options), info, out, options, plan,
                             phase_factor=phase_correction_rows(info, options, N_POINTS), chunk_spokes=200)
    _, now = _data_run(tmp_path / "now", info, samples)
    assert now == out.getvalue()


def test_the_data_frames_do_not_depend_on_the_export_option(tmp_path):
    info, samples = _noisy(tmp_path / "s")
    _, off = _data_run(tmp_path / "off", info, samples)
    _, on = _data_run(tmp_path / "on", info, samples, frame_export="complex", frame_export_prefix=str(tmp_path / "p"))
    assert on == off


# ------------------------------------------------------------------ through the hook
def _hook_env(tmp_path, monkeypatch, info, fid):
    monkeypatch.setattr(hook, "_parse_recon_info", lambda scan: dict(info))
    monkeypatch.setattr(hook, "_get_fid_entry", lambda scan: _Entry(fid))
    return dict(cache_dir=str(tmp_path / "cache"), frame_spokes=FS, estimate_k0=False)


def test_hook_writes_the_files_beside_the_magnitude_result(tmp_path, monkeypatch):
    info, samples = _noisy(tmp_path / "s", n_rep=3)
    kw = _hook_env(tmp_path, monkeypatch, info, _encode(samples))
    prefix = tmp_path / "out" / "sub"
    # off: the cache entry of the data frames, and no side file
    off = _GScan()
    mag_off = hook.get_dataobj(off, None, **kw)
    nii_off = hook.convert(off, mag_off, np.eye(4))
    cache = hook._plan(_GScan(), None, dict(kw))["img_cache_path"]
    cache_bytes = cache.read_bytes()
    assert not list(tmp_path.glob("out/*")) and not list((tmp_path / "cache").glob("*noise*"))
    # on: the same images and cache bytes, the noise frames added, the files written
    scan = _GScan()
    mag = hook.get_dataobj(scan, None, frame_export="complex", frame_export_prefix=str(prefix), **kw)
    nii = hook.convert(scan, mag, np.eye(4))
    assert np.array_equal(mag, mag_off) and cache.read_bytes() == cache_bytes
    assert np.array_equal(np.asarray(nii.dataobj), np.asarray(nii_off.dataobj))
    names = sorted(p.name for p in prefix.parent.iterdir())
    assert names == sorted(f"sub_{n}" for n in (
        "frames_real.nii.gz", "frames_imag.nii.gz", "frames.tsv", "noise_frames_real.nii.gz",
        "noise_frames_imag.nii.gz", "ksigma.json"))
    re = nibabel.load(str(prefix) + "_frames_real.nii.gz")
    im = nibabel.load(str(prefix) + "_frames_imag.nii.gz")
    assert re.get_data_dtype() == np.float32 and re.shape == mag.shape == (*SHAPE, 24)
    assert np.allclose(re.affine, nii.affine) and np.allclose(im.affine, nii.affine)
    assert np.allclose(re.header["pixdim"][1:5], nii.header["pixdim"][1:5])
    cplx = np.asarray(re.dataobj) + 1j * np.asarray(im.dataobj)
    assert np.allclose(np.abs(cplx), mag, rtol=1e-5, atol=1e-6 * mag.max())
    nre = np.asarray(nibabel.load(str(prefix) + "_noise_frames_real.nii.gz").dataobj)
    nim = np.asarray(nibabel.load(str(prefix) + "_noise_frames_imag.nii.gz").dataobj)
    assert nre.shape == re.shape and np.abs(nre + 1j * nim).mean() < np.abs(cplx).mean()
    rows = [line.split("\t") for line in (tmp_path / "out" / "sub_frames.tsv").read_text().splitlines()]
    assert rows[0] == ["frame", "position", "repetition", "first_spoke", "last_spoke"] and len(rows) == 25
    assert rows[1] == ["0", "0", "0", "0", "99"] and rows[10] == ["9", "100", "1", "900", "999"]
    sigma = json.loads((tmp_path / "out" / "sub_ksigma.json").read_text())
    assert sigma["n_repetitions"] == 3 and abs(sigma["sigma_0"] / np.sqrt(SIGMA2) - 1.0) < 0.05


def test_hook_reuses_the_noise_cache_and_follows_the_frame_selection(tmp_path, monkeypatch):
    info, samples = _noisy(tmp_path / "s", n_rep=3)
    kw = _hook_env(tmp_path, monkeypatch, info, _encode(samples))
    kw["frame_export"] = "complex"
    prefix = tmp_path / "o1" / "a"
    first = _GScan()
    hook.convert(first, hook.get_dataobj(first, None, frame_export_prefix=str(prefix), **kw), np.eye(4))

    def refuse(*args, **kwargs):
        raise AssertionError("the noise frames are cached")

    monkeypatch.setattr(hook, "_reconstruct_noise_frames", refuse)
    monkeypatch.setattr(frames, "recon_frames", refuse)
    second = _GScan()
    prefix2 = tmp_path / "o2" / "b"
    mag = hook.get_dataobj(second, None, frame_export_prefix=str(prefix2), frames=[3, 5, 6], **kw)
    hook.convert(second, mag, np.eye(4))
    assert mag.shape == (*SHAPE, 3)
    full = nibabel.load(str(prefix) + "_frames_real.nii.gz").dataobj
    part = nibabel.load(str(prefix2) + "_frames_real.nii.gz").dataobj
    assert np.array_equal(np.asarray(part), np.asarray(full)[..., [3, 5, 6]])
    rows = (tmp_path / "o2" / "b_frames.tsv").read_text().splitlines()
    assert [r.split("\t")[0] for r in rows[1:]] == ["3", "5", "6"]
    assert json.loads((tmp_path / "o2" / "b_ksigma.json").read_text()) == json.loads(
        (tmp_path / "o1" / "a_ksigma.json").read_text())


def test_hook_adds_the_noise_frames_to_images_cached_without_the_option(tmp_path, monkeypatch):
    info, samples = _noisy(tmp_path / "s", n_rep=3)
    kw = _hook_env(tmp_path, monkeypatch, info, _encode(samples))
    hook.get_dataobj(_GScan(), None, **kw)                              # the images are cached, no noise frames
    cache = hook._plan(_GScan(), None, dict(kw))["img_cache_path"]
    before = cache.read_bytes()
    calls = []
    real = frames.recon_frames
    monkeypatch.setattr(frames, "recon_frames", lambda *a, **k: calls.append(k.get("noise_out") is not None) or real(*a, **k))
    scan = _GScan()
    hook.convert(scan, hook.get_dataobj(scan, None, frame_export="complex",
                                        frame_export_prefix=str(tmp_path / "o" / "c"), **kw), np.eye(4))
    assert calls == [True]                                              # only the noise pass ran
    assert cache.read_bytes() == before
    assert (tmp_path / "o" / "c_noise_frames_imag.nii.gz").exists()
