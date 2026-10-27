"""rep_pattern: removal of the repetition-locked undersampling pattern in the frame engine.

Small synthetic scans (16^3 grid, 800 spokes per repetition, frames of 100 spokes so every
repetition holds 8 positions) with data made by the forward model of a smooth phantom, a
small time signal and noise. No real data.
"""
import io

import numpy as np
import pytest

from brkraw_sordino import frames, kcentre, memguard, serial
from brkraw_sordino.recon import phase_correction_rows
from brkraw_sordino.traj import trajectory_rows

from test_golden_frames import _opts
from test_golden_recon import _golden_info
from test_peri_event_corrections import _encode
from test_serial_recon import N_POINTS, SHAPE

FS = 100                 # spokes per frame: 8 positions per repetition
POSITIONS = 8
GAIN = 1.0e4             # int32 data: the quantisation is far below the pattern
NOISE = 2.0e4            # noise per sample, absolute (the data sum over about 4000 voxels)


def _info(n_rep):
    return dict(_golden_info(n_rep), RepetitionTime_ms=1.0)


def _grid():
    return np.meshgrid(*[np.arange(s) - s / 2 for s in SHAPE], indexing="ij")


def _phantom():
    g = _grid()
    r = np.sqrt(((g[0] - 0.6) / 5.0) ** 2 + ((g[1] + 0.4) / 4.2) ** 2 + (g[2] / 3.5) ** 2)
    return 0.5 * (1 - np.tanh((r - 1.0) * 3.0)) + 0.3 * (np.sqrt(sum(v ** 2 for v in g)) < 2.0)


def _roi():
    g = _grid()
    return 1.0 * (np.sqrt((g[0] - 1.0) ** 2 + (g[1] - 1.0) ** 2 + (g[2] + 0.5) ** 2) < 1.8)


def _samples(tmp_path, info, objects, noise=0.0, seed=7):
    """Raw FID samples (n_rep, NPro, 1, n_points): the forward model of ``objects`` (one per repetition),
    divided by the phase factor the engine multiplies back, plus noise."""
    options = _opts(tmp_path)
    npro = int(info["NPro"])
    tr = trajectory_rows(info, options).rows(0, npro)[:, 1:]
    phase = phase_correction_rows(info, options, N_POINTS)[0:npro]
    rng = np.random.default_rng(seed)
    out = np.zeros((len(objects), npro, 1, N_POINTS), dtype=np.complex128)
    for r, x in enumerate(objects):
        d = GAIN * serial.forward(x.astype(np.complex128), tr).reshape(npro, -1)
        d = d + noise * (rng.standard_normal(d.shape) + 1j * rng.standard_normal(d.shape))
        out[r, :, 0, 1:] = d / phase[:, 1:]
    return out


def _run(tmp_path, info, samples, k0=False, k0_out=None, **kw):
    options = _opts(tmp_path, frame_spokes=FS, estimate_k0=k0, **kw)
    plan = frames.make_plan(info, options)
    vt = kcentre.leading_points(info, 1) if k0 else None
    out = io.BytesIO()
    frames.recon_frames(io.BytesIO(_encode(samples)), trajectory_rows(info, options), info, out, options, plan,
                        phase_factor=phase_correction_rows(info, options, N_POINTS), virtual_traj=vt,
                        k0_out=k0_out, chunk_spokes=200,
                        subtract_rep_mean=bool(kw.get("subtract_rep_mean")))
    data = np.frombuffer(out.getvalue(), dtype=np.complex128).reshape((plan.n_frames, *SHAPE[::-1]))
    return plan, data.transpose(0, 3, 2, 1)


def _variance(images):
    """Variance over the positions of one repetition (the time signal is constant there), mean over
    voxels and repetitions."""
    return float(np.mean(np.var(images.reshape(-1, POSITIONS, *SHAPE), axis=1)))


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ------------------------------------------------------------------ removal and signal
def test_model_removes_the_repeated_pattern_and_keeps_the_time_signal(tmp_path):
    info = _info(3)
    x, roi = _phantom(), _roi()
    amp = (0.0, 0.1, -0.1)
    samples = _samples(tmp_path, info, [x + a * roi for a in amp], noise=NOISE)
    plan, none = _run(tmp_path / "none", info, samples)
    _, model = _run(tmp_path / "model", info, samples, rep_pattern="model")
    assert plan.n_frames == 3 * POSITIONS
    _, floor = _run(tmp_path / "floor", info, _samples(tmp_path, info, [0 * x] * 3, noise=NOISE, seed=9))
    assert _variance(none) > 3 * _variance(floor)               # the pattern is there
    assert _variance(model) < 1.5 * _variance(floor)            # what is left is the noise
    assert _variance(model) < 0.15 * _variance(none)
    # the mean over the positions of one repetition is the repetition image: the pattern sums to zero,
    # so the image of every repetition, and with it the time signal, is kept
    mean_none = none.reshape(3, POSITIONS, *SHAPE).mean(axis=1)
    mean_model = model.reshape(3, POSITIONS, *SHAPE).mean(axis=1)
    signal = np.linalg.norm(mean_none[1] - mean_none[0])
    assert signal > 0
    for r in range(3):
        assert np.linalg.norm(mean_model[r] - mean_none[r]) < 1e-3 * signal


def test_model_and_measured_agree_for_a_static_noise_free_object(tmp_path):
    info = _info(2)
    samples = _samples(tmp_path, info, [_phantom()] * 2)
    _, none = _run(tmp_path / "none", info, samples)
    _, model = _run(tmp_path / "model", info, samples, rep_pattern="model")
    _, measured = _run(tmp_path / "measured", info, samples, rep_pattern="measured")
    assert _rel(model, measured) < 0.05
    assert _rel(model, measured) < 0.2 * _rel(none, measured)
    # measured is one image for every frame: the pattern is gone exactly
    assert _rel(measured, np.broadcast_to(measured[:1], measured.shape)) < 1e-6


def test_model_and_measured_agree_with_estimate_k0(tmp_path):
    info = _info(2)
    samples = _samples(tmp_path, info, [_phantom()] * 2)
    k0_model, k0_measured = [], []
    _, model = _run(tmp_path / "model", info, samples, k0=True, k0_out=k0_model, rep_pattern="model")
    _, measured = _run(tmp_path / "measured", info, samples, k0=True, k0_out=k0_measured,
                       rep_pattern="measured")
    np.testing.assert_allclose(k0_model, k0_measured, rtol=1e-9)
    assert _rel(model, measured) < 0.05


# ------------------------------------------------------------------ estimate_k0 with a repetition mean
def test_rep_mean_with_estimate_k0_solves_from_raw_data_and_adds_no_centre_term(tmp_path):
    info = _info(2)
    samples = _samples(tmp_path, info, [_phantom()] * 2)
    k0_plain, k0_mean = [], []
    _run(tmp_path / "plain", info, samples, k0=True, k0_out=k0_plain)
    _, residual = _run(tmp_path / "mean", info, samples, k0=True, k0_out=k0_mean, subtract_rep_mean=True)
    assert abs(k0_plain[0][0]) > 0
    np.testing.assert_allclose(k0_mean, k0_plain, rtol=1e-9)    # the solve saw the object, not zeros
    np.testing.assert_allclose(residual, 0, atol=1e-12)         # identical repetitions: no centre term left


# ------------------------------------------------------------------ guards, key, memory
def test_option_values_and_guards(tmp_path):
    assert frames.parse_rep_pattern(None) == "none"
    assert frames.parse_rep_pattern(" Model ") == "model"
    assert _opts(tmp_path).rep_pattern == "none"
    with pytest.raises(ValueError, match="rep_pattern must be"):
        _opts(tmp_path, rep_pattern="mean")
    info = _info(2)
    with pytest.raises(ValueError, match="sordino: rep_pattern needs a frame plan"):
        frames.make_plan(info, _opts(tmp_path, frame_spokes="repetition", rep_pattern="model"))
    with pytest.raises(ValueError, match="cannot be used together"):
        frames.make_plan(info, _opts(tmp_path, frame_spokes=FS, rep_pattern="model", subtract_rep_mean=True))
    censor = tmp_path / "censor.1D"
    np.savetxt(censor, np.ones(160, dtype=int), fmt="%d")
    with pytest.raises(ValueError, match="peri_censor"):
        frames.make_plan(info, _opts(tmp_path, peri_event="period=0.32,bin=0.04", peri_censor=str(censor),
                                     rep_pattern="measured"))
    with pytest.raises(ValueError, match="at least 2 repetitions"):
        frames.make_plan(_info(1), _opts(tmp_path, frame_spokes=FS, rep_pattern="measured"))
    assert frames.make_plan(_info(1), _opts(tmp_path, frame_spokes=FS, rep_pattern="model")).n_frames == 8


def test_key_entry_and_memory_estimate(tmp_path):
    options = _opts(tmp_path, frame_spokes=FS, rep_pattern="model")
    assert frames.rep_pattern_key(options, _info(2)) == {"mode": "model", "xhat": "cg", "n_iter": kcentre.N_ITER}
    plain = memguard.frames_nbytes(2, 3, (4, 4, 4))
    assert memguard.frames_nbytes(2, 3, (4, 4, 4), rep_pattern=True) - plain == 3 * 3 * 64 * 16


def test_normal_on_is_the_hermitian_normal_operator_of_the_samples():
    rng = np.random.default_rng(3)
    shape = (6, 6, 6)
    traj = rng.uniform(-0.5, 0.5, (300, 3))
    w = rng.uniform(0.5, 1.5, 300)
    x = rng.standard_normal((2, *shape)) + 1j * rng.standard_normal((2, *shape))
    y = rng.standard_normal((2, *shape)) + 1j * rng.standard_normal((2, *shape))
    nx, ny = serial.normal_on(x, traj, w), serial.normal_on(y, traj, w)
    np.testing.assert_allclose(serial.normal_on(x[1], traj, w), nx[1], rtol=1e-12)
    assert abs(np.vdot(y, nx) - np.conj(np.vdot(x, ny))) < 1e-5 * abs(np.vdot(y, nx))
    assert np.vdot(x, nx).real > 0
