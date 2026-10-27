"""Synthetic regression checks for censoring and repetition residuals.

Use exact, even-valued integer FIDs so the two-repetition mean and its residuals
have no quantization error.
"""
import io

import numpy as np
import pytest

from brkraw_sordino import frames
from brkraw_sordino.recon import phase_correction_rows
from brkraw_sordino.traj import trajectory_rows

from test_golden_frames import _direct, _opts, _pieces, WINDOW_TOL
from test_golden_recon import _golden_info
from test_serial_recon import N_POINTS, SHAPE, _rel


SPEC = "period=0.32,bin=0.04"


def _data(n_rep=2):
    info = dict(_golden_info(n_rep), RepetitionTime_ms=1.0)
    rng = np.random.default_rng(804)
    shape = (n_rep, info["NPro"], 1, N_POINTS)
    samples = 2 * (rng.integers(-20, 21, shape) + 1j * rng.integers(-20, 21, shape))
    return info, samples


def _encode(samples):
    return b"".join(np.stack([k.real.T, k.imag.T]).astype("<i4").tobytes(order="F")
                    for k in samples)


def _run(tmp_path, info, samples, *, subtract=False, **kwargs):
    options = _opts(tmp_path, peri_event=SPEC, **kwargs)
    plan = frames.make_plan(info, options)
    output = io.BytesIO()
    frames.recon_frames(
        io.BytesIO(_encode(samples)), trajectory_rows(info, options), info, output, options, plan,
        phase_factor=phase_correction_rows(info, options, N_POINTS),
        chunk_spokes=53, subtract_rep_mean=subtract,
    )
    images = np.frombuffer(output.getvalue(), dtype=np.complex128).reshape(
        (plan.n_frames, *SHAPE[::-1])).transpose(0, 3, 2, 1)
    return plan, images


def test_censor_drops_whole_ranges_and_normalizes_each_surviving_bin(tmp_path):
    info, samples = _data()
    # Each row covers ten spokes. Even one censored spoke invalidates its
    # complete forty-spoke event range; the other spokes are not clipped in.
    mask = np.ones(160, dtype=int)
    mask[[1, 35, 44]] = 0
    censor = tmp_path / "censor.1D"
    np.savetxt(censor, mask, fmt="%d")
    plain = frames.make_plan(info, _opts(tmp_path / "plain", peri_event=SPEC))
    plan, images = _run(tmp_path / "censored", info, samples, peri_censor=str(censor))
    removed = {(0, 40), (320, 360), (440, 480)}
    assert set(plain.ranges) - set(plan.ranges) == removed
    assert set(plan.ranges) == set(plain.ranges) - removed
    assert plan.peri["censored_ranges"] == 3
    counts = [3, 5, 5, 4, 5, 5, 5, 5]
    assert plan.peri["events_per_bin"] == counts
    np.testing.assert_allclose(plan.scales, np.array([800 / (40 * n) for n in counts]))
    assert plan.key() != plain.key()
    # Independent whole-array reconstruction of precisely the surviving spokes
    # also checks that censoring and the per-bin scale affect the written image.
    for b in (0, 3):
        pieces = [piece for span, group in zip(plan.ranges, plan.groups) if group == b
                  for piece in _pieces(span, info["NPro"])]
        reference = _direct(tmp_path / f"reference{b}", info, _encode(samples), pieces, plan.scales[b])
        assert _rel(images[b], reference) < WINDOW_TOL


def test_rep_mean_matches_explicit_residual_fid(tmp_path):
    info, samples = _data()
    residual = samples - samples.mean(axis=0, keepdims=True)
    plan, actual = _run(tmp_path / "subtract", info, samples, subtract=True)
    reference_plan, expected = _run(tmp_path / "explicit", info, residual)
    assert plan.ranges == reference_plan.ranges
    assert np.linalg.norm(expected) > 0
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_rep_mean_removes_identical_repetitions(tmp_path):
    info, samples = _data()
    samples[1] = samples[0]
    _, plain = _run(tmp_path / "plain", info, samples)
    _, residual = _run(tmp_path / "residual", info, samples, subtract=True)
    assert np.linalg.norm(plain) > 0
    np.testing.assert_allclose(residual, 0, atol=1e-12)


def test_rep_mean_requires_multiple_repetitions(tmp_path):
    info, samples = _data(n_rep=1)
    with pytest.raises(ValueError, match="at least 2 repetitions"):
        _run(tmp_path, info, samples, subtract=True)
