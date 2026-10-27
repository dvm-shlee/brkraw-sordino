"""Peri-event reconstruction stage: empty bins, default bin count, censored repetition mean,
options that need a frame plan, and the memory estimate of the per-bin accumulators.

Small synthetic scans (the geometry of ``test_peri_event_corrections``) except the memory test,
which only plans: it reads no data.
"""
import io
import logging

import numpy as np
import pytest

from brkraw_sordino import frames, hook, memguard
from brkraw_sordino.recon import phase_correction_rows
from brkraw_sordino.traj import trajectory_rows

from test_golden_frames import _opts
from test_golden_frames_cp4 import VOL_BYTES, _big_env
from test_golden_recon import _Scan
from test_peri_event_corrections import SPEC, _data, _encode
from test_serial_recon import SHAPE


def _run(tmp_path, info, samples, spec=SPEC, *, subtract=False, **kwargs):
    options = _opts(tmp_path, peri_event=spec, **kwargs)
    plan = frames.make_plan(info, options)
    output = io.BytesIO()
    frames.recon_frames(
        io.BytesIO(_encode(samples)), trajectory_rows(info, options), info, output, options, plan,
        phase_factor=phase_correction_rows(info, options, samples.shape[-1]),
        chunk_spokes=53, subtract_rep_mean=subtract,
    )
    images = np.frombuffer(output.getvalue(), dtype=np.complex128).reshape(
        (plan.n_frames, *SHAPE[::-1])).transpose(0, 3, 2, 1)
    return plan, images


# ------------------------------------------------------------------ a. empty bins
def test_empty_bins_warn_and_are_all_zero_frames(tmp_path, caplog):
    info, samples = _data()
    # 60 bins of 40 spokes span 2400 spokes of each event; only 1600 are read, so bins 40.. are empty
    with caplog.at_level(logging.WARNING, logger="brkraw_sordino.perievent"):
        plan, images = _run(tmp_path, info, samples, "period=0.32,bin=0.04,bins=60")
    assert plan.n_frames == 60
    empty = [b for b, n in enumerate(plan.peri["events_per_bin"]) if n == 0]
    assert empty == list(range(40, 60))
    assert any("all-zero" in r.getMessage() and "40" in r.getMessage() for r in caplog.records)
    assert all(plan.scales[b] == 0.0 for b in empty)
    assert np.all(images[40:] == 0)
    assert np.linalg.norm(images[0]) > 0


def test_a_plan_without_any_spokes_is_still_refused(tmp_path):
    info, _ = _data()
    with pytest.raises(ValueError, match="no spokes|places no event"):
        frames.make_plan(info, _opts(tmp_path, peri_event="period=0.32,bin=0.04,onset=5"))


# ------------------------------------------------------------------ b. default bin count
def test_default_bins_cover_the_whole_period_when_the_bin_does_not_divide_it(tmp_path, caplog):
    info, _ = _data()
    # period 300 spokes, bin 40 spokes: floor would give 7 bins (280 spokes), ceil gives 8 (320)
    with caplog.at_level(logging.WARNING, logger="brkraw_sordino.perievent"):
        plan = frames.make_plan(info, _opts(tmp_path, peri_event="period=0.3,bin=0.04"))
    assert plan.n_frames == 8
    assert plan.peri["bin_edges_s"][-1] * 1000 / info["RepetitionTime_ms"] >= 300
    assert any("does not divide the period" in r.getMessage() for r in caplog.records)


def test_default_bins_of_a_dividing_bin_are_unchanged(tmp_path, caplog):
    info, _ = _data()
    with caplog.at_level(logging.WARNING, logger="brkraw_sordino.perievent"):
        plan = frames.make_plan(info, _opts(tmp_path, peri_event=SPEC))
    assert plan.n_frames == 8
    assert not any("does not divide" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------------ c. censored repetition mean
def test_rep_mean_leaves_censored_spokes_out(tmp_path):
    info, samples = _data()
    mask = np.ones(160, dtype=int)
    mask[85] = 0                    # spokes 850-859: positions 50-59 of repetition 1
    censor = tmp_path / "censor.1D"
    np.savetxt(censor, mask, fmt="%d")
    keep = np.repeat(mask.astype(bool), 10).reshape(2, -1)
    summed = (samples * keep[:, :, None, None]).sum(axis=0)
    mean = summed / np.maximum(keep.sum(axis=0), 1)[:, None, None]
    plan, actual = _run(tmp_path / "a", info, samples, subtract=True, peri_censor=str(censor))
    _, expected = _run(tmp_path / "b", info, samples - mean[None], peri_censor=str(censor))
    assert np.linalg.norm(expected) > 0
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    # the mean over all repetitions (censored spokes included) gives a different image
    naive = samples - samples.mean(axis=0, keepdims=True)
    _, leaked = _run(tmp_path / "c", info, naive, peri_censor=str(censor))
    assert not np.allclose(actual, leaked, rtol=1e-6, atol=1e-6)


# ------------------------------------------------------------------ d. options without a frame plan
def test_subtract_rep_mean_without_a_frame_plan_is_refused(tmp_path):
    info, _ = _data()
    with pytest.raises(ValueError, match=r"sordino: subtract_rep_mean needs a frame plan"):
        frames.make_plan(info, _opts(tmp_path, frame_spokes="repetition", subtract_rep_mean=True))


def test_peri_censor_without_peri_event_is_refused(tmp_path):
    info, _ = _data()
    censor = tmp_path / "censor.1D"
    np.savetxt(censor, np.ones(160, dtype=int), fmt="%d")
    for extra in ({}, {"frame_spokes": 160}, {"frame_spokes": "repetition"}):
        with pytest.raises(ValueError, match=r"sordino: peri_censor needs peri_event"):
            frames.make_plan(info, _opts(tmp_path, peri_censor=str(censor), **extra))


def test_subtract_rep_mean_with_a_frame_option_is_accepted(tmp_path):
    info, _ = _data()
    plan = frames.make_plan(info, _opts(tmp_path, frame_spokes=160, subtract_rep_mean=True))
    assert plan is not None and plan.groups is None


# ------------------------------------------------------------------ e. memory guard
def test_peri_plan_estimate_holds_one_accumulator_per_bin(tmp_path, monkeypatch):
    _big_env(monkeypatch)
    one = hook.get_dataobj_info(_Scan(), None, cache_dir=str(tmp_path), peri_event="period=20,bin=1,bins=1")
    many = hook.get_dataobj_info(_Scan(), None, cache_dir=str(tmp_path), peri_event="period=20,bin=1")
    assert one["frames"] == 1 and many["frames"] == 20
    assert many["recon_frames_open"] == 20
    assert many["recon_nbytes"] - one["recon_nbytes"] == 19 * VOL_BYTES


def test_peri_plan_with_many_bins_is_refused_under_a_small_limit(tmp_path, monkeypatch):
    _big_env(monkeypatch)
    monkeypatch.setattr(memguard, "free_disk_bytes", lambda path: 10 ** 15)
    called = []
    monkeypatch.setattr(frames, "recon_frames", lambda *a, **k: called.append(1))
    kw = {"cache_dir": str(tmp_path / "r")}
    one = hook.get_dataobj_info(_Scan(), None, peri_event="period=20,bin=1,bins=1", **kw)
    # room for the one-bin plan and a few volumes, far short of twenty accumulators
    gb = (one["peak_nbytes"] + 5 * VOL_BYTES) / memguard.GIB
    memguard.check(hook.get_dataobj_info(_Scan(), None, peri_event="period=20,bin=1,bins=1",
                                         max_memory_gb=gb, **kw))
    with pytest.raises(memguard.SordinoResourceError) as exc:
        hook.get_dataobj(_Scan(), None, peri_event="period=20,bin=1", max_memory_gb=gb, **kw)
    assert exc.value.kind == "memory" and not called
