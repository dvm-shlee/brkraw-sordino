"""physio: respiration and heart rate from the innermost spoke samples (physio.py).

Synthetic FID rows only: a direction-fixed complex pattern per spoke position (the same in every
repetition, as the golden-angle list is), an injected modulation of magnitude and phase at a chosen
frequency, and noise. No real data.
"""
import gzip
import io
import json

import numpy as np
import pytest

from brkraw_sordino import frames, golden, hook, physio, recon
from brkraw_sordino.recon import phase_correction_rows
from brkraw_sordino.traj import trajectory_rows

from test_golden_frames import _opts
from test_golden_recon import _golden_info
from test_hook_read import _Fid, _Scan, _recon_info
from test_peri_event_corrections import _encode
from test_serial_recon import N_POINTS

N_REP, NPRO, N_RX, N_INNER = 16, 4000, 2, 4
TR_MS = 0.625                      # 1600 Hz, a repetition of 2.5 s (16 repetitions: the mean over repetitions carries 1/16 of the modulation)
SIGMA = 0.02                       # noise per real or imaginary part
RESP_HZ, CARD_HZ = 3.1234, 10.37   # not locked to the repetition period (2.5 s)
WINDOWS = "2,0.5"


def _options(tmp_path, **kw):
    kw.setdefault("physio", "on")
    kw.setdefault("physio_n_inner", N_INNER)
    kw.setdefault("physio_window_s", WINDOWS)
    return hook._build_options({"cache_dir": str(tmp_path), **kw})


def _inner(resp=None, card=None, pattern=5.0, seed=3, noise=SIGMA, n_rep=N_REP):
    """Inner samples (n_rep * NPRO, N_RX, N_INNER): the position pattern times the modulation, plus noise."""
    rng = np.random.default_rng(seed)
    base = pattern * (rng.standard_normal((NPRO, N_RX, N_INNER)) + 1j * rng.standard_normal((NPRO, N_RX, N_INNER)))
    base += 3.0                                    # the position pattern has a common offset
    t = np.arange(n_rep * NPRO) * TR_MS * 1e-3
    m = np.zeros_like(t)
    for hz, amp in ((resp, 0.05), (card, 0.04)):
        if hz is not None:
            m += amp * np.sin(2 * np.pi * hz * t + 0.7)
    v = np.tile(base, (n_rep, 1, 1)) * ((1 + m) * np.exp(1j * m))[:, None, None]
    v = v + noise * (rng.standard_normal(v.shape) + 1j * rng.standard_normal(v.shape))
    return v, m


def _raw(inner, ign=1, extra=2):
    """Raw rows (spokes, receivers, points): ``ign`` leading samples, the inner ones, ``extra`` after."""
    k = np.zeros(inner.shape[:2] + (ign + inner.shape[2] + extra,), dtype=np.complex128)
    k[:, :, ign:ign + inner.shape[2]] = inner
    k[:, :, :ign] = 99.0
    return k


def _collect(options, inner, chunk=1500, censor=None):
    col = physio.Collector(NPRO, N_REP, N_RX, TR_MS, options, ignore_samples=1,
                           n_points=inner.shape[2] + 3, censor=censor)
    raw = _raw(inner)
    for lo in range(0, raw.shape[0], chunk):
        col.feed(lo, raw[lo:lo + chunk])
    return col


def _band_corr(trace, m, fs, band):
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
    ref = sosfiltfilt(sos, m)
    edge = int(0.5 * fs)
    a, b = trace[edge:-edge], ref[edge:-edge]
    return abs(float(np.corrcoef(a, b)[0, 1]))


# ------------------------------------------------------------------ options
def test_defaults_and_parsing(tmp_path):
    o = hook._build_options({"cache_dir": str(tmp_path)})
    assert (o.physio, o.physio_n_inner, o.physio_resp_band, o.physio_cardiac_band, o.physio_window_s,
            o.physio_pca, o.physio_prefix) == ("off", 6, "0.5,5", "5,15", "3,1", True, None)
    o = hook._build_options({"cache_dir": str(tmp_path), "physio": "ON", "physio_n_inner": "8",
                             "physio_resp_band": [0.4, 6], "physio_cardiac_band": "6 , 14",
                             "physio_window_s": 2.5, "physio_pca": "no"})
    assert (o.physio, o.physio_n_inner, o.physio_resp_band, o.physio_cardiac_band, o.physio_window_s,
            o.physio_pca) == ("on", 8, "0.4,6", "6,14", "2.5,2.5", False)
    assert physio.bands(o) == ((0.4, 6.0), (6.0, 14.0)) and physio.windows(o) == (2.5, 2.5)


@pytest.mark.parametrize("kw", [
    {"physio": "maybe"}, {"physio_n_inner": 0}, {"physio_n_inner": "x"}, {"physio_resp_band": "5,0.5"},
    {"physio_resp_band": "1"}, {"physio_cardiac_band": "a,b"}, {"physio_window_s": "0,1"},
    {"physio_window_s": "1,2,3"},
])
def test_bad_options_are_refused_by_name(tmp_path, kw):
    with pytest.raises(ValueError, match="physio"):
        hook._build_options({"cache_dir": str(tmp_path), **kw})


def test_physio_keys_are_not_in_the_recon_cache_key(tmp_path):
    def params(**kw):
        options = hook._build_options(dict(cache_dir=str(tmp_path), **kw))
        return hook._build_cache_params(_Scan(), None, _Fid(), options, _recon_info(1))

    plain = params()
    on = params(physio="on", physio_n_inner=8, physio_resp_band="1,4", physio_cardiac_band="6,12",
                physio_window_s="4,2", physio_pca="false", physio_prefix=str(tmp_path / "x"))
    assert on["options"] == plain["options"]            # recon_info holds arrays: compare the keyed options and the hashed path
    assert not [k for k in plain["options"] if k.startswith("physio")]
    assert set(physio.PHYSIO_KEYS) <= set(frames.FRAME_KEYS)
    assert recon.build_recon_cache_path(tmp_path, on) == recon.build_recon_cache_path(tmp_path, plain)


# ------------------------------------------------------------------ extraction
@pytest.mark.parametrize("pca", [True, False])
def test_injected_rates_are_recovered(tmp_path, pca):
    inner, m = _inner(resp=RESP_HZ, card=CARD_HZ)
    res = _collect(_options(tmp_path, physio_pca=pca), inner).finish()
    assert res.sampling_frequency == pytest.approx(1600.0)
    assert res.n == N_REP * NPRO
    r, c = res.meta["RespiratoryRate"], res.meta["CardiacRate"]
    assert abs(r["median_hz"] - RESP_HZ) < 0.1                    # stated tolerance: 0.1 Hz
    assert abs(c["median_hz"] - CARD_HZ) < 0.3                    # stated tolerance: 0.3 Hz
    assert r["windows_usable"] == r["windows"] and c["windows_usable"] == c["windows"]
    assert _band_corr(res.respiratory, m, 1600.0, (0.5, 5)) > 0.9
    assert _band_corr(res.cardiac, m, 1600.0, (5, 15)) > 0.9
    assert 0 <= res.quality.min() and res.quality.max() <= 1


def test_each_band_alone(tmp_path):
    for kw, key, hz, tol in (({"resp": 0.97}, "RespiratoryRate", 0.97, 0.1),
                             ({"resp": 1.9}, "RespiratoryRate", 1.9, 0.1),
                             ({"resp": 4.6}, "RespiratoryRate", 4.6, 0.1),
                             ({"card": 6.2}, "CardiacRate", 6.2, 0.3),
                             ({"card": 8.1}, "CardiacRate", 8.1, 0.3),
                             ({"card": 13.4}, "CardiacRate", 13.4, 0.3)):
        inner, _ = _inner(seed=11, **kw)
        res = _collect(_options(tmp_path), inner).finish()
        assert abs(res.meta[key]["median_hz"] - hz) < tol, (kw, res.meta[key])


def test_direction_only_control_is_at_the_noise_floor(tmp_path):
    # a strong direction-fixed pattern, no physiology: the per-position mean removes it
    inner, _ = _inner(pattern=5.0)
    col = _collect(_options(tmp_path), inner)
    use = col.usable()
    feats = physio._features(col.store, use, N_REP, NPRO)
    n_feat = feats.shape[1] // 2
    mag = feats.reshape(-1, 2, N_RX, N_INNER)[:, 0]
    # residual of the mean over R repetitions has variance sigma^2 (1 - 1/R)
    expected = SIGMA * np.sqrt(1 - 1 / N_REP)
    assert np.std(mag) == pytest.approx(expected, rel=0.1)
    raw = inner.real.reshape(N_REP, NPRO, N_RX, N_INNER)
    assert np.std(raw) > 50 * np.std(mag)                         # what the pattern was before removal
    assert n_feat == N_RX * N_INNER
    # and no band shows a rhythm: quality stays below that of a modulated run
    res = col.finish()
    mod = _collect(_options(tmp_path), _inner(resp=RESP_HZ, card=CARD_HZ)[0]).finish()
    assert np.median(res.quality) + 0.15 < np.median(mod.quality)


def test_a_phase_locked_component_is_removed_with_the_pattern(tmp_path):
    # the stated assumption: a rhythm that repeats exactly every repetition is the position mean
    inner, _ = _inner(resp=4.0, noise=0.0)                       # 4 Hz x 2.5 s = 10 cycles
    feats = physio._features(_collect(_options(tmp_path), inner).store,
                             np.ones(N_REP * NPRO, dtype=bool), N_REP, NPRO)
    assert np.abs(feats).max() < 1e-3


def test_censored_and_unread_spokes_lower_the_quality(tmp_path):
    inner, _ = _inner(resp=RESP_HZ, card=CARD_HZ)
    censor = np.ones(N_REP * NPRO, dtype=bool)
    censor[8000:12000] = False                                   # 2.5 s
    res = _collect(_options(tmp_path), inner, censor=censor).finish()
    assert res.meta["Method"]["spokes_unusable"] == 4000
    assert res.meta["RespiratoryRate"]["windows_usable"] < res.meta["RespiratoryRate"]["windows"]
    assert res.quality[9000:11000].max() < 0.2 * res.quality[20000:30000].mean()
    assert abs(res.meta["RespiratoryRate"]["median_hz"] - RESP_HZ) < 0.1


def test_a_spoke_fed_twice_is_kept_once_and_refusals(tmp_path):
    inner, _ = _inner(resp=RESP_HZ)
    once = _collect(_options(tmp_path), inner)
    twice = _collect(_options(tmp_path), inner)
    twice.feed(0, 2 * _raw(inner)[:1500])                        # a second, different read of the same spokes
    assert np.array_equal(once.store, twice.store)
    with pytest.raises(ValueError, match="at least 2 repetitions"):
        physio.Collector(NPRO, 1, N_RX, TR_MS, _options(tmp_path))
    with pytest.raises(ValueError, match="RepetitionTime_ms"):
        physio.Collector(NPRO, N_REP, N_RX, None, _options(tmp_path))
    with pytest.raises(ValueError, match="physio_n_inner"):
        physio.Collector(NPRO, N_REP, N_RX, TR_MS, _options(tmp_path, physio_n_inner=9), n_points=8)


def test_phase_factor_and_offreso_are_applied_as_the_reconstruction_does(tmp_path):
    inner, _ = _inner(resp=RESP_HZ, n_rep=N_REP)
    rng = np.random.default_rng(1)
    phase = np.exp(1j * rng.uniform(-3, 3, (NPRO, 1 + N_INNER + 2)))
    options = _options(tmp_path, offreso_freqs="120,-80")
    col = physio.Collector(NPRO, N_REP, N_RX, TR_MS, options, phase_factor=phase, ignore_samples=1,
                           bandwidth=1.0e5, over_sampling=1.0, n_points=N_INNER + 3)
    raw = _raw(inner)
    col.feed(NPRO, raw[NPRO:NPRO + 100])                          # repetition 1, spoke positions 0..99
    got = col.store[NPRO:NPRO + 100]
    want = (raw[NPRO:NPRO + 100] * phase[:100][:, None, :])[..., 1:1 + N_INNER]
    for ch, hz in enumerate((120.0, -80.0)):
        want[:, ch] = recon.correct_offreso(want[:, ch], hz, eff_bandwidth=1.0e5, over_sampling=1.0)
    assert np.allclose(got, want.astype(np.complex64), atol=1e-4)


# ------------------------------------------------------------------ output files
def test_files_follow_the_physio_layout(tmp_path):
    inner, _ = _inner(resp=RESP_HZ, card=CARD_HZ)
    res = _collect(_options(tmp_path), inner).finish()
    tsv, js = physio.write(res, tmp_path / "out" / "sub-1_task-x")
    assert tsv.name == "sub-1_task-x_physio.tsv.gz" and js.name == "sub-1_task-x_physio.json"
    with gzip.open(tsv, "rt") as fh:
        rows = [line.rstrip("\n").split("\t") for line in fh]
    assert len(rows) == res.n and all(len(r) == 4 for r in rows)
    assert float(rows[0][0]) == 0.0 and float(rows[1][0]) == pytest.approx(0.625e-3)
    assert not rows[0][0].isalpha()                              # no header line
    side = json.loads(js.read_text())
    assert side["SamplingFrequency"] == pytest.approx(1600.0) and side["StartTime"] == 0.0
    assert side["Columns"] == ["time_s", "respiratory", "cardiac", "quality"]
    meth = side["Method"]
    assert meth["respiratory_band_hz"] == [0.5, 5.0] and meth["cardiac_band_hz"] == [5.0, 15.0]
    assert meth["n_inner"] == N_INNER and meth["pca"] is True
    assert float(rows[5000][1]) == pytest.approx(float(res.respiratory[5000]), rel=1e-4, abs=1e-5)


# ------------------------------------------------------------------ the reconstruction pass
def _frames_run(tmp_path, collector=None, **kw):
    from test_rep_pattern import _samples, _phantom

    info = dict(_golden_info(3), RepetitionTime_ms=1.0)
    x = _phantom()
    samples = _samples(tmp_path, info, [x, 1.1 * x, 0.9 * x], noise=2.0e4)
    options = _opts(tmp_path, frame_spokes=100, **kw)
    plan = frames.make_plan(info, options)
    out = io.BytesIO()
    frames.recon_frames(io.BytesIO(_encode(samples)), trajectory_rows(info, options), info, out, options, plan,
                        phase_factor=phase_correction_rows(info, options, N_POINTS), chunk_spokes=200,
                        physio=collector)
    return out.getvalue(), info, options, samples


def test_images_are_byte_identical_with_and_without_the_physio_pass(tmp_path):
    base, info, options, samples = _frames_run(tmp_path / "off")
    n_pro = int(info["NPro"])
    opts = _opts(tmp_path / "on", physio="on", physio_n_inner=2, physio_window_s="0.2,0.1",
                 physio_resp_band="2,40", physio_cardiac_band="45,200")
    col = physio.Collector(n_pro, 3, 1, 1.0, opts, phase_factor=phase_correction_rows(info, opts, N_POINTS),
                           ignore_samples=1, n_points=N_POINTS)
    got, *_ = _frames_run(tmp_path / "on", collector=col, physio="on", physio_n_inner=2)
    assert got == base                                           # physio=on leaves the images alone
    assert col.seen.all()                                        # every spoke came from the engine's reads
    res = col.finish()
    assert res.n == 3 * n_pro and np.isfinite(res.respiratory).all()


def test_the_physio_pass_equals_the_stand_alone_read(tmp_path):
    # the traces of the engine's own reads and of extract_stream (a recon cache already there) agree
    from test_rep_pattern import _samples, _phantom

    info = dict(_golden_info(3), RepetitionTime_ms=1.0)
    x = _phantom()
    samples = _samples(tmp_path, info, [x, 1.1 * x, 0.9 * x], noise=2.0e4)
    n_pro = int(info["NPro"])
    opts = _opts(tmp_path, physio="on", physio_n_inner=2, physio_window_s="0.2,0.1",
                 physio_resp_band="2,40", physio_cardiac_band="45,200", frame_spokes=100)
    phase = phase_correction_rows(info, opts, N_POINTS)
    plan = frames.make_plan(info, opts)
    col = physio.Collector(n_pro, 3, 1, 1.0, opts, phase_factor=phase, ignore_samples=1, n_points=N_POINTS,
                           traj_unit=golden.golden_unit(info))
    frames.recon_frames(io.BytesIO(_encode(samples)), trajectory_rows(info, opts), info, io.BytesIO(), opts,
                        plan, phase_factor=phase, chunk_spokes=200, physio=col)
    ref = col.finish()
    alone = physio.extract_stream(io.BytesIO(_encode(samples)), info, opts, 3, phase_factor=phase,
                                  chunk_spokes=150)
    assert np.array_equal(ref.respiratory, alone.respiratory)
    assert np.array_equal(ref.cardiac, alone.cardiac)
    assert np.array_equal(ref.quality, alone.quality)


# ------------------------------------------------------------------ pre-change identity
def _base_module(name):
    """A module of the code before the physio change, loaded from git next to the current package."""
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
    if shown.returncode or "physio" in shown.stdout.split("def ", 1)[0] and name == "frames":
        pytest.skip("the integration branch already holds the physio change")
    spec = importlib.util.spec_from_loader(f"brkraw_sordino._base_{name}", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "brkraw_sordino"
    sys.modules[spec.name] = mod
    exec(compile(shown.stdout, f"<base {name}>", "exec"), mod.__dict__)
    return mod


def test_off_mode_matches_the_pre_change_code(tmp_path):
    from test_rep_pattern import _samples, _phantom

    base_frames = _base_module("frames")
    base_hook = _base_module("hook")
    info = dict(_golden_info(3), RepetitionTime_ms=1.0)
    x = _phantom()
    samples = _samples(tmp_path, info, [x, 1.1 * x, 0.9 * x], noise=2.0e4)
    options = _opts(tmp_path, frame_spokes=100)
    plan = base_frames.make_plan(info, options)
    out = io.BytesIO()
    base_frames.recon_frames(io.BytesIO(_encode(samples)), trajectory_rows(info, options), info, out, options,
                             plan, phase_factor=phase_correction_rows(info, options, N_POINTS),
                             chunk_spokes=200)
    now, *_ = _frames_run(tmp_path / "now")
    assert now == out.getvalue()                       # off-mode images: the same bytes as before the change
    # off-mode cache entry: the keyed options and the path are those of the pre-change code
    opt = hook._build_options({"cache_dir": str(tmp_path)})
    new_params = hook._build_cache_params(_Scan(), None, _Fid(), opt, _recon_info(1))
    old_params = base_hook._build_cache_params(_Scan(), None, _Fid(), opt, _recon_info(1))
    assert new_params["options"] == old_params["options"]
    assert recon.build_recon_cache_path(tmp_path, new_params) == recon.build_recon_cache_path(tmp_path, old_params)


# ------------------------------------------------------------------ trajectory periodicity
UNIT = 80                          # spokes of a method subset in the synthetic scan (period 2 x 80 = 160 spokes = 10 Hz)
SLOW_CARD_HZ = 7.3                 # not a multiple of 1600 / 160 Hz


def _periodic_inner(seed=5):
    """Cardiac modulation plus a drift that changes from repetition to repetition and multiplies a direction
    pattern with a period of 160 spokes (100 ms at 1600 Hz): the residual carries a 10 Hz line."""
    rng = np.random.default_rng(seed)
    a = np.arange(NPRO)
    base = (3.0 + 0.1 * (rng.standard_normal((NPRO, N_RX, N_INNER)) + 1j * rng.standard_normal((NPRO, N_RX, N_INNER))))
    base *= (1 + 0.8 * np.cos(2 * np.pi * a / (2 * UNIT)))[:, None, None]
    t = np.arange(N_REP * NPRO) * TR_MS * 1e-3
    drift = np.repeat(0.3 * rng.standard_normal(N_REP), NPRO)
    m = 0.04 * np.sin(2 * np.pi * SLOW_CARD_HZ * t + 0.7)
    v = np.tile(base, (N_REP, 1, 1)) * (1 + drift + m)[:, None, None]
    return v + SIGMA * (rng.standard_normal(v.shape) + 1j * rng.standard_normal(v.shape))


def _collect_unit(options, inner, unit=UNIT):
    col = physio.Collector(NPRO, N_REP, N_RX, TR_MS, options, ignore_samples=1,
                           n_points=inner.shape[2] + 3, traj_unit=unit)
    raw = _raw(inner)
    for lo in range(0, raw.shape[0], 1500):
        col.feed(lo, raw[lo:lo + 1500])
    return col


def test_trajectory_periodicity_is_removed_before_the_cardiac_rate(tmp_path):
    inner = _periodic_inner()
    old = _collect_unit(_options(tmp_path, physio_traj_period="off"), inner).finish()
    assert abs(old.meta["CardiacRate"]["median_hz"] - 10.0) < 0.3          # the artefact: the 10 Hz line wins
    assert old.meta["Method"]["trajectory_period_spokes"] is None
    new = _collect_unit(_options(tmp_path), inner).finish()                # auto: 2 x unit = 160 spokes
    assert new.meta["Method"]["trajectory_period_spokes"] == 2 * UNIT
    assert abs(new.meta["CardiacRate"]["median_hz"] - SLOW_CARD_HZ) < 0.3


def test_trajectory_period_option_and_no_unit(tmp_path):
    assert physio.parse_traj_period(None) == "auto" and physio.parse_traj_period(" OFF ") == "off"
    assert physio.parse_traj_period(160) == "160"
    for bad in ("1", "x", -3, "1.5"):
        with pytest.raises(ValueError, match="physio_traj_period"):
            physio.parse_traj_period(bad)
    assert _options(tmp_path).physio_traj_period == "auto"
    inner = _periodic_inner()
    assert _collect_unit(_options(tmp_path), inner, unit=None).traj_period is None      # no golden unit: nothing removed
    assert _collect_unit(_options(tmp_path, physio_traj_period=160), inner, unit=None).traj_period == 160


# ------------------------------------------------------------------ local cycle mean
LOC_P, LOC_NPRO, LOC_REP, LOC_F = 320, 16000, 2, 6      # 320-spoke pattern, 10 s per repetition at 1600 Hz


def _line_power(x, f, bw=0.25):
    spec = np.abs(np.fft.rfft(x[:LOC_NPRO], axis=0)) ** 2
    fr = np.fft.rfftfreq(LOC_NPRO, TR_MS * 1e-3)
    return spec[(fr > f - bw) & (fr < f + bw)].sum()


def _local_case(seed=1):
    """A 320-spoke pattern whose amplitude carries a slow drift and a 0.7 Hz sinusoid, plus a 7 Hz sinusoid."""
    rng = np.random.default_rng(seed)
    t = np.arange(LOC_NPRO) * TR_MS * 1e-3
    pattern = np.tile(rng.standard_normal((LOC_P, LOC_F)) + 2, (LOC_NPRO // LOC_P, 1))
    reps = []
    for r in range(LOC_REP):
        drift = 1 + 0.3 * np.cumsum(rng.standard_normal(LOC_NPRO)) / 60
        reps.append(pattern * (drift + 0.2 * np.sin(2 * np.pi * 0.7 * t + r))[:, None])
    card = np.tile(0.05 * np.sin(2 * np.pi * 7 * t), LOC_REP)[:, None] * np.ones(LOC_F)
    card = card + np.tile(0.3 * np.sin(2 * np.pi * 0.6 * t), LOC_REP)[:, None] * np.ones(LOC_F)
    noise = 0.01 * rng.standard_normal((LOC_REP * LOC_NPRO, LOC_F))
    return np.vstack(reps), card, noise


def test_local_cycle_mean_removes_the_pattern_lines_and_keeps_the_cardiac_rhythm():
    import time

    pattern, card, noise = _local_case()
    use = np.ones(LOC_REP * LOC_NPRO, dtype=bool)
    half = physio.traj_half_window(physio.parse_traj_window(None), LOC_P, TR_MS * 1e-3)
    assert half == 2                                                       # 1 s = 5 cycles of 200 ms
    x = (pattern + card + noise).astype(np.float32)
    before = x.copy()
    t0 = time.perf_counter()
    physio._remove_period(x, use, LOC_REP, LOC_NPRO, LOC_P, half)
    elapsed = time.perf_counter() - t0
    for hz in (5, 10):
        drop = 10 * np.log10(_line_power(before, hz) / _line_power(x, hz))
        assert drop >= 20, (hz, drop)
    alone = card.astype(np.float32)                                        # the operator is linear
    physio._remove_period(alone, use, LOC_REP, LOC_NPRO, LOC_P, half)
    assert _line_power(alone, 7) >= 0.9 * _line_power(card, 7)
    assert _line_power(alone, 0.6) >= 0.9 * _line_power(card, 0.6)         # slow physiology stays
    assert elapsed < 5.0                                                   # O(spokes x features)
    # the per-repetition mean leaves the drift behind
    y = before.copy()
    physio._remove_period(y, use, LOC_REP, LOC_NPRO, LOC_P, 0)
    assert 10 * np.log10(_line_power(before, 5) / _line_power(y, 5)) < 20


def test_local_cycle_mean_leaves_unusable_spokes_out_and_at_zero():
    pattern, card, noise = _local_case(seed=2)
    use = np.ones(LOC_REP * LOC_NPRO, dtype=bool)
    use[500:900] = False
    use[LOC_NPRO + 40::97] = False
    x = (pattern + card + noise).astype(np.float32)
    x[~use] = 0
    physio._remove_period(x, use, LOC_REP, LOC_NPRO, LOC_P, 2)
    assert np.all(x[~use] == 0) and np.isfinite(x).all()
    assert abs(x[use]).mean() < 0.2 * abs(pattern).mean()


def test_stationary_pattern_is_removed_by_the_local_mean_as_before(tmp_path):
    inner = _periodic_inner()
    for window in ("1", "off"):
        res = _collect_unit(_options(tmp_path, physio_traj_window=window), inner).finish()
        assert abs(res.meta["CardiacRate"]["median_hz"] - SLOW_CARD_HZ) < 0.3
    res = _collect_unit(_options(tmp_path), inner).finish()
    assert res.meta["Method"]["trajectory_window_s"] == 1.0


def test_traj_window_option_and_cache_key(tmp_path):
    assert physio.parse_traj_window(None) == 1.0 and physio.parse_traj_window(" OFF ") == "off"
    assert physio.parse_traj_window(0) == "off" and physio.parse_traj_window(0.5) == 0.5
    for bad in ("x", -1, "nan"):
        with pytest.raises(ValueError, match="physio_traj_window"):
            physio.parse_traj_window(bad)
    assert "physio_traj_window" in physio.PHYSIO_KEYS
    assert physio.traj_half_window("off", 320, 6.25e-4) == 0
    assert physio.traj_half_window("0.2", 320, 6.25e-4) == 1


def test_period_off_matches_the_pre_change_code(tmp_path):
    base = _base_module("physio")
    inner = _periodic_inner()
    new = _collect_unit(_options(tmp_path, physio_traj_period="off"), inner).finish()
    col = base.Collector(NPRO, N_REP, N_RX, TR_MS, _options(tmp_path, physio_traj_period="off"),
                         ignore_samples=1, n_points=inner.shape[2] + 3, traj_unit=UNIT)
    raw = _raw(inner)
    for lo in range(0, raw.shape[0], 1500):
        col.feed(lo, raw[lo:lo + 1500])
    old = col.finish()
    assert new.cardiac.tobytes() == old.cardiac.tobytes() and new.respiratory.tobytes() == old.respiratory.tobytes()
    assert new.quality.tobytes() == old.quality.tobytes()
