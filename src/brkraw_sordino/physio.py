"""Respiration and heart rate from the innermost samples of every spoke (``physio=on``).

Every spoke keeps a few samples nearest to k = 0 after the ``ignore_samples`` trim. Read in the
order of the stream they are one complex sample per spoke and channel at the spoke rate 1/TR, and
breathing and heartbeat modulate their magnitude and phase. The golden-angle direction of a spoke
modulates them too (ramp phase, coil geometry), but the direction list repeats in every
repetition, so the mean of a spoke position over the repetitions holds that fixed part and the
residual holds what changes: physiology, bulk motion and noise.

``Collector`` takes the raw rows of the reconstruction's own chunked FID pass (``feed``), so no
second read of the FID is made; ``extract_stream`` is the one-pass fallback for a recon cache that
already exists. ``Collector.finish`` then builds, per channel and inner sample, the magnitude
change and the phase change of the residual (the first-order split of the residual along and
across the mean sample), band-passes them (zero-phase Butterworth), reduces them to one trace per
band (first principal component, or a sign-aligned mean), and estimates rate and quality in
sliding windows from the dominant spectral peak. Output: ``write`` makes the BIDS-style
``<prefix>_physio.tsv.gz`` (no header) and ``<prefix>_physio.json``.

Assumption: the physiological rhythm is not phase-locked to the repetition period. The per-position
mean removes whatever is the same in every repetition.
"""
from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Option names of this module; they are not in the recon cache key (a side file only).
PHYSIO_KEYS = ("physio", "physio_n_inner", "physio_resp_band", "physio_cardiac_band",
               "physio_window_s", "physio_pca", "physio_prefix", "physio_traj_period",
               "physio_traj_window")

DEFAULT_N_INNER = 6
DEFAULT_RESP_BAND = "0.5,5"
DEFAULT_CARDIAC_BAND = "5,15"
DEFAULT_TRAJ_WINDOW = 1.0       # s: five cycles of the 5 Hz golden-scan period
DEFAULT_WINDOW_S = "3,1"        # respiration, cardiac (a few periods of each)
MIN_USABLE = 0.5                # a window with a smaller usable-spoke fraction has quality 0
COLUMNS = ["time_s", "respiratory", "cardiac", "quality"]


# ---------------------------------------------------------------- options
def parse_physio(value: Any) -> str:
    if value is None:
        return "off"
    text = str(value).strip().lower()
    if text in ("off", "false", "0", "no"):
        return "off"
    if text in ("on", "true", "1", "yes"):
        return "on"
    raise ValueError(f"sordino: physio must be off or on, not {value!r}")


def parse_traj_period(value: Any) -> str:
    """``auto`` (twice the method subset of a golden scan), ``off`` or a period in spokes (an integer >= 2)."""
    if value is None:
        return "auto"
    text = str(value).strip().lower()
    if text in ("auto", "off"):
        return text
    if text in ("false", "0", "no"):
        return "off"
    if text in ("true", "yes", "on"):
        return "auto"
    try:
        n = int(text) if not isinstance(value, bool) else -1
    except ValueError:
        n = -1
    if n < 2:
        raise ValueError(f"sordino: physio_traj_period must be auto, off or a number of spokes >= 2, "
                         f"not {value!r}")
    return str(n)


def parse_traj_window(value: Any):
    """``off`` (the per-repetition cycle mean) or the window in seconds (a positive number) of the local mean."""
    if value is None:
        return DEFAULT_TRAJ_WINDOW
    text = str(value).strip().lower()
    if text in ("off", "false", "no", "0"):
        return "off"
    if text in ("true", "yes", "on"):
        return DEFAULT_TRAJ_WINDOW
    try:
        w = float(text) if not isinstance(value, bool) else -1.0
    except ValueError:
        w = -1.0
    if not np.isfinite(w) or w < 0:
        raise ValueError(f"sordino: physio_traj_window must be off or a window in seconds > 0, not {value!r}")
    return "off" if w == 0 else w


def trajectory_period(options, traj_unit: Optional[int], npro: int) -> Optional[int]:
    """Spokes of the trajectory periodicity to remove from the features, or None.

    Cause (golden scans): the direction list is cut into subsets of ``NGoldenSpokesPerSubset`` spokes and
    every subset sweeps the sphere in z-stacks, north to south in even subsets and south to north in odd
    ones. The innermost samples depend on the direction (ramp phase, coil geometry), so the
    direction-dependent part of a spoke repeats with the subset period and, with the alternating sweep,
    with two subsets. With the 0.625 ms spoke time a subset of 160 spokes is 160 x 0.625 ms = 100 ms =
    10 Hz, two subsets are 320 spokes = 200 ms = 5 Hz: the two frequencies the cardiac peak sits on in the
    scans read. The per-position mean over the repetitions removes only the part that is the same in every
    repetition; a slow change from repetition to repetition (drift, breathing, posture) multiplies the
    direction pattern and leaves the periodic line in the residual.
    """
    mode = getattr(options, "physio_traj_period", "auto")
    if mode == "off":
        return None
    period = int(mode) if mode != "auto" else (2 * int(traj_unit) if traj_unit and int(traj_unit) > 1 else 0)
    return period if 2 <= period < int(npro) else None


def parse_n_inner(value: Any) -> int:
    if value is None:
        return DEFAULT_N_INNER
    try:
        n = int(str(value).strip()) if not isinstance(value, bool) else -1
    except ValueError:
        n = -1
    if n < 1:
        raise ValueError(f"sordino: physio_n_inner must be a positive integer, not {value!r}")
    return n


def _numbers(value: Any, count: int, name: str) -> List[float]:
    if isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = str(value).replace(";", ",").split(",")
    try:
        nums = [float(str(p).strip()) for p in parts if str(p).strip() != ""]
    except ValueError:
        raise ValueError(f"sordino: {name} must be numbers, not {value!r}") from None
    if len(nums) == 1 and count == 2 and name == "physio_window_s":
        nums = nums * 2
    if len(nums) != count or not all(np.isfinite(nums)):
        raise ValueError(f"sordino: {name} needs {count} finite numbers, not {value!r}")
    return nums


def _text(nums: List[float]) -> str:
    return ",".join(f"{v:g}" for v in nums)


def parse_band(value: Any, name: str, default: str) -> str:
    """A band ``low,high`` in Hz as normalised text (``"0.5,5"``)."""
    nums = _numbers(default if value is None else value, 2, name)
    if not 0 < nums[0] < nums[1]:
        raise ValueError(f"sordino: {name} needs 0 < low < high (Hz), not {value!r}")
    return _text(nums)


def parse_window(value: Any) -> str:
    """Window lengths in seconds, ``respiration,cardiac`` (one number serves both)."""
    nums = _numbers(DEFAULT_WINDOW_S if value is None else value, 2, "physio_window_s")
    if min(nums) <= 0:
        raise ValueError(f"sordino: physio_window_s must be positive, not {value!r}")
    return _text(nums)


def bands(options) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    r = _numbers(options.physio_resp_band, 2, "physio_resp_band")
    c = _numbers(options.physio_cardiac_band, 2, "physio_cardiac_band")
    return (r[0], r[1]), (c[0], c[1])


def windows(options) -> Tuple[float, float]:
    w = _numbers(options.physio_window_s, 2, "physio_window_s")
    return w[0], w[1]


def is_on(options) -> bool:
    return getattr(options, "physio", "off") == "on"


# ---------------------------------------------------------------- result
@dataclass
class PhysioResult:
    """Traces at the spoke rate and the sidecar content."""
    sampling_frequency: float
    respiratory: np.ndarray
    cardiac: np.ndarray
    quality: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.respiratory.size)

    def sidecar(self) -> Dict[str, Any]:
        out = {"SamplingFrequency": self.sampling_frequency, "StartTime": 0.0, "Columns": list(COLUMNS)}
        out.update(self.meta)
        return out


# ---------------------------------------------------------------- collector
class Collector:
    """The innermost samples of every spoke of the stream, filled from the reconstruction's reads.

    ``feed(g, k)`` takes raw rows (spokes, receivers, points) of stream spokes ``g .. g + m``, applies the
    phase factor and the off-resonance correction exactly as the reconstruction does, and keeps the first
    ``n_inner`` samples after the ``ignore_samples`` trim. A spoke fed twice (overlapping frames) is
    kept once. Spokes never fed count as unusable.
    """

    def __init__(self, npro: int, n_rep: int, n_rx: int, tr_ms: float, options, *, phase_factor=None,
                 ignore_samples: int = 1, bandwidth=None, over_sampling=None, n_points: Optional[int] = None,
                 censor: Optional[np.ndarray] = None, correct_ramptime: Optional[bool] = None,
                 traj_unit: Optional[int] = None):
        if tr_ms in (None, 0, 0.0):
            raise ValueError("sordino: physio needs the repetition time of the spokes "
                             "(RepetitionTime_ms) to give the traces a time axis")
        if int(n_rep) < 2:
            raise ValueError(f"sordino: physio needs at least 2 repetitions (the direction-fixed part is "
                             f"the mean over repetitions), this run reads {int(n_rep)}")
        self.npro, self.n_rep, self.n_rx = int(npro), int(n_rep), int(n_rx)
        self.n_inner = int(options.physio_n_inner)
        self.ign = int(ignore_samples or 1)
        if n_points is not None and self.ign + self.n_inner > int(n_points):
            raise ValueError(f"sordino: physio_n_inner={self.n_inner} with ignore_samples={self.ign} needs "
                             f"{self.ign + self.n_inner} samples per spoke, the FID has {n_points}")
        self.tr_s = float(tr_ms) * 1e-3
        self.options = options
        self.phase_factor = phase_factor
        offreso = getattr(options, "offreso_freqs", None)
        self.offreso = offreso if (isinstance(offreso, tuple) and bandwidth is not None
                                   and over_sampling is not None) else ()
        self.bw, self.osf = bandwidth, over_sampling
        total = self.npro * self.n_rep
        if censor is not None and np.asarray(censor).size != total:
            raise ValueError(f"physio: the censor has {np.asarray(censor).size} values, the stream {total}")
        self.censor = None if censor is None else np.asarray(censor, dtype=bool).reshape(-1)
        self.correct_ramptime = correct_ramptime
        self.traj_period = trajectory_period(options, traj_unit, self.npro)
        self.traj_window = getattr(options, "physio_traj_window", "off")
        self.store = np.zeros((total, self.n_rx, self.n_inner), dtype=np.complex64)
        self.seen = np.zeros(total, dtype=bool)
        logger.info("physio: collecting %s inner sample(s) of %s spokes and %s channel(s) (%.2f GiB).",
                    self.n_inner, total, self.n_rx, self.store.nbytes / 2 ** 30)

    def feed(self, g: int, k: np.ndarray) -> None:
        from .recon import correct_offreso

        g, m = int(g), int(k.shape[0])
        fresh = ~self.seen[g:g + m]
        if not fresh.any():
            return
        stop = self.ign + self.n_inner
        sl = np.array(k[:, :, :stop], dtype=np.complex128)
        if self.phase_factor is not None:
            a = g % self.npro
            sl = sl * np.asarray(self.phase_factor[a:a + m])[:, None, :stop]
        sl = sl[..., self.ign:]
        for ch in range(self.n_rx):
            if len(self.offreso) > ch and self.offreso[ch] is not None:
                sl[:, ch, :] = correct_offreso(sl[:, ch, :], self.offreso[ch], eff_bandwidth=self.bw,
                                               over_sampling=self.osf)
        block = self.store[g:g + m]
        block[fresh] = sl[fresh]
        self.seen[g:g + m] = True

    # ---------------------------------------------------------------- analysis
    def usable(self) -> np.ndarray:
        use = self.seen.copy()
        if self.censor is not None:
            use &= self.censor
        return use

    def finish(self) -> PhysioResult:
        use = self.usable()
        feats = _features(self.store, use, self.n_rep, self.npro)
        self.store = None
        return analyse(feats, use, self.tr_s, self.options, n_rep=self.n_rep, npro=self.npro,
                       ignore_samples=self.ign, n_rx=self.n_rx, correct_ramptime=self.correct_ramptime,
                       traj_period=self.traj_period, traj_window=self.traj_window)


def _features(store: np.ndarray, use: np.ndarray, n_rep: int, npro: int) -> np.ndarray:
    """(N, F) float32: per channel and inner sample the change along (magnitude) and across (phase, rad)
    the position mean, of the residual to the mean over the usable repetitions; 0 for unusable spokes."""
    n_rx, n_inner = store.shape[1:]
    v = store.reshape(n_rep, npro, n_rx, n_inner)
    u = use.reshape(n_rep, npro)
    cnt = u.sum(axis=0).astype(np.float32)
    mean = np.zeros((npro, n_rx, n_inner), dtype=np.complex64)
    for r in range(n_rep):
        mean += v[r] * u[r][:, None, None]
    mean /= np.maximum(cnt, 1)[:, None, None]
    amp = np.abs(mean)
    floor = 0.1 * float(amp.mean()) if amp.size else 0.0
    safe = np.maximum(amp, max(floor, np.finfo(np.float32).tiny))
    unit = (mean / safe).conj()
    out = np.zeros((n_rep, npro, 2, n_rx, n_inner), dtype=np.float32)
    for r in range(n_rep):
        d = (v[r] - mean) * unit
        good = (u[r] & (cnt >= 2))[:, None, None]       # a position seen once has no residual
        out[r, :, 0] = np.where(good, d.real, 0)
        out[r, :, 1] = np.where(good, d.imag / safe, 0)
    return out.reshape(n_rep * npro, 2 * n_rx * n_inner)


def _remove_period(x: np.ndarray, use: np.ndarray, n_rep: int, npro: int, period: int,
                   half: int = 0) -> None:
    """Subtract, in place, the mean of every position of a ``period``-spoke cycle.

    Removes the trajectory periodicity (``trajectory_period``) and its harmonics, a comb of notches at
    multiples of 1 / (period x TR). With ``half`` = 0 the mean runs over all cycles of a repetition; a
    rhythm that is not such a multiple averages out over the many cycles of a repetition (288000 / 320 = 900
    for the usual scan). With ``half`` >= 1 it is local: for the cycle c the mean of the same position over
    the cycles c - half .. c + half of the repetition (cut at its ends), so a pattern whose amplitude drifts
    within a repetition is removed with it. Only the cycle-shaped part goes: the local mean over all
    positions of the same window is subtracted from the position mean, so a component slower than the window
    (respiration) is constant across the positions and stays. Cumulative sums over the cycles keep the cost at
    O(spokes x features) and the working arrays at the size of one repetition of the features. Unusable
    spokes are left out of every mean and stay 0.
    """
    u = use.reshape(n_rep, npro)
    if half <= 0:
        phase = np.arange(npro) % period
        for r in range(n_rep):
            w = u[r].astype(np.float64)
            cnt = np.bincount(phase, weights=w, minlength=period)
            seen = cnt > 0
            for j in range(x.shape[1]):
                col = x[r * npro:(r + 1) * npro, j]
                mean = np.bincount(phase, weights=col.astype(np.float64) * w, minlength=period)
                mean = np.where(seen, mean / np.maximum(cnt, 1), 0.0)
                col -= (mean[phase] * w).astype(np.float32)
        return
    ncyc = -(-npro // period)
    lo = np.clip(np.arange(ncyc) - half, 0, ncyc)
    hi = np.clip(np.arange(ncyc) + half + 1, 0, ncyc)
    n_f = x.shape[1]
    pad = ncyc * period
    for r in range(n_rep):
        w = np.zeros(pad)
        w[:npro] = u[r]
        blk = np.zeros((pad, n_f))
        blk[:npro] = x[r * npro:(r + 1) * npro]
        blk *= w[:, None]
        cs = np.zeros((ncyc + 1, period, n_f))
        np.cumsum(blk.reshape(ncyc, period, n_f), axis=0, out=cs[1:])
        cc = np.zeros((ncyc + 1, period))
        np.cumsum(w.reshape(ncyc, period), axis=0, out=cc[1:])
        num = cs[hi] - cs[lo]
        den = cc[hi] - cc[lo]
        mean = num / np.maximum(den, 1.0)[:, :, None]
        # what is constant across the cycle positions: the mean over all positions of the same cycles, one
        # value per cycle, interpolated linearly between the cycle centres (a step per cycle would be a
        # staircase with a line at the cycle rate)
        per_cycle = num.sum(axis=1) / np.maximum(den.sum(axis=1), 1.0)[:, None]
        at = np.clip((np.arange(pad) + 0.5) / period - 0.5, 0, ncyc - 1)
        i0 = np.minimum(at.astype(np.int64), max(ncyc - 2, 0))
        i1 = np.minimum(i0 + 1, ncyc - 1)
        frac = (at - i0)[:, None]
        overall = per_cycle[i0] * (1 - frac) + per_cycle[i1] * frac
        blk -= (mean.reshape(pad, n_f) - overall) * w[:, None]
        x[r * npro:(r + 1) * npro] = blk[:npro].astype(np.float32)


def traj_half_window(window, period: int, tr_s: float) -> int:
    """Cycles on each side of the local mean for ``physio_traj_window`` (0 keeps the per-repetition mean)."""
    if window == "off":
        return 0
    cycles = float(window) / (int(period) * float(tr_s))
    return max(1, int(round(cycles)) // 2)


def _band_trace(x: np.ndarray, use: np.ndarray, fs: float, band: Tuple[float, float], pca: bool):
    """Band-pass the features and reduce them to one trace; returns (trace, info)."""
    from scipy.signal import butter, sosfiltfilt

    lo, hi = band
    if hi >= fs / 2:
        raise ValueError(f"sordino: a physio band edge of {hi:g} Hz is not below the Nyquist frequency "
                         f"{fs / 2:g} Hz of the spoke rate")
    sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    y = np.empty_like(x)
    for j in range(x.shape[1]):
        y[:, j] = sosfiltfilt(sos, x[:, j].astype(np.float64)).astype(np.float32)
    y *= use[:, None]
    sd = y[use].std(axis=0) if use.any() else np.zeros(y.shape[1])
    keep = sd > 0
    if not keep.any():
        return np.zeros(x.shape[0], dtype=np.float32), {"features": 0}
    y = y[:, keep] / sd[keep].astype(np.float32)
    n_use = max(int(use.sum()), 1)
    cov = (y.T @ y).astype(np.float64) / n_use
    if pca:
        w, vec = np.linalg.eigh(cov)
        vec = vec[:, -1]
        if vec[np.argmax(np.abs(vec))] < 0:
            vec = -vec
        trace = y @ vec.astype(np.float32)
        return trace, {"features": int(keep.sum()), "explained_variance": float(w[-1] / max(w.sum(), 1e-30))}
    ref = int(np.argmax(np.abs(cov).sum(axis=0)))
    sign = np.where(cov[:, ref] < 0, -1.0, 1.0)
    trace = y @ (sign / y.shape[1]).astype(np.float32)
    return trace, {"features": int(keep.sum())}


def _windowed(trace: np.ndarray, use: np.ndarray, fs: float, band: Tuple[float, float], window_s: float):
    """Window centres (sample index), peak frequency (Hz, nan if unusable) and quality in [0, 1]."""
    length = max(int(round(window_s * fs)), 8)
    step = max(length // 2, 1)
    n = trace.size
    if n < length:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    starts = np.arange(0, n - length + 1, step)
    nfft = 1 << int(np.ceil(np.log2(8 * length)))
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    sel = (freqs >= band[0]) & (freqs <= band[1])
    taper = np.hanning(length).astype(np.float32)
    half = 1.5 / window_s                        # about the main lobe of the window
    rate = np.full(starts.size, np.nan)
    qual = np.zeros(starts.size)
    cs = np.concatenate([[0], np.cumsum(use)])
    for b0 in range(0, starts.size, 256):
        idx = starts[b0:b0 + 256]
        seg = trace[idx[:, None] + np.arange(length)[None, :]] * taper
        power = np.abs(np.fft.rfft(seg, nfft, axis=1)) ** 2
        pb = power[:, sel]
        fb = freqs[sel]
        peak = pb.argmax(axis=1)
        total = pb.sum(axis=1)
        for i in range(idx.size):
            frac = (cs[idx[i] + length] - cs[idx[i]]) / length
            j = b0 + i
            if frac < MIN_USABLE or total[i] <= 0:
                continue
            f0 = fb[peak[i]]
            near = np.abs(fb - f0) <= half
            rate[j] = f0
            qual[j] = float(pb[i, near].sum() / total[i]) * float(frac)
    return starts + length // 2, rate, qual


def _summary(rate: np.ndarray, qual: np.ndarray) -> Dict[str, Any]:
    ok = np.isfinite(rate)
    if not ok.any():
        return {"windows": int(rate.size), "windows_usable": 0, "median_hz": None}
    r = rate[ok]
    return {"windows": int(rate.size), "windows_usable": int(ok.sum()),
            "median_hz": float(np.median(r)), "median_per_min": float(np.median(r) * 60),
            "p05_hz": float(np.percentile(r, 5)), "p95_hz": float(np.percentile(r, 95)),
            "median_quality": float(np.median(qual[ok]))}


def analyse(feats: np.ndarray, use: np.ndarray, tr_s: float, options, *, n_rep: int, npro: int,
            ignore_samples: int, n_rx: int, correct_ramptime=None,
            traj_period: Optional[int] = None, traj_window="off") -> PhysioResult:
    """Traces, rates and quality from the (N, F) features (see ``Collector.finish``).

    ``traj_period`` (spokes) removes the trajectory periodicity from the features before the band-pass,
    by the mean of the same cycle position over the cycles within ``traj_window`` seconds (``off``: over
    the repetition).
    """
    fs = 1.0 / tr_s
    if traj_period:
        half = traj_half_window(traj_window, int(traj_period), tr_s)
        _remove_period(feats, use, int(n_rep), int(npro), int(traj_period), half)
    (rband, cband), (rwin, cwin) = bands(options), windows(options)
    pca = bool(options.physio_pca)
    resp, rinfo = _band_trace(feats, use, fs, rband, pca)
    card, cinfo = _band_trace(feats, use, fs, cband, pca)
    rc, rrate, rq = _windowed(resp, use, fs, rband, rwin)
    cc, crate, cq = _windowed(card, use, fs, cband, cwin)
    n = resp.size
    if rc.size and cc.size:
        quality = 0.5 * (np.interp(np.arange(n), rc, rq) + np.interp(np.arange(n), cc, cq))
    else:
        quality = np.zeros(n)
    meta = {
        "Units": {"respiratory": "a.u.", "cardiac": "a.u.", "quality": "fraction", "time_s": "s"},
        "Method": {
            "name": "innermost spoke samples, per-position repetition mean removed",
            "n_inner": int(options.physio_n_inner), "ignore_samples": int(ignore_samples),
            "channels": int(n_rx), "features": ["magnitude change", "phase change (rad)"],
            "pca": pca, "respiratory_band_hz": list(rband), "cardiac_band_hz": list(cband),
            "window_s": {"respiratory": rwin, "cardiac": cwin},
            "bandpass": "Butterworth order 4, zero phase",
            "rate": "dominant spectral peak of a Hann window, zero padded, stepped by half a window",
            "quality": "mean of the two bands: power within 1.5 / window_s Hz of the peak over the band "
                       "power, times the usable-spoke fraction of the window; linear in between",
            "repetitions": int(n_rep), "spokes_per_repetition": int(npro),
            "spokes_unusable": int((~use).sum()), "correct_ramptime": correct_ramptime,
            "trajectory_period_spokes": int(traj_period) if traj_period else None,
            "respiratory_trace": rinfo, "cardiac_trace": cinfo,
            **({"trajectory_window_s": float(traj_window)} if traj_period and traj_window != "off" else {}),
        },
        "RespiratoryRate": _summary(rrate, rq),
        "CardiacRate": _summary(crate, cq),
    }
    return PhysioResult(fs, resp.astype(np.float32), card.astype(np.float32), quality.astype(np.float32), meta)


# ---------------------------------------------------------------- fallback pass and output
def extract_stream(fid_fobj, recon_info: Dict[str, Any], options, n_rep: int, *, phase_factor=None,
                   chunk_spokes: Optional[int] = None, censor: Optional[np.ndarray] = None) -> PhysioResult:
    """The physio traces from one read of the FID, without reconstruction (used when the images are
    already cached; a reconstruction feeds its own ``Collector`` instead)."""
    from . import golden, serial
    from .recon import _read_chunk, parse_fid_info

    fid_shape, fid_dtype = parse_fid_info(recon_info)
    n_points, n_rx, npro = (int(v) for v in fid_shape[1:])
    spoke_bytes = 2 * n_points * n_rx * np.dtype(fid_dtype).itemsize
    ign = getattr(options, "ignore_samples", None) or 1
    col = Collector(npro, n_rep, n_rx, recon_info.get("RepetitionTime_ms"), options, phase_factor=phase_factor,
                    ignore_samples=ign, bandwidth=recon_info.get("EffBandwidth_Hz"),
                    over_sampling=recon_info.get("OverSampling"), n_points=n_points, censor=censor,
                    correct_ramptime=getattr(options, "correct_ramptime", None),
                    traj_unit=golden.golden_unit(recon_info))
    fid_fobj.seek((getattr(options, "offset", None) or 0) * spoke_bytes * npro)
    for r in range(int(n_rep)):
        for lo, hi in serial.spoke_ranges(npro, chunk_spokes or 4096):
            k = _read_chunk(fid_fobj, spoke_bytes * (hi - lo), fid_dtype, (2, n_points, n_rx, hi - lo))
            col.feed(r * npro + lo, k)
    return col.finish()


def write(result: PhysioResult, prefix) -> Tuple[Path, Path]:
    """Write ``<prefix>_physio.tsv.gz`` (no header; columns in the JSON) and ``<prefix>_physio.json``."""
    prefix = str(prefix)
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    tsv, sidecar = Path(prefix + "_physio.tsv.gz"), Path(prefix + "_physio.json")
    dt = 1.0 / result.sampling_frequency
    with gzip.GzipFile(str(tsv), "wb", mtime=0) as fh:
        for lo in range(0, result.n, 200_000):
            hi = min(lo + 200_000, result.n)
            block = np.column_stack([np.arange(lo, hi) * dt, result.respiratory[lo:hi],
                                     result.cardiac[lo:hi], result.quality[lo:hi]])
            text = "\n".join("\t".join(f"{v:.6g}" if j else f"{v:.7f}" for j, v in enumerate(row))
                             for row in block) + "\n"
            fh.write(text.encode("ascii"))
    sidecar.write_text(json.dumps(result.sidecar(), indent=2) + "\n", encoding="utf-8")
    return tsv, sidecar


__all__ = ["PHYSIO_KEYS", "COLUMNS", "Collector", "PhysioResult", "analyse", "extract_stream", "write",
           "parse_physio", "parse_n_inner", "parse_band", "parse_window", "bands", "windows", "is_on"]
