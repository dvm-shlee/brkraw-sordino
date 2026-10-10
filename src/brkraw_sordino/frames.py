"""Frames of golden-angle scans: method subsets, spoke counts, sliding windows, accumulation (WI-0113 CP3).

Options (``TrajectoryMode`` GoldenSampling or GoldenGridSampling only; D-0184 2, D-0189 4, D-0190, D-0191):

* ``frame_spokes``: default (None) and ``"subset"`` are the method subset, the
  smallest spoke count that covers the sphere: ``NGoldenSpokesPerSubset`` for
  Golden Sampling (1 with ``GoldenReorder=No``), one grid frame (cells, x 2 with
  ``GridMirror``) for Golden Grid (D-0190 1). ``"repetition"`` keeps one
  repetition per frame, the reconstruction of ``recon.recon_dataobj``
  (unchanged). An integer is a spoke count.
* ``frame_step``: spokes between frame starts; default ``frame_spokes`` (frames
  side by side). Smaller gives a sliding window, larger leaves spokes out.
* ``frame_accumulate``: every frame starts at the first spoke; frame k ends
  after ``frame_spokes + k * frame_step`` spokes, so with the default step the
  frames hold 1, 2, 3 ... x ``frame_spokes`` (D-0189 4). The start does not move.

The repetitions read (``offset``, ``num_frames``) form one stream of spokes;
global spoke g is spoke g mod NPro of repetition g // NPro (D-0191 2), so
windows run across repetition boundaries. Such windows, and windows that start
inside a method subset, are reported, not refused (D-0187).

Reconstruction: a frame image is the adjoint of its spokes with the product's
density weight |k|^2 normalised over one repetition (as ``recon_dataobj``), times
NPro / (spokes in the frame), so every frame has the brightness of a
one-repetition image. The engine keeps one running sum over the stream and a
copy of it at each frame start; a frame is the difference at its end, so every
spoke is reconstructed once however much the frames overlap.

``estimate_k0`` with frames (D-0191 4): the centre is estimated once from all
spokes read (one least-squares solve, ``kcentre``), and its predicted leading
samples are added spoke by spoke; every frame uses that one estimate, and the
frames stay linear in their spokes. ``correct_spoketiming`` is refused with
frames: it moves every spoke of a repetition to one time point.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import frameplan, golden

logger = logging.getLogger(__name__)

#: Options of this module; they enter the recon cache key through ``FramePlan.key``.
FRAME_KEYS = ("frame_spokes", "frame_step", "frame_accumulate")

#: Measured in WI-0112 (scan 11, 3,200-spoke windows): largest gap between spoke
#: directions from a subset start, from the middle of a subset, across a repetition end.
GAP_SUBSET_START_DEG = 3.58
GAP_INSIDE_SUBSET_DEG = 4.50
GAP_ACROSS_REPETITIONS_DEG = 4.27


# ---------------------------------------------------------------- options
def parse_frame_spokes(value: Any):
    """None, ``"subset"``, ``"repetition"`` or a positive spoke count (ValueError otherwise)."""
    msg = f"frame_spokes must be 'subset', 'repetition' or a positive spoke count, got {value!r}"
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(msg)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("subset", "repetition"):
            return text
        try:
            n = int(text)
        except ValueError:
            raise ValueError(msg) from None
    elif isinstance(value, (int, np.integer)):
        n = int(value)
    elif isinstance(value, (float, np.floating)):
        if not float(value).is_integer():
            raise ValueError(msg)
        n = int(value)
    else:
        raise ValueError(msg)
    if n <= 0:
        raise ValueError(msg)
    return n


def parse_frame_step(value: Any) -> Optional[int]:
    """None or a positive spoke count (ValueError otherwise)."""
    msg = f"frame_step must be a positive spoke count, got {value!r}"
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(msg)
    if isinstance(value, str):
        try:
            n = int(value.strip())
        except ValueError:
            raise ValueError(msg) from None
    elif isinstance(value, (int, np.integer)):
        n = int(value)
    elif isinstance(value, (float, np.floating)):
        if not float(value).is_integer():
            raise ValueError(msg)
        n = int(value)
    else:
        raise ValueError(msg)
    if n <= 0:
        raise ValueError(msg)
    return n


def _explicit(options) -> bool:
    fs = getattr(options, "frame_spokes", None)
    return (fs not in (None, "repetition") or getattr(options, "frame_step", None) is not None
            or bool(getattr(options, "frame_accumulate", False)))


def is_frame_mode(options, recon_info: Dict[str, Any]) -> bool:
    """True when the frames are not one repetition each (golden default, or any frame option)."""
    if golden.trajectory_mode(recon_info) == "Default":
        return _explicit(options)
    return getattr(options, "frame_spokes", None) != "repetition" or _explicit(options)


# ---------------------------------------------------------------- plan
@dataclass(frozen=True)
class FramePlan:
    """Spoke ranges of the frames in the stream of the repetitions read."""

    trajectory_mode: str
    unit: int
    npro: int
    n_rep: int
    window: int
    step: int
    accumulate: bool
    ranges: Tuple[Tuple[int, int], ...]
    scales: Tuple[float, ...]
    crossing: int
    misaligned: int
    tr_ms: Optional[float]

    @property
    def n_frames(self) -> int:
        return len(self.ranges)

    @property
    def frame_interval_s(self) -> Optional[float]:
        """Time between frames: ``frame_step`` x the spoke TR (D-0190 2); None without a TR."""
        return None if self.tr_ms is None else self.step * self.tr_ms * 1e-3

    def key(self) -> Dict[str, Any]:
        """What decides the frame images (recon cache key)."""
        return {"unit": self.unit, "window": self.window, "step": self.step, "accumulate": self.accumulate}

    def describe(self) -> Dict[str, Any]:
        """Result metadata (``scan._sordino_recon_meta["frames"]``)."""
        tr = self.tr_ms
        return {
            "trajectory_mode": self.trajectory_mode, "unit_spokes": self.unit,
            "frame_spokes": self.window, "frame_step": self.step, "accumulate": self.accumulate,
            "n_frames": self.n_frames, "repetitions": self.n_rep, "spokes_per_repetition": self.npro,
            "spoke_ranges": [[lo, hi] for lo, hi in self.ranges],
            "frames_crossing_repetitions": self.crossing, "frames_inside_subsets": self.misaligned,
            "spoke_tr_ms": tr, "frame_interval_s": self.frame_interval_s,
            "frame_centre_s": None if tr is None else [(lo + hi) / 2 * tr * 1e-3 for lo, hi in self.ranges],
            "k0_scope": None,
        }


def make_plan(recon_info: Dict[str, Any], options) -> Optional[FramePlan]:
    """The frame plan, or None for one repetition per frame (``recon_dataobj``)."""
    from .recon import get_num_frames

    if not is_frame_mode(options, recon_info):
        return None
    mode = golden.trajectory_mode(recon_info)
    if mode == "Default":
        raise ValueError(
            "sordino: frame_spokes, frame_step and frame_accumulate apply to golden trajectories only "
            "(TrajectoryMode GoldenSampling or GoldenGridSampling). This scan has TrajectoryMode Default, "
            "whose spokes cover the sphere only as a whole repetition.")
    if getattr(options, "correct_spoketiming", False):
        raise ValueError(
            "sordino: correct_spoketiming cannot be used with golden frames: it moves every spoke of a "
            "repetition to one time point, so frames shorter than a repetition would all show that time. "
            "Use frame_spokes='repetition' with correct_spoketiming.")
    unit = int(golden.golden_unit(recon_info))
    npro = int(recon_info["NPro"])
    n_rep = int(get_num_frames(recon_info, options))
    total = n_rep * npro
    fs = options.frame_spokes
    window = unit if fs in (None, "subset") else npro if fs == "repetition" else int(fs)
    step = int(options.frame_step) if options.frame_step is not None else window
    accumulate = bool(options.frame_accumulate)
    if window > total:
        raise ValueError(f"sordino: frame_spokes {window} is more than the {total} spokes read "
                         f"({n_rep} repetition(s) of {npro}).")
    ranges = tuple(tuple(r) for r in frameplan.frame_ranges(total, window, step, accumulate))
    ends = [hi for _, hi in ranges]
    assert all(b > a for a, b in zip(ends, ends[1:])), "frame ends must increase"
    scales = tuple(frameplan.frame_scales(ranges, npro))
    crossing = frameplan.crossing_count(ranges, npro)
    misaligned = frameplan.misaligned_count(ranges, unit) if unit > 1 else 0
    tr = recon_info.get("RepetitionTime_ms")
    tr_ms = float(tr) if tr not in (None, 0, 0.0) else None
    if misaligned:
        logger.warning(
            "sordino: %s of %s frames start or end inside a method subset (%s spokes). Any spoke range can "
            "be reconstructed, but a window that starts inside a subset covers the sphere less evenly "
            "(WI-0112, 3,200 spokes: largest gap %.2f deg from inside a subset, %.2f deg from a subset "
            "start). Multiples of %s keep whole subsets.",
            misaligned, len(ranges), unit, GAP_INSIDE_SUBSET_DEG, GAP_SUBSET_START_DEG, unit)
    if crossing:
        logger.info(
            "sordino: %s of %s frames cross a repetition boundary; the repetitions are read as one stream "
            "(WI-0112: a window across the boundary had a largest gap of %.2f deg against %.2f deg inside).",
            crossing, len(ranges), GAP_ACROSS_REPETITIONS_DEG, GAP_SUBSET_START_DEG)
    if window > npro and not accumulate:
        logger.info("sordino: frames of %s spokes are longer than one repetition (%s); spokes beyond one "
                    "repetition repeat its directions.", window, npro)
    left = total - ranges[-1][1]
    if left and not accumulate:
        logger.info("sordino: the last %s spoke(s) read are in no frame.", left)
    return FramePlan(mode, unit, npro, n_rep, window, step, accumulate, ranges, scales, crossing,
                     misaligned, tr_ms)


# ---------------------------------------------------------------- engine
class _Stream:
    """Spokes of the FID stream, read forward from the first repetition read (rewind allowed)."""

    def __init__(self, fobj, start: int, spoke_bytes: int, dtype, n_points: int, n_rx: int):
        self.fobj, self.start, self.spoke_bytes = fobj, int(start), int(spoke_bytes)
        self.dtype, self.n_points, self.n_rx = dtype, int(n_points), int(n_rx)
        self.rewind()

    def rewind(self) -> None:
        self.fobj.seek(self.start)
        self.pos = 0

    def read(self, g: int, n: int) -> np.ndarray:
        """Spokes g .. g + n of the stream as (spokes, receivers, points) complex128."""
        from .recon import _read_chunk

        if g < self.pos:
            self.rewind()
        skip = (g - self.pos) * self.spoke_bytes
        while skip > 0:
            got = self.fobj.read(min(skip, 64 * 1024 ** 2))
            if not got:
                raise ValueError("FID data ended early")
            skip -= len(got)
        k = _read_chunk(self.fobj, self.spoke_bytes * n, self.dtype, (2, self.n_points, self.n_rx, n))
        self.pos = g + n
        return k


def recon_frames(fid_fobj, traj, recon_info: Dict[str, Any], img_fobj, options, plan: FramePlan, *,
                 phase_factor=None, virtual_traj=None, k0_out=None, chunk_spokes=None, k0_method=None):
    """Write the frames of ``plan`` to ``img_fobj`` (the layout of ``recon_dataobj``); return their dtype.

    ``traj`` (``TrajectoryRows`` or an array), ``phase_factor`` (``PhaseRows`` or an array) and
    ``virtual_traj`` (``kcentre.leading_points``) are those of one repetition; the stream repeats
    them. ``k0_out`` gets one list (one K0 per channel) for the whole stream.
    """
    from . import memguard, serial
    from .recon import _row_source, correct_offreso, parse_fid_info, parse_volume_shape

    img_fobj.seek(0)
    fid_shape, fid_dtype = parse_fid_info(recon_info)
    vol = [int(v) for v in parse_volume_shape(recon_info, options)]
    n_points, n_rx, npro = (int(v) for v in fid_shape[1:])
    if npro != plan.npro:
        raise ValueError(f"frame plan is for {plan.npro} spokes per repetition, the FID has {npro}")
    ign = getattr(options, "ignore_samples", None) or 1
    offset = getattr(options, "offset", None) or 0
    item = np.dtype(fid_dtype).itemsize
    spoke_bytes = 2 * n_points * n_rx * item
    stream = _Stream(fid_fobj, offset * spoke_bytes * npro, spoke_bytes, fid_dtype, n_points, n_rx)
    rows = _row_source(traj)
    traj_spokes = int(traj.n_pro) if hasattr(traj, "n_pro") else int(np.shape(traj)[0])
    if traj_spokes != npro:
        raise ValueError(f"trajectory has {traj_spokes} spokes, the FID has {npro}")
    if chunk_spokes is None:
        chunk_spokes = memguard.recon_plan(npro, n_points, n_rx, vol,
                                           estimate_k0=virtual_traj is not None)["chunk_spokes"]
    offreso = getattr(options, "offreso_freqs", None)
    bw, osf = recon_info.get("EffBandwidth_Hz"), recon_info.get("OverSampling")

    def _offreso(ch):
        if isinstance(offreso, tuple) and len(offreso) > ch and bw is not None and osf is not None:
            return offreso[ch]
        return None

    # density maximum over one repetition (and the virtual samples), as recon_dataobj; with
    # estimate_k0 the normal operator of the n_rep repetitions (n_rep x one repetition's kernel)
    kernel = None
    if virtual_traj is not None:
        choice = memguard.k0_method(npro, n_points, n_rx, vol)
        method = choice["method"] if k0_method is None else str(k0_method)
        if method == "toeplitz":
            kernel = serial.ToeplitzKernel(vol)
        elif method == "samples":
            kernel = serial.SampleNormal(vol, npro * (n_points - ign))
        else:
            raise ValueError(f"k0_method must be None, 'toeplitz' or 'samples', not {k0_method!r}")
    dmax = 0.0
    for lo, hi in serial.spoke_ranges(npro, chunk_spokes):
        tr = rows(lo, hi)[:, ign:]
        d = serial.density(tr)
        dmax = max(dmax, float(d.max()))
        if kernel is not None:
            kernel.add(tr, d)
        del tr, d
    w_virtual = None
    if virtual_traj is not None:
        dmax = max(dmax, float(serial.density(virtual_traj).max()))
        kernel.finish(float(plan.n_rep) / dmax)
        n_virtual = int(virtual_traj.shape[1])
        w_virtual = (serial.density(virtual_traj) / dmax).reshape(npro, n_virtual)
    nf = serial.norm_factor(vol)
    adj = serial.Adjoint(vol)

    def add_range(acc, g_lo, g_hi, preds=None):
        """acc += raw adjoint of stream spokes g_lo .. g_hi (and their predicted centre samples)."""
        g = g_lo
        while g < g_hi:
            r = g // npro
            lo = g - r * npro
            hi = min(g_hi - r * npro, npro)
            for a, b in serial.spoke_ranges(hi - lo, chunk_spokes):
                a, b = a + lo, b + lo
                k = stream.read(r * npro + a, b - a)
                if phase_factor is not None:
                    k = k * phase_factor[a:b][:, None, :]
                k = k[..., ign:]
                for ch in range(n_rx):
                    freq = _offreso(ch)
                    if freq is not None:
                        k[:, ch, :] = correct_offreso(k[:, ch, :], freq, eff_bandwidth=bw, over_sampling=osf)
                tr = rows(a, b)[:, ign:]
                w = serial.density(tr) / dmax
                adj.setpts(tr)
                del tr
                for ch in range(n_rx):
                    adj.add(acc[ch], k[:, ch, :].reshape(-1) * w)
                del k
                if preds is not None:
                    adj.setpts(virtual_traj[a:b])
                    wv = w_virtual[a:b].reshape(-1)
                    for ch in range(n_rx):
                        adj.add(acc[ch], preds[ch][a:b].reshape(-1) * wv)
            g = r * npro + hi

    shape = (n_rx,) + tuple(vol)
    preds = None
    if virtual_traj is not None:
        from .kcentre import N_ITER

        full = np.zeros(shape, dtype=np.complex128)
        add_range(full, 0, plan.n_rep * npro)
        stream.rewind()
        preds, k0 = [], []
        for ch in range(n_rx):
            x, _ = serial.conjugate_gradient(kernel.normal, full[ch], N_ITER)
            k0.append(complex(x.sum()))
            preds.append(serial.forward(x, virtual_traj).reshape(npro, -1))
            del x
        del full, kernel
        if k0_out is not None:
            k0_out.append(k0)
        logger.info("estimate_k0 with frames: one estimate from all %s spokes read, used by every frame.",
                    plan.n_rep * npro)

    starts_last: Dict[int, int] = {}
    for idx, (lo, _) in enumerate(plan.ranges):
        starts_last[lo] = idx
    opens: Dict[int, int] = {}
    closes: Dict[int, int] = {}
    for lo, hi in plan.ranges:
        opens[lo] = opens.get(lo, 0) + 1
        closes[hi] = closes.get(hi, 0) + 1
    points = sorted(set(opens) | set(closes))
    run = np.zeros(shape, dtype=np.complex128)
    snaps: Dict[int, np.ndarray] = {}
    nxt, active = 0, 0
    for i, b in enumerate(points):
        while nxt < plan.n_frames and plan.ranges[nxt][1] == b:
            lo = plan.ranges[nxt][0]
            img = (run - snaps[lo]) * (plan.scales[nxt] / nf)
            out = img if n_rx > 1 else img[0]
            img_fobj.write(np.ascontiguousarray(out.T).tobytes())
            del img, out
            if starts_last[lo] == nxt:
                del snaps[lo]
            nxt += 1
        if b in starts_last and starts_last[b] >= nxt:
            snaps[b] = run.copy()
        active += opens.get(b, 0) - closes.get(b, 0)
        if i + 1 < len(points) and active > 0:
            add_range(run, b, points[i + 1], preds)
    assert nxt == plan.n_frames
    logger.debug("frames: %s written (%s x %s spokes, step %s, accumulate %s)", plan.n_frames, plan.n_frames,
                 plan.window, plan.step, plan.accumulate)
    return np.dtype(np.complex128)


__all__ = ["FRAME_KEYS", "FramePlan", "parse_frame_spokes", "parse_frame_step", "is_frame_mode",
           "make_plan", "recon_frames"]
