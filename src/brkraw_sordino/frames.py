"""Frames of golden-angle scans: method subsets, spoke counts, sliding windows, accumulation (WI-0113 CP3).

Options (golden trajectories, and since D-0194 the Default list with a spoke count; D-0184 2,
D-0189 4, D-0190, D-0191, D-0194):

* ``frame_spokes``: default (None) and ``"subset"`` are the method subset, the
  smallest spoke count that covers the sphere: ``NGoldenSpokesPerSubset`` for
  Golden Sampling (1 with ``GoldenReorder=No``), one grid frame (cells, x 2 with
  ``GridMirror``) for Golden Grid (D-0190 1). ``"repetition"`` keeps one
  repetition per frame, the reconstruction of ``recon.recon_dataobj``
  (unchanged). ``"all"`` is one frame of every spoke read (``anat_solver``). An integer is a spoke count.
* ``frame_step``: spokes between frame starts; default ``frame_spokes`` (frames
  side by side). Smaller gives a sliding window, larger leaves spokes out.
* ``frame_accumulate``: every frame starts at the first spoke; frame k ends
  after ``frame_spokes + k * frame_step`` spokes, so with the default step the
  frames hold 1, 2, 3 ... x ``frame_spokes`` (D-0189 4). The start does not move.

The repetitions read (``offset``, ``num_frames``) form one stream of spokes;
global spoke g is spoke g mod NPro of repetition g // NPro (D-0191 2), so
windows run across repetition boundaries. Such windows, and windows that start
inside a method subset, are reported, not refused (D-0187).

Reconstruction: a frame image is the adjoint of its spokes with the density
weights of one repetition (as ``recon_dataobj``: |k|^2 normalised over the
repetition, or the ``weights`` the hook passes for golden scans, the
``golden_weights`` choice, WI-0113 runs 5-6), times
NPro / (spokes in the frame), so every frame has the brightness of a
one-repetition image. The engine keeps one running sum over the stream and a
copy of it at each frame start; a frame is the difference at its end, so every
spoke is reconstructed once however much the frames overlap.

``estimate_k0`` with frames (D-0191 4): the centre is estimated once from all
spokes read (one least-squares solve, ``kcentre``), and its predicted leading
samples are added spoke by spoke; every frame uses that one estimate, and the
frames stay linear in their spokes. ``correct_spoketiming`` is refused with
frames: it moves every spoke of a repetition to one time point.

``rep_pattern`` (``model`` or ``measured``): every frame position k has the same spokes in every
repetition, so every frame carries the same undersampling pattern. ``model`` subtracts
P_k = (s_k / nf) T_k x_hat - x_full from the complex frame, with T_k = A_k^H W_k A_k the normal
operator of the frame's own kept samples, x_hat a few CG steps on the all-spoke normal operator and
x_full the all-spoke image at the frame gain; ``measured`` writes the residual of ``subtract_rep_mean``
plus x_full. Theory: rep-pattern-removal (the technical note, section 10.2).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import frameplan, golden, perievent

logger = logging.getLogger(__name__)

#: Options of this module; they enter the recon cache key through ``FramePlan.key``.
FRAME_KEYS = ("frame_spokes", "frame_step", "frame_accumulate") + perievent.PERI_KEYS

#: Measured in WI-0112 (scan 11, 3,200-spoke windows): largest gap between spoke
#: directions from a subset start, from the middle of a subset, across a repetition end.
GAP_SUBSET_START_DEG = 3.58
GAP_INSIDE_SUBSET_DEG = 4.50
GAP_ACROSS_REPETITIONS_DEG = 4.27


# ---------------------------------------------------------------- options
def parse_frame_spokes(value: Any):
    """None, ``"subset"``, ``"repetition"``, ``"all"`` or a positive spoke count (ValueError otherwise)."""
    msg = f"frame_spokes must be 'subset', 'repetition', 'all' or a positive spoke count, got {value!r}"
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(msg)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("subset", "repetition", "all"):
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


REP_PATTERNS = ("none", "model", "measured")


def parse_rep_pattern(value: Any) -> str:
    """``none`` (also None and False), ``model`` or ``measured`` (ValueError otherwise)."""
    if value is None or value is False:
        return "none"
    text = value.strip().lower() if isinstance(value, str) else None
    if text in REP_PATTERNS:
        return text
    if text in ("", "false", "off"):
        return "none"
    raise ValueError(f"sordino: rep_pattern must be 'none', 'model' or 'measured', got {value!r}")


ANAT_SOLVERS = ("adjoint", "cg")
ANAT_ITER_DEFAULT = 10


def parse_anat_solver(value: Any) -> str:
    """``adjoint`` (also None) or ``cg`` (ValueError otherwise)."""
    text = "adjoint" if value is None else (value.strip().lower() if isinstance(value, str) else None)
    if text in ANAT_SOLVERS:
        return text
    raise ValueError(f"sordino: anat_solver must be 'adjoint' or 'cg', got {value!r}")


def _parse_iter(value: Any, name: str) -> int:
    msg = f"sordino: {name} must be a positive iteration count, got {value!r}"
    if value is None:
        return ANAT_ITER_DEFAULT
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(msg)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(msg) from None
    if not number.is_integer() or number < 1:
        raise ValueError(msg)
    return int(number)


def _parse_lambda(value: Any, name: str) -> float:
    msg = f"sordino: {name} must be a number of at least 0, got {value!r}"
    if value is None:
        return 0.0
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(msg)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(msg) from None
    if not np.isfinite(number) or number < 0:
        raise ValueError(msg)
    return number


def parse_anat_cg_iter(value: Any) -> int:
    """A positive iteration count; None is the default (ValueError otherwise)."""
    return _parse_iter(value, "anat_cg_iter")


def parse_anat_lambda(value: Any) -> float:
    """A finite Tikhonov weight of at least 0; None is 0 (ValueError otherwise)."""
    return _parse_lambda(value, "anat_lambda")


FRAME_STATICS = ("adjoint", "cg")


def parse_frame_static(value: Any) -> str:
    """``adjoint`` (also None) or ``cg`` (ValueError otherwise)."""
    text = "adjoint" if value is None else (value.strip().lower() if isinstance(value, str) else None)
    if text in FRAME_STATICS:
        return text
    raise ValueError(f"sordino: frame_static must be 'adjoint' or 'cg', got {value!r}")


def parse_frame_static_iter(value: Any) -> int:
    """A positive iteration count; None is the default (ValueError otherwise)."""
    return _parse_iter(value, "frame_static_iter")


def parse_frame_static_lambda(value: Any) -> float:
    """A finite Tikhonov weight of at least 0; None is 0 (ValueError otherwise)."""
    return _parse_lambda(value, "frame_static_lambda")


def frame_static_key(options) -> Dict[str, Any]:
    """Cache-key entry of ``frame_static='cg'`` (used only when it is not ``adjoint``)."""
    return {"static": options.frame_static, "n_iter": int(options.frame_static_iter),
            "lambda": float(options.frame_static_lambda)}


def _check_static(options) -> None:
    """Option guards of ``frame_static`` (ValueError): cg needs ``rep_pattern='model'``."""
    if getattr(options, "frame_static", "adjoint") == "adjoint":
        if (getattr(options, "frame_static_iter", ANAT_ITER_DEFAULT) != ANAT_ITER_DEFAULT
                or getattr(options, "frame_static_lambda", 0.0) != 0.0):
            raise ValueError("sordino: frame_static_iter and frame_static_lambda need frame_static='cg'; "
                             "with the adjoint they would be ignored.")
        return
    if getattr(options, "rep_pattern", "none") != "model":
        raise ValueError("sordino: frame_static='cg' needs rep_pattern='model' (the static image is the "
                         "model estimate x_hat); without it the option would be ignored.")
    if getattr(options, "anat_solver", "adjoint") != "adjoint":
        raise ValueError("sordino: frame_static='cg' cannot be used with anat_solver='cg'.")


def anat_key(options) -> Dict[str, Any]:
    """Cache-key entry of ``anat_solver='cg'`` (used only when the solver is not ``adjoint``)."""
    return {"solver": options.anat_solver, "n_iter": int(options.anat_cg_iter),
            "lambda": float(options.anat_lambda)}


def _check_anat(options) -> None:
    """Option guards of ``anat_solver`` (ValueError): cg needs ``frame_spokes='all'`` and nothing else."""
    solver = getattr(options, "anat_solver", "adjoint")
    if solver == "adjoint":
        if (getattr(options, "anat_cg_iter", ANAT_ITER_DEFAULT) != ANAT_ITER_DEFAULT
                or getattr(options, "anat_lambda", 0.0) != 0.0):
            raise ValueError("sordino: anat_cg_iter and anat_lambda need anat_solver='cg'; "
                             "with the adjoint they would be ignored.")
        return
    if getattr(options, "frame_spokes", None) != "all":
        raise ValueError("sordino: anat_solver='cg' reconstructs the whole scan as one image; "
                         "use frame_spokes='all'.")
    clash = [name for name, on in (
        ("peri_event", getattr(options, "peri_event", None) not in (None, False, "")),
        ("frame_step", getattr(options, "frame_step", None) is not None),
        ("frame_accumulate", bool(getattr(options, "frame_accumulate", False))),
        ("rep_pattern", getattr(options, "rep_pattern", "none") != "none"),
        ("subtract_rep_mean", bool(getattr(options, "subtract_rep_mean", False)))) if on]
    if clash:
        raise ValueError(f"sordino: anat_solver='cg' cannot be used with {', '.join(clash)}.")


def rep_pattern_key(options, recon_info: Dict[str, Any]) -> Dict[str, Any]:
    """Cache-key entry of ``rep_pattern`` (used only when the mode is not ``none``)."""
    from .kcentre import k0_iterations

    return {"mode": getattr(options, "rep_pattern", "none"), "xhat": "cg", "n_iter": k0_iterations(recon_info)}


def _explicit(options) -> bool:
    fs = getattr(options, "frame_spokes", None)
    return (fs not in (None, "repetition") or getattr(options, "frame_step", None) is not None
            or bool(getattr(options, "frame_accumulate", False))
            or getattr(options, "peri_event", None) not in (None, False, ""))


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
    #: Peri-event plans only (perievent.py): ``groups[i]`` is the output bin of
    #: ``ranges[i]``, so one bin is the union of several ranges and ``scales`` is
    #: per bin, not per range. None for the ordinary one-range-per-frame plans.
    groups: Optional[Tuple[int, ...]] = None
    peri: Optional[Dict[str, Any]] = None

    @property
    def n_frames(self) -> int:
        # scales is per output image in both plans (per range, or per peri-event bin)
        return len(self.scales)

    @property
    def frame_interval_s(self) -> Optional[float]:
        """Time between frames: ``frame_step`` x the spoke TR (D-0190 2); None without a TR.

        Peri-event bins are consecutive windows on the clock of the event, not of
        the scan, so what separates two output images is the bin width; the step
        (the event period) is the distance between the ranges inside one bin.
        """
        if self.tr_ms is None:
            return None
        spokes = self.window if self.groups is not None else self.step
        return spokes * self.tr_ms * 1e-3

    @property
    def max_open(self) -> int:
        """Most image-sized arrays besides the written frame the engine holds at once (memory).

        Ordinary plans: frame-start copies of the running sum. Peri-event plans: one accumulator
        per bin, all allocated before the first spoke (``recon_frames``), so ``n_frames``.
        """
        import bisect

        if self.groups is not None:
            return int(self.n_frames)
        last: Dict[int, int] = {}
        for lo, hi in self.ranges:
            last[lo] = hi
        starts = sorted(last)
        ends = sorted(last.values())
        best = 0
        for i, b in enumerate(starts):
            # copies held just after the copy at start b: starts <= b whose last frame ends after b
            # (a copy is freed at its last frame end, before a new copy is made at the same point)
            best = max(best, (i + 1) - bisect.bisect_right(ends, b))
        return best

    def key(self) -> Dict[str, Any]:
        """What decides the frame images (recon cache key)."""
        key = {"unit": self.unit, "window": self.window, "step": self.step, "accumulate": self.accumulate}
        if self.groups is not None:
            # the ranges themselves decide a peri-event image, so two different event
            # layouts must not share a cache entry
            key["peri"] = {"ranges": [[lo, hi] for lo, hi in self.ranges], "groups": list(self.groups)}
        return key

    def describe(self) -> Dict[str, Any]:
        """Result metadata (``scan._sordino_recon_meta["frames"]``)."""
        tr = self.tr_ms
        meta = {
            "trajectory_mode": self.trajectory_mode, "unit_spokes": self.unit,
            "frame_spokes": self.window, "frame_step": self.step, "accumulate": self.accumulate,
            "n_frames": self.n_frames, "repetitions": self.n_rep, "spokes_per_repetition": self.npro,
            "spoke_ranges": [[lo, hi] for lo, hi in self.ranges],
            "frames_crossing_repetitions": self.crossing, "frames_inside_subsets": self.misaligned,
            "spoke_tr_ms": tr, "frame_interval_s": self.frame_interval_s,
            "frame_centre_s": None if tr is None else [(lo + hi) / 2 * tr * 1e-3 for lo, hi in self.ranges],
            "k0_scope": None,
        }
        if self.groups is not None:
            # a peri-event frame is a bin relative to the event, not a point on the scan clock
            meta["groups"] = list(self.groups)
            meta["peri_event"] = dict(self.peri or {})
            meta["frame_centre_s"] = None if tr is None else [
                (self.peri or {}).get("start_spokes", 0) * tr * 1e-3 + (b + 0.5) * self.window * tr * 1e-3
                for b in range(self.n_frames)]
        return meta


def make_plan(recon_info: Dict[str, Any], options) -> Optional[FramePlan]:
    """The frame plan, or None for one repetition per frame (``recon_dataobj``)."""
    from .recon import get_num_frames

    _check_anat(options)
    _check_static(options)
    peri_on = getattr(options, "peri_event", None) not in (None, False, "")
    if getattr(options, "peri_censor", None) and not peri_on:
        raise ValueError("sordino: peri_censor needs peri_event (a peri-event frame plan, for example "
                         "peri_event=period=20,bin=1); without it the censor file would be ignored.")
    pattern = getattr(options, "rep_pattern", "none")
    if pattern != "none":
        if getattr(options, "subtract_rep_mean", False):
            raise ValueError("sordino: rep_pattern and subtract_rep_mean cannot be used together; "
                             "rep_pattern='measured' keeps the image that subtract_rep_mean removes.")
        if pattern == "measured" and getattr(options, "peri_censor", None):
            raise ValueError("sordino: rep_pattern='measured' cannot be used with peri_censor yet; use "
                             "rep_pattern='model'.")
    if not is_frame_mode(options, recon_info):
        if pattern != "none":
            raise ValueError("sordino: rep_pattern needs a frame plan (peri_event or a frame option such "
                             "as frame_spokes); without one the option would be ignored.")
        if getattr(options, "subtract_rep_mean", False):
            raise ValueError("sordino: subtract_rep_mean needs a frame plan (peri_event or a frame "
                             "option such as frame_spokes); without one the option would be ignored.")
        return None
    mode = golden.trajectory_mode(recon_info)
    if mode == "Default" and options.frame_spokes == "subset":
        raise ValueError(
            "sordino: frame_spokes='subset' needs a golden trajectory; this scan has TrajectoryMode Default, "
            "which has no method subset. Give a spoke count (frame_spokes=<n>).")
    if getattr(options, "correct_spoketiming", False):
        raise ValueError(
            "sordino: correct_spoketiming cannot be used with golden frames: it moves every spoke of a "
            "repetition to one time point, so frames shorter than a repetition would all show that time. "
            "Use frame_spokes='repetition' with correct_spoketiming.")
    unit = int(golden.golden_unit(recon_info) or 1)          # Default list: no method subset
    npro = int(recon_info["NPro"])
    n_rep = int(get_num_frames(recon_info, options))
    total = n_rep * npro
    if pattern == "measured" and n_rep < 2:
        raise ValueError(f"sordino: rep_pattern='measured' needs at least 2 repetitions, this run reads {n_rep}")
    tr = recon_info.get("RepetitionTime_ms")
    tr_ms = float(tr) if tr not in (None, 0, 0.0) else None
    if getattr(options, "peri_event", None) not in (None, False, ""):
        # one image per time bin relative to a repeating event; a bin is the union of one
        # range per event, so the plan carries groups and the engine sums per bin
        if mode == "Default":
            logger.warning(
                "sordino: TrajectoryMode Default lists its spokes in sphere order, so a peri-event bin "
                "covers only part of the sphere; peri_event is meant for golden trajectories.")
        ranges, groups, scales, info = perievent.build(recon_info, options, n_rep)
        misaligned = frameplan.misaligned_count(ranges, unit) if unit > 1 else 0
        if misaligned:
            logger.warning(
                "sordino: %s of %s peri-event range(s) start or end inside a method subset (%s spokes); "
                "such a range covers the sphere less evenly. Bin and period that are multiples of %s "
                "spokes keep whole subsets.", misaligned, len(ranges), unit, unit)
        return FramePlan(mode, unit, npro, n_rep, info["bin_spokes"], info["period_spokes"], False,
                         ranges, scales, frameplan.crossing_count(ranges, npro), misaligned, tr_ms,
                         groups=groups, peri=info)
    fs = options.frame_spokes
    if fs == "repetition" or (fs is None and mode == "Default"):
        window = npro                                     # Default list: one repetition unless a count is given
    elif fs == "all":
        window = total                                    # every spoke of every repetition in one image
    elif fs in (None, "subset"):
        window = unit
    else:
        window = int(fs)
    step = int(options.frame_step) if options.frame_step is not None else window
    accumulate = bool(options.frame_accumulate)
    if window > total:
        raise ValueError(f"sordino: frame_spokes {window} is more than the {total} spokes read "
                         f"({n_rep} repetition(s) of {npro}).")
    ranges = tuple(tuple(r) for r in frameplan.frame_ranges(total, window, step, accumulate))
    if accumulate and ranges[-1][1] < total:
        # the last accumulated frame holds every spoke read (D-0194 1: "until all spokes")
        logger.info("sordino: the last accumulated frame holds all %s spokes read (%s more than the step "
                    "before it).", total, total - ranges[-1][1])
        ranges = ranges + ((0, total),)
    if mode == "Default" and window < npro:
        logger.warning(
            "sordino: TrajectoryMode Default lists its spokes in sphere order, so a frame of %s spokes (fewer "
            "than the %s of a repetition) covers only part of the sphere; such frames show the k-space "
            "coverage growing, not whole images.", window, npro)
    ends = [hi for _, hi in ranges]
    assert all(b > a for a, b in zip(ends, ends[1:])), "frame ends must increase"
    scales = tuple(frameplan.frame_scales(ranges, npro))
    crossing = frameplan.crossing_count(ranges, npro)
    misaligned = frameplan.misaligned_count(ranges, unit) if unit > 1 else 0
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


def _anat_image(options, kernel, xs, full, preds, virtual_traj, w_virtual, adj, plan, npro, chunk_spokes,
                nf, anat_out, n_it=None, lam=None, name="anat_solver"):
    """The ``anat_solver='cg'`` image, one volume per channel (complex128), at the gain of the adjoint image.

    Solves (N + lambda' I) x = A^H W d on the normal operator N of all spokes read (the measured samples
    and, with ``estimate_k0``, the predicted centre samples of every repetition) by CG from zero, with
    lambda' = lambda, the absolute Tikhonov weight (its useful scale depends on the data gain).
    ``xs``: one slot per channel, overwritten by the solution. Fills ``anat_out`` with solver, iterations, lambda and the
    final relative residual (worst channel).
    """
    from . import serial

    lam = float(options.anat_lambda if lam is None else lam)
    n_it = int(options.anat_cg_iter if n_it is None else n_it)
    n_rx = len(xs)
    hist = []
    if virtual_traj is not None:
        flat_w = np.asarray(w_virtual, dtype=np.float64).reshape(-1)
        vtraj = np.asarray(virtual_traj)

        def normal(x):
            y = kernel.normal(x)
            y += plan.n_rep * serial.normal_on(x, vtraj, flat_w)
            if lam:
                y += lam * x
            return y
        for ch in range(n_rx):
            b = full[ch].copy()          # raw A^H W d of the measured samples, plus the predicted centre
            for a, e in serial.spoke_ranges(npro, chunk_spokes):
                adj.setpts(vtraj[a:e])
                adj.add(b, plan.n_rep * preds[ch][a:e].reshape(-1) * w_virtual[a:e].reshape(-1))
            x, hist_ch = serial.conjugate_gradient(normal, b, n_it)
            xs[ch] = x
            hist.append(hist_ch)
            del b
    else:
        def normal(x):
            return kernel.normal(x) + lam * x if lam else kernel.normal(x)
        for ch in range(n_rx):
            x, hist_ch = serial.conjugate_gradient(normal, full[ch], n_it)
            xs[ch] = x
            hist.append(hist_ch)
    residual = max(h[-1] / h[0] if h[0] > 0 else 0.0 for h in hist)
    iterations = max(len(h) - 1 for h in hist)
    if anat_out is not None:
        anat_out.update({"solver": "cg", "n_iter": n_it, "iterations_run": int(iterations),
                         "lambda": lam, "relative_residual": float(residual)})
    logger.info("%s='cg': %s iteration(s) of %s, lambda %g, final relative residual %.3g.",
                name, iterations, n_it, lam, residual)
    return np.stack(xs)


def recon_frames(fid_fobj, traj, recon_info: Dict[str, Any], img_fobj, options, plan: FramePlan, *,
                 phase_factor=None, virtual_traj=None, k0_out=None, chunk_spokes=None, k0_method=None,
                 weights=None, spoke_mask=None, out_scale=1.0, subtract_rep_mean=False, anat_out=None):
    """Write the frames of ``plan`` to ``img_fobj`` (the layout of ``recon_dataobj``); return their dtype.

    ``traj`` (``TrajectoryRows`` or an array), ``phase_factor`` (``PhaseRows`` or an array) and
    ``virtual_traj`` (``kcentre.leading_points``) are those of one repetition; the stream repeats
    them. ``k0_out`` gets one list (one K0 per channel) for the whole stream. ``weights``
    (``dcf.SampleWeights`` of one repetition; golden trajectories, WI-0113 run 5) replace
    |k|^2 / max for every repetition and frame.

    ``spoke_mask`` (entry for WI-0118; not a user option): boolean array over the stream
    (``n_rep * NPro``), True = use the spoke. Excluded spokes add nothing (their data are
    zeroed before the adjoint); frame scales are unchanged. Not with ``virtual_traj``
    (the K0 solve would still hold the excluded spokes): WI-0118 decides that case.

    ``anat_out`` (a dict): with ``anat_solver='cg'`` (one frame of every spoke) or ``frame_static='cg'`` it gets the solver, the
    iterations, lambda and the final relative residual (``_anat_image``).

    ``out_scale`` and a ``traj`` with ``n_keep`` (``recon_matrix``, WI-0132): as in ``recon_dataobj``.

    ``subtract_rep_mean``: before reconstructing, average each spoke position over the
    repetitions read and subtract that average from every occurrence of the position.
    Each bin of a peri-event plan is then built from residuals rather than raw spokes,
    so whatever is fixed for a spoke position -- the undersampling pattern its own
    trajectory produces, and the time-averaged anatomy -- cancels, and only what varies
    between the repetitions of that position survives.

    This is only task-neutral when the repetitions sample the event cycle evenly. For
    these scans a repetition is 181 s against a 20 s cycle, so successive repetitions
    advance the task phase by exactly 1 s and the 20 repetitions of any spoke cover all
    20 phases once each: the average is the task-free background. With 1800 subsets the
    repetition would be 180 s, every repetition of a spoke would land on the same task
    phase, and the average would remove the response along with the background. The
    caller is responsible for checking that; the engine only reports what it subtracted.

    Costs one extra read of the FID and one array of
    (NPro, receivers, points) complex64.
    """
    from . import memguard, serial
    from .recon import (_row_source, correct_offreso, grid_shift, grid_shift_factor, parse_fid_info,
                        parse_volume_shape)

    mode = getattr(options, "rep_pattern", "none")
    if mode != "none" and subtract_rep_mean:
        raise ValueError("sordino: rep_pattern and subtract_rep_mean cannot be used together")
    if mode != "none" and spoke_mask is not None:
        raise ValueError("sordino: rep_pattern with a spoke mask is not supported yet")
    anat = getattr(options, "anat_solver", "adjoint") == "cg"
    if anat and spoke_mask is not None:
        raise ValueError("sordino: anat_solver='cg' with a spoke mask is not supported")
    static_cg = getattr(options, "frame_static", "adjoint") == "cg"
    if static_cg and mode != "model":
        raise ValueError("sordino: frame_static='cg' needs rep_pattern='model'")
    need_x = virtual_traj is not None or mode == "model" or anat   # a CG solve on the all-spoke normal operator
    gain = 1.0          # frame_static='cg': residual term to the units of x_hat (1 / c, technical note 10.4)
    img_fobj.seek(0)
    fid_shape, fid_dtype = parse_fid_info(recon_info)
    vol = [int(v) for v in parse_volume_shape(recon_info, options)]
    n_points, n_rx, npro = (int(v) for v in fid_shape[1:])
    if npro != plan.npro:
        raise ValueError(f"frame plan is for {plan.npro} spokes per repetition, the FID has {npro}")
    if spoke_mask is not None:
        spoke_mask = np.asarray(spoke_mask, dtype=bool).reshape(-1)
        if spoke_mask.size != plan.n_rep * npro:
            raise ValueError(f"spoke_mask has {spoke_mask.size} values, the stream has {plan.n_rep * npro} spokes")
        if virtual_traj is not None:
            raise ValueError("spoke_mask with estimate_k0 is not supported yet (WI-0118 decides how the K0 solve "
                             "leaves spokes out)")
        if spoke_mask.all():
            spoke_mask = None
    ign = getattr(options, "ignore_samples", None) or 1
    offset = getattr(options, "offset", None) or 0
    item = np.dtype(fid_dtype).itemsize
    spoke_bytes = 2 * n_points * n_rx * item
    stream = _Stream(fid_fobj, offset * spoke_bytes * npro, spoke_bytes, fid_dtype, n_points, n_rx)
    rows = _row_source(traj)
    traj_spokes = int(traj.n_pro) if hasattr(traj, "n_pro") else int(np.shape(traj)[0])
    if traj_spokes != npro:
        raise ValueError(f"trajectory has {traj_spokes} spokes, the FID has {npro}")
    n_stop = getattr(traj, "n_keep", None)          # recon_matrix (WI-0132)
    n_used = n_points if n_stop is None else int(n_stop)
    if chunk_spokes is None:
        chunk_spokes = memguard.recon_plan(npro, n_points, n_rx, vol,
                                           estimate_k0=need_x)["chunk_spokes"]
    offreso = getattr(options, "offreso_freqs", None)
    bw, osf = recon_info.get("EffBandwidth_Hz"), recon_info.get("OverSampling")

    def _offreso(ch):
        if isinstance(offreso, tuple) and len(offreso) > ch and bw is not None and osf is not None:
            return offreso[ch]
        return None

    # density maximum over one repetition (and the virtual samples), as recon_dataobj; with
    # estimate_k0 the normal operator of the n_rep repetitions (n_rep x one repetition's kernel)
    kernel = None
    if need_x:
        choice = memguard.k0_method(npro, n_points, n_rx, vol)
        method = choice["method"] if k0_method is None else str(k0_method)
        if method == "toeplitz":
            kernel = serial.ToeplitzKernel(vol)
        elif method == "samples":
            kernel = serial.SampleNormal(vol, npro * (n_used - ign))
        else:
            raise ValueError(f"k0_method must be None, 'toeplitz' or 'samples', not {k0_method!r}")
    dmax = 0.0
    w_virtual = None
    if weights is None:
        for lo, hi in serial.spoke_ranges(npro, chunk_spokes):
            tr = rows(lo, hi)[:, ign:]
            d = serial.density(tr)
            dmax = max(dmax, float(d.max()))
            if kernel is not None:
                kernel.add(tr, d)
            del tr, d
        if virtual_traj is not None:
            dmax = max(dmax, float(serial.density(virtual_traj).max()))
        if kernel is not None:
            kernel.finish(float(plan.n_rep) / dmax)
        if virtual_traj is not None:
            n_virtual = int(virtual_traj.shape[1])
            w_virtual = (serial.density(virtual_traj) / dmax).reshape(npro, n_virtual)
    else:
        if kernel is not None:
            for lo, hi in serial.spoke_ranges(npro, chunk_spokes):
                kernel.add(rows(lo, hi)[:, ign:], weights.chunk(lo, hi))
        if kernel is not None:
            kernel.finish(float(plan.n_rep))
        if virtual_traj is not None:
            w_virtual = weights.virtual().reshape(npro, int(virtual_traj.shape[1]))
    rep_mean = None
    if subtract_rep_mean or mode == "measured":
        name = "subtract_rep_mean" if subtract_rep_mean else "rep_pattern='measured'"
        if plan.n_rep < 2:
            raise ValueError(f"sordino: {name} needs at least 2 repetitions, "
                             f"this run reads {plan.n_rep}")
        # censored spokes (peri_censor) are left out of the average so they cannot leak into the
        # residual of the spokes that are used; a position censored in every repetition gets mean 0
        censor = perievent.load_censor(getattr(options, "peri_censor", None), plan.n_rep * npro)
        rep_mean = np.zeros((npro, n_rx, n_points), dtype=np.complex64)
        counts = np.full(npro, plan.n_rep, dtype=np.float32) if censor is None else np.zeros(npro, np.float32)
        for r in range(plan.n_rep):
            for a, b in serial.spoke_ranges(npro, chunk_spokes):
                k = stream.read(r * npro + a, b - a).astype(np.complex64)
                if censor is not None:
                    use = censor[r * npro + a:r * npro + b]
                    k[~use] = 0
                    counts[a:b] += use
                rep_mean[a:b] += k
        rep_mean /= np.maximum(counts, 1).astype(np.float32)[:, None, None]
        stream.rewind()
        logger.info("%s: averaged %s repetitions of each of the %s spoke positions "
                    "(%.2f GiB, %s censored spoke(s) left out); every spoke is reconstructed as its "
                    "deviation from that average.", name, plan.n_rep, npro, rep_mean.nbytes / 2**30,
                    0 if censor is None else int((~censor).sum()))

    nf = serial.norm_factor(vol)
    shift = grid_shift(recon_info)       # derived from the affine by the hook (WI-0119)
    adj = serial.Adjoint(vol)

    def add_range(acc, g_lo, g_hi, preds=None, raw=False):
        """acc += adjoint of stream spokes g_lo .. g_hi (and their predicted centre samples).

        ``raw`` leaves the repetition mean out: the all-spoke passes (the K0 solve, x_hat, x_full)
        must see the data, not their residuals, which sum to zero over the repetitions."""
        g = g_lo
        while g < g_hi:
            r = g // npro
            lo = g - r * npro
            hi = min(g_hi - r * npro, npro)
            for a, b in serial.spoke_ranges(hi - lo, chunk_spokes):
                a, b = a + lo, b + lo
                k = stream.read(r * npro + a, b - a)
                if rep_mean is not None and not raw:
                    k = k - rep_mean[a:b]          # residual from this position's own average
                if phase_factor is not None:
                    k = k * phase_factor[a:b][:, None, :]
                k = k[..., ign:n_stop]
                if spoke_mask is not None:
                    k[~spoke_mask[r * npro + a:r * npro + b]] = 0.0
                for ch in range(n_rx):
                    freq = _offreso(ch)
                    if freq is not None:
                        k[:, ch, :] = correct_offreso(k[:, ch, :], freq, eff_bandwidth=bw, over_sampling=osf)
                tr = rows(a, b)[:, ign:]
                w = serial.density(tr) / dmax if weights is None else weights.chunk(a, b)
                g = grid_shift_factor(tr, shift)
                if g is not None:          # measured samples only; the predicted centre samples
                    w = w * g.reshape(-1)  # come from the solve on the shifted grid already
                del g
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
    xs = None
    x_full = None
    if mode != "none" or virtual_traj is not None or anat:
        from .kcentre import k0_iterations
        n_iter = k0_iterations(recon_info)     # N_ITER unless the centre gap is wide (WI-0137)

        # all-spoke image of the RAW data (add_range(raw=True)): with a repetition mean the
        # residuals would sum to zero here and the solve below would see no object
        full = np.zeros(shape, dtype=np.complex128)
        add_range(full, 0, plan.n_rep * npro, raw=True)
        stream.rewind()
        if need_x:
            preds, k0, xs = [], [], []
            for ch in range(n_rx):
                if (anat or static_cg) and virtual_traj is None:
                    xs.append(None)               # solved once, with lambda, in _anat_image
                    continue
                x, _ = serial.conjugate_gradient(kernel.normal, full[ch], n_iter)
                if virtual_traj is not None:
                    k0.append(complex(x.sum()))
                    preds.append(serial.forward(x, virtual_traj).reshape(npro, -1))
                xs.append(x)
            if virtual_traj is None:
                preds = None
            if anat:
                img = _anat_image(options, kernel, xs, full, preds, virtual_traj, w_virtual, adj, plan, npro,
                                  chunk_spokes, nf, anat_out)
                del kernel, full
                if out_scale != 1.0:
                    img *= out_scale
                out = img if n_rx > 1 else img[0]
                img_fobj.write(np.ascontiguousarray(out.T).tobytes())
                if k0_out is not None and virtual_traj is not None:
                    k0_out.append(k0)
                return np.dtype(np.complex128)
            if static_cg:
                # x_hat of the frames: CG with frame_static_iter and frame_static_lambda
                _anat_image(options, kernel, xs, full, preds, virtual_traj, w_virtual, adj, plan, npro,
                            chunk_spokes, nf, anat_out, n_it=int(options.frame_static_iter),
                            lam=float(options.frame_static_lambda), name="frame_static")
            if mode == "model":
                # the all-spoke image of the model: the same operator that the frames use
                full = np.stack([kernel.normal(x) for x in xs])
                xs = np.stack(xs)
                if static_cg:
                    # gain c of T_all / (n_rep nf) on x_hat (Rayleigh quotient over all channels);
                    # the residual term of a frame is divided by it (technical note, section 10.4)
                    den = float(np.vdot(xs, xs).real)
                    c = float(np.vdot(xs, full).real) / (plan.n_rep * nf * den) if den > 0 else 1.0
                    gain = 1.0 / c if c > 0 else 1.0
            else:
                xs = None
            del kernel
            if k0_out is not None and virtual_traj is not None:
                k0_out.append(k0)
            logger.info("%s: one estimate from all %s spokes read, used by every frame.",
                        "estimate_k0 with frames" if virtual_traj is not None else "rep_pattern='model'",
                        plan.n_rep * npro)
        if mode != "none":
            # x_full: the all-spoke image at the frame gain (the time mean of the frames), with the
            # predicted centre samples of every repetition; one image per channel
            if static_cg:
                x_full = xs            # frame_static='cg': the static image is x_hat itself
                del full
            else:
                x_full = full
                if preds is not None:
                    for a, b in serial.spoke_ranges(npro, chunk_spokes):
                        adj.setpts(virtual_traj[a:b])
                        wv = w_virtual[a:b].reshape(-1)
                        for ch in range(n_rx):
                            adj.add(x_full[ch], plan.n_rep * preds[ch][a:b].reshape(-1) * wv)
                x_full /= plan.n_rep * nf
        else:
            del full
    # residual frames (subtract_rep_mean, measured) carry no predicted centre term: the centre samples
    # are the same in every repetition, so their residual is zero; x_full carries them for measured
    frame_preds = None if rep_mean is not None else preds

    ranges_of = None
    if plan.groups is not None:
        ranges_of = [[] for _ in range(plan.n_frames)]
        for rng_, bin_index in zip(plan.ranges, plan.groups):
            ranges_of[bin_index].append(rng_)
    else:
        ranges_of = [[rng_] for rng_ in plan.ranges]

    def pattern_term(index):
        """What is added to image ``index`` before the output gain: x_full - (s / nf) T x_hat (model),
        x_full (measured). An empty (zero-scale) frame stays zero."""
        scale = plan.scales[index]
        if mode == "none" or scale == 0.0:
            return None
        if mode == "measured":
            return x_full
        t = np.zeros(shape, dtype=np.complex128)
        for lo_, hi_ in ranges_of[index]:
            g = lo_
            while g < hi_:
                r = g // npro
                lo = g - r * npro
                hi = min(hi_ - r * npro, npro)
                for a, b in serial.spoke_ranges(hi - lo, chunk_spokes):
                    a, b = a + lo, b + lo
                    tr = rows(a, b)[:, ign:]
                    w = serial.density(tr) / dmax if weights is None else weights.chunk(a, b)
                    t += serial.normal_on(xs, tr, w)
                    if virtual_traj is not None:
                        t += serial.normal_on(xs, virtual_traj[a:b], w_virtual[a:b])
                g = r * npro + hi
        t *= -scale * gain / nf
        t += x_full
        return t

    if plan.groups is not None:
        # Peri-event: a bin is the union of one range per event, which no difference of two
        # prefix sums can express, so each bin gets its own accumulator. The ranges are sorted
        # by start (perievent.build), so the stream still runs forward and grids each spoke once.
        n_out = plan.n_frames
        logger.info("peri_event: %s bin accumulator(s) of %s complex128 (%.2f GiB) over %s range(s).",
                    n_out, "x".join(str(v) for v in shape), n_out * int(np.prod(shape)) * 16 / 2**30,
                    len(plan.ranges))
        accs = [np.zeros(shape, dtype=np.complex128) for _ in range(n_out)]
        for (lo, hi), bin_index in zip(plan.ranges, plan.groups):
            add_range(accs[bin_index], lo, hi, frame_preds)
        for bin_index in range(n_out):
            img = accs[bin_index] * (plan.scales[bin_index] * gain / nf)
            term = pattern_term(bin_index)
            if term is not None:
                img += term
                del term
            if out_scale != 1.0:
                img *= out_scale
            out = img if n_rx > 1 else img[0]
            img_fobj.write(np.ascontiguousarray(out.T).tobytes())
            accs[bin_index] = None
            del img, out
        logger.debug("peri_event: %s bin(s) written from %s range(s) of %s spokes.",
                     n_out, len(plan.ranges), plan.window)
        return np.dtype(np.complex128)

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
            img = (run - snaps[lo]) * (plan.scales[nxt] * gain / nf)
            term = pattern_term(nxt)
            if term is not None:
                img += term
                del term
            if out_scale != 1.0:
                img *= out_scale
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
            add_range(run, b, points[i + 1], frame_preds)
    assert nxt == plan.n_frames
    logger.debug("frames: %s written (%s x %s spokes, step %s, accumulate %s)", plan.n_frames, plan.n_frames,
                 plan.window, plan.step, plan.accumulate)
    return np.dtype(np.complex128)


__all__ = ["FRAME_KEYS", "REP_PATTERNS", "FramePlan", "parse_frame_spokes", "parse_frame_step",
           "parse_rep_pattern", "rep_pattern_key", "parse_frame_static", "frame_static_key", "is_frame_mode", "make_plan", "recon_frames"]
