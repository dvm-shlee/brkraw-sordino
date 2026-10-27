"""Peri-event frames: one image per time bin relative to a repeating event.

Event-related reconstruction. A stimulus repeats every ``period`` seconds; this
builds one output image per time bin relative to event onset, each combining the
spokes of that bin from every event.

Unlike the frames of ``frames.make_plan``, a peri-event bin is a *union* of many
disjoint spoke ranges (one per event), so it cannot be written as the difference
of two prefix sums. The plan therefore carries a ``groups`` array mapping each
range to its output bin, and ``frames.recon_frames`` accumulates per bin in one
forward pass over the FID (each spoke read and gridded exactly once).

Option syntax (``--hook-arg sordino:peri_event=...``), comma-separated pairs::

    peri_event=period=20,bin=1,onset=1

    period   seconds between events (required)
    bin      seconds per output bin (required)
    onset    seconds from the start of the data to the first event (default 0)
    start    seconds from event onset to the first bin (default 0, may be negative)
    bins     number of bins (default: ceil(period / bin), covering the whole period)
    events   maximum number of events to use (default: every event that fits)

Bins tile ``[start, start + bins * bin)`` relative to each onset. With the
default ``bins`` the tiling covers one whole period, so every spoke of the scan
lands in exactly one bin and the late bins act as the pre-stimulus baseline of
the next event. When ``bin`` does not divide ``period`` the last default bin
runs past the period end (a warning reports by how much). A bin that gets no
spokes is reported with a warning and written as an all-zero frame; only a plan
with no spokes at all is refused.

Sampling note: the golden trajectory repeats every ``NPro`` spokes, so the
distinct trajectory phases an event can land on is ``NPro / gcd(period_spokes,
NPro)``. When the period divides the repetition evenly that number is small and
every event re-samples the same few angles; ``make_plan`` logs the unique-spoke
count so the result can be judged against the Nyquist spoke count of the
reconstruction matrix.
"""

from __future__ import annotations

import logging
from math import gcd
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

#: Option names owned by this module (excluded from the raw option cache key).
PERI_KEYS = ("peri_event", "peri_censor")

_FIELDS = {"period", "bin", "onset", "start", "bins", "events"}

#: ``peri_censor``: path to a 1D file of 0/1 over the spokes of the stream,
#: 1 = usable. Any peri-event range whose spokes are not all usable is left out
#: of its bin; the per-bin scale already divides by the events that reached it,
#: so dropping ranges changes the averaging weight and nothing else.
CENSOR_KEY = "peri_censor"


def parse_peri_event(value: Any) -> Optional[Dict[str, float]]:
    """``peri_event`` as a dict of floats, or None when unset.

    Accepts the comma-separated ``key=value`` string of ``--hook-arg`` and the
    mapping a YAML hook-args file gives.
    """
    if value is None or value is False or value == "":
        return None
    if isinstance(value, dict):
        items = {str(k).strip(): v for k, v in value.items()}
    else:
        # ``--hook-arg sordino:peri_event=period=20,bin=1`` reaches the hook as a sequence of
        # "key=value" strings, or as the repr of one, because the CLI splits the value on commas.
        if isinstance(value, (list, tuple)):
            parts = [str(v) for v in value]
        else:
            text = str(value).strip()
            if not text:
                return None
            parts = text.split(",")
        items = {}
        for part in parts:
            part = part.strip().strip("()[]").strip().strip("'\"").strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(
                    f"sordino: peri_event needs key=value pairs separated by commas, got {part!r}. "
                    f"Example: peri_event=period=20,bin=1,onset=1")
            key, _, val = part.partition("=")
            items[key.strip().strip("'\"")] = val.strip().strip("'\"")
    unknown = sorted(set(items) - _FIELDS)
    if unknown:
        raise ValueError(f"sordino: peri_event has unknown field(s) {', '.join(unknown)}; "
                         f"known fields are {', '.join(sorted(_FIELDS))}.")
    out: Dict[str, float] = {}
    for key, val in items.items():
        try:
            out[key] = float(val)
        except (TypeError, ValueError):
            raise ValueError(f"sordino: peri_event {key}={val!r} is not a number.") from None
    for required in ("period", "bin"):
        if required not in out:
            raise ValueError(f"sordino: peri_event needs {required}=<seconds>. "
                             f"Example: peri_event=period=20,bin=1,onset=1")
    if out["period"] <= 0 or out["bin"] <= 0:
        raise ValueError("sordino: peri_event period and bin must be positive.")
    for key in ("bins", "events"):
        if key in out:
            if out[key] != int(out[key]) or out[key] < 1:
                raise ValueError(f"sordino: peri_event {key} must be a positive whole number.")
    return out


def _spokes(seconds: float, tr_ms: float, name: str, *, signed: bool = False) -> int:
    """Seconds as a whole number of spokes at the spoke TR."""
    exact = seconds * 1000.0 / tr_ms
    n = int(round(exact))
    if abs(exact - n) > 1e-6:
        logger.warning("sordino: peri_event %s=%ss is %.3f spokes at a %s ms spoke TR; using %s.",
                       name, seconds, exact, tr_ms, n)
    if not signed and n <= 0:
        raise ValueError(f"sordino: peri_event {name}={seconds}s is {n} spokes; it must be at least 1.")
    return n


def _load_censor(value, total: int):
    """A 0/1 mask over the spokes of the stream, or None when unset.

    The file is a 1D column of 0/1 per *frame* of the reconstruction it
    was measured on; it is stretched to spokes here. A range is used only when
    every spoke in it is marked usable, so a censored frame removes the whole
    peri-event bin it falls in rather than half of one.
    """
    if value in (None, False, ""):
        return None
    import numpy as np

    col = np.loadtxt(str(value))
    col = np.asarray(col).reshape(-1)
    if col.size == 0:
        raise ValueError(f"sordino: peri_censor file {value} is empty")
    if not np.isin(col, (0, 1)).all():
        raise ValueError(f"sordino: peri_censor file {value} must hold only 0 and 1")
    if total % col.size:
        raise ValueError(f"sordino: peri_censor has {col.size} rows, which does not divide the "
                         f"{total} spokes of the stream")
    per = total // col.size
    keep = np.repeat(col.astype(bool), per)
    logger.info("sordino: peri_censor %s rows x %s spokes; %s of %s spokes usable (%.1f%%).",
                col.size, per, int(keep.sum()), total, 100.0 * keep.mean())
    return keep


def build(recon_info: Dict[str, Any], options, n_rep: int) -> Tuple[
        Tuple[Tuple[int, int], ...], Tuple[int, ...], Tuple[float, ...], Dict[str, Any]]:
    """``(ranges, groups, scales, info)`` for the peri-event plan.

    ``ranges`` are half-open spoke ranges in the stream of the repetitions read,
    ``groups[i]`` is the output bin of ``ranges[i]``, and ``scales[b]`` normalises
    bin ``b`` to the brightness of a one-repetition image.
    """
    spec = parse_peri_event(getattr(options, "peri_event", None))
    if spec is None:
        raise ValueError("sordino: peri_event is not set")
    tr = recon_info.get("RepetitionTime_ms")
    if tr in (None, 0, 0.0):
        raise ValueError("sordino: peri_event needs the spoke TR (RepetitionTime_ms), which this scan "
                         "does not report.")
    tr_ms = float(tr)
    npro = int(recon_info["NPro"])
    total = int(n_rep) * npro

    period = _spokes(spec["period"], tr_ms, "period")
    width = _spokes(spec["bin"], tr_ms, "bin")
    onset = _spokes(spec.get("onset", 0.0), tr_ms, "onset", signed=True) if spec.get("onset") else 0
    start = _spokes(spec.get("start", 0.0), tr_ms, "start", signed=True) if spec.get("start") else 0
    # default: ceil(period / width) bins, so the whole period is binned; when the bin width does
    # not divide the period the last bin runs past the period end into the next event's first bin
    n_bins = int(spec["bins"]) if "bins" in spec else max(1, -(-period // width))
    if width * n_bins > period:
        if "bins" in spec or period % width == 0:
            logger.warning("sordino: peri_event bins cover %s spokes, more than the %s of one period; "
                           "bins of neighbouring events overlap and spokes are gridded more than once.",
                           width * n_bins, period)
        else:
            logger.warning("sordino: peri_event bin (%s spokes) does not divide the period (%s spokes); "
                           "the default %s bins cover the whole period and the last bin runs %s spokes "
                           "into the next event, so those spokes are gridded in two bins.",
                           width, period, n_bins, width * n_bins - period)
    if width > total:
        raise ValueError(f"sordino: peri_event bin is {width} spokes, more than the {total} read.")

    max_events = int((total - onset - start) // period) + 1
    if "events" in spec:
        max_events = min(max_events, int(spec["events"]))
    if max_events < 1:
        raise ValueError(f"sordino: peri_event places no event inside the {total} spokes read "
                         f"(onset {onset}, period {period}).")

    keep = _load_censor(getattr(options, "peri_censor", None), total)

    ranges, groups = [], []
    used, dropped, censored = [0] * n_bins, 0, 0
    for k in range(max_events):
        base = onset + period * k + start
        for b in range(n_bins):
            lo = base + width * b
            hi = lo + width
            if lo < 0 or hi > total:
                dropped += 1
                continue
            if keep is not None and not keep[lo:hi].all():
                censored += 1
                continue
            ranges.append((lo, hi))
            groups.append(b)
            used[b] += 1
    if not ranges:
        raise ValueError("sordino: peri_event selected no spokes; check period, bin, onset and start.")
    empty = [b for b, n in enumerate(used) if n == 0]
    if empty:
        logger.warning("sordino: peri_event bin(s) %s got no spokes (data too short, events too few, or "
                       "all ranges censored); they are written as all-zero frames.", empty)
    # an empty bin has nothing to normalise: scale 0 keeps it an all-zero frame
    scales = tuple(float(npro) / (used[b] * width) if used[b] else 0.0 for b in range(n_bins))

    order = sorted(range(len(ranges)), key=lambda i: ranges[i])
    ranges = tuple(ranges[i] for i in order)
    groups = tuple(groups[i] for i in order)

    n_events = max(used)
    phases = npro // gcd(period % npro or npro, npro) if period % npro else 1
    unique = min(phases * width, npro)
    info = {
        "period_spokes": period, "bin_spokes": width, "onset_spokes": onset, "start_spokes": start,
        "n_bins": n_bins, "n_events": n_events, "events_per_bin": used, "dropped_bins": dropped,
        "trajectory_phases": phases, "unique_spokes_per_bin": unique,
        "censored_ranges": censored,
        "bin_edges_s": [(start + width * b) * tr_ms * 1e-3 for b in range(n_bins + 1)],
    }
    logger.info("sordino: peri_event %s bin(s) of %s spokes (%.3fs) from %s event(s) every %s spokes "
                "(%.3fs); %s spoke(s) per bin, %s distinct trajectory phase(s) -> %s unique spoke(s) "
                "per bin.", n_bins, width, width * tr_ms * 1e-3, n_events, period, period * tr_ms * 1e-3,
                n_events * width, phases, unique)
    if dropped:
        logger.info("sordino: %s peri-event bin(s) fell outside the spokes read and were left out.", dropped)
    if censored:
        logger.info("sordino: peri_censor left out %s of %s range(s) (%.1f%%); bins now hold %s-%s event(s) "
                    "and each bin's scale divides by its own count.", censored, censored + len(ranges),
                    100.0 * censored / (censored + len(ranges)), min(used), max(used))
    if unique < n_events * width:
        logger.warning(
            "sordino: the event period (%s spokes) and the repetition (%s spokes) share a factor, so the "
            "events land on only %s of the %s trajectory phases: a bin holds %s spoke(s) but only %s "
            "distinct direction(s), each sampled %.0f times. Averaging raises SNR but not k-space "
            "coverage; judge %s against the Nyquist spoke count of the reconstruction matrix.",
            period, npro, phases, npro // (width or 1), n_events * width, unique,
            (n_events * width) / unique, unique)
    return ranges, groups, scales, info


def load_censor(value, total: int):
    """Public form of the censor loader: a boolean mask over ``total`` stream spokes, or None."""
    return _load_censor(value, total)


__all__ = ["PERI_KEYS", "CENSOR_KEY", "parse_peri_event", "build", "load_censor"]
