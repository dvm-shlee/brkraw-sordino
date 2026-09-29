"""Sequence timing for the SORDINO ramp-time and phase corrections.

Every delay or timing value used by the trajectory (traj.py) and the FID
phase correction (recon.py) is declared in this module, once:

* ``SeqTiming``      [read]  values taken from method/acqp (or computed from
                             them) and fixed pulse-program delays. Do not edit.
* ``TimingTuning``   [tune]  per-version adjustments, all zero by default.
  ``TIMING_TUNING``          Edit these to adjust the model while comparing
                             results (they are part of the cache keys).

Time origin: the centre of the excitation pulse (RF centre). All times in
this module are microseconds (us) unless the name says otherwise.

Model (WI-0056, BRK-0056): during projection i the gradient moves linearly
from the previous vector g(i-1) to the current vector g(i),

    G(t) = g(i-1) + (g(i) - g(i-1)) * f(t),
    f(t) = clip((t - ramp_start) / ramp_length, 0, 1),

and the sample position is k(t) = integral of G from the RF centre to t.
The receiver frequency of projection i (ACQ_O1_list[i]) matches g(i), so the
FID also carries the phase 2*pi*(O1[i-1] - O1[i]) * integral of (1 - f).
Only the ramp window (ramp_start, ramp_length) differs between versions:

    version  sequence package          ramp starts                     source
    v1       mjm_zte_231005            10 us after TR start, i.e.       mjm ppg:57,60,64,65
                                       RampDelay + 10 us before RF start
    v2       sordino_260122_trig       at the end of the RF pulse       260122 ppg:70-72
    v3       sordino                   at the end of the RF pulse       sordino ppg:95,106
    zte      general ZTE (no RampTime) before the RF; the gradient is   (model: ramp finished
                                       constant during RF and sampling   before the first sample)

v2 and v3 are one model: how much of the ramp falls inside the acquisition
follows from RampTime versus the acquisition time. v1 with a short RampTime
reaches the target before the first sample; the same formula covers it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional

from .ramp import phase_delay, ramp_integral

# ---------------------------------------------------------------------------
# Fixed pulse-program delays [read, constant in the sequence source]
# ---------------------------------------------------------------------------

#: v1: the transmit-frequency event between the ramp window and the RF pulse
#: (``10u fList:f1``, mjm_zte_231005.ppg:64). The ramp itself starts at the
#: beginning of the ``rampDelay`` event (ppg:60), after the 10 us receive
#: event (ppg:57). [read, us]
V1_TX_EVENT_US = 10.0


# ---------------------------------------------------------------------------
# [read] values from method/acqp
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeqTiming:
    """[read] Timing of one scan, from recon_info (method/acqp).

    version:             "v1" | "v2" | "v3" | "zte" (detect_version).
    rf_len_us:           excitation pulse length; method ``ExcPul`` (first
                         field, ms) x 1000. All versions.
    acq_delay_total_us:  RF centre to the first sample; method
                         ``AcqDelayTotal`` (= P0/2 + DE + AcqDelay, mjm
                         backbone.c:253). All versions.
    dwell_us:            sample interval; 1e6 / (``PVM_EffSWh`` x
                         ``OverSampling``). All versions.
    ramp_time_us:        ramp length; method ``RampTime`` (ms) x 1000.
                         v1 (user value, up to TR - 0.01 ms), v2 (AcqDelay +
                         DE + Acq + 5 us, 260122 backbone.c:148-152), v3
                         (TR - GradSettle when MaximizeRampTime, sordino
                         backbone.c:406-425). None for zte.
    ramp_delay_us:       v1: length of the ramp window before the RF
                         (``RampDelay``, ms x 1000, mjm backbone.c:307).
                         Also stored by v2 (a wait with no ramp); unused there.
    """

    version: str
    rf_len_us: float
    acq_delay_total_us: float
    dwell_us: float
    ramp_time_us: Optional[float] = None
    ramp_delay_us: Optional[float] = None


# ---------------------------------------------------------------------------
# [tune] per-version adjustments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimingTuning:
    """[tune] Adjustments added to the model. All default to 0 (pure model).

    ramp_start_offset_us:  added to the ramp start time.
    ramp_len_offset_us:    added to RampTime.
    acq_start_offset_us:   added to every sample time (first-sample timing).
    phase_ref_us:          phase reference time relative to the RF centre
                           (0 = RF centre, observed on v2 in WI-0056 run 3).

    Observation to adjust against (WI-0056 run 3, not applied): the phase of
    the first samples followed the model on v2 but was offset by about
    -2.5 us on v1 and +2.4 us on v3 (delay of the O1-step phase versus the
    sample time). The cause is not known; try ``acq_start_offset_us`` or
    ``phase_ref_us`` with the evaluation tool before changing the model.
    """

    ramp_start_offset_us: float = 0.0
    ramp_len_offset_us: float = 0.0
    acq_start_offset_us: float = 0.0
    phase_ref_us: float = 0.0


#: [tune] Edit here. One entry per version.
TIMING_TUNING: Dict[str, TimingTuning] = {
    "v1": TimingTuning(),
    "v2": TimingTuning(),
    "v3": TimingTuning(),
    "zte": TimingTuning(),
}


# ---------------------------------------------------------------------------
# Version detection and timing from recon_info
# ---------------------------------------------------------------------------


def detect_version(recon_info: Mapping[str, Any]) -> str:
    """Sequence version from parameter keys unique to each package.

    v3 (`sordino`): ``MaximizeRampTime`` or ``RFWait`` present.
    v2 (`sordino_260122_trig`): ``TrigSegmentMode`` present.
    v1 (`mjm_zte_231005` family): ``RampDelay`` and ``RampTime`` present.
    zte (general ZTE): no ``RampTime``; the gradient is taken as constant
    during RF and sampling.
    (parsDefinition.h comparison, WI-0056 source-review brief.)
    """
    if recon_info.get("MaximizeRampTime") is not None or recon_info.get("RFWait_ms") is not None:
        return "v3"
    if recon_info.get("TrigSegmentMode") is not None:
        return "v2"
    if recon_info.get("RampTime_ms") is not None and recon_info.get("RampDelay_ms") is not None:
        return "v1"
    if recon_info.get("RampTime_ms") is None:
        return "zte"
    raise ValueError("SORDINO version not recognised from method parameters "
                     "(RampTime present without RampDelay/TrigSegmentMode/MaximizeRampTime)")


def read_timing(recon_info: Mapping[str, Any]) -> SeqTiming:
    """Build ``SeqTiming`` from recon_info (see recon_spec.yaml)."""
    version = detect_version(recon_info)
    rf_ms = recon_info.get("ExcPulLength_ms")
    adt = recon_info.get("AcqDelayTotal_us")
    bw = recon_info.get("EffBandwidth_Hz")
    os_ = recon_info.get("OverSampling")
    missing = [n for n, v in (("ExcPul", rf_ms), ("AcqDelayTotal", adt),
                              ("PVM_EffSWh", bw), ("OverSampling", os_)) if v is None]
    if missing:
        raise ValueError(f"timing values missing for the ramp model: {missing}")
    ramp = recon_info.get("RampTime_ms")
    delay = recon_info.get("RampDelay_ms")
    if version in ("v1", "v2", "v3") and ramp is None:
        raise ValueError(f"{version}: RampTime missing")
    return SeqTiming(
        version=version,
        rf_len_us=float(rf_ms) * 1e3,
        acq_delay_total_us=float(adt),
        dwell_us=1e6 / (float(bw) * float(os_)),
        ramp_time_us=None if ramp is None else float(ramp) * 1e3,
        ramp_delay_us=None if delay is None else float(delay) * 1e3,
    )


def tuning_for(version: str, override: Optional[TimingTuning] = None) -> TimingTuning:
    """Tuning in effect for a version (``override`` replaces the table)."""
    return override if override is not None else TIMING_TUNING.get(version, TimingTuning())


# ---------------------------------------------------------------------------
# Derived quantities used by traj.py and recon.py
# ---------------------------------------------------------------------------


def ramp_window(t: SeqTiming, tune: TimingTuning) -> Optional[tuple]:
    """(ramp_start_us, ramp_length_us) relative to the RF centre, or None.

    None means the gradient is constant (equal to g(i)) during sampling.
    """
    if t.version == "zte":
        return None
    length = float(t.ramp_time_us) + tune.ramp_len_offset_us
    if length <= 0:
        raise ValueError("ramp length must be > 0 (RampTime + ramp_len_offset_us)")
    if t.version == "v1":
        start = -(V1_TX_EVENT_US + float(t.ramp_delay_us) + t.rf_len_us / 2.0)
    else:  # v2, v3: ramp starts at the end of the RF pulse
        start = t.rf_len_us / 2.0
    return start + tune.ramp_start_offset_us, length


def sample_times_us(t: SeqTiming, tune: TimingTuning, n_samples: int) -> List[float]:
    """Time of each sample from the RF centre."""
    t0 = t.acq_delay_total_us + tune.acq_start_offset_us
    return [t0 + j * t.dwell_us for j in range(n_samples)]


def ramp_terms(t: SeqTiming, tune: TimingTuning, n_samples: int):
    """Per-sample (time, ramp integral, phase delay), all in us.

    time_j:   sample time from the RF centre;
    F_j:      integral of f from the RF centre to time_j (k-space ramp term);
    tau_j:    integral of (1 - f) from phase_ref to time_j (phase term).
    For a constant gradient (zte) F_j = time_j and tau_j = 0.
    """
    times = sample_times_us(t, tune, n_samples)
    win = ramp_window(t, tune)
    if win is None:
        return times, list(times), [0.0] * n_samples
    start, length = win
    F = [ramp_integral(x, start, length) for x in times]
    ref = phase_delay(tune.phase_ref_us, start, length)
    tau = [phase_delay(x, start, length) - ref for x in times]
    return times, F, tau


def dead_time_kgrid(t: SeqTiming, over_sampling: float) -> float:
    """[read] Radius of the unsampled k-space centre in k-grid units (1/FOV).

    The first sample is acq_delay_total_us / dwell_us samples after the RF
    centre; one k-grid unit is ``over_sampling`` samples along a spoke.
    About 0.5 for the SORDINO data in WI-0056; 2.5 for the general-ZTE
    fixture `triggerzte3`.
    """
    return t.acq_delay_total_us / t.dwell_us / float(over_sampling)


def kspace_gap(t: SeqTiming, over_sampling: float, ignore_samples: int = 1) -> Dict[str, Any]:
    """[read] The unsampled k-space centre of a scan, for logs and the result
    metadata (BRK-0060). Nothing here fills the centre (BRK-0059).

    gap_kgrid:       radius of the first acquired sample (sample 0), i.e. the
                     dead time, in k-grid units (1/FOV); ``dead_time_kgrid``.
    gap_used_kgrid:  radius of the first sample the reconstruction keeps
                     (sample ``ignore_samples``); the adjoint leaves this
                     radius empty.
    Radii are along the spoke: exact for a constant gradient (zte); for the
    ramped SORDINO trajectory the vector step moves a sample by a few
    hundredths of a k-grid unit (WI-0056 run 5), which this ignores.
    """
    step = 1.0 / t.dwell_us / float(over_sampling)   # k-grid units per us
    n_skip = max(int(ignore_samples), 0)
    return {
        "sequence_version": t.version,
        "dead_time_us": float(t.acq_delay_total_us),
        "gap_kgrid": float(t.acq_delay_total_us * step),
        "ignore_samples": n_skip,
        "gap_used_kgrid": float((t.acq_delay_total_us + n_skip * t.dwell_us) * step),
        "centre_filled": False,
    }


def describe(t: SeqTiming, tune: TimingTuning) -> Dict[str, Any]:
    """Plain dict for logs, metadata and cache keys."""
    out = {"timing": asdict(t), "tuning": asdict(tune)}
    win = ramp_window(t, tune)
    out["ramp_window_us"] = None if win is None else [win[0], win[1]]
    return out


__all__ = [
    "SeqTiming", "TimingTuning", "TIMING_TUNING", "V1_TX_EVENT_US",
    "detect_version", "read_timing", "tuning_for", "ramp_window",
    "sample_times_us", "ramp_terms", "dead_time_kgrid", "kspace_gap", "describe",
]
