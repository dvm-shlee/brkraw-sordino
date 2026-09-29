"""Evaluation tool for SORDINO ramp-time and phase corrections (development).

Plots the three checks the method owner uses to judge a reconstruction change:

(a) within a volume: how constant each spoke's FID "first peak" is across the
    spokes of one volume. Because of the TR delay the second sampled point
    (sample index 1 by default) is the maximum of every spoke and lies near
    the k-space centre;
(b) across volumes: how constant that within-volume pattern is from volume to
    volume (the first volumes before steady state are excluded), and the time
    course of the first-peak FID before reconstruction;
(c) after reconstruction: how large the oscillation of the centre-region value
    is over volumes, with the current ``correct_ramptime`` off and on.

It also writes a phase-timing observation used to check the phase model: for
the first samples of each spoke it finds the delay ``tau`` that best aligns
the spoke phases with the per-projection frequency steps of ``ACQ_O1_list``
(see ``phase_timing``). It does not correct anything.

Data are read with the brkraw Python API and reconstructed with the
brkraw-sordino functions the converter hook uses, so a baseline made now can
be compared with a later corrected version by running the same tool again.
The raw data are streamed one volume at a time (zip members are read
sequentially; nothing is extracted).

This file is a development tool: it is not part of the installed package.

Example (Python API)::

    from eval_ramp import EvalConfig, run
    run(EvalConfig(dataset="study.zip", scan_id=5, out_dir="eval/v1"))

Example (command line)::

    python tools/eval_ramp.py study.zip 5 eval/v1 --exclude 10 --sample-index 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:  # pure-Python summary statistics (same folder)
    import evalstats
except ImportError:  # pragma: no cover - run from another folder
    from tools import evalstats  # type: ignore


# ---------------------------------------------------------------------------
# Configuration: every value that changes what is measured is an argument.
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    """Arguments of one evaluation run.

    dataset:        PvDataset folder or zip (read only).
    scan_id:        scan to evaluate (a SORDINO functional scan).
    out_dir:        folder for arrays, figures and the JSON summary.
    exclude:        leading volumes left out of (b) and (c) (not at steady
                    state yet).
    sample_index:   FID sample used as the "first peak" in (a) and (b).
    n_keep:         leading samples kept per spoke for all volumes (phase
                    timing uses them); must be > sample_index.
    max_volumes:    stop after this many volumes (None = all).
    recon_start:    first volume reconstructed for (c) (None = ``exclude``).
    recon_count:    number of volumes reconstructed for (c) (0 = skip (c)).
    roi_half:       half width in voxels of the centre cube averaged in (c).
    ramp_modes:     trajectory variants compared in (c), see ``TRAJ_MODES``.
    ignore_samples: leading samples dropped before NUFFT (hook default 1).
    tau_range_us:   search range of the phase-timing delay (± microseconds).
    tau_step_us:    grid step of that search.
    phase_samples:  leading samples averaged over steady-state volumes for the
                    phase-timing observation (also split in even/odd volumes).
    cache_dir:      trajectory cache for brkraw-sordino (default out_dir/cache).
    label:          free text shown in figure titles (e.g. "v1 baseline").
    """

    dataset: str
    scan_id: int
    out_dir: str
    exclude: int = 10
    sample_index: int = 1
    n_keep: int = 8
    max_volumes: Optional[int] = None
    recon_start: Optional[int] = None
    recon_count: int = 60
    roi_half: int = 4
    ramp_modes: Tuple[str, ...] = ("off", "pre", "post_traj", "post")
    ignore_samples: int = 1
    tau_range_us: float = 400.0
    tau_step_us: float = 0.05
    phase_samples: int = 64
    cache_dir: Optional[str] = None
    label: str = ""
    spoke_volumes: Tuple[int, ...] = field(default_factory=tuple)

    def check(self) -> None:
        if self.exclude < 0:
            raise ValueError("exclude must be >= 0")
        if not 0 <= self.sample_index < self.n_keep:
            raise ValueError("sample_index must be in [0, n_keep)")
        if self.recon_count < 0 or self.roi_half < 0:
            raise ValueError("recon_count and roi_half must be >= 0")


# ---------------------------------------------------------------------------
# Reading through the brkraw / brkraw-sordino Python API
# ---------------------------------------------------------------------------


def open_scan(dataset: str, scan_id: int):
    """Open a scan with brkraw and parse the sordino reconstruction info.

    Returns (scan, recon_info, fid_entry, meta) where meta holds the extra
    method/acqp values this tool needs (O1 list, timing keys).
    """
    import brkraw
    from brkraw.resolver import fid as fid_resolver
    from brkraw_sordino.hook import _parse_recon_info

    loader = brkraw.load(dataset)
    scan = loader.get_scan(scan_id)
    recon_info = _parse_recon_info(scan)
    fid_entry = fid_resolver.get_fid(scan)
    if fid_entry is None:
        raise ValueError(f"scan {scan_id} has no fid/rawdata")
    method, acqp = scan.method, scan.acqp

    def val(params, key):
        value = params.get(key)
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.dtype.kind in "fiu":
            return arr.astype(float).tolist() if arr.ndim else float(arr)
        return str(value)

    keys_method = [
        "Method", "RepetitionTime", "PVM_RepetitionTime", "RampTime",
        "RampDelay", "TRWait", "GradSettle", "RFWait", "ADCEndWait",
        "EndOfScan", "AcqDelay", "AcqDelayTotal", "PVM_AcquisitionTime",
        "PVM_EffSWh", "OverSampling", "NPoints", "PVM_DummyScans",
        "MaximizeRampTime", "GradRes", "TrigSegmentMode",
    ]
    meta: Dict[str, Any] = {k: val(method, k) for k in keys_method}
    meta["DE"] = val(acqp, "DE")
    o1 = acqp.get("ACQ_O1_list")
    meta["o1_list"] = None if o1 is None else np.asarray(o1, dtype=float)
    return scan, recon_info, fid_entry, meta


def iter_volumes(fid_entry, recon_info: Dict[str, Any],
                 max_volumes: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (volume index, complex array [n_receivers, n_pro, n_points]).

    Uses the same FID layout as ``brkraw_sordino.recon.recon_dataobj``
    ([2, NPoints, NReceivers, NPro], Fortran order), one volume at a time.
    """
    from brkraw_sordino.recon import parse_fid_info

    fid_shape, fid_dtype = parse_fid_info(recon_info)
    buffer_size = int(np.prod(fid_shape) * fid_dtype.itemsize)
    n_total = int(recon_info["NRepetitions"])
    n_vol = n_total if max_volumes is None else min(n_total, max_volumes)
    with fid_entry.open() as fobj:
        for v in range(n_vol):
            buf = fobj.read(buffer_size)
            if len(buf) < buffer_size:
                break
            raw = np.frombuffer(buf, dtype=fid_dtype).reshape(fid_shape, order="F")
            cplx = (raw[0] + 1j * raw[1]).astype(np.complex64)  # [pts, rx, pro]
            yield v, np.transpose(cplx, (1, 2, 0))


def reconstruct(kspace_pro_pts: np.ndarray, traj: np.ndarray,
                volume_shape: Sequence[int], ignore_samples: int,
                phase_factor: Optional[np.ndarray] = None) -> np.ndarray:
    """One-channel adjoint NUFFT exactly as the hook does it."""
    from brkraw_sordino.recon import nufft_adjoint

    k = kspace_pro_pts if phase_factor is None else kspace_pro_pts * phase_factor
    k = k[..., ignore_samples:]
    return nufft_adjoint(k, traj[:, ignore_samples:, ...], volume_shape, 1)


TRAJ_MODES = ("off", "pre", "post_traj", "post")
"""Reconstruction variants compared in (c), all through brkraw-sordino.

off:        correct_ramptime=False (constant vector per spoke).
pre:        ramp_model="legacy", correct_phase=False: the code before WI-0056
            (brkraw-sordino bf4447b), k_j = s_j (g_prev + (g_cur - g_prev) j/N).
post_traj:  ramp_model="integral", correct_phase=False: integral trajectory
            only (BRK-0056), to separate its effect from the phase correction.
post:       ramp_model="integral", correct_phase=True: the new default.
"""

_MODE_OPTIONS = {
    "off": {"correct_ramptime": False, "correct_phase": False},
    "pre": {"ramp_model": "legacy", "correct_phase": False},
    "post_traj": {"ramp_model": "integral", "correct_phase": False},
    "post": {"ramp_model": "integral", "correct_phase": True},
}


def mode_options(mode: str, cache_dir: Path):
    """brkraw-sordino Options for one of ``TRAJ_MODES``."""
    from brkraw_sordino.hook import _build_options

    if mode not in TRAJ_MODES:
        raise ValueError(f"unknown mode {mode!r}")
    return _build_options(dict(_MODE_OPTIONS[mode], cache_dir=str(cache_dir)))


def trajectory(recon_info: Dict[str, Any], mode: str, cache_dir: Path):
    """(trajectory [n_pro, n_samples, 3], phase factor or None) for a mode."""
    from brkraw_sordino.recon import phase_correction_factor
    from brkraw_sordino.traj import get_trajectory

    options = mode_options(mode, cache_dir)
    traj = get_trajectory(recon_info, options)
    phase = phase_correction_factor(recon_info, options, int(traj.shape[1]))
    return traj, phase


# ---------------------------------------------------------------------------
# Phase timing observation (no correction)
# ---------------------------------------------------------------------------


def detect_version(meta: Dict[str, Any]) -> Optional[str]:
    """v1/v2/v3 from parameter keys unique to each sequence package.

    v3 (`sordino`): MaximizeRampTime or RFWait exist; v2
    (`sordino_260122_trig`): TrigSegmentMode exists; v1 (`mjm_zte_231005`
    family): RampDelay exists without the others. None otherwise.
    (Source-review brief, WI-0056, parsDefinition.h comparison.)
    """
    if meta.get("MaximizeRampTime") is not None or meta.get("RFWait") is not None:
        return "v3"
    if meta.get("TrigSegmentMode") is not None:
        return "v2"
    if meta.get("RampDelay") is not None:
        return "v1"
    return None


def phase_timing(z: np.ndarray, o1: np.ndarray, tau_range_us: float,
                 tau_step_us: float) -> Dict[str, Any]:
    """Delay that best aligns spoke phases with the O1 frequency steps.

    z:  complex [n_pro, n_samples], averaged over steady-state volumes.
    o1: ACQ_O1_list [n_pro] in Hz.

    Model tested (one sample j at a time): z[i, j] ~ A * exp(i*2*pi*d_i*tau_j)
    with d_i = o1[i-1] - o1[i] (the frequency step from the previous to the
    current projection; i = 0 uses the last projection). tau_j is found by
    maximising the coherence |sum_i z_ij exp(-i 2 pi d_i tau)| / sum_i |z_ij|
    over a grid. A second search does the same with o1[i] itself
    (tau_abs_j), which would be non-zero if the receiver phase reference were
    not the excitation. Returns tau (us), coherence at tau and at 0.
    """
    d = np.roll(o1, 1) - o1  # d_i = o1[i-1] - o1[i]
    out: Dict[str, Any] = {"tau_range_us": tau_range_us, "tau_step_us": tau_step_us}
    for name, freq in (("step", d), ("abs", o1)):
        out[name] = [_best_tau(z[:, j], freq, tau_range_us, tau_step_us, j)
                     for j in range(z.shape[1])]
    return out


def _coherence(zj: np.ndarray, freq: np.ndarray, taus_us: np.ndarray) -> np.ndarray:
    norm = float(np.sum(np.abs(zj))) or 1.0
    res = np.empty(taus_us.size)
    for start in range(0, taus_us.size, 256):
        t = taus_us[start:start + 256] * 1e-6
        res[start:start + 256] = np.abs(np.exp(-2j * np.pi * np.outer(t, freq)) @ zj) / norm
    return res


def _best_tau(zj, freq, tau_range_us, tau_step_us, j) -> Dict[str, Any]:
    """Coarse (1 us) then fine (tau_step_us) search of the coherence peak."""
    coarse = np.arange(-tau_range_us, tau_range_us + 0.5, 1.0)
    c = _coherence(zj, freq, coarse)
    t0 = float(coarse[int(np.argmax(c))])
    fine = np.arange(t0 - 2.0, t0 + 2.0 + tau_step_us / 2, tau_step_us)
    cf = _coherence(zj, freq, fine)
    k = int(np.argmax(cf))
    return {"sample": j, "tau_us": float(fine[k]), "coherence": float(cf[k]),
            "coherence_at_0": float(_coherence(zj, freq, np.array([0.0]))[0])}


def fit_line(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Least-squares y = a + b x with the standard error of b."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    se = float(np.sqrt(np.sum(res ** 2) / max(len(x) - 2, 1) / np.sum((x - x.mean()) ** 2)))
    return {"intercept": float(a), "slope": float(b), "slope_se": se,
            "rms_residual": float(np.sqrt(np.mean(res ** 2)))}


def current_code_tau_us(meta: Dict[str, Any], n_samples: int) -> List[float]:
    """tau_j implied by the current brkraw-sordino trajectory (traj.py).

    The code places sample j at k = t_j * (g_prev + (g_cur - g_prev) * j/N)
    for every version (ramp assumed to span the acquisition, N = NPoints), so
    the phase at the FOV offset relative to O1 = c.g_cur is
    2*pi*d_i*t_j*(1 - j/N): tau_j = t_j * (1 - j/N), t_j from RF centre.
    """
    try:
        adt = float(meta["AcqDelayTotal"])
        dwell = 1e6 / (float(meta["PVM_EffSWh"]) * float(meta["OverSampling"]))
        n = float(meta["NPoints"])
    except (KeyError, TypeError, ValueError):
        return []
    return [(adt + j * dwell) * (1.0 - j / n) for j in range(n_samples)]


def model_tau_us(meta: Dict[str, Any], version: str, n_samples: int,
                 curvature: float = 0.5) -> List[float]:
    """Predicted tau_j (us) of ``phase_timing`` for the clipped-ramp model.

    tau(t) = integral from RF centre to t of (1 - f(s)) ds, f = ramp fraction
    clip((s - t_ramp_start) / RampTime, 0, 1); curvature=0.5 is the gradient
    integral (other values scale the ramp term, for sensitivity checks only;
    the current code's own form is ``current_code_tau_us``).
    Times from the source-review brief (WI-0056): v1 ramp starts at TR start +
    10 us; v2/v3 at RF end. Returns [] if a needed value is missing.
    """
    try:
        p0_us = float(np.asarray(meta["ExcPulLength_ms"])) * 1e3
        ramp_us = float(meta["RampTime"]) * 1e3
        adt_us = float(meta["AcqDelayTotal"])
        dwell_us = 1e6 / (float(meta["PVM_EffSWh"]) * float(meta["OverSampling"]))
    except (KeyError, TypeError, ValueError):
        return []
    if version == "v1":
        ramp_delay_us = float(meta["RampDelay"]) * 1e3
        # t measured from TR start: ramp starts at 10 us, RF starts at
        # 10 us + RampDelay + 10 us (ppg lines 57, 60, 64, 65)
        t_rs = 10.0
        t_rfc = 20.0 + ramp_delay_us + p0_us / 2
    else:
        t_rfc = 0.0
        t_rs = p0_us / 2  # ramp starts at RF end
    out = []
    for j in range(n_samples):
        t = t_rfc + adt_us + j * dwell_us
        # tau = (t - t_rfc) - 2 * curvature * integral of f; with
        # curvature = 0.5 this is the integral of (1 - f).
        s = np.linspace(t_rfc, t, 2001)
        f = np.clip((s - t_rs) / ramp_us, 0.0, 1.0)
        out.append(float((t - t_rfc) - 2.0 * curvature * np.trapezoid(f, s)))
    return out


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------


def run(cfg: EvalConfig) -> Dict[str, Any]:
    """Stream the scan once, save arrays, figures and a JSON summary."""
    cfg.check()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cfg.cache_dir) if cfg.cache_dir else out / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    scan, recon_info, fid_entry, meta = open_scan(cfg.dataset, cfg.scan_id)
    exc = scan.method.get("ExcPul")
    try:
        meta["ExcPulLength_ms"] = float(np.asarray(exc[0] if isinstance(exc, (list, tuple)) else exc).ravel()[0])
    except Exception:
        meta["ExcPulLength_ms"] = None
    n_rx = int(recon_info["EncNReceivers"])
    n_pro = int(recon_info["NPro"])
    n_tot = int(recon_info["NRepetitions"])
    n_vol = n_tot if cfg.max_volumes is None else min(n_tot, cfg.max_volumes)

    from brkraw_sordino.recon import parse_volume_shape

    class _Opt:  # parse_volume_shape only reads ext_factors
        ext_factors = (1.0, 1.0, 1.0)

    vol_shape = parse_volume_shape(recon_info, _Opt())
    rs = cfg.exclude if cfg.recon_start is None else cfg.recon_start
    recon_ids = set(range(rs, min(n_vol, rs + cfg.recon_count)))
    trajs = {m: trajectory(recon_info, m, cache_dir) for m in cfg.ramp_modes} if recon_ids else {}
    # phase factor of the new default, also used for the first-peak phase metric
    post_phase = trajs["post"][1] if "post" in trajs else trajectory(recon_info, "post", cache_dir)[1]

    keep = np.zeros((n_vol, n_rx, n_pro, cfg.n_keep), dtype=np.complex64)
    n_ph = cfg.phase_samples
    zsum = {k: np.zeros((n_pro, n_ph), dtype=np.complex128) for k in ("all", "even", "odd")}
    zcnt = {k: 0 for k in zsum}
    roi = {m: [] for m in trajs}
    mean_img = {m: None for m in trajs}
    spoke_save = {}
    c = [s // 2 for s in vol_shape]
    h = cfg.roi_half
    sl = tuple(slice(ci - h, ci + h + 1) for ci in c)
    n_read = 0
    for v, vol in iter_volumes(fid_entry, recon_info, n_vol):
        keep[v] = vol[..., : cfg.n_keep]
        if v >= cfg.exclude:
            for k in ("all", "even" if v % 2 == 0 else "odd"):
                zsum[k] += vol[0, :, :n_ph]
                zcnt[k] += 1
        if v in cfg.spoke_volumes:
            spoke_save[v] = vol.copy()
        if v in recon_ids:
            for m, (tr, ph) in trajs.items():
                img = reconstruct(vol[0], tr, vol_shape, cfg.ignore_samples, ph)
                mag = np.abs(img)
                roi[m].append(float(mag[sl].mean()))
                mean_img[m] = mag if mean_img[m] is None else mean_img[m] + mag
        n_read = v + 1
    keep = keep[:n_read]
    np.save(out / "first_samples.npy", keep)
    for v, arr in spoke_save.items():
        np.save(out / f"volume_{v:04d}_raw.npy", arr)
    for m in mean_img:
        if mean_img[m] is not None:
            np.save(out / f"recon_mean_{m}.npy", mean_img[m] / len(roi[m]))

    # ---- metrics (a), (b), (c) --------------------------------------------
    fp = np.abs(keep[:, 0, :, cfg.sample_index]).astype(float)  # [vol, spoke]
    cv_per_vol = [evalstats.cv(fp[v].tolist()) for v in range(n_read)]
    ss = fp[cfg.exclude:]
    stab = (evalstats.pattern_stability(fp.tolist(), exclude=cfg.exclude)
            if n_read - cfg.exclude >= 2 else [])
    tc = fp.mean(axis=1).tolist()
    osc_fid = (evalstats.oscillation(tc, exclude=cfg.exclude)
               if n_read - cfg.exclude >= 3 else None)
    cv_over_vol = (np.std(ss, axis=0) / np.mean(ss, axis=0)).tolist() if ss.shape[0] > 1 else []
    osc_recon = {str(m): (evalstats.oscillation(roi[m]) if len(roi[m]) >= 3 else None)
                 for m in roi}
    sharp = {}
    for m, img in mean_img.items():
        if img is not None:
            mean = img / len(roi[m])
            grad = np.sqrt(sum(np.square(np.gradient(mean, axis=a)) for a in range(3)))
            sharp[m] = float(grad.mean() / mean.mean())
    # first-peak phase coherence across spokes, |sum z| / sum |z| per volume:
    # the phase correction has unit magnitude, so |first peak| (a, b) cannot
    # change; its effect on the raw FID shows in the phase agreement.
    z1 = keep[:, 0, :, cfg.sample_index].astype(np.complex128)
    coh_pre = (np.abs(z1.sum(axis=1)) / np.abs(z1).sum(axis=1)).tolist()
    coh_post = None
    if post_phase is not None:
        z1p = z1 * post_phase[None, :, cfg.sample_index]
        coh_post = (np.abs(z1p.sum(axis=1)) / np.abs(z1p).sum(axis=1)).tolist()
    rel_diff = {}
    for a, b in (("pre", "post"), ("pre", "post_traj"), ("post_traj", "post"), ("off", "pre")):
        if mean_img.get(a) is not None and mean_img.get(b) is not None:
            ma, mb = mean_img[a] / len(roi[a]), mean_img[b] / len(roi[b])
            rel_diff[f"{a}-{b}"] = float(np.linalg.norm(ma - mb) / np.linalg.norm(mb))

    # ---- phase timing observation ----------------------------------------
    phase = None
    o1 = meta.get("o1_list")
    version = detect_version(meta)
    if o1 is not None and o1.size == n_pro and zcnt["all"] >= 1:
        np.save(out / "phase_mean_samples.npy", zsum["all"] / zcnt["all"])
        phase = {k: phase_timing(zsum[k] / zcnt[k], o1, cfg.tau_range_us, cfg.tau_step_us)
                 for k in zsum if zcnt[k] >= 1}
        dwell = 1e6 / (float(meta["PVM_EffSWh"]) * float(meta["OverSampling"]))
        t_us = [float(meta["AcqDelayTotal"]) + j * dwell for j in range(n_ph)]
        phase["sample_time_from_rf_centre_us"] = t_us
        phase["version"] = version
        phase["model_tau_integral_us"] = model_tau_us(meta, version or "", n_ph, 0.5) if version else []
        phase["model_tau_code_form_us"] = current_code_tau_us(meta, n_ph)
        fits = {}
        for n_fit in (8, 16, 32, n_ph):
            n_fit = min(n_fit, n_ph)
            taus = [r["tau_us"] for r in phase["all"]["step"][:n_fit]]
            fits[str(n_fit)] = fit_line(t_us[:n_fit], taus)
        phase["line_fit_tau_vs_time"] = fits

    summary = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
        "method": meta.get("Method"),
        "version": version,
        "n_volumes_read": n_read, "n_pro": n_pro, "volume_shape": list(vol_shape),
        "timing": {k: v for k, v in meta.items() if k != "o1_list"},
        "a_cv_within_volume": {
            "median": float(np.median(cv_per_vol[cfg.exclude:] or cv_per_vol)),
            "per_volume_file": "cv_per_volume.json"},
        "b_pattern_stability_r": ({"min": float(min(stab)), "median": float(np.median(stab))}
                                  if stab else None),
        "b_first_peak_timecourse_oscillation": osc_fid,
        "b_cv_over_volumes_median": float(np.median(cv_over_vol)) if cv_over_vol else None,
        "c_recon_roi_oscillation": osc_recon,
        "c_mean_image_sharpness": sharp,
        "c_mean_image_relative_difference": rel_diff,
        "note_ab": ("(a) and (b) use |raw FID|: neither the trajectory nor the "
                    "unit-magnitude phase correction can change them"),
        "a_first_peak_phase_coherence": {
            "pre_median": float(np.median(coh_pre[cfg.exclude:] or coh_pre)),
            "post_median": (float(np.median(coh_post[cfg.exclude:] or coh_post))
                            if coh_post is not None else None),
            "per_volume_file": "phase_coherence_per_volume.json"},
        "phase_timing": phase,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "cv_per_volume.json").write_text(json.dumps(cv_per_vol))
    (out / "phase_coherence_per_volume.json").write_text(
        json.dumps({"pre": coh_pre, "post": coh_post}))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    plot(out, cfg, fp, cv_per_vol, stab, tc, cv_over_vol, roi, mean_img, phase,
         coh_pre, coh_post)
    return summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_orthogonal(path: Path, images: Dict[str, np.ndarray], title: str,
                    pair: Tuple[str, str] = ("pre", "post")) -> None:
    """Axial / coronal / sagittal centre planes of each image and of b - a.

    Array axes are taken as (x, y, z) in reconstruction order; the planes are
    the centre z (axial), y (coronal) and x (sagittal) slices. Display only:
    no orientation correction is applied.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in pair if n in images]
    rows = names + ([f"{pair[1]} - {pair[0]}"] if len(names) == 2 else [])
    fig, ax = plt.subplots(len(rows), 3, figsize=(12, 4 * len(rows)))
    ax = np.atleast_2d(ax)
    ref = images[names[0]]
    cx, cy, cz = (s // 2 for s in ref.shape)
    vmax = float(np.percentile(ref, 99.5))

    def planes(img):
        return img[:, :, cz].T, img[:, cy, :].T, img[cx, :, :].T

    diff = images[names[1]] - images[names[0]] if len(names) == 2 else None
    lim = float(np.percentile(np.abs(diff), 99.5)) if diff is not None else 1.0
    for r, name in enumerate(rows):
        img = diff if (diff is not None and r == len(rows) - 1) else images[name]
        for c, (plane, label) in enumerate(zip(planes(img), ("axial (z)", "coronal (y)", "sagittal (x)"))):
            if img is diff:
                im = ax[r, c].imshow(plane, cmap="coolwarm", vmin=-lim, vmax=lim, origin="lower")
            else:
                im = ax[r, c].imshow(plane, cmap="gray", vmin=0, vmax=vmax, origin="lower")
            ax[r, c].set_title(f"{name}: {label}")
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        if img is diff:
            rel = float(np.linalg.norm(diff) / np.linalg.norm(images[names[0]]))
            ax[r, 0].set_ylabel(f"|diff|/|{pair[0]}| = {rel:.4f}")
            fig.colorbar(im, ax=ax[r, 2], fraction=0.046)
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(path, dpi=110); plt.close(fig)


def plot(out: Path, cfg: EvalConfig, fp, cv_per_vol, stab, tc, cv_over_vol,
         roi, mean_img, phase, coh_pre=None, coh_post=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    title = f"{cfg.label or Path(cfg.dataset).name} scan {cfg.scan_id}"
    n_vol = fp.shape[0]
    ex = cfg.exclude

    # (a) first peak across spokes (+ phase coherence pre/post)
    fig, ax = plt.subplots(3, 1, figsize=(11, 10))
    for v in sorted({min(ex, n_vol - 1), n_vol // 2, n_vol - 1}):
        ax[0].plot(fp[v], lw=0.5, label=f"volume {v}")
    ax[0].set_xlabel("spoke"); ax[0].set_ylabel(f"|FID[{cfg.sample_index}]|")
    ax[0].legend(); ax[0].set_title(f"(a) first peak across spokes - {title}")
    ax[1].plot(cv_per_vol, lw=0.8)
    ax[1].axvline(ex - 0.5, color="grey", ls="--", lw=0.8)
    ax[1].set_xlabel("volume"); ax[1].set_ylabel("CV of |first peak| across spokes")
    if coh_pre is not None:
        ax[2].plot(coh_pre, lw=0.8, label="pre (raw FID)")
        if coh_post is not None:
            ax[2].plot(coh_post, lw=0.8, label="post (phase corrected)")
        ax[2].set_xlabel("volume"); ax[2].set_ylabel("|sum z| / sum |z|")
        ax[2].set_title("first-peak phase coherence across spokes (1 = all in phase)")
        ax[2].legend()
    fig.tight_layout(); fig.savefig(out / "a_first_peak_within_volume.png", dpi=120); plt.close(fig)

    # (b) pattern across volumes and first-peak time course
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    rel = fp / fp[ex:].mean(axis=0, keepdims=True) if n_vol > ex else fp
    im = ax[0, 0].imshow(rel, aspect="auto", cmap="coolwarm", vmin=0.8, vmax=1.2,
                         interpolation="nearest")
    fig.colorbar(im, ax=ax[0, 0]); ax[0, 0].set_xlabel("spoke"); ax[0, 0].set_ylabel("volume")
    ax[0, 0].set_title("(b) first peak / steady-state mean")
    ax[0, 1].plot(range(ex, ex + len(stab)), stab, lw=0.8)
    ax[0, 1].set_xlabel("volume"); ax[0, 1].set_ylabel("r(pattern, mean pattern)")
    ax[0, 1].set_title(f"(b) pattern stability (first {ex} excluded)")
    ax[1, 0].plot(tc, lw=0.8); ax[1, 0].axvline(ex - 0.5, color="grey", ls="--", lw=0.8)
    ax[1, 0].set_xlabel("volume"); ax[1, 0].set_ylabel("mean |first peak|")
    ax[1, 0].set_title("(b) first-peak FID time course (before recon)")
    ax[1, 1].plot(cv_over_vol, lw=0.5)
    ax[1, 1].set_xlabel("spoke"); ax[1, 1].set_ylabel("CV over volumes")
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(out / "b_first_peak_across_volumes.png", dpi=120); plt.close(fig)

    # (c) reconstruction: centre ROI over volumes, centre-line profile
    if any(roi.values()):
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        for m, vals in roi.items():
            if vals:
                ax[0].plot(vals, lw=0.8, label=f"trajectory={m}")
        ax[0].set_xlabel(f"volume (from {cfg.recon_start if cfg.recon_start is not None else ex})")
        ax[0].set_ylabel(f"mean |image| in centre {2*cfg.roi_half+1}^3 voxels")
        ax[0].legend(); ax[0].set_title("(c) centre-region value over volumes")
        for m, img in mean_img.items():
            if img is not None:
                mean = img / max(len(roi[m]), 1)
                cz, cy = mean.shape[2] // 2, mean.shape[1] // 2
                ax[1].plot(mean[:, cy, cz], lw=0.8, label=f"trajectory={m}")
        ax[1].set_xlabel("voxel along axis 0 (centre line)"); ax[1].set_ylabel("mean |image|")
        ax[1].legend(); ax[1].set_title("(c) centre line of the mean image")
        fig.suptitle(title); fig.tight_layout()
        fig.savefig(out / "c_recon_centre.png", dpi=120); plt.close(fig)

        # (c) mid slices of the mean image per mode and differences
        means = {m: img / max(len(roi[m]), 1) for m, img in mean_img.items() if img is not None}
        if means:
            modes = list(means)
            pairs = [(a, b) for a, b in (("post", "pre"), ("post", "post_traj")) if a in means and b in means]
            if "pre" in means and "post" in means:
                plot_orthogonal(out / "c_orthogonal_pre_post.png", means,
                                f"{title}: mean image, pre vs post")
            ncol = len(modes) + len(pairs)
            fig, ax = plt.subplots(1, ncol, figsize=(4 * ncol, 4.2))
            ax = np.atleast_1d(ax)
            ref = means[modes[0]]
            zc = ref.shape[2] // 2
            vmax = float(np.percentile(ref[:, :, zc], 99.5))
            for k, m in enumerate(modes):
                ax[k].imshow(means[m][:, :, zc].T, cmap="gray", vmin=0, vmax=vmax, origin="lower")
                ax[k].set_title(f"{m} (mid slice)")
            for k, (a, b) in enumerate(pairs):
                diff = (means[a] - means[b])[:, :, zc].T
                lim = float(np.percentile(np.abs(diff), 99.5)) or 1.0
                im = ax[len(modes) + k].imshow(diff, cmap="coolwarm", vmin=-lim, vmax=lim, origin="lower")
                rel = float(np.linalg.norm(means[a] - means[b]) / np.linalg.norm(means[b]))
                ax[len(modes) + k].set_title(f"{a} - {b} (|diff|/|img| = {rel:.4f})")
                fig.colorbar(im, ax=ax[len(modes) + k], fraction=0.046)
            for a_ in ax:
                a_.set_xticks([]); a_.set_yticks([])
            fig.suptitle(title); fig.tight_layout()
            fig.savefig(out / "c_recon_slices.png", dpi=120); plt.close(fig)

    # phase timing observation
    if phase:
        t = phase["sample_time_from_rf_centre_us"]
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
        for key, style in (("all", "o"), ("even", "."), ("odd", "x")):
            if key in phase:
                ax[0].plot(t, [r["tau_us"] for r in phase[key]["step"]], style, ms=3,
                           label=f"measured ({key} volumes)")
        if phase.get("model_tau_integral_us"):
            ax[0].plot(t, phase["model_tau_integral_us"], "-", label="model: gradient integral")
            ax[0].plot(t, phase["model_tau_code_form_us"], "--", label="model: current code form")
        ax[0].plot(t, t, ":", color="grey", label="tau = t (no ramp)")
        ax[0].set_xlabel("sample time from RF centre (us)"); ax[0].set_ylabel("tau (us)")
        ax[0].legend(fontsize=7); ax[0].set_title("phase delay vs O1 step (per sample)")
        if phase.get("model_tau_integral_us"):
            meas = np.array([r["tau_us"] for r in phase["all"]["step"]])
            ax[1].plot(t, meas - np.array(phase["model_tau_integral_us"]), "o", ms=3,
                       label="measured - integral model")
            ax[1].plot(t, meas - np.array(phase["model_tau_code_form_us"]), "x", ms=3,
                       label="measured - code-form model")
            ax[1].axhline(0, color="grey", lw=0.8); ax[1].legend(fontsize=7)
        ax[1].set_xlabel("sample time from RF centre (us)"); ax[1].set_ylabel("residual (us)")
        ax[1].set_title("residual to models")
        rows = phase["all"]["step"]
        ax[2].plot(t, [r["coherence"] for r in rows], "o-", ms=3, label="O1 step: at tau")
        ax[2].plot(t, [r["coherence_at_0"] for r in rows], ":", label="O1 step: at 0")
        rows = phase["all"]["abs"]
        ax[2].plot(t, [r["coherence"] for r in rows], "s--", ms=3, label="O1 itself: at tau")
        ax[2].set_xlabel("sample time from RF centre (us)"); ax[2].set_ylabel("coherence")
        ax[2].legend(fontsize=7); ax[2].set_title("coherence (1 = all spokes aligned)")
        fig.suptitle(f"{title} ({phase.get('version')})"); fig.tight_layout()
        fig.savefig(out / "phase_timing.png", dpi=120); plt.close(fig)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dataset"); p.add_argument("scan_id", type=int); p.add_argument("out_dir")
    p.add_argument("--exclude", type=int, default=10)
    p.add_argument("--sample-index", type=int, default=1)
    p.add_argument("--n-keep", type=int, default=8)
    p.add_argument("--max-volumes", type=int, default=None)
    p.add_argument("--recon-start", type=int, default=None)
    p.add_argument("--recon-count", type=int, default=60)
    p.add_argument("--roi-half", type=int, default=4)
    p.add_argument("--label", default="")
    a = p.parse_args(argv)
    run(EvalConfig(dataset=a.dataset, scan_id=a.scan_id, out_dir=a.out_dir,
                   exclude=a.exclude, sample_index=a.sample_index, n_keep=a.n_keep,
                   max_volumes=a.max_volumes, recon_start=a.recon_start,
                   recon_count=a.recon_count, roi_half=a.roi_half, label=a.label))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
