"""Timing self-calibration and k-space centre filling for SORDINO (WI-0058).

Two reconstruction-level methods, tried as extra stages of the WI-0056
stage matrix (``eval_stages``), no product (``src/``) change:

S4p / S4 — self-calibrated first-sample timing
    The run-3 phase-timing observation (WI-0056) found the O1-step phase of
    the first samples offset from the BRK-0056 model by about -2.5 us (v1) and
    +2.4 us (v3). A sample-time shift delta enters the same model as

        t_j = AcqDelayTotal + delta + j * dwell            (timing.py tuning
                                                             acq_start_offset_us)

    The run-3 style estimate (``estimate_delay``: maximise the O1-step
    coherence C_j(delta) = |sum_i z_ij exp(-i 2 pi d_i tau_j(delta))| / sum_i |z_ij|)
    is biased by the object's own spoke phase: an off-centre object gives
    about -2 us with no delay at all (``simulate``). The estimate used by the
    stages is therefore data consistency (``estimate_timing``):

        delta_hat = argmin_delta || W^1/2 (A_delta x_delta - y) || / || W^1/2 y ||,
        x_delta   = least-squares image for that delta (conjugate gradient),

    searched separately for the trajectory (delta_traj) and the O1-step phase
    (delta_phase), alternating. The object's phase is part of x, so it cannot
    mimic the per-spoke phase or the radial shift. This is the SORDINO
    counterpart of data-driven delay estimation in radial MRI (Peters 2003,
    Deshmane 2016, Rosenzweig 2019) and of Arc-ZTE's empirical timing tuning
    by image quality (Ramachandran 2026).
    Settings found necessary on real data (WI-0058 brief): samples and the
    weight fixed by the delta = 0 trajectory; an image grid of 4x the FOV
    (radial ZTE also receives signal from outside the FOV); the residual
    measured with uniform weights (the |k|^2 weight puts it on noise-dominated
    outer samples); a low-k sub-problem (factor 4). Even so, on real data the
    criterion also "prefers" a k-scale of about 1.18 on every scan tested,
    the control included, so a real-data radial estimate is an observation,
    not a validated correction.
    S4p applies delta_phase only (trajectory as S3); S4 applies both.

S4z — estimated k-space centre (dead-time gap)
    The first kept sample (sample 1; sample 0 is dropped by the hook) lies
    0.5-1.2 k-grid units from the centre. S4z solves the weighted
    least-squares problem

        x_hat = argmin_x || W^{1/2} (A x - y) ||^2       (conjugate gradient,
                                                          x0 = 0, 10 iterations,
                                                          grid 2x the FOV)

    on the S4 trajectory and phase; the finite grid is the support constraint
    that determines the unsampled centre (Kuethe 1999; Weiger 2010/2011;
    Froidevaux 2018: algebraic filling works for gaps up to about three
    Nyquist dwells). Only its prediction at virtual leading samples (times
    t_first - m * dwell down to the RF centre, ``leading_points``) is used:
    the image is the product's adjoint over those estimated samples plus all
    measured ones (``centre_fill_reconstruct``). The CG image itself converges
    slowly at high k on real data (blurred after 10 iterations), while the
    centre values converge in a few iterations.

This file is a development tool: it is not part of the installed package.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import curvefit
    import eval_ramp
    import peakfit
except ImportError:  # pragma: no cover - run from another folder
    from tools import curvefit, eval_ramp, peakfit  # type: ignore


# ---------------------------------------------------------------------------
# Timing with a sample-time shift delta (tool side of timing.TimingTuning)
# ---------------------------------------------------------------------------


def _n_samples(recon_info: Dict[str, Any]) -> int:
    return int(int(recon_info["Matrix"][0]) / 2 * float(recon_info["OverSampling"]))


def tuned_terms(recon_info: Dict[str, Any], delta_us: float, n_points: Optional[int] = None,
                ramp_offset_us: float = 0.0):
    """(times, F, tau) in us with ``acq_start_offset_us`` increased by delta_us
    (and ``ramp_start_offset_us`` by ramp_offset_us: a gradient delay moves the
    ramp, not the sample times).

    The version's own tuning table entry is kept and only the offsets are
    added, so zero offsets reproduce the product exactly.
    """
    from brkraw_sordino import timing

    seq = timing.read_timing(recon_info)
    base = timing.tuning_for(seq.version)
    tune = replace(base, acq_start_offset_us=base.acq_start_offset_us + float(delta_us),
                   ramp_start_offset_us=base.ramp_start_offset_us + float(ramp_offset_us))
    n = _n_samples(recon_info) if n_points is None else int(n_points)
    return timing.ramp_terms(seq, tune, n)


def delayed_trajectory(recon_info: Dict[str, Any], delta_us: float, ramp_offset_us: float = 0.0) -> np.ndarray:
    """(n_pro, N, 3) integral trajectory with the sample times shifted by
    delta_us (and the ramp start by ramp_offset_us)."""
    from brkraw_sordino import timing
    from brkraw_sordino.traj import calc_radial_grad3d, calc_radial_traj3d_integral

    seq = timing.read_timing(recon_info)
    matrix = int(recon_info["Matrix"][0])
    grad = calc_radial_grad3d(matrix, int(recon_info["NPro"]), bool(recon_info["HalfAcquisition"]),
                              bool(recon_info["UseOrigin"]), bool(recon_info["Reorder"]))
    times, f_int, _ = tuned_terms(recon_info, delta_us, ramp_offset_us=ramp_offset_us)
    return calc_radial_traj3d_integral(grad, matrix, float(recon_info["OverSampling"]),
                                       times, f_int, seq.dwell_us)


def o1_steps(recon_info: Dict[str, Any]) -> Optional[np.ndarray]:
    """d_i = O1[i-1] - O1[i] (Hz), or None when the list does not match NPro."""
    o1 = np.asarray(recon_info.get("O1List_Hz") or [], dtype=float)
    if o1.size != int(recon_info["NPro"]) or o1.size < 2:
        return None
    return np.roll(o1, 1) - o1


def delayed_phase_factor(recon_info: Dict[str, Any], delta_us: float,
                         n_points: int) -> Optional[np.ndarray]:
    """(n_pro, n_points) exp(-i phi) of the integral model with shifted sample times."""
    d = o1_steps(recon_info)
    if d is None:
        return None
    _, _, tau = tuned_terms(recon_info, delta_us, n_points)
    tau_s = np.asarray(tau, dtype=float) * 1e-6
    return np.exp(-2j * np.pi * np.outer(d, tau_s)).astype(np.complex64)


# ---------------------------------------------------------------------------
# Delay estimation from the data
# ---------------------------------------------------------------------------


def coherence_vs_delay(z: np.ndarray, recon_info: Dict[str, Any], deltas_us: Sequence[float],
                       samples: Sequence[int]) -> np.ndarray:
    """C[k, m] = coherence of sample samples[m] after the phase of deltas_us[k].

    z: complex (n_pro, >= max(samples)+1), averaged over steady-state volumes.
    """
    d = o1_steps(recon_info)
    if d is None:
        raise ValueError("ACQ_O1_list does not match NPro; the delay cannot be estimated")
    samples = list(samples)
    n_need = max(samples) + 1
    out = np.zeros((len(deltas_us), len(samples)))
    zs = z[:, samples].astype(np.complex128)
    norm = np.maximum(np.abs(zs).sum(axis=0), 1e-300)
    for k, delta in enumerate(deltas_us):
        _, _, tau = tuned_terms(recon_info, float(delta), n_need)
        tau_s = np.asarray(tau, dtype=float)[samples] * 1e-6
        ph = np.exp(-2j * np.pi * np.outer(d, tau_s))
        out[k] = np.abs((zs * ph).sum(axis=0)) / norm
    return out


def estimate_delay(z: np.ndarray, recon_info: Dict[str, Any], samples: Sequence[int],
                   span_us: float = 8.0, step_us: float = 0.1,
                   min_coherence: float = 0.9) -> Dict[str, Any]:
    """delta_hat (us) maximising the mean coherence over the usable ``samples``.

    delta enters the phase as 2 pi d_i delta (1 - f), the same form as any
    part of the object's own spoke phase that follows d_i, so only samples
    where the spokes are nearly in phase are used: a sample is kept when its
    coherence at delta = 0 is >= ``min_coherence`` (at least the first two
    samples are always kept). Grid search in [-span, span] with ``step_us``,
    refined by a parabola through the peak (Lee's ``peakfit.refine_peak``).
    Also returns the best delta of each sample alone and their
    coherence-contrast-weighted mean and spread (``peakfit.weighted_mean_std``):
    a constant delay should explain every kept sample.
    """
    deltas = np.round(np.arange(-span_us, span_us + step_us / 2, step_us), 6)
    c_all = coherence_vs_delay(z, recon_info, deltas, samples)
    i0 = int(np.argmin(np.abs(deltas)))
    keep = [m for m in range(len(samples)) if c_all[i0, m] >= min_coherence]
    if len(keep) < 2:
        keep = [0, 1]
    c = c_all[:, keep]
    samples_all = list(samples)
    samples = [samples_all[m] for m in keep]
    obj = c.mean(axis=1)
    d_hat, obj_hat = peakfit.refine_peak(deltas.tolist(), obj.tolist())
    per = []
    weights = []
    for m, j in enumerate(samples):
        dj, cj = peakfit.refine_peak(deltas.tolist(), c[:, m].tolist())
        contrast = float(c[:, m].max() - c[:, m].min())
        per.append({"sample": int(j), "delta_us": float(dj), "coherence": float(cj),
                    "coherence_at_0": float(c[int(np.argmin(np.abs(deltas))), m]),
                    "contrast": contrast})
        weights.append(contrast)
    interior = [p for p in per if -span_us < p["delta_us"] < span_us]
    wm = ws = None
    if interior and sum(p["contrast"] for p in interior) > 0:
        wm, ws = peakfit.weighted_mean_std([p["delta_us"] for p in interior],
                                           [p["contrast"] for p in interior])
    return {"delta_us": float(d_hat), "objective": float(obj_hat),
            "objective_at_0": float(obj[int(np.argmin(np.abs(deltas)))]),
            "at_edge": bool(abs(d_hat) >= span_us - 1e-9),
            "per_sample": per, "per_sample_weighted_mean_us": wm, "per_sample_weighted_std_us": ws,
            "grid_us": deltas.tolist(), "objective_curve": obj.tolist(),
            "samples": [int(j) for j in samples], "samples_offered": [int(j) for j in samples_all],
            "coherence_at_0_offered": c_all[i0].tolist(), "min_coherence": min_coherence,
            "span_us": span_us, "step_us": step_us}


def _direction_terms(g: np.ndarray, order: int) -> List[np.ndarray]:
    """Regressors for the object's own spoke phase: 1st order g_x, g_y, g_z
    (the centroid term -2 pi |k| g.x_c) and, for order 2, the six products."""
    cols = [g[:, a] for a in range(3)]
    if order >= 2:
        cols += [g[:, a] * g[:, b] for a in range(3) for b in range(a, 3)]
    return cols


def estimate_delay_regression(z: np.ndarray, recon_info: Dict[str, Any], samples: Sequence[int],
                              order: int = 1) -> Dict[str, Any]:
    """delta_hat (us) with the object's spoke phase modelled jointly.

    For each sample j, the spoke phase (relative to the circular mean) is fitted
    by weighted least squares (weights |z_ij|^2):

        arg z_ij = c_j + sum_m a_jm h_m(g_i) + 2 pi d_i tau_obs_j * 1e-6

    with h_m the direction terms of ``_direction_terms`` (order 1: the centroid
    phase of an off-centre object; order 0: no object term, the O1-step term
    alone). delta_j = (tau_obs_j - tau_model_j(0)) / (1 - f_j), f_j the ramp
    fraction at the sample; delta_hat is the inverse-variance weighted mean
    over samples (``peakfit.weighted_mean_std``).
    """
    from brkraw_sordino import timing
    from brkraw_sordino.ramp import ramp_fraction

    d = o1_steps(recon_info)
    if d is None:
        raise ValueError("ACQ_O1_list does not match NPro; the delay cannot be estimated")
    import eval_stages

    g = eval_stages.gradient_vectors(recon_info).T
    g = g / np.linalg.norm(g, axis=1, keepdims=True)
    samples = list(samples)
    times, _, tau0 = tuned_terms(recon_info, 0.0, max(samples) + 1)
    seq = timing.read_timing(recon_info)
    win = timing.ramp_window(seq, timing.tuning_for(seq.version))
    per = []
    for j in samples:
        zj = z[:, j].astype(np.complex128)
        ref = np.angle(zj.sum())
        y = np.angle(zj * np.exp(-1j * ref))
        w = np.abs(zj) ** 2
        cols = [np.ones_like(d)] + (_direction_terms(g, order) if order > 0 else []) + [2 * np.pi * d * 1e-6]
        x = np.stack(cols, axis=1)
        sw = np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(x * sw, y * sw[:, 0], rcond=None)
        res = y - x @ coef
        dof = max(len(y) - x.shape[1], 1)
        # nominal: cov = (sum w r^2 / dof) (X^T W X)^-1; the residual is mostly
        # unmodelled object phase, not noise, so this SE is a lower bound
        s2 = float((w * res ** 2).sum()) / dof
        cov = s2 * np.linalg.pinv((x * sw).T @ (x * sw))
        tau_obs = float(coef[-1])
        tau_se = float(np.sqrt(max(cov[-1, -1], 0.0)))
        f = 0.0 if win is None else ramp_fraction(times[j], win[0], win[1])
        per.append({"sample": int(j), "tau_obs_us": tau_obs, "tau_model_us": float(tau0[j]),
                    "delta_us": (tau_obs - float(tau0[j])) / (1.0 - f), "delta_se_us": tau_se / (1.0 - f),
                    "rms_residual_rad": float(np.sqrt((w * res ** 2).sum() / w.sum()))})
    wts = [1.0 / max(p["delta_se_us"], 1e-9) ** 2 for p in per]
    mean, spread = peakfit.weighted_mean_std([p["delta_us"] for p in per], wts)
    return {"delta_us": float(mean), "spread_us": float(spread),
            "se_us": float(1.0 / np.sqrt(sum(wts))), "order": int(order), "per_sample": per,
            "samples": [int(j) for j in samples]}


def reduce_problem(y: np.ndarray, tr: np.ndarray, ph: Optional[np.ndarray], shape: Sequence[int],
                   factor: int = 1, spoke_stride: int = 1, n_keep: Optional[int] = None,
                   ext: int = 1):
    """Low-k sub-problem for the consistency search: keep every spoke_stride-th
    spoke and the leading samples whose radius fits a grid ``factor`` times
    coarser (|k| <= 0.5 / factor on every spoke; or exactly ``n_keep``
    samples), scale the trajectory by ``factor`` and divide the matrix by it.
    Returns (y, traj, phase, shape)."""
    if factor == 1 and spoke_stride == 1 and n_keep is None:
        return y, tr, ph, list(shape)
    sel = slice(None, None, int(spoke_stride))
    if n_keep is None:
        rad = np.linalg.norm(tr, axis=-1).max(axis=0)
        n_keep = int(np.searchsorted(rad, 0.5 / factor, side="right"))
    if n_keep < 3:
        raise ValueError("reduction leaves fewer than 3 samples per spoke")
    y2 = y[sel, :n_keep]
    t2 = tr[sel, :n_keep] * factor
    p2 = None if ph is None else ph[sel, :n_keep]
    # ext > 1: the image grid covers ext x the FOV at the same voxel size, so
    # signal from outside the nominal FOV can be represented (k units unchanged)
    return y2, t2, p2, [int(s) // int(factor) * int(ext) for s in shape]


def consistency_residual(y: np.ndarray, recon_info: Dict[str, Any], shape: Sequence[int],
                         delta_traj_us: float, delta_phase_us: float, n_iter: int = 20,
                         ignore_samples: int = 1, factor: int = 1, spoke_stride: int = 1,
                         ext: int = 1, cv: bool = False, resid_weight: str = "uniform",
                         ramp_offset_us: float = 0.0, k_scale: float = 1.0,
                         k_margin: float = 0.45) -> float:
    """Relative weighted residual ||W^1/2 (A x - y)|| / ||W^1/2 y|| of the
    least-squares image x (``cg_reconstruct``, n_iter) for a trajectory with
    sample times shifted by delta_traj_us and a phase with delta_phase_us
    (optionally on the ``reduce_problem`` sub-problem).

    The data set and the weight W are fixed by the delta = 0 trajectory (same
    samples, same |k|^2 weights for every candidate), so only the model changes
    between candidates, not the norm the residual is measured in (with noisy
    real data a candidate-dependent W or sample cut biases the comparison).
    cv=True: two-fold cross-validation over spokes (fit on even spokes, residual
    on odd ones and the reverse), so fitted noise does not lower the residual.
    resid_weight: "dcf" measures the residual with W (as the fit), "uniform"
    with equal weights (the high-SNR low-k samples then count as much as the
    noise-dominated outer ones)."""
    n = y.shape[1]   # may be fewer than N samples (leading samples only)
    tr0 = delayed_trajectory(recon_info, 0.0)[:, :n]
    # margin: |k| <= 0.45 / factor at delta 0, so shifted candidates stay inside the grid
    n_keep = int(np.searchsorted(np.linalg.norm(tr0, axis=-1).max(axis=0), k_margin / factor, side="right"))
    _, t0r, _, _ = reduce_problem(y, tr0, None, shape, factor, spoke_stride, n_keep=n_keep)
    w = density_weight(t0r[:, ignore_samples:, ...])
    tr = delayed_trajectory(recon_info, delta_traj_us, ramp_offset_us)[:, :n] * float(k_scale)
    ph = delayed_phase_factor(recon_info, delta_phase_us, n)
    y, tr, ph, shape = reduce_problem(y, tr, ph, shape, factor, spoke_stride, n_keep=n_keep, ext=ext)
    k = y if ph is None else y * ph
    if not cv:
        x, info = cg_reconstruct(y, tr, shape, ignore_samples, ph, n_iter=n_iter, return_info=True, weights=w)
        yy = k[..., ignore_samples:].reshape(-1)
        t2 = tr[:, ignore_samples:, ...]
        pred = _operator(t2, shape).op((x * info["adjoint_scale"]).astype(np.complex64))
        wr = w if resid_weight == "dcf" else np.ones_like(w)
        return float(np.sqrt((wr * np.abs(pred - yy) ** 2).sum() / (wr * np.abs(yy) ** 2).sum()))
    # two-fold cross-validation over spokes: fit on one half, residual on the other
    w2 = w.reshape(tr.shape[0], -1)
    num = den = 0.0
    for fit in (0, 1):
        a = np.arange(tr.shape[0]) % 2 == fit
        b = ~a
        x, info = cg_reconstruct(y[a], tr[a], shape, ignore_samples, None if ph is None else ph[a],
                                 n_iter=n_iter, return_info=True, weights=w2[a])
        t_b = tr[b][:, ignore_samples:, ...]
        yy = k[b][:, ignore_samples:].reshape(-1)
        pred = _operator(t_b, shape).op((x * info["adjoint_scale"]).astype(np.complex64))
        wb = w2[b].reshape(-1) if resid_weight == "dcf" else np.ones(w2[b].size)
        num += float((wb * np.abs(pred - yy) ** 2).sum())
        den += float((wb * np.abs(yy) ** 2).sum())
    return float(np.sqrt(num / den))


def estimate_delay_consistency(y: np.ndarray, recon_info: Dict[str, Any], shape: Sequence[int],
                               deltas_us: Sequence[float], mode: str = "both", n_iter: int = 20,
                               fixed_us: float = 0.0, ignore_samples: int = 1, factor: int = 1,
                               spoke_stride: int = 1, ext: int = 1, cv: bool = False) -> Dict[str, Any]:
    """delta_hat minimising ``consistency_residual`` over a grid (parabola refined).

    mode "both": the same delta in trajectory and phase (a first-sample time
    error); "traj": delta in the trajectory, phase at ``fixed_us``; "phase":
    delta in the phase, trajectory at ``fixed_us``. The object's own phase is
    part of the least-squares image, so it cannot mimic the per-spoke O1-step
    phase or the radial shift.
    """
    if mode not in ("both", "traj", "phase"):
        raise ValueError(mode)
    res = []
    for dl in deltas_us:
        dt = dl if mode in ("both", "traj") else fixed_us
        dp = dl if mode in ("both", "phase") else fixed_us
        res.append(consistency_residual(y, recon_info, shape, dt, dp, n_iter, ignore_samples,
                                        factor, spoke_stride, ext, cv))
    x_hat, neg = peakfit.refine_peak([float(d) for d in deltas_us], [-r for r in res])
    return {"delta_us": float(x_hat), "residual_min": float(-neg), "mode": mode, "n_iter": n_iter,
            "grid_us": [float(d) for d in deltas_us], "residual_curve": res, "fixed_us": fixed_us,
            "factor": factor, "spoke_stride": spoke_stride, "ext": ext, "cv": cv,
            "at_edge": bool(x_hat <= min(deltas_us) or x_hat >= max(deltas_us))}


def estimate_sample_gain(y: np.ndarray, recon_info: Dict[str, Any], shape: Sequence[int],
                         settled_from: int, delta_traj_us: float = 0.0, delta_phase_us: float = 0.0,
                         n_iter: int = 20, factor: int = 1, spoke_stride: int = 1,
                         ext: int = 1) -> Dict[str, Any]:
    """Per-sample complex gain h_j of the leading samples (receiver-filter settling).

    The least-squares image is fitted to the settled samples j >= settled_from
    only; its forward prediction p_ij at every kept sample then gives
    h_j = sum_i conj(p_ij) y_ij / sum_i |p_ij|^2 (one value per sample, common
    to all spokes). h_j = 1 means the sample follows the model; samples
    after settled_from are the control (fitted, so close to 1 by construction).
    """
    n = y.shape[1]
    tr = delayed_trajectory(recon_info, delta_traj_us)[:, :n]
    ph = delayed_phase_factor(recon_info, delta_phase_us, n)
    y2, t2, p2, shape2 = reduce_problem(y, tr, ph, shape, factor, spoke_stride,
                                        n_keep=None if (factor, spoke_stride, ext) == (1, 1, 1) else
                                        int(np.searchsorted(np.linalg.norm(tr, axis=-1).max(axis=0),
                                                            0.5 / factor, side="right")), ext=ext)
    x, info = cg_reconstruct(y2, t2, shape2, settled_from, p2, n_iter=n_iter, return_info=True)
    k = y2 if p2 is None else y2 * p2
    pred = _operator(t2, shape2).op((x * info["adjoint_scale"]).astype(np.complex64)).reshape(t2.shape[:2])
    num = (np.conj(pred) * k).sum(axis=0)
    den = np.maximum((np.abs(pred) ** 2).sum(axis=0), 1e-300)
    h = num / den
    return {"settled_from": int(settled_from), "gain_abs": np.abs(h).tolist(),
            "gain_phase_rad": np.angle(h).tolist(), "n_samples": int(t2.shape[1]),
            "delta_traj_us": delta_traj_us, "delta_phase_us": delta_phase_us}


def estimate_timing(y: np.ndarray, recon_info: Dict[str, Any], shape: Sequence[int],
                    deltas_us: Sequence[float] = tuple(np.arange(-6.0, 6.01, 1.0)), n_iter: int = 20,
                    rounds: int = 2, factor: int = 1, spoke_stride: int = 1,
                    ignore_samples: int = 1, ext: int = 1, cv: bool = False) -> Dict[str, Any]:
    """Alternating 1-D consistency searches: delta_traj with the phase delay
    fixed, then delta_phase with the trajectory delay fixed (start 0, 0).

    delta_traj_us: first-sample time error seen by the trajectory (radial k
    shift); delta_phase_us: the one seen by the O1-step phase. Equal values
    point to a sample-time error; a phase-only value to the phase reference.
    """
    dt = dp = 0.0
    steps = []
    for _ in range(int(rounds)):
        rt = estimate_delay_consistency(y, recon_info, shape, deltas_us, "traj", n_iter, dp,
                                        ignore_samples, factor, spoke_stride, ext, cv)
        dt = rt["delta_us"]
        rp = estimate_delay_consistency(y, recon_info, shape, deltas_us, "phase", n_iter, dt,
                                        ignore_samples, factor, spoke_stride, ext, cv)
        dp = rp["delta_us"]
        steps.append({"traj": rt, "phase": rp})
    r00 = consistency_residual(y, recon_info, shape, 0.0, 0.0, n_iter, ignore_samples, factor, spoke_stride,
                               ext, cv)
    r11 = consistency_residual(y, recon_info, shape, dt, dp, n_iter, ignore_samples, factor, spoke_stride,
                               ext, cv)
    return {"delta_traj_us": float(dt), "delta_phase_us": float(dp), "residual_at_0": r00,
            "residual_at_estimate": r11, "steps": steps, "n_iter": n_iter, "factor": factor,
            "spoke_stride": spoke_stride, "ext": ext, "cv": cv}


def delay_pass(dataset: str, scan_id: int, out_dir: str, samples: Sequence[int] = tuple(range(1, 13)),
               exclude: int = 10, max_volumes: Optional[int] = None, n_keep: int = 32,
               span_us: float = 8.0, step_us: float = 0.1, label: str = "",
               min_coherence: float = 0.9, factor: int = 1, spoke_stride: int = 1,
               cons_grid_us: Sequence[float] = tuple(np.arange(-6.0, 6.01, 1.0)), cons_iter: int = 20,
               cons_rounds: int = 2, ext: int = 4, cv: bool = False) -> Dict[str, Any]:
    """Stream a scan once and estimate the timing two ways.

    1. ``coherence`` (all / even / odd volumes): the run-3 style O1-step
       coherence estimate (``estimate_delay``); biased by the object's own
       spoke phase, kept as an observation.
    2. ``consistency``: ``estimate_timing`` on the mean low-k data of the
       steady-state volumes (reduced by ``factor`` / ``spoke_stride``),
       giving delta_traj_us and delta_phase_us for stages S4p/S4/S4z.

    Saves ``delay.json``, ``delay_mean_first_samples.npy`` and ``delay_estimate.png``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _, recon_info, fid_entry, meta = eval_ramp.open_scan(dataset, scan_id)
    n_pro = int(recon_info["NPro"])
    n_tot = int(recon_info["NRepetitions"])
    n_vol = n_tot if max_volumes is None else min(n_tot, max_volumes)
    ex = min(exclude, max(n_vol - 1, 0))
    n_all = _n_samples(recon_info)
    n_keep = min(n_keep, n_all)
    # samples needed by the reduced consistency problem (+ margin for the delay grid)
    rad0 = np.linalg.norm(delayed_trajectory(recon_info, 0.0), axis=-1).max(axis=0)
    n_cons = min(n_all, int(np.searchsorted(rad0, 0.5 / factor, side="right")) + 6)
    n_acc = max(n_keep, n_cons)
    acc = {k: np.zeros((n_pro, n_acc), dtype=np.complex128) for k in ("all", "even", "odd")}
    cnt = {k: 0 for k in acc}
    mag_prof = np.zeros(n_keep)
    n_read = 0
    for v, vol in eval_ramp.iter_volumes(fid_entry, recon_info, n_vol):
        n_read = v + 1
        if v < ex:
            continue
        z = vol[0, :, :n_acc]
        for k in ("all", "even" if v % 2 == 0 else "odd"):
            acc[k] += z
            cnt[k] += 1
        mag_prof += np.abs(z[:, :n_keep]).mean(axis=0)
    for k in acc:
        acc[k] /= max(cnt[k], 1)
    mag_prof /= max(cnt["all"], 1)
    np.save(out / "delay_mean_first_samples.npy", acc["all"][:, :n_keep].astype(np.complex64))
    t_read = time.time() - t0
    res: Dict[str, Any] = {"n_volumes_read": n_read, "exclude_used": ex, "n_keep": n_keep,
                           "version": eval_ramp.detect_version(meta), "method": meta.get("Method"),
                           "stream_s": round(t_read, 1)}
    res["coherence"] = {}
    for k in acc:
        if cnt[k] == 0:
            res["coherence"][k] = None
            continue
        res["coherence"][k] = estimate_delay(acc[k][:, :n_keep], recon_info, samples, span_us, step_us,
                                             min_coherence)
        res["coherence"][k]["n_volumes"] = cnt[k]
    t1 = time.time()
    shape = [int(x) for x in recon_info["Matrix"]]
    res["consistency"] = estimate_timing(acc["all"][:, :n_cons].astype(np.complex64), recon_info, shape,
                                         cons_grid_us, cons_iter, cons_rounds, factor, spoke_stride, ext=ext,
                                         cv=cv)
    res["consistency"]["n_samples_used"] = n_cons
    res["consistency_s"] = round(time.time() - t1, 1)
    # radius of each sample in k-grid units (integral model, delta 0 and delta_hat)
    matrix = int(recon_info["Matrix"][0])
    os_ = float(recon_info["OverSampling"])
    times0, _, _ = tuned_terms(recon_info, 0.0, n_keep)
    from brkraw_sordino import timing

    dwell = timing.read_timing(recon_info).dwell_us
    res["sample_radius_kgrid"] = [t / dwell / os_ for t in times0]
    res["dwell_us"] = dwell
    res["over_sampling"] = os_
    res["mean_abs_vs_sample"] = mag_prof.tolist()
    res["elapsed_s"] = round(time.time() - t0, 1)
    (out / "delay.json").write_text(json.dumps(res, indent=1, default=str))
    plot_delay(out / "delay_estimate.png", res, f"{label or Path(dataset).name} scan {scan_id}")
    return res


def plot_delay(path: Path, res: Dict[str, Any], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coh = res["coherence"]
    fig, ax = plt.subplots(1, 4, figsize=(24, 4.8))
    for k, ls in (("all", "-"), ("even", "--"), ("odd", ":")):
        r = coh.get(k)
        if r:
            ax[0].plot(r["grid_us"], np.asarray(r["objective_curve"]) - r["objective_at_0"], ls,
                       label=f"{k}: delta = {r['delta_us']:+.2f} us")
    ax[0].axvline(0, color="grey", lw=0.8)
    ax[0].set_xlabel("sample-time shift delta (us)")
    ax[0].set_ylabel("mean coherence(delta) - coherence(0)")
    ax[0].set_title("O1-step coherence (run-3 style, object-biased), samples "
                    + ",".join(str(j) for j in coh["all"]["samples"]), fontsize=9)
    ax[0].legend(fontsize=8)
    per = coh["all"]["per_sample"]
    ax[1].plot([p["sample"] for p in per], [p["delta_us"] for p in per], "o-")
    ax[1].axhline(coh["all"]["delta_us"], color="C3", lw=0.8, label="joint estimate")
    ax[1].set_xlabel("sample index"); ax[1].set_ylabel("best delta of this sample alone (us)")
    ax[1].set_title("O1-step coherence: per-sample delta", fontsize=9); ax[1].legend(fontsize=8)
    cons = res["consistency"]
    last = cons["steps"][-1]
    for key, lab in (("traj", "delta_traj (phase fixed)"), ("phase", "delta_phase (traj fixed)")):
        r = last[key]
        ax[2].plot(r["grid_us"], r["residual_curve"], "o-", label=f"{lab}: {r['delta_us']:+.2f} us")
    ax[2].set_xlabel("delay (us)"); ax[2].set_ylabel("relative weighted residual of LS image")
    ax[2].set_title(f"data consistency (factor {cons['factor']}, stride {cons['spoke_stride']}, "
                    f"{cons['n_iter']} CG it.)", fontsize=9)
    ax[2].legend(fontsize=8)
    ax[3].plot(res["sample_radius_kgrid"], res["mean_abs_vs_sample"], "o-")
    for j in (0, 1, 2):
        ax[3].annotate(str(j), (res["sample_radius_kgrid"][j], res["mean_abs_vs_sample"][j]), fontsize=8)
    ax[3].set_xlabel("sample radius, k-grid units (1/FOV), delta = 0")
    ax[3].set_ylabel("mean |FID| over spokes and volumes")
    ax[3].set_title(f"|FID| vs k radius (OS {res['over_sampling']:g}, dwell {res['dwell_us']:.2f} us)", fontsize=9)
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(path, dpi=110); plt.close(fig)


# ---------------------------------------------------------------------------
# Algebraic (least-squares) reconstruction for the dead-time gap
# ---------------------------------------------------------------------------


def density_weight(traj: np.ndarray) -> np.ndarray:
    """The product's density weight |k|^2 / max (recon.nufft_adjoint)."""
    w = np.square(traj).sum(-1).reshape(-1)
    return w / w.max()


def _operator(traj: np.ndarray, shape: Sequence[int]):
    """finufft operator at k positions ``traj`` (|k| = 0.5 is Nyquist).

    mrinufft (``proper_trajectory(normalize="pi")``, 1.5.1) multiplies a
    trajectory by 2 pi when its largest |omega| is below 0.5 rad, assuming it
    was given in [-0.5, 0.5). The radians passed here are below 0.5 rad
    whenever all points lie within about 5 k-grid units of the centre (the
    virtual leading samples, the gap points, a probe of the first samples),
    so such an operator silently evaluated k at 2 pi times the radius
    (WI-0058 run 2). The samples are therefore set again, unchanged, through
    ``update_samples(unsafe=True)``; full spokes (|omega| up to pi) are not
    affected either way.
    """
    import warnings

    from mrinufft import get_operator

    omega = np.asarray(traj.reshape(-1, 3) / 0.5 * np.pi, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Samples will be rescaled")
        op = get_operator("finufft")(omega, shape=tuple(int(s) for s in shape), density=False)
    op.update_samples(np.asarray(omega, order="F"), unsafe=True)
    return op


def cg_reconstruct(kspace: np.ndarray, traj: np.ndarray, shape: Sequence[int], ignore_samples: int = 1,
                   phase_factor: Optional[np.ndarray] = None, n_iter: int = 10,
                   return_info: bool = False, weights: Optional[np.ndarray] = None, ext: int = 1):
    """Weighted least squares by conjugate gradient on A^H W A x = A^H W y.

    Same data handling as ``eval_ramp.reconstruct`` (phase factor, samples
    dropped). The result is scaled by 1/c with c = <A x_adj, y>_W / ||A x_adj||_W^2
    of the adjoint image x_adj = A^H W y, i.e. expressed in the adjoint's units.
    ext > 1 solves on a grid covering ext x the FOV at the same voxel size
    (radial ZTE also receives signal from outside the nominal FOV, which a 1x
    grid cannot represent) and returns the central FOV.
    """
    k = kspace if phase_factor is None else kspace * phase_factor
    y = k[..., ignore_samples:].reshape(-1).astype(np.complex128)
    tr = traj[:, ignore_samples:, ...]
    w = density_weight(tr) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    out_shape = tuple(int(s) for s in shape)
    shape = tuple(int(s) * int(ext) for s in out_shape)
    op = _operator(tr, shape)

    def normal(x):
        return op.adj_op((w * op.op(x.astype(np.complex64))).astype(np.complex64)).reshape(x.shape)

    b = op.adj_op((w * y).astype(np.complex64)).reshape(tuple(int(s) for s in shape)).astype(np.complex128)
    x_adj = b.copy()
    ax_adj = op.op(x_adj.astype(np.complex64)).astype(np.complex128)
    c = np.vdot(ax_adj, w * y) / max(np.vdot(ax_adj, w * ax_adj).real, 1e-300)
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs = np.vdot(r, r).real
    hist = [float(np.sqrt(rs))]
    for _ in range(int(n_iter)):
        ap = normal(p).astype(np.complex128)
        alpha = rs / max(np.vdot(p, ap).real, 1e-300)
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = np.vdot(r, r).real
        hist.append(float(np.sqrt(rs_new)))
        if rs_new <= 1e-24 * hist[0] ** 2:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    x_units = x / c
    if int(ext) > 1:
        lo = [(s - o) // 2 for s, o in zip(shape, out_shape)]
        x_units = x_units[tuple(slice(a, a + o) for a, o in zip(lo, out_shape))]
    if return_info:
        # x_full: the whole (ext) grid in data units, for forward predictions
        return x_units, {"adjoint_scale": complex(c), "residual_norms": hist, "x_full": x,
                         "grid_shape": list(shape)}
    return x_units


def leading_points(recon_info: Dict[str, Any], delta_traj_us: float = 0.0,
                   ignore_samples: int = 1) -> np.ndarray:
    """(n_pro, M, 3) k positions of virtual leading samples: times
    t_first - m * dwell (m = 1..M, t >= 0) before the first kept sample, on each
    spoke's own integral-model trajectory, i.e. the dropped samples and the dead
    time at the real sample spacing, down to the RF centre (k = 0)."""
    from brkraw_sordino import timing

    seq = timing.read_timing(recon_info)
    times, _, _ = tuned_terms(recon_info, delta_traj_us, ignore_samples + 1)
    t_first = times[ignore_samples]
    ts = []
    m = 1
    while t_first - m * seq.dwell_us >= 0:
        ts.append(t_first - m * seq.dwell_us)
        m += 1
    if not ts:
        raise ValueError("no room for virtual samples before the first kept sample")
    return points_at_times(recon_info, delta_traj_us, sorted(ts))


def points_at_times(recon_info: Dict[str, Any], delta_traj_us: float, ts: Sequence[float]) -> np.ndarray:
    """(n_pro, len(ts), 3) integral-model k positions at times ts (us, RF centre = 0)."""
    from brkraw_sordino import timing
    from brkraw_sordino.ramp import ramp_integral
    from brkraw_sordino.traj import calc_radial_grad3d

    seq = timing.read_timing(recon_info)
    base = timing.tuning_for(seq.version)
    tune = replace(base, acq_start_offset_us=base.acq_start_offset_us + float(delta_traj_us))
    win = timing.ramp_window(seq, tune)
    fs = list(ts) if win is None else [ramp_integral(t, win[0], win[1]) for t in ts]
    n = _n_samples(recon_info)
    unit = 1.0 / (n - 1) / 2.0
    g = calc_radial_grad3d(int(recon_info["Matrix"][0]), int(recon_info["NPro"]),
                           bool(recon_info["HalfAcquisition"]), bool(recon_info["UseOrigin"]),
                           bool(recon_info["Reorder"]))
    g_prev = np.roll(g, 1, axis=1).T
    delta = g.T - g_prev
    t = np.asarray(ts, dtype=float) / seq.dwell_us
    f = np.asarray(fs, dtype=float) / seq.dwell_us
    return unit * (t[None, :, None] * g_prev[:, None, :] + f[None, :, None] * delta[:, None, :])


def centre_fill_reconstruct(kspace: np.ndarray, traj: np.ndarray, shape: Sequence[int],
                            ignore_samples: int, phase_factor: Optional[np.ndarray],
                            virtual_traj: np.ndarray, n_iter: int = 10, ext: int = 2,
                            return_info: bool = False):
    """S4z: the product's adjoint over measured + estimated leading samples.

    1. least-squares image on an ext x FOV grid (``cg_reconstruct``);
    2. its forward prediction at ``virtual_traj`` (``leading_points``) gives the
       unsampled centre (dead time and dropped samples), in data units;
    3. adjoint NUFFT with the product's |k|^2 weight over both sample sets.
    Only the centre comes from the iterative solution (it converges in a few
    iterations); every measured sample enters as in the product adjoint.
    """
    from brkraw_sordino.recon import nufft_adjoint

    _, info = cg_reconstruct(kspace, traj, shape, ignore_samples, phase_factor, n_iter=n_iter,
                             return_info=True, ext=ext)
    k = kspace if phase_factor is None else kspace * phase_factor
    pred = _operator(virtual_traj, info["grid_shape"]).op(info["x_full"].astype(np.complex64))
    pred = pred.reshape(virtual_traj.shape[:2])
    data = np.concatenate([pred, k[..., ignore_samples:]], axis=1)
    tr = np.concatenate([virtual_traj, traj[:, ignore_samples:, ...]], axis=1)
    img = nufft_adjoint(data, tr, shape, 1)
    if return_info:
        return img, {"virtual_samples": int(virtual_traj.shape[1]), "cg_residual_norms": info["residual_norms"],
                     "virtual_values": pred}
    return img


# ---------------------------------------------------------------------------
# S3c: first samples restored from the FID curve (WI-0058 run 2)
# ---------------------------------------------------------------------------

#: Fit window of the curve (WI-0058 run 2). The receiver-filter settling is a
#: time-domain effect counted in samples (sample 0 about 0.55x, sample 1 about
#: 1.2x the smooth curve on every scan; from sample 2 on within about 5 %), so
#: the fit starts at sample CURVE_FIRST_SAMPLE; the object's k-space profile is
#: a k-space effect, so it ends at CURVE_MAX_KGRID k-grid units (spoke-mean
#: radius; beyond about 2 units the profile nears its first zero and the phase
#: jumps), extended by index to at least CURVE_MIN_SAMPLES samples.
CURVE_FIRST_SAMPLE: int = 2
CURVE_MAX_KGRID: float = 1.6
CURVE_MIN_SAMPLES: int = 3


def leading_times(recon_info: Dict[str, Any], ignore_samples: int = 1) -> np.ndarray:
    """Times (us, RF centre = 0) of ``leading_points``: t_first - m * dwell >= 0, increasing."""
    from brkraw_sordino import timing

    seq = timing.read_timing(recon_info)
    times, _, _ = tuned_terms(recon_info, 0.0, ignore_samples + 1)
    t_first = times[ignore_samples]
    ts = []
    m = 1
    while t_first - m * seq.dwell_us >= 0:
        ts.append(t_first - m * seq.dwell_us)
        m += 1
    return np.asarray(sorted(ts), dtype=float)


def curve_window(traj: np.ndarray, matrix: int, first: int = CURVE_FIRST_SAMPLE,
                 max_kgrid: float = CURVE_MAX_KGRID, min_samples: int = CURVE_MIN_SAMPLES) -> List[int]:
    """Fit samples first, first + 1, ... while the spoke-mean radius is <= max_kgrid
    (k-grid units), at least min_samples of them."""
    r = np.linalg.norm(traj, axis=-1).mean(axis=0) * int(matrix)
    n = traj.shape[1]
    last = int(first)
    while last + 1 < n and r[last + 1] <= max_kgrid:
        last += 1
    last = max(last, int(first) + int(min_samples) - 1)
    if last >= n:
        raise ValueError(f"curve window from sample {first} needs {min_samples} samples; only {n - first}")
    return list(range(int(first), last + 1))


def curve_restore(z: np.ndarray, traj: np.ndarray, times_us: np.ndarray, fit_cols: Sequence[int],
                  new_traj: np.ndarray, new_times_us: np.ndarray, matrix: int,
                  model: str = "k2") -> np.ndarray:
    """Per-spoke FID-curve values at new positions (Kuethe 1999 style, 1-D per spoke).

    z: (n_pro, N) complex FID (after the phase factor); traj: (n_pro, N, 3);
    times_us: (N,) sample times; new_traj (n_pro, M, 3) and new_times_us (M,)
    the positions to fill. Magnitude: log|z| fitted over fit_cols, linear in
    x = |k|^2 (model "k2", k in k-grid units: a Gaussian, flat at k = 0) or
    in x = t (model "t": a straight line on a log scale in time). Phase:
    unwrapped phase linear in t over the same samples. Uses curvefit (Lee
    Minjun, wi-0058-lee-2). Returns (n_pro, M) complex."""
    if model not in ("k2", "t"):
        raise ValueError(model)
    cols = list(fit_cols)
    tf = [float(times_us[j]) for j in cols]
    tn = [float(t) for t in new_times_us]
    if model == "k2":
        xf_all = np.square(traj[:, cols, :] * matrix).sum(-1)
        xn_all = np.square(new_traj * matrix).sum(-1)
    out = np.zeros((z.shape[0], len(tn)), dtype=np.complex128)
    for i in range(z.shape[0]):
        zi = z[i, cols]
        xf = xf_all[i].tolist() if model == "k2" else tf
        xn = xn_all[i].tolist() if model == "k2" else tn
        mag = curvefit.extrapolate_log_magnitude(xf, np.abs(zi).tolist(), xn, degree=1)
        ph = curvefit.extrapolate_phase(tf, np.angle(zi).tolist(), tn, degree=1)
        out[i] = np.asarray(mag) * np.exp(1j * np.asarray(ph))
    return out


def curve_fill_data(kspace: np.ndarray, traj: np.ndarray, recon_info: Dict[str, Any],
                    ignore_samples: int, phase_factor: Optional[np.ndarray], model: str = "k2",
                    first: int = CURVE_FIRST_SAMPLE, max_kgrid: float = CURVE_MAX_KGRID):
    """S3c data: the FID curve at the dropped samples and the dead time
    (``leading_points``, down to the RF centre) and at the kept samples before
    the fit window (from ignore_samples up to first - 1: receiver-filter
    settling); measured values from the fit window on.
    Returns (data (n_pro, M + N - first), traj (n_pro, M + N - first, 3), info)."""
    matrix = int(recon_info["Matrix"][0])
    k = kspace if phase_factor is None else kspace * phase_factor
    n = k.shape[1]
    times, _, _ = tuned_terms(recon_info, 0.0, n)
    times = np.asarray(times, dtype=float)
    if first < ignore_samples:
        raise ValueError("the fit window cannot start at a dropped sample")
    cols = curve_window(traj, matrix, first, max_kgrid)
    vtraj = leading_points(recon_info, 0.0, ignore_samples)
    vt = leading_times(recon_info, ignore_samples)
    rep = list(range(int(ignore_samples), int(first)))
    new_traj = np.concatenate([vtraj, traj[:, rep, :]], axis=1)
    new_t = np.concatenate([vt, times[rep]])
    est = curve_restore(k, traj, times, cols, new_traj, new_t, matrix, model)
    data = np.concatenate([est, k[:, int(first):]], axis=1)
    tr = np.concatenate([new_traj, traj[:, int(first):, :]], axis=1)
    r = np.linalg.norm(traj, axis=-1).mean(axis=0) * matrix
    info = {"fit_cols": cols, "fit_radius_kgrid": [float(r[cols[0]]), float(r[cols[-1]])],
            "virtual_samples": int(vtraj.shape[1]), "replaced_kept": rep, "estimated": est,
            "model": model, "first_sample": int(first), "max_kgrid": float(max_kgrid)}
    return data, tr, info


def curve_fill_reconstruct(kspace: np.ndarray, traj: np.ndarray, shape: Sequence[int],
                           recon_info: Dict[str, Any], ignore_samples: int,
                           phase_factor: Optional[np.ndarray], model: str = "k2",
                           return_info: bool = False):
    """S3c: the product's adjoint (|k|^2 weight) over ``curve_fill_data``."""
    from brkraw_sordino.recon import nufft_adjoint

    data, tr, info = curve_fill_data(kspace, traj, recon_info, ignore_samples, phase_factor, model)
    img = nufft_adjoint(data, tr, shape, 1)
    return (img, info) if return_info else img


def centre_points(recon_info: Dict[str, Any], delta_us: float,
                  fractions: Sequence[float] = (0.0, 0.25, 0.5, 0.75)) -> np.ndarray:
    """(n_pro, len(fractions), 3) k positions inside the gap: times
    fraction * t(first kept sample) on each spoke's own model trajectory."""
    from brkraw_sordino import timing
    from brkraw_sordino.ramp import ramp_integral
    from brkraw_sordino.traj import calc_radial_grad3d

    seq = timing.read_timing(recon_info)
    base = timing.tuning_for(seq.version)
    tune = replace(base, acq_start_offset_us=base.acq_start_offset_us + float(delta_us))
    t1 = timing.sample_times_us(seq, tune, 2)[1]
    win = timing.ramp_window(seq, tune)
    ts = [f * t1 for f in fractions]
    fs = ts if win is None else [ramp_integral(t, win[0], win[1]) for t in ts]
    n = _n_samples(recon_info)
    unit = 1.0 / (n - 1) / 2.0
    g = calc_radial_grad3d(int(recon_info["Matrix"][0]), int(recon_info["NPro"]),
                           bool(recon_info["HalfAcquisition"]), bool(recon_info["UseOrigin"]),
                           bool(recon_info["Reorder"]))
    g_prev = np.roll(g, 1, axis=1).T
    delta = g.T - g_prev
    t = np.asarray(ts) / seq.dwell_us
    f = np.asarray(fs) / seq.dwell_us
    return unit * (t[None, :, None] * g_prev[:, None, :] + f[None, :, None] * delta[:, None, :])


# ---------------------------------------------------------------------------
# Simulation with known delay and known centre
# ---------------------------------------------------------------------------


def smooth_phantom(shape: Sequence[int]) -> np.ndarray:
    """Brain-like test object: an off-centre ellipsoid (semi-axes 0.32, 0.27,
    0.22 of the matrix) with an inner ellipsoid at half intensity and two
    small bright spheres, edges smoothed over about one voxel (real image)."""
    n = np.asarray(shape, dtype=float)
    grid = np.meshgrid(*[np.arange(int(s)) - s / 2 for s in shape], indexing="ij")

    def ellipsoid(centre, axes):
        r = np.sqrt(sum(((g - c * s) / (a * s)) ** 2 for g, c, a, s in zip(grid, centre, axes, n)))
        return 0.5 * (1 - np.tanh((r - 1.0) * float(min(axes) * n.min()) / 1.0))

    img = ellipsoid((0.04, -0.03, 0.02), (0.32, 0.27, 0.22))
    img = img + 0.5 * ellipsoid((0.02, 0.0, 0.0), (0.14, 0.12, 0.1))
    img = img + 1.5 * ellipsoid((-0.12, 0.1, 0.05), (0.04, 0.04, 0.04))
    img = img + 1.5 * ellipsoid((0.15, -0.08, -0.06), (0.05, 0.05, 0.05))
    return img.astype(np.complex64)


def simulate(recon_info: Dict[str, Any], true_traj_us: float, true_phase_us: Optional[float] = None,
             cg_iters: Sequence[int] = (5, 10, 20), noise_rel: float = 0.0,
             samples: Sequence[int] = tuple(range(1, 13)), seed: int = 0, obj: str = "smooth",
             min_coherence: float = 0.9, cons_grid_us: Sequence[float] = tuple(np.arange(-6.0, 6.01, 1.0)),
             cons_iter: int = 20, factor: int = 1, spoke_stride: int = 1, cv: bool = False) -> Dict[str, Any]:
    """Model data with known timing errors, then S3 / S4p / S4 / S4z.

    Data: ``obj`` = "smooth" (``smooth_phantom``, brain-like, slightly off
    the FOV centre) or "points" (``eval_stages.phantom``, four bright points
    and a cube), forward NUFFT on the integral trajectory with sample times
    shifted by true_traj_us and the integral-model phase with true_phase_us
    (default: the same); complex Gaussian noise with std = noise_rel * max|y|
    if noise_rel > 0. The phantom is real: its spoke phase is its own
    (position) phase plus the model phase.
    Estimates: the O1-step coherence (``estimate_delay``, for comparison)
    and the data-consistency pair (``estimate_timing``) that S4p/S4/S4z use.
    Per stage: the error of the magnitude image against the phantom after a
    least-squares scale (``rel_error_vs_phantom``), and the error of the gap
    k-space values predicted by each image (``gap_rel_error``; truth = forward
    model of the phantom at ``centre_points`` of the true timing).
    """
    import eval_stages

    if true_phase_us is None:
        true_phase_us = true_traj_us
    shape = [int(x) for x in recon_info["Matrix"]]
    img = smooth_phantom(shape) if obj == "smooth" else eval_stages.phantom(shape)[0]
    n = _n_samples(recon_info)
    tr_true = delayed_trajectory(recon_info, true_traj_us)
    ph_true = delayed_phase_factor(recon_info, true_phase_us, n)
    y = _operator(tr_true, shape).op(img).reshape(tr_true.shape[:2])
    if ph_true is not None:
        y = y / ph_true
    if noise_rel > 0:
        rng = np.random.default_rng(seed)
        s = noise_rel * float(np.abs(y).max())
        y = y + s * (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape)) / np.sqrt(2)
    est_coh = estimate_delay(y[:, :max(samples) + 1], recon_info, samples, min_coherence=min_coherence)
    est = estimate_timing(y.astype(np.complex64), recon_info, shape, cons_grid_us, cons_iter, 2,
                          factor, spoke_stride, cv=cv)
    d_t, d_p = est["delta_traj_us"], est["delta_phase_us"]
    cp_true = centre_points(recon_info, true_traj_us)
    gap_truth = _operator(cp_true, shape).op(img)
    op_gap = _operator(cp_true, shape)

    def image_error(rec_complex) -> float:
        """|rec| against |phantom| after the least-squares scale of |rec|."""
        mag = np.abs(rec_complex)
        a = float((mag * np.abs(img)).sum() / max((mag * mag).sum(), 1e-300))
        return float(np.linalg.norm(a * mag - np.abs(img)) / np.linalg.norm(np.abs(img)))

    def gap_error(rec_data_units) -> float:
        """Gap k-space values predicted by an image in data units vs truth."""
        pred = op_gap.op(rec_data_units.astype(np.complex64))
        return float(np.linalg.norm(pred - gap_truth) / np.linalg.norm(gap_truth))

    tr0 = delayed_trajectory(recon_info, 0.0)
    trd = delayed_trajectory(recon_info, d_t)
    ph0 = delayed_phase_factor(recon_info, 0.0, n)
    phd = delayed_phase_factor(recon_info, d_p, n)
    stages = {"S3": (tr0, ph0), "S4p": (tr0, phd), "S4": (trd, phd), "true_timing": (tr_true, ph_true)}
    res: Dict[str, Any] = {"true_traj_us": true_traj_us, "true_phase_us": true_phase_us, "obj": obj,
                           "estimated_traj_us": d_t, "estimated_phase_us": d_p,
                           "coherence_estimate_us": est_coh["delta_us"],
                           "consistency": {k: v for k, v in est.items() if k != "steps"},
                           "noise_rel": noise_rel, "stages": {}, "images": {"phantom": np.abs(img)}}
    # the adjoint with the true timing: the reference for the timing stages (an
    # adjoint keeps the gap, so its distance to the phantom mixes the gap and
    # the timing; the distance to this reference is the timing error alone)
    ref_adj = np.abs(eval_ramp.reconstruct(y, tr_true, shape, 1, ph_true))

    def vs_ref(mag) -> float:
        a = float((mag * ref_adj).sum() / max((mag * mag).sum(), 1e-300))
        return float(np.linalg.norm(a * mag - ref_adj) / np.linalg.norm(ref_adj))

    for name, (tr, ph) in stages.items():
        rec_adj = eval_ramp.reconstruct(y, tr, shape, 1, ph)
        # adjoint units -> data units by the least-squares scale (same rule as cg_reconstruct)
        _, info = cg_reconstruct(y, tr, shape, 1, ph, n_iter=0, return_info=True)
        res["stages"][name] = {"rel_error_vs_phantom": image_error(rec_adj),
                               "rel_error_vs_true_timing_adjoint": vs_ref(np.abs(rec_adj)),
                               "gap_rel_error": gap_error(rec_adj * info["adjoint_scale"])}
        res["images"][name] = np.abs(rec_adj)
    for it in cg_iters:
        x, info = cg_reconstruct(y, trd, shape, 1, phd, n_iter=it, return_info=True)
        res["stages"][f"S4z_cg{it}"] = {"rel_error_vs_phantom": image_error(x),
                                         "rel_error_vs_true_timing_adjoint": vs_ref(np.abs(x)),
                                         "gap_rel_error": gap_error(x * info["adjoint_scale"]),
                                         "residual_norms": info["residual_norms"]}
        res["images"][f"S4z_cg{it}"] = np.abs(x)
    # S4z as used on real data: adjoint over measured + estimated leading samples.
    # Reference without a gap: the same adjoint with the TRUE values at the true
    # leading-sample positions (forward model of the phantom there).
    from brkraw_sordino.recon import nufft_adjoint

    virt_true = leading_points(recon_info, true_traj_us, 1)
    yv_true = _operator(virt_true, shape).op(img).reshape(virt_true.shape[:2])
    k_true = y * ph_true if ph_true is not None else y
    nogap = np.abs(nufft_adjoint(np.concatenate([yv_true, k_true[:, 1:]], axis=1),
                                 np.concatenate([virt_true, tr_true[:, 1:]], axis=1), shape, 1))
    res["images"]["no_gap_reference"] = nogap

    def vs_nogap(mag) -> float:
        a = float((mag * nogap).sum() / max((mag * mag).sum(), 1e-300))
        return float(np.linalg.norm(a * mag - nogap) / np.linalg.norm(nogap))

    virt = leading_points(recon_info, d_t, 1)
    for it in cg_iters:
        fill, finfo = centre_fill_reconstruct(y, trd, shape, 1, phd, virt, n_iter=it, ext=1, return_info=True)
        res["stages"][f"S4z_fill_cg{it}"] = {"rel_error_vs_phantom": image_error(fill),
                                              "rel_error_vs_no_gap_reference": vs_nogap(np.abs(fill)),
                                              "virtual_samples": finfo["virtual_samples"]}
        res["images"][f"S4z_fill_cg{it}"] = np.abs(fill)
    for name in ("S3", "S4", "true_timing"):
        res["stages"][name]["rel_error_vs_no_gap_reference"] = vs_nogap(res["images"][name])
    res["stages"]["no_gap_reference"] = {"rel_error_vs_phantom": image_error(nogap)}
    return res


__all__ = ["tuned_terms", "delayed_trajectory", "delayed_phase_factor", "o1_steps",
           "coherence_vs_delay", "estimate_delay", "estimate_delay_regression", "reduce_problem",
           "consistency_residual", "estimate_delay_consistency", "estimate_timing", "delay_pass",
           "plot_delay", "density_weight", "cg_reconstruct", "centre_points", "smooth_phantom", "simulate",
           "CURVE_FIRST_SAMPLE", "CURVE_MAX_KGRID", "CURVE_MIN_SAMPLES", "leading_times", "curve_window", "curve_restore", "curve_fill_data",
           "curve_fill_reconstruct"]
