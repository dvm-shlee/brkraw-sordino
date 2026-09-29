"""Staged comparison of the SORDINO ramp-time corrections (WI-0056 run 5).

Runs every evaluation metric of ``eval_ramp`` for a fixed set of correction
stages side by side, so that a change can be read stage by stage:

    stage   trajectory model            FID phase          product options
    S0      no ramp (constant vector)   none               correct_ramptime=False
    S1      legacy (curvature x2)       none               ramp_model="legacy"
    S1+p    legacy                      implied by S1      (tool only)
    S1h     legacy, curvature halved    none               (tool only)
    S1h+p   legacy halved               implied by S1h     (tool only)
    S2      gradient integral           none               ramp_model="integral", correct_phase=False
    S3      gradient integral           integral model     ramp_model="integral", correct_phase=True

One model G(t) defines both the trajectory and the phase of a stage: with
k(t) = c * integral(G) and the receiver frequency set for the target vector
g(i), the FID of projection i carries the phase 2*pi*(O1[i-1] - O1[i]) *
tau_j with tau_j = t_j - F_j, where F_j is the ramp integral of that model.
For S0 F_j = t_j, so tau_j = 0 and "phase on" does not apply. "S1+p" and
"S1h+p" use the phase implied by the legacy trajectory (tau_j = t_j (1 -
c j/N), c = 1 or 1/2) and exist only in this tool; the product never had a
phase term for the legacy model. S1h ("curvature halved", the simple /2 fix)
is also tool-only.

What can change at which stage (by principle):

    metric                                   depends on
    (a) |first peak| per spoke, CV           nothing (raw FID magnitude)
    (a) first-peak phase, re, im, circular   phase model only
        std, coherence
    first-peak k position                    trajectory model only
    (b) |first peak| pattern stability,      nothing (raw FID magnitude)
        time course, CV over volumes
    (b) coherence time course                phase model only
    (c) recon oscillation, sharpness,        trajectory and phase
        image difference, orthogonal views
    simulation error                         trajectory and phase

This file is a development tool: it is not part of the installed package.
Product code is not changed by it; S1h and the implied legacy phase are
computed here from the product's own gradient vectors and timing.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import circstats
    import eval_ramp
    import evalstats
    import stagetable
except ImportError:  # pragma: no cover - run from another folder
    from tools import circstats, eval_ramp, evalstats, stagetable  # type: ignore


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

TRAJ_MODELS = ("off", "legacy", "legacy_half", "integral", "integral_delay")
PHASE_MODELS = ("none", "legacy", "legacy_half", "integral")
#: WI-0058: the integral phase with the self-calibrated sample-time shift.
DELAY_PHASE_MODELS = ("integral_delay",)


@dataclass(frozen=True)
class Stage:
    name: str
    traj: str
    phase: str
    label: str
    recon: str = "adjoint"   # "adjoint" (product) or "fill" (WI-0058 S4z: adjoint + estimated centre)


STAGES: Tuple[Stage, ...] = (
    Stage("S0", "off", "none", "S0 no ramp"),
    Stage("S1", "legacy", "none", "S1 legacy (2x curvature)"),
    Stage("S1p", "legacy", "legacy", "S1+phase"),
    Stage("S1h", "legacy_half", "none", "S1h legacy /2"),
    Stage("S1hp", "legacy_half", "legacy_half", "S1h+phase"),
    Stage("S2", "integral", "none", "S2 integral traj"),
    Stage("S3", "integral", "integral", "S3 integral + phase"),
)
STAGE_NAMES = tuple(s.name for s in STAGES)

#: WI-0058 stages (tool only; ``eval_timing_centre``). They need
#: ``StageConfig.delay_traj_us`` / ``delay_phase_us`` (the per-scan estimates
#: of ``eval_timing_centre.delay_pass``).
STAGES_WI0058: Tuple[Stage, ...] = (
    Stage("S4p", "integral", "integral_delay", "S4p S3 + phase delay"),
    Stage("S4", "integral_delay", "integral_delay", "S4 S3 + traj and phase delays"),
    Stage("S4z", "integral_delay", "integral_delay", "S4z S4 + estimated centre", "fill"),
    Stage("S3z", "integral", "integral", "S3z S3 + estimated centre", "fill"),
)
ALL_STAGE_NAMES = STAGE_NAMES + tuple(s.name for s in STAGES_WI0058)


def stage(name: str) -> Stage:
    for s in STAGES + STAGES_WI0058:
        if s.name == name:
            return s
    raise ValueError(f"unknown stage {name!r}; known: {ALL_STAGE_NAMES}")


# ---------------------------------------------------------------------------
# Trajectory and phase builders
# ---------------------------------------------------------------------------


def _n_samples(recon_info: Dict[str, Any]) -> int:
    return int(int(recon_info["Matrix"][0]) / 2 * float(recon_info["OverSampling"]))


def _offset_samples(recon_info: Dict[str, Any]) -> float:
    """AcqDelayTotal in samples (the product's ``offset_factor``)."""
    return (float(recon_info["AcqDelayTotal_us"]) * 1e-6
            * float(recon_info["EffBandwidth_Hz"]) * float(recon_info["OverSampling"]))


def gradient_vectors(recon_info: Dict[str, Any]) -> np.ndarray:
    """(3, n_pro) unit vectors exactly as the product computes them."""
    from brkraw_sordino.traj import calc_radial_grad3d

    return calc_radial_grad3d(int(recon_info["Matrix"][0]), int(recon_info["NPro"]),
                              bool(recon_info["HalfAcquisition"]), bool(recon_info["UseOrigin"]),
                              bool(recon_info["Reorder"]))


def legacy_trajectory(recon_info: Dict[str, Any], curvature: float = 1.0) -> np.ndarray:
    """The pre-WI-0056 trajectory with a scaled curvature term (tool only).

    Product form (traj.py ``calc_radial_traj3d``, correct_ramptime=True):
        k_ij = s_j * (g(i-1) + (g(i) - g(i-1)) * j / N),  s_j = (j + o) / (2 (N-1))
    with o = AcqDelayTotal in samples; the last projection gets no
    correction (k = s_j g(i)). ``curvature`` scales the j/N term: 1.0 is the
    product's own form, 0.5 is stage S1h ("curvature halved").
    """
    if bool(recon_info.get("UseOrigin")):
        raise ValueError("legacy_trajectory is defined for UseOrigin=False scans only")
    g = gradient_vectors(recon_info)
    n = _n_samples(recon_info)
    j = np.arange(n, dtype=float)
    s = ((j + _offset_samples(recon_info)) / (n - 1)) / 2.0
    g_cur = g.T                                    # (n_pro, 3)
    g_prev = np.roll(g_cur, 1, axis=0)
    frac = curvature * j / n                       # (n,)
    traj = s[None, :, None] * (g_prev[:, None, :] + frac[None, :, None] * (g_cur - g_prev)[:, None, :])
    traj[-1] = s[:, None] * g_cur[-1][None, :]     # legacy: last projection uncorrected
    return traj


def stage_trajectory(recon_info: Dict[str, Any], traj_model: str, cache_dir: Path,
                     delay_us: float = 0.0) -> np.ndarray:
    """(n_pro, N, 3) trajectory of one ``TRAJ_MODELS`` entry (``delay_us``
    is used by ``integral_delay`` only)."""
    if traj_model == "integral_delay":
        import eval_timing_centre

        return eval_timing_centre.delayed_trajectory(recon_info, delay_us)
    if traj_model == "off":
        return eval_ramp.trajectory(recon_info, "off", cache_dir)[0]
    if traj_model == "legacy":
        return eval_ramp.trajectory(recon_info, "pre", cache_dir)[0]
    if traj_model == "legacy_half":
        return legacy_trajectory(recon_info, 0.5)
    if traj_model == "integral":
        return eval_ramp.trajectory(recon_info, "post_traj", cache_dir)[0]
    raise ValueError(f"unknown trajectory model {traj_model!r}")


def implied_tau_us(recon_info: Dict[str, Any], phase_model: str, n_points: int) -> Optional[np.ndarray]:
    """tau_j (us, length n_points) of the phase model, or None for ``none``.

    legacy / legacy_half: tau_j = t_j * (1 - c * j / N) with c = 1 / 0.5, the
    phase implied by the legacy trajectory (its F_j = c t_j j / N).
    integral: the product's ``timing.ramp_terms`` (t_j - F_j).
    t_j = AcqDelayTotal + j * dwell from the RF centre (timing.py).
    """
    from brkraw_sordino import timing

    if phase_model == "none":
        return None
    seq = timing.read_timing(recon_info)
    tune = timing.tuning_for(seq.version)
    times, _, tau = timing.ramp_terms(seq, tune, n_points)
    if phase_model == "integral":
        return np.asarray(tau, dtype=float)
    c = {"legacy": 1.0, "legacy_half": 0.5}[phase_model]
    n = float(_n_samples(recon_info))
    j = np.arange(n_points, dtype=float)
    return np.asarray(times, dtype=float) * (1.0 - c * j / n)


def stage_phase_factor(recon_info: Dict[str, Any], phase_model: str, n_points: int,
                       cache_dir: Path, delay_us: float = 0.0) -> Optional[np.ndarray]:
    """(n_pro, n_points) complex factor exp(-i phi) of one ``PHASE_MODELS`` entry.

    ``integral`` is taken from the product (``recon.phase_correction_factor``);
    ``legacy``/``legacy_half`` are built here from ``implied_tau_us`` with the
    same O1 step d_i = O1[i-1] - O1[i] and sign; the last projection has no
    ramp in the legacy model, so its factor is 1. ``integral_delay`` (WI-0058)
    is the integral model with the sample times shifted by ``delay_us``.
    """
    if phase_model == "none":
        return None
    if phase_model == "integral_delay":
        import eval_timing_centre

        return eval_timing_centre.delayed_phase_factor(recon_info, delay_us, n_points)
    if phase_model == "integral":
        from brkraw_sordino.recon import phase_correction_factor

        return phase_correction_factor(recon_info, eval_ramp.mode_options("post", cache_dir), n_points)
    o1 = np.asarray(recon_info.get("O1List_Hz") or [], dtype=float)
    n_pro = int(recon_info["NPro"])
    if o1.size != n_pro:
        return None
    tau = implied_tau_us(recon_info, phase_model, n_points) * 1e-6
    d = np.roll(o1, 1) - o1
    factor = np.exp(-2j * np.pi * np.outer(d, tau)).astype(np.complex64)
    factor[-1] = 1.0
    return factor


def first_peak_k(traj: np.ndarray, grad: np.ndarray, sample_index: int, matrix: int) -> Dict[str, Any]:
    """Where the first-peak sample sits in k space, per spoke.

    radius_kgrid: |k| in k-grid units (1/FOV; the product's 0.5 is matrix/2
    units); angle_deg: angle between k and the target vector g(i).
    """
    k = traj[:, sample_index, :]
    radius = np.linalg.norm(k, axis=1) * matrix
    g = grad.T / np.linalg.norm(grad.T, axis=1, keepdims=True)
    cosang = np.clip(np.sum(k * g, axis=1) / np.maximum(np.linalg.norm(k, axis=1), 1e-300), -1.0, 1.0)
    angle = np.degrees(np.arccos(cosang))
    return {"radius_kgrid": radius, "angle_deg": angle,
            "radius_median": float(np.median(radius)), "radius_min": float(radius.min()),
            "radius_max": float(radius.max()), "angle_median_deg": float(np.median(angle)),
            "angle_max_deg": float(angle.max())}


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------


@dataclass
class StageConfig:
    """Arguments of one staged run (see ``eval_ramp.EvalConfig`` for the
    shared meanings).

    stages:          stage names run in (c); all seven by default.
    phase_samples:   leading samples per spoke used for the coherence-vs-sample
                     curves and the phase figures (per volume, all models).
    """

    dataset: str
    scan_id: int
    out_dir: str
    exclude: int = 10
    sample_index: int = 1
    max_volumes: Optional[int] = None
    recon_start: Optional[int] = None
    recon_count: int = 60
    roi_half: int = 4
    stages: Tuple[str, ...] = STAGE_NAMES
    ignore_samples: int = 1
    phase_samples: int = 32
    cache_dir: Optional[str] = None
    label: str = ""
    delay_traj_us: Optional[float] = None   # WI-0058: sample-time shift seen by the trajectory (S4, S4z)
    delay_phase_us: Optional[float] = None  # WI-0058: shift seen by the O1-step phase (S4p, S4, S4z)
    cg_iters: int = 10                 # WI-0058: conjugate-gradient iterations of S4z
    cg_ext: int = 2                    # WI-0058: S4z grid covers cg_ext x the FOV (central FOV kept)

    def check(self) -> None:
        if self.exclude < 0 or self.recon_count < 0 or self.roi_half < 0:
            raise ValueError("exclude, recon_count and roi_half must be >= 0")
        if not 0 <= self.sample_index < self.phase_samples:
            raise ValueError("sample_index must be in [0, phase_samples)")
        for s in self.stages:
            st = stage(s)
            if "delay" in st.traj and self.delay_traj_us is None:
                raise ValueError(f"stage {s} needs delay_traj_us (eval_timing_centre.delay_pass)")
            if "delay" in st.phase and self.delay_phase_us is None:
                raise ValueError(f"stage {s} needs delay_phase_us (eval_timing_centre.delay_pass)")
        if self.cg_iters < 0:
            raise ValueError("cg_iters must be >= 0")


def _coherence(z: np.ndarray) -> np.ndarray:
    """|sum_i z_ij| / sum_i |z_ij| per column (1 = all spokes in phase)."""
    return np.abs(z.sum(axis=0)) / np.maximum(np.abs(z).sum(axis=0), 1e-300)


def _circular_std(z: np.ndarray) -> np.ndarray:
    """Circular std of angle(z) per column (unweighted), sqrt(-2 ln R)."""
    u = z / np.maximum(np.abs(z), 1e-300)
    r = np.abs(u.mean(axis=0))
    with np.errstate(divide="ignore"):
        return np.where(r >= 1.0, 0.0, np.sqrt(-2.0 * np.log(np.maximum(r, 1e-300))))


def run(cfg: StageConfig) -> Dict[str, Any]:
    """Stream the scan once, reconstruct every stage, save arrays, figures and
    ``summary.json`` and ``stage_table.md``."""
    cfg.check()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cfg.cache_dir) if cfg.cache_dir else out / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    scan, recon_info, fid_entry, meta = eval_ramp.open_scan(cfg.dataset, cfg.scan_id)
    n_pro = int(recon_info["NPro"])
    n_tot = int(recon_info["NRepetitions"])
    n_vol = n_tot if cfg.max_volumes is None else min(n_tot, cfg.max_volumes)
    n_pts = _n_samples(recon_info)
    n_ph = min(cfg.phase_samples, n_pts)
    matrix = int(recon_info["Matrix"][0])
    grad = gradient_vectors(recon_info)

    from brkraw_sordino.recon import parse_volume_shape

    class _Opt:
        ext_factors = (1.0, 1.0, 1.0)

    vol_shape = parse_volume_shape(recon_info, _Opt())
    stages_run = [stage(s) for s in cfg.stages]
    traj_models = sorted({s.traj for s in stages_run}, key=TRAJ_MODELS.index)
    d_traj = 0.0 if cfg.delay_traj_us is None else float(cfg.delay_traj_us)
    d_phase = 0.0 if cfg.delay_phase_us is None else float(cfg.delay_phase_us)
    phase_list = PHASE_MODELS + (DELAY_PHASE_MODELS if cfg.delay_phase_us is not None else ())
    phase_models_all = [p for p in phase_list if p != "none"]  # always measured in (a)
    trajs = {m: stage_trajectory(recon_info, m, cache_dir, d_traj) for m in traj_models}
    phases = {p: stage_phase_factor(recon_info, p, n_pts, cache_dir, d_phase) for p in phase_models_all}
    phases["none"] = None
    have_phase = all(phases[p] is not None for p in phase_models_all)

    # first-peak k position per trajectory model (data independent)
    kpos = {m: first_peak_k(trajs[m], grad, cfg.sample_index, matrix) for m in traj_models}
    k_off = (trajs["off"] if "off" in trajs else stage_trajectory(recon_info, "off", cache_dir))[:, cfg.sample_index, :]
    for m in traj_models:
        disp = np.linalg.norm(trajs[m][:, cfg.sample_index, :] - k_off, axis=1) * matrix
        kpos[m]["displacement_kgrid"] = disp
        kpos[m]["displacement_median"] = float(np.median(disp))
        kpos[m]["displacement_max"] = float(disp.max())

    rs = cfg.exclude if cfg.recon_start is None else cfg.recon_start
    recon_ids = set(range(rs, min(n_vol, rs + cfg.recon_count)))
    first_peak = np.zeros((n_vol, n_pro), dtype=np.complex64)
    coh = {p: np.zeros((n_vol, n_ph)) for p in phase_list}
    cstd = {p: np.zeros((n_vol, n_ph)) for p in phase_list}
    roi = {s.name: [] for s in stages_run}
    mean_img = {s.name: None for s in stages_run}
    c = [x // 2 for x in vol_shape]
    h = cfg.roi_half
    sl = tuple(slice(ci - h, ci + h + 1) for ci in c)
    rep_vol = None
    virtual: Dict[str, np.ndarray] = {}   # WI-0058 S3z/S4z: virtual leading samples per trajectory model
    n_read = 0
    for v, vol in eval_ramp.iter_volumes(fid_entry, recon_info, n_vol):
        z = vol[0, :, :n_ph].astype(np.complex128)
        first_peak[v] = vol[0, :, cfg.sample_index]
        for p in phase_list:
            zp = z if phases[p] is None else z * phases[p][:, :n_ph]
            coh[p][v] = _coherence(zp)
            cstd[p][v] = _circular_std(zp)
        if v in recon_ids:
            for s in stages_run:
                if s.recon == "fill":
                    import eval_timing_centre

                    if s.traj not in virtual:   # virtual samples on this stage's own trajectory model
                        virtual[s.traj] = eval_timing_centre.leading_points(
                            recon_info, d_traj if s.traj == "integral_delay" else 0.0, cfg.ignore_samples)
                    img = eval_timing_centre.centre_fill_reconstruct(
                        vol[0], trajs[s.traj], vol_shape, cfg.ignore_samples, phases[s.phase], virtual[s.traj],
                        n_iter=cfg.cg_iters, ext=cfg.cg_ext)
                else:
                    img = eval_ramp.reconstruct(vol[0], trajs[s.traj], vol_shape, cfg.ignore_samples,
                                                phases[s.phase])
                mag = np.abs(img)
                roi[s.name].append(float(mag[sl].mean()))
                mean_img[s.name] = mag if mean_img[s.name] is None else mean_img[s.name] + mag
        n_read = v + 1
    first_peak = first_peak[:n_read]
    for p in phase_list:
        coh[p] = coh[p][:n_read]
        cstd[p] = cstd[p][:n_read]
    ex = min(cfg.exclude, max(n_read - 1, 0))
    rep_vol = max(ex, (n_read - 1 + ex) // 2)
    np.save(out / "first_peak_raw.npy", first_peak)
    means = {}
    for s in stages_run:
        if mean_img[s.name] is not None:
            means[s.name] = mean_img[s.name] / len(roi[s.name])
            np.save(out / f"recon_mean_{s.name}.npy", means[s.name])

    # ---- (a) within a volume ------------------------------------------------
    fp_mag = np.abs(first_peak).astype(float)
    cv_per_vol = [evalstats.cv(fp_mag[v].tolist()) for v in range(n_read)]
    a_phase = {}
    gain = {p: coh[p] - coh["none"] for p in phase_list}   # > 0: spokes more aligned than raw
    for p in phase_list:
        zr = first_peak[rep_vol].astype(np.complex128)
        if phases[p] is not None:
            zr = zr * phases[p][:, cfg.sample_index]
        st = circstats.circular_stats(np.angle(zr).tolist())
        a_phase[p] = {
            "coherence_median": float(np.median(coh[p][ex:, cfg.sample_index])),
            "coherence_gain_median": float(np.median(gain[p][ex:, cfg.sample_index])),
            "circular_std_median": float(np.median(cstd[p][ex:, cfg.sample_index])),
            "rep_volume": int(rep_vol),
            "rep_circular_std": float(st["std"]),
            "rep_resultant_length": float(st["resultant_length"]),
            "rep_real_cv": float(np.std(zr.real) / max(abs(zr.real.mean()), 1e-300)),
            "rep_imag_std_over_mag": float(np.std(zr.imag) / max(np.abs(zr).mean(), 1e-300)),
            "coherence_vs_sample": coh[p][ex:].mean(axis=0).tolist(),
            "coherence_gain_vs_sample": gain[p][ex:].mean(axis=0).tolist(),
            "circular_std_vs_sample": cstd[p][ex:].mean(axis=0).tolist(),
            "correction_phase_first_peak_abs_max_rad": (
                0.0 if phases[p] is None else float(np.abs(np.angle(phases[p][:, cfg.sample_index])).max())),
            "correction_phase_last_kept_abs_max_rad": (
                0.0 if phases[p] is None else float(np.abs(np.angle(phases[p][:, n_ph - 1])).max())),
        }
    np.savez(out / "phase_metrics.npz", **{f"coh_{p}": coh[p] for p in phase_list},
             **{f"cstd_{p}": cstd[p] for p in phase_list})

    # ---- (b) across volumes -------------------------------------------------
    stab = (evalstats.pattern_stability(fp_mag.tolist(), exclude=ex) if n_read - ex >= 2 else [])
    tc = fp_mag.mean(axis=1).tolist()
    osc_fid = evalstats.oscillation(tc, exclude=ex) if n_read - ex >= 3 else None
    ss = fp_mag[ex:]
    cv_over_vol = (np.std(ss, axis=0) / np.mean(ss, axis=0)).tolist() if ss.shape[0] > 1 else []
    b_phase = {}
    for p in phase_list:
        series = coh[p][:, cfg.sample_index].tolist()
        b_phase[p] = {
            "coherence_timecourse_oscillation": (evalstats.oscillation(series, exclude=ex)
                                                 if n_read - ex >= 3 else None)}

    # ---- (c) reconstruction -------------------------------------------------
    c_stage = {}
    for s in stages_run:
        vals = roi[s.name]
        entry: Dict[str, Any] = {"n_recon": len(vals)}
        entry["roi_oscillation"] = evalstats.oscillation(vals) if len(vals) >= 3 else None
        if s.name in means:
            m = means[s.name]
            gradm = np.sqrt(sum(np.square(np.gradient(m, axis=a)) for a in range(3)))
            entry["sharpness"] = float(gradm.mean() / m.mean())
            for ref in ("S3", "S1", "S0", "S4"):
                if ref in means and ref != s.name:
                    entry[f"rel_diff_vs_{ref}"] = float(np.linalg.norm(m - means[ref]) / np.linalg.norm(means[ref]))
        c_stage[s.name] = entry

    summary = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
        "method": meta.get("Method"), "version": eval_ramp.detect_version(meta),
        "n_volumes_read": n_read, "n_pro": n_pro, "n_points": n_pts,
        "volume_shape": list(vol_shape), "exclude_used": ex,
        "timing": {k: v for k, v in meta.items() if k != "o1_list"},
        "stages": {s.name: {"traj": s.traj, "phase": s.phase, "label": s.label, "recon": s.recon}
                   for s in stages_run},
        "delay_traj_us": cfg.delay_traj_us, "delay_phase_us": cfg.delay_phase_us,
        "phase_factors_available": have_phase,
        "a_magnitude_cv_median": float(np.median(cv_per_vol[ex:] or cv_per_vol)),
        "a_phase": a_phase,
        "k_first_peak": {m: {k: v for k, v in kpos[m].items() if not isinstance(v, np.ndarray)}
                         for m in traj_models},
        "b_magnitude": {
            "pattern_stability_r": ({"min": float(min(stab)), "median": float(np.median(stab))}
                                    if stab else None),
            "timecourse_oscillation": osc_fid,
            "cv_over_volumes_median": float(np.median(cv_over_vol)) if cv_over_vol else None},
        "b_phase": b_phase,
        "c": c_stage,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "stage_table.md").write_text(stage_table(summary))
    plot_all(out, cfg, summary, first_peak, phases, kpos, coh, cstd, cv_per_vol, stab, tc,
             cv_over_vol, roi, means, rep_vol)
    return summary


# ---------------------------------------------------------------------------
# Where is the first peak? argmax of |FID| per spoke and volume
# ---------------------------------------------------------------------------


def argmax_pass(dataset: str, scan_id: int, out_dir: str, max_volumes: Optional[int] = None,
                exclude: int = 10, label: str = "") -> Dict[str, Any]:
    """Sample index of max |FID| for every spoke of every volume (channel 0).

    The evaluation assumes the first peak is sample 1 (Director's model: the
    TR delay puts the maximum at the second sample). This pass checks it:
    saves ``argmax.npy`` [n_vol, n_pro] (int16), a histogram, the per-spoke
    mode over steady-state volumes against the spoke direction and the size
    of the vector step |g(i) - g(i-1)|, the per-volume fraction of spokes
    whose maximum is not at sample 1, and a JSON summary. Observation only.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scan, recon_info, fid_entry, meta = eval_ramp.open_scan(dataset, scan_id)
    n_pro = int(recon_info["NPro"])
    n_tot = int(recon_info["NRepetitions"])
    n_vol = n_tot if max_volumes is None else min(n_tot, max_volumes)
    am = np.zeros((n_vol, n_pro), dtype=np.int16)
    mag_first = np.zeros((n_vol, n_pro, 4), dtype=np.float32)   # |FID| at samples 0..3
    n_read = 0
    for v, vol in eval_ramp.iter_volumes(fid_entry, recon_info, n_vol):
        mag = np.abs(vol[0])
        am[v] = np.argmax(mag, axis=1)
        mag_first[v] = mag[:, :4]
        n_read = v + 1
    am, mag_first = am[:n_read], mag_first[:n_read]
    ex = min(exclude, max(n_read - 1, 0))
    np.save(out / "argmax.npy", am)
    ss = am[ex:]
    values, counts = np.unique(ss, return_counts=True)
    hist = {int(a): int(c) for a, c in zip(values, counts)}
    frac_not1_per_vol = (am != 1).mean(axis=1)
    # per-spoke mode over steady-state volumes and how often a spoke changes
    mode = np.array([np.bincount(ss[:, i]).argmax() for i in range(n_pro)])
    changes = (ss != mode[None, :]).mean(axis=0)
    g = gradient_vectors(recon_info).T
    step = np.linalg.norm(g - np.roll(g, 1, axis=0), axis=1)
    not1 = mode != 1
    summary = {
        "method": meta.get("Method"), "version": eval_ramp.detect_version(meta),
        "n_volumes_read": n_read, "n_pro": n_pro, "exclude_used": ex,
        "histogram_steady_state": hist,
        "fraction_argmax_is_1": float((ss == 1).mean()),
        "fraction_not1_per_volume_median": float(np.median(frac_not1_per_vol[ex:])),
        "fraction_not1_per_volume_max": float(frac_not1_per_vol[ex:].max()),
        "spokes_with_mode_not_1": int(not1.sum()),
        "spokes_changing_over_volumes_frac": float((changes > 0).mean()),
        "mode_histogram": {int(a): int(c) for a, c in zip(*np.unique(mode, return_counts=True))},
        "step_size_median_all": float(np.median(step)),
        "step_size_median_mode_not_1": float(np.median(step[not1])) if not1.any() else None,
        "direction_mean_abs_z_all": float(np.abs(g[:, 2]).mean()),
        "direction_mean_abs_z_mode_not_1": float(np.abs(g[not1, 2]).mean()) if not1.any() else None,
        "ratio_sample1_over_sample0_median": float(np.median(mag_first[ex:, :, 1] / np.maximum(mag_first[ex:, :, 0], 1e-30))),
        "ratio_sample2_over_sample1_median": float(np.median(mag_first[ex:, :, 2] / np.maximum(mag_first[ex:, :, 1], 1e-30))),
    }
    (out / "argmax_summary.json").write_text(json.dumps(summary, indent=1, default=str))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    title = f"{label or Path(dataset).name} scan {scan_id}"
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    ax[0, 0].bar(list(hist), list(hist.values()))
    ax[0, 0].set_yscale("log"); ax[0, 0].set_xlabel("argmax |FID| sample index"); ax[0, 0].set_ylabel("count (spokes x volumes)")
    ax[0, 0].set_title(f"argmax histogram, volumes >= {ex}: sample 1 in {summary['fraction_argmax_is_1']:.4%}")
    ax[0, 1].plot(mode, ".", ms=2, label="mode of argmax over volumes")
    ax2 = ax[0, 1].twinx()
    ax2.plot(step, lw=0.4, color="grey", alpha=0.6, label="|g(i) - g(i-1)|")
    ax2.set_ylabel("vector step size")
    ax[0, 1].set_xlabel("spoke"); ax[0, 1].set_ylabel("argmax sample (mode)")
    ax[0, 1].set_title(f"per-spoke argmax mode ({int(not1.sum())} spokes not at sample 1) and vector step")
    ax[0, 1].legend(loc="upper left", fontsize=7); ax2.legend(loc="upper right", fontsize=7)
    im = ax[1, 0].imshow(am, aspect="auto", cmap="viridis", interpolation="nearest",
                         vmin=0, vmax=max(3, int(np.percentile(am, 99.9))))
    fig.colorbar(im, ax=ax[1, 0]); ax[1, 0].set_xlabel("spoke"); ax[1, 0].set_ylabel("volume")
    ax[1, 0].set_title("argmax sample per spoke and volume")
    ax[1, 1].plot(frac_not1_per_vol, lw=0.7); ax[1, 1].axvline(ex - 0.5, color="grey", ls="--", lw=0.8)
    ax[1, 1].set_xlabel("volume"); ax[1, 1].set_ylabel("fraction of spokes with argmax != 1")
    ax[1, 1].set_title("spokes whose maximum is not sample 1, per volume")
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(out / "argmax_first_peak.png", dpi=110); plt.close(fig)
    return summary


# ---------------------------------------------------------------------------
# Simulation with known ground truth
# ---------------------------------------------------------------------------


def phantom(shape: Sequence[int]):
    """Four bright points and one cube with sharp edges (complex image)."""
    img = np.zeros(tuple(shape), dtype=np.complex64)
    c = np.array(shape) // 2
    q = max(int(min(shape) // 8), 1)
    points = [c + d for d in ((q, 0, 0), (0, -q - 2, q // 2), (-q - 4, q, -q), (q // 2, q + 4, q + 2))]
    for p in points:
        img[tuple(int(x) for x in p)] = 10.0
    lo, hi = c - max(q // 2, 1), c + max(q // 2, 1)
    img[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] += 1.0
    return img, [tuple(int(x) for x in p) for p in points]


def simulate(recon_info: Dict[str, Any], cache_dir: Path, stages: Sequence[str] = STAGE_NAMES,
             ignore_samples: int = 1) -> Dict[str, Any]:
    """Reconstruct model-generated data with every stage.

    data = forward NUFFT of the phantom on the S2 (integral) trajectory, with
    the integral model's accumulated phase added; ``ideal`` is the adjoint
    of the phase-free data on the true trajectory. So S3 equals ideal by
    construction, and the other stages show their error IF the scanner
    follows the model; the simulation cannot show that it does.
    Returns per stage: rel_error_vs_ideal, point peak ratio and centroid
    offset, and the magnitude images (``images``) for figures.
    """
    from mrinufft import get_operator

    shape = [int(x) for x in recon_info["Matrix"]]
    img, points = phantom(shape)
    tr_true = stage_trajectory(recon_info, "integral", cache_dir)
    ph_true = stage_phase_factor(recon_info, "integral", tr_true.shape[1], cache_dir)
    op = get_operator("finufft")(tr_true.reshape(-1, 3) / 0.5 * np.pi, shape=img.shape, density=False)
    data_ideal = op.op(img).reshape(tr_true.shape[:2])
    data_acq = data_ideal if ph_true is None else data_ideal / ph_true
    ideal = np.abs(eval_ramp.reconstruct(data_ideal, tr_true, shape, ignore_samples))
    res: Dict[str, Any] = {"n_pro": int(tr_true.shape[0]),
                           "phase_abs_max_rad": 0.0 if ph_true is None else float(np.abs(np.angle(ph_true)).max()),
                           "stages": {}, "images": {"ideal": ideal}}
    trajs: Dict[str, np.ndarray] = {}
    for name in stages:
        s = stage(name)
        if s.traj not in trajs:
            trajs[s.traj] = stage_trajectory(recon_info, s.traj, cache_dir)
        ph = stage_phase_factor(recon_info, s.phase, tr_true.shape[1], cache_dir)
        rec = np.abs(eval_ramp.reconstruct(data_acq, trajs[s.traj], shape, ignore_samples, ph))
        pts = []
        for p in points:
            blk_sl = tuple(slice(max(x - 2, 0), x + 3) for x in p)
            blk, ref = rec[blk_sl], ideal[blk_sl]
            w = blk ** 2
            grid = np.indices(blk.shape)
            cen = [float((g * w).sum() / w.sum()) - (x - max(x - 2, 0)) for g, x in zip(grid, p)]
            pts.append({"peak_rel": float(blk.max() / ref.max()), "centroid_offset": cen})
        res["stages"][name] = {
            "rel_error_vs_ideal": float(np.linalg.norm(rec - ideal) / np.linalg.norm(ideal)),
            "point_peak_rel_min": float(min(q["peak_rel"] for q in pts)),
            "point_peak_rel_max": float(max(q["peak_rel"] for q in pts)),
            "centroid_offset_max": float(max(abs(c) for q in pts for c in q["centroid_offset"])),
            "points": pts}
        res["images"][name] = rec
    return res


# ---------------------------------------------------------------------------
# Tables and figures
# ---------------------------------------------------------------------------

PHASE_OF_STAGE = {s.name: s.phase for s in STAGES + STAGES_WI0058}
TRAJ_OF_STAGE = {s.name: s.traj for s in STAGES + STAGES_WI0058}


def stage_table(summary: Dict[str, Any]) -> str:
    """One Markdown table, rows = metrics, columns = stages (Lee's formatter)."""
    names = list(summary["stages"])
    cells: Dict[Tuple[str, str], Any] = {}
    rows: List[str] = []
    fmt: Dict[str, str] = {}

    def add(row, fn, f=".4f"):
        rows.append(row)
        fmt[row] = f
        for n in names:
            try:
                cells[(row, n)] = fn(n)
            except (KeyError, TypeError, IndexError):
                cells[(row, n)] = None

    a, k, b, c = summary["a_phase"], summary["k_first_peak"], summary["b_phase"], summary["c"]
    add("(a) |first peak| CV across spokes", lambda n: summary["a_magnitude_cv_median"])
    add("(a) first-peak phase coherence (median)", lambda n: a[PHASE_OF_STAGE[n]]["coherence_median"], ".5f")
    add("(a) first-peak circular std, rad (median)", lambda n: a[PHASE_OF_STAGE[n]]["circular_std_median"])
    add("(a) first-peak coherence gain vs raw (median)", lambda n: a[PHASE_OF_STAGE[n]]["coherence_gain_median"], ".2e")
    add("(a) coherence gain vs raw at sample 8", lambda n: a[PHASE_OF_STAGE[n]]["coherence_gain_vs_sample"][8], ".2e")
    add("(a) coherence at last kept sample", lambda n: a[PHASE_OF_STAGE[n]]["coherence_vs_sample"][-1], ".5f")
    add("(a) coherence gain vs raw at last kept sample", lambda n: a[PHASE_OF_STAGE[n]]["coherence_gain_vs_sample"][-1], ".2e")
    add("correction phase at first peak, |max| rad", lambda n: a[PHASE_OF_STAGE[n]]["correction_phase_first_peak_abs_max_rad"], ".3f")
    add("first-peak |k|, k-grid units (median)", lambda n: k[TRAJ_OF_STAGE[n]]["radius_median"])
    add("first-peak displacement from S0, k-grid units (median)", lambda n: k[TRAJ_OF_STAGE[n]]["displacement_median"])
    add("first-peak angle to target vector, deg (median)", lambda n: k[TRAJ_OF_STAGE[n]]["angle_median_deg"], ".2f")
    add("(b) |first peak| pattern r (min)", lambda n: summary["b_magnitude"]["pattern_stability_r"]["min"], ".5f")
    add("(b) |first peak| time course rel. std", lambda n: summary["b_magnitude"]["timecourse_oscillation"]["rel_std"])
    add("(b) coherence time course rel. std",
        lambda n: b[PHASE_OF_STAGE[n]]["coherence_timecourse_oscillation"]["rel_std"], ".5f")
    add("(c) centre ROI oscillation rel. std", lambda n: c[n]["roi_oscillation"]["rel_std"], ".5f")
    add("(c) mean image sharpness", lambda n: c[n]["sharpness"])
    add("(c) mean image rel. diff vs S3", lambda n: c[n]["rel_diff_vs_S3"], ".1%")
    add("(c) mean image rel. diff vs S1", lambda n: c[n]["rel_diff_vs_S1"], ".1%")
    if "S4" in names:
        add("(c) mean image rel. diff vs S4", lambda n: c[n]["rel_diff_vs_S4"], ".1%")
    return stagetable.format_stage_table(rows, names, cells, fmt=fmt, first_header="metric")


def plot_orthogonal_grid(path: Path, images: Dict[str, np.ndarray], title: str,
                         ref: Optional[str] = None) -> None:
    """Rows = images (or image - ref when ``ref`` is given), columns = axial /
    coronal / sagittal centre planes. Display only, no orientation change."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in images if n != ref] if ref else list(images)
    fig, ax = plt.subplots(len(names), 3, figsize=(12, 3.6 * len(names)))
    ax = np.atleast_2d(ax)
    base = images[ref] if ref else images[names[0]]
    cx, cy, cz = (s // 2 for s in base.shape)
    vmax = float(np.percentile(base, 99.5))
    diffs = {n: images[n] - base for n in names} if ref else {}
    lim = float(np.percentile(np.abs(np.stack(list(diffs.values()))), 99.5)) if ref else 1.0

    def planes(img):
        return img[:, :, cz].T, img[:, cy, :].T, img[cx, :, :].T

    for r, name in enumerate(names):
        img = diffs[name] if ref else images[name]
        for col, (plane, lab) in enumerate(zip(planes(img), ("axial (z)", "coronal (y)", "sagittal (x)"))):
            if ref:
                im = ax[r, col].imshow(plane, cmap="coolwarm", vmin=-lim, vmax=lim, origin="lower")
            else:
                im = ax[r, col].imshow(plane, cmap="gray", vmin=0, vmax=vmax, origin="lower")
            ax[r, col].set_xticks([]); ax[r, col].set_yticks([])
            ax[r, col].set_title(f"{name}{' - ' + ref if ref else ''}: {lab}", fontsize=9)
        if ref:
            rel = float(np.linalg.norm(diffs[name]) / np.linalg.norm(base))
            ax[r, 0].set_ylabel(f"|diff|/|{ref}| = {rel:.4f}")
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    if ref:
        fig.colorbar(im, ax=ax[:, 2].tolist(), fraction=0.02, pad=0.02)
    fig.savefig(path, dpi=100); plt.close(fig)


def plot_all(out: Path, cfg: StageConfig, summary, first_peak, phases, kpos, coh, cstd,
             cv_per_vol, stab, tc, cv_over_vol, roi, means, rep_vol) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    title = f"{cfg.label or Path(cfg.dataset).name} scan {cfg.scan_id}"
    ex = summary["exclude_used"]
    si = cfg.sample_index
    plabel = {"none": "phase off (S0/S1/S1h/S2)", "legacy": "S1 phase", "legacy_half": "S1h phase",
              "integral": "S3 phase (integral)", "integral_delay": "S4 phase (integral + delay)"}
    tlabel = {"off": "S0", "legacy": "S1", "legacy_half": "S1h", "integral": "S2/S3",
              "integral_delay": "S4/S4z"}

    # (a) first peak within one volume, per phase model
    fig, ax = plt.subplots(4, 2, figsize=(14, 16))
    zr = first_peak[rep_vol].astype(np.complex128)
    ax[0, 0].plot(np.abs(zr), lw=0.5)
    ax[0, 0].set_title(f"(a) |first peak| (sample {si}) across spokes, volume {rep_vol}: same for every stage")
    ax[0, 0].set_xlabel("spoke"); ax[0, 0].set_ylabel("|FID|")
    for p in coh:
        z = zr if phases[p] is None else zr * phases[p][:, si]
        unw = circstats.unwrap(np.angle(z).tolist())
        ax[0, 1].plot(unw, lw=0.6, label=plabel[p])
        ax[1, 0].plot(z.real, lw=0.5, label=plabel[p])
        ax[1, 1].plot(z.imag, lw=0.5, label=plabel[p])
        ax[2, 0].plot(coh[p][:, si], lw=0.7, label=plabel[p])
        ax[2, 1].plot(cstd[p][:, si], lw=0.7, label=plabel[p])
        if phases[p] is not None:
            ax[3, 0].plot(np.angle(phases[p][:, si]), lw=0.5, label=plabel[p])
            ax[3, 1].plot(np.angle(phases[p][:, coh[p].shape[1] - 1]), lw=0.5, label=plabel[p])
    ax[3, 0].set_title(f"correction phase applied at the first peak (sample {si}), rad, per spoke")
    ax[3, 1].set_title(f"correction phase applied at sample {coh['none'].shape[1] - 1}, rad, per spoke")
    ax[3, 0].set_xlabel("spoke"); ax[3, 1].set_xlabel("spoke")
    ax[0, 1].set_title("(a) first-peak phase across spokes, unwrapped (rad)"); ax[0, 1].set_xlabel("spoke")
    ax[1, 0].set_title("(a) first-peak real part across spokes"); ax[1, 0].set_xlabel("spoke")
    ax[1, 1].set_title("(a) first-peak imaginary part across spokes"); ax[1, 1].set_xlabel("spoke")
    ax[2, 0].set_title("(a) first-peak phase coherence per volume, |sum z| / sum |z|")
    ax[2, 1].set_title("(a) first-peak circular std across spokes per volume (rad)")
    for a_ in (ax[2, 0], ax[2, 1]):
        a_.axvline(ex - 0.5, color="grey", ls="--", lw=0.8); a_.set_xlabel("volume")
    for a_ in ax.ravel()[1:]:
        a_.legend(fontsize=7)
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(out / "a_first_peak_stages.png", dpi=110); plt.close(fig)

    # (a) coherence vs sample index, and the gain of each phase model over raw
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    for p in coh:
        ax[0].plot(coh[p][ex:].mean(axis=0), "o-", ms=3, lw=0.8, label=plabel[p])
        ax[1].plot(cstd[p][ex:].mean(axis=0), "o-", ms=3, lw=0.8, label=plabel[p])
        if p != "none":
            ax[2].plot((coh[p] - coh["none"])[ex:].mean(axis=0), "o-", ms=3, lw=0.8, label=plabel[p])
    ax[0].axvline(si, color="grey", ls=":", lw=0.8, label=f"first peak (sample {si})")
    ax[0].set_xlabel("sample index"); ax[0].set_ylabel("coherence across spokes (mean over volumes)")
    ax[0].set_title("(a) phase coherence vs sample")
    ax[1].set_xlabel("sample index"); ax[1].set_ylabel("circular std across spokes (rad)")
    ax[1].set_title("(a) circular std vs sample")
    ax[2].axhline(0, color="grey", lw=0.8); ax[2].axvline(si, color="grey", ls=":", lw=0.8)
    ax[2].set_xlabel("sample index"); ax[2].set_ylabel("coherence(model) - coherence(raw)")
    ax[2].set_title("(a) coherence gain over raw: > 0 = spokes more aligned after correction")
    for a_ in ax:
        a_.legend(fontsize=7)
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(out / "a_phase_coherence_vs_sample.png", dpi=110); plt.close(fig)

    # first-peak k position per trajectory model
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    r0 = kpos["off"]["radius_median"] if "off" in kpos else None
    for m, kp in kpos.items():
        ax[0].plot(kp["displacement_kgrid"], lw=0.6, label=tlabel[m])
        ax[1].plot(kp["angle_deg"], lw=0.6, label=tlabel[m])
    ax[0].set_xlabel("spoke"); ax[0].set_ylabel("|k_stage - k_S0| of first peak (k-grid units, 1/FOV)")
    ax[0].set_title(f"first-peak (sample {si}) displacement from S0 per spoke"
                    + (f" (S0 radius {r0:.4f})" if r0 is not None else ""))
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("spoke"); ax[1].set_ylabel("angle to target vector g(i) (deg)")
    ax[1].set_title("first-peak direction error per spoke"); ax[1].legend(fontsize=7)
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(out / "k_first_peak_position.png", dpi=110); plt.close(fig)

    # (b) across volumes: magnitude (stage-invariant) and coherence time courses
    n_vol = first_peak.shape[0]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fp_mag = np.abs(first_peak).astype(float)
    rel = fp_mag / fp_mag[ex:].mean(axis=0, keepdims=True) if n_vol > ex else fp_mag
    im = ax[0, 0].imshow(rel, aspect="auto", cmap="coolwarm", vmin=0.8, vmax=1.2, interpolation="nearest")
    fig.colorbar(im, ax=ax[0, 0]); ax[0, 0].set_xlabel("spoke"); ax[0, 0].set_ylabel("volume")
    ax[0, 0].set_title("(b) |first peak| / steady-state mean: same for every stage")
    if stab:
        ax[0, 1].plot(range(ex, ex + len(stab)), stab, lw=0.8)
    ax[0, 1].set_title(f"(b) |first peak| pattern stability r (first {ex} excluded): same for every stage")
    ax[0, 1].set_xlabel("volume")
    ax[1, 0].plot(tc, lw=0.8); ax[1, 0].axvline(ex - 0.5, color="grey", ls="--", lw=0.8)
    ax[1, 0].set_title("(b) mean |first peak| time course: same for every stage"); ax[1, 0].set_xlabel("volume")
    for p in coh:
        ax[1, 1].plot(coh[p][:, si], lw=0.7, label=plabel[p])
    ax[1, 1].axvline(ex - 0.5, color="grey", ls="--", lw=0.8)
    ax[1, 1].set_title("(b) first-peak coherence time course per phase model"); ax[1, 1].set_xlabel("volume")
    ax[1, 1].legend(fontsize=7)
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(out / "b_across_volumes_stages.png", dpi=110); plt.close(fig)

    # (c) reconstruction
    if means:
        fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
        for name, vals in roi.items():
            if vals:
                ax[0].plot(vals, lw=0.8, label=name)
        ax[0].set_xlabel(f"volume (from {cfg.recon_start if cfg.recon_start is not None else ex})")
        ax[0].set_ylabel(f"mean |image| in centre {2 * cfg.roi_half + 1}^3 voxels")
        ax[0].set_title("(c) centre-region value over volumes per stage"); ax[0].legend(fontsize=7)
        for name, m in means.items():
            cz, cy = m.shape[2] // 2, m.shape[1] // 2
            ax[1].plot(m[:, cy, cz], lw=0.8, label=name)
        ax[1].set_xlabel("voxel along axis 0 (centre line)"); ax[1].set_ylabel("mean |image|")
        ax[1].set_title("(c) centre line of the mean image per stage"); ax[1].legend(fontsize=7)
        fig.suptitle(title); fig.tight_layout()
        fig.savefig(out / "c_recon_centre_stages.png", dpi=110); plt.close(fig)
        plot_orthogonal_grid(out / "c_orthogonal_stages.png", means, f"{title}: mean image per stage")
        for ref in ("S3", "S1"):
            if ref in means and len(means) > 1:
                plot_orthogonal_grid(out / f"c_orthogonal_diff_vs_{ref}.png", means,
                                     f"{title}: stage - {ref}", ref=ref)
        new = {k: means[k] for k in ("S3", "S4p", "S4", "S4z") if k in means}
        if len(new) > 1 and "S3" in new:   # WI-0058 stages only
            plot_orthogonal_grid(out / "c_orthogonal_wi0058.png", new,
                                 f"{title}: S3 and WI-0058 stages (delay traj {cfg.delay_traj_us} us, "
                                 f"phase {cfg.delay_phase_us} us)")
            plot_orthogonal_grid(out / "c_orthogonal_wi0058_diff_vs_S3.png", new,
                                 f"{title}: WI-0058 stage - S3", ref="S3")


__all__ = ["STAGES", "STAGES_WI0058", "STAGE_NAMES", "ALL_STAGE_NAMES", "Stage", "StageConfig", "stage",
           "legacy_trajectory",
           "stage_trajectory", "implied_tau_us", "stage_phase_factor", "first_peak_k", "run",
           "simulate", "phantom", "stage_table", "plot_orthogonal_grid"]
