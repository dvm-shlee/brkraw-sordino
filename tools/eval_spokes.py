"""Per-spoke comparison of the reconstruction conditions (WI-0056 run 6).

Two figure families the Director asked for, drawn for every condition

    S0, S1, S1+p, S1h, S1h+p, S2, S3   (run 5 stage matrix, ``eval_stages``)
    S3z                                (WI-0058: S3 + estimated k-space centre)
    S3c                                (WI-0058 run 2: S3 + first samples restored
                                        from the FID curve)

1. accumulated FID: every spoke of one volume drawn semi-transparent against
   the sample index, the spoke mean on top (|FID| only: Director, WI-0058
   run 2, "magnitude 한 값을 넣는게 맞는것같아요");
2. first-peak phase within the volume: the phase across spokes and the
   difference between neighbouring spokes, raw and after each condition's
   correction, plus the correction itself and the first-peak k position.

Layout (Director, 2026-09-29): conditions are never overlaid on one axis;
each condition is one row, all rows on the same y scale, shifted vertically
(``offsetstack.stack_layout``, Lee Minjun, wi-0056-lee-5).

What a condition changes in these curves (by principle):

    trajectory only (S0, S1, S1h, S2):   the FID is the raw FID; only the
                                         first-peak k position differs
    phase (S1+p, S1h+p, S3):             FID x exp(-i phi_ij) of its own model
    S3z:                                 S3's FID plus the estimated values at
                                         the unsampled leading positions
                                         (dropped sample 0 and the dead time,
                                         down to the RF centre)
    S3c:                                 S3's FID with the FID-curve values there
                                         and at the kept samples inside 0.7
                                         k-grid units (receiver-filter settling)

Development tool, not part of the installed package; product code unchanged.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import circstats
    import eval_ramp
    import eval_stages
    import offsetstack
except ImportError:  # pragma: no cover - run from another folder
    from tools import circstats, eval_ramp, eval_stages, offsetstack  # type: ignore


CONDITIONS: Tuple[str, ...] = ("S0", "S1", "S1p", "S1h", "S1hp", "S2", "S3", "S3z", "S3c")
LABELS = {"S0": "S0 no ramp", "S1": "S1 legacy", "S1p": "S1+phase", "S1h": "S1h legacy/2",
          "S1hp": "S1h+phase", "S2": "S2 integral", "S3": "S3 integral+phase",
          "S3z": "S3z S3+centre", "S3c": "S3c S3+FID curve"}


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


def neighbour_phase_diff(z: np.ndarray) -> np.ndarray:
    """angle(z_i * conj(z_{i-1})) along the last axis, wrapped to (-pi, pi];
    i = 0 pairs with the last spoke (the product's g(i-1) convention)."""
    d = np.angle(z * np.conj(np.roll(z, 1, axis=-1)))
    return np.where(d <= -np.pi, d + 2 * np.pi, d)


def neighbour_stats(z: np.ndarray) -> Dict[str, float]:
    """First-peak statistics of one volume (z: complex, one value per spoke).

    nd_circular_std: circular std of the neighbour differences, sqrt(-2 ln R);
    nd_mean_abs:     mean |neighbour difference| (rad);
    coherence:       |sum z| / sum |z| (1 = all spokes in phase).
    """
    d = neighbour_phase_diff(z)
    r = float(np.abs(np.exp(1j * d).mean()))
    return {"nd_circular_std": float(np.sqrt(-2.0 * np.log(max(r, 1e-300)))) if r < 1.0 else 0.0,
            "nd_mean_abs": float(np.abs(d).mean()),
            "coherence": float(np.abs(z.sum()) / max(np.abs(z).sum(), 1e-300))}


def condition_fid(vol: np.ndarray, factor: Optional[np.ndarray], virtual: Optional[np.ndarray] = None,
                  ignore_samples: int = 1, replaced: Optional[Dict[int, np.ndarray]] = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """(sample index, FID) of one condition for one volume (vol: n_pro x N).

    factor: exp(-i phi) of the condition's phase model or None (raw FID).
    virtual: S3z only, (n_pro, M) estimated values at the M leading positions
    before the first kept sample, times t_first - m * dwell in increasing
    order; they take the sample indices ignore_samples - M .. ignore_samples - 1
    and replace the dropped samples. Measured samples keep their index.
    replaced: S3c only, {kept sample index: (n_pro,) curve values} put in place
    of those measured samples.
    """
    z = vol if factor is None else vol * factor
    if replaced:
        z = np.array(z, copy=True)
        for j, val in replaced.items():
            z[:, j] = val
    n = z.shape[1]
    if virtual is None:
        return np.arange(n, dtype=float), z
    m = virtual.shape[1]
    x = np.concatenate([np.arange(ignore_samples - m, ignore_samples, dtype=float),
                        np.arange(ignore_samples, n, dtype=float)])
    return x, np.concatenate([virtual.astype(z.dtype), z[:, ignore_samples:]], axis=1)


def fid_curves(x: np.ndarray, z: np.ndarray, norm: float, sample_x: float, n_phase: int,
               line_idx: np.ndarray) -> Dict[str, Any]:
    """Curves of one condition: |z|/norm per spoke (subset ``line_idx``) and
    its spoke mean, the coherent mean |mean_i z|/norm, and the phase relative
    to each spoke's own first peak, angle(z_ij conj(z_i,first)), over the first
    samples below index ``n_phase`` (S3z: plus its leading samples), with its
    circular mean."""
    j1 = int(np.argmin(np.abs(x - sample_x)))
    mag = np.abs(z) / norm
    rel = z[:, :] * np.conj(z[:, j1:j1 + 1]) / np.maximum(np.abs(z[:, j1:j1 + 1]), 1e-300)
    ph_sl = x < n_phase   # the same window for every condition (S3z adds its leading samples)
    ph = np.angle(rel[:, ph_sl])
    return {"x": x, "mag_lines": mag[line_idx], "mag_mean": mag.mean(axis=0),
            "mag_coherent": np.abs(z.mean(axis=0)) / norm,
            "x_phase": x[ph_sl], "phase_lines": ph[line_idx],
            "phase_mean": np.angle(np.exp(1j * ph).sum(axis=0))}


# ---------------------------------------------------------------------------
# Pass
# ---------------------------------------------------------------------------


def spoke_pass(dataset: str, scan_id: int, out_dir: str, volume: int, stat_start: int = 10,
               stat_count: int = 100, sample_index: int = 1, n_phase: int = 16, n_zoom: int = 32,
               ignore_samples: int = 1, max_lines: int = 3200, cache_dir: Optional[str] = None,
               cg_iters: int = 10, cg_ext: int = 1, conditions: Sequence[str] = CONDITIONS,
               label: str = "") -> Dict[str, Any]:
    """Stream volumes 0 .. max(volume, stat_start + stat_count - 1) of a scan
    once; draw the per-spoke figures for ``volume``; neighbour statistics of the
    first peak for the volumes [stat_start, stat_start + stat_count).

    Writes ``spoke_fid_accumulated.png``, ``spoke_fid_correction_phase.png``,
    ``first_peak_phase_within_volume.png``, ``first_peak_parts_and_k.png``,
    ``first_peak_neighbour_over_volumes.png``, ``spokes_summary.json`` and
    ``spoke_curves.npz`` (curve means only, no raw FID).
    """
    import eval_timing_centre

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir) if cache_dir else out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    scan, recon_info, fid_entry, meta = eval_ramp.open_scan(dataset, scan_id)
    n_pro = int(recon_info["NPro"])
    n_tot = int(recon_info["NRepetitions"])
    n_pts = eval_stages._n_samples(recon_info)
    matrix = int(recon_info["Matrix"][0])
    stat_start = min(stat_start, n_tot - 1)
    stat_stop = min(n_tot, stat_start + stat_count)
    volume = min(volume, n_tot - 1)
    n_read = max(volume + 1, stat_stop)

    stages = {c: eval_stages.stage(c) for c in conditions}
    traj_models = sorted({s.traj for s in stages.values()} | {"off"}, key=eval_stages.TRAJ_MODELS.index)
    trajs = {m: eval_stages.stage_trajectory(recon_info, m, cache) for m in traj_models}
    phase_models = sorted({s.phase for s in stages.values()}, key=eval_stages.PHASE_MODELS.index)
    factors = {p: eval_stages.stage_phase_factor(recon_info, p, n_pts, cache) for p in phase_models}
    grad = eval_stages.gradient_vectors(recon_info)
    kpos = {m: eval_stages.first_peak_k(trajs[m], grad, sample_index, matrix) for m in traj_models}
    k_off = trajs["off"][:, sample_index, :]
    for m in traj_models:
        kpos[m]["displacement_kgrid"] = np.linalg.norm(trajs[m][:, sample_index, :] - k_off, axis=1) * matrix

    stats = {c: [] for c in conditions}
    fp_vol = None
    vol_keep = None
    for v, vol in eval_ramp.iter_volumes(fid_entry, recon_info, n_read):
        if stat_start <= v < stat_stop:
            for c, s in stages.items():
                f = factors[s.phase]
                z = vol[0, :, sample_index].astype(np.complex128)
                stats[c].append(neighbour_stats(z if f is None else z * f[:, sample_index]))
        if v == volume:
            vol_keep = vol[0].astype(np.complex64)
    assert vol_keep is not None
    fp_vol = vol_keep[:, sample_index].astype(np.complex128)

    # S3z: estimated values at the leading positions (same solve as eval_stages S3z)
    virtual = None
    virt_info: Dict[str, Any] = {}
    if "S3z" in stages:
        s = stages["S3z"]
        vtraj = eval_timing_centre.leading_points(recon_info, 0.0, ignore_samples)
        _, info = eval_timing_centre.centre_fill_reconstruct(
            vol_keep, trajs[s.traj], eval_ramp_volume_shape(recon_info), ignore_samples,
            factors[s.phase], vtraj, n_iter=cg_iters, ext=cg_ext, return_info=True)
        virtual = info["virtual_values"]
        virt_info = {"virtual_samples": int(virtual.shape[1]),
                     "virtual_radius_kgrid": (np.linalg.norm(vtraj, axis=2).mean(axis=0) * matrix).tolist()}

    # S3c: FID-curve values at the same leading positions and the kept samples inside the window start
    curve_virtual = None
    curve_replaced: Dict[int, np.ndarray] = {}
    if "S3c" in stages:
        s = stages["S3c"]
        _, _, cinfo = eval_timing_centre.curve_fill_data(vol_keep, trajs[s.traj], recon_info, ignore_samples,
                                                         factors[s.phase])
        m = cinfo["virtual_samples"]
        curve_virtual = cinfo["estimated"][:, :m]
        curve_replaced = {j: cinfo["estimated"][:, m + q] for q, j in enumerate(cinfo["replaced_kept"])}
        virt_info["s3c"] = {"fit_cols": cinfo["fit_cols"], "replaced_kept": cinfo["replaced_kept"],
                            "virtual_samples": m, "model": cinfo["model"],
                            "fit_radius_kgrid": cinfo["fit_radius_kgrid"]}

    norm = float(np.abs(vol_keep[:, sample_index]).mean())
    step = max(1, int(np.ceil(n_pro / max_lines)))
    line_idx = np.arange(0, n_pro, step)
    curves: Dict[str, Dict[str, Any]] = {}
    for c, s in stages.items():
        f = factors[s.phase]
        if s.recon == "curve":
            x, z = condition_fid(vol_keep, f, curve_virtual, ignore_samples, curve_replaced)
        else:
            x, z = condition_fid(vol_keep, f, virtual if s.recon == "fill" else None, ignore_samples)
        curves[c] = fid_curves(x, z, norm, float(sample_index), n_phase, line_idx)
        curves[c]["correction"] = None if f is None else np.angle(f)
        if s.recon == "fill":
            m = virtual.shape[1]
            est0 = np.abs(z[:, m - ignore_samples]) if m >= ignore_samples else None
            virt_info.update({
                "est_sample0_over_first_peak": None if est0 is None else float(est0.mean() / norm),
                "measured_sample0_over_first_peak": float(np.abs(vol_keep[:, 0]).mean() / norm),
                "est_centre_over_first_peak": float(np.abs(z[:, 0]).mean() / norm)})

    summary: Dict[str, Any] = {
        "method": meta.get("Method"), "version": eval_ramp.detect_version(meta),
        "n_pro": n_pro, "n_points": n_pts, "n_volumes_total": n_tot, "n_volumes_read": n_read,
        "figure_volume": int(volume), "stat_volumes": [int(stat_start), int(stat_stop)],
        "sample_index": sample_index, "ignore_samples": ignore_samples,
        "lines_drawn": int(line_idx.size), "line_step": step, "norm_first_peak_mean_abs": norm,
        "conditions": {c: {"traj": s.traj, "phase": s.phase, "recon": s.recon} for c, s in stages.items()},
        "s3z": virt_info, "per_condition": {}}
    for c in conditions:
        st = stats[c]
        f = factors[stages[c].phase]
        nd = neighbour_phase_diff(fp_vol if f is None else fp_vol * f[:, sample_index])
        summary["per_condition"][c] = {
            "nd_circular_std_median": float(np.median([q["nd_circular_std"] for q in st])) if st else None,
            "nd_mean_abs_median": float(np.median([q["nd_mean_abs"] for q in st])) if st else None,
            "coherence_median": float(np.median([q["coherence"] for q in st])) if st else None,
            "figure_volume_nd_circular_std": neighbour_stats(fp_vol if f is None else fp_vol * f[:, sample_index])["nd_circular_std"],
            "figure_volume_nd_abs_max": float(np.abs(nd).max()),
            "correction_first_peak_abs_max_rad": 0.0 if f is None else float(np.abs(np.angle(f[:, sample_index])).max()),
            "correction_last_sample_abs_max_rad": 0.0 if f is None else float(np.abs(np.angle(f[:, -1])).max()),
            "coherent_mean_at_first_peak": float(curves[c]["mag_coherent"][int(np.argmin(np.abs(curves[c]["x"] - sample_index)))]),
            "k_displacement_median_kgrid": float(np.median(kpos[stages[c].traj]["displacement_kgrid"])),
        }
    summary["elapsed_s"] = round(time.time() - t0, 1)
    (out / "spokes_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    np.savez(out / "spoke_curves.npz", **{f"{c}_{k}": curves[c][k] for c in conditions
                                          for k in ("x", "mag_mean", "mag_coherent", "x_phase", "phase_mean")})
    title = f"{label or Path(dataset).name} scan {scan_id}, volume {volume}"
    plot_accumulated(out / "spoke_fid_accumulated.png", curves, conditions, title, sample_index, ignore_samples,
                     n_zoom, n_phase)
    plot_correction(out / "spoke_fid_correction_phase.png", curves, conditions, title, line_idx)
    plot_first_peak(out, fp_vol, factors, stages, kpos, sample_index, title)
    plot_neighbour_over_volumes(out / "first_peak_neighbour_over_volumes.png", stats, conditions,
                                stat_start, f"{label or Path(dataset).name} scan {scan_id}")
    return summary


def eval_ramp_volume_shape(recon_info: Dict[str, Any]) -> List[int]:
    from brkraw_sordino.recon import parse_volume_shape

    class _Opt:
        ext_factors = (1.0, 1.0, 1.0)

    return [int(x) for x in parse_volume_shape(recon_info, _Opt())]


# ---------------------------------------------------------------------------
# Offset-stack drawing (one row per condition, same y scale)
# ---------------------------------------------------------------------------


def draw_stack(ax, names: Sequence[str], rows: Dict[str, Dict[str, Any]], *, gap: float = 0.25,
               clip_pct: Optional[float] = None, alpha: float = 0.03, zero_line: bool = True,
               note: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Draw one row per name. rows[name]: ``x``, ``mean`` (1-D), optional
    ``lines`` (2-D, drawn semi-transparent), ``x_lines``, ``dashed`` (1-D).
    Every row uses the same y scale; ``offsetstack.stack_layout`` shifts it.
    With ``clip_pct`` the band is the [clip_pct, 100 - clip_pct] percentile
    range of all rows and drawn values are clipped to it, so rows never
    overlap. Returns the layout (lo, hi, step, offsets, centres)."""
    from matplotlib.collections import LineCollection

    mins, maxs = [], []
    for n in names:
        r = rows[n]
        vals = [np.asarray(r["mean"], dtype=float).ravel()]
        if r.get("lines") is not None:
            vals.append(np.asarray(r["lines"], dtype=float).ravel())
        if r.get("dashed") is not None:
            vals.append(np.asarray(r["dashed"], dtype=float).ravel())
        v = np.concatenate(vals)
        v = v[np.isfinite(v)]
        if clip_pct:
            mins.append(float(np.percentile(v, clip_pct)))
            maxs.append(float(np.percentile(v, 100 - clip_pct)))
        else:
            mins.append(float(v.min()))
            maxs.append(float(v.max()))
    if clip_pct:
        lo_all, hi_all = min(mins), max(maxs)
        mins, maxs = [lo_all] * len(names), [hi_all] * len(names)
    lay = offsetstack.stack_layout(mins, maxs, gap)
    lo, hi = lay["lo"], lay["hi"]
    for k, n in enumerate(names):
        r = rows[n]
        off = lay["offsets"][k]
        base = lo + off
        ax.axhspan(base, hi + off, color="0.5" if k % 2 else "0.8", alpha=0.08, lw=0)
        if zero_line and lo <= 0.0 <= hi:
            ax.axhline(off, color="0.6", lw=0.5, ls=":")
        if r.get("lines") is not None:
            lines = np.clip(np.asarray(r["lines"], dtype=float), lo, hi)
            xl = np.asarray(r.get("x_lines", r["x"]), dtype=float)
            segs = np.stack([np.broadcast_to(xl, lines.shape), lines + off], axis=-1)
            ax.add_collection(LineCollection(segs, colors="C0", alpha=alpha, linewidths=0.5))
        if r.get("dashed") is not None:
            ax.plot(r["x"], np.clip(r["dashed"], lo, hi) + off, color="C3", lw=0.9, ls="--")
        ax.plot(r["x"], np.clip(np.asarray(r["mean"], dtype=float), lo, hi) + off, color="k", lw=1.1)
        if note and n in note:
            ax.text(1.0, hi + off, note[n], transform=ax.get_yaxis_transform(), ha="right", va="top", fontsize=7)
    ax.set_yticks(lay["centres"])
    ax.set_yticklabels([LABELS.get(n, n) for n in names], fontsize=8)
    ax.set_ylim(lo + lay["offsets"][-1] - 0.05 * (hi - lo), hi + lay["offsets"][0] + 0.05 * (hi - lo))
    ax.autoscale(axis="x")
    return lay


def _scale_text(lay: Dict[str, Any], unit: str) -> str:
    return f"each row spans {lay['lo']:.3g} .. {lay['hi']:.3g} {unit} (same scale for every row)"


def plot_accumulated(path: Path, curves, names, title: str, sample_index: int, ignore_samples: int,
                     n_zoom: int, n_phase: int = 16) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(14, 1.35 * len(names) + 2.5))
    full = {n: {"x": curves[n]["x"], "mean": curves[n]["mag_mean"], "lines": curves[n]["mag_lines"],
                "dashed": curves[n]["mag_coherent"]} for n in names}
    lay = draw_stack(ax[0], names, full)
    ax[0].set_title("|FID| / mean |first peak|, whole readout\n" + _scale_text(lay, ""), fontsize=9)
    zoom = {}
    for n in names:
        x = curves[n]["x"]
        sel = x < n_zoom
        zoom[n] = {"x": x[sel], "mean": curves[n]["mag_mean"][sel], "lines": curves[n]["mag_lines"][:, sel],
                   "dashed": curves[n]["mag_coherent"][sel]}
    lay = draw_stack(ax[1], names, zoom, alpha=0.05)
    ax[1].axvline(ignore_samples - 0.5, color="C2", lw=0.8, ls="--")
    ax[1].axvline(sample_index, color="0.4", lw=0.6, ls=":")
    ax[1].set_title(f"|FID|, first {n_zoom} samples (green: product drops samples left of the line;\n"
                    f"S3z, S3c: estimated values there, down to k = 0) " + _scale_text(lay, ""), fontsize=9)
    # |FID| only (Director, WI-0058 run 2); the phase stays in spoke_fid_correction_phase.png
    for a in ax:
        a.set_xlabel("sample index")
    fig.suptitle(f"{title}: all spokes (blue, semi-transparent), spoke mean (black), "
                 "|coherent mean| (red dashed)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_correction(path: Path, curves, names, title: str, line_idx: np.ndarray) -> None:
    """Correction phase angle(exp(-i phi_ij)) per spoke vs sample (offset stack)
    and as spoke x sample maps, one per phase condition, same colour scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    have = [n for n in names if curves[n]["correction"] is not None]
    if not have:
        return
    fig = plt.figure(figsize=(21, 1.4 * len(names) + 2.5))
    gs = fig.add_gridspec(len(have), 2, width_ratios=[1.2, 1.0])
    ax0 = fig.add_subplot(gs[:, 0])
    rows = {}
    x = np.arange(curves[have[0]]["correction"].shape[1], dtype=float)
    for n in names:
        corr = curves[n]["correction"]
        if corr is None:
            rows[n] = {"x": x, "mean": np.zeros_like(x)}
        else:
            rows[n] = {"x": np.arange(corr.shape[1], dtype=float), "mean": corr.mean(axis=0),
                       "lines": corr[line_idx]}
    lay = draw_stack(ax0, names, rows, alpha=0.04)
    ax0.set_xlabel("sample index")
    ax0.set_title("correction phase applied, angle(exp(-i phi)), rad (0 = phase off)\n" + _scale_text(lay, "rad"),
                  fontsize=9)
    lim = max(float(np.abs(curves[n]["correction"]).max()) for n in have)
    for r, n in enumerate(have):
        a = fig.add_subplot(gs[r, 1])
        im = a.imshow(curves[n]["correction"].T, aspect="auto", origin="lower", cmap="coolwarm",
                      vmin=-lim, vmax=lim, interpolation="nearest")
        a.set_ylabel("sample"); a.set_title(f"{LABELS.get(n, n)}: correction phase, spoke x sample", fontsize=8)
        if r == len(have) - 1:
            a.set_xlabel("spoke")
        fig.colorbar(im, ax=a, fraction=0.03, pad=0.01)
    fig.suptitle(f"{title}: FID phase correction per condition")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_first_peak(out: Path, fp: np.ndarray, factors, stages, kpos, sample_index: int, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(stages)
    spokes = np.arange(fp.size, dtype=float)
    z = {}
    for n, s in stages.items():
        f = factors[s.phase]
        z[n] = fp if f is None else fp * f[:, sample_index]
    same = {"S3z": "= S3 (S3z adds samples before the first kept one only)"}

    fig, ax = plt.subplots(1, 3, figsize=(21, 1.35 * len(names) + 2.5))
    unw = {n: {"x": spokes, "mean": np.asarray(circstats.unwrap(np.angle(z[n]).tolist()))} for n in names}
    lay = draw_stack(ax[0], names, unw, zero_line=False, note=same)
    ax[0].set_title(f"first-peak (sample {sample_index}) phase across spokes, unwrapped, rad\n" + _scale_text(lay, "rad"),
                    fontsize=9)
    nd = {n: {"x": spokes, "mean": np.asarray(offsetstack.neighbour_diff(np.angle(z[n]).tolist()))} for n in names}
    note = {n: f"circ. std {neighbour_stats(z[n])['nd_circular_std']:.4f} rad" for n in names}
    lay = draw_stack(ax[1], names, nd, note=note)
    ax[1].set_title("neighbour-spoke phase difference, wrap(phi_i - phi_(i-1)), rad\n" + _scale_text(lay, "rad"),
                    fontsize=9)
    corr = {n: {"x": spokes, "mean": (np.zeros_like(spokes) if factors[stages[n].phase] is None
                                      else np.angle(factors[stages[n].phase][:, sample_index]))} for n in names}
    lay = draw_stack(ax[2], names, corr)
    ax[2].set_title("correction applied at the first peak (after - before), rad\n" + _scale_text(lay, "rad"),
                    fontsize=9)
    for a in ax:
        a.set_xlabel("spoke")
    fig.suptitle(f"{title}: first-peak phase within the volume, raw and after each condition's correction")
    fig.tight_layout()
    fig.savefig(out / "first_peak_phase_within_volume.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(26, 1.35 * len(names) + 2.5))
    re = {n: {"x": spokes, "mean": z[n].real} for n in names}
    im = {n: {"x": spokes, "mean": z[n].imag} for n in names}
    disp = {n: {"x": spokes, "mean": kpos[stages[n].traj]["displacement_kgrid"]} for n in names}
    ang = {n: {"x": spokes, "mean": kpos[stages[n].traj]["angle_deg"]} for n in names}
    for a, rows, t, u in ((ax[0], re, "first-peak real part", ""), (ax[1], im, "first-peak imaginary part", ""),
                          (ax[2], disp, "first-peak k displacement from S0, k-grid units", "k-grid"),
                          (ax[3], ang, "first-peak angle to the target vector g(i), deg", "deg")):
        lay = draw_stack(a, names, rows)
        a.set_title(t + "\n" + _scale_text(lay, u), fontsize=9)
        a.set_xlabel("spoke")
    fig.suptitle(f"{title}: first-peak parts and k position per condition")
    fig.tight_layout()
    fig.savefig(out / "first_peak_parts_and_k.png", dpi=110)
    plt.close(fig)


def plot_neighbour_over_volumes(path: Path, stats, names, stat_start: int, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not stats[names[0]]:
        return
    fig, ax = plt.subplots(1, 2, figsize=(15, 1.3 * len(names) + 2.5))
    vols = np.arange(stat_start, stat_start + len(stats[names[0]]), dtype=float)
    for a, key, t in ((ax[0], "nd_circular_std", "circular std of neighbour-spoke first-peak phase differences, rad"),
                      (ax[1], "coherence", "first-peak coherence |sum z| / sum |z|")):
        rows = {n: {"x": vols, "mean": np.array([q[key] for q in stats[n]])} for n in names}
        lay = draw_stack(a, names, rows, zero_line=False)
        a.set_title(t + "\n" + _scale_text(lay, ""), fontsize=9)
        a.set_xlabel("volume")
    fig.suptitle(f"{title}: first-peak within-volume statistics per volume and condition")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


__all__ = ["CONDITIONS", "LABELS", "neighbour_phase_diff", "neighbour_stats", "condition_fid",
           "fid_curves", "spoke_pass", "draw_stack", "plot_accumulated", "plot_correction",
           "plot_first_peak", "plot_neighbour_over_volumes"]
