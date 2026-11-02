"""Complex frames and pure-noise frames as a side output (``frame_export=complex``).

The frames of a frame plan are complex images; the NIfTI result keeps their magnitude. With
``frame_export=complex`` the same reconstruction also writes, next to it,

* ``<prefix>_frames_real.nii.gz`` and ``<prefix>_frames_imag.nii.gz``: the complex frames (float32),
* ``<prefix>_frames.tsv``: one row per frame (position, repetition, first and last spoke),
* ``<prefix>_noise_frames_real.nii.gz`` and ``<prefix>_noise_frames_imag.nii.gz``: frames of pure noise
  made from the k-space residuals n'_kr = (d_kr - mean_r d_kr) sqrt(R / (R - 1)) of every spoke position k
  over the R repetitions, reconstructed by the same engine (trajectory, weights, scales, frame plan,
  off-resonance) as the data (``frames.recon_frames(noise_out=...)``),
* ``<prefix>_ksigma.json``: the k-space noise level measured from the same residuals (``ksigma``).

The option is not part of the recon cache key (``EXPORT_KEYS`` are excluded like the physio keys): the
complex frames are the recon cache, the noise frames get a cache file of their own beside it.
Theory: the technical note, section 10.6.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

EXPORT_KEYS = ("frame_export", "frame_export_prefix")
EXPORT_MODES = ("off", "complex")

#: Equal-width bands of |k| (trajectory units, 0.5 = Nyquist) between 0 and the largest |k| read.
N_BANDS = 8

NOISE_SUFFIX = "_noise"


def parse_export(value: Any) -> str:
    """``off`` (also None and False) or ``complex`` (ValueError otherwise)."""
    msg = f"sordino: frame_export must be 'off' or 'complex', got {value!r}"
    if value is None or value is False:
        return "off"
    if isinstance(value, str):
        text = value.strip().lower()
        if text in EXPORT_MODES:
            return text
        if text in ("", "false"):
            return "off"
    raise ValueError(msg)


def is_on(options) -> bool:
    return getattr(options, "frame_export", "off") == "complex"


def check(options, frame_plan, n_rx: int) -> None:
    """Refuse what ``frame_export=complex`` cannot do (ValueError, naming the option)."""
    if not is_on(options):
        if getattr(options, "frame_export_prefix", None):
            raise ValueError("sordino: frame_export_prefix needs frame_export='complex'; "
                             "without it the option would be ignored.")
        return
    if frame_plan is None:
        raise ValueError("sordino: frame_export='complex' needs a frame plan (a frame option such as "
                         "frame_spokes, for example frame_spokes=repetition).")
    if frame_plan.groups is not None:
        raise ValueError("sordino: frame_export='complex' cannot be used with peri_event "
                         "(a bin is the union of several ranges, not one position and repetition).")
    if getattr(options, "subtract_rep_mean", False):
        raise ValueError("sordino: frame_export='complex' cannot be used with subtract_rep_mean.")
    if getattr(options, "anat_solver", "adjoint") != "adjoint":
        raise ValueError("sordino: frame_export='complex' cannot be used with anat_solver='cg'.")
    if int(n_rx) != 1:
        raise ValueError(f"sordino: frame_export='complex' needs a single receive channel, this scan has {n_rx}.")
    if int(frame_plan.n_rep) < 2:
        raise ValueError(
            f"sordino: frame_export='complex' needs at least 2 repetitions: the pure-noise frames are the "
            f"deviations of every spoke position from its mean over the R repetitions and the Bessel factor "
            f"sqrt(R/(R-1)) is undefined for R={int(frame_plan.n_rep)}.")


def bessel_factor(n_rep: int) -> float:
    """sqrt(R / (R - 1)): the residual of one repetition has variance (R - 1) / R of the noise."""
    if int(n_rep) < 2:
        raise ValueError(f"the Bessel factor needs at least 2 repetitions, got {n_rep}")
    return math.sqrt(float(n_rep) / float(n_rep - 1))


# ---------------------------------------------------------------- k-space sigma
def ksigma(read, rep_mean: np.ndarray, rows, n_rep: int, npro: int, ranges, ign: int,
           n_stop: Optional[int], n_bands: int = N_BANDS) -> Dict[str, Any]:
    """k-space noise level from the repetition residuals (technical note, section 10.6).

    ``read(g, n)`` returns stream spokes g .. g + n as (spokes, receivers, points); ``rep_mean`` is the
    (npro, receivers, points) mean over the repetitions, ``rows(a, b)`` the trajectory rows, ``ranges``
    the spoke ranges to walk (``serial.spoke_ranges``). For every sample j of position k
    s2_kj = sum_r |d_kjr - mean_r d_kjr|^2 / (R - 1), the unbiased estimate of E|n|^2 (real and imaginary
    part each carry half). It is pooled over the samples of a band of |k| (and over positions), over the
    samples of a position, and over the outer half of the bands per position for ``sigma_0``.
    """
    n_rep, npro = int(n_rep), int(npro)
    if n_rep < 2:
        raise ValueError(f"ksigma needs at least 2 repetitions, got {n_rep}")
    kmax = 0.0
    for a, b in ranges:
        kmax = max(kmax, float(np.sqrt(np.square(rows(a, b)[:, ign:]).sum(-1)).max()))
    if not kmax > 0:
        raise ValueError("ksigma: the trajectory has no sample off k = 0")
    edges = np.linspace(0.0, kmax, n_bands + 1)
    ss = np.zeros((npro, n_bands))             # sum over repetitions and samples of |residual|^2
    cnt = np.zeros((npro, n_bands))            # samples of the position in the band (one repetition)
    for r in range(n_rep):
        for a, b in ranges:
            k = read(r * npro + a, b - a)
            res = k[..., ign:n_stop] - rep_mean[a:b][..., ign:n_stop]
            power = (res.real ** 2 + res.imag ** 2).sum(axis=1)             # (spokes, points)
            kk = np.sqrt(np.square(rows(a, b)[:, ign:]).sum(-1))
            if kk.shape != power.shape:
                raise ValueError(f"ksigma: {power.shape} samples against {kk.shape} trajectory points")
            band = np.digitize(kk, edges[1:-1])
            flat = (band + n_bands * np.arange(b - a)[:, None]).reshape(-1)
            ss[a:b] += np.bincount(flat, weights=power.reshape(-1), minlength=(b - a) * n_bands).reshape(b - a, -1)
            if r == 0:
                cnt[a:b] += np.bincount(flat, minlength=(b - a) * n_bands).reshape(b - a, -1)
    dof = float(n_rep - 1)

    def sigma(num, den):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sqrt(num / (dof * den))

    band_sigma = sigma(ss.sum(0), cnt.sum(0))
    position_sigma = sigma(ss.sum(1), cnt.sum(1))
    outer = slice(n_bands // 2, n_bands)
    pos_outer = ss[:, outer].sum(1) / (dof * np.where(cnt[:, outer].sum(1) > 0, cnt[:, outer].sum(1), np.nan))
    sigma_0 = float(np.sqrt(np.nanmedian(pos_outer)))
    pooled_outer = float(sigma(ss[:, outer].sum(), cnt[:, outer].sum()))

    def num(x):
        return None if not np.isfinite(x) else float(x)

    return {
        "n_repetitions": n_rep,
        "bessel_factor": bessel_factor(n_rep),
        "n_positions": npro,
        "sigma_0": num(sigma_0),
        "sigma_outer_pooled": num(pooled_outer),
        "band_edges": [float(v) for v in edges],
        "sigma_per_band": [num(v) for v in band_sigma],
        "samples_per_band": [int(v) for v in cnt.sum(0)],
        "sigma_per_position": [num(v) for v in position_sigma],
    }


def ksigma_document(stats: Dict[str, Any], options=None) -> Dict[str, Any]:
    """``ksigma.json``: the statistics with the conventions needed to read them."""
    doc = {
        "definition": ("sigma^2 = E|n|^2 of one complex k-space sample (the real and the imaginary part each "
                       "have variance sigma^2 / 2), from s2 = sum_r |d_r - mean_r d_r|^2 / (R - 1) per sample "
                       "of every spoke position, pooled as named below"),
        "k_unit": "trajectory units (cycles per sample; 0.5 is Nyquist), bands of equal width between 0 and the largest |k| read",
        "notes": {
            "sigma_0": "sqrt of the median over spoke positions of the variance pooled over the outer half of the bands",
            "sigma_outer_pooled": "the same outer-band variance pooled over all positions",
            "sigma_per_band": "pooled over all positions, one value per band_edges interval",
            "sigma_per_position": "pooled over all samples, one value per spoke position of a repetition",
            "bessel_factor": "the noise frames are the residuals times this factor, so their variance is sigma^2",
        },
    }
    doc.update(stats)
    return doc


# ---------------------------------------------------------------- frame table
def frame_rows(plan, frame_list: Optional[List[int]] = None) -> List[Tuple[int, int, int, int, int]]:
    """Rows ``(frame, position, repetition, first_spoke, last_spoke)`` of the frames of ``plan``.

    Spokes are counted in the stream of the repetitions read (0 is the first spoke of the first
    repetition read), ``last_spoke`` is the last spoke of the frame (inclusive). ``position`` is the
    spoke position of the first spoke inside its repetition and ``repetition`` the repetition it falls
    in, counted from the first one read.
    """
    order = range(plan.n_frames) if frame_list is None else [int(i) for i in frame_list]
    rows = []
    for i in order:
        lo, hi = plan.ranges[i]
        rows.append((int(i), int(lo % plan.npro), int(lo // plan.npro), int(lo), int(hi - 1)))
    return rows


def write_tsv(path, plan, frame_list: Optional[List[int]] = None) -> None:
    head = "frame\tposition\trepetition\tfirst_spoke\tlast_spoke\n"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(head)
        for row in frame_rows(plan, frame_list):
            handle.write("\t".join(str(v) for v in row) + "\n")


def write_json(path, doc: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, indent=1, sort_keys=True)
        handle.write("\n")


# ---------------------------------------------------------------- cache and files
def noise_cache_path(img_cache_path):
    """The noise frames' cache file: the data cache's name with ``_noise`` before the suffix."""
    return img_cache_path.with_name(img_cache_path.stem + NOISE_SUFFIX + img_cache_path.suffix)


def read_parts(path, dtype, cached_shape, frame_list, orient) -> Tuple[np.ndarray, np.ndarray]:
    """Real and imaginary parts (float32, x y z frame) of the cache frames in ``frame_list``.

    One frame at a time (the complex128 cache is never held whole); ``orient`` is applied to every
    volume (``orientation.correct``), as for the magnitude result.
    """
    from .cacheio import read_recon_frames

    order = list(range(int(cached_shape[-1]))) if frame_list is None else [int(i) for i in frame_list]
    re = im = None
    for j, k in enumerate(order):
        vol = read_recon_frames(path, dtype, cached_shape, [k], as_complex=True)[..., 0]
        vol = orient(vol)
        if re is None:
            re = np.empty(vol.shape + (len(order),), dtype=np.float32, order="F")
            im = np.empty_like(re)
        re[..., j] = vol.real
        im[..., j] = vol.imag
    return re, im


def nifti_like(data: np.ndarray, ref):
    """A float32 NIfTI of ``data`` with the affine and header of ``ref`` (the magnitude result)."""
    from nibabel.nifti1 import Nifti1Image

    aff = ref.affine
    img = Nifti1Image(np.asarray(data, dtype=np.float32), aff, header=ref.header.copy())
    img.set_data_dtype(np.float32)
    img.header.set_slope_inter(None, None)
    img.set_qform(aff, 1)
    img.set_sform(aff, 0)
    return img


__all__ = ["EXPORT_KEYS", "EXPORT_MODES", "N_BANDS", "parse_export", "is_on", "check", "bessel_factor",
           "ksigma", "ksigma_document", "frame_rows", "write_tsv", "write_json", "noise_cache_path",
           "read_parts", "nifti_like"]
