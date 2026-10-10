"""Sample-based density weights for golden trajectories (WI-0113 run 5, D-0197 decision 2).

The product's weight |k|^2 / max (``serial.density``) assumes straight spokes
spread evenly over the sphere. With the ramp model each spoke bends from the
previous direction; the golden orders make neighbouring directions far apart, and
the Golden Grid ring order bends them unevenly, so |k|^2 leaves direction-dependent
streaks (WI-0113 run 3: synthetic data on scan 17's trajectory, Dice 0.75 and NRMSE
0.30 with |k|^2 against 0.93 and 0.12 with these weights).

Golden trajectories therefore use Pipe-Menon weights of the actual sample positions
(J. G. Pipe and P. Menon, Magn. Reson. Med. 41:179-186, 1999), as mrinufft's
``MRIfinufft.pipe``: w <- w / |interp(spread(w))| with finufft spreading and
interpolation only (``spreadinterponly``, ``upsampfac`` 2, ``spread_kerevalmeth`` 0, the
same tolerance as the reconstruction), ``PIPE_ITER`` iterations from w = 1. The spread
of all samples is a sum, so it is accumulated chunk by chunk on one grid; then every
chunk is interpolated from it and its weights updated. No whole trajectory is built
(D-0133); one weight per sample of one repetition is held (``memguard.dcf_nbytes``).

Cap (WI-0113 run 5, measured): Pipe also *raises* the weight of samples in sparsely covered
k-space, which amplifies their noise. On scan 17 the 12,732 head spokes (Default list) got 14.7
times the per-spoke weight of the golden spokes and the image became noisy (NRMSE to scan 13
0.234, |k|^2 0.244); on scan 11 the image got worse (0.157 against 0.115). So each sample's
weight is capped at its |k|^2 / max weight, the weight of uniform radial sampling at that
radius, and then the whole set is multiplied by one factor lambda >= 1 that restores the sum.
The final weights are lambda x min(Pipe, |k|^2/max): relative to |k|^2 they are clipped at one
common level (lambda: 1.32 for scan 11, 1.91 for scan 17; 25 % and 19 % of the samples sit at
it), so the uncapped tail (outer-shell ratios up to 6.6 and 21.6) is gone, while below the clip
Pipe's relative profile (lower where curved spokes bunch) is kept. The common factor only
scales the brightness. Measured: 17 0.126, 11 0.108 (Dice 0.937, 0.932; ``r5_cap.json``,
``r5_lambda.json``).

Scale: the weights are multiplied so that their sum over the measured samples equals
the sum of |k|^2 / max over the same samples (max also over the virtual samples with
``estimate_k0``, as the product normalises), so the image keeps the brightness of the
|k|^2 image. With ``estimate_k0`` the virtual leading samples are part of the set (the
final adjoint is over virtual and measured samples, as ``kcentre.fill_centre``), and the
K0 solve uses the same weights. One repetition's weights serve every repetition and
frame, so the frames still add up to the repetition image.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Sequence

import finufft
import numpy as np

from . import golden, serial

logger = logging.getLogger(__name__)

#: Iterations and kernel upsampling of the Pipe-Menon estimate (the WI-0113 run-3 probe values).
PIPE_ITER = 10
PIPE_UPSAMPFAC = 2.0
#: The rule as it enters the recon cache key of golden scans (hook._build_cache_params).
RULE: Dict[str, Any] = {"method": "pipe-menon", "iterations": PIPE_ITER, "upsampfac": PIPE_UPSAMPFAC,
                        "scale": "sum of |k|^2/max", "virtual_samples": "included with estimate_k0",
                        "cap": "min(Pipe, |k|^2/max), then the same sum"}


def uses_sample_weights(recon_info: Dict[str, Any]) -> bool:
    """True for golden trajectories (Default keeps |k|^2 / max)."""
    return golden.trajectory_mode(recon_info) != "Default"


class SampleWeights:
    """Density weights of one repetition: ``w`` (n_pro, samples kept), ``w_virtual`` (n_pro, M) or None."""

    def __init__(self, w: np.ndarray, w_virtual: Optional[np.ndarray] = None):
        self.w = np.asarray(w, dtype=np.float64)
        self.w_virtual = None if w_virtual is None else np.asarray(w_virtual, dtype=np.float64)

    def chunk(self, lo: int, hi: int) -> np.ndarray:
        """Flat weights of spokes lo .. hi - 1 (the order of ``rows(lo, hi)[:, ign:]``)."""
        return self.w[lo:hi].reshape(-1)

    def virtual(self, lo: Optional[int] = None, hi: Optional[int] = None) -> Optional[np.ndarray]:
        """Flat weights of the virtual samples of spokes lo .. hi - 1 (all spokes by default)."""
        if self.w_virtual is None:
            return None
        return self.w_virtual[lo:hi].reshape(-1)

    @property
    def nbytes(self) -> int:
        return int(self.w.nbytes + (0 if self.w_virtual is None else self.w_virtual.nbytes))


def _plan(nufft_type: int, shape: Sequence[int]):
    return finufft.Plan(nufft_type, tuple(int(s) for s in shape), n_trans=1, eps=serial.NUFFT_EPS,
                        dtype="complex128", upsampfac=PIPE_UPSAMPFAC, spreadinterponly=1, spread_kerevalmeth=0)


def pipe_weights(rows: Any, n_pro: int, ignore_samples: int, shape: Sequence[int], chunk_spokes: int,
                 virtual_traj: Optional[np.ndarray] = None, n_iter: int = PIPE_ITER,
                 cap: bool = True) -> SampleWeights:
    """Pipe-Menon weights of one repetition's samples, chunk by chunk (see the module text).

    ``cap=False`` returns the uncapped estimate (diagnostics and figures; the hook always caps).

    ``rows(lo, hi)`` (``traj.TrajectoryRows.rows`` or an array's slice) gives the untrimmed
    rows; the first ``ignore_samples`` samples of each spoke are left out, as in the
    reconstruction. ``virtual_traj`` ((n_pro, M, 3), ``kcentre.leading_points``) joins the set.
    """
    from .recon import _row_source

    t0 = time.perf_counter()
    rows = _row_source(rows) if not callable(rows) else rows
    n_pro = int(n_pro)
    ign = int(ignore_samples)
    shape = tuple(int(s) for s in shape)
    ranges = serial.spoke_ranges(n_pro, int(chunk_spokes))
    n_v = 0 if virtual_traj is None else int(virtual_traj.shape[1])
    n_s = int(rows(0, 1).shape[1]) - ign

    def points(lo, hi):
        tr = rows(lo, hi)[:, ign:]
        if n_v:
            tr = np.concatenate([virtual_traj[lo:hi], tr], axis=1)
        return serial.omega_columns(tr)

    w = np.ones((n_pro, n_v + n_s), dtype=np.float64)
    spread, interp = _plan(1, shape), _plan(2, shape)
    for _ in range(int(n_iter)):
        grid = np.zeros(shape, dtype=np.complex128)
        for lo, hi in ranges:
            spread.setpts(*points(lo, hi))
            grid += spread.execute(np.ascontiguousarray(w[lo:hi].reshape(-1), dtype=np.complex128)).reshape(shape)
        for lo, hi in ranges:
            interp.setpts(*points(lo, hi))
            d = interp.execute(grid)
            w[lo:hi] /= np.abs(d).reshape(hi - lo, -1)
        del grid
    del spread, interp
    # scale: sum over the measured samples = sum of |k|^2 / max over them (max also over the virtual ones)
    k2_sum, k2_max = 0.0, 0.0
    for lo, hi in ranges:
        d = serial.density(rows(lo, hi)[:, ign:])
        k2_sum += float(d.sum())
        k2_max = max(k2_max, float(d.max()))
    if n_v:
        k2_max = max(k2_max, float(serial.density(virtual_traj).max()))
    w *= (k2_sum / k2_max) / float(w[:, n_v:].sum())
    # cap: never above the |k|^2 / max weight of the same sample, then the same sum again (module text)
    if cap:
        for lo, hi in ranges:
            d = serial.density(rows(lo, hi)[:, ign:]).reshape(hi - lo, n_s) / k2_max
            np.minimum(w[lo:hi, n_v:], d, out=w[lo:hi, n_v:])
        if n_v:
            dv = serial.density(virtual_traj).reshape(n_pro, n_v) / k2_max
            np.minimum(w[:, :n_v], dv, out=w[:, :n_v])
        w *= (k2_sum / k2_max) / float(w[:, n_v:].sum())
    logger.info("Golden trajectory: sample-based density weights (Pipe-Menon, %s iterations) for %s samples "
                "in %.1f s.", n_iter, n_pro * (n_v + n_s), time.perf_counter() - t0)
    return SampleWeights(w[:, n_v:], w[:, :n_v] if n_v else None)


__all__ = ["PIPE_ITER", "PIPE_UPSAMPFAC", "RULE", "uses_sample_weights", "SampleWeights", "pipe_weights"]
