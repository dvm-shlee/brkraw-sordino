"""The pre-WI-0056 ("legacy") ramp trajectory, moved out of ``src/`` (BRK-0066).

Development tool, not part of the installed package. The product places every
sample on the integral of the ramping gradient (``traj.py``); this earlier
form is kept only so evaluation tools can still compare against it (stage
S1 in ``eval_stages``, mode ``pre`` in ``eval_ramp``):

    k_ij = s_j * (g(i-1) + (g(i) - g(i-1)) * j / N),  s_j = (j + o) / (2 (N - 1))

with N samples per spoke and o the acquisition delay in samples. It is
wrong by a curvature term of twice the integral form; the last projection
gets no correction (``k = s_j g(i)``).
"""

from typing import Any, Dict, Optional

import numpy as np


def calc_radial_traj3d_legacy(grad_array: np.ndarray, matrix_size: int, over_sampling: float,
                              traj_offset: Optional[float] = None) -> np.ndarray:
    """Legacy trajectory (n_pro, N, 3) for gradient vectors ``grad_array`` (3, n_pro).

    ``traj_offset`` is AcqDelayTotal expressed in samples
    (``AcqDelayTotal_us * 1e-6 * EffBandwidth_Hz * OverSampling``).
    """
    g = np.asarray(grad_array, dtype=float)
    n = int(matrix_size / 2 * over_sampling)
    off = traj_offset or 0
    j = np.arange(n, dtype=float)
    samp = ((j + off) / (n - 1)) / 2.0
    g_cur = g.T
    g_prev = np.roll(g_cur, 1, axis=0)
    correction = (g_cur - g_prev)[:, None, :] / n * j[None, :, None]
    traj = samp[None, :, None] * (g_prev[:, None, :] + correction)
    traj[-1] = samp[:, None] * g_cur[-1][None, :]
    return traj


def legacy_from_recon_info(recon_info: Dict[str, Any]) -> np.ndarray:
    """Legacy trajectory of a scan described by ``recon_info`` (UseOrigin=False scans only)."""
    from brkraw_sordino.traj import calc_radial_grad3d

    if bool(recon_info["UseOrigin"]):
        raise ValueError("the legacy trajectory is defined for UseOrigin=False scans only")
    matrix = int(recon_info["Matrix"][0])
    over_sampling = float(recon_info["OverSampling"])
    grad = calc_radial_grad3d(matrix, int(recon_info["NPro"]), bool(recon_info["HalfAcquisition"]),
                              False, bool(recon_info["Reorder"]))
    offset = float(recon_info["AcqDelayTotal_us"]) * 1e-6 * float(recon_info["EffBandwidth_Hz"]) * over_sampling
    return calc_radial_traj3d_legacy(grad, matrix, over_sampling, offset)


__all__ = ["calc_radial_traj3d_legacy", "legacy_from_recon_info"]
