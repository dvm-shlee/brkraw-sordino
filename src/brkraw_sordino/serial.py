"""Serial (spoke-chunk) SORDINO reconstruction (WI-0097, D-0133; design WI-0096).

The adjoint NUFFT image is a sum over samples,

    img = A^H (w * y) / nf,   w_j = |k_j|^2 / max_all |k|^2,   nf = sqrt(prod(N) * 2^3),

so for contiguous spoke ranges C_1 .. C_K it equals sum_c A_c^H (w_c * y_c) / nf
exactly; only the floating-point summation order changes. Two things must be the
same as in the whole-scan reconstruction: the weight is normalised by the
maximum over all spokes (not per chunk), and each chunk uses the trajectory and
phase rows of its own spokes (``traj.TrajectoryRows``, ``recon.PhaseRows``).
Overlapping or random ranges give nothing for a linear sum (WI-0096).

finufft is called directly (D-0133 2) with the settings the product used through
mrinufft 1.5.1 (``operators/interfaces/finufft.py``): type 1 for the adjoint
(default sign +1), tolerance 1e-6, default upsampling, complex128 for float64
points; mrinufft divided the result by ``norm_factor``. One plan serves every
channel of a chunk (points set once, one execute per channel); a batched plan
(``n_trans`` = channels) was 2.4-2.9 times slower in WI-0096.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import finufft
import numpy as np

#: finufft tolerance, as mrinufft's finufft backend (1.5.1) that the product used up to 366f5fc.
NUFFT_EPS = 1e-6


def norm_factor(shape: Sequence[int]) -> float:
    """mrinufft's operator scale ``sqrt(prod(shape) * 2**ndim)``."""
    return math.sqrt(float(np.prod([int(s) for s in shape])) * 2 ** len(shape))


def spoke_ranges(n_pro: int, chunk_spokes: int) -> List[Tuple[int, int]]:
    """Contiguous ranges of at most ``chunk_spokes`` spokes, as equal in size as possible."""
    n_pro = int(n_pro)
    chunk_spokes = int(chunk_spokes)
    if n_pro <= 0 or chunk_spokes <= 0:
        raise ValueError("spoke_ranges needs n_pro > 0 and chunk_spokes > 0")
    n_chunks = -(-n_pro // chunk_spokes)
    size = -(-n_pro // n_chunks)
    return [(lo, min(lo + size, n_pro)) for lo in range(0, n_pro, size)]


def omega_columns(traj: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Radian coordinates (x, y, z columns) of k positions ``traj`` (|k| = 0.5 is Nyquist);
    the same expression as ``recon.nufft_adjoint``."""
    om = traj.reshape(-1, traj.shape[-1]) / 0.5 * np.pi
    return tuple(np.ascontiguousarray(om[:, i], dtype=np.float64) for i in range(3))


def density(traj: np.ndarray) -> np.ndarray:
    """The product's density weight before normalisation, |k|^2, flat
    (``recon.nufft_adjoint`` writes it as ``sqrt(sum k^2) ** 2``)."""
    return np.sqrt(np.square(traj).sum(-1)).reshape(-1) ** 2


class Adjoint:
    """Type-1 NUFFT plan on ``shape``: points set once per chunk, one execute per channel."""

    def __init__(self, shape: Sequence[int], *, upsampfac: float = None):
        self.shape = tuple(int(s) for s in shape)
        kw = {} if upsampfac is None else {"upsampfac": float(upsampfac)}
        self.plan = finufft.Plan(1, self.shape, n_trans=1, eps=NUFFT_EPS, dtype="complex128", **kw)
        self.n_points = 0
        self._cols = None

    def setpts(self, traj: np.ndarray) -> None:
        self._cols = omega_columns(traj)       # kept alive while the plan uses them
        self.plan.setpts(*self._cols)
        self.n_points = int(self._cols[0].size)

    def add(self, out: np.ndarray, coeffs: np.ndarray) -> None:
        """``out += type1(coeffs)`` at the points set last (raw sum, no ``norm_factor``)."""
        c = np.ascontiguousarray(np.asarray(coeffs).reshape(-1), dtype=np.complex128)
        out += self.plan.execute(c).reshape(self.shape)


__all__ = ["NUFFT_EPS", "norm_factor", "spoke_ranges", "omega_columns", "density", "Adjoint"]
