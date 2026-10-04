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


#: Upsampling of the kernel NUFFT on the 2N grid (WI-0096: 1.25 used 4.60 GiB and 10.4 s
#: against 8.39 GiB and 15.3 s for 2.0, with a smaller K0 error).
KERNEL_UPSAMPFAC = 1.25


class ToeplitzKernel:
    """The normal operator A^H W A of the estimate_k0 solve as a convolution (WI-0097 stage 2).

    (A^H W A x)(n) = sum_m T(n - m) x(m) with T(d) = sum_j w_j exp(i omega_j . d) for
    d in [-(N-1), N-1] (A^H: type 1, sign +1; A: type 2, sign -1, as finufft's and
    mrinufft's defaults). T is a type-1 NUFFT of the weights onto the 2N grid, a sum
    over samples, so it is accumulated chunk by chunk and does not depend on the
    channel or the frame. The circulant embedding c = ifftshift(T) (index d mod 2N)
    makes every normal-operator step two FFTs of size 2N per axis; the crop to
    [0, N) uses only |d| <= N - 1. FFT(c) is kept as its real part: c is Hermitian
    except on the d = -N planes, which no cropped output uses, so the real part (the
    FFT of the Hermitian average) gives the same cropped result in half the memory
    (WI-0096 design note 6). References: Wajer and Pruessmann, ISMRM 2001; Fessler et
    al., IEEE TSP 53(9), 2005.
    """

    def __init__(self, shape: Sequence[int]):
        self.shape = tuple(int(s) for s in shape)
        self.big = tuple(2 * s for s in self.shape)
        self._adj = Adjoint(self.big, upsampfac=KERNEL_UPSAMPFAC)
        self._t = np.zeros(self.big, dtype=np.complex128)
        self.chat = None

    def add(self, traj: np.ndarray, weight: np.ndarray) -> None:
        """Add the samples at ``traj`` with real weights ``weight`` (flat)."""
        self._adj.setpts(traj)
        self._adj.add(self._t, weight)

    def finish(self, scale: float = 1.0) -> None:
        """Scale the accumulated kernel and keep the real part of its FFT."""
        import scipy.fft as sfft

        self._adj = None                      # release the NUFFT plan and its grid first
        c = np.fft.ifftshift(self._t)
        self._t = None
        if scale != 1.0:
            c *= scale
        f = sfft.fftn(c, workers=-1, overwrite_x=True)
        del c
        self.chat = np.ascontiguousarray(f.real)
        del f

    def normal(self, x: np.ndarray) -> np.ndarray:
        """A^H W A x (raw sums, no ``norm_factor``) for x on the N grid."""
        import scipy.fft as sfft

        xp = np.zeros(self.big, dtype=np.complex128)
        xp[tuple(slice(0, s) for s in self.shape)] = x
        f = sfft.fftn(xp, workers=-1, overwrite_x=True)
        del xp
        f *= self.chat
        y = sfft.ifftn(f, workers=-1, overwrite_x=True)
        del f
        return np.ascontiguousarray(y[tuple(slice(0, s) for s in self.shape)])


class SampleNormal:
    """The normal operator A^H W A of the estimate_k0 solve at the measured samples (WI-0099).

    The form of ``kcentre.least_squares_image``: a type-2 NUFFT to the samples, the
    weights, a type-1 NUFFT back (raw sums, no ``norm_factor``), in complex128. The
    points of every spoke are collected chunk by chunk in the dmax pass (as the
    Toeplitz kernel is), then both plans are made once and serve every channel and
    frame. Its memory grows with the sample count of a frame, the Toeplitz form's
    with the 2N grid, so ``memguard.k0_method`` picks the smaller (D-0136). Both
    plans use ``KERNEL_UPSAMPFAC`` (fine grid 1.25 per axis instead of 2) at the
    same tolerance, ``NUFFT_EPS``.
    """

    def __init__(self, shape: Sequence[int], n_samples: int):
        self.shape = tuple(int(s) for s in shape)
        n = int(n_samples)
        self._cols = tuple(np.empty(n, dtype=np.float64) for _ in range(3))
        self._w = np.empty(n, dtype=np.float64)
        self._n = 0
        self._t1 = self._t2 = None

    def add(self, traj: np.ndarray, weight: np.ndarray) -> None:
        """Add the samples at ``traj`` with real weights ``weight`` (flat)."""
        cols = omega_columns(traj)
        m = int(cols[0].size)
        lo, hi = self._n, self._n + m
        if hi > self._w.size:
            raise ValueError(f"SampleNormal holds {self._w.size} samples, got {hi}")
        for dst, src in zip(self._cols, cols):
            dst[lo:hi] = src
        self._w[lo:hi] = np.asarray(weight, dtype=np.float64).reshape(-1)
        self._n = hi

    def finish(self, scale: float = 1.0) -> None:
        """Scale the weights and make the two plans on all samples."""
        if self._n != self._w.size:
            raise ValueError(f"SampleNormal expected {self._w.size} samples, got {self._n}")
        if scale != 1.0:
            self._w *= scale
        kw = {"upsampfac": KERNEL_UPSAMPFAC}
        self._t2 = finufft.Plan(2, self.shape, n_trans=1, eps=NUFFT_EPS, dtype="complex128", **kw)
        self._t2.setpts(*self._cols)
        self._t1 = finufft.Plan(1, self.shape, n_trans=1, eps=NUFFT_EPS, dtype="complex128", **kw)
        self._t1.setpts(*self._cols)

    def normal(self, x: np.ndarray) -> np.ndarray:
        """A^H W A x (raw sums, no ``norm_factor``) for x on the N grid."""
        y = self._t2.execute(np.ascontiguousarray(x, dtype=np.complex128))
        y *= self._w
        return self._t1.execute(y).reshape(self.shape)


def conjugate_gradient(normal, b: np.ndarray, n_iter: int):
    """``kcentre.least_squares_image``'s CG loop (x0 = 0, same stop rule), complex128."""
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs = np.vdot(r, r).real
    hist = [float(np.sqrt(rs))]
    for _ in range(int(n_iter)):
        ap = normal(p)
        alpha = rs / max(np.vdot(p, ap).real, 1e-300)
        x += alpha * p
        r -= alpha * ap
        del ap
        rs_new = np.vdot(r, r).real
        hist.append(float(np.sqrt(rs_new)))
        if rs_new <= 1e-24 * hist[0] ** 2:
            break
        p *= rs_new / rs
        p += r
        rs = rs_new
    return x, hist


def forward(x: np.ndarray, traj: np.ndarray) -> np.ndarray:
    """Type-2 NUFFT of ``x`` (image grid) at ``traj`` (raw, no ``norm_factor``), flat."""
    cols = omega_columns(traj)
    plan = finufft.Plan(2, tuple(x.shape), n_trans=1, eps=NUFFT_EPS, dtype="complex128")
    plan.setpts(*cols)
    out = plan.execute(np.ascontiguousarray(x, dtype=np.complex128))
    del plan
    return out


__all__ = ["NUFFT_EPS", "KERNEL_UPSAMPFAC", "norm_factor", "spoke_ranges", "omega_columns",
           "density", "Adjoint", "ToeplitzKernel", "SampleNormal", "conjugate_gradient", "forward"]
