"""Estimated k-space centre for SORDINO (option ``estimate_k0``, BRK-0066).

The dead time leaves the k-space centre unmeasured: the first kept sample lies
0.5-1.2 k-grid units from k = 0. This module fills it algebraically:

1. Solve the weighted least-squares problem
   ``x = argmin || W^(1/2) (A x - y) ||^2`` (conjugate gradient, ``x0 = 0``,
   ``N_ITER`` iterations) for the measured samples ``y`` on the integral
   trajectory. The image grid is the output volume shape itself
   (``GRID_FACTOR = 1``, that is the FOV, or the extended FOV when
   ``ext_factors`` is not 1); the finite grid is the support constraint that
   determines the centre.
2. Predict the image's k-space values at virtual leading samples: the times
   ``t_first - m * dwell`` (m = 1, 2, ..., t >= 0) before the first kept
   sample, on each spoke's own trajectory, down to the RF centre.
3. Run the product's adjoint NUFFT (``recon.nufft_adjoint``, |k|^2 weight)
   over the predicted and the measured samples.

Only the centre comes from the iterative solution (it converges in a few
iterations); every measured sample enters as in the plain adjoint. K0, the
estimated signal at k = 0, is one value per volume and channel (all spokes
cross k = 0), the least-squares image's forward value there.

The product reconstruction (``recon.recon_dataobj``, WI-0097) computes the same
three steps chunk by chunk: the normal operator ``A^H W A`` is applied as a
convolution on a 2N grid (``serial.ToeplitzKernel``), so no step holds all
samples. ``fill_centre`` stays as the whole-array reference that the tests
(``tests/test_serial_recon.py``) compare it with.

The grid factor and the iteration count are fixed on purpose (BRK-0066: the
option is an on/off switch). Development history and the comparison with
other centre estimates: ``tools/eval_timing_centre.py`` (WI-0058), which keeps
its own copy of this method for the parity test in ``tests/test_kcentre.py``.
"""

from typing import Any, Dict, Sequence, Tuple

import numpy as np

from . import timing as timing_mod
from .ramp import ramp_integral
from .recon import make_nufft_operator, nufft_adjoint
from .traj import gradient_list

#: Conjugate-gradient iterations of the least-squares image.
N_ITER: int = 10
#: The least-squares image covers this many times the FOV per axis.
GRID_FACTOR: int = 1


def density_weight(traj: np.ndarray) -> np.ndarray:
    """The product's density weight |k|^2 / max (``recon.nufft_adjoint``), flat."""
    w = np.square(traj).sum(-1).reshape(-1)
    return w / w.max()


def _operator(traj: np.ndarray, shape: Sequence[int]):
    """finufft operator at k positions ``traj`` (|k| = 0.5 is Nyquist).

    ``recon.make_nufft_operator`` keeps the radian coordinates unchanged even
    when they all lie near the centre (WI-0058 run 2, BRK-0063).
    """
    omega = np.asarray(traj.reshape(-1, 3) / 0.5 * np.pi, dtype=np.float64)
    return make_nufft_operator(omega, tuple(int(s) for s in shape), False)


def least_squares_image(kspace: np.ndarray, traj: np.ndarray, shape: Sequence[int],
                        n_iter: int = N_ITER) -> Tuple[np.ndarray, Any]:
    """Weighted least squares by conjugate gradient on ``A^H W A x = A^H W y``.

    Args:
        kspace: (n_pro, n_samples) measured values (phase and off-resonance
            corrected, dropped samples already removed).
        traj: (n_pro, n_samples, 3) trajectory of ``kspace``.
        shape: image grid (FOV) in voxels; the solve uses ``GRID_FACTOR`` times it.
        n_iter: number of iterations.

    Returns:
        The image on the solve grid (data units) and the residual norm history.
    """
    y = np.asarray(kspace).reshape(-1).astype(np.complex128)
    w = density_weight(traj)
    grid = tuple(int(s) * GRID_FACTOR for s in shape)
    op = _operator(traj, grid)

    def normal(x):
        return op.adj_op((w * op.op(x.astype(np.complex64))).astype(np.complex64)).reshape(x.shape)

    b = op.adj_op((w * y).astype(np.complex64)).reshape(grid).astype(np.complex128)
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
    return x, hist


def leading_points(recon_info: Dict[str, Any], ignore_samples: int = 1,
                   grad: np.ndarray = None) -> np.ndarray:
    """(n_pro, M, 3) k positions of the virtual leading samples.

    Times ``t_first - m * dwell`` (m = 1..M, t >= 0) before the first kept
    sample, on each spoke's own integral-model trajectory (the dropped samples
    and the dead time at the real sample spacing, down to the RF centre).
    Same units and ramp model as ``traj.calc_radial_traj3d_integral``.
    """
    seq = timing_mod.read_timing(recon_info)
    tune = timing_mod.tuning_for(seq.version)
    n = int(int(recon_info["Matrix"][0]) / 2 * float(recon_info["OverSampling"]))
    t_first = timing_mod.sample_times_us(seq, tune, int(ignore_samples) + 1)[int(ignore_samples)]
    ts = []
    m = 1
    while t_first - m * seq.dwell_us >= 0:
        ts.append(t_first - m * seq.dwell_us)
        m += 1
    if not ts:
        raise ValueError("no room for virtual samples before the first kept sample")
    ts.sort()
    win = timing_mod.ramp_window(seq, tune)
    fs = list(ts) if win is None else [ramp_integral(t, win[0], win[1]) for t in ts]
    if grad is None:
        grad, _ = gradient_list(recon_info)     # the scan's TrajectoryMode (WI-0113)
    g_prev = np.roll(grad, 1, axis=1).T
    delta = grad.T - g_prev
    unit = 1.0 / (n - 1) / 2.0
    t = np.asarray(ts, dtype=float) / seq.dwell_us
    f = np.asarray(fs, dtype=float) / seq.dwell_us
    return unit * (t[None, :, None] * g_prev[:, None, :] + f[None, :, None] * delta[:, None, :])


def fill_centre(kspace: np.ndarray, traj: np.ndarray, virtual_traj: np.ndarray,
                shape: Sequence[int], n_iter: int = N_ITER) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Adjoint image over estimated leading samples plus all measured ones.

    Args:
        kspace: (n_pro, n_samples) measured values, as passed to the adjoint.
        traj: (n_pro, n_samples, 3) their trajectory.
        virtual_traj: (n_pro, M, 3) from ``leading_points``.
        shape: image shape (FOV) in voxels.

    Returns:
        The complex image and a dict with ``k0`` (complex, the least-squares
        image's value at k = 0), ``virtual_samples`` (M) and
        ``cg_residual_norms``.
    """
    x, hist = least_squares_image(kspace, traj, shape, n_iter)
    grid = tuple(int(s) * GRID_FACTOR for s in shape)
    x64 = x.astype(np.complex64)
    pred = np.asarray(_operator(virtual_traj, grid).op(x64)).reshape(virtual_traj.shape[:2])
    k0 = complex(np.asarray(_operator(np.zeros((1, 1, 3)), grid).op(x64)).reshape(-1)[0])
    data = np.concatenate([pred, kspace], axis=1)
    tr = np.concatenate([virtual_traj, traj], axis=1)
    img = nufft_adjoint(data, tr, shape, 1)
    return img, {"k0": k0, "virtual_samples": int(virtual_traj.shape[1]), "cg_residual_norms": hist}


__all__ = ["N_ITER", "GRID_FACTOR", "density_weight", "least_squares_image", "leading_points",
           "fill_centre"]
