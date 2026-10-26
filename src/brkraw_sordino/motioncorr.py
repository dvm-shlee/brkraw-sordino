"""Rigid-body motion correction applied to the spokes, in k-space.

SORDINO reads while the gradient is on, so a rigid motion of the head during the
scan shows up two ways in the data of a spoke:

* a **translation** d adds a linear phase along the readout. With the engine's
  convention (``omega = 2*pi*traj``, ``|traj| = 0.5`` at Nyquist, voxel indices
  for x, finufft type 1 with sign +1, so the forward model is
  ``S(k) = sum_x f[x] exp(-2i*pi* k.x)``), a displacement of d voxels multiplies
  the measured sample by ``exp(-2i*pi* k.d)``. Dividing it out is the correction.

* a **rotation** R turns the k-space frame with the object, so a spoke nominally
  along k was really acquired along ``R^T k``. Gridding it at its true location
  is the correction.

Formally, for ``g(x) = f(R^-1 (x - d))``::

    S'(k) = exp(-2i*pi* k.d) * S(R^T k)

so a sample measured at nominal k is corrected by

    value    *= exp(+2i*pi* k.d)
    location  = R^T k            ( = ``traj @ R`` for row-vector trajectories)

Both are verified against a synthetic ground truth in
``scripts/validate/validate_kspace_motion.py``: the corrected samples match the
motion-free object at ``R^T k`` to 7e-17 (rounding), while a flipped phase sign
leaves a 0.25 relative error and a transposed rotation 0.025. That test
exists because a flipped sign produces an image that looks plausible and is
worse than no correction at all.

Unlike censoring, this keeps every spoke.

## Frames

``3dvolreg -1Dfile`` writes ``roll pitch yaw dS dL dP``: degrees counter-clockwise
about the I-S, R-L and A-P axes, then displacements toward Superior, Left and
Posterior in mm. Those are anatomical; the trajectory lives in the voxel frame of
the image the motion was measured on. The conversion goes through RAS using that
image's affine, so it follows the data rather than assuming an orientation.

## What this does not do

* The density weights are computed for the nominal trajectory. Rotating spokes
  changes the local sampling density slightly; at the rotations seen here
  (median 0.12 deg against a 0.38 deg spoke spacing) that is second order, but it
  is an approximation.
* Motion is only known at the rate it was estimated (one value per frame of the
  series ``3dvolreg`` ran on). Movement *within* a frame is not corrected; a
  sliding-window estimate would be needed for that.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Option names owned by this module.
MOTION_KEYS = ("motion_file", "motion_spokes")


def _axis_rotation(angle_rad: np.ndarray, axis: int) -> np.ndarray:
    """(n, 3, 3) rotations by ``angle_rad`` about RAS axis ``axis`` (0=x, 1=y, 2=z)."""
    n = angle_rad.shape[0]
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    out = np.zeros((n, 3, 3))
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    out[:, axis, axis] = 1.0
    out[:, i, i] = c
    out[:, j, j] = c
    out[:, i, j] = -s
    out[:, j, i] = s
    return out


def volreg_to_voxel(params: np.ndarray, affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``3dvolreg`` parameters -> (rotations, displacements) in the voxel frame.

    ``params`` is (n, 6): roll, pitch, yaw (deg), dS, dL, dP (mm).
    Returns ``R`` (n, 3, 3) and ``d`` (n, 3) in voxel units, both in the voxel
    frame of the image ``affine`` belongs to.
    """
    params = np.asarray(params, dtype=np.float64)
    if params.ndim != 2 or params.shape[1] != 6:
        raise ValueError(f"motion parameters must be (n, 6), got {params.shape}")
    roll, pitch, yaw = np.radians(params[:, 0]), np.radians(params[:, 1]), np.radians(params[:, 2])
    dS, dL, dP = params[:, 3], params[:, 4], params[:, 5]

    # anatomical -> RAS: roll about I-S (z), pitch about R-L (x), yaw about A-P (y)
    R_ras = _axis_rotation(roll, 2) @ _axis_rotation(yaw, 1) @ _axis_rotation(pitch, 0)
    d_ras = np.stack([-dL, -dP, dS], axis=1)          # Left/Posterior/Superior -> R/A/S, in mm

    M = np.asarray(affine, dtype=np.float64)[:3, :3]  # voxel -> RAS (rotation x scale)
    Minv = np.linalg.inv(M)
    # a rotation conjugated into the voxel frame; scales cancel for a rigid rotation
    R_vox = np.einsum('ij,njk,kl->nil', Minv, R_ras, M)
    d_vox = d_ras @ Minv.T                            # mm in RAS -> voxel units
    return R_vox, d_vox


def per_spoke(params: np.ndarray, affine: np.ndarray, spokes_per_sample: int,
              n_spokes_total: int) -> Tuple[np.ndarray, np.ndarray]:
    """Expand per-frame motion to one rotation and displacement per spoke.

    The estimate is held constant across the spokes of its own frame -- it was
    measured from exactly those spokes, so interpolating between frame centres
    would imply a time resolution the estimate does not have.
    """
    R, d = volreg_to_voxel(params, affine)
    idx = np.minimum(np.arange(n_spokes_total) // int(spokes_per_sample), len(R) - 1)
    return R[idx], d[idx]


def correct_chunk(k: np.ndarray, traj: np.ndarray, R: Optional[np.ndarray],
                  d: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the correction to one chunk.

    ``k`` is (spokes, receivers, samples), ``traj`` is (spokes, samples, 3),
    ``R`` is (spokes, 3, 3) and ``d`` is (spokes, 3), both in voxel units.
    Returns the corrected ``(k, traj)``.
    """
    if d is not None:
        phase = np.exp(2j * np.pi * np.einsum('smj,sj->sm', traj, d))
        k = k * phase[:, None, :]
    if R is not None:
        traj = np.einsum('smj,sjk->smk', traj, R)     # R^T k for row vectors
    return k, traj


def summarise(params: np.ndarray) -> str:
    """One line for the log: how much motion is being corrected."""
    p = np.asarray(params, dtype=np.float64)
    rot = np.sqrt((np.diff(p[:, :3], axis=0) ** 2).sum(1))
    tra = np.sqrt((np.diff(p[:, 3:], axis=0) ** 2).sum(1))
    return (f"{len(p)} motion samples; frame-to-frame rotation "
            f"median {np.median(rot):.4f} deg (max {rot.max():.4f}), translation "
            f"median {np.median(tra):.4f} mm (max {tra.max():.4f})")


__all__ = ["MOTION_KEYS", "volreg_to_voxel", "per_spoke", "correct_chunk", "summarise"]
