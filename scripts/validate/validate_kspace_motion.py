"""Synthetic round-trip test for k-space rigid-body motion correction.

The sign and axis conventions are the part of this that goes wrong silently: a
flipped sign produces an image that looks plausible and is worse than no
correction at all. So the correction is derived here against a known ground
truth before it is allowed anywhere near real data.

Conventions taken from brkraw_sordino.serial:
    omega = 2*pi*traj,  |traj| = 0.5 at Nyquist,  x in voxel indices
    adjoint = finufft type 1, isign +1  ->  img[x] = sum_j c_j exp(+i omega_j . x)
    so the forward model is   S(k_j) = sum_x f[x] exp(-2i*pi* traj_j . x)

With an object displaced by d (voxels) and rotated by R,

    g[x] = f(R^-1 (x - d))
    S'(k) = exp(-2i*pi* k.d) * S(R^T k)

so the correction for a sample measured at nominal k is

    value    *= exp(+2i*pi* k.d)      remove the translation phase
    location  = R^T k                 place the spoke on its true trajectory

For row-vector trajectories (..., 3) the second is ``traj @ R``.

Manual check, not collected by pytest (no data needed):

    python scripts/validate/validate_kspace_motion.py      # exit 0 = passes
"""

import numpy as np

try:
    import finufft
except ImportError:  # pragma: no cover
    raise SystemExit('finufft is required; run with the brkraw-sordino venv')

N = 32                      # matrix
NSPOKE = 4000               # radial spokes
NSAMP = N // 2              # samples per spoke
EPS = 1e-9
rng = np.random.default_rng(0)


def phantom(n):
    """Three off-centre ellipsoids, so the result is sensitive to rotation."""
    g = np.stack(np.meshgrid(*[np.arange(n) - n / 2] * 3, indexing='ij'), -1).astype(float)
    img = np.zeros((n, n, n))
    for c, r, v in (((-4, 2, 0), (7, 4, 5), 1.0),
                    ((5, -3, 2), (3, 3, 6), 0.6),
                    ((0, 6, -4), (4, 2, 2), 0.9)):
        d = ((g - np.array(c)) / np.array(r)) ** 2
        img[d.sum(-1) <= 1] += v
    return img.astype(np.complex128)


def spokes(nspoke, nsamp):
    """Golden-ish 3D radial half-spokes, |traj| up to 0.5."""
    i = np.arange(nspoke) + 0.5
    z = 1 - 2 * i / nspoke
    r = np.sqrt(np.maximum(0.0, 1 - z * z))
    phi = np.pi * (1 + 5 ** 0.5) * i
    direc = np.stack([r * np.cos(phi), r * np.sin(phi), z], -1)
    t = np.linspace(0, 0.5, nsamp, endpoint=False)
    return direc[:, None, :] * t[None, :, None]        # (spoke, sample, 3)


def rot(deg):
    """Rotation matrix from three small angles about x, y, z (degrees)."""
    ax, ay, az = np.radians(deg)
    def R(a, i, j):
        m = np.eye(3); c, s = np.cos(a), np.sin(a)
        m[i, i] = m[j, j] = c; m[i, j] = -s; m[j, i] = s
        return m
    return R(az, 0, 1) @ R(ay, 2, 0) @ R(ax, 1, 2)


def forward(img, traj):
    """S(k) = sum_x f[x] exp(-2i pi traj.x): finufft type 2, isign -1."""
    om = (traj.reshape(-1, 3) * 2 * np.pi).T
    return finufft.nufft3d2(*[np.ascontiguousarray(o) for o in om],
                            np.ascontiguousarray(img), isign=-1, eps=1e-12)


def adjoint(vals, traj, n):
    """Density-compensated adjoint, as the engine does (|k|^2 weights)."""
    om = (traj.reshape(-1, 3) * 2 * np.pi).T
    w = (traj.reshape(-1, 3) ** 2).sum(-1)
    w = w / max(w.max(), EPS)
    out = finufft.nufft3d1(*[np.ascontiguousarray(o) for o in om],
                           np.ascontiguousarray(vals.reshape(-1) * w),
                           (n, n, n), isign=1, eps=1e-12)
    return out


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), EPS))


def main():
    img = phantom(N)
    traj = spokes(NSPOKE, NSAMP)
    ref = adjoint(forward(img, traj), traj, N)      # no motion: the target

    # half the spokes are acquired while the object sits at (d, R)
    moved = rng.random(NSPOKE) < 0.5
    d = np.array([0.9, -0.5, 0.35])                 # voxels
    R = rot([0.8, -1.3, 0.6])                       # degrees

    # simulate: those spokes measure  exp(-2i pi k.d) * S(R^T k)
    k_true = traj @ R                                # R^T k for row vectors
    vals = forward(img, traj).reshape(NSPOKE, NSAMP)
    vals_moved = forward(img, k_true).reshape(NSPOKE, NSAMP)
    phase = np.exp(-2j * np.pi * np.einsum('sij,j->si', traj, d))
    meas = np.where(moved[:, None], vals_moved * phase, vals)

    uncorrected = adjoint(meas, traj, N)

    # correction: undo the phase, and grid on the true trajectory
    corr = meas.copy()
    corr[moved] *= np.exp(+2j * np.pi * np.einsum('sij,j->si', traj[moved], d))
    tr_corr = traj.copy()
    tr_corr[moved] = traj[moved] @ R
    corrected = adjoint(corr, tr_corr, N)

    # translation only, to show each term's share
    only_phase = meas.copy()
    only_phase[moved] *= np.exp(+2j * np.pi * np.einsum('sij,j->si', traj[moved], d))
    phase_only = adjoint(only_phase, traj, N)
    tr_rot = traj.copy(); tr_rot[moved] = traj[moved] @ R
    rot_only = adjoint(meas, tr_rot, N)

    # sign check: the wrong sign must be clearly worse, or the test proves nothing
    wrong = meas.copy()
    wrong[moved] *= np.exp(-2j * np.pi * np.einsum('sij,j->si', traj[moved], d))
    tr_w = traj.copy(); tr_w[moved] = traj[moved] @ R.T

    # Data domain -- the exact test of the conventions. After correction, every
    # sample must equal what the motion-free object gives at the location it is
    # gridded on. This holds to rounding for the right conventions only.
    def data_err(sign, Rg):
        c = meas[moved] * np.exp(sign * 2j * np.pi * np.einsum('sij,j->si', traj[moved], d))
        return rel(c, forward(img, traj[moved] @ Rg).reshape(-1, NSAMP))

    err = data_err(+1, R)
    flips = {'phase sign flipped': data_err(-1, R),
             'rotation transposed': data_err(+1, R.T),
             'both flipped': data_err(-1, R.T)}

    print(f'{NSPOKE} spokes, {moved.sum()} moved by d={d} voxels, R={[0.8, -1.3, 0.6]} deg\n')
    print('  data domain (exact test of the conventions; relative error of the moved samples)')
    print(f'    right conventions      {err:.2e}')
    for name, e in flips.items():
        print(f'    {name:22s} {e:.2e}')

    # Image domain -- informative, not exact: the rotated spokes sample k-space at
    # other locations than the motion-free reference, and the density weights are
    # those of the nominal trajectory, so even a perfect correction leaves a residual.
    print('\n  image domain (relative error against the motion-free image)')
    print(f'    uncorrected            {rel(uncorrected, ref):.4f}')
    print(f'    translation phase only {rel(phase_only, ref):.4f}')
    print(f'    rotation only          {rel(rot_only, ref):.4f}')
    print(f'    both (full correction) {rel(corrected, ref):.4f}')
    print(f'    both flipped           {rel(adjoint(wrong, tr_w, N), ref):.4f}')

    ok = err < 1e-10 and min(flips.values()) > 1e-3 and \
        rel(corrected, ref) < 0.5 * rel(uncorrected, ref)
    print(f'\n  {"PASSES" if ok else "FAILS"}: right conventions exact in the data domain, every '
          f'flip clearly wrong, and the correction more than halves the image error')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
