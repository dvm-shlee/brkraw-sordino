"""NUFFT coordinate safeguard in recon (BRK-0063, WI-0057).

mrinufft rescales a trajectory by 2 pi when its largest |omega| is below
about 0.5 rad. ``recon.make_nufft_operator`` must keep the radians it was
given, so an operator built on near-centre points only gives the same
values as the same points inside a full trajectory.
"""
import numpy as np
import pytest

pytest.importorskip("mrinufft")

from brkraw_sordino import recon  # noqa: E402

SHAPE = (16, 16, 16)


def _near_centre_traj(n=40, radius=0.02, seed=3):
    """Points within |k| <= radius (|k| = 0.5 is Nyquist), shape (n, 3)."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = radius * np.sqrt(rng.uniform(0.05, 1.0, size=(n, 1)))
    return d * r


def _far_traj(n=20, seed=4):
    """Points near the Nyquist radius, so a trajectory holding them is never rescaled."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d * 0.49


def test_near_centre_operator_keeps_the_radians():
    omega = _near_centre_traj() / 0.5 * np.pi
    assert np.abs(omega).max() < 0.5          # the case mrinufft would rescale
    dcf = np.ones(len(omega))
    op = recon.make_nufft_operator(omega, SHAPE, dcf)
    np.testing.assert_allclose(np.asarray(op.samples).reshape(omega.shape), omega, rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(op.density), dcf)


def test_full_trajectory_is_unchanged():
    omega = np.concatenate([_near_centre_traj(), _far_traj()]) / 0.5 * np.pi
    op = recon.make_nufft_operator(omega, SHAPE, np.ones(len(omega)))
    np.testing.assert_allclose(np.asarray(op.samples).reshape(omega.shape), omega, rtol=0, atol=1e-6)


def test_near_centre_forward_values_match_the_full_trajectory():
    """op(x) at the near points must not depend on whether far points are in the plan."""
    near = _near_centre_traj() / 0.5 * np.pi
    full = np.concatenate([near, _far_traj() / 0.5 * np.pi])
    rng = np.random.default_rng(5)
    x = (rng.normal(size=SHAPE) + 1j * rng.normal(size=SHAPE)).astype(np.complex128)
    p_near = recon.make_nufft_operator(near, SHAPE, False).op(x)
    p_full = recon.make_nufft_operator(full, SHAPE, False).op(x)[: len(near)]
    np.testing.assert_allclose(p_near, p_full, rtol=1e-5, atol=1e-6 * np.abs(p_full).max())


def test_nufft_adjoint_near_centre_matches_full_trajectory():
    """Public entry: the adjoint over near-centre samples equals the adjoint over
    a full trajectory whose far samples carry zero data (density |k|^2 / max,
    so the two results differ only by the ratio of the maxima)."""
    near = _near_centre_traj()[None]                      # (1, n, 3) like (n_pro, n_samples, 3)
    far = _far_traj()[None]
    full = np.concatenate([near, far], axis=1)
    rng = np.random.default_rng(6)
    y = rng.normal(size=near.shape[1]) + 1j * rng.normal(size=near.shape[1])
    img_near = recon.nufft_adjoint(y, near, SHAPE)
    img_full = recon.nufft_adjoint(np.concatenate([y, np.zeros(far.shape[1])]), full, SHAPE)
    scale = (np.square(full).sum(-1).max()) / (np.square(near).sum(-1).max())
    np.testing.assert_allclose(img_near, img_full * scale, rtol=1e-5,
                               atol=1e-6 * np.abs(img_near).max())


def test_without_the_safeguard_the_backend_rescales():
    """Documents the backend behaviour the safeguard exists for (mrinufft 1.5.1);
    skipped if a later backend no longer rescales."""
    import warnings

    from mrinufft import get_operator

    omega = _near_centre_traj() / 0.5 * np.pi
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        op = get_operator("finufft")(omega, shape=SHAPE, density=False)
    samples = np.asarray(op.samples).reshape(omega.shape)
    if np.allclose(samples, omega, rtol=0, atol=1e-6):
        pytest.skip("this mrinufft version keeps near-centre radians")
    np.testing.assert_allclose(samples, 2 * np.pi * omega, rtol=1e-6)
