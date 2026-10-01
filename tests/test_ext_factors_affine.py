"""``ext_factors`` keeps the reconstruction (acquisition) axis order and the affine keeps
every object where it was (WI-0081, D-0104).

``ext_factors[i]`` scales reconstruction axis ``i``: the ``PVM_Matrix`` order, read/phase/slice
of the acquisition (the trajectory columns r, p, s). The reconstruction grid of axis ``i`` has
``int(Matrix[i] * ext_factors[i])`` voxels at the same voxel size, so the field of view grows.
``get_dataobj`` then reorders the axes (``orientation.correct``), so the affine columns follow
the output order, not the reconstruction order.

Where an object lands: the adjoint NUFFT puts the grid centre at index ``N // 2`` for odd and
even ``N`` (``test_nufft_places_the_centre_at_n_floor_half``), so a point at offset ``d`` from
the centre is at voxel ``N // 2 + d``. The affine therefore moves the origin by
``-(N // 2 - N0 // 2)`` voxels along each *output* axis (``N0`` = the ``ext_factors=1`` size).
Two defects were fixed (WI-0080): (1) the shift was computed in reconstruction order but applied
to the output-ordered columns, so a factor other than 1 on index 0 or 1 moved the object when the
orientation swaps axes; (2) the base size was recomputed as ``N / factor`` with centre
``(N - 1) / 2``, a sub-voxel error whenever ``Matrix * factor`` is not a whole number.
"""
import numpy as np
import pytest

from brkraw_sordino import hook
from brkraw_sordino.orientation import correct
from brkraw_sordino.recon import build_recon_cache_path, parse_volume_shape

from test_hook_read import _Fid, _Scan, _recon_info  # noqa: F401

D = np.array([2, -1, 1])   # point offset from the grid centre, reconstruction order

#: GradientOrientation of the approved coronal fixture (pv360-3.1-01 scan 4): the
#: orientation step returns (recon 1, recon 0, recon 2).
R_FIXTURE = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
GEOMETRIES = {
    "coronal_fixture": (R_FIXTURE, "coronal"),
    "axial_identity": (np.eye(3), "axial"),
    "coronal_identity": (np.eye(3), "coronal"),
    "sagittal_identity": (np.eye(3), "sagittal"),
    "axial_swapped": (np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]), "axial"),
}
EXTS = ([1.5, 1, 1], [1, 1.5, 1], [1, 1, 1.5], [1.25, 1.5, 2], [1.1, 1, 1.3], [2, 1, 1.5],
        [1, 1.1, 1], [1.3, 1.3, 1.3], [1.5, 1.5, 1.5], [2, 2, 2])
MATRICES = ((16, 16, 16), (15, 15, 15), (16, 20, 12), (17, 14, 15))


def _rot(axis, deg):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    i, j = [k for k in range(3) if k != axis]
    rot = np.eye(3)
    rot[i, i], rot[i, j], rot[j, i], rot[j, j] = c, -s, s, c
    return rot


def _base_affine(rot=np.eye(3)):
    """Any affine of the ext_factors=1 output (anisotropic voxels, a rotation, an offset)."""
    a = np.eye(4)
    a[:3, :3] = rot @ np.array([[0.4, 0, 0], [0, 0, 0.3], [0, 0.5, 0]])
    a[:3, 3] = [-5.2, 2.8, -4.2]
    return a


def _info(matrix, R, plane):
    return {"Matrix": [float(m) for m in matrix], "GradientOrientation": np.asarray(R, float),
            "SliceOrientation": plane}


def _marker_volume(shape):
    """Reconstruction-order volume with one bright voxel at N // 2 + D (NUFFT convention)."""
    vol = np.zeros(shape)
    vol[tuple(int(n) // 2 + int(d) for n, d in zip(shape, D))] = 1.0
    return vol


def _hook_affine(monkeypatch, tmp_path, info, base_affine, ext):
    monkeypatch.setattr(hook, "get_affine_helper", lambda *a, **k: base_affine.copy())
    monkeypatch.setattr(hook, "_parse_recon_info", lambda scan: dict(info))
    kwargs = {"cache_dir": str(tmp_path)}
    if ext is not None:
        kwargs["ext_factors"] = ext
    return np.asarray(hook.get_affine(_Scan(), 1, **kwargs))


def _world_of_marker(tmp_path, info, affine, ext):
    options = hook._build_options({"ext_factors": ext, "cache_dir": str(tmp_path)}) if ext is not None else None
    shape = parse_volume_shape(info, options) if options else [int(m) for m in info["Matrix"]]
    out = correct(_marker_volume(shape), info)
    peak = np.array(np.unravel_index(np.argmax(out), out.shape), float)
    return (affine @ np.r_[peak, 1.0])[:3], out.shape


def _point_error_mm(monkeypatch, tmp_path, info, base_affine, ext):
    ref, _ = _world_of_marker(tmp_path, info, base_affine, None)
    aff = _hook_affine(monkeypatch, tmp_path, info, base_affine, ext)
    world, shape = _world_of_marker(tmp_path, info, aff, ext)
    return world - ref, aff, shape


@pytest.mark.parametrize("geom", sorted(GEOMETRIES))
@pytest.mark.parametrize("matrix", MATRICES, ids=lambda m: "x".join(map(str, m)))
@pytest.mark.parametrize("ext", EXTS, ids=str)
def test_the_object_stays_in_place(monkeypatch, tmp_path, geom, matrix, ext):
    R, plane = GEOMETRIES[geom]
    err, _, _ = _point_error_mm(monkeypatch, tmp_path, _info(matrix, R, plane), _base_affine(), ext)
    assert np.abs(err).max() < 1e-9, err


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("deg", [20, 30, 44, 46, 60])
@pytest.mark.parametrize("ext", ([1.5, 1, 1], [1, 1.5, 1], [1.25, 1.5, 2], [1.1, 1, 1.3]), ids=str)
def test_oblique_scans_keep_the_object_in_place(monkeypatch, tmp_path, axis, deg, ext):
    """Synthetic oblique scan: the gradient frame and the affine rotated together; the
    orientation permutation switches near 45 degrees, the position must not move either way."""
    rot = _rot(axis, deg)
    info = _info((16, 20, 12), R_FIXTURE @ rot.T, "coronal")
    err, _, _ = _point_error_mm(monkeypatch, tmp_path, info, _base_affine(rot), ext)
    assert np.abs(err).max() < 1e-9, err


def test_index_follows_the_reconstruction_axis_on_the_coronal_fixture(monkeypatch, tmp_path):
    """Each index grows its own reconstruction axis; on the fixture geometry the output order is
    (recon 1, recon 0, recon 2), so index 0 grows output axis 1 and index 1 output axis 0."""
    info = _info((16, 16, 16), R_FIXTURE, "coronal")
    for i, out_axis in ((0, 1), (1, 0), (2, 2)):
        ext = [1.0, 1.0, 1.0]
        ext[i] = 1.5
        _, _, shape = _point_error_mm(monkeypatch, tmp_path, info, _base_affine(), ext)
        expected = [16, 16, 16]
        expected[out_axis] = 24
        assert list(shape) == expected


def test_ext_one_returns_the_brkraw_affine_unchanged(monkeypatch, tmp_path):
    info = _info((16, 20, 12), R_FIXTURE, "coronal")
    base = _base_affine(_rot(2, 30))
    for ext in (None, 1, 1.0, [1, 1, 1]):
        assert np.array_equal(_hook_affine(monkeypatch, tmp_path, info, base, ext), base)


def test_shift_is_whole_voxels_n_floor_half(monkeypatch, tmp_path):
    """Size is a whole number of voxels and the shift is -(N//2 - N0//2) per output axis."""
    info = _info((16, 16, 16), np.eye(3), "axial")
    base = _base_affine()
    for e, n in ((1.1, 17), (1.25, 20), (1.3, 20), (1.5, 24), (2.0, 32)):
        aff = _hook_affine(monkeypatch, tmp_path, info, base, [1, 1, e])
        assert parse_volume_shape(info, hook._build_options({"ext_factors": [1, 1, e], "cache_dir": str(tmp_path)}))[2] == n
        expected = base @ np.r_[0.0, 0.0, -(n // 2 - 16 // 2), 1.0]
        assert np.allclose(aff[:, 3], expected)


def _old_affine(affine, shape, factors):
    """The a14a82f rule, kept here to show where it was already right."""
    factors = np.asarray(factors, float)
    scaled = np.asarray(shape, float)
    origin = (scaled / factors - 1.0) / 2.0 - (scaled - 1.0) / 2.0
    updated = affine.copy()
    updated[:, 3] = updated.dot(origin.tolist() + [1.0])
    return updated


@pytest.mark.parametrize("ext", [1.5, 2.0, [1.5, 1.5, 1.5], [2, 2, 2]], ids=str)
def test_scalar_whole_sizes_keep_the_previous_affine(monkeypatch, tmp_path, ext):
    """Scalar (or three equal) factors with whole sizes on an even isotropic matrix: same affine
    as before the fix, so existing conversions of that kind do not move. (On an odd matrix the
    old (N - 1) / 2 centre was half a voxel off, so those affines do change: the 15x15x15 cases
    of test_the_object_stays_in_place failed before the fix.)"""
    info = _info((16, 16, 16), R_FIXTURE, "coronal")
    base = _base_affine(_rot(0, 20))
    options = hook._build_options({"ext_factors": ext, "cache_dir": str(tmp_path)})
    shape = parse_volume_shape(info, options)
    assert np.allclose(_hook_affine(monkeypatch, tmp_path, info, base, ext),
                       _old_affine(base, shape, options.ext_factors))


#: Recon cache file names and reconstruction shapes measured on a14a82f (before the fix) with
#: the synthetic scan of test_hook_read (Matrix 4x5x6): the fix touches the affine only.
PINNED = {
    "none": ("recon_1cde6edc034b627a79daaefffddca0d9bb0c43ce.bin", [4, 5, 6]),
    "1.5": ("recon_ece117d44d804a04b44169da219007e18625fae8.bin", [6, 7, 9]),
    "2": ("recon_8544b6db3a1bc8f64d000e66778f22d8b47e27c7.bin", [8, 10, 12]),
    "1.5,1,1": ("recon_d3e34b22e9016c565a8b2a54b67f44ebf6f6c557.bin", [6, 5, 6]),
    "1.1": ("recon_e59a7e8a6ca0deb2cef3caff216e5cdd5d0e560d.bin", [4, 5, 6]),
}


@pytest.mark.parametrize("ext,pin", [
    (None, "none"), (1, "none"), (1.0, "none"), ([1, 1, 1], "none"),
    (1.5, "1.5"), ([1.5, 1.5, 1.5], "1.5"), (2, "2"), ([2.0, 2.0, 2.0], "2"), ([1.5, 1, 1], "1.5,1,1"),
    (1.1, "1.1"),
], ids=str)
def test_recon_cache_key_and_shape_are_unchanged(tmp_path, ext, pin):
    kw = {"cache_dir": str(tmp_path)}
    if ext is not None:
        kw["ext_factors"] = ext
    options = hook._build_options(kw)
    params = hook._build_cache_params(_Scan(), None, _Fid(), options, _recon_info(1))
    name, shape = PINNED[pin]
    assert build_recon_cache_path(tmp_path, params).name == name
    assert list(parse_volume_shape(_recon_info(1), options)) == shape


@pytest.mark.parametrize("n", [15, 16, 17, 18])
def test_nufft_places_the_centre_at_n_floor_half(n):
    """The rule the affine relies on: a point at offset d is reconstructed at N // 2 + d."""
    from brkraw_sordino.recon import nufft_adjoint
    rng = np.random.default_rng(81)
    traj = rng.uniform(-0.5, 0.5, size=(20000, 3))
    ksp = np.exp(-1j * ((traj / 0.5 * np.pi) @ D.astype(float)))
    img = np.abs(np.asarray(nufft_adjoint(ksp, traj, (n, n, n))).reshape(n, n, n))
    peak = np.array(np.unravel_index(np.argmax(img), img.shape))
    assert (peak - n // 2).tolist() == D.tolist()
