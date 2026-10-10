"""Golden trajectory generators against the sequence C code (WI-0113, CP0/CP1).

``golden_cref`` is a statement-by-statement copy of ``goldensamp.c`` and
``goldengrid.c`` (sequence ``sordino_260801``); ``brkraw_sordino.golden`` is the
vectorised port used by the reconstruction. They must give the same spokes in
the same order (to floating-point rounding: numpy and the C library may differ
in the last bit of cos/sin), and the invariants of each scheme must hold.
"""
import math

import numpy as np
import pytest

import golden_cref as cref


def _golden():
    from brkraw_sordino import golden
    return golden


def _arr(*cols):
    return np.asarray(cols, dtype=float)


TOL = 1e-12


def test_golden_samples_match_the_c_loop():
    g, theta = _golden().golden_samples(2000)
    g1, g2, g3, th = cref.generate_golden_samples(2000)
    assert g.shape == (3, 2000) and theta.shape == (2000,)
    assert np.abs(g - _arr(g1, g2, g3)).max() <= TOL
    assert np.abs(theta - np.asarray(th)).max() <= TOL
    assert np.allclose(np.linalg.norm(g, axis=0), 1.0, atol=1e-12)


REORDER_CASES = [
    # n_subsets, per_subset, zstack_deg, use_origin, golden_reorder, n_pro
    (6, 40, 22.5, False, True, None),
    (7, 33, 30.0, False, True, None),
    (5, 37, 25.0, False, True, None),      # 180 / 25 is not whole: some spokes are not placed
    (4, 50, 40.0, True, True, None),
    (3, 64, 22.5, True, True, None),
    (12, 160, 22.5, False, True, None),
    (3, 40, 22.5, False, False, 100),      # GoldenReorder=No, NPro (NGoldenSteps) < 3 * 40
    (3, 40, 22.5, True, False, 150),       # NPro > 3 * 40: the tail stays zero
    (3, 40, 22.5, False, False, None),
]


@pytest.mark.parametrize("n_sub,per,zdeg,origin,reorder,n_pro", REORDER_CASES)
def test_reorder_matches_the_c_loop(n_sub, per, zdeg, origin, reorder, n_pro):
    got = _golden().reorder_golden_samples(n_sub, per, zdeg, use_origin=origin,
                                           golden_reorder=reorder, n_pro=n_pro)
    o1, o2, o3, _ = cref.reorder_golden_samples(n_sub, per, zdeg, origin, reorder, n_pro)
    ref = _arr(o1, o2, o3)
    assert got.shape == ref.shape
    assert got.dtype == np.float64
    assert np.abs(got - ref).max() <= TOL


@pytest.mark.parametrize("zdeg,expected", [(22.5, 0), (25.0, 80), (40.0, 511)])
def test_unplaced_spokes(zdeg, expected):
    """Counts measured in WI-0112 (50 subsets x 160) and the C copy agree."""
    assert _golden().unplaced_spokes(50, 160, zdeg) == expected
    assert cref.reorder_golden_samples(50, 160, zdeg, False)[3] == expected


def test_each_subset_holds_its_own_golden_segment():
    """Reordering only permutes the spokes inside a subset (22.5 deg: none left out)."""
    golden = _golden()
    n_sub, per = 9, 160
    out = golden.reorder_golden_samples(n_sub, per, 22.5)
    plain, _ = golden.golden_samples(n_sub * per)
    for s in range(n_sub):
        a = out[:, s * per:(s + 1) * per].T
        b = plain[:, s * per:(s + 1) * per].T
        assert np.array_equal(a[np.lexsort(a.T[::-1])], b[np.lexsort(b.T[::-1])])


def test_unplaced_spokes_leave_zeros_at_the_end_of_their_subset():
    golden = _golden()
    n_sub, per, zdeg = 6, 160, 40.0
    out = golden.reorder_golden_samples(n_sub, per, zdeg)
    plain, _ = golden.golden_samples(n_sub * per)
    total = 0
    for s in range(n_sub):
        block = out[:, s * per:(s + 1) * per]
        zero = ~np.any(block, axis=0)
        k = int(zero.sum())
        total += k
        assert not zero[:per - k].any() and zero[per - k:].all()
        placed = block[:, :per - k].T
        seg = plain[:, s * per:(s + 1) * per].T
        # every placed spoke is one of the subset's own golden spokes
        d = np.abs(placed[:, None, :] - seg[None, :, :]).max(axis=2).min(axis=1)
        assert d.max() == 0.0
    assert total == golden.unplaced_spokes(n_sub, per, zdeg)


def test_use_origin_overwrites_spoke_zero_and_keeps_the_count():
    golden = _golden()
    a = golden.reorder_golden_samples(4, 40, 22.5, use_origin=False)
    b = golden.reorder_golden_samples(4, 40, 22.5, use_origin=True)
    assert a.shape == b.shape == (3, 160)
    assert np.array_equal(b[:, 0], np.zeros(3))
    assert np.array_equal(a[:, 1:], b[:, 1:])


def test_sreag_grid_rings_and_equal_area():
    grid = _golden().sreag_grid(10)
    assert list(grid.n_lon) == [3, 9, 14, 18, 20, 20, 18, 14, 9, 3]
    assert grid.n_cell == 128
    lat = np.asarray(grid.lat_edges)
    assert lat[0] == 90.0 and abs(lat[-1] + 90.0) < 1e-6
    assert np.all(np.diff(lat) < 0)
    # every cell has the area 4 pi / n_cell on the unit sphere
    z = np.sin(np.deg2rad(lat))
    ring_area = 2 * np.pi * (z[:-1] - z[1:])
    assert np.allclose(ring_area / np.asarray(grid.n_lon), 4 * np.pi / 128, rtol=1e-9, atol=0)
    ref = cref.generate_sreag_grid(10)
    assert list(grid.n_lon) == ref["nLon"] and grid.n_cell == ref["nCell"]
    assert np.abs(lat - np.asarray(ref["latEdges"])).max() <= TOL


@pytest.mark.parametrize("n_ring,n_frames,mirror", [(10, 5, True), (10, 4, False), (6, 3, True),
                                                    (1, 3, False), (17, 2, True)])
def test_sreag_trajectory_matches_the_c_loop(n_ring, n_frames, mirror):
    got = _golden().sreag_trajectory(n_ring, n_frames, mirror)
    ref = _arr(*cref.generate_sreag_trajectory(n_ring, n_frames, mirror))
    assert got.shape == ref.shape and got.dtype == np.float64
    assert np.abs(got - ref).max() <= TOL


def test_sreag_mirror_half_is_the_opposite_direction():
    golden = _golden()
    n_cell = golden.sreag_grid(10).n_cell
    g = golden.sreag_trajectory(10, 6, True)
    for f in range(6):
        first = g[:, 2 * f * n_cell:(2 * f + 1) * n_cell]
        second = g[:, (2 * f + 1) * n_cell:(2 * f + 2) * n_cell]
        assert np.array_equal(second, -first)
    # without mirror the frames are the first halves
    assert np.array_equal(golden.sreag_trajectory(10, 6, False)[:, :n_cell], g[:, :n_cell])


def test_sreag_point_lies_in_its_cell():
    golden = _golden()
    grid = golden.sreag_grid(10)
    lat = np.asarray(grid.lat_edges)
    for m in (1, 2, 7, 1125):
        k = golden.sreag_cells(grid, m)
        assert k.shape == (3, grid.n_cell)
        c = 0
        for i, n in enumerate(grid.n_lon):
            z_u, z_l = math.sin(math.radians(lat[i])), math.sin(math.radians(lat[i + 1]))
            for j in range(n):
                z = k[2, c]
                az = math.degrees(math.atan2(k[1, c], k[0, c])) % 360.0
                assert z_l - 1e-12 <= z <= z_u + 1e-12
                lo, hi = 360.0 * j / n, 360.0 * (j + 1) / n
                assert lo - 1e-9 <= az <= hi + 1e-9 or (j == 0 and az > 360.0 - 1e-9)
                c += 1


def test_sreag_cells_frame_m_is_the_frame_in_the_trajectory():
    golden = _golden()
    grid = golden.sreag_grid(10)
    g = golden.sreag_trajectory(10, 4, True)
    n = grid.n_cell
    assert np.array_equal(golden.sreag_cells(grid, 3), g[:, 2 * 2 * n:2 * 2 * n + n])
