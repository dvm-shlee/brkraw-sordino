"""Golden trajectories against what the scanner recorded (WI-0113, CP1; local data only).

Runs only when ``BRKRAW_SORDINO_GOLDEN_ZIP`` names the phantom dataset with the
scans 11 (GoldenSampling), 13 (Default), 17 (GoldenGridSampling) and 18 (Default,
matrix 60); skipped otherwise. No FID is read.

* Order: ``ACQ_O1_list[i] = offR*GradR[i] + offP*GradP[i] + offS*GradS[i]``
  (``BaseLevelRelations.c``); a least-squares fit of a constant and the three
  offsets to the computed list must leave a relative residual of rounding size.
* Axes and signs: the online-reconstruction ``traj`` file holds
  ``(i / matrix - 0.5) * (GradR, GradP, GradS)`` per line (``radialTraj3D``); the
  ParaVision reconstruction handles only GoldenSampling correctly, so the
  comparison uses the lines that hold the acquired directions: scan 11 the
  first NPro/2 lines, scan 13 all lines, scan 17 the lines after the default
  directions ParaVision put first (WI-0112).
"""
import os
import re
import zipfile

import numpy as np
import pytest

ZIP = os.environ.get("BRKRAW_SORDINO_GOLDEN_ZIP", "")
pytestmark = pytest.mark.skipif(not (ZIP and os.path.isfile(ZIP)),
                                reason="phantom dataset not available (BRKRAW_SORDINO_GOLDEN_ZIP)")

TOL = 1e-12


@pytest.fixture(scope="module")
def scans():
    import brkraw
    from brkraw_sordino.hook import _parse_recon_info

    loader = brkraw.load(ZIP)
    return {s: _parse_recon_info(loader.get_scan(s)) for s in (11, 13, 17, 18)}


def _member(zf, scan, name):
    pattern = re.compile(rf"(^|/){scan}/{name}$")
    hits = [n for n in zf.namelist() if pattern.search(n)]
    assert len(hits) == 1, hits
    return zf.read(hits[0])


@pytest.mark.parametrize("scan,mode", [(11, "GoldenSampling"), (13, "Default"),
                                       (17, "GoldenGridSampling"), (18, "Default")])
def test_mode_is_read_from_the_method(scans, scan, mode):
    from brkraw_sordino.golden import trajectory_mode
    assert trajectory_mode(scans[scan]) == mode


@pytest.mark.parametrize("scan", [11, 13, 17, 18])
def test_spoke_order_matches_acq_o1_list(scans, scan):
    from brkraw_sordino.traj import gradient_list, o1_order_residual

    info = scans[scan]
    grad, _ = gradient_list(info)
    o1 = np.asarray(info["O1List_Hz"], dtype=float)
    assert grad.shape == (3, int(info["NPro"])) and o1.size == grad.shape[1]
    assert o1_order_residual(o1, grad) <= TOL


@pytest.mark.parametrize("scan,first", [(11, 0), (13, 0), (17, 12732)])
def test_axes_and_signs_match_the_scanner_traj_file(scans, scan, first):
    from brkraw_sordino.traj import gradient_list

    info = scans[scan]
    grad, _ = gradient_list(info)
    mat = int(info["Matrix"][0])
    with zipfile.ZipFile(ZIP) as zf:
        t = np.frombuffer(_member(zf, scan, "traj"), dtype="<f8").reshape(-1, mat, 3)
    n_lines = t.shape[0]
    assert n_lines == int(info["NPro"]) // 2
    samp = np.arange(mat) / mat - 0.5
    worst = 0.0
    for lo in range(first, n_lines, 8192):
        hi = min(lo + 8192, n_lines)
        expected = samp[None, :, None] * grad[:, lo:hi].T[:, None, :]
        worst = max(worst, float(np.abs(t[lo:hi] - expected).max()))
    assert worst <= TOL


def test_golden_grid_head_is_found_from_the_traj_file(scans):
    """WI-0113 run 5 (D-0197 decision 1): scan 17's traj file holds the Default list at its head."""
    h = scans[17]["GoldenGridHead"]
    assert h["applied"] is True and h["n"] == 12732 and h["source"] == "traj file"
    assert h["method"] == "sordino_260801" and h["traj_vs_default"] <= 1e-12
    for s in (11, 13, 18):
        assert "GoldenGridHead" not in scans[s]


@pytest.mark.parametrize("scan", [13, 18])
def test_default_scans_keep_the_default_list(scans, scan):
    from brkraw_sordino.traj import calc_radial_grad3d, gradient_list

    info = scans[scan]
    grad, _ = gradient_list(info)
    ref = calc_radial_grad3d(int(info["Matrix"][0]), int(info["NPro"]), bool(info["HalfAcquisition"]),
                             bool(info["UseOrigin"]), bool(info["Reorder"]))
    assert np.array_equal(grad, ref)
