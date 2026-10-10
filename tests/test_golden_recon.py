"""A golden scan through the hook, one repetition per frame (WI-0113 CP2).

With ``frame_spokes="repetition"`` a golden scan is reconstructed as a Default
scan is: one repetition (all NPro spokes) per frame. (CP2 had this as the
default, D-0186; since D-0190 the default is the method subset, see
test_golden_frames.py.) The hook must use the
golden list for the trajectory, the ramp model and the phase rows; the result
equals the whole-array reference (``recon.nufft_adjoint`` after the phase
factor) on the golden trajectory, and differs from a reconstruction that uses
another spoke order.
"""
import io
import logging

import numpy as np
import pytest

from brkraw_sordino import hook
from brkraw_sordino.hook import _build_options
from brkraw_sordino.recon import nufft_adjoint, phase_correction_factor
from brkraw_sordino.traj import get_trajectory, gradient_list

from test_serial_recon import N_POINTS, SHAPE, _fid_frames, _info as _serial_info, _rel

N_SUB, PER = 20, 40          # GoldenSampling, 800 spokes (Default at this matrix: about 800)
OFF = (2100.0, -1300.0, 800.0)


def _golden_info(n_frames=1, mode="GoldenSampling"):
    info = _serial_info(1, n_frames, phase=False)
    if mode == "GoldenSampling":
        info.update(TrajectoryMode=mode, NGoldenSubsets=N_SUB, NGoldenSpokesPerSubset=PER,
                    NGoldenSteps=N_SUB * PER, GoldenReorder=True, ZStackAngleDeg=22.5,
                    NPro=N_SUB * PER)
    else:
        from brkraw_sordino.golden import sreag_grid

        info.update(TrajectoryMode=mode, NGridRing=6, NGridFrames=6, GridMirror=True,
                    NPro=6 * 2 * sreag_grid(6).n_cell)
    grad, _ = gradient_list(info)
    info["O1List_Hz"] = list(OFF[0] * grad[0] + OFF[1] * grad[1] + OFF[2] * grad[2] - 50.0)
    return info


class _Entry:
    name = "fid-wi0113"

    def __init__(self, data):
        self.data = data

    def open(self):
        return io.BytesIO(self.data)


class _Scan:
    scan_id = 11


def _hook_image(tmp_path, monkeypatch, info, fid, **kw):
    monkeypatch.setattr(hook, "_parse_recon_info", lambda scan: dict(info))
    monkeypatch.setattr(hook, "_get_fid_entry", lambda scan: _Entry(fid))
    kw.setdefault("frame_spokes", "repetition")
    real, imag = hook.get_dataobj(_Scan(), None, cache_dir=str(tmp_path), as_complex=True, **kw)
    return real + 1j * imag


@pytest.mark.parametrize("mode", ["GoldenSampling", "GoldenGridSampling"])
def test_golden_scan_equals_the_whole_array_reference(tmp_path, monkeypatch, mode):
    from brkraw_sordino.orientation import correct as correct_orientation

    info = _golden_info(1, mode)
    options = _build_options({"cache_dir": str(tmp_path / "t")})
    traj = get_trajectory(info, options)
    factor = phase_correction_factor(info, options, N_POINTS)
    assert factor is not None
    (fid,) = _fid_frames(info, traj, 1, 1, factor)
    img = _hook_image(tmp_path / "h", monkeypatch, info, fid)[..., 0]
    vol = np.frombuffer(fid, dtype="<i4").reshape((2, N_POINTS, 1, info["NPro"]), order="F")
    k = ((vol[0] + 1j * vol[1]).T[:, 0, :]) * factor
    ref = correct_orientation(nufft_adjoint(k[:, 1:], traj[:, 1:, :], SHAPE, 1), info)
    assert img.shape == ref.shape
    assert _rel(img, ref) < 1e-9


def test_another_spoke_order_gives_another_image_and_a_warning(tmp_path, monkeypatch, caplog):
    info = _golden_info(1)
    options = _build_options({"cache_dir": str(tmp_path / "t")})
    traj = get_trajectory(info, options)
    factor = phase_correction_factor(info, options, N_POINTS)
    (fid,) = _fid_frames(info, traj, 1, 1, factor)
    good = _hook_image(tmp_path / "good", monkeypatch, info, fid)
    wrong = dict(info, GoldenReorder=False)           # plain golden order, same spoke count
    with caplog.at_level(logging.WARNING, logger="brkraw_sordino"):
        bad = _hook_image(tmp_path / "bad", monkeypatch, wrong, fid)
    assert _rel(bad, good) > 0.3
    assert any("ACQ_O1_list" in r.getMessage() for r in caplog.records)


def test_repetition_option_gives_one_repetition_per_frame(tmp_path, monkeypatch):
    info = _golden_info(2)
    options = _build_options({"cache_dir": str(tmp_path / "t")})
    traj = get_trajectory(info, options)
    factor = phase_correction_factor(info, options, N_POINTS)
    fid = b"".join(_fid_frames(info, traj, 2, 1, factor))
    monkeypatch.setattr(hook, "_parse_recon_info", lambda scan: dict(info))
    monkeypatch.setattr(hook, "_get_fid_entry", lambda scan: _Entry(fid))
    got = hook.get_dataobj_info(_Scan(), None, cache_dir=str(tmp_path / "h"), frame_spokes="repetition")
    assert got["frames"] == 2 and got["shape"][-1] == 2
