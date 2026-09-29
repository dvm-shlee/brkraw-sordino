"""Tests for kcentre.py (estimate_k0, BRK-0066): the src S3z against the
independent evaluation-tool implementation (tools/eval_timing_centre.py), and
the reconstruction loop with virtual centre samples."""
import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import eval_timing_centre as etc  # noqa: E402
from brkraw_sordino import kcentre  # noqa: E402
from brkraw_sordino.hook import _build_options  # noqa: E402
from brkraw_sordino.recon import nufft_adjoint, recon_dataobj  # noqa: E402
from brkraw_sordino.traj import calc_npro, get_trajectory  # noqa: E402

MATRIX, OS = 16, 8.0
SHAPE = [MATRIX] * 3


def _info(o1=None):
    npro = 2 * calc_npro(MATRIX, 1.0)
    return {
        "Matrix": SHAPE, "NPro": npro, "HalfAcquisition": False,
        "UseOrigin": False, "Reorder": False, "OverSampling": OS,
        "AcqDelayTotal_us": 6.75, "EffBandwidth_Hz": 75000.0,
        "ExcPulLength_ms": 0.004, "RampTime_ms": 0.1164, "RampDelay_ms": 0.16,
        "TrigSegmentMode": "Off", "MaximizeRampTime": None, "RFWait_ms": None,
        "O1List_Hz": [0.0] if o1 is None else o1,
        "NRepetitions": 2, "EncNReceivers": 1, "NPoints": int(MATRIX / 2 * OS),
    }


def _data(info):
    img = etc.smooth_phantom(SHAPE)
    tr = etc.delayed_trajectory(info, 0.0)
    y = etc._operator(tr, SHAPE).op(img).reshape(tr.shape[:2])
    return img, tr, y


def test_constants_are_the_approved_settings():
    assert kcentre.N_ITER == 10 and kcentre.GRID_FACTOR == 1


def test_leading_points_equal_the_tool():
    info = _info()
    np.testing.assert_allclose(kcentre.leading_points(info, 1), etc.leading_points(info, 0.0, 1),
                               rtol=0, atol=1e-15)
    v = kcentre.leading_points(info, 1)
    assert v.shape[0] == info["NPro"] and v.shape[1] >= 2
    r = np.linalg.norm(v[5], axis=-1)                                # inside the gap, increasing outwards
    assert np.all(np.diff(r) > 0) and r[-1] < np.linalg.norm(etc.delayed_trajectory(info, 0.0)[5, 1])


def test_src_s3z_equals_tool_s3z_on_simulated_data():
    info = _info()
    img, tr, y = _data(info)
    ph = None
    vt_tool = etc.leading_points(info, 0.0, 1)
    tool_img, tool_info = etc.centre_fill_reconstruct(y, tr, SHAPE, 1, ph, vt_tool, n_iter=10, ext=1,
                                                      return_info=True)
    vt = kcentre.leading_points(info, 1)
    src_img, src_info = kcentre.fill_centre(y[:, 1:], tr[:, 1:], vt, SHAPE)
    rel = np.linalg.norm(src_img - tool_img) / np.linalg.norm(tool_img)
    assert rel < 1e-6
    assert abs(src_info["k0"] - tool_info["k0"]) <= 1e-6 * abs(tool_info["k0"])
    assert src_info["virtual_samples"] == tool_info["virtual_samples"]
    true_k0 = complex(np.asarray(etc._operator(np.zeros((1, 1, 3)), SHAPE).op(img)).reshape(-1)[0])
    assert abs(src_info["k0"] - true_k0) < 0.02 * abs(true_k0)      # recovers the dead-time centre


def test_recon_loop_with_virtual_samples_matches_fill_centre(tmp_path):
    info = _info()
    _, tr, y = _data(info)
    options = _build_options({"cache_dir": str(tmp_path), "estimate_k0": True})
    traj = get_trajectory(info, options)
    np.testing.assert_allclose(traj, tr, atol=1e-12)
    n_pro, n_pts = y.shape
    frames = [y, 2.0 * y]
    raw = b"".join(
        np.stack([f.real.T, f.imag.T])[:, :, None, :].astype("<f4").tobytes(order="F") for f in frames)
    info["FIDDataType"] = np.dtype("<f4")
    vt = kcentre.leading_points(info, 1)
    k0s: list = []
    out = io.BytesIO()
    dtype = recon_dataobj(io.BytesIO(raw), traj, info, out, options, virtual_traj=vt, k0_out=k0s)
    assert np.issubdtype(dtype, np.complexfloating) and len(k0s) == 2 and all(len(f) == 1 for f in k0s)
    vol = np.frombuffer(out.getvalue(), dtype=dtype).reshape(2, -1)
    ref0, i0 = kcentre.fill_centre(y[:, 1:].astype(np.complex64), traj[:, 1:], vt, SHAPE)
    got0 = vol[0].reshape(SHAPE[::-1]).T
    assert np.linalg.norm(got0 - ref0) / np.linalg.norm(ref0) < 1e-4
    assert abs(k0s[0][0] - i0["k0"]) < 1e-4 * abs(i0["k0"])
    assert abs(k0s[1][0] - 2 * k0s[0][0]) < 1e-3 * abs(k0s[1][0])    # linear in the data
    # without virtual samples: the plain adjoint, no K0 list
    out2, k0_plain = io.BytesIO(), []
    recon_dataobj(io.BytesIO(raw), traj, info, out2, options, k0_out=k0_plain)
    plain = np.frombuffer(out2.getvalue(), dtype=dtype).reshape(2, -1)[0].reshape(SHAPE[::-1]).T
    ref_plain = nufft_adjoint(y[:, 1:].astype(np.complex64), traj[:, 1:], SHAPE, 1)
    assert np.linalg.norm(plain - ref_plain) / np.linalg.norm(ref_plain) < 1e-5
    assert k0_plain == []
    assert np.linalg.norm(plain - got0) > 1e-3 * np.linalg.norm(plain)   # the fill changes the image


def test_k0_metadata_flags(tmp_path):
    from brkraw_sordino.hook import _recon_metadata

    info = _info()
    plain = _recon_metadata(info, _build_options({"cache_dir": str(tmp_path)}))
    est = _recon_metadata(info, _build_options({"cache_dir": str(tmp_path), "estimate_k0": "true"}))
    assert plain["kspace_gap"]["centre_filled"] is False and plain["k0"] is None
    assert est["kspace_gap"]["centre_filled"] is True


def test_recon_loop_estimates_every_channel(tmp_path):
    info = _info()
    info["EncNReceivers"], info["NRepetitions"] = 2, 1
    _, tr, y = _data(info)
    options = _build_options({"cache_dir": str(tmp_path), "estimate_k0": True})
    traj = get_trajectory(info, options)
    chans = np.stack([y, 0.5 * y], axis=1)                       # (n_pro, 2, n_pts)
    raw = np.stack([chans.real.transpose(2, 1, 0), chans.imag.transpose(2, 1, 0)]).astype("<f4")
    info["FIDDataType"] = np.dtype("<f4")
    vt = kcentre.leading_points(info, 1)
    k0s: list = []
    out = io.BytesIO()
    recon_dataobj(io.BytesIO(raw.tobytes(order="F")), traj, info, out, options, virtual_traj=vt, k0_out=k0s)
    assert len(k0s) == 1 and len(k0s[0]) == 2
    assert abs(k0s[0][1] - 0.5 * k0s[0][0]) < 1e-3 * abs(k0s[0][0])
    n_vox = int(np.prod(SHAPE))
    assert len(out.getvalue()) == 2 * n_vox * np.dtype(np.complex128).itemsize
    single, _ = kcentre.fill_centre(y[:, 1:].astype(np.complex64), traj[:, 1:], vt, SHAPE)
    # written as recon_vol.T in C order: (z, y, x, channel)
    first = np.frombuffer(out.getvalue(), dtype=np.complex128).reshape(SHAPE[::-1] + [2])[..., 0].T
    assert np.linalg.norm(first - single) / np.linalg.norm(single) < 1e-4
