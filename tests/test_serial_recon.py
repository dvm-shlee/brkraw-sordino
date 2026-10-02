"""Serial (spoke-chunk) reconstruction and Toeplitz K0 (WI-0097, D-0133; design WI-0096).

The reconstruction is a sum over samples, so cutting the spokes into contiguous
chunks and adding the chunk images gives the whole-scan image; the estimate_k0
least-squares solve uses the same normal operator written as a convolution on
a 2N grid (Toeplitz), so it gives the same centre estimate.

Part 1 pins the result against the whole-array reference that the product used
up to 366f5fc (``recon.nufft_adjoint`` and ``kcentre.fill_centre`` on the whole
trajectory, per frame and channel, after the phase factor and the off-resonance
correction). It ran on 366f5fc before the change (``chunk_spokes`` did not exist
then and is dropped by ``_call``) and must keep passing with any chunk size.

Part 2 tests the new pieces: per-range trajectory and phase rows, chunk
independence, the number of samples given to one NUFFT call, the chunk planner
in ``memguard`` and the hook passing the plan to the reconstruction.
"""
import inspect
import io
import math

import numpy as np
import pytest

from brkraw_sordino import kcentre, recon
from brkraw_sordino.hook import _build_options
from brkraw_sordino.recon import nufft_adjoint, phase_correction_factor, recon_dataobj
from brkraw_sordino.traj import calc_npro, get_trajectory

MATRIX, OS = 16, 8.0
SHAPE = [MATRIX] * 3
N_POINTS = int(MATRIX / 2 * OS)
N_PRO = 2 * calc_npro(MATRIX, 1.0)
PLAIN_TOL = 1e-9          # double precision NUFFT, only the summation order differs
K0_IMG_TOL = 1e-6         # the 366f5fc solve runs its NUFFTs in complex64
K0_VAL_TOL = 1e-5


def _info(n_rx=1, n_frames=2, phase=True):
    rng = np.random.default_rng(97)
    o1 = list(rng.uniform(-3000.0, 3000.0, N_PRO)) if phase else [0.0]
    return {
        "Matrix": list(SHAPE), "NPro": N_PRO, "HalfAcquisition": False,
        "UseOrigin": False, "Reorder": False, "OverSampling": OS,
        "AcqDelayTotal_us": 6.75, "EffBandwidth_Hz": 75000.0,
        "ExcPulLength_ms": 0.004, "RampTime_ms": 0.1164, "RampDelay_ms": 0.16,
        "TrigSegmentMode": "Off", "MaximizeRampTime": None, "RFWait_ms": None,
        "O1List_Hz": o1, "NRepetitions": n_frames, "EncNReceivers": n_rx,
        "NPoints": N_POINTS, "FIDDataType": np.dtype("<i4"),
        "GradientOrientation": np.eye(3), "SliceOrientation": "axial",
    }


def _phantom_kspace(traj):
    """Smooth object (real, off-centre ellipsoids) sampled on ``traj`` (all samples)."""
    grid = np.meshgrid(*[np.arange(s) - s / 2 for s in SHAPE], indexing="ij")
    r = np.sqrt(((grid[0] - 0.6) / 5.0) ** 2 + ((grid[1] + 0.4) / 4.2) ** 2 + (grid[2] / 3.5) ** 2)
    img = 0.5 * (1 - np.tanh((r - 1.0) * 3.0)) + 0.3 * (np.sqrt(sum(g ** 2 for g in grid)) < 2.0)
    op = recon.make_nufft_operator(traj.reshape(-1, 3) / 0.5 * np.pi, SHAPE, False)
    return np.asarray(op.op(img.astype(np.complex128))).reshape(traj.shape[:2])


def _fid_frames(info, traj, n_frames, n_rx, factor):
    """int32 FID bytes, layout (2, n_points, rx, n_pro) Fortran per frame. The stored
    values are divided by the phase factor so the product's multiplication restores them."""
    y = _phantom_kspace(traj) * 4.0e4
    if factor is not None:
        y = y / factor
    rng = np.random.default_rng(5)
    frames = []
    for f in range(n_frames):
        chans = []
        for c in range(n_rx):
            gain = (1.0 + 0.4 * f) * (1.0 - 0.3 * c) * np.exp(0.7j * c)
            v = gain * y + 30.0 * (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape))
            chans.append(v)
        k = np.stack(chans, axis=1)                                  # (n_pro, rx, n_points)
        arr = np.stack([np.round(k.real).T, np.round(k.imag).T])     # (2, n_points, rx, n_pro)
        frames.append(arr.astype("<i4").tobytes(order="F"))
    return frames


def _reference(frames, info, options, traj, factor, virtual_traj, offset=0, n_frames=None):
    """The 366f5fc method on whole arrays: per frame and channel, phase factor, trim,
    off-resonance, then nufft_adjoint or kcentre.fill_centre."""
    n_rx = int(info["EncNReceivers"])
    ign = options.ignore_samples or 1
    tr = traj[:, ign:, :]
    freqs = options.offreso_freqs
    imgs, k0s = [], []
    sel = frames[offset:] if n_frames is None else frames[offset:offset + n_frames]
    for raw in sel:
        vol = np.frombuffer(raw, dtype="<i4").reshape((2, N_POINTS, n_rx, N_PRO), order="F")
        k = (vol[0] + 1j * vol[1]).T                                  # (n_pro, rx, n_points)
        if factor is not None:
            k = k * factor[:, None, :]
        k = k[..., ign:]
        chans, kk = [], []
        for c in range(n_rx):
            kc = k[:, c, :]
            if isinstance(freqs, tuple) and len(freqs) > c:
                kc = recon.correct_offreso(kc, freqs[c], eff_bandwidth=info["EffBandwidth_Hz"],
                                           over_sampling=info["OverSampling"])
            if virtual_traj is None:
                chans.append(nufft_adjoint(kc, tr, SHAPE, 1))
            else:
                img, i = kcentre.fill_centre(kc, tr, virtual_traj, SHAPE)
                chans.append(img)
                kk.append(i["k0"])
        imgs.append(np.stack(chans))
        k0s.append(kk)
    return imgs, k0s


def _call(fid, traj, info, out, options, chunk_spokes=None, **kw):
    """recon_dataobj with ``chunk_spokes`` when this version has it (366f5fc does not)."""
    if "chunk_spokes" in inspect.signature(recon_dataobj).parameters:
        kw["chunk_spokes"] = chunk_spokes
    return recon_dataobj(fid, traj, info, out, options, **kw)


def _written(out, n_rx, n_frames):
    data = np.frombuffer(out.getvalue(), dtype=np.complex128)
    per = int(np.prod(SHAPE)) * n_rx
    assert data.size == per * n_frames
    imgs = []
    for f in range(n_frames):
        v = data[f * per:(f + 1) * per].reshape(SHAPE[::-1] + [n_rx])   # recon_vol.T, C order
        imgs.append(v.T)                                                # (rx, x, y, z)
    return imgs


def _rel(a, b):
    return float(np.linalg.norm(np.ravel(a - b)) / np.linalg.norm(np.ravel(b)))


def _setup(tmp_path, n_rx, n_frames, phase=True, k0=False, **opts):
    info = _info(n_rx, n_frames, phase)
    options = _build_options({"cache_dir": str(tmp_path), "estimate_k0": k0, **opts})
    traj = get_trajectory(info, options)
    factor = phase_correction_factor(info, options, N_POINTS)
    assert (factor is not None) == phase
    frames = _fid_frames(info, traj, n_frames, n_rx, factor)
    vt = kcentre.leading_points(info, options.ignore_samples or 1) if k0 else None
    return info, options, traj, factor, frames, vt


#: estimate_k0 cases of part 2 (the Toeplitz solve is stage 2 of WI-0097)
STAGE_K0 = [False]
CHUNKS = [None, N_PRO, math.ceil(N_PRO / 3), 7]
CHUNK_IDS = ["default", "one", "three", "spokes7"]


# ----------------------------------------------------------------------------- part 1
@pytest.mark.parametrize("chunk", CHUNKS, ids=CHUNK_IDS)
@pytest.mark.parametrize("n_rx", [1, 2])
def test_plain_equals_whole_array_reference(tmp_path, n_rx, chunk):
    info, options, traj, factor, frames, _ = _setup(tmp_path, n_rx, 2)
    out = io.BytesIO()
    dtype = _call(io.BytesIO(b"".join(frames)), traj, info, out, options, chunk,
                  phase_factor=factor)
    assert np.dtype(dtype) == np.complex128
    ref, _ = _reference(frames, info, options, traj, factor, None)
    got = _written(out, n_rx, 2)
    for g, r in zip(got, ref):
        assert _rel(g, r) < PLAIN_TOL


@pytest.mark.parametrize("chunk", CHUNKS, ids=CHUNK_IDS)
@pytest.mark.parametrize("n_rx", [1, 2])
def test_estimate_k0_equals_fill_centre(tmp_path, n_rx, chunk):
    info, options, traj, factor, frames, vt = _setup(tmp_path, n_rx, 2, k0=True)
    out, k0s = io.BytesIO(), []
    _call(io.BytesIO(b"".join(frames)), traj, info, out, options, chunk,
          phase_factor=factor, virtual_traj=vt, k0_out=k0s)
    ref, ref_k0 = _reference(frames, info, options, traj, factor, vt)
    got = _written(out, n_rx, 2)
    assert len(k0s) == 2 and all(len(f) == n_rx for f in k0s)
    for g, r in zip(got, ref):
        assert _rel(g, r) < K0_IMG_TOL
    for fk, rk in zip(k0s, ref_k0):
        for a, b in zip(fk, rk):
            assert isinstance(a, complex)
            assert abs(a - b) <= K0_VAL_TOL * abs(b)


@pytest.mark.parametrize("chunk", [None, 11], ids=["default", "spokes11"])
def test_offresonance_per_channel_equals_reference(tmp_path, chunk):
    info, options, traj, factor, frames, _ = _setup(tmp_path, 2, 1, offreso_freqs=[120.0, -75.0])
    out = io.BytesIO()
    _call(io.BytesIO(b"".join(frames)), traj, info, out, options, chunk, phase_factor=factor)
    ref, _ = _reference(frames, info, options, traj, factor, None)
    assert _rel(_written(out, 2, 1)[0], ref[0]) < PLAIN_TOL


@pytest.mark.parametrize("chunk", [None, 13], ids=["default", "spokes13"])
def test_offset_and_num_frames_read_the_right_frames(tmp_path, chunk):
    info, options, traj, factor, frames, _ = _setup(tmp_path, 1, 4, phase=False,
                                                    offset=1, num_frames=2)
    out = io.BytesIO()
    _call(io.BytesIO(b"".join(frames)), traj, info, out, options, chunk, phase_factor=factor)
    ref, _ = _reference(frames, info, options, traj, factor, None, offset=1, n_frames=2)
    for g, r in zip(_written(out, 1, 2), ref):
        assert _rel(g, r) < PLAIN_TOL


@pytest.mark.parametrize("chunk", [None, 17], ids=["default", "spokes17"])
def test_spoketiming_buffer_path_equals_reference(tmp_path, chunk):
    """The spoke-timing cache holds float64 frames from position 0 (override buffer/dtype)."""
    info, options, traj, factor, frames, vt = _setup(tmp_path, 2, 2, k0=True)
    as_f8 = [np.frombuffer(f, dtype="<i4").astype("<f8").tobytes() for f in frames]
    buf = int(2 * N_POINTS * 2 * N_PRO * 8)
    out, k0s = io.BytesIO(), []
    _call(io.BytesIO(b"".join(as_f8)), traj, info, out, options, chunk,
          override_buffer_size=buf, override_dtype=np.dtype("<f8"),
          phase_factor=factor, virtual_traj=vt, k0_out=k0s)
    ref, ref_k0 = _reference(frames, info, options, traj, factor, vt)
    for g, r in zip(_written(out, 2, 2), ref):
        assert _rel(g, r) < K0_IMG_TOL
    for fk, rk in zip(k0s, ref_k0):
        for a, b in zip(fk, rk):
            assert abs(a - b) <= K0_VAL_TOL * abs(b)


# ----------------------------------------------------------------------------- part 2
@pytest.mark.parametrize("ramp", [True, False])
def test_trajectory_rows_equal_the_whole_trajectory(tmp_path, ramp):
    from brkraw_sordino.traj import trajectory_rows

    info = _info()
    options = _build_options({"cache_dir": str(tmp_path / "rows"), "correct_ramptime": ramp})
    rows = trajectory_rows(info, options)
    assert list((tmp_path / "rows").iterdir()) == []                 # nothing written
    full = get_trajectory(info, _build_options({"cache_dir": str(tmp_path / "full"),
                                                "correct_ramptime": ramp}))
    assert rows.n_pro == N_PRO and rows.n_samples == N_POINTS
    for lo, hi in [(0, N_PRO), (0, 1), (5, 40), (N_PRO - 3, N_PRO)]:
        assert np.array_equal(rows.rows(lo, hi), full[lo:hi])


def test_phase_rows_equal_the_whole_factor(tmp_path):
    from brkraw_sordino.recon import phase_correction_rows

    info = _info()
    options = _build_options({"cache_dir": str(tmp_path)})
    full = phase_correction_factor(info, options, N_POINTS)
    rows = phase_correction_rows(info, options, N_POINTS)
    for lo, hi in [(0, N_PRO), (3, 9), (N_PRO - 1, N_PRO)]:
        got = rows[lo:hi]
        assert got.dtype == full.dtype and np.array_equal(got, full[lo:hi])
    assert phase_correction_rows(_info(phase=False), options, N_POINTS) is None


@pytest.mark.parametrize("k0", STAGE_K0)
def test_one_chunk_and_three_chunks_agree(tmp_path, k0):
    from brkraw_sordino.traj import trajectory_rows

    info, options, traj, factor, frames, vt = _setup(tmp_path, 2, 1, k0=k0)
    rows = trajectory_rows(info, options)
    res = []
    for chunk in (N_PRO, math.ceil(N_PRO / 3)):
        out, k0s = io.BytesIO(), []
        recon_dataobj(io.BytesIO(frames[0]), rows, info, out, options, phase_factor=factor,
                      virtual_traj=vt, k0_out=k0s, chunk_spokes=chunk)
        res.append((_written(out, 2, 1)[0], k0s))
    assert _rel(res[1][0], res[0][0]) < 1e-10
    if k0:
        for a, b in zip(res[1][1][0], res[0][1][0]):
            assert abs(a - b) <= 1e-10 * abs(b)


@pytest.mark.parametrize("k0", STAGE_K0)
def test_no_nufft_call_gets_more_samples_than_one_chunk(tmp_path, monkeypatch, k0):
    """Deterministic memory test (WI-0096 test design 2): count the points given to
    every NUFFT plan instead of measuring memory."""
    from brkraw_sordino import serial

    seen = []
    real_plan = serial.finufft.Plan

    class Counting:
        def __init__(self, *a, **kw):
            self.plan = real_plan(*a, **kw)

        def setpts(self, *cols, **kw):
            seen.append(len(cols[0]))
            return self.plan.setpts(*cols, **kw)

        def execute(self, *a, **kw):
            return self.plan.execute(*a, **kw)

    monkeypatch.setattr(serial.finufft, "Plan", Counting)
    info, options, traj, factor, frames, vt = _setup(tmp_path, 2, 2, k0=k0)
    chunk = 29
    out = io.BytesIO()
    recon_dataobj(io.BytesIO(b"".join(frames)), traj, info, out, options, phase_factor=factor,
                  virtual_traj=vt, k0_out=[], chunk_spokes=chunk)
    n_kept = N_POINTS - 1
    n_chunks = math.ceil(N_PRO / chunk)
    # the virtual leading points (estimate_k0: M per spoke, all spokes) are one small
    # fixed set; every other call is a chunk of measured samples
    data_calls = [s for s in seen if s != N_PRO * vt.shape[1]] if k0 else seen
    assert data_calls and max(data_calls) <= chunk * n_kept
    # per frame one call per chunk (all channels share it); with estimate_k0 one more
    # pass for the convolution kernel (once for all frames) and the virtual points per frame
    assert len(data_calls) == (2 + int(k0)) * n_chunks


def test_planner_caps_and_equalises_the_chunks():
    from brkraw_sordino import memguard

    plan = memguard.recon_plan(80876, 640, 2, (160, 160, 160))
    assert plan["chunk_spokes"] * 640 <= memguard.CHUNK_SAMPLES_CAP
    assert plan["n_chunks"] == math.ceil(80876 / plan["chunk_spokes"])
    assert plan["n_chunks"] * plan["chunk_spokes"] - 80876 < plan["n_chunks"]   # equal sizes
    assert plan["fits"] is True
    small = memguard.recon_plan(1000, 64, 1, (32, 32, 32))
    assert small["n_chunks"] == 1 and small["chunk_spokes"] == 1000
    assert memguard.recon_plan(80876, 640, 2, (160, 160, 160)) == plan          # deterministic


@pytest.mark.parametrize("k0", STAGE_K0)
def test_planner_uses_the_budget_and_reports_what_does_not_fit(k0):
    from brkraw_sordino import memguard

    big = memguard.recon_plan(80876, 640, 2, (160, 160, 160), estimate_k0=k0)
    budget = big["recon_nbytes"] - 1
    tighter = memguard.recon_plan(80876, 640, 2, (160, 160, 160), estimate_k0=k0,
                                  budget_nbytes=budget)
    assert tighter["fits"] is True and tighter["recon_nbytes"] <= budget
    assert tighter["chunk_spokes"] < big["chunk_spokes"]
    none = memguard.recon_plan(80876, 640, 2, (160, 160, 160), estimate_k0=k0, budget_nbytes=1)
    assert none["fits"] is False
    assert none["chunk_spokes"] == memguard.MIN_CHUNK_SPOKES
    assert none["recon_nbytes"] > 1
    # the estimate does not grow with the spoke count once the chunk is capped
    a = memguard.recon_plan(80876, 640, 2, (160, 160, 160), estimate_k0=k0)["recon_nbytes"]
    b = memguard.recon_plan(8 * 80876, 640, 2, (160, 160, 160), estimate_k0=k0)["recon_nbytes"]
    assert abs(b - a) <= memguard.chunk_nbytes(1, 640, 2)


@pytest.mark.parametrize("k0", STAGE_K0)
def test_recon_nbytes_is_the_planned_peak(k0):
    from brkraw_sordino import memguard

    plan = memguard.recon_plan(80876, 640, 2, (160, 160, 160), estimate_k0=k0, budget_nbytes=8 << 30)
    assert memguard.recon_nbytes(80876, 640, 2, (160, 160, 160), estimate_k0=k0,
                                 budget_nbytes=8 << 30) == plan["recon_nbytes"]
    assert memguard.recon_nbytes(80876, 640, 2, (160, 160, 160), estimate_k0=k0,
                                 spoketiming_nbytes=plan["recon_nbytes"] + 5) == plan["recon_nbytes"] + 5


class _FidEntry:
    name = "fid-wi0097"

    def __init__(self, data):
        self.data = data

    def open(self):
        return io.BytesIO(self.data)


class _Scan:
    scan_id = 3


def test_hook_plans_the_chunks_from_the_limit_and_the_result_does_not_change(tmp_path, monkeypatch):
    from brkraw_sordino import hook, memguard

    info, options, traj, factor, frames, _ = _setup(tmp_path / "ref", 2, 1)
    monkeypatch.setattr(hook, "_parse_recon_info", lambda scan: dict(info))
    monkeypatch.setattr(hook, "_get_fid_entry", lambda scan: _FidEntry(b"".join(frames)))
    seen = []
    real = hook.recon_dataobj

    def spy(*a, **kw):
        seen.append(kw.get("chunk_spokes"))
        return real(*a, **kw)

    monkeypatch.setattr(hook, "recon_dataobj", spy)
    results, infos = [], []
    for name, gb in (("big", None), ("small", None)):
        kw = {"cache_dir": str(tmp_path / name), "as_complex": True, "split_ch": True}
        if name == "small":
            first = hook.get_dataobj_info(_Scan(), None, **kw)
            read_part = first["peak_nbytes"] - first["recon_nbytes"]
            plan = memguard.recon_plan(N_PRO, N_POINTS, 2, SHAPE)
            # enough for the fixed part and a quarter of the spokes
            gb = (read_part + plan["recon_nbytes"] - 0.7 * plan["chunk_nbytes"]) / memguard.GIB
            kw["max_memory_gb"] = gb
        infos.append(hook.get_dataobj_info(_Scan(), None, **kw))
        results.append(hook.get_dataobj(_Scan(), None, **kw))
    assert infos[0]["recon_chunks"] == 1 and infos[1]["recon_chunks"] > 1
    assert seen == [infos[0]["recon_chunk_spokes"], infos[1]["recon_chunk_spokes"]]
    for a, b in zip(results[0], results[1]):
        assert np.allclose(a, b, rtol=0, atol=1e-10 * np.abs(results[0][0]).max())
