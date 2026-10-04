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

Part 3 (WI-0099, D-0136, D-0143): estimate_k0 solves with the Toeplitz form, or with
the NUFFT pair at the samples (``serial.SampleNormal``) when that halves the memory
estimate; part 1 runs with both, and the two agree.
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
#: K0 is one number per frame and channel, the sum of the least-squares image: the
#: complex64 rounding of the 366f5fc solve (relative 6e-8 per value) accumulates over
#: the 10 CG steps and the sum over all voxels, so one value is held to 10 x the image
#: tolerance (gate 1 note, wi-0097-choi-2). Measured at 160^3: 3e-8 relative.
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


def _call(fid, traj, info, out, options, chunk_spokes=None, k0_method=None, **kw):
    """recon_dataobj with ``chunk_spokes`` and ``k0_method`` when this version has them
    (366f5fc has neither, 858de87 has no ``k0_method``)."""
    params = inspect.signature(recon_dataobj).parameters
    if "chunk_spokes" in params:
        kw["chunk_spokes"] = chunk_spokes
    if "k0_method" in params:
        kw["k0_method"] = k0_method
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
STAGE_K0 = [False, True]
#: the two estimate_k0 solves (WI-0099); the test geometry picks Toeplitz by itself (D-0143 rule)
K0_METHODS = ["toeplitz", "samples"]
#: plain, then estimate_k0 with each solve
K0_CASES = [pytest.param(False, None, id="plain"), pytest.param(True, "toeplitz", id="k0-toeplitz"),
            pytest.param(True, "samples", id="k0-samples")]
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


@pytest.mark.parametrize("method", K0_METHODS)
@pytest.mark.parametrize("chunk", CHUNKS, ids=CHUNK_IDS)
@pytest.mark.parametrize("n_rx", [1, 2])
def test_estimate_k0_equals_fill_centre(tmp_path, n_rx, chunk, method):
    info, options, traj, factor, frames, vt = _setup(tmp_path, n_rx, 2, k0=True)
    out, k0s = io.BytesIO(), []
    _call(io.BytesIO(b"".join(frames)), traj, info, out, options, chunk, method,
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


@pytest.mark.parametrize("method", K0_METHODS)
@pytest.mark.parametrize("chunk", [None, 17], ids=["default", "spokes17"])
def test_spoketiming_buffer_path_equals_reference(tmp_path, chunk, method):
    """The spoke-timing cache holds float64 frames from position 0 (override buffer/dtype)."""
    info, options, traj, factor, frames, vt = _setup(tmp_path, 2, 2, k0=True)
    as_f8 = [np.frombuffer(f, dtype="<i4").astype("<f8").tobytes() for f in frames]
    buf = int(2 * N_POINTS * 2 * N_PRO * 8)
    out, k0s = io.BytesIO(), []
    _call(io.BytesIO(b"".join(as_f8)), traj, info, out, options, chunk, method,
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


@pytest.mark.parametrize("k0,method", K0_CASES)
def test_one_chunk_and_three_chunks_agree(tmp_path, k0, method):
    from brkraw_sordino.traj import trajectory_rows

    info, options, traj, factor, frames, vt = _setup(tmp_path, 2, 1, k0=k0)
    rows = trajectory_rows(info, options)
    res = []
    for chunk in (N_PRO, math.ceil(N_PRO / 3)):
        out, k0s = io.BytesIO(), []
        recon_dataobj(io.BytesIO(frames[0]), rows, info, out, options, phase_factor=factor,
                      virtual_traj=vt, k0_out=k0s, chunk_spokes=chunk, k0_method=method)
        res.append((_written(out, 2, 1)[0], k0s))
    assert _rel(res[1][0], res[0][0]) < 1e-10
    if k0:
        for a, b in zip(res[1][1][0], res[0][1][0]):
            assert abs(a - b) <= 1e-10 * abs(b)


@pytest.mark.parametrize("k0,method", K0_CASES)
def test_no_nufft_call_gets_more_samples_than_one_chunk(tmp_path, monkeypatch, k0, method):
    """Deterministic memory test (WI-0096 test design 2): count the points given to
    every NUFFT plan instead of measuring memory. The sample-based K0 solve (WI-0099)
    holds all samples of a frame in its two plans by design (its estimate counts them,
    ``memguard.k0_samples_nbytes``); every other call stays within one chunk."""
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
                  virtual_traj=vt, k0_out=[], chunk_spokes=chunk, k0_method=method)
    n_kept = N_POINTS - 1
    n_chunks = math.ceil(N_PRO / chunk)
    if method == "samples":
        whole = [s for s in seen if s == N_PRO * n_kept]
        assert len(whole) == 2                 # type 2 and type 1, made once for all frames
        seen = [s for s in seen if s != N_PRO * n_kept]
    # the virtual leading points (estimate_k0: M per spoke, all spokes) are one small
    # fixed set; every other call is a chunk of measured samples
    data_calls = [s for s in seen if s != N_PRO * vt.shape[1]] if k0 else seen
    assert data_calls and max(data_calls) <= chunk * n_kept
    # per frame one call per chunk (all channels share it); with the Toeplitz solve one
    # more pass for the convolution kernel (once for all frames; the sample-based solve
    # copies the chunk points instead) and the virtual points per frame
    assert len(data_calls) == (2 + int(method == "toeplitz")) * n_chunks


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


# ----------------------------------------------------------------------------- part 3
#: The two estimate_k0 solves (WI-0099, acceptance 4): every NUFFT runs at tolerance 1e-6
#: (``serial.NUFFT_EPS``), so the image is held to that and one K0 value to 10 x it, as
#: K0_VAL_TOL against fill_centre. Observed: 4e-10 (image) and 2e-9 (K0) on this geometry;
#: 2e-8 to 6e-8 and 2e-7 at 32^3-128^3 and 2e-11 and 1.2e-6 at 160^3 (WI-0099 bench).
METHODS_IMG_TOL = 1e-6
METHODS_K0_TOL = 1e-5


@pytest.mark.parametrize("chunk", [None, 7], ids=["default", "spokes7"])
@pytest.mark.parametrize("phase", [True, False])
def test_the_two_k0_solves_agree(tmp_path, chunk, phase):
    info, options, traj, factor, frames, vt = _setup(tmp_path, 2, 2, phase=phase, k0=True)
    got = {}
    for method in K0_METHODS:
        out, k0s = io.BytesIO(), []
        recon_dataobj(io.BytesIO(b"".join(frames)), traj, info, out, options, phase_factor=factor,
                      virtual_traj=vt, k0_out=k0s, chunk_spokes=chunk, k0_method=method)
        got[method] = (_written(out, 2, 2), k0s)
    for a, b in zip(got["samples"][0], got["toeplitz"][0]):
        assert _rel(a, b) < METHODS_IMG_TOL
    for fa, fb in zip(got["samples"][1], got["toeplitz"][1]):
        for a, b in zip(fa, fb):
            assert abs(a - b) <= METHODS_K0_TOL * abs(b)


#: (n_pro, n_points, n_rx, grid, solve) under the D-0143 rule; the measured shapes of the
#: WI-0099 bench (estimate ratio samples / Toeplitz in the comment)
RULE_CASES = [
    (80876, 640, 2, 160, "toeplitz"),   # 160^3 fixture geometry (WI-0095 scan, D-0135), 1.05
    (12800, 64, 1, 128, "samples"),     # WI-0097 decision 2 case, 0.32
    (3200, 32, 1, 32, "samples"),       # 0.50
    (12800, 64, 1, 64, "toeplitz"),     # sample solve smaller (0.60) but 4 x slower
    (28796, 96, 4, 96, "toeplitz"),     # 0.72
    (51128, 128, 1, 128, "toeplitz"),   # 0.63
]


@pytest.mark.parametrize("n_pro,n_points,n_rx,n,solve", RULE_CASES)
def test_k0_method_takes_samples_only_when_they_halve_the_estimate(n_pro, n_points, n_rx, n, solve):
    from brkraw_sordino import memguard

    vol = (n, n, n)
    c = memguard.k0_method(n_pro, n_points, n_rx, vol)
    assert c["method"] == solve
    plain = memguard.recon_plan(n_pro, n_points, n_rx, vol)
    assert c["samples_total_nbytes"] == plain["recon_nbytes"] + memguard.k0_samples_nbytes(n_pro, n_points, vol)
    assert c["toeplitz_total_nbytes"] == plain["recon_nbytes"] + memguard.k0_fixed_nbytes(vol)
    assert (c["samples_total_nbytes"] <= memguard.K0_SAMPLES_MAX_FRACTION * c["toeplitz_total_nbytes"]) \
        == (solve == "samples")
    assert c["k0_nbytes"] == (c["samples_nbytes"] if solve == "samples" else c["toeplitz_nbytes"])
    # the planner adds the chosen term; the budget does not change the choice
    for budget in (None, 1, 64 << 30):
        with_k0 = memguard.recon_plan(n_pro, n_points, n_rx, vol, estimate_k0=True, budget_nbytes=budget)
        assert with_k0["k0_method"] == solve
        if budget is None:
            assert with_k0["fixed_nbytes"] - plain["fixed_nbytes"] == c["k0_nbytes"]
    assert plain["k0_method"] is None


def test_k0_method_half_is_inclusive(monkeypatch):
    from brkraw_sordino import memguard

    ratio = memguard.k0_method(12800, 64, 1, (64, 64, 64))["ratio"]
    monkeypatch.setattr(memguard, "K0_SAMPLES_MAX_FRACTION", ratio)
    assert memguard.k0_method(12800, 64, 1, (64, 64, 64))["method"] == "samples"
    monkeypatch.setattr(memguard, "K0_SAMPLES_MAX_FRACTION", ratio * (1 - 1e-9))
    assert memguard.k0_method(12800, 64, 1, (64, 64, 64))["method"] == "toeplitz"


@pytest.mark.parametrize("chosen", K0_METHODS)
def test_recon_uses_and_logs_the_chosen_solve(tmp_path, monkeypatch, caplog, chosen):
    import logging

    from brkraw_sordino import memguard, serial

    real = memguard.k0_method

    def pick(*a):
        res = dict(real(*a))
        res["method"] = chosen
        return res

    made = []
    for cls in ("ToeplitzKernel", "SampleNormal"):
        orig = getattr(serial, cls)

        def spy(*a, _orig=orig, _name=cls, **kw):
            made.append(_name)
            return _orig(*a, **kw)

        monkeypatch.setattr(serial, cls, spy)
    monkeypatch.setattr(memguard, "k0_method", pick)
    info, options, traj, factor, frames, vt = _setup(tmp_path, 1, 1, k0=True)
    with caplog.at_level(logging.INFO, logger="brkraw_sordino.recon"):
        recon_dataobj(io.BytesIO(frames[0]), traj, info, io.BytesIO(), options, phase_factor=factor,
                      virtual_traj=vt, k0_out=[])
    assert made == ["ToeplitzKernel" if chosen == "toeplitz" else "SampleNormal"]
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("estimate_k0:")]
    assert len(lines) == 1
    assert ("sample-based" in lines[0]) == (chosen == "samples")
    assert "(forced)" not in lines[0] and "GiB" in lines[0]


def test_unknown_k0_method_is_refused(tmp_path):
    info, options, traj, factor, frames, vt = _setup(tmp_path, 1, 1, k0=True)
    with pytest.raises(ValueError, match="k0_method"):
        recon_dataobj(io.BytesIO(frames[0]), traj, info, io.BytesIO(), options, phase_factor=factor,
                      virtual_traj=vt, k0_out=[], k0_method="cg")
