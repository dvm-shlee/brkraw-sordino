"""Reconstruction memory (WI-0071, D-0098 2 and 4): garbage is collected after every
frame, and the memory check counts the reconstruction step when no cache exists."""
import gc as gc_mod

import numpy as np
import pytest

from brkraw_sordino import hook, memguard, recon
from brkraw_sordino.memguard import SordinoResourceError
from brkraw_sordino.typing import Options

from test_hook_read import NF, VOL, _Scan, setup  # noqa: F401  (fixture)


def _opts(**kw):
    base = dict(ext_factors=(1.0, 1.0, 1.0), ignore_samples=1, offset=0, num_frames=None,
                correct_spoketiming=False, correct_ramptime=False, offreso_freqs=(),
                mem_limit=0.5, clear_cache=False, split_ch=False, cache_dir=".", as_complex=False)
    base.update(kw)
    return Options(**base)


class _FidFile:
    def __init__(self, frames):
        self.data = b"".join(frames)
        self.pos = 0

    def seek(self, pos, *a):
        self.pos = pos

    def read(self, n):
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out


class _Sink:
    def __init__(self):
        self.chunks = []

    def seek(self, *a):
        return 0

    def write(self, b):
        self.chunks.append(bytes(b))
        return len(b)


def _recon(n_frames, n_rx=1):
    n, n_pro, n_points = 8, 24, 8
    info = {"Matrix": [n, n, n], "NPoints": n_points, "NPro": n_pro, "EncNReceivers": n_rx,
            "FIDDataType": np.dtype("<i4"), "NRepetitions": n_frames}
    rng = np.random.default_rng(4)
    traj = rng.uniform(-0.45, 0.45, size=(n_pro, n_points, 3))
    frames = [rng.integers(-500, 500, size=(2, n_points, n_rx, n_pro)).astype("<i4").tobytes(order="F")
              for _ in range(n_frames)]
    sink = _Sink()
    recon.recon_dataobj(_FidFile(frames), traj, info, sink, _opts())
    return sink.chunks


@pytest.mark.parametrize("n_rx", [1, 2])
def test_garbage_is_collected_after_every_frame(monkeypatch, n_rx):
    calls = []
    real = gc_mod.collect
    monkeypatch.setattr(recon.gc, "collect", lambda *a: calls.append(1) or real(*a))
    chunks = _recon(3, n_rx)
    assert len(calls) == 3 and len(chunks) == 3


def test_collecting_does_not_change_the_values(monkeypatch):
    with_gc = _recon(3)
    monkeypatch.setattr(recon.gc, "collect", lambda *a: 0)
    without = _recon(3)
    for a, b in zip(with_gc, without):
        x = np.frombuffer(a, dtype="<c16")
        y = np.frombuffer(b, dtype="<c16")
        np.testing.assert_allclose(x, y, rtol=1e-9, atol=1e-12 * np.abs(y).max())


def test_estimate_adds_the_reconstruction_when_no_cache_exists(setup):
    state = setup(1, write=False)
    info = hook.get_dataobj_info(_Scan(), None, **state["kwargs"])
    ri = state["info"]
    want = memguard.recon_nbytes(ri["NPro"], ri["NPoints"], 1, list(VOL))
    assert info["cached"] is False
    assert info["recon_nbytes"] == want
    frame_bytes = int(np.prod(VOL)) * 16
    assert info["peak_nbytes"] == info["nbytes"] + 3 * frame_bytes + want


def test_estimate_has_no_reconstruction_share_with_a_cache(setup):
    state = setup(1, write=True)
    info = hook.get_dataobj_info(_Scan(), None, **state["kwargs"])
    assert info["cached"] is True and info["recon_nbytes"] == 0
    assert info["peak_nbytes"] == info["nbytes"] + 3 * int(np.prod(VOL)) * 16


def test_multichannel_estimate_uses_the_receivers(setup):
    state = setup(3, write=False)
    info = hook.get_dataobj_info(_Scan(), None, **state["kwargs"])
    ri = state["info"]
    assert info["recon_nbytes"] == memguard.recon_nbytes(ri["NPro"], ri["NPoints"], 3, list(VOL))


def test_reconstruction_share_alone_can_stop_the_reconstruction(setup, monkeypatch):
    state = setup(1, write=False)
    info = hook.get_dataobj_info(_Scan(), None, **state["kwargs"])
    read_only = info["peak_nbytes"] - info["recon_nbytes"]
    limit_gb = (read_only + 1) / memguard.GIB      # enough for the data, not for the recon

    def fail(*a, **k):
        raise AssertionError("reconstruction started after the memory check failed")

    monkeypatch.setattr(hook, "get_trajectory", fail)
    monkeypatch.setattr(hook, "recon_dataobj", fail)
    with pytest.raises(SordinoResourceError, match="Nothing was reconstructed") as err:
        hook.get_dataobj(_Scan(), None, **state["kwargs"], max_memory_gb=limit_gb)
    assert err.value.kind == "memory"


def test_memory_stop_offers_a_retry_value_that_passes(setup, monkeypatch):
    state = setup(1, write=True)
    info = hook.get_dataobj_info(_Scan(), None, **state["kwargs"])
    with pytest.raises(SordinoResourceError) as err:
        hook.get_dataobj(_Scan(), None, **state["kwargs"], max_memory_gb=1e-6)
    retry = err.value.retry_kwargs
    assert set(retry) == {"max_memory_gb"}
    assert memguard.memory_limit_bytes(retry["max_memory_gb"])["limit_nbytes"] >= info["peak_nbytes"]
    assert hook.get_dataobj(_Scan(), None, **state["kwargs"], **retry) is not None


def _fid_bytes(state):
    ri = state["info"]
    return 2 * ri["NPoints"] * ri["NPro"] * ri["EncNReceivers"] * 4 * NF      # int32 FID, all frames


def _only_the_stage(monkeypatch):
    """Zero the reconstruction-step terms so the spoke-timing stage is what is returned."""
    for name in ("SERIAL_SAMPLE_BYTES", "SERIAL_SAMPLE_RX_BYTES"):
        monkeypatch.setattr(memguard, name, 0)
    monkeypatch.setattr(memguard, "serial_fixed_nbytes", lambda *a: 0)


def test_spoketiming_stage_without_a_limit_is_the_whole_fid(setup, monkeypatch):
    """mem_limit=0: one segment, SPOKETIMING_FACTOR x the FID. The trajectory and the
    phase factor are made per chunk and not held during the stage (WI-0097)."""
    _only_the_stage(monkeypatch)
    state = setup(1, write=False, correct_spoketiming=True, mem_limit=0)
    info = hook.get_dataobj_info(_Scan(), None, **state["kwargs"])
    stage = int(np.ceil(memguard.SPOKETIMING_FACTOR * _fid_bytes(state)))
    assert info["recon_nbytes"] == stage


def test_spoketiming_stage_follows_the_segments(setup, monkeypatch):
    """With mem_limit the stage is one segment (spoketiming.get_num_segment)."""
    from brkraw_sordino import spoketiming
    _only_the_stage(monkeypatch)
    monkeypatch.setattr(spoketiming, "get_num_segment", lambda gb, ri, o: np.array([5, 5]))   # 5 of 10
    state = setup(1, write=False, correct_spoketiming=True, mem_limit=1e-9)
    info = hook.get_dataobj_info(_Scan(), None, **state["kwargs"])
    stage = int(np.ceil(memguard.SPOKETIMING_FACTOR * _fid_bytes(state) * 0.5))
    assert info["recon_nbytes"] == stage


@pytest.mark.parametrize("opts,frames_in_file,scale", [
    ({}, NF, 1.0),                                   # every frame
    ({"num_frames": 2}, 2, 1.0),                     # frames 0-1 read: the file holds at least 2
    ({"offset": 1, "num_frames": 3}, 4, 1.0),        # frames 1-3 read
    ({"offset": 1, "num_frames": 10}, NF, 4 / 10),   # more than there are: scaled as spoketiming does
])
def test_segments_use_the_smallest_possible_fid_file(setup, monkeypatch, opts, frames_in_file, scale):
    from brkraw_sordino import spoketiming
    seen = []
    monkeypatch.setattr(spoketiming, "get_num_segment",
                        lambda gb, ri, o: seen.append(gb) or np.array([ri["NPro"]]))
    state = setup(1, write=False, correct_spoketiming=True, mem_limit=0.5, **opts)
    hook.get_dataobj_info(_Scan(), None, **state["kwargs"])
    ri = state["info"]
    frame = 2 * ri["NPoints"] * ri["NPro"] * 4
    assert seen and seen[-1] == pytest.approx(frame * frames_in_file * scale / memguard.GIB)


def test_the_larger_stage_is_used(setup):
    state = setup(1, write=False, correct_spoketiming=True, mem_limit=0)
    info = hook.get_dataobj_info(_Scan(), None, **state["kwargs"])
    ri = state["info"]
    stage = int(np.ceil(memguard.SPOKETIMING_FACTOR * _fid_bytes(state)))
    assert info["recon_nbytes"] == max(memguard.recon_nbytes(ri["NPro"], ri["NPoints"], 1, list(VOL)), stage)


def test_no_spoketiming_stage_when_off_or_cached(setup):
    state = setup(1, write=False, correct_spoketiming=False, mem_limit=0)
    info = hook.get_dataobj_info(_Scan(), None, **state["kwargs"])
    ri = state["info"]
    assert info["recon_nbytes"] == memguard.recon_nbytes(ri["NPro"], ri["NPoints"], 1, list(VOL))


def test_estimate_k0_adds_its_share():
    """The Toeplitz solve (WI-0097 stage 2) adds a grid-only part: the same for any spoke,
    sample or receiver count; the measured rows are pinned in test_recon_memory_measured."""
    for npro, npts, nrx in [(12800, 64, 1), (25600, 64, 2), (80876, 640, 4)]:
        plain = memguard.recon_nbytes(npro, npts, nrx, (64, 64, 64))
        k0 = memguard.recon_nbytes(npro, npts, nrx, (64, 64, 64), estimate_k0=True)
        assert k0 - plain == memguard.k0_fixed_nbytes((64, 64, 64))
    assert memguard.k0_fixed_nbytes((128, 128, 128)) - memguard.k0_fixed_nbytes((64, 64, 64)) \
        == memguard.K0_VOXEL_BYTES * (128 ** 3 - 64 ** 3)


def test_estimate_k0_option_reaches_the_estimate(setup, monkeypatch):
    monkeypatch.setattr(hook, "_resolve_k0", lambda options, info: options)    # keep it on
    state = setup(1, write=False, estimate_k0=True)
    info = hook.get_dataobj_info(_Scan(), None, **state["kwargs"])
    ri = state["info"]
    assert info["recon_nbytes"] == memguard.recon_nbytes(ri["NPro"], ri["NPoints"], 1, list(VOL),
                                                         estimate_k0=True)


def test_a_stop_before_reconstructing_writes_nothing(setup, tmp_path):
    state = setup(1, write=False)
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(SordinoResourceError) as err:
        hook.get_dataobj(_Scan(), None, **state["kwargs"], max_memory_gb=1e-6)
    assert sorted(p.name for p in tmp_path.iterdir()) == before       # no .partial, no traj file
    retry = err.value.retry_kwargs
    info = hook.get_dataobj_info(_Scan(), None, **state["kwargs"])
    assert memguard.memory_limit_bytes(retry["max_memory_gb"])["limit_nbytes"] >= info["peak_nbytes"]


@pytest.mark.parametrize("peak,want", [(2 * 2 ** 30, 2.0), (2 * 2 ** 30 + 1, 2.1), (1, 0.1)])
def test_retry_value_edges(peak, want):
    info = {"peak_nbytes": peak, "limit_nbytes": 0, "frames": 1, "shape": [1], "dtype": "<f8", "count": 1,
            "limit_source": "x", "cached": True, "disk_nbytes": 0, "disk_free_nbytes": None, "cache_dir": "/"}
    with pytest.raises(SordinoResourceError) as err:
        memguard.check(info)
    assert err.value.retry_kwargs == {"max_memory_gb": want}
    assert memguard.memory_limit_bytes(want)["limit_nbytes"] >= peak


def test_default_limit_is_half_of_the_physical_memory(monkeypatch):
    """D-0098 4: the default stays half of the memory (the option raises it)."""
    monkeypatch.setattr(memguard, "physical_memory_bytes", lambda: 64 * memguard.GIB)
    assert memguard.memory_limit_bytes(None) == {"limit_nbytes": 32 * memguard.GIB,
                                                 "limit_source": "half of physical memory"}
