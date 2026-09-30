"""get_dataobj reads the recon cache frame by frame, honours frames/axis, reports its size
and checks memory and disk first (WI-0071; D-0097 1)."""
import io
import json
import logging

import numpy as np
import pytest

from brkraw_sordino import cacheio, hook, memguard
from brkraw_sordino.memguard import SordinoResourceError
from brkraw_sordino.orientation import correct as correct_orientation

VOL = (4, 5, 6)
NF = 5


class _Fid:
    name = "fid"


class _Scan:
    scan_id = 7


def _recon_info(n_rx):
    return {
        "Matrix": list(VOL), "NPoints": 8, "NPro": 10, "EncNReceivers": n_rx,
        "FIDDataType": np.dtype("<i4"), "NRepetitions": NF,
        "GradientOrientation": np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
        "SliceOrientation": "coronal",
    }


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """A scan whose recon cache exists (synthetic complex128 values)."""
    state = {}

    def make(n_rx=1, write=True, **opts):
        info = _recon_info(n_rx)
        monkeypatch.setattr(hook, "_parse_recon_info", lambda scan: dict(info))
        monkeypatch.setattr(hook, "_get_fid_entry", lambda scan: _Fid())
        # the synthetic cache is written under the key of the options the test uses
        # (as_complex/split_ch left the key in D-0098 3; test_recon_cache_key.py)
        kwargs = {"cache_dir": str(tmp_path), **opts}
        plan = hook._plan(_Scan(), None, dict(kwargs))
        shape = ([n_rx] if n_rx > 1 else []) + list(VOL) + [NF]
        rng = np.random.default_rng(5)
        data = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype("<c16")
        if write:
            data.flatten(order="F").tofile(plan["img_cache_path"])
            meta = {"dtype": "<c16", "shape": shape, "kspace_gap": None, "k0": None}
            hook._cache_meta_path(plan["img_cache_path"]).write_text(json.dumps(meta))
        state.update(info=info, data=data, shape=shape, kwargs=kwargs, plan=plan)
        return state

    return make


def _reference(state, *, as_complex=False, split_ch=False):
    """The pre-WI-0071 method: whole file, abs, channel combination, orientation."""
    d = state["data"]
    info = state["info"]
    if not as_complex:
        d = np.abs(d)
    multi = d.ndim == 5
    if multi and not split_ch:
        d = np.sum(d, axis=0) if as_complex else np.sqrt(np.sum(d ** 2, axis=0))
        multi = False
    arrays = list(d) if multi else [d]
    if as_complex:
        return [f(correct_orientation(a, info)) for a in arrays for f in (np.real, np.imag)]
    return [correct_orientation(a, info) for a in arrays]


def _as_list(out):
    return list(out) if isinstance(out, tuple) else [out]


def _get(state, **kw):
    return _as_list(hook.get_dataobj(_Scan(), None, **state["kwargs"], **kw))


@pytest.mark.parametrize("n_rx", [1, 3])
@pytest.mark.parametrize("as_complex", [False, True])
@pytest.mark.parametrize("split_ch", [False, True])
def test_values_equal_the_whole_file_method(setup, n_rx, as_complex, split_ch):
    state = setup(n_rx, as_complex=as_complex, split_ch=split_ch)
    got = _get(state)
    ref = _reference(state, as_complex=as_complex, split_ch=split_ch)
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        assert g.shape == r.shape and g.dtype == r.dtype
        assert np.array_equal(g, r)


@pytest.mark.parametrize("frames,index", [
    (2, 2), ([3, 1], [3, 1]), ("1:4", slice(1, 4)), ("::2", slice(None, None, 2)), (-1, -1),
])
@pytest.mark.parametrize("axis", [None, 3, "cycle"])
def test_frames_follow_brkraw_rules(setup, frames, index, axis):
    state = setup(1)
    (full,) = _reference(state)
    (got,) = _get(state, frames=frames, axis=axis)
    assert np.array_equal(got, full[..., index])


def test_frames_on_multichannel_split_output(setup):
    state = setup(3, split_ch=True)
    ref = _reference(state, split_ch=True)
    got = _get(state, frames=[4, 0])
    assert len(got) == 3
    for g, r in zip(got, ref):
        assert np.array_equal(g, r[..., [4, 0]])


def test_frame_selection_errors(setup):
    state = setup(1)
    with pytest.raises(ValueError, match="axis needs frames"):
        _get(state, axis=3)
    with pytest.raises(ValueError, match="not the frame axis"):
        _get(state, axis=0, frames=1)
    with pytest.raises(ValueError, match="sordino frames"):
        _get(state, frames=NF)
    with pytest.raises(ValueError, match="sordino frames"):
        _get(state, frames=[1, 1])


def test_one_frame_reads_one_frame(setup, monkeypatch):
    state = setup(1)
    counted = {"bytes": 0}

    class Counting(io.FileIO):
        def readinto(self, b):
            n = super().readinto(b)
            counted["bytes"] += n or 0
            return n

        def read(self, *a):
            data = super().read(*a)
            counted["bytes"] += len(data)
            return data

    monkeypatch.setattr(cacheio, "open", lambda p, mode="rb": Counting(str(p), "r"), raising=False)
    _get(state, frames=3)
    assert counted["bytes"] == int(np.prod(VOL)) * 16


def test_read_options_do_not_change_the_recon_cache_key_or_warn(setup, caplog):
    state = setup(1)
    with caplog.at_level(logging.WARNING, logger="brkraw_sordino.hook"):
        plan = hook._plan(_Scan(), None, dict(state["kwargs"], frames=[1], axis=3, max_memory_gb=2))
    assert plan["img_cache_path"] == state["plan"]["img_cache_path"]
    assert not [r for r in caplog.records if "unknown option" in r.getMessage()]


@pytest.mark.parametrize("opts,kw", [
    ({}, {}), ({}, {"frames": 2}), ({}, {"frames": [0, 3]}), ({"as_complex": True}, {}),
    ({"split_ch": True}, {}), ({"split_ch": True, "as_complex": True}, {"frames": "0:2"}),
])
@pytest.mark.parametrize("n_rx", [1, 3])
def test_info_matches_what_is_returned(setup, opts, kw, n_rx):
    state = setup(n_rx, **opts)
    info = hook.get_dataobj_info(_Scan(), None, **state["kwargs"], **kw)
    got = _get(state, **kw)
    assert info["cached"] is True
    assert info["count"] == len(got)
    assert all(list(g.shape) == info["shape"] for g in got)
    assert all(g.dtype.str == info["dtype"] for g in got)
    assert info["nbytes"] == sum(g.nbytes for g in got)
    assert info["peak_nbytes"] >= info["nbytes"]
    assert info["cache_nbytes"] == int(np.prod(state["shape"])) * 16


def test_memory_limit_stops_before_reading(setup, monkeypatch):
    state = setup(1)
    one_frame = hook.get_dataobj_info(_Scan(), None, **state["kwargs"], frames=0)
    every = hook.get_dataobj_info(_Scan(), None, **state["kwargs"])
    limit_gb = (one_frame["peak_nbytes"] + 1) / memguard.GIB
    assert every["peak_nbytes"] > one_frame["peak_nbytes"] + 1

    def fail(*a, **k):
        raise AssertionError("read after the memory check failed")

    (got,) = _get(state, frames=0, max_memory_gb=limit_gb)   # under the limit: read
    assert got.shape == tuple(one_frame["shape"])
    monkeypatch.setattr(cacheio, "read_recon_frames", fail)
    with pytest.raises(SordinoResourceError) as err:
        _get(state, max_memory_gb=limit_gb)
    assert err.value.kind == "memory"
    assert "max_memory_gb" in str(err.value) and "Nothing was read" in str(err.value)
    assert err.value.info["peak_nbytes"] == every["peak_nbytes"]


def test_memory_limit_stops_before_reconstructing(setup, monkeypatch):
    state = setup(1, write=False)
    info = hook.get_dataobj_info(_Scan(), None, **state["kwargs"])
    assert info["cached"] is False and info["cache_dtype"] == "<c16"

    def fail(*a, **k):
        raise AssertionError("reconstruction started after the memory check failed")

    monkeypatch.setattr(hook, "get_trajectory", fail)
    monkeypatch.setattr(hook, "recon_dataobj", fail)
    with pytest.raises(SordinoResourceError, match="Nothing was reconstructed") as err:
        _get(state, max_memory_gb=1e-6)
    assert err.value.kind == "memory"


def test_disk_check_before_reconstructing(setup, monkeypatch):
    state = setup(1, write=False)
    monkeypatch.setattr(memguard, "free_disk_bytes", lambda path: 10)

    def fail(*a, **k):
        raise AssertionError("reconstruction started after the disk check failed")

    monkeypatch.setattr(hook, "get_trajectory", fail)
    with pytest.raises(SordinoResourceError) as err:
        _get(state)
    assert err.value.kind == "disk"
    # a valid cache needs no disk: the same free space does not stop a read
    state = setup(1, write=True)
    assert _get(state)


def test_default_limit_is_half_the_memory(monkeypatch):
    monkeypatch.setattr(memguard, "physical_memory_bytes", lambda: 8 * memguard.GIB)
    got = memguard.memory_limit_bytes(None)
    assert got["limit_nbytes"] == 4 * memguard.GIB
    assert got["limit_source"] == "half of physical memory"
    monkeypatch.setattr(memguard, "physical_memory_bytes", lambda: None)
    assert memguard.memory_limit_bytes(None)["limit_nbytes"] == 4 * memguard.GIB
    assert memguard.memory_limit_bytes("16")["limit_nbytes"] == 16 * memguard.GIB
    for bad in ("lots", 0, -1):
        with pytest.raises(ValueError, match="max_memory_gb"):
            memguard.memory_limit_bytes(bad)


def test_physical_memory_is_read_on_this_computer():
    ram = memguard.physical_memory_bytes()
    assert ram is None or ram > 256 * 1024 ** 2


def test_resource_error_is_a_memory_error():
    assert issubclass(SordinoResourceError, MemoryError)


def test_int_frame_on_split_complex_multichannel(setup):
    state = setup(3, split_ch=True, as_complex=True)
    ref = _reference(state, as_complex=True, split_ch=True)
    got = _get(state, frames=1)
    assert len(got) == 6
    for g, r in zip(got, ref):
        assert np.array_equal(g, r[..., 1])


def test_negative_out_of_range_frame(setup):
    state = setup(1)
    with pytest.raises(ValueError, match="sordino frames"):
        _get(state, frames=-(NF + 1))


def test_info_before_a_cache_exists(setup):
    state = setup(1, write=False)
    info = hook.get_dataobj_info(_Scan(), None, **state["kwargs"], frames=[0, 2])
    assert info["cached"] is False
    assert info["shape"] == _oriented(state) + [2]
    assert info["nbytes"] == int(np.prod(VOL)) * 2 * 8
    assert info["disk_nbytes"] == int(np.prod(VOL)) * NF * 16


def _oriented(state):
    return list(correct_orientation(np.zeros(VOL), state["info"]).shape)


def test_the_reconstruction_writes_the_assumed_cache_dtype():
    """get_dataobj_info assumes RECON_CACHE_DTYPE before a cache exists (review F1)."""
    from brkraw_sordino.recon import nufft_adjoint

    rng = np.random.default_rng(2)
    traj = rng.uniform(-0.45, 0.45, size=(6, 8, 3))
    fid = rng.integers(-100, 100, size=(2, 8, 6)).astype("<i4")   # int FID as in recon_dataobj
    k = (fid[0] + 1j * fid[1]).T
    img = nufft_adjoint(k, traj, [8, 8, 8])
    assert np.dtype(img.dtype) == hook.RECON_CACHE_DTYPE
