"""A FID shorter than the parameters say (acquisition stopped early, WI-0109).

The hook measures the FID before it reconstructs. When the file holds fewer
complete frames than ``PVM_NRepetitions``, it warns once (bytes found, bytes the
parameters need, frames kept) and reconstructs the complete frames; the bytes of
the incomplete last frame are not used. The result equals the complete scan read
with ``num_frames`` set to that count, with and without spoke-timing correction and
``estimate_k0``. No complete frame, or an ``offset`` at or after the last complete
frame, stops with a reason. ``allow_short_fid=false`` restores the stop (before
anything is reconstructed). A complete FID keeps its recon cache key.

"Equals" is within the tolerances of test_serial_recon.py: two identical complete
runs already differ by up to 1.7e-10 (relative to the maximum; the NUFFT is not
bit-reproducible from run to run, WI-0109 probe), so no bit equality is asked.
"""
import io
import json
import logging
import zipfile

import numpy as np
import pytest

from brkraw_sordino import hook
from test_serial_recon import K0_IMG_TOL, K0_VAL_TOL, PLAIN_TOL, _rel, _setup

PLANNED = 4


class _FidEntry:
    name = "fid-wi0109"

    def __init__(self, data):
        self.data = data

    def open(self):
        return io.BytesIO(self.data)


class _NoSize:
    """A FID entry whose size cannot be read (as the cache-only read tests use)."""
    name = "fid-wi0109"            # the same name as _FidEntry: the name is in the cache key


class _Scan:
    scan_id = 9


def _patch(monkeypatch, info, entry):
    monkeypatch.setattr(hook, "_parse_recon_info", lambda scan: dict(info))
    monkeypatch.setattr(hook, "_get_fid_entry", lambda scan: entry)
    monkeypatch.setattr(hook, "_resolve_k0", lambda options, recon_info: options)


def _get(tmp, **kw):
    out = hook.get_dataobj(_Scan(), None, cache_dir=str(tmp), as_complex=True, split_ch=True, **kw)
    return list(out) if isinstance(out, tuple) else [out]


def _short(frames, n_complete, extra):
    """The first ``n_complete`` frames and ``extra`` bytes of the next one."""
    return b"".join(frames[:n_complete]) + frames[n_complete][:extra]


def _info(tmp_path, n_rx=1, phase=True, k0=False, n_frames=PLANNED, **opts):
    info, _, _, _, frames, _ = _setup(tmp_path / "setup", n_rx, n_frames, phase=phase, k0=k0, **opts)
    info["RepetitionTime_ms"] = 4.0           # spoke-timing correction needs the TR
    return info, frames


def _same(a, b, tol=PLAIN_TOL):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.shape == y.shape
        assert _rel(x, y) < tol


@pytest.mark.parametrize("n_rx", [1, 2])
def test_short_fid_gives_the_complete_frames_and_warns(tmp_path, monkeypatch, caplog, n_rx):
    info, frames = _info(tmp_path, n_rx)
    frame_nbytes = len(frames[0])
    data = _short(frames, 2, frame_nbytes // 3)
    _patch(monkeypatch, info, _FidEntry(b"".join(frames)))
    ref = _get(tmp_path / "full", num_frames=2)
    _patch(monkeypatch, info, _FidEntry(data))
    with caplog.at_level(logging.WARNING, logger="brkraw_sordino.hook"):
        got = _get(tmp_path / "short")
    _same(got, ref)
    assert got[0].shape[-1] == 2
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    short = [m for m in msgs if "FID" in m and "short" in m]
    assert len(short) == 1, msgs
    text = short[0]
    for number in (len(data), PLANNED * frame_nbytes, PLANNED * frame_nbytes - len(data),
                   frame_nbytes // 3):
        assert f"{number:,}" in text, text
    assert "2 of 4" in text
    assert text.startswith("sordino: the FID is short")
    assert "allow_short_fid=false" in text
    assert all(len(line) < 100 for line in text.splitlines()), text     # short lines


@pytest.mark.parametrize("n_complete", [2, 3])
def test_short_fid_with_spoketiming_equals_the_complete_scan_cut(tmp_path, monkeypatch, n_complete):
    info, frames = _info(tmp_path, 2, correct_spoketiming=True)
    data = _short(frames, n_complete, 100)
    _patch(monkeypatch, info, _FidEntry(b"".join(frames)))
    ref = _get(tmp_path / "full", num_frames=n_complete, correct_spoketiming=True)
    _patch(monkeypatch, info, _FidEntry(data))
    got = _get(tmp_path / "short", correct_spoketiming=True)
    _same(got, ref)


def test_short_fid_with_one_frame_says_spoketiming_is_skipped(tmp_path, monkeypatch, caplog):
    info, frames = _info(tmp_path, 1)
    _patch(monkeypatch, info, _FidEntry(b"".join(frames)))
    ref = _get(tmp_path / "full", num_frames=1)          # one frame: no spoke-timing step
    _patch(monkeypatch, info, _FidEntry(_short(frames, 1, 64)))
    with caplog.at_level(logging.WARNING, logger="brkraw_sordino.hook"):
        got = _get(tmp_path / "short", correct_spoketiming=True)
    _same(got, ref)
    assert any("spoke-timing" in r.getMessage() for r in caplog.records)


def test_short_fid_with_estimate_k0_keeps_one_k0_per_complete_frame(tmp_path, monkeypatch):
    info, frames = _info(tmp_path, 1, k0=True, n_frames=3)
    kw = {"estimate_k0": True, "max_memory_gb": 64}
    _patch(monkeypatch, info, _FidEntry(b"".join(frames)))
    full_scan = _Scan()
    ref = hook.get_dataobj(full_scan, None, cache_dir=str(tmp_path / "full"), num_frames=2, **kw)
    _patch(monkeypatch, info, _FidEntry(_short(frames, 2, 8)))
    scan = _Scan()
    got = hook.get_dataobj(scan, None, cache_dir=str(tmp_path / "short"), **kw)
    _same([got], [ref], K0_IMG_TOL)
    k0, ref_k0 = scan._sordino_recon_meta["k0"], full_scan._sordino_recon_meta["k0"]
    assert len(k0) == 2 and len(ref_k0) == 2
    for frame, ref_frame in zip(k0, ref_k0):
        for a, b in zip(frame, ref_frame):
            assert abs(complex(*a) - complex(*b)) <= K0_VAL_TOL * abs(complex(*b))


def test_offset_and_num_frames_count_within_the_complete_frames(tmp_path, monkeypatch):
    info, frames = _info(tmp_path, 1, phase=False)
    _patch(monkeypatch, info, _FidEntry(b"".join(frames)))
    ref = _get(tmp_path / "full", offset=1, num_frames=2)
    _patch(monkeypatch, info, _FidEntry(_short(frames, 3, 10)))
    got = _get(tmp_path / "short", offset=1, num_frames=5)
    _same(got, ref)


def test_offset_at_or_after_the_last_complete_frame_stops_with_the_reason(tmp_path, monkeypatch):
    info, frames = _info(tmp_path, 1)
    _patch(monkeypatch, info, _FidEntry(_short(frames, 2, 10)))
    with pytest.raises(ValueError, match="offset 2"):
        hook.get_dataobj(_Scan(), None, cache_dir=str(tmp_path / "c"), offset=2)


def test_no_complete_frame_stops_before_reconstructing(tmp_path, monkeypatch):
    info, frames = _info(tmp_path, 1)
    _patch(monkeypatch, info, _FidEntry(frames[0][:-4]))

    def fail(*a, **k):
        raise AssertionError("reconstructed")

    monkeypatch.setattr(hook, "recon_dataobj", fail)
    with pytest.raises(ValueError, match="no complete frame"):
        hook.get_dataobj(_Scan(), None, cache_dir=str(tmp_path / "c"))
    with pytest.raises(ValueError, match="no complete frame"):
        hook.get_dataobj_info(_Scan(), None, cache_dir=str(tmp_path / "c"))


def test_allow_short_fid_false_stops_before_reconstructing_even_with_a_cache(tmp_path, monkeypatch):
    info, frames = _info(tmp_path, 1)
    _patch(monkeypatch, info, _FidEntry(_short(frames, 2, 10)))
    cache = str(tmp_path / "c")
    hook.get_dataobj(_Scan(), None, cache_dir=cache)          # the short result is cached
    real = hook.recon_dataobj

    def fail(*a, **k):
        raise AssertionError("reconstructed")

    monkeypatch.setattr(hook, "recon_dataobj", fail)
    for value in (False, "false", "no"):
        with pytest.raises(ValueError, match="allow_short_fid") as err:
            hook.get_dataobj(_Scan(), None, cache_dir=cache, allow_short_fid=value)
        assert "2 of 4" in str(err.value)
    monkeypatch.setattr(hook, "recon_dataobj", real)
    # true (the default) and a complete FID with false both pass
    hook.get_dataobj(_Scan(), None, cache_dir=cache, allow_short_fid=True)
    _patch(monkeypatch, info, _FidEntry(b"".join(frames)))
    out = hook.get_dataobj(_Scan(), None, cache_dir=str(tmp_path / "full"), allow_short_fid=False)
    assert out.shape[-1] == PLANNED


def test_allow_short_fid_is_known_and_not_in_the_cache_key(tmp_path, monkeypatch, caplog):
    info, frames = _info(tmp_path, 1)
    _patch(monkeypatch, info, _FidEntry(_short(frames, 2, 10)))
    kw = {"cache_dir": str(tmp_path / "c")}
    with caplog.at_level(logging.WARNING, logger="brkraw_sordino.hook"):
        a = hook._plan(_Scan(), None, dict(kw))
        b = hook._plan(_Scan(), None, dict(kw, allow_short_fid=True))
    assert a["img_cache_path"] == b["img_cache_path"]
    assert not [r for r in caplog.records if "unknown option" in r.getMessage()]


def test_a_complete_fid_keeps_its_recon_cache_key_and_does_not_warn(tmp_path, monkeypatch, caplog):
    info, frames = _info(tmp_path, 1)
    kw = {"cache_dir": str(tmp_path / "c")}
    _patch(monkeypatch, info, _NoSize())
    before = hook._plan(_Scan(), None, dict(kw))["img_cache_path"]
    longer = b"".join(frames) + b"\0" * 12                   # trailing bytes are not a short FID
    for data in (b"".join(frames), longer):
        _patch(monkeypatch, info, _FidEntry(data))
        with caplog.at_level(logging.WARNING, logger="brkraw_sordino.hook"):
            plan = hook._plan(_Scan(), None, dict(kw))
        assert plan["img_cache_path"] == before
        assert plan["recon_info"]["NRepetitions"] == PLANNED
        assert set(plan["recon_info"]) == set(info)
    assert not [r for r in caplog.records if "FID" in r.getMessage()]


def test_info_reports_planned_and_short_bytes(tmp_path, monkeypatch):
    info, frames = _info(tmp_path, 1)
    frame_nbytes = len(frames[0])
    kw = {"cache_dir": str(tmp_path / "c")}
    _patch(monkeypatch, info, _FidEntry(_short(frames, 3, 50)))
    got = hook.get_dataobj_info(_Scan(), None, **kw)
    assert got["frames_planned"] == PLANNED
    assert got["frames_reconstructed"] == 3 and got["frames"] == 3 and got["shape"][-1] == 3
    assert got["fid_short_nbytes"] == PLANNED * frame_nbytes - (3 * frame_nbytes + 50)
    _patch(monkeypatch, info, _FidEntry(b"".join(frames)))
    full = hook.get_dataobj_info(_Scan(), None, **kw)
    assert full["frames_planned"] == PLANNED and full["fid_short_nbytes"] == 0
    _patch(monkeypatch, info, _NoSize())
    unknown = hook.get_dataobj_info(_Scan(), None, **kw)
    assert unknown["frames_planned"] == PLANNED and unknown["fid_short_nbytes"] is None


def test_short_fid_facts_are_kept_with_the_result(tmp_path, monkeypatch):
    info, frames = _info(tmp_path, 1)
    frame_nbytes = len(frames[0])
    data = _short(frames, 3, 50)
    _patch(monkeypatch, info, _FidEntry(data))
    scan = _Scan()
    hook.get_dataobj(scan, None, cache_dir=str(tmp_path / "c"))
    facts = scan._sordino_recon_meta["short_fid"]
    assert facts == {"fid_nbytes": len(data), "expected_nbytes": PLANNED * frame_nbytes,
                     "frame_nbytes": frame_nbytes, "frames_planned": PLANNED, "frames_complete": 3}
    plan = hook._plan(_Scan(), None, {"cache_dir": str(tmp_path / "c")})
    meta = json.loads(hook._cache_meta_path(plan["img_cache_path"]).read_text())
    assert meta["short_fid"] == facts
    again = _Scan()                                           # read from the cache: same facts
    hook.get_dataobj(again, None, cache_dir=str(tmp_path / "c"))
    assert again._sordino_recon_meta["short_fid"] == facts
    _patch(monkeypatch, info, _FidEntry(b"".join(frames)))
    whole = _Scan()
    hook.get_dataobj(whole, None, cache_dir=str(tmp_path / "w"))
    assert whole._sordino_recon_meta["short_fid"] is None


def test_zip_member_size_is_read_from_the_archive_without_opening_it(tmp_path, monkeypatch):
    from brkraw.core.zip import ZippedFile

    path = tmp_path / "scan.zip"
    payload = bytes(range(256)) * 37
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("1/fid", payload)
    with zipfile.ZipFile(path) as zf:
        entry = ZippedFile(name="fid", arcname="1/fid", zipobj=zf)

        def fail(self):
            raise AssertionError("opened")

        monkeypatch.setattr(ZippedFile, "open", fail)
        assert hook._fid_nbytes(entry) == len(payload)
    monkeypatch.undo()
    with zipfile.ZipFile(path) as zf:                          # an entry that opens a zip stream

        class _Stream:
            name = "fid"

            def open(self):
                handle = zf.open("1/fid")
                handle.seek = None                             # seeking would decompress
                return handle

        assert hook._fid_nbytes(_Stream()) == len(payload)
    assert hook._fid_nbytes(_FidEntry(b"x" * 1234)) == 1234
    assert hook._fid_nbytes(_NoSize()) is None
    assert hook._fid_nbytes(None) is None


@pytest.mark.parametrize("change", [{"NRepetitions": None}, {"NPro": 0}, {"NPoints": None}])
def test_parameters_without_frame_size_or_count_leave_the_fid_alone(tmp_path, change):
    info, frames = _info(tmp_path, 1)
    info.update(change)
    options = hook._build_options({"cache_dir": str(tmp_path / "c")})
    before = dict(info)
    got = hook._check_fid_size(_Scan(), info, _FidEntry(b""), 10, options, True)
    assert got is None
    assert set(info) == set(before) and info["NRepetitions"] == before["NRepetitions"]


def test_one_scan_object_warns_once_and_a_new_one_warns_again(tmp_path, monkeypatch, caplog):
    """The viewer asks get_dataobj_info, then get_dataobj, on the same scan: one warning."""
    info, frames = _info(tmp_path, 1)
    _patch(monkeypatch, info, _FidEntry(_short(frames, 2, 10)))
    kw = {"cache_dir": str(tmp_path / "c")}

    def count():
        return len([r for r in caplog.records if "the FID is short" in r.getMessage()])

    with caplog.at_level(logging.WARNING, logger="brkraw_sordino.hook"):
        scan = _Scan()
        hook.get_dataobj_info(scan, None, **kw)
        hook.get_dataobj(scan, None, **kw)
        assert count() == 1
        hook.get_dataobj(_Scan(), None, **kw)                 # a new scan (cache read): warns again
        assert count() == 2
