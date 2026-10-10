"""Golden frames: memory and disk check, NIfTI frame time, ignored Default-trajectory keys, docs (WI-0113 CP4).

* The size check runs before anything is reconstructed: the frames' recon cache
  (complex128, one volume per frame) is the disk need, and the frame engine's
  images (the running sum, one copy per open frame start, the frame written) are
  in the memory estimate. A scan-11-sized default (1,800 subset frames of 120^3,
  49.8 GB of cache) stops with the size when the disk is too small (D-0190 1).
* NIfTI: with golden frames the time step (pixdim[4]) is ``frame_step`` x the
  spoke TR in the requested time unit (D-0190 2); one repetition per frame keeps
  the header as before.
* The golden spoke lists do not use ``ProUnderSampling``, ``Reorder`` or
  ``HalfAcquisition`` (Director 2026-10-10): values left in a golden protocol
  change nothing.
"""
import io
import re
from pathlib import Path

import numpy as np
import pytest

from brkraw_sordino import hook, memguard
from brkraw_sordino.traj import gradient_list

from test_golden_frames import _hook_env, _setup
from test_golden_recon import _Scan, _golden_info

ROOT = Path(__file__).resolve().parents[1]


class _BigHandle(io.BytesIO):
    _orig_file_size = 288000 * 240 * 2 * 4


class _BigEntry:
    name = "fid-big"

    def open(self):
        return _BigHandle(b"")


def _scan11_like():
    info = _golden_info()
    info.update(Matrix=[120, 120, 120], NPoints=240, NGoldenSubsets=1800, NGoldenSpokesPerSubset=160,
                NGoldenSteps=288000, NPro=288000, O1List_Hz=[0.0], NRepetitions=1, RepetitionTime_ms=0.625)
    return info


def _big_env(monkeypatch):
    info = _scan11_like()
    monkeypatch.setattr(hook, "_parse_recon_info", lambda scan: dict(info))
    monkeypatch.setattr(hook, "_get_fid_entry", lambda scan: _BigEntry())
    return info


VOL_BYTES = 120 ** 3 * 16


def test_scan11_sized_default_needs_the_frames_cache_on_disk(tmp_path, monkeypatch):
    _big_env(monkeypatch)
    info = hook.get_dataobj_info(_Scan(), None, cache_dir=str(tmp_path))
    assert info["frames"] == 1800 and info["cache_nbytes"] == 1800 * VOL_BYTES   # 49.77e9 bytes
    assert info["disk_nbytes"] == 1800 * VOL_BYTES
    rep = hook.get_dataobj_info(_Scan(), None, cache_dir=str(tmp_path), frame_spokes="repetition")
    assert rep["frames"] == 1 and rep["disk_nbytes"] == VOL_BYTES


def test_scan11_sized_default_stops_before_start_when_the_disk_is_small(tmp_path, monkeypatch):
    _big_env(monkeypatch)
    monkeypatch.setattr(memguard, "free_disk_bytes", lambda path: 10 * memguard.GIB)
    called = []
    monkeypatch.setattr(hook, "recon_dataobj", lambda *a, **k: called.append(1))
    from brkraw_sordino import frames
    monkeypatch.setattr(frames, "recon_frames", lambda *a, **k: called.append(1))
    with pytest.raises(memguard.SordinoResourceError) as exc:
        hook.get_dataobj(_Scan(), None, cache_dir=str(tmp_path), max_memory_gb=1000)
    assert exc.value.kind == "disk" and not called
    assert "46.35 GiB" in str(exc.value) and "10.00 GiB" in str(exc.value)


def test_open_frame_starts_are_in_the_memory_estimate(tmp_path, monkeypatch):
    _big_env(monkeypatch)
    side = hook.get_dataobj_info(_Scan(), None, cache_dir=str(tmp_path), frame_spokes=3200)
    slide = hook.get_dataobj_info(_Scan(), None, cache_dir=str(tmp_path), frame_spokes=3200, frame_step=160)
    acc = hook.get_dataobj_info(_Scan(), None, cache_dir=str(tmp_path), frame_spokes=3200, frame_accumulate=True)
    assert side["recon_frames_open"] == 1 and slide["recon_frames_open"] == 20 and acc["recon_frames_open"] == 1
    assert slide["recon_nbytes"] - side["recon_nbytes"] == 19 * VOL_BYTES


def test_a_heavily_overlapping_window_stops_before_start_with_the_size(tmp_path, monkeypatch):
    _big_env(monkeypatch)
    called = []
    monkeypatch.setattr(hook, "recon_dataobj", lambda *a, **k: called.append(1))
    with pytest.raises(memguard.SordinoResourceError) as exc:
        hook.get_dataobj(_Scan(), None, cache_dir=str(tmp_path), frame_spokes=3200, frame_step=1,
                         num_frames=1)
    assert exc.value.kind == "memory" and not called
    assert re.search(r"needs about [0-9.]+ GiB", str(exc.value))


# ------------------------------------------------------------------ NIfTI frame time (D-0190 2)
@pytest.mark.parametrize("t_units,factor", [("sec", 1e-3), ("msec", 1.0), ("usec", 1e3)])
def test_nifti_time_step_is_frame_step_times_the_spoke_tr(tmp_path, monkeypatch, t_units, factor):
    info, fid = _setup(tmp_path)
    info = dict(info, RepetitionTime_ms=0.625)
    _hook_env(monkeypatch, info, fid)
    scan = _Scan()
    data = hook.get_dataobj(scan, None, cache_dir=str(tmp_path / "h"), frame_spokes=120, frame_step=40)
    nii = hook.convert(scan, data, np.eye(4), t_units=t_units)
    assert nii.header["pixdim"][4] == pytest.approx(40 * 0.625 * factor, rel=1e-6)
    assert nii.header.get_xyzt_units()[1] == t_units


def test_one_repetition_per_frame_keeps_the_header(tmp_path, monkeypatch):
    info, fid = _setup(tmp_path, n_rep=2)
    info = dict(info, RepetitionTime_ms=0.625)
    _hook_env(monkeypatch, info, fid)
    scan = _Scan()
    data = hook.get_dataobj(scan, None, cache_dir=str(tmp_path / "h"), frame_spokes="repetition")
    nii = hook.convert(scan, data, np.eye(4))
    assert nii.header["pixdim"][4] == 1.0          # as before CP4 (the hook never set it)


# ------------------------------------------------------------------ Default-trajectory keys
@pytest.mark.parametrize("mode", ["GoldenSampling", "GoldenGridSampling"])
def test_golden_lists_ignore_the_default_trajectory_keys(mode):
    base = _golden_info(mode=mode)
    g0, p0 = gradient_list(base)
    for extra in ({"UnderSampling": 3.61468271736276}, {"UnderSampling": 1.0}, {"UnderSampling": 10.0},
                  {"Reorder": True}, {"HalfAcquisition": True}):
        g, p = gradient_list(dict(base, **extra))
        assert np.array_equal(g, g0) and p == p0, extra


# ------------------------------------------------------------------ docs
def _golden_section(path):
    text = path.read_text()
    m = re.search(r"^## Golden trajectories and frames\n(.*?)(?=^## )", text, re.S | re.M)
    assert m, path
    return m.group(1)


def test_docs_golden_section_is_the_same_in_readme_and_docs():
    assert _golden_section(ROOT / "README.md") == _golden_section(ROOT / "src/brkraw_sordino/docs.md")


def test_docs_golden_options_are_real_options(tmp_path):
    import yaml

    sec = _golden_section(ROOT / "README.md")
    names = set(re.findall(r"`(frame_[a-z_]+)`", sec))
    assert names == {"frame_spokes", "frame_step", "frame_accumulate"}
    blocks = re.findall(r"```yaml\n(.*?)```", sec, re.S)
    assert blocks
    for b in blocks:
        args = yaml.safe_load(b)["hooks"]["sordino"]
        hook._build_options(dict(args, cache_dir=str(tmp_path)))
    for line in re.findall(r"--hook-arg sordino:([a-z_]+)=", sec):
        assert line in {"frame_spokes", "frame_step", "frame_accumulate", "estimate_k0"}
    for word in ("ProUnderSampling", "49.8 GB", "pixdim"):
        assert word in sec
