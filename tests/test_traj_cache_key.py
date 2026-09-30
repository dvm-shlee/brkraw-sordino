"""Trajectory cache key: only the values that generate the trajectory (WI-0071, D-0097 2).

The trajectory is saved in the cache folder under a hash and reused when the
hash matches. Before WI-0071 the hash was the plain concatenation of
``str()`` values, so two different parameter sets could give the same text
(below: OverSampling 2.0 with NPro 512 and OverSampling 2.05 with NPro 12
both give "...2.0512..."), and it held ``ext_factors`` (and, with the ramp
model, every timing and tuning value), which do not change the trajectory.
"""
from dataclasses import replace

import numpy as np
import pytest

from brkraw_sordino import timing, traj as traj_mod
from brkraw_sordino.hook import _build_options
from brkraw_sordino.traj import get_trajectory

from test_ramp_model import _info


def _opts(cache_dir, **kw):
    return _build_options(dict(cache_dir=str(cache_dir), **kw))


def _files(cache_dir):
    return sorted(p.name for p in cache_dir.glob("*.npy"))


def test_different_trajectories_never_share_a_cache_file(tmp_path):
    """The collision: the second scan must get its own trajectory, not the first one's."""
    a = dict(_info("v2"), OverSampling=2.0, NPro=5120)
    b = dict(_info("v2"), OverSampling=2.05, NPro=120)
    shared, alone = tmp_path / "shared", tmp_path / "alone"
    ta = get_trajectory(a, _opts(shared, correct_ramptime=False))
    tb = get_trajectory(b, _opts(shared, correct_ramptime=False))
    tb_alone = get_trajectory(b, _opts(alone, correct_ramptime=False))
    assert ta.shape != tb_alone.shape
    assert tb.shape == tb_alone.shape and np.array_equal(tb, tb_alone)
    assert len(_files(shared)) == 2


@pytest.mark.parametrize("ext", [(2.0, 2.0, 2.0), (1.0, 1.5, 1.0)])
@pytest.mark.parametrize("ramp", [True, False])
def test_ext_factors_do_not_change_the_trajectory_or_its_file(tmp_path, ext, ramp):
    info = _info("v2")
    base = get_trajectory(info, _opts(tmp_path, correct_ramptime=ramp))
    other = get_trajectory(info, _opts(tmp_path, correct_ramptime=ramp, ext_factors=list(ext)))
    assert np.array_equal(base, other)
    assert len(_files(tmp_path)) == 1


def test_options_outside_the_trajectory_do_not_change_its_file(tmp_path):
    info = _info("v2")
    get_trajectory(info, _opts(tmp_path))
    for kw in (dict(ignore_samples=3), dict(offset=2), dict(num_frames=5),
               dict(correct_spoketiming=True), dict(offreso_freqs=(120.0,)),
               dict(mem_limit=2.0), dict(clear_cache=False), dict(split_ch=True),
               dict(as_complex=True), dict(estimate_k0=True)):
        get_trajectory(info, _opts(tmp_path, **kw))
    assert len(_files(tmp_path)) == 1


def test_scan_values_outside_the_trajectory_do_not_change_its_file(tmp_path):
    info = _info("v2")
    get_trajectory(info, _opts(tmp_path))
    # v2 stores RampDelay but does not use it; O1 list, repetitions and receivers
    # belong to the phase correction and the reconstruction, not the trajectory
    for extra in (dict(RampDelay_ms=0.5), dict(O1List_Hz=[10.0, -10.0]),
                  dict(NRepetitions=9), dict(EncNReceivers=4)):
        get_trajectory(dict(info, **extra), _opts(tmp_path))
    assert len(_files(tmp_path)) == 1


def test_phase_reference_tuning_does_not_change_the_trajectory_file(tmp_path, monkeypatch):
    info = _info("v2")
    base = get_trajectory(info, _opts(tmp_path))
    tuned = replace(timing.TIMING_TUNING["v2"], phase_ref_us=3.0)
    monkeypatch.setitem(timing.TIMING_TUNING, "v2", tuned)
    again = get_trajectory(info, _opts(tmp_path))
    assert np.array_equal(base, again)
    assert len(_files(tmp_path)) == 1


CHANGES = [
    ("fixed", dict(AcqDelayTotal_us=9.0)),
    ("fixed", dict(EffBandwidth_Hz=50000.0)),
    ("fixed", dict(Reorder=True)),
    ("fixed", dict(HalfAcquisition=True)),
    ("integral", dict(AcqDelayTotal_us=9.0)),
    ("integral", dict(EffBandwidth_Hz=50000.0)),
    ("integral", dict(RampTime_ms=0.2)),
    ("integral", dict(ExcPulLength_ms=0.012)),
    ("integral", dict(Reorder=True)),
]


@pytest.mark.parametrize("model,change", CHANGES)
def test_every_generating_value_gives_its_own_file(tmp_path, model, change):
    info = _info("v2")
    opts = _opts(tmp_path, correct_ramptime=(model == "integral"))
    base = get_trajectory(info, opts)
    other = get_trajectory(dict(info, **change), opts)
    assert len(_files(tmp_path)) == 2
    assert base.shape != other.shape or not np.array_equal(base, other)


def test_ramp_model_on_and_off_are_separate_files(tmp_path):
    info = _info("v2")
    a = get_trajectory(info, _opts(tmp_path, correct_ramptime=True))
    b = get_trajectory(info, _opts(tmp_path, correct_ramptime=False))
    assert not np.array_equal(a, b)
    assert len(_files(tmp_path)) == 2


def test_first_sample_tuning_changes_the_trajectory_file(tmp_path, monkeypatch):
    info = _info("v2")
    get_trajectory(info, _opts(tmp_path))
    tuned = replace(timing.TIMING_TUNING["v2"], acq_start_offset_us=1.0)
    monkeypatch.setitem(timing.TIMING_TUNING, "v2", tuned)
    get_trajectory(info, _opts(tmp_path))
    assert len(_files(tmp_path)) == 2


def test_cache_hit_returns_the_saved_trajectory(tmp_path, monkeypatch):
    info = _info("v2")
    first = get_trajectory(info, _opts(tmp_path))

    def fail(*a, **k):
        raise AssertionError("trajectory recomputed on a cache hit")

    monkeypatch.setattr(traj_mod, "calc_radial_traj3d_integral", fail)
    again = get_trajectory(info, _opts(tmp_path))
    assert np.array_equal(first, again)


def test_a_damaged_cache_file_is_computed_again(tmp_path):
    info = _info("v2")
    first = get_trajectory(info, _opts(tmp_path))
    (path,) = list(tmp_path.glob("*.npy"))
    np.save(path, first[:3])                     # wrong shape under the right name
    again = get_trajectory(info, _opts(tmp_path))
    assert np.array_equal(first, again)
    path.write_bytes(b"not a numpy file")
    again = get_trajectory(info, _opts(tmp_path))
    assert np.array_equal(first, again)


def test_no_partial_file_is_left(tmp_path):
    get_trajectory(_info("v2"), _opts(tmp_path))
    assert not [p for p in tmp_path.iterdir() if not p.name.endswith(".npy")]


MORE_CHANGES = [
    ("v2", True, dict(Matrix=[20, 20, 20], NPro=1340)),
    ("v2", True, dict(NPro=512)),
    ("v2", False, dict(UseOrigin=True)),
    ("v2", True, dict(OverSampling=2.5)),
    ("v1", True, dict(RampDelay_ms=0.3)),
]


@pytest.mark.parametrize("version,ramp,change", MORE_CHANGES)
def test_more_generating_values_give_their_own_file(tmp_path, version, ramp, change):
    info = _info(version)
    opts = _opts(tmp_path, correct_ramptime=ramp)
    base = get_trajectory(info, opts)
    other = get_trajectory(dict(info, **change), opts)
    assert len(_files(tmp_path)) == 2
    assert base.shape != other.shape or not np.array_equal(base, other)


@pytest.mark.parametrize("field", ["ramp_start_offset_us", "ramp_len_offset_us"])
def test_ramp_window_tuning_changes_the_trajectory_file(tmp_path, monkeypatch, field):
    info = _info("v2")
    get_trajectory(info, _opts(tmp_path))
    tuned = replace(timing.TIMING_TUNING["v2"], **{field: 1.0})
    monkeypatch.setitem(timing.TIMING_TUNING, "v2", tuned)
    get_trajectory(info, _opts(tmp_path))
    assert len(_files(tmp_path)) == 2


def test_version_bump_changes_the_file_name(tmp_path, monkeypatch):
    info = _info("v2")
    get_trajectory(info, _opts(tmp_path))
    monkeypatch.setattr(traj_mod, "TRAJ_CACHE_VERSION", traj_mod.TRAJ_CACHE_VERSION + 1)
    get_trajectory(info, _opts(tmp_path))
    assert len(_files(tmp_path)) == 2


def test_a_file_under_the_old_md5_name_is_not_read(tmp_path):
    import hashlib
    info = _info("v2")
    opts = _opts(tmp_path, correct_ramptime=False)
    old_text = "".join(str(v) for v in (
        float(info["AcqDelayTotal_us"]), info["Matrix"][0], info["EffBandwidth_Hz"],
        info["OverSampling"], int(info["NPro"]), 1.0, False, False, False, False))
    fresh = get_trajectory(info, _opts(tmp_path / "fresh", correct_ramptime=False))
    planted = tmp_path / f"{hashlib.md5(old_text.encode()).hexdigest()}.npy"
    np.save(planted, np.zeros_like(fresh))
    got = get_trajectory(info, opts)
    assert np.array_equal(got, fresh)


def test_a_float32_file_is_computed_again(tmp_path):
    info = _info("v2")
    first = get_trajectory(info, _opts(tmp_path))
    (path,) = list(tmp_path.glob("*.npy"))
    np.save(path, first.astype(np.float32))
    again = get_trajectory(info, _opts(tmp_path))
    assert again.dtype == np.float64 and np.array_equal(first, again)


def test_saving_twice_and_a_stale_partial_do_not_fail(tmp_path):
    arr = np.arange(24, dtype=float).reshape(2, 4, 3)
    path = tmp_path / "traj_x.npy"
    stale = tmp_path / "traj_x.npy.partial"
    stale.write_bytes(b"left by a killed run")
    traj_mod._save_trajectory(path, arr)
    traj_mod._save_trajectory(path, arr)
    assert np.array_equal(np.load(path), arr)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["traj_x.npy", "traj_x.npy.partial"]


def test_a_failed_save_leaves_no_temporary_file(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(traj_mod.np, "save", boom)
    with pytest.raises(OSError):
        traj_mod._save_trajectory(tmp_path / "traj_y.npy", np.zeros((1, 2, 3)))
    assert list(tmp_path.iterdir()) == []


GOLDEN = {  # sha256 of np.round(traj, 10) for _info("v2"), shape (856, 16, 3); WI-0071
    True: "d65ff9e469b5179730f3be9afad4637f33b682b744ef2a27008b1b631718d16f",
    False: "57484c15679efdd9d3c2351528e986a18e88a6ae2f182e37182d382a420b804e",
}


@pytest.mark.parametrize("ramp", [True, False])
def test_trajectory_formulas_are_unchanged(tmp_path, ramp):
    """Fails when a trajectory formula changes: then increase TRAJ_CACHE_VERSION and update
    this value, so that files saved by the older formula are not reused."""
    import hashlib
    t = get_trajectory(_info("v2"), _opts(tmp_path, correct_ramptime=ramp))
    assert t.shape == (856, 16, 3)
    assert hashlib.sha256(np.round(t, 10).tobytes()).hexdigest() == GOLDEN[ramp]
