"""Trajectory mode from the method, golden gradient lists in the trajectory (WI-0113, CP0/CP1).

The sequence chooses the spoke directions with ``TrajectoryMode`` (``Default``,
``GoldenSampling``, ``GoldenGridSampling``). brkraw-sordino reads it from the
method (no override option); a scan without the key is ``Default`` (older
sequences). Only the list of spoke directions changes; the ramp model and the
phase correction are the same for every mode. The ``Default`` path must stay
bit for bit what it was.
"""
import logging
from pathlib import Path

import numpy as np
import pytest
import yaml

from brkraw_sordino import kcentre, traj as traj_mod
from brkraw_sordino.hook import _build_options
from brkraw_sordino.traj import calc_radial_grad3d, get_trajectory, trajectory_rows

from test_ramp_model import _info, MATRIX, N


def _golden():
    from brkraw_sordino import golden
    return golden


def _gs(**kw):
    info = dict(_info("v3"), TrajectoryMode="GoldenSampling", NGoldenSubsets=12,
                NGoldenSpokesPerSubset=40, NGoldenSteps=480, GoldenReorder=True,
                ZStackAngleDeg=22.5, NPro=480)
    info.update(kw)
    return info


def _gg(**kw):
    info = dict(_info("v3"), TrajectoryMode="GoldenGridSampling", NGridRing=10, NGridFrames=4,
                GridMirror=True, NPro=4 * 128 * 2)
    info.update(kw)
    return info


def _opts(tmp_path, **kw):
    return _build_options(dict(cache_dir=str(tmp_path), **kw))


# ------------------------------------------------------------------ mode
def test_missing_mode_is_default():
    golden = _golden()
    assert golden.trajectory_mode(_info("v2")) == "Default"
    assert golden.trajectory_mode(dict(_info("v2"), TrajectoryMode=None)) == "Default"
    assert golden.trajectory_mode(dict(_info("v2"), TrajectoryMode=" Default ")) == "Default"
    assert golden.trajectory_mode(_gs()) == "GoldenSampling"
    assert golden.trajectory_mode(_gg()) == "GoldenGridSampling"


def test_unknown_mode_stops_and_names_the_value():
    with pytest.raises(ValueError, match="Spiral3D"):
        _golden().trajectory_mode(dict(_info("v2"), TrajectoryMode="Spiral3D"))
    with pytest.raises(ValueError, match="Spiral3D"):
        traj_mod.gradient_list(dict(_info("v2"), TrajectoryMode="Spiral3D"))


# ------------------------------------------------------------------ Default unchanged
@pytest.mark.parametrize("mode", [None, "Default"])
def test_default_gradient_list_and_cache_fields_are_unchanged(mode):
    info = _info("v2")
    if mode is not None:
        info = dict(info, TrajectoryMode=mode, NGoldenSubsets=1800, NGridRing=10)
    grad, params = traj_mod.gradient_list(info)
    ref = calc_radial_grad3d(MATRIX, info["NPro"], False, False, False)
    assert np.array_equal(grad, ref)
    assert params == {"matrix_size": MATRIX, "npro_target": info["NPro"], "half_sphere": False,
                      "use_origin": False, "reorder": False}
    assert traj_mod.TRAJ_CACHE_VERSION == 2


@pytest.mark.parametrize("version,ramp", [("v1", True), ("v2", True), ("v3", True), ("v2", False)])
def test_default_trajectory_is_the_unchanged_formula(tmp_path, version, ramp):
    """Default: the radial list and the formulas as before, bit for bit."""
    from brkraw_sordino import timing
    from brkraw_sordino.traj import calc_radial_traj3d, calc_radial_traj3d_integral

    info = _info(version)
    t = get_trajectory(info, _opts(tmp_path, correct_ramptime=ramp))
    g = calc_radial_grad3d(MATRIX, info["NPro"], False, False, False)
    if ramp:
        seq = timing.read_timing(info)
        times, f, _ = timing.ramp_terms(seq, timing.tuning_for(seq.version), N)
        ref = calc_radial_traj3d_integral(g, MATRIX, info["OverSampling"], times, f, seq.dwell_us)
    else:
        off = float(info["AcqDelayTotal_us"] * 1e-6 * info["EffBandwidth_Hz"] * info["OverSampling"])
        ref = calc_radial_traj3d(g, MATRIX, info["OverSampling"], off)
    assert np.array_equal(t, ref)


# ------------------------------------------------------------------ golden lists
def test_golden_sampling_list_in_the_trajectory(tmp_path):
    golden = _golden()
    info = _gs()
    grad, params = traj_mod.gradient_list(info)
    assert np.array_equal(grad, golden.reorder_golden_samples(12, 40, 22.5))
    assert params["mode"] == "GoldenSampling"
    t = get_trajectory(info, _opts(tmp_path, correct_ramptime=False))
    assert t.shape == (480, N, 3)
    samp = t[:, -1, :] / np.linalg.norm(t[:, -1, :], axis=1, keepdims=True)
    assert np.abs(samp - grad.T).max() < 1e-12


def test_golden_grid_list_in_the_trajectory(tmp_path):
    golden = _golden()
    info = _gg()
    grad, params = traj_mod.gradient_list(info)
    assert np.array_equal(grad, golden.sreag_trajectory(10, 4, True))
    assert params["mode"] == "GoldenGridSampling"
    t = get_trajectory(info, _opts(tmp_path, correct_ramptime=False))
    assert t.shape == (1024, N, 3)


@pytest.mark.parametrize("make", [_gs, _gg])
@pytest.mark.parametrize("ramp", [True, False])
def test_rows_equal_the_whole_trajectory(tmp_path, make, ramp):
    info = make()
    opts = _opts(tmp_path, correct_ramptime=ramp)
    whole = get_trajectory(info, opts)
    rows = trajectory_rows(info, opts)
    assert rows.shape == whole.shape
    for lo, hi in ((0, 7), (100, 333), (whole.shape[0] - 5, whole.shape[0])):
        assert np.array_equal(rows.rows(lo, hi), whole[lo:hi])


def test_golden_reorder_no_uses_ngoldensteps():
    info = _gs(GoldenReorder=False, NGoldenSteps=300, NPro=300)
    grad, _ = traj_mod.gradient_list(info)
    ref = _golden().reorder_golden_samples(12, 40, 22.5, golden_reorder=False, n_pro=300)
    assert grad.shape == (3, 300) and np.array_equal(grad, ref)


def test_use_origin_reaches_the_golden_list():
    grad, _ = traj_mod.gradient_list(_gs(UseOrigin=True))
    assert np.array_equal(grad[:, 0], np.zeros(3))


@pytest.mark.parametrize("info,numbers", [
    (_gs(NPro=481), ("481", "480")),
    (_gg(NPro=512), ("512", "1024")),
])
def test_spoke_count_mismatch_stops_with_both_numbers(info, numbers):
    with pytest.raises(ValueError) as exc:
        traj_mod.gradient_list(info)
    for text in numbers:
        assert text in str(exc.value)


@pytest.mark.parametrize("info,key", [
    (_gs(NGoldenSpokesPerSubset=None), "NGoldenSpokesPerSubset"),
    (_gs(ZStackAngleDeg=None), "ZStackAngleDeg"),
    (_gg(NGridRing=None), "NGridRing"),
    (_gg(GridMirror=None), "GridMirror"),
])
def test_missing_golden_parameter_is_named(info, key):
    with pytest.raises(ValueError, match=key):
        traj_mod.gradient_list(info)


def test_modes_have_their_own_trajectory_files(tmp_path):
    base = _info("v3")
    npro = 4 * 128 * 2
    get_trajectory(dict(base, NPro=npro), _opts(tmp_path))
    get_trajectory(_gg(), _opts(tmp_path))
    get_trajectory(_gg(NGridFrames=8, GridMirror=False), _opts(tmp_path))
    get_trajectory(_gs(), _opts(tmp_path))
    get_trajectory(_gs(ZStackAngleDeg=30.0), _opts(tmp_path))
    assert len(list(tmp_path.glob("*.npy"))) == 5


def test_golden_unit_is_the_method_subset():
    golden = _golden()
    assert golden.golden_unit(_gs()) == 40
    assert golden.golden_unit(_gs(GoldenReorder=False, NGoldenSteps=300, NPro=300)) == 1
    assert golden.golden_unit(_gg()) == 256
    assert golden.golden_unit(_gg(GridMirror=False, NPro=512)) == 128
    assert golden.golden_unit(_info("v2")) is None


# ------------------------------------------------------------------ ACQ_O1_list order check
def _o1(grad, off=(310.0, -120.0, 45.0), c=7.0):
    return list(c + off[0] * grad[0] + off[1] * grad[1] + off[2] * grad[2])


def test_o1_residual_is_zero_for_the_acquired_list_and_one_for_another():
    grad, _ = traj_mod.gradient_list(_gs())
    assert traj_mod.o1_order_residual(np.asarray(_o1(grad)), grad) < 1e-12
    other = _golden().golden_samples(480)[0]
    assert traj_mod.o1_order_residual(np.asarray(_o1(other)), grad) > 0.5


@pytest.mark.parametrize("make", [_gs, _gg])
def test_o1_check_warns_only_when_the_order_differs(tmp_path, caplog, make):
    info = make()
    grad, _ = traj_mod.gradient_list(info)
    with caplog.at_level(logging.DEBUG, logger="brkraw_sordino"):
        trajectory_rows(dict(info, O1List_Hz=_o1(grad)), _opts(tmp_path))
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("ACQ_O1_list" in r.getMessage() for r in caplog.records)
    caplog.clear()
    shuffled = grad[:, np.random.default_rng(3).permutation(grad.shape[1])]
    with caplog.at_level(logging.WARNING, logger="brkraw_sordino"):
        trajectory_rows(dict(info, O1List_Hz=_o1(shuffled)), _opts(tmp_path))
    warn = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warn) == 1 and "ACQ_O1_list" in warn[0] and info["TrajectoryMode"] in warn[0]


def test_o1_check_is_quiet_without_a_fov_offset_and_for_default(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="brkraw_sordino"):
        trajectory_rows(dict(_gs(), O1List_Hz=[0.0]), _opts(tmp_path))
        info = _info("v3")
        g = calc_radial_grad3d(MATRIX, info["NPro"], False, False, False)
        rng = np.random.default_rng(1)
        trajectory_rows(dict(info, O1List_Hz=_o1(g[:, rng.permutation(g.shape[1])])),
                        _opts(tmp_path))
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# ------------------------------------------------------------------ users of the list
@pytest.mark.parametrize("make", [_gs, _gg])
def test_k0_leading_points_follow_the_mode(make):
    info = make()
    grad, _ = traj_mod.gradient_list(info)
    assert np.array_equal(kcentre.leading_points(info, 1), kcentre.leading_points(info, 1, grad=grad))


def test_recon_spec_reads_the_mode_parameters():
    spec = yaml.safe_load((Path(traj_mod.__file__).parent / "specs" / "recon_spec.yaml").read_text())
    for key in ("TrajectoryMode", "NGoldenSubsets", "NGoldenSpokesPerSubset", "NGoldenSteps",
                "GoldenReorder", "ZStackAngleDeg", "NGridRing", "NGridFrames", "GridMirror"):
        assert spec[key]["sources"] == [{"file": "method", "key": key}], key


def test_yes_no_transform_keeps_a_missing_value_missing():
    from brkraw_sordino.specs import utils
    assert utils.yes_no("Yes") is True and utils.yes_no("No") is False
    assert utils.yes_no(None) is None
