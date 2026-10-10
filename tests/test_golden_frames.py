"""Frames of golden scans: subsets, user spoke counts, sliding windows, accumulation (WI-0113 CP3).

Options (golden trajectories only; D-0184 2, D-0189 4, D-0190, D-0191):

* ``frame_spokes``: default (None) is the method subset, the smallest spoke
  count that covers the sphere: ``NGoldenSpokesPerSubset`` for Golden Sampling,
  one grid frame (cells x 2 with mirror) for Golden Grid (D-0190 1).
  ``"subset"`` says the same; ``"repetition"`` keeps one repetition per frame
  (the reconstruction before CP3, unchanged); an integer is a spoke count.
* ``frame_step``: spokes between frame starts (default ``frame_spokes``; smaller
  gives a sliding window). With ``frame_accumulate`` every frame starts at the
  first spoke and ``frame_step`` is how much each frame grows (default
  ``frame_spokes``: the frames hold 1, 2, 3 ... x ``frame_spokes``, D-0189 4).
* The repetitions read form one stream of spokes (D-0191 2): windows,
  accumulation and sliding windows run across repetition boundaries.

A frame image is the adjoint of its spokes with the density weight normalised
over one repetition, scaled by NPro / (spokes in the frame) so every frame has
the brightness of a one-repetition image. Frames are differences of one running
sum (linearity): every spoke is reconstructed once. A frame may start inside a
method subset (a warning, not a refusal: D-0187 question). With ``estimate_k0``
the centre is estimated once from all spokes read and its predicted samples are
added spoke by spoke, so every frame gets the same estimate (D-0191 4).
"""
import io
import logging

import numpy as np
import pytest

from brkraw_sordino import hook, serial
from brkraw_sordino.hook import _build_options
from brkraw_sordino.recon import phase_correction_rows, recon_dataobj
from brkraw_sordino.traj import get_trajectory, trajectory_rows

from test_golden_recon import _Entry, _Scan, _golden_info
from test_serial_recon import N_POINTS, SHAPE, _fid_frames, _rel

WINDOW_TOL = 1e-8


def _frames():
    from brkraw_sordino import frames
    return frames


def _opts(tmp_path, **kw):
    return _build_options({"cache_dir": str(tmp_path), **kw})


# ------------------------------------------------------------------ options
@pytest.mark.parametrize("value,expected", [
    (None, None), ("repetition", "repetition"), ("subset", "subset"), (" Subset ", "subset"),
    (160, 160), ("160", 160), (160.0, 160),
])
def test_frame_spokes_option_values(tmp_path, value, expected):
    kw = {} if value is None else {"frame_spokes": value}
    assert _opts(tmp_path, **kw).frame_spokes == expected


@pytest.mark.parametrize("value", ["half", 0, -40, 12.5, True, "0"])
def test_bad_frame_spokes_is_refused_by_name(tmp_path, value):
    with pytest.raises(ValueError, match="frame_spokes"):
        _opts(tmp_path, frame_spokes=value)


def test_other_frame_options(tmp_path):
    o = _opts(tmp_path, frame_step="40", frame_accumulate="yes")
    assert o.frame_step == 40 and o.frame_accumulate is True
    d = _opts(tmp_path)
    assert d.frame_step is None and d.frame_accumulate is False
    for bad in (0, -1, 2.5, True, "x"):
        with pytest.raises(ValueError, match="frame_step"):
            _opts(tmp_path, frame_step=bad)


def test_frame_mode_follows_the_trajectory(tmp_path):
    from test_serial_recon import _info as serial_info
    f = _frames()
    golden, default = _golden_info(), serial_info(1, 1, phase=False)
    assert f.is_frame_mode(_opts(tmp_path), golden)                  # golden default: subsets
    assert not f.is_frame_mode(_opts(tmp_path, frame_spokes="repetition"), golden)
    assert f.is_frame_mode(_opts(tmp_path, frame_spokes="repetition", frame_step=400), golden)
    assert not f.is_frame_mode(_opts(tmp_path), default)
    assert not f.is_frame_mode(_opts(tmp_path, frame_spokes="repetition"), default)


# ------------------------------------------------------------------ plan
def _plan(info, tmp_path, n_rep=1, **kw):
    info = dict(info, NRepetitions=n_rep)
    return _frames().make_plan(info, _opts(tmp_path, **kw))


def test_default_golden_frames_are_the_method_subsets(tmp_path):
    for kw in ({}, {"frame_spokes": "subset"}):
        p = _plan(_golden_info(), tmp_path, **kw)                       # 20 subsets x 40
        assert p.unit == 40 and p.window == 40 and p.step == 40 and p.n_frames == 20
        assert p.ranges[0] == (0, 40) and p.ranges[-1] == (760, 800)
        assert p.scales == tuple([20.0] * 20)
    assert _plan(_golden_info(), tmp_path, frame_spokes="repetition") is None


def test_sliding_window_and_accumulation(tmp_path):
    p = _plan(_golden_info(), tmp_path, frame_spokes=120, frame_step=40)
    assert p.n_frames == (800 - 120) // 40 + 1 and p.ranges[1] == (40, 160)
    a = _plan(_golden_info(), tmp_path, frame_spokes=160, frame_accumulate=True)
    assert a.ranges == ((0, 160), (0, 320), (0, 480), (0, 640), (0, 800))
    assert a.scales == (5.0, 2.5, 800 / 480, 1.25, 1.0)
    b = _plan(_golden_info(), tmp_path, frame_spokes=160, frame_step=80, frame_accumulate=True)
    assert b.ranges[:3] == ((0, 160), (0, 240), (0, 320)) and b.ranges[-1] == (0, 800)


def test_frame_interval_is_frame_step_times_the_spoke_tr(tmp_path):
    info = dict(_golden_info(), RepetitionTime_ms=0.625)
    p = _plan(info, tmp_path, frame_spokes=120, frame_step=40)
    assert p.frame_interval_s == pytest.approx(40 * 0.625e-3)
    assert _plan(info, tmp_path).frame_interval_s == pytest.approx(40 * 0.625e-3)
    acc = _plan(info, tmp_path, frame_spokes=160, frame_accumulate=True)
    assert acc.frame_interval_s == pytest.approx(160 * 0.625e-3)
    assert _plan(_golden_info(), tmp_path).frame_interval_s is None       # no TR in the header


def test_gaps_between_frames_are_allowed(tmp_path):
    p = _plan(_golden_info(), tmp_path, frame_spokes=40, frame_step=200)
    assert p.ranges == ((0, 40), (200, 240), (400, 440), (600, 640))


def test_frames_may_start_inside_a_subset_with_a_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="brkraw_sordino.frames"):
        p = _plan(_golden_info(), tmp_path, frame_spokes=100, frame_step=50)
    assert p.n_frames == (800 - 100) // 50 + 1 and p.misaligned > 0
    text = caplog.text
    assert "subset" in text and "40" in text and "4.50" in text and "3.58" in text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="brkraw_sordino.frames"):
        _plan(_golden_info(), tmp_path, frame_spokes=120, frame_step=40)
    assert "subset" not in caplog.text


@pytest.mark.parametrize("kw,words", [
    ({"frame_spokes": 840}, ("frame_spokes", "840", "800")),
    ({"frame_spokes": "subset", "correct_spoketiming": True}, ("correct_spoketiming",)),
])
def test_refused_plans_say_why(tmp_path, kw, words):
    with pytest.raises(ValueError) as exc:
        _plan(_golden_info(), tmp_path, **kw)
    for w in words:
        assert w in str(exc.value)


def test_estimate_k0_is_allowed_with_frames(tmp_path):
    p = _plan(_golden_info(), tmp_path, frame_spokes="subset", estimate_k0=True)
    assert p.n_frames == 20


def test_default_trajectory_refuses_frame_options(tmp_path):
    from test_serial_recon import _info as serial_info
    for kw in ({"frame_spokes": 100}, {"frame_spokes": "subset"}, {"frame_accumulate": True},
               {"frame_step": 100}):
        with pytest.raises(ValueError, match="TrajectoryMode"):
            _plan(serial_info(1, 1, phase=False), tmp_path, **kw)


def test_reorder_no_takes_any_spoke_count(tmp_path):
    info = _golden_info()
    info.update(GoldenReorder=False)
    p = _plan(info, tmp_path, frame_spokes=100, frame_step=30)
    assert p.unit == 1 and p.ranges[1] == (30, 130) and p.misaligned == 0


def test_golden_grid_unit_is_one_grid_frame(tmp_path):
    info = _golden_info(mode="GoldenGridSampling")
    p = _plan(info, tmp_path)
    from brkraw_sordino.golden import sreag_grid
    assert p.unit == 2 * sreag_grid(6).n_cell and p.n_frames == 6


def test_repetitions_form_one_stream(tmp_path):
    p = _plan(_golden_info(), tmp_path, n_rep=2, frame_spokes=120, frame_step=80)
    assert p.n_frames == (1600 - 120) // 80 + 1
    assert p.crossing == sum(1 for r in p.ranges if r[0] < 800 < r[1]) > 0
    acc = _plan(_golden_info(), tmp_path, n_rep=2, frame_spokes=400, frame_accumulate=True)
    assert acc.ranges == ((0, 400), (0, 800), (0, 1200), (0, 1600))
    sub = _plan(_golden_info(), tmp_path, n_rep=2)
    assert sub.n_frames == 40 and sub.crossing == 0


def test_offset_and_num_frames_choose_the_repetitions(tmp_path):
    info = dict(_golden_info(), NRepetitions=3)
    p = _frames().make_plan(info, _opts(tmp_path, frame_spokes="subset", offset=1, num_frames=1))
    assert p.n_rep == 1 and p.n_frames == 20


# ------------------------------------------------------------------ engine
def _setup(tmp_path, n_rep=1, mode="GoldenSampling"):
    info = _golden_info(n_rep, mode)
    options = _opts(tmp_path / "t")
    traj = get_trajectory(info, options)
    factor = phase_correction_rows(info, options, N_POINTS)[:]
    fid = b"".join(_fid_frames(info, traj, n_rep, 1, factor))
    return info, fid


class _Concat:
    """Spokes (rep, lo, hi) pieces of a stream as one list: rows, phase rows and FID bytes."""

    def __init__(self, info, options, fid, pieces):
        self.rows_src = trajectory_rows(info, options)
        self.phase_src = phase_correction_rows(info, options, N_POINTS)
        spoke = 2 * N_POINTS * 1 * 4
        npro = int(info["NPro"])
        self.n_pro = sum(hi - lo for _, lo, hi in pieces)
        self.fid = b"".join(fid[(r * npro + lo) * spoke:(r * npro + hi) * spoke] for r, lo, hi in pieces)
        self._r = np.concatenate([self.rows_src.rows(lo, hi) for _, lo, hi in pieces])
        self._p = np.concatenate([self.phase_src[lo:hi] for _, lo, hi in pieces])

    def rows(self, lo, hi):
        return self._r[lo:hi]

    def __getitem__(self, idx):
        return self._p[idx]


def _direct(tmp_path, info, fid, pieces, scale):
    """recon_dataobj on exactly the frame's spokes, rescaled to the one-repetition density maximum."""
    options = _opts(tmp_path / "d")
    c = _Concat(info, options, fid, pieces)
    sub = dict(info, NPro=c.n_pro, NRepetitions=1)
    out = io.BytesIO()
    recon_dataobj(io.BytesIO(c.fid), c, sub, out, options, phase_factor=c)
    img = np.frombuffer(out.getvalue(), dtype=np.complex128).reshape(SHAPE[::-1]).T
    full = trajectory_rows(info, options)
    d_all = serial.density(full.rows(0, full.n_pro)[:, 1:]).max()
    d_sub = serial.density(c._r[:, 1:]).max()
    return img * (d_sub / d_all) * scale


def _engine(tmp_path, info, fid, k0_out=None, **kw):
    f = _frames()
    options = _opts(tmp_path / "e", **kw)
    plan = f.make_plan(info, options)
    vt = None
    if options.estimate_k0:
        from brkraw_sordino import kcentre
        vt = kcentre.leading_points(info, options.ignore_samples or 1)
    out = io.BytesIO()
    f.recon_frames(io.BytesIO(fid), trajectory_rows(info, options), info, out, options, plan,
                   phase_factor=phase_correction_rows(info, options, N_POINTS), virtual_traj=vt,
                   k0_out=k0_out)
    per = int(np.prod(SHAPE))
    data = np.frombuffer(out.getvalue(), dtype=np.complex128)
    assert data.size == per * plan.n_frames
    return plan, [data[k * per:(k + 1) * per].reshape(SHAPE[::-1]).T for k in range(plan.n_frames)]


def _pieces(rng, npro):
    lo, hi = rng
    out = []
    while lo < hi:
        r = lo // npro
        top = min(hi, (r + 1) * npro)
        out.append((r, lo - r * npro, top - r * npro))
        lo = top
    return out


@pytest.mark.parametrize("kw", [
    {},
    {"frame_spokes": 120, "frame_step": 40},
    {"frame_spokes": 160, "frame_accumulate": True},
    {"frame_spokes": 40, "frame_step": 200},
    {"frame_spokes": 100, "frame_step": 70},
], ids=["subset-default", "sliding", "accumulate", "gaps", "inside-subsets"])
def test_frames_equal_direct_reconstruction_of_their_spokes(tmp_path, kw):
    info, fid = _setup(tmp_path)
    plan, imgs = _engine(tmp_path, info, fid, **kw)
    for k in sorted({0, 1, plan.n_frames // 2, plan.n_frames - 1}):
        ref = _direct(tmp_path / f"f{k}", info, fid, _pieces(plan.ranges[k], 800), plan.scales[k])
        assert _rel(imgs[k], ref) < WINDOW_TOL, k


def test_golden_grid_subset_frames_equal_direct(tmp_path):
    info, fid = _setup(tmp_path, mode="GoldenGridSampling")
    plan, imgs = _engine(tmp_path, info, fid)
    npro = int(info["NPro"])
    for k in (0, plan.n_frames - 1):
        ref = _direct(tmp_path / f"g{k}", info, fid, _pieces(plan.ranges[k], npro), plan.scales[k])
        assert _rel(imgs[k], ref) < WINDOW_TOL


def test_frames_across_repetitions(tmp_path):
    info, fid = _setup(tmp_path, n_rep=2)
    plan, imgs = _engine(tmp_path, info, fid, frame_spokes=200, frame_step=120)
    crossing = [k for k, r in enumerate(plan.ranges) if r[0] < 800 < r[1]]
    assert crossing
    for k in (crossing[0], plan.n_frames - 1):
        ref = _direct(tmp_path / f"c{k}", info, fid, _pieces(plan.ranges[k], 800), plan.scales[k])
        assert _rel(imgs[k], ref) < WINDOW_TOL


def test_whole_repetition_windows_equal_the_repetition_frames(tmp_path):
    info, fid = _setup(tmp_path, n_rep=2)
    plan, imgs = _engine(tmp_path, info, fid, frame_spokes="repetition", frame_step=800)
    options = _opts(tmp_path / "def", frame_spokes="repetition")
    out = io.BytesIO()
    recon_dataobj(io.BytesIO(fid), trajectory_rows(info, options), info, out, options,
                  phase_factor=phase_correction_rows(info, options, N_POINTS))
    per = int(np.prod(SHAPE))
    data = np.frombuffer(out.getvalue(), dtype=np.complex128)
    assert plan.n_frames == 2
    for k in range(2):
        ref = data[k * per:(k + 1) * per].reshape(SHAPE[::-1]).T
        assert _rel(imgs[k], ref) < 1e-9


def test_subset_frames_add_up_to_the_repetition_image(tmp_path):
    info, fid = _setup(tmp_path)
    plan, imgs = _engine(tmp_path, info, fid)
    total = sum(img / s for img, s in zip(imgs, plan.scales))
    options = _opts(tmp_path / "rep", frame_spokes="repetition")
    out = io.BytesIO()
    recon_dataobj(io.BytesIO(fid), trajectory_rows(info, options), info, out, options,
                  phase_factor=phase_correction_rows(info, options, N_POINTS))
    ref = np.frombuffer(out.getvalue(), dtype=np.complex128).reshape(SHAPE[::-1]).T
    assert _rel(total, ref) < 1e-9


def test_each_spoke_is_reconstructed_once(tmp_path, monkeypatch):
    info, fid = _setup(tmp_path)
    calls = []
    real = serial.Adjoint.setpts

    def spy(self, tr):
        calls.append(int(tr.shape[0]))
        return real(self, tr)

    monkeypatch.setattr(serial.Adjoint, "setpts", spy)
    plan, _ = _engine(tmp_path, info, fid, frame_spokes=200, frame_step=40)
    assert sum(calls) == 800          # every spoke once (setpts gets (spokes, samples, 3)), frames overlap 5 times


# ------------------------------------------------------------------ estimate_k0 with frames (D-0191 4)
def test_k0_frames_add_up_to_the_repetition_k0_image(tmp_path):
    info, fid = _setup(tmp_path)
    k0_frames = []
    plan, imgs = _engine(tmp_path, info, fid, k0_out=k0_frames, estimate_k0=True)
    total = sum(img / s for img, s in zip(imgs, plan.scales))
    from brkraw_sordino import kcentre
    options = _opts(tmp_path / "rep", frame_spokes="repetition", estimate_k0=True)
    vt = kcentre.leading_points(info, options.ignore_samples or 1)
    out, k0_ref = io.BytesIO(), []
    recon_dataobj(io.BytesIO(fid), trajectory_rows(info, options), info, out, options,
                  phase_factor=phase_correction_rows(info, options, N_POINTS), virtual_traj=vt, k0_out=k0_ref)
    ref = np.frombuffer(out.getvalue(), dtype=np.complex128).reshape(SHAPE[::-1]).T
    assert _rel(total, ref) < WINDOW_TOL
    assert len(k0_frames) == 1 and len(k0_frames[0]) == 1          # one estimate for the whole stream
    assert abs(k0_frames[0][0] - k0_ref[0][0]) / abs(k0_ref[0][0]) < 1e-9


def test_k0_of_two_repetitions_is_the_estimate_of_their_mean(tmp_path):
    info, fid = _setup(tmp_path, n_rep=2)
    k0_frames = []
    _engine(tmp_path, info, fid, k0_out=k0_frames, estimate_k0=True)
    npro = int(info["NPro"])
    half = len(fid) // 2
    a = np.frombuffer(fid[:half], dtype="<i4").astype(np.float64)
    b = np.frombuffer(fid[half:], dtype="<i4").astype(np.float64)
    mean = ((a + b) / 2.0).tobytes()
    from brkraw_sordino import kcentre
    options = _opts(tmp_path / "m", frame_spokes="repetition", estimate_k0=True)
    k0_ref = []
    recon_dataobj(io.BytesIO(mean), trajectory_rows(info, options), dict(info, NRepetitions=1), io.BytesIO(),
                  options, override_buffer_size=len(mean), override_dtype=np.float64,
                  phase_factor=phase_correction_rows(info, options, N_POINTS),
                  virtual_traj=kcentre.leading_points(info, 1), k0_out=k0_ref)
    assert npro == 800
    assert abs(k0_frames[0][0] - k0_ref[0][0]) / abs(k0_ref[0][0]) < 1e-9


# ------------------------------------------------------------------ through the hook
def _hook_env(monkeypatch, info, fid):
    monkeypatch.setattr(hook, "_parse_recon_info", lambda scan: dict(info))
    monkeypatch.setattr(hook, "_get_fid_entry", lambda scan: _Entry(fid))


def test_hook_returns_the_frames_and_records_them(tmp_path, monkeypatch):
    info, fid = _setup(tmp_path)
    info = dict(info, RepetitionTime_ms=0.625)
    _hook_env(monkeypatch, info, fid)
    kw = {"cache_dir": str(tmp_path / "h"), "frame_spokes": 120, "frame_step": 40}
    got = hook.get_dataobj_info(_Scan(), None, **kw)
    assert got["frames"] == 18 and got["shape"][-1] == 18
    scan = _Scan()
    re, im = hook.get_dataobj(scan, None, as_complex=True, **kw)
    assert re.shape[-1] == 18
    _, imgs = _engine(tmp_path, info, fid, frame_spokes=120, frame_step=40)
    from brkraw_sordino.orientation import correct as correct_orientation
    assert _rel(re[..., 3] + 1j * im[..., 3], correct_orientation(imgs[3], info)) < 1e-12
    meta = scan._sordino_recon_meta["frames"]
    assert meta["trajectory_mode"] == "GoldenSampling" and meta["unit_spokes"] == 40
    assert meta["frame_spokes"] == 120 and meta["frame_step"] == 40 and meta["n_frames"] == 18
    assert meta["accumulate"] is False and meta["repetitions"] == 1
    assert meta["spoke_ranges"][3] == [120, 240]
    assert meta["frame_interval_s"] == pytest.approx(40 * 0.625e-3)
    assert meta["frame_centre_s"][3] == pytest.approx(180 * 0.625e-3)
    one = hook.get_dataobj(_Scan(), None, frames=3, **kw)
    assert one.shape == tuple(re.shape[:-1])


def test_hook_default_for_a_golden_scan_is_subset_frames(tmp_path, monkeypatch):
    info, fid = _setup(tmp_path)
    _hook_env(monkeypatch, info, fid)
    got = hook.get_dataobj_info(_Scan(), None, cache_dir=str(tmp_path / "h"))
    assert got["frames"] == 20
    rep = hook.get_dataobj_info(_Scan(), None, cache_dir=str(tmp_path / "h"), frame_spokes="repetition")
    assert rep["frames"] == 1


def test_hook_k0_with_frames_records_one_estimate(tmp_path, monkeypatch):
    info, fid = _setup(tmp_path)
    _hook_env(monkeypatch, info, fid)
    scan = _Scan()
    hook.get_dataobj(scan, None, cache_dir=str(tmp_path / "h"), estimate_k0=True)
    meta = scan._sordino_recon_meta
    assert len(meta["k0"]) == 1 and meta["frames"]["k0_scope"] == "all spokes read"


def test_frame_plan_is_in_the_cache_key_and_repetition_keys_are_unchanged(tmp_path, monkeypatch):
    info, fid = _setup(tmp_path)
    _hook_env(monkeypatch, info, fid)
    paths = set()
    for kw in ({}, {"frame_spokes": "repetition"}, {"frame_spokes": 80}, {"frame_spokes": 80, "frame_step": 40},
               {"frame_spokes": 80, "frame_accumulate": True}):
        paths.add(hook.get_dataobj_info(_Scan(), None, cache_dir=str(tmp_path / "k"), **kw)["cache_path"])
    assert len(paths) == 5
    # the default and "subset" give the same plan, so the same cache
    a = hook.get_dataobj_info(_Scan(), None, cache_dir=str(tmp_path / "k"))["cache_path"]
    b = hook.get_dataobj_info(_Scan(), None, cache_dir=str(tmp_path / "k"), frame_spokes="subset")["cache_path"]
    assert a == b
    # one repetition per frame: no frame entry and no frame option in the key (keys from before CP3)
    options = _opts(tmp_path / "k", frame_spokes="repetition")
    params = hook._build_cache_params(_Scan(), None, _Entry(fid), options, dict(info))
    assert "frames" not in params
    assert not set(params["options"]) & {"frame_spokes", "frame_step", "frame_accumulate"}
