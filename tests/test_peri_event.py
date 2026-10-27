"""Peri-event frames: one image per time bin relative to a repeating event (perievent.py).

The geometry throughout is the Golden SORDINO resting scan: NPro 288,000 spokes
per repetition, 160 spokes per method subset, a 0.625 ms spoke TR, so one subset
is 0.1 s, one repetition 180 s, and an event every 20 s is every 32,000 spokes.

What the tests hold to:

* the plan's ranges and groups place bin b of event k at the right spokes, and
  bins that tile one period use every spoke of the scan exactly once;
* ``scales`` normalises each bin to the brightness of a one-repetition image,
  whatever number of events reached it;
* the ranges are sorted by start, so the forward-only FID stream never rewinds;
* the recon cache key separates two different event layouts;
* the reconstruction equals summing the same spokes by hand, and a one-bin
  peri-event plan equals the ordinary frame of the same single range.
"""
from pathlib import Path

import numpy as np
import pytest

from brkraw_sordino import frames, perievent
from brkraw_sordino.typing import Options

from test_golden_recon import _golden_info

NPRO = 288000
SUBSET = 160
TR_MS = 0.625
PERIOD_S = 20.0
PERIOD_SPOKES = 32000          # 20 s / 0.625 ms


def _info(n_rep=20):
    info = _golden_info()
    info.update(Matrix=[120, 120, 120], NPoints=240, NGoldenSubsets=1800,
                NGoldenSpokesPerSubset=SUBSET, NGoldenSteps=NPRO, NPro=NPRO,
                O1List_Hz=[0.0], NRepetitions=n_rep, RepetitionTime_ms=TR_MS)
    return info


def _opts(**kw):
    base = dict(ext_factors=(1.0, 1.0, 1.0), ignore_samples=1, offset=0, num_frames=None,
                correct_spoketiming=False, correct_ramptime=True, offreso_freqs=(), mem_limit=0.5,
                clear_cache=True, split_ch=False, cache_dir=Path("/tmp"), as_complex=False)
    base.update(kw)
    return Options(**base)


def _plan(spec, n_rep=20, **kw):
    return frames.make_plan(_info(n_rep), _opts(peri_event=spec, **kw))


# ---------------------------------------------------------------- option parsing
@pytest.mark.parametrize("text, want", [
    ("period=20,bin=1", {"period": 20.0, "bin": 1.0}),
    ("period=20,bin=1,onset=1", {"period": 20.0, "bin": 1.0, "onset": 1.0}),
    (" period = 20 , bin = 0.1 ", {"period": 20.0, "bin": 0.1}),
    ({"period": 20, "bin": 2}, {"period": 20.0, "bin": 2.0}),
    # the CLI splits --hook-arg values on commas, so the hook sees a sequence of pairs
    # or the repr of one, not the string that was typed
    (("period=20", "bin=1", "onset=1"), {"period": 20.0, "bin": 1.0, "onset": 1.0}),
    (["period=20", "bin=1"], {"period": 20.0, "bin": 1.0}),
    ("('period=20', 'bin=1', 'onset=1')", {"period": 20.0, "bin": 1.0, "onset": 1.0}),
    ('["period=20", "bin=1"]', {"period": 20.0, "bin": 1.0}),
])
def test_parse_accepts_the_cli_string_and_a_mapping(text, want):
    assert perievent.parse_peri_event(text) == want


@pytest.mark.parametrize("value", [None, False, ""])
def test_parse_unset_is_none(value):
    assert perievent.parse_peri_event(value) is None


@pytest.mark.parametrize("text, message", [
    ("period=20", "bin"),                      # bin missing
    ("bin=1", "period"),                       # period missing
    ("period=20,bin=1,nope=3", "unknown"),     # unknown field
    ("period=20,bin", "key=value"),            # not a pair
    ("period=x,bin=1", "not a number"),
    ("period=0,bin=1", "must be positive"),
    ("period=20,bin=1,bins=0", "whole number"),
])
def test_parse_rejects_bad_specs(text, message):
    with pytest.raises(ValueError, match=message):
        perievent.parse_peri_event(text)


# ---------------------------------------------------------------- the plan
def test_bins_tile_one_period_by_default():
    plan = _plan("period=20,bin=1")
    assert plan.n_frames == 20                      # 20 s / 1 s
    assert plan.window == 1600                      # 1 s at 0.625 ms
    assert plan.step == PERIOD_SPOKES


def test_ranges_are_the_spokes_of_bin_b_of_event_k():
    plan = _plan("period=20,bin=1,onset=1")
    onset = 1600                                    # 1 s
    by_bin = {}
    for (lo, hi), b in zip(plan.ranges, plan.groups):
        by_bin.setdefault(b, []).append((lo, hi))
    for b in range(plan.n_frames):
        for k, (lo, hi) in enumerate(sorted(by_bin[b])):
            assert lo == onset + PERIOD_SPOKES * k + 1600 * b
            assert hi == lo + 1600


def test_tiling_bins_use_every_spoke_exactly_once():
    plan = _plan("period=20,bin=1")
    total = 20 * NPRO
    used = np.zeros(total, dtype=np.int32)
    for lo, hi in plan.ranges:
        used[lo:hi] += 1
    assert used.min() == 1 and used.max() == 1      # a partition of the scan


def test_a_short_span_leaves_spokes_out_but_never_doubles_them():
    plan = _plan("period=20,bin=1,bins=5")          # 5 s of each 20 s period
    total = 20 * NPRO
    used = np.zeros(total, dtype=np.int32)
    for lo, hi in plan.ranges:
        used[lo:hi] += 1
    assert used.max() == 1
    assert used.sum() == len(plan.ranges) * 1600


def test_ranges_are_sorted_so_the_fid_stream_runs_forward():
    plan = _plan("period=20,bin=1,onset=1")
    starts = [lo for lo, _ in plan.ranges]
    assert starts == sorted(starts)


def test_scales_normalise_each_bin_to_one_repetition():
    plan = _plan("period=20,bin=1")
    counts = plan.peri["events_per_bin"]
    for b, scale in enumerate(plan.scales):
        assert scale == pytest.approx(NPRO / (counts[b] * 1600))


def test_a_bin_cut_off_at_the_end_of_the_data_keeps_the_events_it_has():
    # one repetition holds 9 events from t=1 s; the last event's final bin runs past the end
    plan = _plan("period=20,bin=1,onset=1", n_rep=1)
    counts = plan.peri["events_per_bin"]
    assert counts[:-1] == [9] * 19 and counts[-1] == 8
    assert plan.peri["dropped_bins"] == 1


def test_a_bin_short_of_events_is_scaled_by_its_own_count():
    plan = _plan("period=20,bin=1,onset=1", n_rep=1)
    counts = plan.peri["events_per_bin"]
    # the short bin is brightened to match the others, not left dim
    assert plan.scales[-1] == pytest.approx(NPRO / (counts[-1] * 1600))
    assert plan.scales[-1] > plan.scales[0]


def test_an_event_shifted_before_the_data_is_dropped_not_clipped():
    # start=-1 s puts bin 0 of event 0 before the first spoke; no range may be negative
    plan = _plan("period=20,bin=1,onset=0,start=-1")
    assert min(lo for lo, _ in plan.ranges) >= 0
    assert plan.peri["dropped_bins"] >= 1


def test_events_caps_the_event_count():
    plan = _plan("period=20,bin=1,onset=1,events=90")
    assert max(plan.peri["events_per_bin"]) == 90


def test_the_number_of_events_follows_the_repetitions_read():
    assert max(_plan("period=20,bin=1", n_rep=10).peri["events_per_bin"]) == 90
    assert max(_plan("period=20,bin=1", n_rep=20).peri["events_per_bin"]) == 180


def test_subset_sized_and_longer_bins_both_work():
    for bin_s, spokes in ((0.1, 160), (0.2, 320), (1.0, 1600), (2.0, 3200)):
        plan = _plan(f"period=20,bin={bin_s}")
        assert plan.window == spokes
        assert plan.n_frames == int(round(PERIOD_S / bin_s))


def test_the_reported_phases_match_the_trajectory_period():
    # 20 s = 32,000 spokes and NPro = 288,000 share a factor of 32,000: 9 phases
    assert _plan("period=20,bin=1").peri["trajectory_phases"] == 9
    # 19 s = 30,400 spokes is coprime enough to reach 1,800 of them
    assert _plan("period=19,bin=1").peri["trajectory_phases"] > 9


def test_a_bin_without_spokes_is_kept_as_an_empty_bin():
    # one repetition is 180 s; 400 bins of 1 s reach far past it, so the late bins are empty
    plan = _plan("period=20,bin=1,bins=400", n_rep=1)
    counts = plan.peri["events_per_bin"]
    assert plan.n_frames == 400 and counts[0] > 0 and counts[-1] == 0
    assert plan.scales[-1] == 0.0


def test_peri_event_turns_on_frame_mode():
    assert frames.is_frame_mode(_opts(peri_event="period=20,bin=1"), _info())


# ---------------------------------------------------------------- cache key
def test_two_event_layouts_do_not_share_a_cache_key():
    a = _plan("period=20,bin=1,onset=1").key()
    b = _plan("period=20,bin=1,onset=2").key()
    c = _plan("period=20,bin=1,onset=1,events=90").key()
    assert a != b and a != c and b != c


def test_an_ordinary_frame_plan_keeps_its_old_key():
    plan = frames.make_plan(_info(), _opts(frame_spokes=3200))
    assert set(plan.key()) == {"unit", "window", "step", "accumulate"}
    assert plan.groups is None


def test_the_frame_interval_is_the_bin_width_not_the_event_period():
    """pixdim[4] must be the spacing of the output images: one bin, not one period."""
    plan = _plan("period=20,bin=1")
    assert plan.frame_interval_s == pytest.approx(1.0)
    assert _plan("period=20,bin=0.2").frame_interval_s == pytest.approx(0.2)
    # an ordinary plan still reports frame_step x TR
    ordinary = frames.make_plan(_info(), _opts(frame_spokes=1600, frame_step=3200))
    assert ordinary.frame_interval_s == pytest.approx(2.0)


def test_describe_reports_the_bins_and_their_times():
    meta = _plan("period=20,bin=1,onset=1").describe()
    assert meta["peri_event"]["n_bins"] == 20
    assert len(meta["groups"]) == len(meta["spoke_ranges"])
    centres = meta["frame_centre_s"]
    assert len(centres) == 20
    assert centres[0] == pytest.approx(0.5)         # centre of the 0-1 s bin
    assert centres[-1] == pytest.approx(19.5)


# ---------------------------------------------------------------- reconstruction
# The small golden scan of test_golden_frames (NPro 800, 20 subsets of 40, a 16^3
# grid). A spoke TR of 1 ms makes a second 1,000 spokes, so the specs below are
# round spoke counts: period=0.2 s is 200 spokes, bin=0.04 s is one 40-spoke subset.
from test_golden_frames import _engine, _setup, _rel, WINDOW_TOL      # noqa: E402

TR_1MS = 1.0


def _small(tmp_path, n_rep=3):
    info, fid = _setup(tmp_path, n_rep=n_rep)
    info = dict(info, RepetitionTime_ms=TR_1MS)
    return info, fid


def test_one_bin_equals_the_mean_of_the_ordinary_frames_of_the_same_ranges(tmp_path):
    """A bin is the union of its ranges, so it is the mean of those frames.

    Each ordinary frame is NPro / window x the raw adjoint of its spokes; the bin
    is NPro / (events x window) x the sum of the same adjoints. The bin is
    therefore exactly the mean of the frames, which is what an event-related
    average should be.
    """
    info, fid = _small(tmp_path)
    peri, bins = _engine(tmp_path / "p", info, fid, peri_event="period=0.2,bin=0.04,bins=1")
    plan, frames_imgs = _engine(tmp_path / "f", info, fid, frame_spokes=40, frame_step=200)
    assert peri.n_frames == 1
    assert plan.ranges == peri.ranges                      # the same spokes, grouped differently
    assert _rel(bins[0], np.mean(frames_imgs, axis=0)) < WINDOW_TOL


def test_every_bin_equals_a_one_bin_plan_shifted_to_it(tmp_path):
    """Bin b of a tiling plan is the same image as a single bin started at b x bin."""
    info, fid = _small(tmp_path)
    peri, bins = _engine(tmp_path / "p", info, fid, peri_event="period=0.2,bin=0.04")
    assert peri.n_frames == 5                              # 200 / 40
    for b in range(peri.n_frames):
        one, only = _engine(tmp_path / f"s{b}", info, fid,
                            peri_event=f"period=0.2,bin=0.04,bins=1,start={0.04 * b:.2f}")
        assert one.n_frames == 1
        assert [lo for lo, _ in one.ranges] == [lo for (lo, _), g in zip(peri.ranges, peri.groups)
                                                if g == b]
        assert _rel(bins[b], only[0]) < WINDOW_TOL, b


def test_a_bin_that_is_one_whole_repetition_equals_that_repetition_frame(tmp_path):
    """period = bin = one repetition: each bin is one repetition, so no averaging happens."""
    info, fid = _small(tmp_path, n_rep=2)
    peri, bins = _engine(tmp_path / "p", info, fid, peri_event="period=0.8,bin=0.8,bins=1")
    plan, imgs = _engine(tmp_path / "f", info, fid, frame_spokes=800, frame_step=800)
    assert peri.n_frames == 1 and plan.n_frames == 2
    assert _rel(bins[0], np.mean(imgs, axis=0)) < WINDOW_TOL


def test_bins_are_written_in_time_order(tmp_path):
    """Bin b must hold the b-th window of every event, not the b-th range of the stream."""
    info, fid = _small(tmp_path)
    peri, bins = _engine(tmp_path / "p", info, fid, peri_event="period=0.2,bin=0.04")
    for b in range(peri.n_frames):
        starts = [lo for (lo, _), g in zip(peri.ranges, peri.groups) if g == b]
        assert all(s % 200 == 40 * b for s in starts), b
