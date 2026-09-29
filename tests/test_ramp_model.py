"""Tests for the integral ramp model and phase correction (WI-0056, BRK-0056)."""
import numpy as np
import pytest

from brkraw_sordino import ramp, timing
from brkraw_sordino.hook import _build_options
from brkraw_sordino.recon import phase_correction_factor
from brkraw_sordino.traj import (
    calc_npro,
    calc_radial_grad3d,
    calc_radial_traj3d,
    calc_radial_traj3d_integral,
    get_trajectory,
)

MATRIX, OS = 16, 2.0
N = int(MATRIX / 2 * OS)


def _info(version="v2", ramp_ms=None, delay_ms=0.1645833, adt=6.75, rf_ms=0.004,
          bw=75000.0, o1=None):
    npro = 2 * calc_npro(MATRIX, 1.0)
    dwell_ms = 1e3 / (bw * OS)
    acq_ms = N * dwell_ms
    info = {
        "Matrix": [MATRIX] * 3, "NPro": npro, "HalfAcquisition": False,
        "UseOrigin": False, "Reorder": False, "OverSampling": OS,
        "AcqDelayTotal_us": adt, "EffBandwidth_Hz": bw, "ExcPulLength_ms": rf_ms,
        "RampTime_ms": ramp_ms if ramp_ms is not None else (4.75e-3 + acq_ms + 0.005),
        "RampDelay_ms": delay_ms, "RFWait_ms": None, "MaximizeRampTime": None,
        "TrigSegmentMode": "Off", "O1List_Hz": o1 if o1 is not None else [0.0],
    }
    if version == "v1":
        info["TrigSegmentMode"] = None
    elif version == "v3":
        info["TrigSegmentMode"], info["RampDelay_ms"] = None, None
        info["MaximizeRampTime"], info["RFWait_ms"] = "Yes", 0.01
    elif version == "zte":
        info["TrigSegmentMode"], info["RampDelay_ms"], info["RampTime_ms"] = None, None, None
    return info


def _grad(info):
    return calc_radial_grad3d(MATRIX, info["NPro"], False, False, False)


def _traj(info):
    seq = timing.read_timing(info)
    tune = timing.tuning_for(seq.version)
    times, F, _ = timing.ramp_terms(seq, tune, N)
    return calc_radial_traj3d_integral(_grad(info), MATRIX, OS, times, F, seq.dwell_us), seq


def _numeric_k(g_prev, g_cur, t_end_us, start, length, dwell_us, unit):
    s = np.linspace(0.0, t_end_us, 40001)
    f = np.clip((s - start) / length, 0.0, 1.0)
    G = g_prev[None, :] + (g_cur - g_prev)[None, :] * f[:, None]
    return unit * np.trapezoid(G, s, axis=0) / dwell_us


def test_detect_version():
    assert timing.detect_version(_info("v1")) == "v1"
    assert timing.detect_version(_info("v2")) == "v2"
    assert timing.detect_version(_info("v3")) == "v3"
    assert timing.detect_version(_info("zte")) == "zte"


def test_ramp_windows_follow_the_sequence_source():
    rf = 4.0
    v1 = timing.ramp_window(timing.read_timing(_info("v1", delay_ms=0.0436375)), timing.TimingTuning())
    assert v1[0] == pytest.approx(-(10.0 + 43.6375 + rf / 2))   # before the RF
    v2 = timing.ramp_window(timing.read_timing(_info("v2")), timing.TimingTuning())
    v3 = timing.ramp_window(timing.read_timing(_info("v3")), timing.TimingTuning())
    assert v2[0] == pytest.approx(rf / 2) and v3[0] == pytest.approx(rf / 2)  # RF end
    assert timing.ramp_window(timing.read_timing(_info("zte")), timing.TimingTuning()) is None


@pytest.mark.parametrize("version,ramp_ms", [("v1", 0.615), ("v2", None), ("v3", 0.3)])
def test_closed_form_matches_numeric_integral(version, ramp_ms):
    info = _info(version, ramp_ms=ramp_ms, delay_ms=0.0436375)
    traj, seq = _traj(info)
    g = _grad(info)
    start, length = timing.ramp_window(seq, timing.TimingTuning())
    unit = 1.0 / (N - 1) / 2.0
    times, _, _ = timing.ramp_terms(seq, timing.TimingTuning(), N)
    for i in (1, 7, len(g[0]) - 1):
        for j in (0, N // 2, N - 1):
            k = _numeric_k(g[:, i - 1], g[:, i], times[j], start, length, seq.dwell_us, unit)
            np.testing.assert_allclose(traj[i, j], k, atol=1e-6)


def test_constant_gradient_equals_correct_ramptime_off():
    info = _info("zte")
    traj, seq = _traj(info)
    off = seq.acq_delay_total_us / seq.dwell_us
    ref = calc_radial_traj3d(_grad(info), MATRIX, False, OS, correct_ramptime=False, traj_offset=off)
    np.testing.assert_allclose(traj, ref, atol=1e-12)


def test_short_ramp_v1_reaches_target_before_sampling():
    # ramp of 20 us that ends before the RF centre (starts ~70 us before it)
    info = _info("v1", ramp_ms=0.020, delay_ms=0.060)
    traj, seq = _traj(info)
    g = _grad(info)
    unit = 1.0 / (N - 1) / 2.0
    step = np.diff(traj, axis=1)
    np.testing.assert_allclose(step, unit * g.T[:, None, :] * np.ones((1, N - 1, 1)), atol=1e-12)
    _, _, tau = timing.ramp_terms(seq, timing.TimingTuning(), N)
    assert np.allclose(tau, 0.0)            # no residual phase: constant gradient since before RF
    # a short ramp ending between the RF centre and the first sample: constant tau
    # (starts 12 us before the RF centre, 15 us long: ends at +3 us < 6.75 us)
    info2 = _info("v1", ramp_ms=0.015, delay_ms=0.0)
    seq2 = timing.read_timing(info2)
    _, _, tau2 = timing.ramp_terms(seq2, timing.TimingTuning(), N)
    start, length = timing.ramp_window(seq2, timing.TimingTuning())
    assert 0 < start + length < seq2.acq_delay_total_us
    assert np.allclose(tau2, tau2[0]) and tau2[0] > 0


def test_v2_v3_are_one_model_driven_by_ramp_vs_acquisition():
    info_v2 = _info("v2")                                   # ramp = acquisition
    seq = timing.read_timing(info_v2)
    times, _, _ = timing.ramp_terms(seq, timing.TimingTuning(), N)
    start, length = timing.ramp_window(seq, timing.TimingTuning())
    # 16 synthetic samples: last sample one dwell + 5 us before the ramp end
    assert ramp.ramp_fraction(times[-1], start, length) == pytest.approx(
        (times[-1] - start) / length)
    assert 0.85 < ramp.ramp_fraction(times[-1], start, length) < 1.0
    # real v2 scan (#10): NPoints 256, dwell 1/600 kHz, RampTime 0.43641667 ms
    f_real = ramp.ramp_fraction(6.75 + 255 * 1e6 / 600000.0, 2.0, 436.41667)
    assert 0.98 < f_real < 1.0
    long_ramp = (info_v2["RampTime_ms"]) * 2                # ramp ~ 2x acquisition
    t2, _ = _traj(_info("v2", ramp_ms=long_ramp))
    t3, _ = _traj(_info("v3", ramp_ms=long_ramp))
    np.testing.assert_allclose(t2, t3, atol=1e-12)          # same model, same numbers
    seq3 = timing.read_timing(_info("v3", ramp_ms=long_ramp))
    s3, l3 = timing.ramp_window(seq3, timing.TimingTuning())
    f_last = ramp.ramp_fraction(times[-1], s3, l3)
    assert 0.4 < f_last < 0.6                               # ends midway, target not reached


def test_legacy_option_reproduces_previous_trajectory(tmp_path):
    info = _info("v2")
    options = _build_options({"cache_dir": str(tmp_path), "ramp_model": "legacy"})
    traj = get_trajectory(info, options)
    off = info["AcqDelayTotal_us"] * 1e-6 * info["EffBandwidth_Hz"] * OS
    ref = calc_radial_traj3d(_grad(info), MATRIX, False, OS, correct_ramptime=True, traj_offset=off)
    np.testing.assert_array_equal(traj, ref)
    integral = get_trajectory(info, _build_options({"cache_dir": str(tmp_path)}))
    assert not np.allclose(integral, ref)                    # different cache entry and values


def test_phase_factor():
    info = _info("v2")

    class O:  # minimal options
        correct_phase = True
        correct_ramptime = True
        ramp_model = "integral"
    assert phase_correction_factor(info, O, N) is None       # O1 list of length 1
    g = _grad(info)
    o1 = (1000.0 * g[0] - 500.0 * g[1] + 250.0 * g[2]).tolist()
    info["O1List_Hz"] = o1
    p = phase_correction_factor(info, O, N)
    assert p.shape == (info["NPro"], N)
    seq = timing.read_timing(info)
    _, _, tau = timing.ramp_terms(seq, timing.TimingTuning(), N)
    d = np.roll(np.asarray(o1), 1) - np.asarray(o1)
    np.testing.assert_allclose(np.angle(p[5, 3]),
                               np.angle(np.exp(-2j * np.pi * d[5] * tau[3] * 1e-6)), atol=1e-5)
    O.ramp_model = "legacy"
    assert phase_correction_factor(info, O, N) is None
    O.ramp_model, O.correct_phase = "integral", False
    assert phase_correction_factor(info, O, N) is None


def test_ramp_model_option_validation(tmp_path):
    with pytest.raises(ValueError):
        _build_options({"cache_dir": str(tmp_path), "ramp_model": "quadratic"})
    o = _build_options({"cache_dir": str(tmp_path), "correct_phase": "false"})
    assert o.correct_phase is False and o.ramp_model == "integral"


def test_general_zte_warns_about_dead_time(tmp_path, caplog):
    # triggerzte3-like timing: 6.2 us dead time, 0.625 us dwell, oversampling 4
    info = _info("zte", adt=6.2, bw=400000.0)
    info["OverSampling"] = 4.0
    seq = timing.read_timing(info)
    assert timing.dead_time_kgrid(seq, 4.0) == pytest.approx(6.2 / 0.625 / 4)
    info["Matrix"] = [8, 8, 8]
    info["NPro"] = 2 * calc_npro(8, 1.0)
    with caplog.at_level("WARNING", logger="brkraw_sordino.traj"):
        get_trajectory(info, _build_options({"cache_dir": str(tmp_path)}))
    assert any("General ZTE" in r.message for r in caplog.records)
    # real SORDINO v2 numbers (6.75 us, dwell 1/600 kHz, oversampling 8): ~0.5 unit
    seq2 = timing.SeqTiming("v2", 4.0, 6.75, 1e6 / 600000.0, 436.4, 164.6)
    assert timing.dead_time_kgrid(seq2, 8.0) == pytest.approx(0.50625)
