"""Tests for tools/eval_spokes.py (WI-0056 run 6). Synthetic data only."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import eval_spokes  # noqa: E402
import offsetstack  # noqa: E402

RNG = np.random.default_rng(3)


def test_neighbour_phase_diff_matches_lee_helper():
    ph = RNG.uniform(-math.pi, math.pi, 257)
    z = 2.0 * np.exp(1j * ph)
    got = eval_spokes.neighbour_phase_diff(z)
    want = np.asarray(offsetstack.neighbour_diff(ph.tolist()))
    assert np.allclose(got, want, atol=1e-12)
    assert np.all(got > -math.pi) and np.all(got <= math.pi)


def test_neighbour_stats_principles():
    const = np.full(100, 3.0 * np.exp(0.7j))
    s = eval_spokes.neighbour_stats(const)
    assert s["nd_circular_std"] == pytest.approx(0.0, abs=1e-6)
    assert s["coherence"] == pytest.approx(1.0)
    ramp = np.exp(1j * 0.1 * np.arange(100))
    s = eval_spokes.neighbour_stats(ramp)
    # constant neighbour step 0.1 except the cyclic pair (0 vs 99)
    assert s["nd_mean_abs"] == pytest.approx((99 * 0.1 + abs(math.remainder(-9.9, 2 * math.pi))) / 100)
    assert s["coherence"] < 1.0


def test_condition_fid_phase_and_virtual():
    vol = (RNG.normal(size=(6, 10)) + 1j * RNG.normal(size=(6, 10))).astype(np.complex64)
    x, z = eval_spokes.condition_fid(vol, None)
    assert np.array_equal(x, np.arange(10)) and z is vol
    f = np.exp(-1j * RNG.uniform(-0.1, 0.1, (6, 10))).astype(np.complex64)
    x, z = eval_spokes.condition_fid(vol, f)
    assert np.allclose(z, vol * f)
    virt = np.full((6, 3), 5.0 + 0j)
    x, z = eval_spokes.condition_fid(vol, f, virt, ignore_samples=1)
    assert x.tolist() == [-2.0, -1.0, 0.0] + list(range(1, 10))
    assert np.allclose(z[:, :3], 5.0)                  # estimated leading values, sample 0 replaced
    assert np.allclose(z[:, 3:], (vol * f)[:, 1:])     # measured samples unchanged


def test_fid_curves_normalised_at_first_peak():
    vol = (np.exp(-np.arange(12) / 4.0)[None, :] * np.exp(1j * RNG.uniform(-3, 3, (8, 1)))
           * np.exp(0.05j * np.arange(12))[None, :])
    x, z = eval_spokes.condition_fid(vol, None)
    norm = float(np.abs(z[:, 1]).mean())
    c = eval_spokes.fid_curves(x, z, norm, 1.0, 6, np.arange(0, 8, 2))
    assert c["mag_mean"][1] == pytest.approx(1.0)
    assert c["mag_lines"].shape == (4, 12)
    assert c["x_phase"].tolist() == [0, 1, 2, 3, 4, 5]
    assert np.allclose(c["phase_mean"], 0.05 * (np.arange(6) - 1))   # relative to each spoke's sample 1
    assert c["mag_coherent"][1] < 1.0                                 # spokes have random phases


def _curves(names, n=16, n_lines=5, with_virtual=True):
    out = {}
    for k, name in enumerate(names):
        x = np.arange(n, dtype=float)
        if with_virtual and name == "S3z":
            x = np.arange(-3, n, dtype=float)
        lines = np.abs(RNG.normal(size=(n_lines, x.size))) + k * 1e-3
        corr = None if name in ("S0", "S1", "S2") else RNG.uniform(-0.02, 0.02, (n_lines, n))
        out[name] = {"x": x, "mag_lines": lines, "mag_mean": lines.mean(0), "mag_coherent": lines.mean(0) / 2,
                     "x_phase": x[x < 8], "phase_lines": RNG.normal(size=(n_lines, int((x < 8).sum()))),
                     "phase_mean": np.zeros(int((x < 8).sum())), "correction": corr}
    return out


def test_draw_stack_rows_never_overlap(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["S0", "S1", "S3", "S3z"]
    rows = {n: {"x": np.arange(5.0), "mean": RNG.normal(size=5) * (i + 1),
                "lines": RNG.normal(size=(3, 5)) * 4} for i, n in enumerate(names)}
    fig, ax = plt.subplots()
    lay = eval_spokes.draw_stack(ax, names, rows, clip_pct=1.0)
    plt.close(fig)
    bands = [(lay["lo"] + o, lay["hi"] + o) for o in lay["offsets"]]
    for upper, lower in zip(bands, bands[1:]):
        assert upper[0] >= lower[1]
    assert lay["step"] == pytest.approx((lay["hi"] - lay["lo"]) * 1.25)


def test_figures_are_written(tmp_path):
    names = ["S0", "S1", "S1p", "S3", "S3z"]
    curves = _curves(names)
    eval_spokes.plot_accumulated(tmp_path / "acc.png", curves, names, "t", 1, 1, 12, 8)
    eval_spokes.plot_correction(tmp_path / "corr.png", curves, names, "t", np.arange(5))

    class S:
        def __init__(self, traj, phase):
            self.traj, self.phase, self.recon = traj, phase, "adjoint"

    stages = {"S0": S("off", "none"), "S1": S("legacy", "none"), "S3": S("integral", "integral"),
              "S3z": S("integral", "integral")}
    fp = np.exp(1j * RNG.uniform(-3, 3, 20))
    factors = {"none": None, "integral": np.exp(-1j * RNG.uniform(-0.01, 0.01, (20, 4)))}
    kpos = {m: {"displacement_kgrid": RNG.uniform(0, 0.05, 20), "angle_deg": RNG.uniform(0, 4, 20)}
            for m in ("off", "legacy", "integral")}
    eval_spokes.plot_first_peak(tmp_path, fp, factors, stages, kpos, 1, "t")
    stats = {n: [eval_spokes.neighbour_stats(fp) for _ in range(4)] for n in stages}
    eval_spokes.plot_neighbour_over_volumes(tmp_path / "nb.png", stats, list(stages), 10, "t")
    for name in ("acc.png", "corr.png", "first_peak_phase_within_volume.png", "first_peak_parts_and_k.png",
                 "nb.png"):
        assert (tmp_path / name).stat().st_size > 1000
