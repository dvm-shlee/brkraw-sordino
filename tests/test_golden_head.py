"""The Default head of Golden Grid scans of ``sordino_260801`` (WI-0113 run 5, D-0197 decision 1).

The sequence's reco relation (``RecoRelations.c``) fills the gradient lists with
``radialGrad3D(matrix, ProUnderSampling, ...)`` whenever ``GoldenSampTraj`` is No,
which is the case for Golden Grid: the first N entries (N = the count that
call writes) hold the Default list, the golden list stays from N on. The
scanner's ``traj`` file shows exactly this, and the data of scans 17 and 21 fit
these directions with the receiver frequency of the golden list (WI-0113 run 4,
``r4_head.json``; an inference from the data, not documented ParaVision
behaviour). The product reconstructs such scans with:

* the played list: golden, entries 0..N-1 replaced by the Default list of
  ``radialGrad3D`` with ``ProUnderSampling`` itself;
* the phase rows: the ramp step of the true receiver frequencies
  (O1_true = offset . direction on the head, the recorded value elsewhere) and the
  receiver error e = O1_true - O1_rec over the sample time from the RF centre;
* the ``ACQ_O1_list`` order check on the golden list (the recorded list is golden).

The rule is automatic (no option) and warns once. A fixed sequence is told apart by
the ``traj`` file (head equal to the golden list: no rule); without the file, only
the known version ``sordino_260801`` gets the rule.
"""
import io
import logging

import numpy as np
import pytest

from brkraw_sordino import golden, hook, kcentre
from brkraw_sordino.hook import _build_options
from brkraw_sordino.recon import nufft_adjoint, phase_correction_rows
from brkraw_sordino.traj import (calc_npro, calc_radial_grad3d, get_trajectory, gradient_list,
                                 trajectory_rows)

from test_golden_recon import OFF, _golden_info
from test_serial_recon import N_POINTS, SHAPE, _fid_frames, _rel

US = 4.0          # ProUnderSampling of the synthetic scan


def _grid_info(**head):
    info = _golden_info(1, "GoldenGridSampling")
    info["UnderSampling"] = US
    if head:
        info["GoldenGridHead"] = dict(head)
    return info


def _n_head():
    return 2 * calc_npro(SHAPE[0], US)


# ------------------------------------------------------------------ count and list
def test_head_count_is_what_radialgrad3d_writes():
    assert golden.default_head_count(120, 3.61468271736276, False, False) == 12732     # scans 17, 21
    assert golden.default_head_count(120, 3.61468271736276, True, False) == 6366
    assert golden.default_head_count(120, 3.61468271736276, False, True) == 12733
    assert golden.default_head_count(16, US, False, False) == _n_head()


def test_default_list_from_the_undersampling_itself():
    from brkraw_sordino.traj import radial_grad3d_undersampling

    g = radial_grad3d_undersampling(120, 3.61468271736276, False, False, False)
    assert g.shape == (3, 12732)
    ref = calc_radial_grad3d(120, 12732, False, False, False)      # the probe's list (run 4: 1.6e-15)
    assert np.abs(g - ref).max() < 1e-12
    gs = radial_grad3d_undersampling(16, US, False, False, False)
    assert gs.shape == (3, _n_head())


# ------------------------------------------------------------------ decision
def _traj_dirs(kind, n):
    info = _grid_info()
    if kind == "default":
        from brkraw_sordino.traj import radial_grad3d_undersampling
        return radial_grad3d_undersampling(16, US, False, False, False)[:, :n]
    if kind == "golden":
        return gradient_list(info)[0][:, :n]
    return np.full((3, n), 0.5)


def test_traj_file_with_the_default_head_turns_the_rule_on(caplog):
    n = _n_head()
    with caplog.at_level(logging.WARNING, logger="brkraw_sordino.golden"):
        h = golden.grid_head(_grid_info(), method="<User:sordino_260801>", golden_samp_traj=False,
                             traj_dirs=_traj_dirs("default", n))
    assert h["applied"] is True and h["n"] == n and h["source"] == "traj file"
    assert h["method"] == "sordino_260801" and h["traj_match"] == "default"
    text = caplog.text
    assert text.count("sordino:") == 1 and str(n) in text and "Default" in text and "traj" in text


def test_traj_file_with_the_golden_head_is_a_fixed_sequence(caplog):
    with caplog.at_level(logging.WARNING, logger="brkraw_sordino.golden"):
        h = golden.grid_head(_grid_info(), method="<User:sordino_260801>", golden_samp_traj=False,
                             traj_dirs=_traj_dirs("golden", _n_head()))
    assert h["applied"] is False and h["n"] == 0 and h["traj_match"] == "golden"
    assert caplog.text == ""


def test_traj_file_that_matches_neither_list_warns_and_keeps_the_golden_list(caplog):
    with caplog.at_level(logging.WARNING, logger="brkraw_sordino.golden"):
        h = golden.grid_head(_grid_info(), method="<User:sordino_260801>", golden_samp_traj=False,
                             traj_dirs=_traj_dirs("neither", _n_head()))
    assert h["applied"] is False and h["traj_match"] == "neither"
    assert "traj" in caplog.text and "neither" in caplog.text


@pytest.mark.parametrize("method,applied", [("<User:sordino_260801>", True), ("sordino_260801", True),
                                            ("<User:sordino_261015>", False), (None, False)])
def test_without_a_traj_file_only_the_known_version_gets_the_rule(caplog, method, applied):
    with caplog.at_level(logging.WARNING, logger="brkraw_sordino.golden"):
        h = golden.grid_head(_grid_info(), method=method, golden_samp_traj=False, traj_dirs=None)
    assert h["applied"] is applied and h["n"] == (_n_head() if applied else 0)
    assert h["source"] == "sequence version"
    if applied:
        assert "no traj file" in caplog.text


def test_golden_samp_traj_yes_or_other_modes_have_no_head():
    h = golden.grid_head(_grid_info(), method="sordino_260801", golden_samp_traj=True,
                         traj_dirs=_traj_dirs("default", _n_head()))
    assert h["applied"] is False
    assert golden.grid_head(_golden_info(1, "GoldenSampling"), method="sordino_260801",
                            golden_samp_traj=True, traj_dirs=None) is None
    from test_serial_recon import _info as serial_info
    assert golden.grid_head(serial_info(1, 1), method="sordino_260801", golden_samp_traj=False,
                            traj_dirs=None) is None


# ------------------------------------------------------------------ the hook reads the scan
class _File:
    def __init__(self, name, data):
        self.name, self._data = name, data

    def open(self):
        return io.BytesIO(self._data)


class _Params(dict):
    pass


class _TrajScan:
    def __init__(self, traj_bytes, method="<User:sordino_260801>", gst="No"):
        self._files = [_File("method", b""), _File("fid", b"")]
        if traj_bytes is not None:
            self._files.append(_File("traj", traj_bytes))
        self.method = _Params(Method=method, GoldenSampTraj=gst)

    def iterdir(self):
        return iter(self._files)


def _traj_bytes(dirs, extra_lines=3):
    mat = SHAPE[0]
    d = np.concatenate([dirs, np.tile([[0.0], [0.0], [1.0]], extra_lines)], axis=1)
    s = np.arange(mat) / mat - 0.5
    return (s[None, :, None] * d.T[:, None, :]).astype("<f8").tobytes()


@pytest.mark.parametrize("kind,applied", [("default", True), ("golden", False)])
def test_hook_decides_from_the_scan_traj_file(kind, applied):
    n = _n_head()
    scan = _TrajScan(_traj_bytes(_traj_dirs(kind, n)))
    h = hook._golden_grid_head(scan, _grid_info())
    assert h["applied"] is applied and h["source"] == "traj file"


def test_hook_without_traj_file_uses_the_version():
    assert hook._golden_grid_head(_TrajScan(None), _grid_info())["applied"] is True
    assert hook._golden_grid_head(_TrajScan(None, method="<User:other>"), _grid_info())["applied"] is False
    assert hook._golden_grid_head(_TrajScan(None, gst="Yes"), _grid_info())["applied"] is False


# ------------------------------------------------------------------ what the rule changes
def _head_info():
    return _grid_info(applied=True, n=_n_head(), source="traj file", method="sordino_260801")


def test_played_list_has_the_default_head_and_the_golden_rest():
    from brkraw_sordino.traj import radial_grad3d_undersampling, spoke_directions

    n = _n_head()
    gg, params = gradient_list(_head_info())
    gp, pp = spoke_directions(_head_info())
    assert np.array_equal(gp[:, :n], radial_grad3d_undersampling(16, US, False, False, False))
    assert np.array_equal(gp[:, n:], gg[:, n:])
    assert pp == dict(params, grid_head=n)
    g0, p0 = spoke_directions(_grid_info())
    assert np.array_equal(g0, gg) and p0 == params


def test_trajectory_and_k0_points_follow_the_played_list(tmp_path):
    from brkraw_sordino.traj import spoke_directions

    opts = _build_options({"cache_dir": str(tmp_path)})
    gp, _ = spoke_directions(_head_info())
    rows = trajectory_rows(_head_info(), opts)
    plain = trajectory_rows(_grid_info(), opts)
    n = _n_head()
    assert not np.allclose(rows.rows(0, n), plain.rows(0, n))
    assert np.array_equal(rows.rows(n + 1, rows.n_pro), plain.rows(n + 1, plain.n_pro))
    whole = get_trajectory(_head_info(), opts)
    assert np.array_equal(whole, rows.rows(0, rows.n_pro))
    vt = kcentre.leading_points(_head_info(), 1)
    assert np.array_equal(vt, kcentre.leading_points(_grid_info(), 1, grad=gp))


def test_o1_order_check_uses_the_golden_list(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="brkraw_sordino.traj"):
        trajectory_rows(_head_info(), _build_options({"cache_dir": str(tmp_path)}))
    assert "ACQ_O1_list" not in caplog.text


def _head_phase_reference(info, ramp=True):
    """Model (c) of run 4 written out: true O1 on the head, ramp step of the true list, + e * t."""
    from brkraw_sordino import timing as tm
    from brkraw_sordino.traj import radial_grad3d_undersampling

    n = info["GoldenGridHead"]["n"]
    o1 = np.asarray(info["O1List_Hz"], float)
    gg, _ = gradient_list(info)
    a = np.concatenate([np.ones((gg.shape[1], 1)), gg.T], axis=1)
    coef = np.linalg.lstsq(a, o1, rcond=None)[0]
    true = o1.copy()
    true[:n] = coef[0] + radial_grad3d_undersampling(16, US, False, False, False).T @ coef[1:]
    seq = tm.read_timing(info)
    times, _, tau = tm.ramp_terms(seq, tm.tuning_for(seq.version), N_POINTS)
    d = (np.roll(true, 1) - true) if ramp else np.zeros_like(true)
    tau = np.asarray(tau) * 1e-6 if ramp else np.zeros(N_POINTS)
    ph = np.outer(d, tau) + np.outer(true - o1, np.asarray(times) * 1e-6)
    return np.exp(-2j * np.pi * ph).astype(np.complex64), true - o1


@pytest.mark.parametrize("ramp", [True, False])
def test_phase_rows_carry_the_true_ramp_step_and_the_receiver_error(tmp_path, ramp):
    info = _head_info()
    opts = _build_options({"cache_dir": str(tmp_path), "correct_ramptime": ramp})
    rows = phase_correction_rows(info, opts, N_POINTS)
    ref, e = _head_phase_reference(info, ramp)
    n = _n_head()
    assert np.abs(e[:n]).max() > 100.0 and np.all(e[n:] == 0.0)
    assert np.abs(rows[:] - ref).max() < 1e-6
    if ramp:      # from spoke n + 1 on the rows are the plain golden rows
        plain = phase_correction_rows(_grid_info(), opts, N_POINTS)
        assert np.array_equal(rows[n + 1:], plain[n + 1:])
    else:
        assert phase_correction_rows(_grid_info(), opts, N_POINTS) is None


def test_hook_reconstructs_the_head_on_the_played_list(tmp_path, monkeypatch):
    """Data simulated on the played list with the head phase: the hook equals the whole-array
    reference on that list (same weights as the hook, the sample-based ones for golden scans)."""
    from brkraw_sordino.orientation import correct as correct_orientation
    from test_golden_density import _pipe_reference, _weighted_adjoint
    from test_golden_recon import _hook_image

    info = _head_info()
    opts = _build_options({"cache_dir": str(tmp_path / "t")})
    traj = get_trajectory(info, opts)
    factor = phase_correction_rows(info, opts, N_POINTS)[:]
    (fid,) = _fid_frames(info, traj, 1, 1, factor)
    img = _hook_image(tmp_path / "h", monkeypatch, info, fid)[..., 0]
    vol = np.frombuffer(fid, dtype="<i4").reshape((2, N_POINTS, 1, info["NPro"]), order="F")
    k = ((vol[0] + 1j * vol[1]).T[:, 0, :]) * factor
    w, _ = _pipe_reference(traj[:, 1:, :], SHAPE)
    ref = correct_orientation(_weighted_adjoint(k[:, 1:], traj[:, 1:, :], SHAPE, w), info)
    assert _rel(img, ref) < 1e-8
    plain = correct_orientation(nufft_adjoint(k[:, 1:], get_trajectory(_grid_info(), opts)[:, 1:, :],
                                              SHAPE, 1), info)
    assert _rel(img, plain) > 1e-3


def test_cache_key_of_a_golden_grid_scan_carries_the_head(tmp_path):
    from test_golden_recon import _Entry, _Scan

    opts = _build_options({"cache_dir": str(tmp_path)})
    a = hook._build_cache_params(_Scan(), None, _Entry(b""), opts, _head_info())
    b = hook._build_cache_params(_Scan(), None, _Entry(b""), opts, _grid_info())
    assert a != b and a["recon_info"]["GoldenGridHead"]["n"] == _n_head()
