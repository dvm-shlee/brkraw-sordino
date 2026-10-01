"""estimate_k0 is announced with one INFO line also when get_dataobj reads the recon cache
(WI-0087, D-0115 1). The line is only a log: values, recon cache key, trajectory and the
returned bytes are the same with and without it."""
import logging

import numpy as np
import pytest

from brkraw_sordino import hook

from test_hook_read import _Scan, _get, _reference, setup  # noqa: F401


@pytest.fixture
def make(setup, monkeypatch):  # noqa: F811
    # the synthetic scan has no sequence timing: keep estimate_k0 as the caller set it
    monkeypatch.setattr(hook, "_resolve_k0", lambda options, recon_info: options)

    def fail(*a, **k):
        raise AssertionError("reconstructed or built a trajectory on a cache hit")

    monkeypatch.setattr(hook, "recon_dataobj", fail)
    monkeypatch.setattr(hook, "get_trajectory", fail)
    return setup


def _k0_lines(caplog):
    return [r for r in caplog.records
            if r.name == "brkraw_sordino.hook" and r.levelno == logging.INFO
            and "estimate_k0" in r.getMessage()]


def test_cache_hit_logs_one_info_line_when_estimate_k0_is_on(make, caplog):
    state = make(estimate_k0=True)
    with caplog.at_level(logging.INFO, logger="brkraw_sordino.hook"):
        _get(state)
    assert len(_k0_lines(caplog)) == 1


def test_cache_hit_logs_nothing_when_estimate_k0_is_off(make, caplog):
    state = make()
    with caplog.at_level(logging.INFO, logger="brkraw_sordino.hook"):
        _get(state)
    assert not _k0_lines(caplog)


@pytest.mark.parametrize("flag", ["True", "true", True])
def test_the_line_follows_the_parsed_option(make, caplog, flag):
    state = make(estimate_k0=flag)
    with caplog.at_level(logging.INFO, logger="brkraw_sordino.hook"):
        _get(state)
    assert len(_k0_lines(caplog)) == 1


def test_values_and_cache_key_do_not_depend_on_the_log(make, caplog):
    state = make(estimate_k0=True)
    path = state["plan"]["img_cache_path"]
    meta_path = hook._cache_meta_path(path)
    before = (path.read_bytes(), meta_path.read_bytes())
    files = sorted(p.name for p in path.parent.iterdir())

    quiet = _get(state)
    with caplog.at_level(logging.INFO, logger="brkraw_sordino.hook"):
        loud = _get(state)
    assert len(_k0_lines(caplog)) == 1

    assert len(quiet) == len(loud)
    for a, b in zip(quiet, loud):
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
    for a, b in zip(loud, _reference(state)):
        assert np.array_equal(a, b)
    # same key: the plan finds the same cache file, nothing was written or added
    assert hook._plan(_Scan(), None, dict(state["kwargs"]))["img_cache_path"] == path
    assert (path.read_bytes(), meta_path.read_bytes()) == before
    assert sorted(p.name for p in path.parent.iterdir()) == files
