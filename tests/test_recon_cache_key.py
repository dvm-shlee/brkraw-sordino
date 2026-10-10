"""The recon cache key holds only what changes the cache content (WI-0071, D-0098 3).

``as_complex``, ``split_ch``, ``clear_cache`` and ``cache_dir`` change what
``get_dataobj`` returns from the cache, whether leftovers are deleted, or where
the cache lives, but not the reconstructed values, so they are not in the key:
changing them reads the same cache instead of reconstructing again. Every
other option stays in the key (a new option is in the key unless it is listed).
"""
from dataclasses import asdict, fields

import numpy as np
import pytest

from brkraw_sordino import hook
from brkraw_sordino.recon import build_recon_cache_path
from brkraw_sordino.spoketiming import build_spoketiming_cache_path
from brkraw_sordino.typing import Options

from test_hook_read import _Fid, _Scan, _recon_info, NF, VOL  # noqa: F401


def _params(tmp_path, **kw):
    options = hook._build_options(dict(cache_dir=str(tmp_path), **kw))
    return hook._build_cache_params(_Scan(), None, _Fid(), options, _recon_info(1))


def _paths(tmp_path, **kw):
    p = _params(tmp_path, **kw)
    return build_recon_cache_path(tmp_path, p), build_spoketiming_cache_path(tmp_path, p)


def test_the_excluded_options_are_exactly_the_four():
    assert hook.RECON_KEY_EXCLUDED == ("as_complex", "split_ch", "clear_cache", "cache_dir")
    names = {f.name for f in fields(Options)}
    assert set(hook.RECON_KEY_EXCLUDED) <= names


def test_every_other_option_is_in_the_key(tmp_path):
    # the frame options of golden scans (WI-0113 CP3) are keyed through the resolved frame
    # plan ("frames" entry, tests/test_golden_frames.py), so keys without frames are unchanged
    from brkraw_sordino.frames import FRAME_KEYS

    keyed = set(_params(tmp_path)["options"])
    names = {f.name for f in fields(Options)}
    assert keyed == names - set(hook.RECON_KEY_EXCLUDED) - set(FRAME_KEYS)


@pytest.mark.parametrize("kw", [
    {"as_complex": True}, {"split_ch": True}, {"clear_cache": False},
    {"as_complex": True, "split_ch": True, "clear_cache": False},
])
def test_excluded_options_share_one_cache(tmp_path, kw):
    assert _paths(tmp_path, **kw) == _paths(tmp_path)


def test_cache_dir_is_not_in_the_key(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    pa = _params(a)
    pb = _params(b)
    assert build_recon_cache_path(a, pa).name == build_recon_cache_path(b, pb).name
    assert build_spoketiming_cache_path(a, pa).name == build_spoketiming_cache_path(b, pb).name
    assert "cache_dir" not in pa["options"]


@pytest.mark.parametrize("kw", [
    {"ext_factors": 2}, {"ignore_samples": 2}, {"offset": 1}, {"num_frames": 2},
    {"correct_spoketiming": True}, {"correct_ramptime": False}, {"offreso_freqs": 120.0},
    {"mem_limit": 1.0}, {"estimate_k0": True},
])
def test_content_options_change_the_key(tmp_path, kw):
    assert _paths(tmp_path, **kw)[0] != _paths(tmp_path)[0]


def test_existing_caches_are_not_reused_once(tmp_path):
    """The old key (all options) never equals the new one: old recon caches are
    reconstructed once, never read under a wrong key."""
    options = hook._build_options(dict(cache_dir=str(tmp_path)))
    new = _params(tmp_path)
    old = dict(new, options=asdict(options))
    assert build_recon_cache_path(tmp_path, old) != build_recon_cache_path(tmp_path, new)


def test_as_complex_reads_the_magnitude_cache_without_reconstructing(tmp_path, monkeypatch):
    info = _recon_info(1)
    monkeypatch.setattr(hook, "_parse_recon_info", lambda scan: dict(info))
    monkeypatch.setattr(hook, "_get_fid_entry", lambda scan: _Fid())
    plan = hook._plan(_Scan(), None, {"cache_dir": str(tmp_path)})
    shape = list(VOL) + [NF]
    rng = np.random.default_rng(9)
    data = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype("<c16")
    data.flatten(order="F").tofile(plan["img_cache_path"])
    import json
    hook._cache_meta_path(plan["img_cache_path"]).write_text(
        json.dumps({"dtype": "<c16", "shape": shape, "kspace_gap": None, "k0": None}))

    def fail(*a, **k):
        raise AssertionError("reconstructed although a cache with the same content exists")

    monkeypatch.setattr(hook, "recon_dataobj", fail)
    monkeypatch.setattr(hook, "get_trajectory", fail)
    for kw in ({}, {"as_complex": True}, {"split_ch": True}, {"clear_cache": False}):
        out = hook.get_dataobj(_Scan(), None, cache_dir=str(tmp_path), **kw)
        assert out is not None
    re, im = hook.get_dataobj(_Scan(), None, cache_dir=str(tmp_path), as_complex=True)
    from brkraw_sordino.orientation import correct
    assert np.array_equal(re, correct(np.real(data), info))
    assert np.array_equal(im, correct(np.imag(data), info))
