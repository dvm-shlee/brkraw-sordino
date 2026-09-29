"""Hook options after BRK-0066: bool parsing, removed keys, estimate_k0 rules."""
import logging

import pytest

from brkraw_sordino.hook import _build_options, _resolve_k0


def _opts(tmp_path, **kw):
    return _build_options(dict(cache_dir=str(tmp_path), **kw))


def test_defaults(tmp_path):
    o = _opts(tmp_path)
    assert o.correct_ramptime is True and o.estimate_k0 is False
    assert not hasattr(o, "ramp_model") and not hasattr(o, "correct_phase")


@pytest.mark.parametrize("name", ["correct_spoketiming", "correct_ramptime", "clear_cache",
                                  "split_ch", "as_complex", "estimate_k0"])
def test_every_bool_option_reads_strings_case_insensitively(tmp_path, name):
    for text, value in (("false", False), ("False", False), ("FALSE", False), ("true", True),
                        ("True", True), ("TRUE", True), (True, True), (False, False)):
        kw = {name: text}
        if name == "estimate_k0":
            kw["correct_ramptime"] = True
        assert getattr(_opts(tmp_path, **kw), name) is value, (name, text)
    with pytest.raises(ValueError, match=name):
        _opts(tmp_path, **{name: "maybe"})


def test_string_false_is_not_true(tmp_path):
    # the defect BRK-0066 fixes: bool("false") is True
    assert _opts(tmp_path, correct_ramptime="false").correct_ramptime is False
    assert _opts(tmp_path, split_ch="false").split_ch is False


def test_estimate_k0_needs_correct_ramptime(tmp_path):
    with pytest.raises(ValueError, match="estimate_k0"):
        _opts(tmp_path, estimate_k0="true", correct_ramptime="false")
    assert _opts(tmp_path, estimate_k0="TRUE").estimate_k0 is True
    # off together is fine
    o = _opts(tmp_path, estimate_k0="false", correct_ramptime="false")
    assert (o.estimate_k0, o.correct_ramptime) == (False, False)


def test_removed_and_unknown_keys_warn_once(tmp_path, caplog):
    with caplog.at_level(logging.DEBUG, logger="brkraw_sordino.hook"):
        o = _opts(tmp_path, ramp_model="legacy", correct_phase=False, nonsense=1)
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1
    text = warns[0].getMessage()
    assert "correct_phase" in text and "nonsense" in text and "ramp_model" in text
    assert o.correct_ramptime is True                      # ignored, not aliased
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="brkraw_sordino.hook"):
        _opts(tmp_path, correct_ramptime=True)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def _info(version):
    from test_ramp_model import _info as info
    return info(version)


def test_estimate_k0_is_switched_off_for_general_zte(tmp_path, caplog):
    o = _opts(tmp_path, estimate_k0=True)
    with caplog.at_level(logging.INFO, logger="brkraw_sordino.hook"):
        o = _resolve_k0(o, _info("zte"))
    assert o.estimate_k0 is False
    assert [r for r in caplog.records if "general ZTE" in r.getMessage()]
    o2 = _resolve_k0(_opts(tmp_path, estimate_k0=True), _info("v2"))
    assert o2.estimate_k0 is True
    o3 = _resolve_k0(_opts(tmp_path), _info("zte"))
    assert o3.estimate_k0 is False


def test_estimate_k0_changes_the_recon_cache_key(tmp_path):
    from dataclasses import asdict
    from brkraw_sordino.recon import build_recon_cache_path

    a, b = _opts(tmp_path), _opts(tmp_path, estimate_k0=True)
    pa = build_recon_cache_path(tmp_path, {"options": asdict(a)})
    pb = build_recon_cache_path(tmp_path, {"options": asdict(b)})
    assert pa != pb
