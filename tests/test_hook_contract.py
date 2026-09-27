"""Smoke tests: the hook imports against the installed brkraw and keeps the
converter-hook contract (brkraw passes a hook only the keyword arguments
its functions accept; **kwargs lets --hook-arg values through)."""

import inspect


def test_hook_imports_with_installed_brkraw():
    from brkraw_sordino import hook  # every brkraw name the hook uses must exist

    assert set(hook.HOOK) == {"get_dataobj", "get_affine", "convert"}


def test_every_hook_function_accepts_kwargs():
    from brkraw_sordino.hook import HOOK

    for name, func in HOOK.items():
        kinds = {p.kind for p in inspect.signature(func).parameters.values()}
        assert inspect.Parameter.VAR_KEYWORD in kinds, name
