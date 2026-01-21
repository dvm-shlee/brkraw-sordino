import logging
import os
import sys
import time
from typing import Any, Optional, TextIO, cast

logger = logging.getLogger("brkraw.sordino")


def progressbar(iterable: Any, *, desc: str = "", ncols: int = 100, disable: Optional[bool] = None):
    """Lightweight progress iterator (tqdm-like) without external deps."""
    if disable is None:
        disable = str(os.environ.get("BRKRAW_NO_PROGRESS", "")).strip().lower() in {"1", "true", "yes", "y", "on"}
    stream: TextIO = cast(
        TextIO,
        sys.__stderr__
        or sys.stderr
        or sys.__stdout__
        or sys.stdout
        or open(os.devnull, "w", encoding="utf-8"),
    )
    try:
        is_tty = bool(getattr(stream, "isatty", lambda: False)())
    except Exception:
        is_tty = False
    if disable or not is_tty or not logger.isEnabledFor(logging.INFO):
        return iterable

    try:
        total = len(iterable)  # type: ignore[arg-type]
    except Exception:
        total = 0

    if total <= 0:
        return iterable

    bar_width = max(10, min(40, ncols - max(0, len(desc)) - 20))
    start = time.time()
    last_emit = 0.0

    def _emit(i: int) -> None:
        nonlocal last_emit
        now = time.time()
        if now - last_emit < 0.1 and i < total:
            return
        last_emit = now
        frac = min(1.0, max(0.0, i / total))
        filled = int(bar_width * frac)
        bar = "#" * filled + "-" * (bar_width - filled)
        elapsed = max(0.001, now - start)
        rate = i / elapsed if i > 0 else 0.0
        remaining = max(0, total - i)
        eta = int(remaining / rate) if rate > 0 else -1
        eta_txt = f"{eta}s" if eta >= 0 else "?"
        prefix = f"{desc} " if desc else ""
        line = f"{prefix}[{bar}] {i}/{total} ETA {eta_txt}"
        try:
            stream.write("\r" + line)
            stream.flush()
        except Exception:
            pass

    def _done() -> None:
        try:
            stream.write("\r" + (" " * (ncols if ncols > 0 else 120)) + "\r\n")
            stream.flush()
        except Exception:
            pass

    def _iter():
        for i, item in enumerate(iterable, start=1):
            _emit(i)
            yield item
        _done()

    return _iter()


__all__ = [
    "progressbar",
]
