"""Memory and disk check before a SORDINO reconstruction or cache read (WI-0071, D-0097 1).

``get_dataobj`` estimates, before it reconstructs or reads anything, how much
memory the returned data need and how much disk the recon cache needs, and
stops with ``SordinoResourceError`` when an estimate is above the limit, so a
full reconstruction of a large scan is not started by accident on a small
computer (baseline: an 8 GB laptop, D-0094).

Memory estimate (read path, measured in WI-0071 M1 with the frame-by-frame
reader: 1.0 x the result): the returned arrays plus three cache frames (one
read buffer and one frame temporary). When no cache exists, the reconstruction
runs first in the same process and its memory is not part of this estimate: on
a 900-frame v1 scan the whole call peaked at 2.3 x the estimate (WI-0071 M2).

Limit: the ``max_memory_gb`` option, otherwise half of this computer's
physical memory (4 GB when it cannot be read).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

GIB = 1024 ** 3
DEFAULT_FRACTION = 0.5
FALLBACK_LIMIT_BYTES = 4 * GIB


class SordinoResourceError(MemoryError):
    """An estimate is above the memory limit or the free disk space.

    ``kind`` is ``"memory"`` or ``"disk"``; ``info`` is the estimate
    (``get_dataobj_info``) that was checked.
    """

    def __init__(self, message: str, *, kind: str, info: Dict[str, Any]):
        super().__init__(message)
        self.kind = kind
        self.info = info


def physical_memory_bytes() -> Optional[int]:
    """Installed memory of this computer, or None when the system does not tell."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    if pages <= 0 or size <= 0:
        return None
    return int(pages) * int(size)


def memory_limit_bytes(max_memory_gb: Any = None) -> Dict[str, Any]:
    """The limit in bytes and where it comes from (``option``, ``half of RAM``, ``fallback``)."""
    if max_memory_gb is not None:
        try:
            value = float(max_memory_gb)
        except (TypeError, ValueError):
            raise ValueError(f"max_memory_gb must be a number of GB, not {max_memory_gb!r}") from None
        if not value > 0:
            raise ValueError(f"max_memory_gb must be > 0, not {max_memory_gb!r}")
        return {"limit_nbytes": int(value * GIB), "limit_source": "max_memory_gb option"}
    ram = physical_memory_bytes()
    if ram is None:
        return {"limit_nbytes": FALLBACK_LIMIT_BYTES, "limit_source": "fallback 4 GB"}
    return {"limit_nbytes": int(ram * DEFAULT_FRACTION), "limit_source": "half of physical memory"}


def free_disk_bytes(path: Path) -> Optional[int]:
    try:
        return int(shutil.disk_usage(str(path)).free)
    except OSError:
        return None


def _gib(n: Optional[int]) -> str:
    return "unknown" if n is None else f"{n / GIB:.2f} GiB"


def check(info: Dict[str, Any]) -> None:
    """Raise ``SordinoResourceError`` when ``info`` is above a limit; do nothing otherwise."""
    if info["peak_nbytes"] > info["limit_nbytes"]:
        raise SordinoResourceError(
            "SORDINO: returning {n} frame(s) of shape {shape} ({dtype}, {count} array(s)) needs about "
            "{peak} of memory, above the limit of {limit} ({source}). Nothing was {what}. "
            "Ask for fewer frames (frames=..., or num_frames=... and offset=...), or raise the limit "
            "with max_memory_gb=<GB> if this computer can hold it.".format(
                n=info["frames"], shape=tuple(info["shape"]), dtype=info["dtype"],
                count=info["count"], peak=_gib(info["peak_nbytes"]),
                limit=_gib(info["limit_nbytes"]), source=info["limit_source"],
                what="read" if info["cached"] else "reconstructed"),
            kind="memory", info=info)
    if not info["cached"] and info["disk_free_nbytes"] is not None \
            and info["disk_nbytes"] > info["disk_free_nbytes"]:
        raise SordinoResourceError(
            "SORDINO: the reconstruction cache needs about {need} on disk in {where}, but only {free} "
            "is free. Nothing was reconstructed. Free disk space (brkraw cache clear), set "
            "cache_dir=... to a larger disk, or reconstruct fewer frames (num_frames=...).".format(
                need=_gib(info["disk_nbytes"]), where=info["cache_dir"],
                free=_gib(info["disk_free_nbytes"])),
            kind="disk", info=info)


__all__ = [
    "SordinoResourceError", "physical_memory_bytes", "memory_limit_bytes",
    "free_disk_bytes", "check",
]
