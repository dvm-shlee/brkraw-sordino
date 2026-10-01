"""Memory and disk check before a SORDINO reconstruction or cache read (WI-0071, D-0097 1).

``get_dataobj`` estimates, before it reconstructs or reads anything, how much
memory the returned data need and how much disk the recon cache needs, and
stops with ``SordinoResourceError`` when an estimate is above the limit, so a
full reconstruction of a large scan is not started by accident on a small
computer (baseline: an 8 GB laptop, D-0094).

Memory estimate: the returned arrays plus three cache frames (read path,
measured in WI-0071 M1 with the frame-by-frame reader: 1.0 x the result). When
no cache exists, the reconstruction runs first in the same process and
``recon_nbytes`` is added (D-0098 2): with ``gc.collect()`` after every frame
its working memory does not grow with the frame count, so it is a size of one
frame's work (trajectory, k-space, NUFFT grid), fitted to measurements. The
two are added although they do not peak at the same time (conservative).

Limit: the ``max_memory_gb`` option, otherwise half of this computer's
physical memory (4 GB when it cannot be read; D-0098 4).
"""
from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

GIB = 1024 ** 3
DEFAULT_FRACTION = 0.5
FALLBACK_LIMIT_BYTES = 4 * GIB
MIB = 1024 ** 2
#: Reconstruction working memory model (WI-0071, D-0098 2), fitted to reconstructions
#: with gc.collect() after every frame: 21 synthetic shapes up to 128^3 and 6.6 M
#: samples per frame, 1-16 receivers, and 2 real v1 runs. Checked again in WI-0095
#: at 160^3 with 51.8 M samples per frame (NPoints 640, OverSampling 8), 1-4
#: receivers, with and without estimate_k0. At or above every measured peak RSS,
#: +3 % to +55 % (pinned in tests/test_recon_memory_measured.py).
RECON_TRAJ_FACTOR = 6.0       # x trajectory bytes (3 float64 per sample)
RECON_KSPACE_FACTOR = 12.0    # x one frame of complex128 k-space, all channels
RECON_GRID_FACTOR = 1.1       # x one NUFFT grid (2x oversampled per axis, complex128)
RECON_FIXED_BYTES = 16 * MIB
#: estimate_k0 adds a least-squares solve per frame and channel (kcentre.fill_centre):
#: measured +9.1 x trajectory + 0.8 x grid on 5 synthetic shapes (after wi-0071-choi-4 F4).
RECON_K0_TRAJ_FACTOR = 10.0
RECON_K0_GRID_FACTOR = 1.0
#: Spoke-timing correction works on one FID segment of all selected frames at a time:
#: measured 4.1 x the segment (real v1, 30 frames, one segment); 5.0 keeps a margin.
SPOKETIMING_FACTOR = 5.0


class SordinoResourceError(MemoryError):
    """An estimate is above the memory limit or the free disk space.

    ``kind`` is ``"memory"`` or ``"disk"``; ``info`` is the estimate
    (``get_dataobj_info``) that was checked. ``retry_kwargs`` is the keyword
    arguments that would pass the check (the memory stop gives
    ``{"max_memory_gb": GB}``, used by the brkraw CLI to offer a retry), or None.
    """

    def __init__(self, message: str, *, kind: str, info: Dict[str, Any], retry_kwargs=None):
        super().__init__(message)
        self.kind = kind
        self.info = info
        self.retry_kwargs = dict(retry_kwargs) if retry_kwargs else None


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


def recon_nbytes(n_pro, n_points, n_receivers, volume_shape, *, estimate_k0=False,
                 spoketiming_nbytes=0) -> int:
    """Memory the reconstruction step needs on top of the returned data.

    ``estimate_k0`` adds the least-squares centre estimate. The spoke-timing
    stage, when it runs first, needs ``spoketiming_nbytes`` (its segment work)
    plus the trajectory and phase factor that are already held; the larger of
    the two stages is returned.
    """
    n_pro = int(n_pro)
    n_points = int(n_points)
    n_receivers = int(n_receivers)
    vol = [int(v) for v in volume_shape]
    if n_pro <= 0 or n_points <= 0 or n_receivers <= 0:
        raise ValueError("recon_nbytes needs positive sizes")
    if len(vol) != 3 or any(v <= 0 for v in vol):
        raise ValueError("volume_shape must have 3 positive entries")
    samples = n_pro * n_points
    traj = samples * 3 * 8
    kspace = samples * n_receivers * 16
    grid = 8 * vol[0] * vol[1] * vol[2] * 16
    est = int(math.ceil(RECON_TRAJ_FACTOR * traj + RECON_KSPACE_FACTOR * kspace
                        + RECON_GRID_FACTOR * grid)) + RECON_FIXED_BYTES
    if estimate_k0:
        est += int(math.ceil(RECON_K0_TRAJ_FACTOR * traj + RECON_K0_GRID_FACTOR * grid))
    stc = int(spoketiming_nbytes)
    if stc > 0:
        stc += traj + samples * 8      # trajectory (float64 x 3) and phase factor (complex64)
    return max(est, stc)


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
        # the smallest limit in 0.1 GB steps that holds the estimate (retry_kwargs)
        peak = int(info["peak_nbytes"])
        gb = math.ceil(peak * 10 / GIB) / 10
        if gb < 0.1:
            gb = 0.1
        if int(gb * GIB) < peak:
            gb = round(gb + 0.1, 1)
        gb = float(gb)
        raise SordinoResourceError(
            "SORDINO: returning {n} frame(s) of shape {shape} ({dtype}, {count} array(s)) needs about "
            "{peak} of memory, above the limit of {limit} ({source}). Nothing was {what}. "
            "Ask for fewer frames (frames=..., or num_frames=... and offset=...), or raise the limit "
            "with max_memory_gb=<GB> if this computer can hold it.".format(
                n=info["frames"], shape=tuple(info["shape"]), dtype=info["dtype"],
                count=info["count"], peak=_gib(info["peak_nbytes"]),
                limit=_gib(info["limit_nbytes"]), source=info["limit_source"],
                what="read" if info["cached"] else "reconstructed"),
            kind="memory", info=info, retry_kwargs={"max_memory_gb": gb})
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
    "recon_nbytes", "free_disk_bytes", "check",
]
