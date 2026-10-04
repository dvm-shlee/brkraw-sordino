"""Memory and disk check before a SORDINO reconstruction or cache read (WI-0071, D-0097 1).

``get_dataobj`` estimates, before it reconstructs or reads anything, how much
memory the returned data need and how much disk the recon cache needs, and
stops with ``SordinoResourceError`` when an estimate is above the limit, so a
full reconstruction of a large scan is not started by accident on a small
computer (baseline: an 8 GB laptop, D-0094).

Memory estimate: the returned arrays plus three cache frames (read path,
measured in WI-0071 M1 with the frame-by-frame reader: 1.0 x the result). When
no cache exists, the reconstruction runs first in the same process and
``recon_nbytes`` is added (D-0098 2). The reconstruction is serial (WI-0097,
D-0133): each frame is cut into spoke chunks, so its working memory is a fixed
part (image grids) plus one chunk, independent of the spoke and frame counts
(``recon_plan``); ``estimate_k0`` adds the smaller of its two solves: the Toeplitz
form, grids only (``k0_fixed_nbytes``, WI-0097 stage 2), or the sample-based form,
which grows with the samples of a frame (``k0_samples_nbytes``; ``k0_method``,
WI-0099, D-0136). What the limit leaves after the read is
the budget that sets the chunk size; the check stops only when even the
smallest chunk does not fit. The read and reconstruction parts are added
although they do not peak at the same time (conservative).

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
#: Interpreter-side fixed allowance of the reconstruction step (WI-0071).
RECON_FIXED_BYTES = 16 * MIB
#: Serial reconstruction (WI-0097, D-0133; design WI-0096): chunk cap in samples
#: (about 13 M; larger chunks did not save time at 160^3), the smallest chunk tried
#: when the limit is tight, and the per-sample bytes of one chunk (WI-0096 fit on 29
#: measured rows at 160^3: 120 B per sample plus 60 B per sample and receiver).
CHUNK_SAMPLES_CAP = 13_000_000
MIN_CHUNK_SPOKES = 256
SERIAL_SAMPLE_BYTES = 120
SERIAL_SAMPLE_RX_BYTES = 60
#: estimate_k0 with the Toeplitz solve (WI-0097 stage 2, D-0133 1): grids only, no samples.
#: Counted per output voxel (complex128 = 16 B; the 2N grid has 8 voxels per voxel): the
#: kernel on 2N (128 B), its NUFFT fine grid at upsampling 1.25 (250 B), the real kernel
#: FFT (64 B), three 2N FFT buffers of one CG step (384 B), five CG vectors (80 B); 906 B,
#: rounded up to 1024 B although the kernel and CG buffers do not coexist (conservative).
#: Plus a fixed part for the extra FFT and NUFFT plans. Fitted against 34 measured rows
#: (32^3 to 160^3, 1-16 receivers, tests/test_recon_memory_measured.py).
K0_VOXEL_BYTES = 1024
K0_FIXED_BYTES = 64 * MIB
#: estimate_k0 with the sample-based normal operator (WI-0099, D-0136; ``serial.SampleNormal``):
#: per output voxel the two NUFFT fine grids at upsampling 1.25 (2 x 31 B) and the CG vectors
#: x, r, p, A^H W A p (64 B), 126 B, rounded to 128 B; per sample of a frame the three radian
#: columns and the weight (32 B), two plans' sort indices (16 B) and the type-2 result (16 B),
#: 64 B, rounded up to 80 B; 16 MiB for the plans. Fitted against 30 measured rows (32^3 to
#: 160^3, 1-16 receivers; tests/test_recon_memory_measured.py): estimate 1.18-1.84 x measured.
K0_SAMPLE_VOXEL_BYTES = 128
K0_SAMPLE_BYTES = 80
K0_SAMPLE_FIXED_BYTES = 16 * MIB
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


def _check_sizes(n_pro, n_points, n_receivers, volume_shape):
    n_pro = int(n_pro)
    n_points = int(n_points)
    n_receivers = int(n_receivers)
    vol = [int(v) for v in volume_shape]
    if n_pro <= 0 or n_points <= 0 or n_receivers <= 0:
        raise ValueError("recon_nbytes needs positive sizes")
    if len(vol) != 3 or any(v <= 0 for v in vol):
        raise ValueError("volume_shape must have 3 positive entries")
    return n_pro, n_points, n_receivers, vol


def serial_fixed_nbytes(n_receivers, volume_shape) -> int:
    """Chunk-independent part of the serial adjoint: the frame image of every channel
    and its written copy, one NUFFT output and the NUFFT grid (2x per axis), complex128."""
    vox = int(volume_shape[0]) * int(volume_shape[1]) * int(volume_shape[2])
    return int(n_receivers) * vox * 16 * 2 + vox * 16 + 8 * vox * 16 + RECON_FIXED_BYTES


def chunk_nbytes(chunk_spokes, n_points, n_receivers) -> int:
    """Chunk-dependent part: per sample the trajectory rows, radians, weights and NUFFT
    point data, per sample and channel the FID bytes and complex k-space copies."""
    samples = int(chunk_spokes) * int(n_points)
    return int(math.ceil(samples * (SERIAL_SAMPLE_BYTES + int(n_receivers) * SERIAL_SAMPLE_RX_BYTES)))


def k0_fixed_nbytes(volume_shape) -> int:
    """Extra chunk-independent memory of estimate_k0 (Toeplitz solve, WI-0097 stage 2):
    independent of the spoke, sample and receiver counts (the solve runs one channel at
    a time on grids; the kernel is shared by all channels and frames)."""
    vox = int(volume_shape[0]) * int(volume_shape[1]) * int(volume_shape[2])
    return K0_VOXEL_BYTES * vox + K0_FIXED_BYTES


def k0_samples_nbytes(n_pro, n_points, volume_shape) -> int:
    """Extra chunk-independent memory of estimate_k0 with the sample-based normal operator
    (``serial.SampleNormal``, WI-0099): it holds the points of every spoke of a frame, so it
    grows with ``n_pro * n_points``; independent of the receiver count (one channel at a
    time, plans shared by channels and frames)."""
    vox = int(volume_shape[0]) * int(volume_shape[1]) * int(volume_shape[2])
    samples = int(n_pro) * int(n_points)
    return K0_SAMPLE_VOXEL_BYTES * vox + K0_SAMPLE_BYTES * samples + K0_SAMPLE_FIXED_BYTES


def k0_method(n_pro, n_points, volume_shape) -> Dict[str, Any]:
    """The estimate_k0 solve with the smaller memory estimate (WI-0099, D-0136).

    ``"samples"`` (``serial.SampleNormal``) when its estimate is below the Toeplitz one
    (``k0_fixed_nbytes``), otherwise ``"toeplitz"`` (``serial.ToeplitzKernel``; also on a
    tie, the 858de87 behaviour). Depends only on the scan geometry, not on the memory limit
    or the receiver count, so a scan always takes the same path; both give the same K0 and
    image within the NUFFT tolerance (tests/test_serial_recon.py).

    Returns ``method``, ``k0_nbytes`` (the chosen estimate), ``samples_nbytes`` and
    ``toeplitz_nbytes``.
    """
    samples = k0_samples_nbytes(n_pro, n_points, volume_shape)
    toeplitz = k0_fixed_nbytes(volume_shape)
    method = "samples" if samples < toeplitz else "toeplitz"
    return {"method": method, "k0_nbytes": int(min(samples, toeplitz)),
            "samples_nbytes": int(samples), "toeplitz_nbytes": int(toeplitz)}


def recon_plan(n_pro, n_points, n_receivers, volume_shape, *, estimate_k0=False,
               budget_nbytes=None) -> Dict[str, Any]:
    """Chunk size of the serial reconstruction and its memory (WI-0097, D-0133 1 and 4).

    The chunk is the largest that keeps ``chunk_spokes * n_points`` at or below
    ``CHUNK_SAMPLES_CAP`` and, with ``budget_nbytes``, the estimate at or below the
    budget; the spokes are then cut into equal chunks. Larger chunks than the cap
    do not save time (WI-0096: 4 chunks of 12.9 M samples took the time of one
    chunk, 4.2 s, with 3.42 instead of 11.13 GiB), so the cap applies even when
    memory is plentiful. When even ``MIN_CHUNK_SPOKES`` spokes do not fit,
    ``fits`` is False and the estimate is the one for that chunk (the memory check
    then stops and offers that limit). Deterministic: no seed, no measurement.

    With ``estimate_k0`` the K0 term is the smaller of the two solves (``k0_method``).

    Returns ``chunk_spokes``, ``n_chunks``, ``chunk_samples``, ``fixed_nbytes``,
    ``chunk_nbytes``, ``recon_nbytes`` (the estimate), ``fits`` and ``k0_method``
    (``"samples"``, ``"toeplitz"`` or None without ``estimate_k0``).
    """
    n_pro, n_points, n_receivers, vol = _check_sizes(n_pro, n_points, n_receivers, volume_shape)
    cap = max(1, CHUNK_SAMPLES_CAP // n_points)
    smallest = min(MIN_CHUNK_SPOKES, n_pro)
    fixed = serial_fixed_nbytes(n_receivers, vol)
    method = None
    if estimate_k0:
        k0 = k0_method(n_pro, n_points, vol)
        method = k0["method"]
        fixed += k0["k0_nbytes"]
    chunk = min(n_pro, cap)
    per_spoke = chunk_nbytes(1, n_points, n_receivers)
    if budget_nbytes is not None and per_spoke > 0:
        room = int(budget_nbytes) - fixed
        chunk = min(chunk, max(room // per_spoke, 0))
        chunk = max(chunk, smallest)
    n_chunks = -(-n_pro // chunk)
    chunk = -(-n_pro // n_chunks)                       # equal chunks
    part = chunk_nbytes(chunk, n_points, n_receivers)
    total = fixed + part
    fits = budget_nbytes is None or total <= int(budget_nbytes)
    return {"chunk_spokes": int(chunk), "n_chunks": int(n_chunks),
            "chunk_samples": int(chunk * n_points), "fixed_nbytes": int(fixed),
            "chunk_nbytes": int(part), "recon_nbytes": int(total), "fits": bool(fits),
            "k0_method": method}


def recon_nbytes(n_pro, n_points, n_receivers, volume_shape, *, estimate_k0=False,
                 spoketiming_nbytes=0, budget_nbytes=None) -> int:
    """Memory the reconstruction step needs on top of the returned data.

    The serial reconstruction (``recon_plan``): the fixed part plus one chunk, which
    does not grow with the spoke count. The spoke-timing stage, when it runs
    first, needs ``spoketiming_nbytes`` (its segment work); the larger of the two
    stages is returned. The trajectory and phase factor are made per chunk and
    are not held during that stage (WI-0097).
    """
    est = recon_plan(n_pro, n_points, n_receivers, volume_shape, estimate_k0=estimate_k0,
                     budget_nbytes=budget_nbytes)["recon_nbytes"]
    return max(est, int(spoketiming_nbytes))


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
    "recon_plan", "recon_nbytes", "serial_fixed_nbytes", "chunk_nbytes", "k0_fixed_nbytes",
    "k0_samples_nbytes", "k0_method",
    "free_disk_bytes", "check",
]
