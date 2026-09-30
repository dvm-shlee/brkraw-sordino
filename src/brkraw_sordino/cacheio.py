"""Frame-by-frame reading of the SORDINO reconstruction cache (WI-0071).

The cache is a headerless file of complex values: frame ``n`` is one block of
``prod(frame_shape) * itemsize`` bytes at byte ``n * block``, Fortran order
inside the block (``recon.recon_dataobj`` writes it). Reading one frame at a
time into one reusable buffer keeps the memory near one result, where reading
the whole file and then taking the magnitude needed about three results
(WI-0071 M1: 3.01-3.11 x on real v1 data).
"""
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np


def frame_nbytes(dtype, frame_shape) -> int:
    """Size of one cache frame in bytes."""
    return int(np.prod(frame_shape)) * np.dtype(dtype).itemsize


def read_recon_frames(
    path: Union[str, Path],
    dtype,
    cached_shape: Sequence[int],
    frames: Optional[Sequence[int]] = None,
    *,
    as_complex: bool = False,
    combine_channels: bool = False,
) -> np.ndarray:
    """Read ``frames`` (default: all) of a recon cache, one frame at a time.

    ``cached_shape`` is ``(x, y, z, F)`` or ``(ch, x, y, z, F)``. Each frame is
    converted to magnitude unless ``as_complex``, and with ``combine_channels``
    multi-channel frames are combined over the channel axis (root sum of
    squares of the magnitudes, or the complex sum). The values equal reading
    the whole file and applying the same steps. The frame axis is last, in the
    order of ``frames``.
    """
    if len(cached_shape) not in (4, 5):
        raise ValueError("cached_shape must have 4 or 5 entries")
    dtype = np.dtype(dtype)
    frame_shape = tuple(int(v) for v in cached_shape[:-1])
    n_total = int(cached_shape[-1])
    nbytes = frame_nbytes(dtype, frame_shape)

    if frames is None:
        order = list(range(n_total))
    else:
        order = [int(k) for k in frames]
        if not order:
            raise ValueError("frames selects no frame")
        for k in order:
            if not 0 <= k < n_total:
                raise IndexError(f"frame {k} out of range for {n_total} frames")

    combine = combine_channels and len(frame_shape) == 4
    buf = np.empty(int(np.prod(frame_shape)), dtype=dtype)
    raw = memoryview(buf).cast("B")
    out: Optional[np.ndarray] = None
    with open(path, "rb") as handle:
        for i, k in enumerate(order):
            handle.seek(k * nbytes)
            if handle.readinto(raw) != nbytes:
                raise ValueError(f"cache file is truncated at frame {k}")
            frame = buf.reshape(frame_shape, order="F")
            # the buffer is reused for the next frame; out[..., i] = result copies it
            result = frame if as_complex else np.abs(frame)
            if combine:
                result = np.sum(result, axis=0) if as_complex else np.sqrt(np.sum(result ** 2, axis=0))
            if out is None:
                out = np.empty(tuple(result.shape) + (len(order),), dtype=result.dtype, order="F")
            out[..., i] = result
    assert out is not None
    return out


__all__ = ["frame_nbytes", "read_recon_frames"]
