import os
import json
import logging
import numpy as np
from dataclasses import asdict, replace
from pathlib import Path

from typing import Any, Optional, Tuple, Dict, Union, cast

from nibabel.nifti1 import Nifti1Image

from brkraw.specs.remapper import load_spec, map_parameters
from brkraw.resolver import fid as fid_resolver
from brkraw.resolver import datatype as dtype_resolver
from brkraw.core import config as config_core
from brkraw.core.fs import DatasetFile
from brkraw.core.zip import ZippedFile
from brkraw.apps.loader.helper import get_affine as get_affine_helper

from numpy.typing import NDArray
from .typing import Options
from .boolopt import parse_bool
from . import kcentre
from .traj import get_trajectory, trajectory_rows  # noqa: F401  (get_trajectory: tests patch it)
from .recon import (
    build_recon_cache_path,
    get_dataobj_shape,
    get_num_frames,
    parse_fid_info,
    parse_volume_shape,
    recon_dataobj,
)
from .spoketiming import (
    build_spoketiming_cache_path,
    prep_fid_segmentation,
    correct_spoketiming,
)
from .orientation import axis_order as orientation_axis_order
from .orientation import correct as correct_orientation
from .recon import phase_correction_factor, phase_correction_rows  # noqa: F401
from .timing import TIMING_TUNING

FileIO = Union[DatasetFile, ZippedFile]
logger = logging.getLogger(__name__)
config_core.configure_logging()

def _normalize_ext_factors(value: Any) -> Tuple[float, float, float]:
    if value is None:
        return (1.0, 1.0, 1.0)
    if isinstance(value, (int, float)):
        val = float(value)
        return (val, val, val)
    if isinstance(value, (list, tuple, np.ndarray)):
        items = list(value)
        if len(items) == 1:
            val = float(items[0])
            return (val, val, val)
        if len(items) == 3:
            return (float(items[0]), float(items[1]), float(items[2]))
    raise ValueError("ext_factors must be a scalar or a 3-item sequence")


def _offreso_item(item: Any) -> Optional[float]:
    if item is None:
        return None
    if isinstance(item, (bool, np.bool_)):
        raise ValueError(f"offreso_freqs must be numbers in Hz, got {item!r}")
    if isinstance(item, str):
        try:
            number = float(item.strip())
        except ValueError:
            raise ValueError(f"offreso_freqs must be numbers in Hz, got {item!r}") from None
    elif isinstance(item, (int, float, np.integer, np.floating)):
        number = float(item)
    else:
        raise ValueError(f"offreso_freqs must be numbers in Hz, got {item!r}")
    if not np.isfinite(number):
        raise ValueError(f"offreso_freqs must be finite numbers in Hz, got {item!r}")
    return number


def _normalize_offreso_freqs(value: Any) -> Tuple[Optional[float], ...]:
    """``offreso_freqs`` as a tuple of floats in Hz, one per receive channel (WI-0106).

    A number, a string (``"120"``, ``"120,-80"``, ``"120 -80"``, ``"[120, -80]"``) and a
    list, tuple or array all read as the same values; nothing means no correction.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if len(text) >= 2 and (text[0], text[-1]) in (("[", "]"), ("(", ")")):
            text = text[1:-1]
        parts = text.replace(",", " ").split()
        return tuple(_offreso_item(part) for part in parts)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(_offreso_item(item) for item in value)
    return (_offreso_item(value),)


def _get_cache_dir(path: Optional[Union[str, Path]]) -> Path:
    if path:
        base = Path(path).expanduser()
    else:
        base = config_core.resolve_root(None) / "cache" / "sordino"
    base.mkdir(parents=True, exist_ok=True)
    return base


#: Read-time options (WI-0071): they choose what get_dataobj returns from the
#: recon cache and how much memory it may use, so they are not part of
#: ``Options`` and not part of the recon cache key. ``allow_short_fid``
#: (WI-0109) only decides whether a short FID stops the run; the frames kept
#: from it are in ``recon_info`` (and so in the key) either way.
READ_KEYS = ("frames", "axis", "max_memory_gb", "allow_short_fid")
#: Names accepted for the frame axis (the repetition axis, data axis 3).
FRAME_AXIS_NAMES = ("cycle", "repetition")


def _build_options(kwargs: Dict[str, Any]) -> Options:
    logger.debug("Sordino hook kwargs: %s", kwargs)
    known_keys = {
        "cache_dir",
        "ext_factors",
        "ignore_samples",
        "offset",
        "num_frames",
        "correct_spoketiming",
        "correct_ramptime",
        "offreso_freqs",
        "mem_limit",
        "clear_cache",
        "split_ch",
        "as_complex",
        "estimate_k0",
    } | set(READ_KEYS)
    unknown_keys = sorted(set(kwargs.keys()) - known_keys)
    if unknown_keys:
        removed = [key for key in unknown_keys if key in ("ramp_model", "correct_phase")]
        hint = (f"\n  {' and '.join(removed)} {'was' if len(removed) == 1 else 'were'} removed; "
                "correct_ramptime now covers both") if removed else ""
        logger.warning("sordino: ignoring unknown option(s): %s%s", ", ".join(unknown_keys), hint)
    cache_dir = _get_cache_dir(kwargs.get("cache_dir"))
    logger.debug("Cache dir: %s", cache_dir)
    offreso_freqs = _normalize_offreso_freqs(kwargs.get("offreso_freqs"))

    correct_ramptime = parse_bool("correct_ramptime", kwargs.get("correct_ramptime", True))
    estimate_k0 = parse_bool("estimate_k0", kwargs.get("estimate_k0", False))
    if estimate_k0 and not correct_ramptime:
        raise ValueError(
            "estimate_k0=true needs correct_ramptime=true: the estimated centre samples "
            "lie on the ramp-corrected trajectory")

    return Options(
        ext_factors=_normalize_ext_factors(kwargs.get("ext_factors")),
        ignore_samples=int(kwargs.get("ignore_samples", 1)),
        offset=int(kwargs.get("offset", 0)),
        num_frames=kwargs.get("num_frames"),
        correct_spoketiming=parse_bool("correct_spoketiming", kwargs.get("correct_spoketiming", False)),
        correct_ramptime=correct_ramptime,
        offreso_freqs=offreso_freqs,
        mem_limit=float(kwargs.get("mem_limit", 0.5)),
        clear_cache=parse_bool("clear_cache", kwargs.get("clear_cache", True)),
        split_ch=parse_bool("split_ch", kwargs.get("split_ch", False)),
        cache_dir=cache_dir,
        as_complex=parse_bool("as_complex", kwargs.get("as_complex", False)),
        estimate_k0=estimate_k0,
    )


def _resolve_k0(options: Options, recon_info: Dict[str, Any]) -> Options:
    """``estimate_k0`` applies to SORDINO v1-v3 only (BRK-0066).

    A general ZTE keeps the plain adjoint: the option is switched off in the
    options (so the cache key equals the plain run) and an info line says so.
    """
    if not options.estimate_k0:
        return options
    from . import timing as timing_mod

    try:
        version = timing_mod.read_timing(recon_info).version
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"estimate_k0 needs the sequence timing parameters: {exc}") from exc
    if version == "zte":
        logger.info("estimate_k0 applies to SORDINO v1-v3 only; general ZTE data are reconstructed "
                    "without it.")
        options.estimate_k0 = False
    return options


def _parse_recon_info(scan):
    spec_path = Path(__file__).parent / "specs" / "recon_spec.yaml"
    spec, transforms = load_spec(spec_path, validate=True)
    recon_info = map_parameters(scan, spec, transforms)
    dtype_info = dtype_resolver.resolve(scan)
    if not dtype_info or "dtype" not in dtype_info:
        raise ValueError("Failed to resolve FID dtype from acqp.")
    recon_info['FIDDataType'] = dtype_info["dtype"]
    return recon_info


def _recon_metadata(recon_info: Dict[str, Any], options: Options) -> Dict[str, Any]:
    """Facts about the reconstruction kept with the result (BRK-0060).

    ``kspace_gap`` is ``timing.kspace_gap`` (radius of the unsampled k-space
    centre in k-grid units; ``centre_filled`` is true only with
    ``estimate_k0``), or None when the timing values are not available.
    ``k0`` is added after the reconstruction with ``estimate_k0``: for every
    frame a list of ``[real, imag]``, one per channel (None otherwise).
    Stored on the scan as ``scan._sordino_recon_meta`` and in the recon cache
    ``.json``.
    """
    from . import timing as timing_mod

    try:
        seq = timing_mod.read_timing(recon_info)
    except (KeyError, TypeError, ValueError) as exc:
        logger.debug("No k-space gap metadata: %s", exc)
        return {"kspace_gap": None, "k0": None}
    gap = timing_mod.kspace_gap(seq, float(recon_info["OverSampling"]),
                                getattr(options, "ignore_samples", None) or 1)
    gap["centre_filled"] = bool(getattr(options, "estimate_k0", False))
    return {"kspace_gap": gap, "k0": None}


def _get_fid_identity(fid_entry: FileIO) -> str:
    if isinstance(fid_entry, DatasetFile):
        return fid_entry.path
    if isinstance(fid_entry, ZippedFile):
        return fid_entry.arcname
    return getattr(fid_entry, "name", "fid")


#: Options that do not change the reconstructed values, so they are not part of the
#: recon (and spoke-timing) cache key (WI-0071, D-0098 3): ``as_complex`` and
#: ``split_ch`` choose what is returned from the cache, ``clear_cache`` whether
#: leftover temporary files are removed, ``cache_dir`` where the cache lives.
#: Every other option is in the key, including options added later.
RECON_KEY_EXCLUDED = ("as_complex", "split_ch", "clear_cache", "cache_dir")


def _build_cache_params(
    scan: Any,
    reco_id: Optional[int],
    fid_entry: FileIO,
    options: Options,
    recon_info: Dict[str, Any],
) -> Dict[str, Any]:
    keyed = {k: v for k, v in asdict(options).items() if k not in RECON_KEY_EXCLUDED}
    return {
        "scan_id": getattr(scan, "scan_id", None),
        "reco_id": reco_id,
        "fid": _get_fid_identity(fid_entry),
        "options": keyed,
        "recon_info": recon_info,
        "timing_tuning": {k: asdict(v) for k, v in TIMING_TUNING.items()},
    }


def _cache_meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _load_cache_meta(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _write_cache_meta(path: Path, meta: Dict[str, Any]) -> None:
    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        return value

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(meta), handle, sort_keys=True)


def _is_cache_valid(path: Path, *, expected_size: Optional[int] = None) -> bool:
    if not path.exists():
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if expected_size is not None and size != expected_size:
        return False
    return True


def _get_fid_entry(scan: Any) -> FileIO:
    fid_entry = fid_resolver.get_fid(scan)
    if fid_entry is None:
        logger.warning("No FID/rawdata entry found for scan %s.", 
                       getattr(scan, "scan_id", "?"))
    return cast(FileIO, fid_entry)


def _fid_nbytes(fid_entry: Any) -> Optional[int]:
    """Size of the FID in bytes, or None when it cannot be read (WI-0109).

    A zip member's size comes from the archive directory (nothing is
    decompressed); any other entry is opened and its end position is read (a
    zip member stream reports the size it was opened with, since seeking to its
    end would decompress it).
    """
    if fid_entry is None:
        return None
    zipobj = getattr(fid_entry, "zipobj", None)
    arcname = getattr(fid_entry, "arcname", None)
    if zipobj is not None and arcname is not None:
        try:
            return int(zipobj.getinfo(arcname).file_size)
        except Exception:
            return None
    opener = getattr(fid_entry, "open", None)
    if opener is None:
        return None
    try:
        with opener() as handle:
            known = getattr(handle, "_orig_file_size", None)      # zipfile.ZipExtFile
            if isinstance(known, int):
                return known
            handle.seek(0, os.SEEK_END)
            return int(handle.tell())
    except Exception:
        return None


def _check_fid_size(scan: Any, recon_info: Dict[str, Any], fid_entry: Any, size: Optional[int],
                    options: Options, allow_short: bool) -> Optional[Dict[str, int]]:
    """Keep the complete frames of a FID shorter than the parameters say (WI-0109).

    A scan stopped before its last repetition leaves a FID with fewer bytes than
    ``PVM_NRepetitions`` frames need. When the size can be read and is short,
    ``recon_info["NRepetitions"]`` becomes the number of complete frames (the
    planned number is kept as ``NRepetitionsPlanned``), so the reconstruction,
    spoke-timing correction, size report, frame selection and cache key all use
    the frames that exist, and one warning says what is missing. The bytes of
    the incomplete last frame are not used. ``size`` is ``_fid_nbytes(fid_entry)``.
    Returns those facts, or None when the FID is complete, longer, of unknown
    size, or the parameters do not give the frame size or count (then nothing
    changes and the reconstruction meets the FID as before).

    Stops with ``ValueError`` when no frame is complete, when ``offset`` is at
    or after the last complete frame, or when ``allow_short`` is false (the stop
    of earlier versions, now before anything is reconstructed).
    """
    if size is None or recon_info.get("NRepetitions") is None:
        return None
    try:
        fid_shape, fid_dtype = parse_fid_info(recon_info)
        frame_nbytes = int(np.prod(fid_shape)) * np.dtype(fid_dtype).itemsize
        planned = int(recon_info["NRepetitions"])
    except (KeyError, TypeError, ValueError):
        return None
    if frame_nbytes <= 0 or planned <= 0:
        return None
    expected = frame_nbytes * planned
    if size >= expected:
        return None
    complete = size // frame_nbytes
    rest = size - complete * frame_nbytes
    facts = (f"the FID is short ({size:,} of {expected:,} bytes, {planned} frames of "
             f"{frame_nbytes:,}; {expected - size:,} missing; the scan may have stopped early)")
    if complete == 0:
        raise ValueError(f"sordino: {facts}: no complete frame to reconstruct.")
    tail = (f"the last {rest:,} bytes (an incomplete frame)" if rest
            else "nothing (no incomplete frame is left over)")
    if not allow_short:
        raise ValueError(f"sordino: {facts}; only {complete} of {planned} frames are complete. "
                         "Stopped because allow_short_fid=false; leave it out (or set it true) "
                         "to reconstruct the complete frames.")
    offset = int(options.offset or 0)
    if offset >= complete:
        raise ValueError(f"sordino: {facts}; only {complete} of {planned} frames are complete, "
                         f"so offset {offset} is at or after the last complete frame.")
    recon_info["NRepetitionsPlanned"] = planned
    recon_info["NRepetitions"] = complete
    notes = ""
    if options.correct_spoketiming and planned > 1 and complete == 1:
        notes = "\n  note      with one frame the spoke-timing correction (2 or more frames) is skipped"
    key = (_get_fid_identity(fid_entry), size, planned)
    if getattr(scan, "_sordino_short_fid_warned", None) != key:
        logger.warning("sordino: the FID is short; the scan may have stopped early.\n"
                       "  found     %s bytes\n"
                       "  needed    %s bytes (%s frames of %s)\n"
                       "  missing   %s bytes\n"
                       "  using     %s of %s frames (the complete ones)\n"
                       "  skipped   %s%s\n"
                       "  (allow_short_fid=false stops here instead)",
                       f"{size:,}", f"{expected:,}", planned, f"{frame_nbytes:,}",
                       f"{expected - size:,}", complete, planned, tail, notes)
        try:
            setattr(scan, "_sordino_short_fid_warned", key)
        except Exception:
            pass
    return {"fid_nbytes": int(size), "expected_nbytes": int(expected),
            "frame_nbytes": int(frame_nbytes), "frames_planned": planned,
            "frames_complete": int(complete)}


def _is_frame_axis(axis: Any) -> bool:
    if isinstance(axis, bool):
        return False
    if isinstance(axis, (int, np.integer)):
        return int(axis) in (3, -1)
    if isinstance(axis, str):
        return axis.strip().lower() in FRAME_AXIS_NAMES
    return False


def _frame_selection(axis: Any, frames: Any, n_total: int) -> Tuple[Optional[list], bool]:
    """(frame indices or None for all, frame axis kept), with brkraw's ``frames`` rules.

    As in ``brkraw`` ``get_dataobj``: an int picks one frame and removes the
    axis, a list keeps the axis in that order, ``"start:stop[:step]"`` is a
    Python slice. Frames count the reconstructed frames (0 is frame ``offset``).
    ``axis`` may be omitted (the only frame axis), 3/-1, or "cycle"/"repetition".
    """
    if frames is None:
        if axis is not None:
            raise ValueError("axis needs frames (which frames of that axis to keep).")
        return None, True
    if axis is not None and not _is_frame_axis(axis):
        raise ValueError(
            f"sordino: axis {axis!r} is not the frame axis; SORDINO data have one frame axis "
            "(data axis 3, 'cycle' or 'repetition').")
    from brkraw.specs.context_map.output import parse_frames

    try:
        _, kept, norm, notes = parse_frames(frames, int(n_total))
    except ValueError as exc:
        raise ValueError(str(exc).replace("split: ", "sordino frames: ", 1)) from None
    for note in notes:
        logger.warning(note.replace("split: ", "sordino frames: ", 1))
    return list(norm), bool(kept)


def _oriented_spatial_shape(vol_shape, recon_info: Dict[str, Any]) -> list:
    """Spatial shape after ``orientation.correct`` (a transpose), without data."""
    probe = np.broadcast_to(np.zeros((), dtype=np.uint8), tuple(int(v) for v in vol_shape))
    try:
        return list(correct_orientation(probe, recon_info).shape)
    except Exception as exc:  # orientation values missing: shape as reconstructed
        logger.warning("sordino size report: orientation not applied to the reported shape (%s).", exc)
        return list(vol_shape)


def _output_info(recon_info: Dict[str, Any], options: Options, cached_shape, cache_dtype,
                 frame_list: Optional[list], keep_axis: bool, *, cached: bool,
                 cache_path: Path, max_memory_gb: Any,
                 stc_cache_path: Optional[Path] = None) -> Dict[str, Any]:
    """What get_dataobj would return and need, before reading anything (C9, WI-0071)."""
    from . import memguard

    cached_shape = [int(v) for v in cached_shape]
    cdt = np.dtype(cache_dtype)
    real_dt = np.empty(0, dtype=cdt).real.dtype
    multi = len(cached_shape) == 5
    n_ch = cached_shape[0] if multi else 1
    vol = cached_shape[1:4] if multi else cached_shape[:3]
    n_total = cached_shape[-1]
    n_sel = n_total if frame_list is None else len(frame_list)
    per_ch = multi and options.split_ch
    count = (n_ch if per_ch else 1) * (2 if options.as_complex else 1)
    shape = _oriented_spatial_shape(vol, recon_info)
    if keep_axis:
        shape = shape + [n_sel]
    nbytes = int(count * int(np.prod(shape)) * real_dt.itemsize)
    frame_bytes = int(np.prod(cached_shape[:-1])) * cdt.itemsize
    cache_nbytes = int(np.prod(cached_shape)) * cdt.itemsize
    disk_nbytes = 0 if cached else cache_nbytes
    stc_stage_nbytes = 0
    if not cached and options.correct_spoketiming and int(recon_info.get("NRepetitions") or 1) > 1:
        fid_shape, fid_dtype = parse_fid_info(recon_info)
        stc_nbytes = int(np.prod(fid_shape)) * np.dtype(fid_dtype).itemsize * n_total
        # a valid spoke-timing cache is reused, so it needs no new disk space
        if stc_cache_path is None or not _is_cache_valid(stc_cache_path, expected_size=stc_nbytes):
            disk_nbytes += stc_nbytes
            # the spoke-timing stage works on one segment of projections (all selected
            # frames) at a time; the segment count follows mem_limit and the FID file
            # size as in spoketiming.prep_fid_segmentation. The file is not opened here:
            # its smallest possible size (the frames up to offset + frames read), with
            # the same num_frames scaling, gives the fewest and largest segments, so the
            # estimate is never below the run (wi-0071-choi-5 F1)
            from .spoketiming import get_num_segment
            scale = 1.0
            if options.num_frames is not None:
                scale = get_num_frames(recon_info, options) / options.num_frames
            file_gb = (int(np.prod(fid_shape)) * np.dtype(fid_dtype).itemsize
                       * (int(options.offset or 0) + n_total) * scale / memguard.GIB)
            segs = get_num_segment(file_gb, recon_info, options)
            seg_fraction = float(max(segs)) / float(recon_info["NPro"])
            stc_stage_nbytes = int(np.ceil(memguard.SPOKETIMING_FACTOR * stc_nbytes * seg_fraction))
    # Reconstruction step (D-0098 2): runs first in the same process when no cache
    # exists; with gc after every frame its working memory does not grow with the
    # frame count. Added to the read estimate (conservative: the two are not at their
    # peaks at the same time). The serial reconstruction (WI-0097, D-0133 1 and 4)
    # takes what the limit leaves after the read as its budget and picks the chunk
    # size from it, so the limit sets the chunk size instead of stopping a large scan.
    limit = memguard.memory_limit_bytes(max_memory_gb)
    read_nbytes = nbytes + 3 * frame_bytes
    recon_share = 0
    chunk_spokes = None
    n_chunks = None
    k0_solve = k0_reason = None
    if not cached:
        budget = int(limit["limit_nbytes"]) - read_nbytes
        plan = memguard.recon_plan(
            int(recon_info["NPro"]), int(recon_info["NPoints"]), n_ch, vol,
            estimate_k0=bool(options.estimate_k0), budget_nbytes=budget)
        recon_share = max(plan["recon_nbytes"], int(stc_stage_nbytes))
        chunk_spokes = plan["chunk_spokes"]
        n_chunks = plan["n_chunks"]
        k0_solve = plan["k0_method"]          # estimate_k0: "samples" or "toeplitz" (WI-0099, D-0143)
        k0_reason = plan["k0_reason"]         # "limit": only the sample solve fits (D-0147)
    info: Dict[str, Any] = {
        "shape": shape,
        "dtype": real_dt.str,
        "count": count,
        "nbytes": nbytes,
        "frames": n_sel,
        "frames_reconstructed": n_total,
        "cached": bool(cached),
        "cache_path": str(cache_path),
        "cache_dir": str(options.cache_dir),
        "cache_dtype": cdt.str,
        "cache_nbytes": cache_nbytes,
        "peak_nbytes": read_nbytes + recon_share,
        "recon_nbytes": recon_share,
        "recon_chunk_spokes": chunk_spokes,
        "recon_chunks": n_chunks,
        "recon_k0_method": k0_solve,
        "recon_k0_reason": k0_reason,
        "disk_nbytes": disk_nbytes,
        "disk_free_nbytes": None if cached else memguard.free_disk_bytes(Path(options.cache_dir)),
    }
    info.update(limit)
    return info


#: dtype the reconstruction writes today (complex128, WI-0071 M1/M2; pinned by
#: tests/test_hook_read.py); used for the estimate before a cache exists.
RECON_CACHE_DTYPE = np.dtype("<c16")


def _plan(scan: Any, reco_id: Optional[int], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Options, cache paths, cache state, frame selection and the size estimate."""
    options = _build_options(kwargs)
    recon_info = _parse_recon_info(scan)
    options = _resolve_k0(options, recon_info)
    fid_entry = _get_fid_entry(scan)
    frames_planned = recon_info.get("NRepetitions")
    # a short FID (WI-0109): NRepetitions becomes the complete frames, before the cache key
    fid_nbytes = _fid_nbytes(fid_entry)
    short_fid = _check_fid_size(scan, recon_info, fid_entry, fid_nbytes, options,
                                parse_bool("allow_short_fid", kwargs.get("allow_short_fid", True)))
    cache_params = _build_cache_params(scan, reco_id, fid_entry, options, recon_info)
    img_cache_path = build_recon_cache_path(options.cache_dir, cache_params)
    img_meta = _load_cache_meta(_cache_meta_path(img_cache_path))
    cached_dtype: Optional[np.dtype] = None
    cached_shape: Optional[list[int]] = None
    if img_meta:
        try:
            cached_dtype = np.dtype(img_meta.get("dtype"))
            cached_shape = img_meta.get("shape")
            if cached_shape:
                expected_size = int(np.prod(cached_shape) * cached_dtype.itemsize)
                if not _is_cache_valid(img_cache_path, expected_size=expected_size):
                    cached_dtype = None
                    cached_shape = None
        except Exception:
            cached_dtype = None
            cached_shape = None
    cached = cached_dtype is not None and bool(cached_shape)
    shape_for_plan = cached_shape if cached else list(get_dataobj_shape(recon_info, options))
    frame_list, keep_axis = _frame_selection(kwargs.get("axis"), kwargs.get("frames"),
                                             int(shape_for_plan[-1]))
    info = _output_info(recon_info, options, shape_for_plan,
                        cached_dtype if cached else RECON_CACHE_DTYPE, frame_list, keep_axis,
                        cached=cached, cache_path=img_cache_path,
                        max_memory_gb=kwargs.get("max_memory_gb"),
                        stc_cache_path=build_spoketiming_cache_path(options.cache_dir, cache_params))
    info["frames_planned"] = frames_planned
    if short_fid is not None:
        info["fid_short_nbytes"] = short_fid["expected_nbytes"] - short_fid["fid_nbytes"]
    else:
        info["fid_short_nbytes"] = None if fid_nbytes is None or frames_planned is None else 0
    return {
        "options": options, "recon_info": recon_info, "fid_entry": fid_entry,
        "short_fid": short_fid,
        "cache_params": cache_params, "img_cache_path": img_cache_path, "img_meta": img_meta,
        "cached_dtype": cached_dtype if cached else None,
        "cached_shape": cached_shape if cached else None,
        "frame_list": frame_list, "keep_axis": keep_axis, "info": info,
    }


def get_dataobj_info(scan: Any, reco_id: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
    """Size of what ``get_dataobj`` returns with the same arguments, without reading data.

    For callers that decide before loading (viewer size notice, WI-0069/C9).
    Keys: ``shape`` and ``dtype`` of each returned array, ``count`` (arrays
    returned), ``nbytes`` (all arrays), ``frames``, ``cached`` (a valid recon
    cache exists; if not, ``get_dataobj`` reconstructs first), ``cache_nbytes``,
    ``peak_nbytes`` (memory estimate: the returned arrays, three cache frames and,
    without a cache, ``recon_nbytes`` for the reconstruction step,
    ``memguard.recon_nbytes``), ``limit_nbytes`` and ``limit_source``
    (the memory limit that ``get_dataobj`` applies), ``disk_nbytes`` and
    ``disk_free_nbytes``. Before a cache exists the cache dtype is assumed to
    be complex128.
    """
    return _plan(scan, reco_id, kwargs)["info"]


def get_dataobj(
        scan: Any, reco_id: Optional[int] = None, **kwargs: Any,
    ) -> Optional[Union[np.ndarray, Tuple[np.ndarray, ...]]]:
    """Reconstruct (or read from the recon cache) the SORDINO images.

    Read-time options (WI-0071): ``frames``/``axis`` as in brkraw (only the
    selected frames are read from the cache), ``max_memory_gb`` (limit of the
    memory check; default half of the physical memory). Before reconstructing
    or reading, the expected memory and cache disk space are checked and
    ``memguard.SordinoResourceError`` is raised when they exceed the limit.
    """
    from . import memguard
    from .cacheio import read_recon_frames

    plan = _plan(scan, reco_id, kwargs)
    options: Options = plan["options"]
    recon_info = plan["recon_info"]
    info = plan["info"]
    cache_files: list[str] = []
    setattr(scan, "_sordino_cache_files", cache_files)
    logger.debug("Sordino options correct_spoketiming=%s", options.correct_spoketiming)
    setattr(scan, "_sordino_options", options)
    try:
        spatial_shape = tuple(parse_volume_shape(recon_info, options))
        setattr(scan, "_sordino_spatial_shape", spatial_shape)
    except Exception:
        setattr(scan, "_sordino_spatial_shape", None)
    recon_meta = _recon_metadata(recon_info, options)
    # a short FID (WI-0109): the planned and complete frames, with the result
    recon_meta["short_fid"] = plan["short_fid"]
    setattr(scan, "_sordino_recon_meta", recon_meta)
    setattr(scan, "_sordino_dataobj_info", info)
    memguard.check(info)
    fid_entry = plan["fid_entry"]
    cache_params = plan["cache_params"]
    img_cache_path = plan["img_cache_path"]
    img_meta_path = _cache_meta_path(img_cache_path)
    cached_dtype: Optional[np.dtype] = plan["cached_dtype"]
    cached_shape: Optional[list[int]] = plan["cached_shape"]
    if cached_dtype is not None and options.estimate_k0 and plan["img_meta"]:
        recon_meta["k0"] = plan["img_meta"].get("k0")

    if cached_dtype is None or cached_shape is None:
        with fid_entry.open() as fid_fobj:
            # per-chunk trajectory and phase rows (WI-0097): no whole trajectory, no
            # trajectory cache file, no whole phase factor
            traj = trajectory_rows(recon_info, options)
            phase_factor = phase_correction_rows(
                recon_info, options, int(parse_fid_info(recon_info)[0][1]))
            chunk_spokes = info.get("recon_chunk_spokes")
            virtual_traj = None
            k0_frames: list = []
            if options.estimate_k0:
                virtual_traj = kcentre.leading_points(recon_info, options.ignore_samples or 1)
                logger.info("Estimating the k-space centre (%s virtual sample(s) per spoke).",
                            virtual_traj.shape[1])
                if info.get("recon_k0_reason") == "limit":
                    logger.info("estimate_k0: the Toeplitz solve does not fit the memory limit; "
                                "using the sample-based solve, which fits (slower).")
            img_temp_path = img_cache_path.with_suffix(img_cache_path.suffix + ".partial")
            if img_temp_path.exists():
                try:
                    os.remove(img_temp_path)
                except OSError:
                    pass
            with open(img_temp_path, "w+b") as img_fobj:
                logger.debug("Created temp image file: %s", img_temp_path)
                cache_files.append(str(img_temp_path))

                if options.correct_spoketiming and recon_info['NRepetitions'] > 1:
                    logger.debug("Spoketiming correction enabled.")
                    fid_shape, fid_dtype = parse_fid_info(recon_info)
                    num_frames = get_num_frames(recon_info, options)
                    stc_expected_size = int(np.prod(fid_shape) * fid_dtype.itemsize * num_frames)
                    stc_cache_path = build_spoketiming_cache_path(options.cache_dir, cache_params)
                    stc_meta_path = _cache_meta_path(stc_cache_path)
                    stc_temp_path = stc_cache_path.with_suffix(stc_cache_path.suffix + ".partial")
                    stc_param = {
                        "buffer_size": int(np.prod(fid_shape) * fid_dtype.itemsize),
                        "dtype": fid_dtype,
                    }

                    if _is_cache_valid(stc_cache_path, expected_size=stc_expected_size):
                        logger.debug("Using cached spoketiming file: %s", stc_cache_path)
                    else:
                        if stc_temp_path.exists():
                            try:
                                os.remove(stc_temp_path)
                            except OSError:
                                pass
                        with open(stc_temp_path, "w+b") as stc_fobj:
                            cache_files.append(str(stc_temp_path))
                            logger.debug("Created temp spoketiming file: %s", stc_temp_path)
                            segs = prep_fid_segmentation(fid_fobj, recon_info, options)
                            logger.info("Spoketiming correction: %s segment(s).", segs.shape[0])
                            stc_param = correct_spoketiming(
                                segs, fid_fobj, stc_fobj, recon_info, options
                            )
                        os.replace(stc_temp_path, stc_cache_path)
                        _write_cache_meta(
                            stc_meta_path,
                            {
                                "dtype": np.dtype(stc_param["dtype"]).str,
                                "buffer_size": int(stc_param["buffer_size"]),
                                "size": stc_expected_size,
                            },
                        )

                    with open(stc_cache_path, "rb") as stc_fobj:
                        dtype = recon_dataobj(
                            stc_fobj,
                            traj,
                            recon_info,
                            img_fobj,
                            options,
                            override_buffer_size=stc_param['buffer_size'],
                            override_dtype=stc_param['dtype'],
                            phase_factor=phase_factor,
                            virtual_traj=virtual_traj,
                            k0_out=k0_frames,
                            chunk_spokes=chunk_spokes,
                            k0_method=info.get("recon_k0_method"),
                        )
                else:
                    logger.debug("Spoketiming correction disabled.")
                    dtype = recon_dataobj(fid_fobj, traj, recon_info, img_fobj, options,
                                          phase_factor=phase_factor,
                                          virtual_traj=virtual_traj, k0_out=k0_frames,
                                          chunk_spokes=chunk_spokes,
                                          k0_method=info.get("recon_k0_method"))
            os.replace(img_temp_path, img_cache_path)
        if options.estimate_k0:
            recon_meta["k0"] = [[[float(k.real), float(k.imag)] for k in frame] for frame in k0_frames]
        dataobj_shape = list(get_dataobj_shape(recon_info, options))
        cached_dtype = np.dtype(dtype)
        cached_shape = list(dataobj_shape)
        _write_cache_meta(
            img_meta_path,
            {
                "dtype": cached_dtype.str,
                "shape": list(cached_shape),
                "kspace_gap": recon_meta["kspace_gap"],
                "k0": recon_meta["k0"],
                "short_fid": recon_meta["short_fid"],
            },
        )
    else:
        logger.debug("Using cached recon file: %s", img_cache_path)
        if options.estimate_k0:
            logger.info("Reading the recon cache reconstructed with estimate_k0 "
                        "(k-space centre estimated).")

    if cached_shape is None:
        cached_shape = list(get_dataobj_shape(recon_info, options))
    assert cached_dtype is not None
    # Frame by frame into one result (WI-0071): magnitude unless as_complex, and the
    # channels combined (RSS of magnitudes, or the complex sum) unless split_ch.
    is_multi = len(cached_shape) == 5
    combine = is_multi and not options.split_ch
    logger.debug("Reading recon cache frame by frame (as_complex=%s, combine channels=%s, frames=%s).",
                 options.as_complex, combine,
                 "all" if plan["frame_list"] is None else len(plan["frame_list"]))
    dataobj = read_recon_frames(img_cache_path, cached_dtype, cached_shape, plan["frame_list"],
                                as_complex=options.as_complex, combine_channels=combine)
    if combine:
        is_multi = False
    if not plan["keep_axis"]:
        dataobj = dataobj[..., 0]

    if options.as_complex:
        logger.debug("Formatting complex output.")
        if is_multi:
            logger.debug("Emitting complex output per channel.")
            dataobj_list = []
            for receiver_data in dataobj:
                receiver_arr = cast(NDArray[Any], receiver_data)
                dataobj_list.extend([
                    correct_orientation(np.real(receiver_arr), recon_info),
                    correct_orientation(np.imag(receiver_arr), recon_info),
                ])
            return cast(Tuple[np.ndarray, ...], tuple(dataobj_list))
        logger.debug("Emitting complex output (real/imag pair).")
        return (
            correct_orientation(np.real(dataobj), recon_info),
            correct_orientation(np.imag(dataobj), recon_info),
        )

    if is_multi:
        logger.debug("Emitting magnitude output per channel.")
        return cast(
            Tuple[np.ndarray, ...],
            tuple(correct_orientation(ch, recon_info) for ch in dataobj),
        )
    logger.debug("Emitting single-channel magnitude output.")
    return correct_orientation(cast(np.ndarray, dataobj), recon_info)


def get_affine(
        scan: Any,
        reco_id: Optional[int] = None,
        decimals: Optional[int] = None,
        **kwargs: Any,
    ) -> Optional[Union[np.ndarray, Tuple[np.ndarray, ...]]]:
    """brkraw affine, with the origin moved for ``ext_factors`` (WI-0081, D-0104).

    ``ext_factors[i]`` scales reconstruction axis ``i`` (``PVM_Matrix`` order: read, phase,
    slice of the acquisition). The data reach the caller after ``orientation.correct``, so the
    shift is applied per output axis: ``-(N // 2 - N0 // 2)`` voxels, ``N`` the extended size,
    ``N0`` the ``ext_factors=1`` size, the adjoint NUFFT centre being at ``N // 2``. Voxel size
    and direction are unchanged.
    """
    affine = get_affine_helper(scan, reco_id, decimals=decimals, **kwargs)
    if affine is None:
        return None
    options = getattr(scan, "_sordino_options", None) or _build_options(kwargs)
    if np.allclose(np.asarray(options.ext_factors, dtype=float), 1.0):
        return affine
    try:
        recon_info = _parse_recon_info(scan)
        shift = _ext_factor_shift(recon_info, options)
    except Exception as exc:  # header without Matrix/orientation: keep the brkraw affine
        logger.warning("sordino: ext_factors affine shift not applied (%s).", exc)
        return affine
    affine_list = list(affine) if isinstance(affine, tuple) else [affine]
    new_affine_list = [_apply_ext_factor_affine(aff, shift) for aff in affine_list]
    if isinstance(affine, tuple):
        return tuple(new_affine_list)
    return new_affine_list[0]

def _calc_slope_inter(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    inter = float(np.min(data))
    dmax = float(np.max(data))
    # 65535 steps: the maximum maps to 65535 (2**16 would wrap to 0 in uint16)
    slope = (dmax - inter) / (2**16 - 1) if dmax != inter else 1.0
    if data.ndim > 3:
        converted = np.stack(
            [((data[..., idx] - inter) / slope).round().astype(np.uint16) for idx in range(data.shape[-1])],
            axis=-1,
        )
    else:
        converted = ((data - inter) / slope).round().astype(np.uint16)
    return converted.squeeze(), slope, inter


def _ext_factor_shift(recon_info: Dict[str, Any], options: Options) -> Tuple[int, int, int]:
    """Origin shift in whole voxels per output axis for ``options.ext_factors``.

    Sizes are whole voxels (``int(Matrix * ext_factors)``, as reconstructed), taken in
    reconstruction order and moved to output order with ``orientation.axis_order``.
    """
    base_options = replace(options, ext_factors=(1.0, 1.0, 1.0))
    base = [int(n) for n in parse_volume_shape(recon_info, base_options)]
    scaled = [int(n) for n in parse_volume_shape(recon_info, options)]
    order = orientation_axis_order(recon_info)
    return cast(Tuple[int, int, int],
                tuple(-(scaled[order[j]] // 2 - base[order[j]] // 2) for j in range(3)))


def _apply_ext_factor_affine(affine: np.ndarray, shift: Tuple[int, int, int]) -> np.ndarray:
    if not any(shift):
        return affine
    updated = np.asarray(affine, dtype=float).copy()
    updated[:, 3] = updated.dot([float(v) for v in shift] + [1.0])
    return updated


def _clear_cache_files(scan: Any, *, keep: Optional[Tuple[str, ...]] = None) -> None:
    cache_files = getattr(scan, "_sordino_cache_files", None)
    if not cache_files:
        return
    keep = keep or tuple()
    for path in list(cache_files):
        if any(str(path).endswith(suffix) for suffix in keep):
            continue
        try:
            os.remove(path)
        except OSError:
            pass
    cache_files.clear()
    logger.debug("Cleared sordino cache files.")


def convert(
    scan: Any,
    dataobj: Union[np.ndarray, Tuple[np.ndarray, ...]],
    affine: Union[np.ndarray, Tuple[np.ndarray, ...]],
    *,
    xyz_units: str = "mm",
    t_units: str = "sec",
    override_header: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
):
    options = getattr(scan, "_sordino_options", None) or _build_options(kwargs)
    data_list = list(dataobj) if isinstance(dataobj, tuple) else [dataobj]
    affine_list = list(affine) if isinstance(affine, tuple) else [affine]
    nii_list = []
    for idx, data in enumerate(data_list):
        aff = affine_list[idx] if idx < len(affine_list) else affine_list[0]
        img_u16, slope, inter = _calc_slope_inter(np.asarray(data))
        logger.debug('Calculated slope: %s, inter: %s', slope, inter)
        nii = Nifti1Image(img_u16, aff)
        nii.set_qform(aff, 1)
        nii.set_sform(aff, 0)
        nii.header.set_slope_inter(slope, inter)
        try:
            nii.header.set_xyzt_units(xyz_units, t_units)
        except Exception:
            pass
        if override_header:
            for key, value in override_header.items():
                if value is not None:
                    try:
                        nii.header[key] = value
                    except Exception:
                        pass
        nii_list.append(nii)
    if options.clear_cache:
        _clear_cache_files(scan)
    if isinstance(dataobj, tuple) or isinstance(affine, tuple):
        return tuple(nii_list)
    return nii_list[0] if nii_list else None


HOOK = {"get_dataobj": get_dataobj, "get_affine": get_affine, "convert": convert}

__all__ = ["HOOK", "get_dataobj", "get_dataobj_info", "get_affine", "convert"]
