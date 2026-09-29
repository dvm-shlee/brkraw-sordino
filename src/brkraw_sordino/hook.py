import os
import json
import logging
import numpy as np
from dataclasses import asdict
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
from .traj import get_trajectory
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
from .orientation import correct as correct_orientation
from .recon import phase_correction_factor
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


def _get_cache_dir(path: Optional[Union[str, Path]]) -> Path:
    if path:
        base = Path(path).expanduser()
    else:
        base = config_core.resolve_root(None) / "cache" / "sordino"
    base.mkdir(parents=True, exist_ok=True)
    return base


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
    }
    unknown_keys = sorted(set(kwargs.keys()) - known_keys)
    if unknown_keys:
        logger.warning(
            "Sordino hook: ignoring unknown option(s): %s (ramp_model and correct_phase "
            "were removed; correct_ramptime now covers both)", ", ".join(unknown_keys))
    cache_dir = _get_cache_dir(kwargs.get("cache_dir"))
    logger.debug("Cache dir: %s", cache_dir)
    offreso_freqs = kwargs.get("offreso_freqs")
    if isinstance(offreso_freqs, (int, float)):
        offreso_freqs = (offreso_freqs, )

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
        offreso_freqs=tuple(offreso_freqs) if offreso_freqs else (),
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


def _build_cache_params(
    scan: Any,
    reco_id: Optional[int],
    fid_entry: FileIO,
    options: Options,
    recon_info: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "scan_id": getattr(scan, "scan_id", None),
        "reco_id": reco_id,
        "fid": _get_fid_identity(fid_entry),
        "options": asdict(options),
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


def get_dataobj(
        scan: Any, reco_id: Optional[int] = None, **kwargs: Any,
    ) -> Optional[Union[np.ndarray, Tuple[np.ndarray, ...]]]:
    
    options = _build_options(kwargs)
    cache_files: list[str] = []
    setattr(scan, "_sordino_cache_files", cache_files)
    recon_info = _parse_recon_info(scan)
    options = _resolve_k0(options, recon_info)
    logger.debug("Sordino options correct_spoketiming=%s", options.correct_spoketiming)
    setattr(scan, "_sordino_options", options)
    try:
        spatial_shape = tuple(parse_volume_shape(recon_info, options))
        setattr(scan, "_sordino_spatial_shape", spatial_shape)
    except Exception:
        setattr(scan, "_sordino_spatial_shape", None)
    recon_meta = _recon_metadata(recon_info, options)
    setattr(scan, "_sordino_recon_meta", recon_meta)
    fid_entry = _get_fid_entry(scan)
    cache_params = _build_cache_params(scan, reco_id, fid_entry, options, recon_info)
    img_cache_path = build_recon_cache_path(options.cache_dir, cache_params)
    img_meta_path = _cache_meta_path(img_cache_path)
    img_meta = _load_cache_meta(img_meta_path)
    cached_dtype: Optional[np.dtype] = None
    cached_shape: Optional[list[int]] = None
    if img_meta:
        try:
            cached_dtype = np.dtype(img_meta.get("dtype"))
            cached_shape = img_meta.get("shape")
            if options.estimate_k0:
                recon_meta["k0"] = img_meta.get("k0")
            if cached_shape:
                expected_size = int(np.prod(cached_shape) * cached_dtype.itemsize)
                if not _is_cache_valid(img_cache_path, expected_size=expected_size):
                    cached_dtype = None
                    cached_shape = None
        except Exception:
            cached_dtype = None
            cached_shape = None

    if cached_dtype is None or cached_shape is None:
        with fid_entry.open() as fid_fobj:
            traj = get_trajectory(recon_info, options)
            phase_factor = phase_correction_factor(
                recon_info, options, int(parse_fid_info(recon_info)[0][1]))
            virtual_traj = None
            k0_frames: list = []
            if options.estimate_k0:
                virtual_traj = kcentre.leading_points(recon_info, options.ignore_samples or 1)
                logger.info("Estimating the k-space centre (%s virtual sample(s) per spoke).",
                            virtual_traj.shape[1])
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
                        )
                else:
                    logger.debug("Spoketiming correction disabled.")
                    dtype = recon_dataobj(fid_fobj, traj, recon_info, img_fobj, options,
                                          phase_factor=phase_factor,
                                          virtual_traj=virtual_traj, k0_out=k0_frames)
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
            },
        )
    else:
        logger.debug("Using cached recon file: %s", img_cache_path)

    if cached_shape is None:
        cached_shape = list(get_dataobj_shape(recon_info, options))
    assert cached_dtype is not None
    with open(img_cache_path, "rb") as img_fobj:
        dataobj = np.frombuffer(img_fobj.read(), dtype=cached_dtype).reshape(cached_shape, order='F')
    num_receivers = recon_info.get("EncNReceivers", 1)
    if not options.as_complex:
        logger.debug("Converting to magnitude (as_complex=False).")
        dataobj = np.abs(dataobj)
    else:
        logger.debug("Keeping complex data (as_complex=True).")

    is_multi = num_receivers > 1
    if not options.split_ch and is_multi:
        logger.debug("Combining multi-channel data (split_ch=False).")
        if options.as_complex:
            logger.debug("Combining complex channels by summation.")
            dataobj = np.sum(dataobj, axis=0)
            is_multi = False
        else:
            logger.debug("Combining magnitude channels by RSS.")
            dataobj = np.sqrt(np.sum(dataobj ** 2, axis=0))
            is_multi = False
    elif options.split_ch and is_multi:
        logger.debug("Keeping multi-channel data (split_ch=True).")
    else:
        logger.debug("Single-channel data detected.")

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
    affine = get_affine_helper(scan, reco_id, decimals=decimals, **kwargs)
    if affine is None:
        return None
    options = getattr(scan, "_sordino_options", None) or _build_options(kwargs)
    ext_factors = options.ext_factors
    spatial_shape = getattr(scan, "_sordino_spatial_shape", None)
    if spatial_shape is None:
        try:
            recon_info = _parse_recon_info(scan)
            spatial_shape = tuple(parse_volume_shape(recon_info, options))
            setattr(scan, "_sordino_spatial_shape", spatial_shape)
        except Exception:
            spatial_shape = None
    if spatial_shape is None:
        return affine
    affine_list = list(affine) if isinstance(affine, tuple) else [affine]
    new_affine_list = []
    for aff in affine_list:
        scaled_affine = _apply_ext_factor_affine(aff, tuple(spatial_shape[:3]), ext_factors)
        new_affine_list.append(scaled_affine)
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


def _apply_ext_factor_affine(affine: np.ndarray, shape: Tuple[int, int, int], ext_factors: Tuple[float, float, float]) -> np.ndarray:
    factors = np.asarray(ext_factors, dtype=float)
    if np.allclose(factors, 1.0):
        return affine
    scaled_matrix = np.asarray(shape, dtype=float)
    base_matrix = scaled_matrix / factors
    center = (base_matrix - 1.0) / 2.0
    origin = center - (scaled_matrix - 1.0) / 2.0
    updated = affine.copy()
    updated[:, 3] = updated.dot(origin.tolist() + [1.0])
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

__all__ = ["HOOK", "get_dataobj", "get_affine", "convert"]
