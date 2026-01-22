import os
import logging
import numpy as np
import tempfile
from pathlib import Path

from typing import Any, Optional, Tuple, Dict, Union, cast, TYPE_CHECKING

from nibabel.nifti1 import Nifti1Image

from brkraw.specs.remapper import load_spec, map_parameters
from brkraw.resolver import fid as fid_resolver
from brkraw.resolver import datatype as dtype_resolver
from brkraw.core import config as config_core
from brkraw.core.fs import DatasetFile
from brkraw.core.zip import ZippedFile

from numpy.typing import NDArray
from .typing import Options
from .traj import get_trajectory
from .helper import progressbar
from .recon import get_dataobj_shape, recon_dataobj
from .spoketiming import prep_fid_segmentation, correct_spoketiming
from .orientation import correct as correct_orientation


FileIO = Union[DatasetFile, ZippedFile]
logger = logging.getLogger("brkraw.sordino")


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
        "offreso_ch",
        "offreso_freq",
        "mem_limit",
        "clear_cache",
        "split_ch",
        "as_complex",
    }
    unknown_keys = sorted(set(kwargs.keys()) - known_keys)
    if unknown_keys:
        logger.debug("Sordino hook unknown kwargs: %s", unknown_keys)
    cache_dir = _get_cache_dir(kwargs.get("cache_dir"))
    logger.debug("Cache dir: %s", cache_dir)

    return Options(
        ext_factors=_normalize_ext_factors(kwargs.get("ext_factors")),
        ignore_samples=int(kwargs.get("ignore_samples", 1)),
        offset=int(kwargs.get("offset", 0)),
        num_frames=kwargs.get("num_frames"),
        correct_spoketiming=bool(kwargs.get("correct_spoketiming", False)),
        correct_ramptime=bool(kwargs.get("correct_ramptime", True)),
        offreso_ch=kwargs.get("offreso_ch"),
        offreso_freq=float(kwargs.get("offreso_freq", 0.0)),
        mem_limit=float(kwargs.get("mem_limit", 0.5)),
        clear_cache=bool(kwargs.get("clear_cache", True)),
        split_ch=bool(kwargs.get("split_ch", False)),
        cache_dir=cache_dir,
        as_complex=bool(kwargs.get("as_complex", False)),
    )


def _parse_recon_info(scan):
    spec_path = Path(__file__).parent / "specs" / "recon_spec.yaml"
    spec, transforms = load_spec(spec_path, validate=True)
    recon_info = map_parameters(scan, spec, transforms)
    dtype_info = dtype_resolver.resolve(scan)
    if not dtype_info or "dtype" not in dtype_info:
        raise ValueError("Failed to resolve FID dtype from acqp.")
    recon_info['FIDDataType'] = dtype_info["dtype"]
    return recon_info


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
    logger.debug("Sordino options correct_spoketiming=%s", options.correct_spoketiming)
    setattr(scan, "_sordino_options", options)
    cache_files: list[str] = []
    setattr(scan, "_sordino_cache_files", cache_files)
    recon_info = _parse_recon_info(scan)
    fid_entry = _get_fid_entry(scan)

    with fid_entry.open() as fid_fobj:
        traj = get_trajectory(recon_info, options)

        with tempfile.NamedTemporaryFile(   
            mode="w+b", delete=False, dir=options.cache_dir
        ) as img_fobj:  # where reconstructed dataobj will be stored.
            cache_files.append(img_fobj.name)
            logger.debug("Created temp image file: %s", img_fobj.name)
            
            if options.correct_spoketiming and recon_info['NRepetitions'] > 1:
                logger.debug("Spoketiming correction enabled.")
                with tempfile.NamedTemporaryFile(
                    mode="w+b", delete=False, dir=options.cache_dir
                ) as stc_fobj:  # where spoketiming corrected fid will be stored.
                    cache_files.append(stc_fobj.name)
                    logger.debug("Created temp spoketiming file: %s", stc_fobj.name)
                    
                    segs = prep_fid_segmentation(fid_fobj, recon_info, options)
                    logger.info("Spoketiming correction: %s segment(s).", segs.shape[0])

                    stc_param = correct_spoketiming(segs, fid_fobj, stc_fobj, recon_info, options)
                    dtype = recon_dataobj(stc_fobj, traj, recon_info, img_fobj, options, 
                                          override_buffer_size=stc_param['buffer_size'],
                                          override_dtype=stc_param['dtype'])
            else:
                logger.debug("Spoketiming correction disabled.")
                dtype = recon_dataobj(fid_fobj, traj, recon_info, img_fobj, options)
    
    with open(img_fobj.name, "rb") as img_fobj:
        dataobj_shape = get_dataobj_shape(recon_info, options)
        dataobj = np.frombuffer(img_fobj.read(), dtype=dtype).reshape(dataobj_shape, order='F')
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

def _calc_slope_inter(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    inter = float(np.min(data))
    dmax = float(np.max(data))
    slope = (dmax - inter) / 2**16 if dmax != inter else 1.0
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
    ext_factors = options.ext_factors
    data_list = list(dataobj) if isinstance(dataobj, tuple) else [dataobj]
    affine_list = list(affine) if isinstance(affine, tuple) else [affine]
    nii_list = []
    for idx, data in enumerate(data_list):
        aff = affine_list[idx] if idx < len(affine_list) else affine_list[0]
        scaled_affine = _apply_ext_factor_affine(aff, tuple(data.shape[:3]), ext_factors)
        img_u16, slope, inter = _calc_slope_inter(np.asarray(data))
        nii = Nifti1Image(img_u16, scaled_affine)
        nii.set_qform(scaled_affine, 1)
        nii.set_sform(scaled_affine, 0)
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


HOOK = {"get_dataobj": get_dataobj, "convert": convert}

__all__ = ["HOOK", "get_dataobj", "convert"]
