from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TextIO, cast
import sys

import numpy as np

from brkraw.core import config as config_core
from brkraw.resolver import fid as fid_resolver
from brkraw.resolver import datatype as datatype_resolver

logger = logging.getLogger("brkraw.sordino")


def _progress(iterable: Any, *, desc: str = "", ncols: int = 100, disable: Optional[bool] = None):
    """Lightweight progress iterator (tqdm-like) without external deps."""
    if disable is None:
        disable = logger.isEnabledFor(logging.DEBUG)
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


@dataclass
class _Options:
    ext_factors: Tuple[float, float, float]
    pass_samples: int
    offset: int
    num_frames: Optional[int]
    traj_offset: Optional[float]
    spoketiming: bool
    ramp_time: bool
    offreso_ch: Optional[int]
    offreso_freq: float
    mem_limit: float
    traj_denom: Optional[float]
    clear_cache: bool
    operator: str
    rss: bool


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"yes", "true", "1", "on"}
    return bool(value)


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


def _build_options(kwargs: Dict[str, Any]) -> _Options:
    return _Options(
        ext_factors=_normalize_ext_factors(kwargs.get("ext_factors")),
        pass_samples=int(kwargs.get("pass_samples", 1)),
        offset=int(kwargs.get("offset", 0)),
        num_frames=kwargs.get("num_frames"),
        traj_offset=kwargs.get("traj_offset"),
        spoketiming=bool(kwargs.get("spoketiming", False)),
        ramp_time=bool(kwargs.get("ramp_time", False)),
        offreso_ch=kwargs.get("offreso_ch"),
        offreso_freq=float(kwargs.get("offreso_freq", 0.0)),
        mem_limit=float(kwargs.get("mem_limit", 0.5)),
        traj_denom=kwargs.get("traj_denom"),
        clear_cache=bool(kwargs.get("clear_cache", True)),
        operator=str(kwargs.get("operator", "finufft")),
        rss=bool(kwargs.get("rss", True)),
    )


def _get_cache_dir(path: Optional[str]) -> Path:
    if path:
        base = Path(path).expanduser()
    else:
        base = config_core.resolve_root(None) / "cache" / "sordino"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _resolve_reco_id(scan: Any, reco_id: Optional[int]) -> Optional[int]:
    if reco_id is not None:
        return reco_id
    avail = getattr(scan, "avail", None)
    if isinstance(avail, dict) and avail:
        return sorted(avail.keys())[0]
    return None


def _get_param(params: Any, key: str, default: Any = None) -> Any:
    if params is None:
        return default
    getter = getattr(params, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(params, key, default)


def get_gradient_axis_order(method: Any) -> list[int]:
    axis_decoder = {"axial": "L_R", "sagittal": "A_P", "coronal": "L_R"}
    slice_orient = _get_param(method, "PVM_SPackArrSliceOrient")
    read_orient = _get_param(method, "PVM_SPackArrReadOrient")
    if isinstance(slice_orient, (list, tuple, np.ndarray)):
        slice_orient = slice_orient[0] if slice_orient else None
    if isinstance(read_orient, (list, tuple, np.ndarray)):
        read_orient = read_orient[0] if read_orient else None
    if not slice_orient or not read_orient:
        return [0, 1, 2]
    if axis_decoder.get(str(slice_orient).lower()) != str(read_orient):
        return [1, 0, 2]
    return [0, 1, 2]


def get_orient_info(visu_pars: Any) -> Optional[Dict[str, Any]]:
    if visu_pars is None:
        return None
    orient_raw = _get_param(visu_pars, "VisuCoreOrientation")
    if orient_raw is None:
        return None
    orient_matrix = np.squeeze(np.round(np.asarray(orient_raw))).reshape([3, 3])
    slice_order = str(_get_param(visu_pars, "VisuCoreDiskSliceOrder", "")).lower()
    reversed_slice = "reverse" in slice_order
    position = np.squeeze(np.asarray(_get_param(visu_pars, "VisuCorePosition", [0, 0, 0])))
    return dict(orient_matrix=orient_matrix, reversed_slice=reversed_slice, position=position)


def _parse_params(scan: Any, reco_id: int, options: _Options) -> Dict[str, Any]:
    method = getattr(scan, "method", None)
    acqp = getattr(scan, "acqp", None)
    if method is None or acqp is None:
        raise ValueError("Scan is missing method/acqp parameters.")
    reco = None
    try:
        reco = scan.get_reco(reco_id)
    except Exception:
        reco = None
    visu_pars = getattr(reco, "visu_pars", None) or getattr(scan, "visu_pars", None)
    dtype_info = datatype_resolver.resolve(scan)
    if not dtype_info or "dtype" not in dtype_info:
        raise ValueError("Failed to resolve FID dtype from acqp.")
    dtype = dtype_info["dtype"]

    n_receivers = int(_get_param(method, "PVM_EncNReceivers"))
    n_points = int(_get_param(method, "NPoints"))
    n_pro = int(_get_param(method, "NPro"))
    fid_shape = np.array([2, n_points, n_receivers, n_pro])

    tr = _get_param(method, "RepetitionTime", None)
    if tr is None:
        tr = _get_param(method, "PVM_RepetitionTime", None)
    if tr is None:
        raise ValueError("Missing repetition time in method parameters.")

    eff_bw = float(_get_param(method, "PVM_EffSWh"))
    over_sampling = float(_get_param(method, "OverSampling"))
    dwell_time = 1.0 / (eff_bw * over_sampling) * 1e6
    acq_delay = _get_param(method, "AcqDelayTotal", 0.0)
    traj_offset = acq_delay
    if options.traj_offset is not None:
        traj_offset = float(options.traj_offset)

    orient_info = get_orient_info(visu_pars)
    subj_type = _get_param(visu_pars, "VisuSubjectType", "Biped")
    params = dict(
        subj_type=subj_type,
        axis_order=get_gradient_axis_order(method),
        orient_info=orient_info,
        fid_shape=fid_shape,
        n_frames=int(_get_param(method, "PVM_NRepetitions")),
        matrix_size=np.asarray(_get_param(method, "PVM_Matrix"), dtype=float),
        eff_bandwidth=eff_bw,
        over_sampling=over_sampling,
        under_sampling=float(_get_param(method, "ProUnderSampling", 0.0)),
        resol=np.asarray(_get_param(method, "PVM_SpatResol"), dtype=float),
        fov=np.asarray(_get_param(method, "PVM_Fov"), dtype=float),
        half_acquisition=_as_bool(_get_param(method, "HalfAcquisition")),
        use_origin=_as_bool(_get_param(method, "UseOrigin")),
        reorder=_as_bool(_get_param(method, "Reorder")),
        dtype_code=dtype,
        buffer_size=int(np.prod(fid_shape) * dtype.itemsize),
        repetition_time=float(tr) / 1000.0,
        dwell_time=dwell_time,
        traj_offset=traj_offset,
        ext_factors=np.asarray(options.ext_factors, dtype=float),
        cache=[],
        reco_id=reco_id,
    )
    logger.debug(
        "Parsed params: fid_shape=%s matrix_size=%s n_frames=%s eff_bw=%s over_sampling=%s",
        params["fid_shape"],
        params["matrix_size"],
        params["n_frames"],
        params["eff_bandwidth"],
        params["over_sampling"],
    )
    return params


def radial_angles(n: int, factor: float) -> int:
    return int(np.ceil((np.pi * n * factor) / 2))


def radial_angle(i: int, n: int) -> float:
    return np.pi * (i + 0.5) / n


def recon_output_shape(params: Dict[str, Any]) -> list[int]:
    ext_factors = params["ext_factors"]
    matrix_size = np.array(params["matrix_size"])
    oshape = (matrix_size * ext_factors).astype(int).tolist()
    return oshape


def recon_n_frames(options: _Options, params: Dict[str, Any]) -> int:
    total_frames = params["n_frames"]
    offset = options.offset or 0
    avail_frames = total_frames - offset
    set_frames = options.num_frames or total_frames
    if set_frames > avail_frames:
        set_frames = avail_frames
    return int(set_frames)


def recon_buffer_offset(options: _Options, params: Dict[str, Any]) -> int:
    return int(options.offset or 0) * int(params["buffer_size"])


def get_vol_scantime(params: Dict[str, Any]) -> float:
    return float(params["repetition_time"]) * float(params["fid_shape"][3])


def calc_n_pro(matrix_size: int, under_sampling: float) -> int:
    usamp = np.sqrt(under_sampling)
    n_theta = radial_angles(matrix_size, 1 / usamp)
    n_pro = 0
    for i_theta in range(n_theta):
        theta = radial_angle(i_theta, n_theta)
        n_phi = radial_angles(matrix_size, np.sin(theta) / usamp)
        n_pro += n_phi
    return int(n_pro)


def find_undersamp(matrix_size: int, n_pro_target: int) -> float:
    from scipy.optimize import brentq

    def func(under_sampling: float) -> float:
        n_pro = calc_n_pro(matrix_size, under_sampling)
        return float(n_pro - n_pro_target)

    max_val = calc_n_pro(matrix_size, 1)
    start = 1e-6
    end = max_val / matrix_size
    if func(start) * func(end) > 0:
        raise ValueError("The function does not change sign over the interval.")
    undersamp_solution = brentq(func, start, end, xtol=1e-6)
    if isinstance(undersamp_solution, tuple):
        return float(undersamp_solution[0])
    return float(undersamp_solution)


def calc_radial_traj3d(
    grad_array: np.ndarray,
    matrix_size: int,
    use_origin: bool,
    over_sampling: float,
    ramp_time_corr: bool = False,
    traj_offset: Optional[float] = None,
    traj_denom: Optional[float] = None,
) -> np.ndarray:
    pro_offset = 1 if use_origin else 0
    g = grad_array.copy()
    n_pro = g.shape[-1]
    traj_offset = traj_offset or 0
    num_samples = int(matrix_size / 2 * over_sampling)
    traj = np.zeros([n_pro, num_samples, 3])
    for i_pro in _progress(range(pro_offset, n_pro + pro_offset), desc="traj", ncols=100):
        for i_samp in range(num_samples):
            if traj_denom:
                samp = ((i_samp + traj_offset) / traj_denom) / 2
            else:
                samp = ((i_samp + traj_offset) / (num_samples - 1)) / 2
            if not ramp_time_corr or i_pro == (n_pro + pro_offset) - 1:
                correction = np.zeros(3)
                traj[i_pro, i_samp, :] = samp * (g[:, i_pro] + correction)
            else:
                correction = (g[:, i_pro] - g[:, i_pro - 1]) / num_samples * i_samp
                traj[i_pro, i_samp, :] = samp * (g[:, i_pro - 1] + correction)
    return traj


def calc_radial_grad3d(
    matrix_size: int,
    n_pro_target: int,
    half_sphere: bool,
    use_origin: bool,
    reorder: bool,
) -> np.ndarray:
    n_pro = int(n_pro_target / (1 if half_sphere else 2) - (1 if use_origin else 0))
    usamp = np.sqrt(find_undersamp(matrix_size, n_pro))
    grad = {"r": [], "p": [], "s": []}
    radial_n_phi: list[int] = []

    n_theta = radial_angles(matrix_size, 1.0 / usamp)
    for i_theta in range(n_theta):
        theta = radial_angle(i_theta, n_theta)
        n_phi = radial_angles(matrix_size, float(np.sin(theta) / usamp))
        radial_n_phi.append(n_phi)
        for i_phi in range(n_phi):
            phi = radial_angle(i_phi, n_phi)
            grad["r"].append(np.sin(theta) * np.cos(phi))
            grad["p"].append(np.sin(theta) * np.sin(phi))
            grad["s"].append(np.cos(theta))

    grad_array = np.stack([grad["r"], grad["p"], grad["s"]], axis=0)
    n_pro_created = grad_array.shape[-1] * (1 if half_sphere else 2) + (1 if use_origin else 0)
    if not usamp:
        if n_pro_created != n_pro_target:
            raise ValueError("Target number of projections can't be reached.")

    grad_array = reorder_projections(n_theta, radial_n_phi, grad_array, reorder)

    if not half_sphere:
        grad_array = np.concatenate([grad_array, -1 * grad_array], axis=1)

    if use_origin:
        grad_array = np.concatenate([[[0, 0, 0]], grad_array.T], axis=0).T

    return grad_array


def reorder_projections(
    n_theta: int,
    radial_n_phi: list[int],
    grad_array: np.ndarray,
    reorder: bool,
) -> np.ndarray:
    g = grad_array.copy()
    if reorder:
        def reorder_incr_index(n: int, i: int, d: int) -> tuple[int, int]:
            if (i + d > n - 1) or (i + d < 0):
                d *= -1
            i += d
            return i, d

        n_pro = g.shape[-1]
        n_phi_max = max(radial_n_phi)
        r_g = np.zeros_like(g)
        r_mask = np.zeros([n_theta, n_phi_max])

        for i_theta in range(n_theta):
            for i_phi in range(radial_n_phi[i_theta], n_phi_max):
                r_mask[i_theta][i_phi] = 1

        i_theta = 0
        d_theta = 1
        i_phi = 0
        d_phi = 1

        for i in range(n_pro):
            while not any(r_mask[i_theta] == 0):
                i_theta, d_theta = reorder_incr_index(n_theta, i_theta, d_theta)

            while r_mask[i_theta][i_phi] == 1:
                i_phi, d_phi = reorder_incr_index(n_phi_max, i_phi, d_phi)
            new_i = sum(radial_n_phi[:i_theta]) + i_phi
            r_g[:, i] = g[:, new_i]
            r_mask[i_theta][i_phi] = 1

            i_theta, d_theta = reorder_incr_index(n_theta, i_theta, d_theta)
            i_phi, d_phi = reorder_incr_index(n_phi_max, i_phi, d_phi)
        return r_g

    i = 0
    for i_theta in range(n_theta):
        if i_theta % 2 == 1:
            for i_phi in range(int(radial_n_phi[i_theta] / 2)):
                i0 = i + i_phi
                i1 = i + radial_n_phi[i_theta] - 1 - i_phi
                g[:, i0], g[:, i1] = g[:, i1].copy(), g[:, i0].copy()
        i += radial_n_phi[i_theta]
    return g


def generate_hash(*args: Any) -> str:
    hash_input = "".join(str(arg) for arg in args)
    return hashlib.md5(hash_input.encode()).hexdigest()


def get_trajectory(options: _Options, params: Dict[str, Any], tmpdir: Path) -> np.ndarray:
    matrix_size = int(params["matrix_size"][0])
    eff_bandwidth = float(params["eff_bandwidth"])
    over_sampling = float(params["over_sampling"])
    n_pro = int(params["fid_shape"][3])
    half_acquisition = bool(params["half_acquisition"])
    use_origin = bool(params["use_origin"])
    reorder = bool(params["reorder"])
    ext_factors = params["ext_factors"]
    if options.traj_offset is None:
        traj_offset_time = float(params["traj_offset"])
    else:
        traj_offset_time = float(options.traj_offset)

    grad = calc_radial_grad3d(matrix_size, n_pro, half_acquisition, use_origin, reorder)
    offset_factor = traj_offset_time * (10 ** -6) * eff_bandwidth * over_sampling

    option_for_hash = (
        float(traj_offset_time),
        matrix_size,
        eff_bandwidth,
        over_sampling,
            int(n_pro),
            float(ext_factors[0]),
            bool(half_acquisition),
            bool(use_origin),
            bool(reorder),
        options.ramp_time,
    )
    digest = generate_hash(*option_for_hash)
    traj_path = tmpdir / f"{digest}.npy"
    if traj_path.exists():
        logger.debug("Trajectory cache hit: %s", traj_path)
        traj = np.load(traj_path)
    else:
        logger.info("Computing trajectory (matrix=%s, n_pro=%s).", matrix_size, n_pro)
        traj = calc_radial_traj3d(
            grad,
            matrix_size,
            use_origin,
            over_sampling,
            ramp_time_corr=options.ramp_time,
            traj_offset=offset_factor,
            traj_denom=options.traj_denom,
        )
        np.save(traj_path, traj)
        logger.debug("Saved trajectory cache: %s", traj_path)
    return traj


def correct_spoketiming(
    fid_f: Any,
    stc_f: Any,
    options: _Options,
    params: Dict[str, Any],
    stc_params: Dict[str, Any],
) -> None:
    from scipy.interpolate import interp1d

    pro_loc = 0
    stc_dtype = None
    stc_buffer_size = None

    target_timestamps = stc_params["target_timestamps"]
    for seg_size in _progress(stc_params["segs"], desc="segments", ncols=100):
        pro_offset = pro_loc * stc_params["buffer_size_per_pro"]
        seg_buffer_size = stc_params["buffer_size_per_pro"] * seg_size
        seg = []
        for t in range(recon_n_frames(options, params)):
            frame_offset = t * params["buffer_size"]
            seek_loc = recon_buffer_offset(options, params) + frame_offset + pro_offset
            fid_f.seek(seek_loc)
            seg.append(fid_f.read(seg_buffer_size))
        seg_data = np.frombuffer(b"".join(seg), dtype=params["dtype_code"])
        seg_data = seg_data.reshape(
            [2, np.prod(params["fid_shape"][1:3]), seg_size, recon_n_frames(options, params)],
            order="F",
        )
        corrected_seg_data = np.empty_like(seg_data)

        for pro_id in range(seg_size):
            cur_pro = pro_loc + pro_id
            ref_timestamps = stc_params["base_timestamps"] + (cur_pro * params["repetition_time"])

            for e in range(np.prod(params["fid_shape"][1:3])):
                complex_feed = seg_data[0, e, pro_id, :] + 1j * seg_data[1, e, pro_id, :]
                mag = np.abs(complex_feed)
                phase = np.angle(complex_feed)
                phase_unw = np.unwrap(phase)
                interp_mag = interp1d(
                    ref_timestamps,
                    mag,
                    kind="linear",
                    bounds_error=False,
                    fill_value=cast(Any, "extrapolate"),
                )
                interp_phase = interp1d(
                    ref_timestamps,
                    phase_unw,
                    kind="linear",
                    bounds_error=False,
                    fill_value=cast(Any, "extrapolate"),
                )
                mag_t = interp_mag(target_timestamps)
                phase_t = interp_phase(target_timestamps)
                phase_wrap = (phase_t + np.pi) % (2 * np.pi) - np.pi
                z_t = mag_t * np.exp(1j * phase_wrap)
                corrected_seg_data[0, e, pro_id, :] = z_t.real
                corrected_seg_data[1, e, pro_id, :] = z_t.imag
        for t in range(recon_n_frames(options, params)):
            frame_offset = t * params["buffer_size"]
            stc_f.seek(frame_offset + pro_offset)
            stc_f.write(corrected_seg_data[:, :, :, t].flatten(order="F").tobytes())

        if not stc_dtype:
            stc_dtype = corrected_seg_data.dtype
            stc_buffer_size = np.prod(params["fid_shape"]) * stc_dtype.itemsize
        pro_loc += seg_size
    stc_params["buffer_size"] = stc_buffer_size
    stc_params["dtype"] = stc_dtype


def run_spoketiming_correction(
    fid_entry: Any,
    recon_f: Any,
    options: _Options,
    params: Dict[str, Any],
    tmpdir: Path,
) -> Dict[str, Any]:
    n_pro = params["fid_shape"][3]
    vol_scantime = get_vol_scantime(params)
    base_timestamps = np.arange(recon_n_frames(options, params)) * vol_scantime
    stc_buffer_size = int(params["buffer_size"] / n_pro)
    stc_params: Dict[str, Any] = dict(
        base_timestamps=base_timestamps,
        target_timestamps=base_timestamps + (vol_scantime / 2),
        buffer_size_per_pro=stc_buffer_size,
    )

    with tempfile.NamedTemporaryFile(mode="w+b", delete=False, dir=tmpdir) as stc_f:
        with fid_entry.open() as fid_f:
            fid_f.seek(0, os.SEEK_END)
            file_size = fid_f.tell()
            fid_f.seek(0)
            if params["n_frames"] > 0:
                file_size *= recon_n_frames(options, params) / params["n_frames"]
            file_size_gb = file_size / (1024 ** 3)
            num_segs = int(np.ceil(file_size_gb / options.mem_limit)) if options.mem_limit > 0 else 1
            n_pro_per_seg = int(np.ceil(n_pro / num_segs))
            if residual_pro := n_pro % n_pro_per_seg:
                segs = [n_pro_per_seg for _ in range(num_segs - 1)] + [residual_pro]
            else:
                segs = [n_pro_per_seg for _ in range(num_segs)]
            stc_params["segs"] = np.asarray(segs, dtype=int)
            logger.info("Spoketiming correction: %s segment(s).", len(segs))
            correct_spoketiming(fid_f, stc_f, options, params, stc_params)

    with open(stc_f.name, "r+b") as fid_f:
        recon_params = reconstruct_image(fid_f, recon_f, options, params, stc_params)
    params["cache"].append(stc_f.name)
    return recon_params


def correct_offreso(k: np.ndarray, shift_freq: float, params: Dict[str, Any]) -> np.ndarray:
    bw = params["eff_bandwidth"] * params["over_sampling"]
    m_k = k.copy()
    num_samp = m_k.shape[1]
    for samp_id in range(num_samp):
        m_k[:, samp_id] *= np.exp(-1j * 2 * shift_freq * np.pi * ((samp_id + 1) / bw))
    return m_k


def reconstruct_image(
    fid_f: Any,
    recon_f: Any,
    options: _Options,
    params: Dict[str, Any],
    stc_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    recon_f.seek(0)
    if stc_params:
        fid_f.seek(0)
        buffer_size = stc_params["buffer_size"]
        dtype = stc_params["dtype"]
    else:
        fid_f.seek(recon_buffer_offset(options, params))
        buffer_size = params["buffer_size"]
        dtype = params["dtype_code"]

    traj = params["traj"][:, options.pass_samples :, ...]
    logger.debug("Reconstruction traj shape: %s", traj.shape)
    recon_dtype = np.complex64
    for n in _progress(range(recon_n_frames(options, params)), desc="frames", ncols=100):
        buffer = fid_f.read(buffer_size)
        v = np.frombuffer(buffer, dtype=dtype).reshape(params["fid_shape"], order="F")
        v = (v[0] + 1j * v[1])[np.newaxis, ...]
        ksp = v.squeeze().T[..., options.pass_samples :]
        n_receivers = params["fid_shape"][2]
        if n_receivers > 1:
            recon_vol = []
            for ch_id in range(n_receivers):
                k = ksp[:, ch_id, :]
                if options.offreso_ch and ch_id == options.offreso_ch - 1:
                    k = correct_offreso(k, options.offreso_freq, params)
                recon_vol.append(nufft_adjoint(params, k, traj, operator=options.operator))
            recon_vol = np.stack(recon_vol, axis=0)
        else:
            recon_vol = nufft_adjoint(params, ksp, traj, operator=options.operator)
        if n == 0:
            recon_dtype = recon_vol.dtype
        recon_f.write(recon_vol.T.flatten(order="C").tobytes())
    return dict(dtype=recon_dtype)


def nufft_adjoint(
    params: Dict[str, Any],
    kspace: np.ndarray,
    traj: np.ndarray,
    operator: str = "finufft",
) -> np.ndarray:
    from mrinufft import get_operator

    output_shape = recon_output_shape(params)
    dcf = np.sqrt(np.square(traj).sum(-1)).flatten() ** 2
    dcf /= dcf.max()
    traj = traj.copy() / 0.5 * np.pi
    nufft_op = get_operator(operator)(traj, shape=output_shape, density=dcf)
    complex_img = nufft_op.adj_op(kspace.flatten())
    return complex_img


def run_reconstruction(
    fid_entry: Any,
    options: _Options,
    params: Dict[str, Any],
    tmpdir: Path,
) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(mode="w+b", delete=False, dir=tmpdir) as recon_f:
        if options.spoketiming:
            logger.info("Running spoke timing correction.")
            recon_params = run_spoketiming_correction(fid_entry, recon_f, options, params, tmpdir)
        else:
            logger.info("Running reconstruction.")
            with fid_entry.open() as fid_f:
                recon_params = reconstruct_image(fid_f, recon_f, options, params)
        recon_params["name"] = recon_f.name
    logger.debug("Reconstruction output temp file: %s", recon_params["name"])
    return recon_params


def convert_to_numpy(recon_params: Dict[str, Any], options: _Options, params: Dict[str, Any]) -> np.ndarray:
    n_receivers = params["fid_shape"][2]
    n_frames = recon_n_frames(options, params)
    with open(recon_params["name"], "r+b") as img_f:
        if n_receivers > 1:
            oshape = [n_receivers] + recon_output_shape(params)
        else:
            oshape = recon_output_shape(params)
        logger.debug("Loading recon data as %s with shape %s", recon_params["dtype"], oshape)
        imgs = np.abs(
            np.frombuffer(img_f.read(), dtype=recon_params["dtype"]).reshape(oshape + [n_frames], order="F")
        )
    params["cache"].append(recon_params["name"])
    return imgs


def combine_rss(imgs: np.ndarray) -> np.ndarray:
    logger.info("Applying RSS combine across receiver channels.")
    return np.sqrt(np.sum(imgs ** 2, axis=0))


def _transpose_spatial(data: np.ndarray, axis_order: list[int], *, has_channel: bool, has_time: bool) -> np.ndarray:
    if axis_order == [0, 1, 2]:
        return data
    if has_channel:
        axes = [0] + [a + 1 for a in axis_order]
        if has_time:
            axes.append(data.ndim - 1)
    else:
        axes = axis_order + ([data.ndim - 1] if has_time else [])
    return np.transpose(data, axes)


def _realign_to_2dseq(data: np.ndarray, params: Dict[str, Any], options: _Options) -> np.ndarray:
    axis_order = params.get("axis_order")
    orient_info = params.get("orient_info") or {}
    if not axis_order:
        return data
    n_receivers = params["fid_shape"][2]
    has_channel = n_receivers > 1
    data = _transpose_spatial(data, list(axis_order), has_channel=has_channel, has_time=True)
    if orient_info.get("reversed_slice"):
        slice_axis = axis_order.index(2)
        if has_channel:
            slice_axis += 1
        slicer = [slice(None)] * data.ndim
        slicer[slice_axis] = slice(None, None, -1)
        data = data[tuple(slicer)]
    return data


def _cleanup_cache(params: Dict[str, Any]) -> None:
    for fpath in params.get("cache", []):
        try:
            os.remove(fpath)
        except OSError:
            pass


def get_dataobj(
    scan: Any,
    reco_id: Optional[int] = None,
    **kwargs: Any,
) -> Optional[np.ndarray]:
    options = _build_options(kwargs)
    logger.info("SORDINO hook get_dataobj start (scan_id=%s).", getattr(scan, "scan_id", "?"))
    resolved_reco_id = _resolve_reco_id(scan, reco_id)
    if resolved_reco_id is None:
        logger.warning("No reco id available for SORDINO reconstruction.")
        return None

    fid_entry = fid_resolver.get_fid(scan)
    if fid_entry is None:
        logger.warning("No FID/rawdata entry found for scan %s.", getattr(scan, "scan_id", "?"))
        return None

    tmpdir = _get_cache_dir(kwargs.get("cache_dir"))
    logger.debug("Cache dir: %s", tmpdir)
    params = _parse_params(scan, resolved_reco_id, options)
    params["tmpdir"] = str(tmpdir)

    params["traj"] = get_trajectory(options, params, tmpdir)
    recon_params = run_reconstruction(fid_entry, options, params, tmpdir)
    imgs = convert_to_numpy(recon_params, options, params)
    imgs = _realign_to_2dseq(imgs, params, options)

    if params["fid_shape"][2] > 1 and options.rss:
        imgs = combine_rss(imgs)
    else:
        imgs = imgs.squeeze()

    if options.clear_cache:
        logger.debug("Clearing cache files (%s).", len(params.get("cache", [])))
        _cleanup_cache(params)

    logger.info("SORDINO hook get_dataobj done.")
    return imgs


HOOK = {"get_dataobj": get_dataobj}

__all__ = ["HOOK", "get_dataobj"]
