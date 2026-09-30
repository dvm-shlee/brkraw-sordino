import gc
import os
import json
import hashlib
import platform
from pathlib import Path
import numpy as np
from typing import Any, Dict, Tuple, Optional
from numpy.typing import NDArray
from mrinufft import get_operator
import logging
from .helper import progressbar
from .typing import Options

logger = logging.getLogger(__name__)


def _hash_cache_params(params: Dict[str, Any], *, salt: str) -> str:
    payload = json.dumps(params, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha1(f"{salt}:{payload}".encode("utf-8")).hexdigest()


def build_recon_cache_path(cache_dir: Path, cache_params: Dict[str, Any]) -> Path:
    cache_hash = _hash_cache_params(cache_params, salt="recon")
    return cache_dir / f"recon_{cache_hash}.bin"

def _get_current_rss_gb() -> Optional[float]:
    if platform.system() != "Linux":
        return None
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            parts = handle.read().strip().split()
        if len(parts) < 2:
            return None
        rss_pages = int(parts[1])
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (rss_pages * page_size) / (1024 ** 3)
    except Exception:
        return None


def parse_fid_info(recon_info: Dict[str, Any]) -> Tuple[np.ndarray, np.dtype]:
    """Parse FID dimensions and dtype from reconstruction metadata.

    Args:
        recon_info (Dict[str, Any]): Reconstruction metadata including
            "EncNReceivers", "NPoints", "NPro", and "FIDDataType".

    Returns:
        Tuple[np.ndarray, np.dtype]: FID shape array
        `[2, n_points, n_receivers, n_pro]` and the FID dtype.

    Raises:
        ValueError: If required dimensions are missing or zero.
    """
    n_receivers = int(recon_info.get("EncNReceivers") or 0)
    n_points = int(recon_info.get("NPoints") or 0)
    n_pro = int(recon_info.get("NPro") or 0)
    dtype = recon_info['FIDDataType']
    if not all((n_receivers, n_points, n_pro)):
        raise ValueError("Missing reconstruction dimensions in recon_spec output.")
    return np.array([2, n_points, n_receivers, n_pro]), dtype


def get_num_frames(recon_info: Dict[str, Any], options: Options):
    """Return the number of data frames to reconstruct.

    Args:
        recon_info (Dict[str, Any]): Reconstruction metadata containing
            "NRepetitions".
        options (Options): Reconstruction options that may include "offset"
            and "num_frames".

    Returns:
        int: Number of frames to reconstruct after applying offset and limits.
    """
    total_frames = recon_info['NRepetitions']
    offset = getattr(options, 'offset') or 0
    avail_frames = total_frames - offset
    set_frames = getattr(options, 'num_frames') or total_frames
    
    if set_frames > avail_frames:
        diff = set_frames - avail_frames
        set_frames -= diff
    return set_frames


def parse_volume_shape(recon_info: Dict[str, Any], 
                       options: Options) -> NDArray[np.int_]:
    """Determine the output volume shape for reconstruction.

    Args:
        recon_info (Dict[str, Any]): Reconstruction metadata with "Matrix" or
            "NPoints".
        options (Options): Reconstruction options that may include "ext_factors".

    Returns:
        NDArray[np.int_]: Volume shape array after applying extension factors.
    """
    matrix = recon_info.get("Matrix")
    if matrix is None:
        matrix = [int(recon_info.get("NPoints") or 0)] * 3
        logger.warning(" - Matrix size missing; defaulting to %s.", matrix)
    ext_factors = getattr(options, 'ext_factors', None)
    if ext_factors is None: 
        ext_factors = [1.0, 1.0, 1.0]
    return np.asarray(matrix * np.asarray(ext_factors)).astype(int).tolist()


def get_dataobj_shape(recon_info: Dict[str, Any], 
                      options: Options):
    """Compute the output data object shape including frames and receivers.

    Args:
        recon_info (Dict[str, Any]): Reconstruction metadata.
        options (Options): Reconstruction options.

    Returns:
        list[int]: Shape of the reconstructed data object.
    """
    num_receivers = parse_fid_info(recon_info)[0][2]
    vol_shape = parse_volume_shape(recon_info, options)
    num_frame = get_num_frames(recon_info, options)

    if num_receivers > 1:
        return [num_receivers] + vol_shape + [num_frame]
    else:
        return vol_shape + [num_frame]


def nufft_adjoint(kspace, traj, volume_shape, log_counter=0, operator='finufft'):
    """Run adjoint NUFFT and return the reconstructed image.

    Args:
        kspace (np.ndarray): Input k-space data.
        traj (np.ndarray): Trajectory coordinates.
        volume_shape (Sequence[int]): Output volume shape.
        log_counter (int, optional): Log verbosity flag; logs details on 0.
        operator (str, optional): NUFFT backend name (e.g., "finufft").

    Returns:
        np.ndarray: Reconstructed complex image volume.
    """
    dcf = np.sqrt(np.square(traj).sum(-1)).flatten() ** 2
    dcf /= dcf.max()
    if log_counter == 0:
        logger.debug("Processing NUFFT")
        logger.debug(" - DCF shape: %s", dcf.shape)
        logger.debug(" - Trajectory shape: %s", traj.shape)
        logger.debug(" - Volume shape: %s", volume_shape)
    omega = traj.reshape(-1, traj.shape[-1]) / 0.5 * np.pi
    nufft_op = make_nufft_operator(omega, volume_shape, dcf, operator)
    complex_img = nufft_op.adj_op(kspace.flatten())
    return complex_img


def make_nufft_operator(omega, volume_shape, density, operator='finufft'):
    """NUFFT operator at the radian coordinates ``omega`` (shape (M, 3), [-pi, pi)).

    mrinufft (``proper_trajectory(normalize="pi")``, checked on 1.5.1)
    multiplies a trajectory by 2 pi when its largest |omega| is below about
    0.5 rad, assuming it was given in [-0.5, 0.5). Full spokes reach pi and
    are not touched, but an operator built only on points near the centre
    (within about Matrix / (4 pi) k-grid units, for example a centre fill or a
    probe of the first samples) would silently be evaluated at 2 pi times the
    radius (WI-0058 run 2, BRK-0063). When the stored samples differ from
    ``omega``, they are set again, unchanged, through
    ``update_samples(unsafe=True)`` and the density is restored; if they still
    differ, a ``RuntimeError`` is raised instead of returning a wrong image.
    """
    import warnings

    omega = np.asarray(omega)
    if not np.issubdtype(omega.dtype, np.floating):
        omega = omega.astype(np.float64)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Samples will be rescaled")
        nufft_op = get_operator(operator)(omega, shape=volume_shape, density=density)
    if not _same_samples(nufft_op, omega):
        logger.debug(" - NUFFT samples were rescaled by the backend; setting the radians again")
        nufft_op.update_samples(np.asarray(omega, order="F"), unsafe=True)
        if density is not None and density is not False:
            nufft_op.density = density
        if not _same_samples(nufft_op, omega):
            raise RuntimeError("NUFFT operator does not keep the trajectory coordinates")
    return nufft_op


def _same_samples(nufft_op, omega) -> bool:
    samples = np.asarray(nufft_op.samples).reshape(omega.shape)
    return bool(np.allclose(samples, omega, rtol=0.0, atol=1e-6))


def phase_correction_factor(recon_info: Dict[str, Any], options: Options,
                            n_points: int) -> Optional[np.ndarray]:
    """Per-projection accumulated phase of the ramped gradient (WI-0056).

    The receiver frequency of projection i is ACQ_O1_list[i], set for the
    target vector g(i); while the gradient still moves from g(i-1), spins at
    the FOV offset collect phi_ij = 2*pi*(O1[i-1] - O1[i]) * tau_j, with
    tau_j the integral of (1 - ramp fraction) from the phase reference (RF
    centre) to sample j (timing.ramp_terms). Returns exp(-1j*phi) with shape
    (n_pro, n_points), or None when no correction applies: correct_ramptime
    off, no FOV offset (O1 list of length 1), or a constant gradient (general
    ZTE). The phase correction is part of ``correct_ramptime`` (BRK-0066).
    The sign follows the phase observed on v2 data (WI-0056 run 3).
    """
    from . import timing as timing_mod

    if not getattr(options, "correct_ramptime", True):
        return None
    o1 = np.asarray(recon_info.get("O1List_Hz") or [], dtype=float)
    n_pro = int(recon_info.get("NPro") or 0)
    if o1.size != n_pro or n_pro == 0:
        if o1.size == 1:   # a single constant frequency: no FOV offset, nothing to correct
            logger.debug(" - Phase correction not needed: single O1 value")
        else:
            logger.warning("Phase correction skipped: ACQ_O1_list has %s values, NPro is %s.",
                           o1.size, n_pro)
        return None
    seq = timing_mod.read_timing(recon_info)
    tune = timing_mod.tuning_for(seq.version)
    _, _, tau_us = timing_mod.ramp_terms(seq, tune, n_points)
    tau_s = np.asarray(tau_us, dtype=float) * 1e-6
    if not np.any(tau_s):
        return None
    d = np.roll(o1, 1) - o1
    logger.debug(" - Phase correction: version %s, max |step| %.1f Hz", seq.version, np.abs(d).max())
    return np.exp(-2j * np.pi * np.outer(d, tau_s)).astype(np.complex64)


def correct_offreso(kspace: np.ndarray, shift_freq: float, *, eff_bandwidth: float, over_sampling: float) -> np.ndarray:
    if shift_freq == 0.0:
        return kspace
    bw = float(eff_bandwidth) * float(over_sampling)
    if bw == 0.0:
        return kspace
    num_samp = kspace.shape[1]
    phase = np.exp(-1j * 2 * np.pi * shift_freq * ((np.arange(num_samp) + 1) / bw))
    return kspace * phase[np.newaxis, :]


def recon_dataobj(fid_fobj, 
                  traj, 
                  recon_info: Dict[str, Any],
                  img_fobj,
                  options: Options,
                  override_buffer_size=None,
                  override_dtype=None,
                  phase_factor=None,
                  virtual_traj=None,
                  k0_out=None):
    """Reconstruct image volumes from FID data and write to an output file.

    Args:
        fid_fobj (IO[bytes]): Input FID file handle.
        traj (np.ndarray): K-space trajectory array.
        recon_info (Dict[str, Any]): Reconstruction metadata.
        img_fobj (IO[bytes]): Output image file handle.
        options (Options): Reconstruction options.
        override_buffer_size (Optional[int]): Override FID frame buffer size.
        override_dtype (Optional[np.dtype]): Override FID dtype.
        phase_factor (Optional[np.ndarray]): (n_pro, n_points) factor from
            ``phase_correction_factor``, applied to every frame and channel.
        virtual_traj (Optional[np.ndarray]): (n_pro, M, 3) leading positions
            from ``kcentre.leading_points``; when given (``estimate_k0``), the
            centre is estimated and filled for every frame and channel
            (``kcentre.fill_centre``) instead of the plain adjoint.
        k0_out (Optional[list]): with ``virtual_traj``, one list per frame is
            appended, holding the estimated K0 (complex) of each channel.

    Returns:
        np.dtype: Dtype of the reconstructed output volumes.
    """
    logger.debug("Processing reconstruction")
    img_fobj.seek(0)
    fid_shape, fid_dtype = parse_fid_info(recon_info)
    volume_shape = parse_volume_shape(recon_info, options)
    
    offset = getattr(options, 'offset') or 0
    num_frames = get_num_frames(recon_info, options)
    ignore_samples = getattr(options, 'ignore_samples') or 1

    if all(arg != None for arg in [override_buffer_size, override_buffer_size]):
        logger.debug(" - Use override buffer size and dtype")
        fid_fobj.seek(0)
        buffer_size = override_buffer_size
        fid_dtype = override_dtype
    else:
        buffer_size = int(np.prod(fid_shape) * fid_dtype.itemsize)
        buf_offset = offset * buffer_size
        fid_fobj.seek(buf_offset)
    
    trimmed_traj = traj[:, ignore_samples:, ...]
    logger.debug(" - Reconstruction traj shape: %s", trimmed_traj.shape)
    
    dtype = None
    offreso_freqs = getattr(options, "offreso_freqs", None)
    eff_bandwidth = recon_info.get("EffBandwidth_Hz")
    over_sampling = recon_info.get("OverSampling")

    if virtual_traj is not None:
        from .kcentre import fill_centre

    def _image(k, frame, frame_k0):
        if virtual_traj is None:
            return nufft_adjoint(k, trimmed_traj, volume_shape, frame)
        img, info = fill_centre(k, trimmed_traj, virtual_traj, volume_shape)
        frame_k0.append(info["k0"])
        return img

    for n in progressbar(range(num_frames), desc='frames', ncols=100):
        frame_k0: list = []
        buffer = fid_fobj.read(buffer_size)
        vol = np.frombuffer(buffer, dtype=fid_dtype).reshape(fid_shape, order='F')
        vol = (vol[0] + 1j * vol[1])[np.newaxis, ...]
        k_full = vol.squeeze().T
        if phase_factor is not None:
            # (n_pro, n_points) single channel or (n_pro, n_rx, n_points)
            k_full = k_full * (phase_factor if k_full.ndim == 2 else phase_factor[:, None, :])
        k_space = k_full[..., ignore_samples:]
        rss_gb = _get_current_rss_gb()
        if rss_gb is None:
            logger.debug(" - Reconstruction k-space shape: %s", k_space.shape)
        else:
            logger.debug(
                " - Reconstruction k-space shape: %s (RSS %.2f GB)",
                k_space.shape,
                rss_gb,
            )
        n_receivers = fid_shape[2]

        if n_receivers > 1:
            if n == 0:
                logger.debug(" - Multi-channel reconstruction")
            recon_vol = []
            for ch_id in range(n_receivers):
                if n == 0:
                    logger.debug(" - Channel: %s", ch_id)
                _k_space = k_space[:, ch_id, :]
                apply_offreso = offreso_freqs is not None and len(offreso_freqs) > ch_id
                
                if (
                    apply_offreso
                    and isinstance(offreso_freqs, tuple)
                    and eff_bandwidth is not None
                    and over_sampling is not None
                ):
                    offreso_freq = offreso_freqs[ch_id]
                    if n == 0:
                        logger.info(
                            " - Correcting off-resonance: ch=%s, freq=%.6f Hz",
                            ch_id,
                            offreso_freq,
                        )
                    _k_space = correct_offreso(
                        _k_space,
                        offreso_freq,
                        eff_bandwidth=eff_bandwidth,
                        over_sampling=over_sampling,
                    )
                _vol = _image(_k_space, n, frame_k0)
                recon_vol.append(_vol)
            recon_vol = np.stack(recon_vol, axis=0)
        else:
            if n == 0:
                logger.debug(" - Single-channel reconstruction")
            if (
                isinstance(offreso_freqs, tuple)
                and len(offreso_freqs) > 0
                and eff_bandwidth is not None
                and over_sampling is not None
            ):
                offreso_freq = offreso_freqs[0]
                if n == 0:
                    logger.info(
                        " - Correcting off-resonance: freq=%.6f Hz",
                        offreso_freq,
                    )
                k_space = correct_offreso(
                    k_space,
                    offreso_freq,
                    eff_bandwidth=eff_bandwidth,
                    over_sampling=over_sampling,
                )
            recon_vol = _image(k_space, n, frame_k0)
        if k0_out is not None and virtual_traj is not None:
            k0_out.append(frame_k0)
        if n == 0:
            dtype = recon_vol.dtype
        img_fobj.write(recon_vol.T.flatten(order="C").tobytes())
        # Free the frame's garbage now (WI-0071, D-0098 2): without this, unreachable
        # reference cycles of the NUFFT step pile up between collections and the
        # reconstruction peak grows with the frame count (300 v1 frames: 2369 MiB
        # without, 455 MiB with, about 12 % more time).
        del recon_vol, vol, k_full, k_space
        gc.collect()
    logger.debug("done")
    return dtype

__all__ = [
    'recon_dataobj',
    'get_dataobj_shape',
]
