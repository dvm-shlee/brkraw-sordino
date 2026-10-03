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


class PhaseRows:
    """``phase_correction_factor`` rows computed on demand (WI-0097).

    ``rows[lo:hi]`` (or an index array) equals ``phase_correction_factor(...)[lo:hi]``
    bit for bit: the same expression on the selected projections only, so the
    whole (n_pro, n_points) factor is never held.
    """

    def __init__(self, step_hz: np.ndarray, tau_s: np.ndarray):
        self._d = np.asarray(step_hz, dtype=float)
        self._tau = np.asarray(tau_s, dtype=float)

    @property
    def shape(self) -> tuple:
        return (int(self._d.size), int(self._tau.size))

    def __len__(self) -> int:
        return int(self._d.size)

    def __getitem__(self, idx) -> np.ndarray:
        return np.exp(-2j * np.pi * np.outer(self._d[idx], self._tau)).astype(np.complex64)


def phase_correction_rows(recon_info: Dict[str, Any], options: Options,
                          n_points: int) -> Optional[PhaseRows]:
    """``phase_correction_factor`` as ``PhaseRows``, or None when no correction applies."""
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
    return PhaseRows(d, tau_s)


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
    The sign follows the phase observed on v2 data (WI-0056 run 3). The
    reconstruction uses ``phase_correction_rows`` (the same values per range).
    """
    rows = phase_correction_rows(recon_info, options, n_points)
    return None if rows is None else rows[:]


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
                  k0_out=None,
                  chunk_spokes=None):
    """Reconstruct image volumes from FID data and write to an output file.

    Each frame is reconstructed in contiguous spoke chunks (WI-0097, D-0133):
    a chunk's FID bytes, trajectory rows and phase rows are made, its adjoint
    NUFFT is added to the frame image for every channel, and the chunk is
    released, so the working memory is one chunk plus the image grids
    (``memguard.recon_plan``). The result equals the whole-scan adjoint
    (``nufft_adjoint``) up to the summation order (tests/test_serial_recon.py).

    Args:
        fid_fobj (IO[bytes]): Input FID file handle (read forward only after
            the first seek).
        traj: ``traj.TrajectoryRows`` (the hook) or a whole (n_pro, N, 3)
            trajectory array.
        recon_info (Dict[str, Any]): Reconstruction metadata.
        img_fobj (IO[bytes]): Output image file handle.
        options (Options): Reconstruction options.
        override_buffer_size (Optional[int]): Override FID frame buffer size.
        override_dtype (Optional[np.dtype]): Override FID dtype.
        phase_factor: ``PhaseRows`` from ``phase_correction_rows`` or the
            (n_pro, n_points) array from ``phase_correction_factor``, applied
            to every frame and channel.
        virtual_traj (Optional[np.ndarray]): (n_pro, M, 3) leading positions
            from ``kcentre.leading_points``; when given (``estimate_k0``), the
            centre is estimated and filled for every frame and channel
            (``kcentre.fill_centre``) instead of the plain adjoint.
        k0_out (Optional[list]): with ``virtual_traj``, one list per frame is
            appended, holding the estimated K0 (complex) of each channel.
        chunk_spokes (Optional[int]): largest chunk in spokes (the hook takes it
            from the memory limit); None plans it from the sample cap alone.

    Returns:
        np.dtype: Dtype of the reconstructed output volumes.
    """
    from . import memguard, serial

    logger.debug("Processing reconstruction")
    img_fobj.seek(0)
    fid_shape, fid_dtype = parse_fid_info(recon_info)
    volume_shape = parse_volume_shape(recon_info, options)

    offset = getattr(options, 'offset') or 0
    num_frames = get_num_frames(recon_info, options)
    ignore_samples = getattr(options, 'ignore_samples') or 1

    if override_buffer_size is not None and override_dtype is not None:
        logger.debug(" - Use override buffer size and dtype")
        fid_fobj.seek(0)
        buffer_size = int(override_buffer_size)
        fid_dtype = np.dtype(override_dtype)
    else:
        buffer_size = int(np.prod(fid_shape) * fid_dtype.itemsize)
        buf_offset = offset * buffer_size
        fid_fobj.seek(buf_offset)

    n_points, n_receivers, n_pro = (int(v) for v in fid_shape[1:])
    spoke_bytes = 2 * n_points * n_receivers * np.dtype(fid_dtype).itemsize
    if buffer_size != spoke_bytes * n_pro:
        raise ValueError(f"FID frame of {buffer_size} bytes does not hold {n_pro} spokes "
                         f"of {spoke_bytes} bytes")
    rows = _row_source(traj)
    traj_spokes = int(traj.n_pro) if hasattr(traj, "n_pro") else int(np.shape(traj)[0])
    if traj_spokes != n_pro:
        raise ValueError(f"trajectory has {traj_spokes} spokes, the FID has {n_pro}")
    if chunk_spokes is None:
        chunk_spokes = memguard.recon_plan(n_pro, n_points, n_receivers, volume_shape,
                                           estimate_k0=virtual_traj is not None)["chunk_spokes"]
    ranges = serial.spoke_ranges(n_pro, chunk_spokes)
    logger.debug(" - Reconstruction: %s spokes x %s samples in %s chunk(s) of up to %s spokes",
                 n_pro, n_points - ignore_samples, len(ranges), ranges[0][1] - ranges[0][0])

    offreso_freqs = getattr(options, "offreso_freqs", None)
    eff_bandwidth = recon_info.get("EffBandwidth_Hz")
    over_sampling = recon_info.get("OverSampling")

    def _offreso(ch):
        if (isinstance(offreso_freqs, tuple) and len(offreso_freqs) > ch
                and eff_bandwidth is not None and over_sampling is not None):
            return offreso_freqs[ch]
        return None

    # the density weight is normalised by its maximum over ALL spokes (as the
    # whole-scan adjoint does), so the maximum is found before the first chunk.
    # With estimate_k0 the same pass accumulates the convolution kernel of the
    # least-squares normal operator (once for all frames and channels), and the
    # maximum also covers the virtual leading samples, as kcentre.fill_centre's
    # final adjoint over [virtual, measured] does.
    kernel = None
    if virtual_traj is not None:
        from .kcentre import N_ITER
        kernel = serial.ToeplitzKernel(volume_shape)
    dmax = 0.0
    for lo, hi in ranges:
        tr = rows(lo, hi)[:, ignore_samples:]
        d = serial.density(tr)
        dmax = max(dmax, float(d.max()))
        if kernel is not None:
            kernel.add(tr, d)                  # raw |k|^2; scaled by 1 / dmax below
        del tr, d
    if virtual_traj is not None:
        dmax = max(dmax, float(serial.density(virtual_traj).max()))
        kernel.finish(1.0 / dmax)
        w_virtual = serial.density(virtual_traj) / dmax
    nf = serial.norm_factor(volume_shape)

    dtype = None
    reuse = len(ranges) == 1          # one chunk: points and weights stay set for every frame
    adj = None
    for n in progressbar(range(num_frames), desc='frames', ncols=100):
        frame_k0: list = []
        if n == 0:
            logger.debug(" - %s reconstruction",
                         "Multi-channel" if n_receivers > 1 else "Single-channel")
            for ch in range(n_receivers):
                freq = _offreso(ch)
                if freq is not None:
                    if n_receivers > 1:
                        logger.info(" - Correcting off-resonance: ch=%s, freq=%.6f Hz", ch, freq)
                    else:
                        logger.info(" - Correcting off-resonance: freq=%.6f Hz", freq)
        acc = np.zeros((n_receivers,) + tuple(volume_shape), dtype=np.complex128)
        if adj is None or not reuse:
            adj = serial.Adjoint(volume_shape)
        for lo, hi in ranges:
            k = _read_chunk(fid_fobj, spoke_bytes * (hi - lo), fid_dtype,
                            (2, n_points, n_receivers, hi - lo))
            if phase_factor is not None:
                k = k * phase_factor[lo:hi][:, None, :]
            k = k[..., ignore_samples:]
            for ch in range(n_receivers):
                freq = _offreso(ch)
                if freq is not None:
                    k[:, ch, :] = correct_offreso(k[:, ch, :], freq, eff_bandwidth=eff_bandwidth,
                                                  over_sampling=over_sampling)
            if not reuse or adj.n_points == 0:
                tr = rows(lo, hi)[:, ignore_samples:]
                w = serial.density(tr) / dmax      # kept for every frame when reuse
                adj.setpts(tr)
                del tr
            for ch in range(n_receivers):
                adj.add(acc[ch], k[:, ch, :].reshape(-1) * w)
            del k
        if kernel is not None:
            # estimate_k0 (kcentre.fill_centre, WI-0097 stage 2): acc[ch] is the raw
            # A^H W y; solve A^H W A x = A^H W y by CG on the Toeplitz form, K0 = A_0 x =
            # sum(x), predict the virtual samples A_v x and add their adjoint. The raw x
            # is the product's iterate divided by nf, so the predictions are the same.
            if not reuse:
                adj = None                     # its NUFFT grid is not needed during the solve
            preds = []
            for ch in range(n_receivers):
                x, _ = serial.conjugate_gradient(kernel.normal, acc[ch], N_ITER)
                frame_k0.append(complex(x.sum()))
                preds.append(serial.forward(x, virtual_traj))
                del x
            vadj = serial.Adjoint(volume_shape)
            vadj.setpts(virtual_traj)
            for ch in range(n_receivers):
                vadj.add(acc[ch], preds[ch] * w_virtual)
            del vadj, preds
        acc /= nf
        rss_gb = _get_current_rss_gb()
        if rss_gb is not None:
            logger.debug(" - Frame %s reconstructed (RSS %.2f GB)", n, rss_gb)
        if k0_out is not None and virtual_traj is not None:
            k0_out.append(frame_k0)
        recon_vol = acc if n_receivers > 1 else acc[0]
        if n == 0:
            dtype = recon_vol.dtype
        img_fobj.write(np.ascontiguousarray(recon_vol.T).tobytes())
        if not reuse:
            adj = None
        # Free the frame's garbage now (WI-0071, D-0098 2); chunk arrays are released as
        # each chunk ends, so the working memory is one chunk, not one frame (WI-0097).
        del recon_vol, acc
        gc.collect()
    logger.debug("done")
    return dtype


def _row_source(traj):
    """``rows(lo, hi)``: untrimmed trajectory rows from a ``TrajectoryRows`` or an array."""
    if hasattr(traj, "rows"):
        return traj.rows
    arr = np.asarray(traj)
    return lambda lo, hi: arr[lo:hi]


def _read_chunk(fid_fobj, nbytes: int, dtype, shape) -> np.ndarray:
    """``nbytes`` from the FID stream as complex k-space (spokes, receivers, points)."""
    buf = fid_fobj.read(nbytes)
    if len(buf) < nbytes:
        parts = [buf]
        got = len(buf)
        while got < nbytes:
            more = fid_fobj.read(nbytes - got)
            if not more:
                break
            parts.append(more)
            got += len(more)
        buf = b"".join(parts)
    if len(buf) != nbytes:
        raise ValueError(f"FID data ended early: {len(buf)} of {nbytes} bytes")
    vol = np.frombuffer(buf, dtype=dtype).reshape(shape, order="F")
    k = np.empty(vol.shape[1:], dtype=np.complex128, order="F")
    k.real = vol[0]
    k.imag = vol[1]
    return k.T                                   # (spokes, receivers, points), C order

__all__ = [
    'recon_dataobj',
    'get_dataobj_shape',
]
