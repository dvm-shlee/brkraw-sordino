from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np
import hashlib
import json
import logging
import os
import tempfile
from .helper import progressbar
from .typing import Options

logger = logging.getLogger(__name__)


def radial_angles(n: int, factor: float) -> int:
    return int(np.ceil((np.pi * n * factor) / 2))


def radial_angle(i: int, n: int) -> float:
    return np.pi * (i + 0.5) / n


def recon_output_shape(matrix_size, ext_factors) -> list[int]:
    output_shape = (matrix_size * ext_factors).astype(int).tolist()
    return output_shape


def recon_n_frames(total_frames: int, 
                   offset: int = 0, 
                   num_frame: Optional[int] = None) -> int:
    avail_frames = total_frames - offset
    set_frames = num_frame or total_frames
    if set_frames > avail_frames:
        set_frames = avail_frames
    return int(set_frames)


def recon_buffer_offset(buffer_size: int, offset: Optional[int] = 0) -> int:
    return offset or 0 * buffer_size


def get_vol_scantime(repetition_time: float, fid_shape: np.ndarray) -> float:
    return repetition_time * float(fid_shape[3])


def calc_npro(matrix_size: int, under_sampling: float) -> int:
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
        n_pro = calc_npro(matrix_size, under_sampling)
        return float(n_pro - n_pro_target)

    max_val = calc_npro(matrix_size, 1)
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
    over_sampling: float,
    traj_offset: Optional[float] = None,
) -> np.ndarray:
    """Trajectory with one fixed gradient vector per projection (no ramp model).

    Used by ``correct_ramptime=false`` (every sample of a spoke lies on that
    spoke's target vector, sample j at ``(j + traj_offset) / (2 (N - 1))``).
    The ramp-corrected trajectory is ``calc_radial_traj3d_integral``; the
    earlier curved form was moved to ``tools/legacytraj.py`` (BRK-0066).

    Args:
        grad_array (ndarray): Gradient vector profile for each projection, sized (3 x n_pro).
        matrix_size (int): Matrix size of the final image.
        over_sampling (float): Oversampling factor.
        traj_offset (float, optional): Trajectory offset in samples (acquisition delay).

    Returns:
        ndarray: (n_pro, N, 3) trajectory.
    """
    g = np.asarray(grad_array, dtype=float)
    num_samples = int(matrix_size / 2 * over_sampling)
    off = traj_offset or 0
    samp = ((np.arange(num_samples, dtype=float) + off) / (num_samples - 1)) / 2.0
    logger.debug(" - Fixed-vector trajectory: (%s, %s, 3)", g.shape[-1], num_samples)
    return samp[None, :, None] * g.T[:, None, :]


def calc_radial_traj3d_integral(
    grad_array: np.ndarray,
    matrix_size: int,
    over_sampling: float,
    times_us: list,
    ramp_integral_us: list,
    dwell_us: float,
) -> np.ndarray:
    """Trajectory from the integral of the ramped gradient (WI-0056, BRK-0056).

    For projection i and sample j (times from the RF centre, timing.py):

        k_ij = unit * (t_j * g(i-1) + F_j * (g(i) - g(i-1)))

    with t_j and F_j (integral of the ramp fraction) in samples, and
    unit = 1 / (2 (N - 1)) the same radius scale as calc_radial_traj3d, so a
    constant gradient gives the same trajectory as ``calc_radial_traj3d``
    (the fixed vector).
    g(-1) is the last vector of the list (the dummy spokes and the previous
    frame end on it). Every projection gets the ramp, the last one included.

    Args:
        grad_array: (3, n_pro) gradient vectors.
        matrix_size: matrix size (samples per spoke N = matrix/2 * over_sampling).
        over_sampling: oversampling factor.
        times_us, ramp_integral_us: per-sample t_j and F_j in us (length N).
        dwell_us: sample interval in us.

    Returns:
        (n_pro, N, 3) trajectory.
    """
    g = np.asarray(grad_array, dtype=float)
    n = int(matrix_size / 2 * over_sampling)
    if len(times_us) != n or len(ramp_integral_us) != n:
        raise ValueError("times and ramp integral must have one value per sample")
    unit = 1.0 / (n - 1) / 2.0
    t = np.asarray(times_us, dtype=float) / dwell_us
    f_int = np.asarray(ramp_integral_us, dtype=float) / dwell_us
    g_prev = np.roll(g, 1, axis=1).T              # (n_pro, 3)
    delta = g.T - g_prev
    traj = unit * (t[None, :, None] * g_prev[:, None, :]
                   + f_int[None, :, None] * delta[:, None, :])
    logger.debug(" - Integral ramp trajectory: %s", traj.shape)
    return traj


def calc_radial_grad3d(
    matrix_size: int,
    npro_target: int,
    half_sphere: bool,
    use_origin: bool,
    reorder: bool,
) -> np.ndarray:
    """
    Generate 3D radial gradient profile based on input parameters.

    Args:
        matrix_size (int): Target matrix size.
        n_pro_target (int): Target number of projections.
        half_sphere (bool): If True, only generate for half the sphere.
        use_origin (bool): If True, add center points at the start.
        reorder (bool): Use reorder scheme provided by Bruker ZTE sequence.

    Returns:
        ndarray: The gradient profile as an array.
    """

    n_pro = int(npro_target / (1 if half_sphere else 2) - (1 if use_origin else 0))
    usamp = np.sqrt(find_undersamp(matrix_size, n_pro))

    logger.debug('\n++ Processing SORDINO 3D Radial Gradient Calculation...')
    logger.debug(' + Input arguments')
    logger.debug(f' - Matrix size: {matrix_size}')
    logger.debug(f' - Undersampling factor: {usamp}')
    logger.debug(f' - Number of Projections: {npro_target}')
    logger.debug(f' - Half sphere only: {half_sphere}')
    logger.debug(f' - Use origin: {use_origin}')
    logger.debug(f' - Reorder Gradient: {reorder}')

    grad = {"r": [], "p": [], "s": []}
    radial_n_phi: list[int] = []

    logger.debug(' + Start Calculating Gradient Vectors...')
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
    logger.debug('done')

    grad_array = np.stack([grad["r"], grad["p"], grad["s"]], axis=0)
    n_pro_created = grad_array.shape[-1] * (1 if half_sphere else 2) + (1 if use_origin else 0)
    if not usamp:
        if n_pro_created != npro_target:
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
    """
    Reorder radial projections for improved image spoiling.

    Args:
        n_theta (int): Number of theta angles.
        radial_n_phi (list): Number of phi angles for each theta.
        grad_array (ndarray): Gradient array.
        reorder (bool): Whether to apply the reordering scheme.

    Returns:
        ndarray: Reordered gradient array.
    """
    g = grad_array.copy()
    if reorder:
        logger.debug(' + Reordering projections...')
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
        logger.debug('done')
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


#: Version of the trajectory formulas in the cache key (WI-0071). Increase it
#: whenever a function that shapes the array changes its result:
#: calc_radial_grad3d, find_undersamp (including the brentq tolerance),
#: calc_npro, radial_angles, radial_angle, reorder_projections,
#: calc_radial_traj3d or calc_radial_traj3d_integral, so that trajectories saved
#: by older code are not reused. tests/test_traj_cache_key.py holds a golden
#: value that fails when the formulas change.
TRAJ_CACHE_VERSION = 2


def trajectory_cache_key(grad_params: Dict[str, Any], n_samples: int,
                         model: Dict[str, Any]) -> str:
    """Hash of the values that generate a trajectory, and nothing else (WI-0071).

    ``grad_params`` are the arguments of ``calc_radial_grad3d``, ``n_samples``
    the samples per spoke and ``model`` the per-sample inputs of the chosen
    formula (fixed vector: the offset in samples; integral: sample times, ramp
    integrals and dwell). Values are written as named JSON fields (floats in
    their exact repr), so different values cannot give the same text, and
    options that do not change the trajectory (``ext_factors``, frames,
    channels, the phase-only tuning ``phase_ref_us`` ...) are not part of it.
    """
    payload = {
        "version": TRAJ_CACHE_VERSION,
        "grad": grad_params,
        "n_samples": int(n_samples),
        "model": model,
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_cached_trajectory(path: Path, expected_shape: tuple) -> Optional[np.ndarray]:
    """The saved trajectory, or None when it is missing, unreadable or of another shape."""
    if not path.exists():
        return None
    try:
        traj = np.load(path, allow_pickle=False)
    except Exception as exc:  # damaged or partly written file: compute again
        logger.warning("Trajectory cache %s is unreadable (%s); computing it again.", path.name, exc)
        return None
    if traj.shape != expected_shape or traj.dtype != np.float64:
        logger.warning("Trajectory cache %s has shape %s and dtype %s, expected %s float64; "
                       "computing it again.", path.name, traj.shape, traj.dtype, expected_shape)
        return None
    return traj


def _save_trajectory(path: Path, traj: np.ndarray) -> None:
    """Write through a temporary file of this process and rename it, so a reader never
    sees half a file and two processes saving the same trajectory do not share one file."""
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".partial")
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, traj, allow_pickle=False)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise


def get_trajectory(recon_info: Dict[str, Any],
                   options: Options) -> np.ndarray:

    correct_ramptime = bool(getattr(options, "correct_ramptime", True))

    sample_size = int(recon_info['Matrix'][0])
    npro = int(recon_info['NPro'])
    half_acquisition = bool(recon_info['HalfAcquisition'])
    use_origin = bool(recon_info['UseOrigin'])
    reorder = bool(recon_info['Reorder'])

    eff_bandwidth = recon_info['EffBandwidth_Hz']
    over_sampling = recon_info['OverSampling']
    traj_offset = recon_info['AcqDelayTotal_us']
    n_samples = int(sample_size / 2 * over_sampling)

    grad = calc_radial_grad3d(sample_size,
                              npro,
                              half_acquisition,
                              use_origin,
                              reorder)
    grad_params = {"matrix_size": sample_size, "npro_target": npro,
                   "half_sphere": half_acquisition, "use_origin": use_origin,
                   "reorder": reorder}

    use_integral = correct_ramptime
    if use_integral:
        from . import timing as timing_mod

        seq = timing_mod.read_timing(recon_info)
        tune = timing_mod.tuning_for(seq.version)
        times_us, f_us, _ = timing_mod.ramp_terms(seq, tune, n_samples)
        logger.debug(" + Ramp model: integral, %s", timing_mod.describe(seq, tune))
        model = {"formula": "integral",
                 "times_us": [float(v) for v in times_us],
                 "ramp_integral_us": [float(v) for v in f_us],
                 "dwell_us": float(seq.dwell_us)}
        # BRK-0059/BRK-0060: the unsampled centre is a property of the
        # sequence and the user has nothing to do about it, so it is logged
        # (info for a general ZTE gap over 1 k-grid unit, debug otherwise),
        # never warned; the radius also goes into the result metadata (hook).
        gap = timing_mod.kspace_gap(seq, over_sampling,
                                    getattr(options, "ignore_samples", None) or 1)
        if seq.version == "zte" and gap["gap_kgrid"] > 1.0:
            logger.info(
                "General ZTE: the first sample is %.1f k-grid units from the k-space "
                "centre (dead time %.2f us); the centre is not filled.",
                gap["gap_kgrid"], seq.acq_delay_total_us)
        else:
            logger.debug(" + k-space centre gap: %s", gap)
    else:
        offset_factor = float(traj_offset * (10 ** -6) * eff_bandwidth * over_sampling)
        model = {"formula": "fixed", "traj_offset_samples": offset_factor}

    digest = trajectory_cache_key(grad_params, n_samples, model)
    traj_path = Path(options.cache_dir) / f"traj_{digest}.npy"
    expected_shape = (int(grad.shape[1]), n_samples, 3)
    traj = _load_cached_trajectory(traj_path, expected_shape)
    if traj is not None:
        logger.debug("Trajectory cache hit: %s", traj_path)
        return traj
    logger.info("Computing trajectory (matrix=%s, n_pro=%s).", sample_size, npro)
    if use_integral:
        traj = calc_radial_traj3d_integral(
            grad, sample_size, over_sampling, times_us, f_us, seq.dwell_us)
    else:
        traj = calc_radial_traj3d(grad, sample_size, over_sampling, offset_factor)
    _save_trajectory(traj_path, traj)
    logger.debug("Saved trajectory cache: %s", traj_path)
    return traj

__all__ = [
    'get_trajectory'
]