from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np
import hashlib
import logging
from .helper import progressbar
from .typing import Options

logger = logging.getLogger("brkraw.sordino")


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
    use_origin: bool,
    over_sampling: float,
    correct_ramptime: bool = False,
    traj_offset: Optional[float] = None,
) -> np.ndarray:
    """
    Calculate trajectory for SORDINO imaging.
    For each projection, a vector with sampling positions (trajectory) is created.

    Args:
        grad_array (ndarray): Gradient vector profile for each projection, sized (3 x n_pro).
        matrix_size (int): Matrix size of the final image.
        over_sampling (float): Oversampling factor.
        bandwidth (float): Bandwidth of the imaging process.
        traj_offset (float, optional): Trajectory offset ratio compared to ADC sampling.

    Returns:
        ndarray: Calculated trajectory for each projection.
    """
    pro_offset = 1 if use_origin else 0
    g = grad_array.copy()
    npro = g.shape[-1]
    traj_offset = traj_offset or 0
    num_samples = int(matrix_size / 2 * over_sampling)
    traj = np.zeros([npro, num_samples, 3])
    scale_factor = (num_samples - 1 + traj_offset) / (num_samples-1)

    logger.debug('++ Processing trajectory calculation...')
    logger.debug(' + Input arguments')
    logger.debug(f' - Size of Matrix: {matrix_size}')
    logger.debug(f' - OverSampling: {over_sampling}')
    logger.debug(f' - Trajectory offset ratio: {traj_offset}')
    logger.debug(f' - Ramp-time Correction: {str(correct_ramptime)}')
    logger.debug(f' - Size of Output Trajectory: {traj.shape}')
    logger.debug(f' - Image Scailing Factor (*Subject to be corrected in future version): {scale_factor}')

    for i_pro in progressbar(range(pro_offset, npro + pro_offset), desc="traj", ncols=100):
        for i_samp in range(num_samples):
            samp = ((i_samp + traj_offset) / (num_samples - 1)) / 2
            if not correct_ramptime or i_pro == (npro + pro_offset) - 1:
                correction = np.zeros(3)
                traj[i_pro, i_samp, :] = samp * (g[:, i_pro] + correction)
            else:
                correction = (g[:, i_pro] - g[:, i_pro - 1]) / num_samples * i_samp
                traj[i_pro, i_samp, :] = samp * (g[:, i_pro - 1] + correction)
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


def generate_hash(*args: Any) -> str:
    """Generate a hash from the input arguments."""
    hash_input = "".join(str(arg) for arg in args)
    return hashlib.md5(hash_input.encode()).hexdigest()


def get_trajectory(recon_info: Dict[str, Any], 
                   options: Options) -> np.ndarray:
    
    correct_ramptime = getattr(options, "correct_ramptime", True)
    ext_factors = getattr(options, "ext_factors", [1.0, 1.0, 1.0])
    logger.debug(f' + Extension factors applied to matrix: {ext_factors}')

    sample_size = recon_info['Matrix'][0]
    npro = recon_info['NPro']
    half_acquisition = recon_info['HalfAcquisition']
    use_origin = recon_info['UseOrigin']
    reorder = recon_info['Reorder']
    
    eff_bandwidth = recon_info['EffBandwidth_Hz']
    over_sampling = recon_info['OverSampling']
    traj_offset = recon_info['AcqDelayTotal_us']

    grad = calc_radial_grad3d(sample_size, 
                              npro, 
                              half_acquisition, 
                              use_origin, 
                              reorder)
    offset_factor = traj_offset * (10 ** -6) * eff_bandwidth * over_sampling

    option_for_hash = (
        float(traj_offset),
        sample_size,
        eff_bandwidth,
        over_sampling,
            int(npro),
            float(np.prod(ext_factors)),
            bool(half_acquisition),
            bool(use_origin),
            bool(reorder),
        correct_ramptime,
    )

    digest = generate_hash(*option_for_hash)
    traj_path = options.cache_dir / f"{digest}.npy"
    if traj_path.exists():
        logger.debug("Trajectory cache hit: %s", traj_path)
        traj = np.load(traj_path)
    else:
        logger.info("Computing trajectory (matrix=%s, n_pro=%s).", sample_size, npro)
        traj = calc_radial_traj3d(
            grad,
            sample_size,
            use_origin,
            over_sampling,
            correct_ramptime=correct_ramptime,
            traj_offset=offset_factor,
        )
        np.save(traj_path, traj)
        logger.debug("Saved trajectory cache: %s", traj_path)
    return traj

__all__ = [
    'get_trajectory'
]