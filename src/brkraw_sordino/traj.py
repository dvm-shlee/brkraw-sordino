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
    return _radial_grad3d(matrix_size, usamp, half_sphere, use_origin, reorder, npro_target)


def radial_grad3d_undersampling(matrix_size: int, under_sampling: float, half_sphere: bool,
                                use_origin: bool, reorder: bool) -> np.ndarray:
    """``radialGrad3D`` of the sequence with the method's ``ProUnderSampling`` itself (WI-0113).

    ``calc_radial_grad3d`` finds an undersampling that gives ``NPro`` spokes; the
    sequence's reco relation calls ``radialGrad3D(matrix, ProUnderSampling, ...)``
    directly, which is what Golden Grid scans of ``sordino_260801`` carry at the head
    of their lists (``golden.grid_head``). Same loop, reordering and mirroring.
    """
    return _radial_grad3d(int(matrix_size), np.sqrt(float(under_sampling)), bool(half_sphere),
                          bool(use_origin), bool(reorder), None)


def _radial_grad3d(matrix_size, usamp, half_sphere, use_origin, reorder, npro_target) -> np.ndarray:
    """The ``radialGrad3D`` loop for the square-root undersampling ``usamp``."""
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
    if not usamp and npro_target is not None:
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
#: value that fails when the formulas change. The golden lists (golden.py,
#: WI-0113) are new: their fields carry ``mode`` and never equal a Default
#: field set, so the version did not change and Default files keep their names;
#: increase it when a golden generator changes its result.
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


class TrajectoryRows:
    """Trajectory of any spoke range, computed on demand (WI-0097, D-0133).

    Holds only the per-spoke gradient vectors and the per-sample terms, so the
    whole (n_pro, N, 3) trajectory is never built and no trajectory cache is
    written. ``rows(lo, hi)`` equals ``get_trajectory(...)[lo:hi]`` bit for bit:
    it is the same expression as ``calc_radial_traj3d_integral`` (or
    ``calc_radial_traj3d`` with ``correct_ramptime=false``) on the selected
    spokes only. Spoke i needs g(i-1); ``g_prev`` is the rolled list, so a range
    that starts inside the list still sees the vector of the spoke before it.
    """

    def __init__(self, grad: np.ndarray, n_samples: int, formula: str, **terms: Any):
        g = np.asarray(grad, dtype=float)
        self.n_pro = int(g.shape[1])
        self.n_samples = int(n_samples)
        self.formula = formula
        if formula == "integral":
            n = self.n_samples
            self._unit = 1.0 / (n - 1) / 2.0
            self._t = np.asarray(terms["times_us"], dtype=float) / terms["dwell_us"]
            self._f = np.asarray(terms["ramp_integral_us"], dtype=float) / terms["dwell_us"]
            if len(self._t) != n or len(self._f) != n:
                raise ValueError("times and ramp integral must have one value per sample")
            self._g_prev = np.roll(g, 1, axis=1).T              # (n_pro, 3)
            self._delta = g.T - self._g_prev
        elif formula == "fixed":
            off = terms.get("traj_offset") or 0
            n = self.n_samples
            self._samp = ((np.arange(n, dtype=float) + off) / (n - 1)) / 2.0
            self._gT = g.T
        else:
            raise ValueError(f"unknown trajectory formula {formula!r}")

    @property
    def shape(self) -> tuple:
        return (self.n_pro, self.n_samples, 3)

    def rows(self, lo: int, hi: int) -> np.ndarray:
        """(hi - lo, N, 3) float64 trajectory of spokes lo .. hi - 1."""
        if self.formula == "integral":
            gp = self._g_prev[lo:hi]
            de = self._delta[lo:hi]
            return self._unit * (self._t[None, :, None] * gp[:, None, :]
                                 + self._f[None, :, None] * de[:, None, :])
        return self._samp[None, :, None] * self._gT[lo:hi, None, :]


def gradient_list(recon_info: Dict[str, Any]) -> tuple:
    """(3, NPro) spoke directions of the scan's ``TrajectoryMode`` and their cache-key fields.

    ``Default`` (also a method without ``TrajectoryMode``): ``calc_radial_grad3d``
    with the same fields as before WI-0113, so its trajectory files keep their
    names. ``GoldenSampling`` / ``GoldenGridSampling``: ``golden.golden_gradients``;
    the fields carry ``mode`` and the generating method values, so a golden list
    never shares a file with a Default one. Only the direction list depends on
    the mode; the ramp model and the phase correction are the same for all.
    """
    from . import golden

    mode = golden.trajectory_mode(recon_info)
    if mode != "Default":
        grad, params = golden.golden_gradients(recon_info)
        params = {k: v for k, v in params.items() if k != "n_cell"}
        return grad, params
    sample_size = int(recon_info['Matrix'][0])
    npro = int(recon_info['NPro'])
    half_acquisition = bool(recon_info['HalfAcquisition'])
    use_origin = bool(recon_info['UseOrigin'])
    reorder = bool(recon_info['Reorder'])
    grad = calc_radial_grad3d(sample_size, npro, half_acquisition, use_origin, reorder)
    grad_params = {"matrix_size": sample_size, "npro_target": npro,
                   "half_sphere": half_acquisition, "use_origin": use_origin,
                   "reorder": reorder}
    return grad, grad_params


def grid_head_length(recon_info: Dict[str, Any]) -> int:
    """Spokes at the head of every repetition played along the Default list (0: none).

    Set by ``hook._parse_recon_info`` (``golden.grid_head``) as
    ``recon_info["GoldenGridHead"]`` for Golden Grid scans; absent elsewhere.
    """
    head = recon_info.get("GoldenGridHead")
    if not head or not head.get("applied"):
        return 0
    return int(head["n"])


def grid_head_list(recon_info: Dict[str, Any]) -> np.ndarray:
    """The Default list the sequence's reco relation writes over a Golden Grid list (WI-0113)."""
    return radial_grad3d_undersampling(int(recon_info["Matrix"][0]), float(recon_info["UnderSampling"]),
                                       bool(recon_info["HalfAcquisition"]), bool(recon_info["UseOrigin"]),
                                       bool(recon_info["Reorder"]))


def _with_grid_head(recon_info: Dict[str, Any], grad: np.ndarray, params: Dict[str, Any]) -> tuple:
    n = grid_head_length(recon_info)
    if n == 0:
        return grad, params
    grad = np.array(grad, dtype=float, copy=True)
    grad[:, :n] = grid_head_list(recon_info)[:, :n]
    return grad, dict(params, grid_head=n)


def spoke_directions(recon_info: Dict[str, Any]) -> tuple:
    """(3, NPro) directions the spokes were played along, and their cache-key fields.

    ``gradient_list`` (the list the method parameters describe), except for the
    Golden Grid head of ``sordino_260801`` scans (``golden.grid_head``): there the
    first N entries are the Default list (WI-0113 run 5, D-0197 decision 1). Used
    for the trajectory and the K0 leading points; the ``ACQ_O1_list`` check and the
    receiver frequencies keep the method's list.
    """
    grad, params = gradient_list(recon_info)
    return _with_grid_head(recon_info, grad, params)


#: Relative residual above which the ACQ_O1_list order check warns (WI-0113).
#: The scanner list agrees with the computed one to ~1e-15 (WI-0112: 1.8e-15 to
#: 3.7e-15 on four scans); another order gives ~1.
O1_ORDER_TOL = 1e-6


def o1_order_residual(o1: np.ndarray, grad: np.ndarray) -> float:
    """Relative RMS residual of ``o1 ~ c + offR*gR + offP*gP + offS*gS`` (least squares).

    The sequence records ``ACQ_O1_list[i] = offR*GradR[i] + offP*GradP[i] +
    offS*GradS[i]`` (FOV offset times the spoke direction), so the computed list in
    the acquired order leaves only rounding. The three offsets are fitted, so the
    check confirms the order and the relative directions, not a fixed change of
    axes or signs (WI-0112).
    """
    o1 = np.asarray(o1, dtype=float).ravel()
    g = np.asarray(grad, dtype=float)
    a = np.concatenate([np.ones((g.shape[1], 1)), g.T], axis=1)
    coef, *_ = np.linalg.lstsq(a, o1, rcond=None)
    res = o1 - a @ coef
    spread = float(np.sqrt(np.mean((o1 - o1.mean()) ** 2)))
    if spread == 0.0:
        return float("nan")
    return float(np.sqrt(np.mean(res ** 2)) / spread)


def check_o1_order(recon_info: Dict[str, Any], grad: np.ndarray) -> Optional[float]:
    """Compare the computed spoke order with the scanner's ``ACQ_O1_list`` (WI-0113).

    Only possible when the list has one value per spoke and the FOV centre is
    off the isocentre (otherwise the list is one value, or constant). Logged at
    debug level; a golden mode whose residual is above ``O1_ORDER_TOL`` gives one
    warning, since its image would be wrong. For ``Default`` the residual is only
    logged (its list is the established one). Returns the residual or None.
    """
    from . import golden

    o1 = np.asarray(recon_info.get("O1List_Hz") or [], dtype=float)
    mode = golden.trajectory_mode(recon_info)
    if o1.size != grad.shape[1] or o1.size < 4 or float(np.ptp(o1)) == 0.0:
        logger.debug(" + ACQ_O1_list order check not possible (%s values for %s spokes, no FOV offset)",
                     o1.size, grad.shape[1])
        return None
    res = o1_order_residual(o1, grad)
    logger.debug(" + ACQ_O1_list order check (%s): relative residual %.3g", mode, res)
    if mode != "Default" and not res <= O1_ORDER_TOL:
        logger.warning(
            "sordino: the computed %s spoke order does not match the scanner's ACQ_O1_list\n"
            "  residual  %.3g (expected below %.0e)\n"
            "  the image is likely wrong; check TrajectoryMode and its method parameters",
            mode, res, O1_ORDER_TOL)
    return res


def _trajectory_inputs(recon_info: Dict[str, Any], options: Options) -> Dict[str, Any]:
    """Gradient list, cache-key fields and per-sample terms of the chosen formula."""
    correct_ramptime = bool(getattr(options, "correct_ramptime", True))

    sample_size = int(recon_info['Matrix'][0])
    npro = int(recon_info['NPro'])

    eff_bandwidth = recon_info['EffBandwidth_Hz']
    over_sampling = recon_info['OverSampling']
    traj_offset = recon_info['AcqDelayTotal_us']
    n_samples = int(sample_size / 2 * over_sampling)

    grad, grad_params = gradient_list(recon_info)
    check_o1_order(recon_info, grad)              # the recorded list follows the method's list
    grad, grad_params = _with_grid_head(recon_info, grad, grad_params)
    out: Dict[str, Any] = {"grad": grad, "grad_params": grad_params, "n_samples": n_samples,
                           "sample_size": sample_size, "npro": npro,
                           "over_sampling": over_sampling}
    if correct_ramptime:
        from . import timing as timing_mod

        seq = timing_mod.read_timing(recon_info)
        tune = timing_mod.tuning_for(seq.version)
        times_us, f_us, _ = timing_mod.ramp_terms(seq, tune, n_samples)
        logger.debug(" + Ramp model: integral, %s", timing_mod.describe(seq, tune))
        out["model"] = {"formula": "integral",
                        "times_us": [float(v) for v in times_us],
                        "ramp_integral_us": [float(v) for v in f_us],
                        "dwell_us": float(seq.dwell_us)}
        out["terms"] = {"times_us": times_us, "ramp_integral_us": f_us, "dwell_us": seq.dwell_us}
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
        out["model"] = {"formula": "fixed", "traj_offset_samples": offset_factor}
        out["terms"] = {"traj_offset": offset_factor}
    return out


def trajectory_rows(recon_info: Dict[str, Any], options: Options) -> TrajectoryRows:
    """The trajectory of ``get_trajectory`` as a ``TrajectoryRows`` (nothing saved)."""
    inp = _trajectory_inputs(recon_info, options)
    rows = TrajectoryRows(inp["grad"], inp["n_samples"], inp["model"]["formula"], **inp["terms"])
    logger.debug(" + Trajectory rows (%s formula), %s spokes x %s samples",
                 rows.formula, rows.n_pro, rows.n_samples)
    return rows


def get_trajectory(recon_info: Dict[str, Any],
                   options: Options) -> np.ndarray:
    """Whole (n_pro, N, 3) trajectory, saved in the trajectory cache.

    The reconstruction uses ``trajectory_rows`` (WI-0097); this function stays
    for the tools and as the reference the rows are tested against.
    """
    inp = _trajectory_inputs(recon_info, options)
    grad = inp["grad"]
    sample_size = inp["sample_size"]
    npro = inp["npro"]
    over_sampling = inp["over_sampling"]
    n_samples = inp["n_samples"]
    model = inp["model"]
    use_integral = model["formula"] == "integral"

    digest = trajectory_cache_key(inp["grad_params"], n_samples, model)
    traj_path = Path(options.cache_dir) / f"traj_{digest}.npy"
    expected_shape = (int(grad.shape[1]), n_samples, 3)
    traj = _load_cached_trajectory(traj_path, expected_shape)
    if traj is not None:
        logger.debug("Trajectory cache hit: %s", traj_path)
        return traj
    logger.info("Computing trajectory (matrix=%s, n_pro=%s).", sample_size, npro)
    if use_integral:
        t = inp["terms"]
        traj = calc_radial_traj3d_integral(
            grad, sample_size, over_sampling, t["times_us"], t["ramp_integral_us"], t["dwell_us"])
    else:
        traj = calc_radial_traj3d(grad, sample_size, over_sampling, inp["terms"]["traj_offset"])
    _save_trajectory(traj_path, traj)
    logger.debug("Saved trajectory cache: %s", traj_path)
    return traj


__all__ = [
    'get_trajectory',
    'trajectory_rows',
    'TrajectoryRows',
    'gradient_list',
    'spoke_directions',
    'grid_head_length',
    'grid_head_list',
    'radial_grad3d_undersampling',
    'o1_order_residual',
    'check_o1_order',
]