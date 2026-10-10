"""Golden-angle spoke directions of the SORDINO sequence (WI-0113).

The sequence (``sordino_260801`` and later) chooses the spoke directions with
the method parameter ``TrajectoryMode``:

* ``Default``: the radial list of ``radialGrad3D`` (``traj.calc_radial_grad3d``).
* ``GoldenSampling`` (``goldensamp.c``): the two-dimensional golden means
  (n * 0.4656 for the azimuth, n * 0.6823 for z) cut into subsets of
  ``NGoldenSpokesPerSubset`` consecutive spokes; inside each subset the spokes
  are reordered in z-stacks of ``ZStackAngleDeg`` (north to south in even
  subsets, south to north in odd ones, by azimuth inside a stack, the stack
  edges twisted with the azimuth) to keep the step between spokes short.
  ``GoldenReorder=No`` keeps the plain golden order.
* ``GoldenGridSampling`` (``goldengrid.c``, "Sreag" grid): ``NGridRing``
  latitude rings of equal-area cells; frame m puts one spoke in every cell at a
  golden offset of the cell (plus the opposite spoke with ``GridMirror``).

The functions here are vectorised copies of those C functions; they give the
scanner's lists to floating-point rounding (tests/golden_cref.py is the
statement-by-statement copy they are tested against, and the scanner's
``ACQ_O1_list`` and online-reconstruction ``traj`` file agree to ~1e-15,
WI-0112). The arrays are (3, n) in the scanner's (R, P, S) gradient axes, like
``calc_radial_grad3d``.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Values of ``TrajectoryMode`` this package reconstructs.
MODES = ("Default", "GoldenSampling", "GoldenGridSampling")
GOLDEN_MODES = MODES[1:]

# goldensamp.c / goldengrid.c constants (two-dimensional golden means and the
# per-cell phase offsets of the grid)
PHI1 = 0.465571231876768
PHI2 = 0.6823278038280193
GOLDEN_CONJ = 0.61803398875
SILVER_CONJ = 0.41421356237


# ---------------------------------------------------------------- Golden Sampling
def golden_samples(n_spokes: int) -> Tuple[np.ndarray, np.ndarray]:
    """``GenerateGoldenSamples``: (3, n) unit directions and the azimuth (radians) of each."""
    n = np.arange(int(n_spokes), dtype=float)
    frac1 = np.fmod(n * PHI1, 1.0)
    frac2 = np.fmod(n * PHI2, 1.0)
    z = 1.0 - 2.0 * frac2
    r = np.sqrt(1.0 - z * z)
    theta = (2.0 * math.pi) * frac1
    return np.stack([r * np.cos(theta), r * np.sin(theta), z]), theta


def _zstack_edges(zstack_angle_deg: float) -> Tuple[float, int, np.ndarray, np.ndarray]:
    """Stack angle (rad), stacks per half turn and the lower/upper edges, as in C."""
    angle = float(zstack_angle_deg) * math.pi / 180.0
    if not angle > 0:
        raise ValueError(f"ZStackAngleDeg must be positive, got {zstack_angle_deg!r}")
    n_half = int(math.floor(math.pi / angle))
    if n_half < 1:
        raise ValueError(f"ZStackAngleDeg {zstack_angle_deg!r} is wider than 180 degrees: "
                         "no z-stack holds a spoke")
    lower = np.empty(2 * n_half)
    upper = np.empty(2 * n_half)
    for i in range(1, n_half + 1):
        lower[i - 1] = math.cos(angle * i)
    upper[0] = 1.0
    for i in range(1, n_half):
        upper[i] = lower[i - 1]
    for i in range(n_half):
        lower[n_half + i] = lower[n_half - 1 - i]
        upper[n_half + i] = upper[n_half - 1 - i]
    return angle, n_half, lower, upper


def _bucket_order(n_subsets: int, n_per: int, zstack_angle_deg: float):
    """Golden samples, their azimuths and each spoke's z-stack visit position (-1: none)."""
    angle, n_half, lower, upper = _zstack_edges(zstack_angle_deg)
    g, theta = golden_samples(n_subsets * n_per)
    subset = np.repeat(np.arange(n_subsets), n_per)
    even = (subset % 2) == 0
    shift = (1.0 - theta / (2.0 * math.pi)) * angle
    polar = np.arccos(g[2])
    shifted = np.where(even, np.cos(polar + shift), np.cos(polar - shift))
    visit = np.full(shifted.shape, -1, dtype=np.int64)
    for j in range(n_half):
        for is_even, bucket in ((True, j), (False, n_half + j)):
            m = (even == is_even) & (shifted > lower[bucket]) & (shifted <= upper[bucket])
            visit[m] = j
    return g, theta, subset, visit


def reorder_golden_samples(n_subsets: int, n_spokes_per_subset: int, zstack_angle_deg: float,
                           use_origin: bool = False, golden_reorder: bool = True,
                           n_pro: Optional[int] = None) -> np.ndarray:
    """``ReorderGoldenSamples`` into a zero-filled list of ``n_pro`` spokes, (3, n_pro).

    ``n_pro`` defaults to ``n_subsets * n_spokes_per_subset`` (``NGoldenSteps``
    with ``GoldenReorder=Yes``). Spokes that fall in no z-stack (a stack angle
    that does not divide 180 degrees) are left out; their places at the end of
    the subset keep the zero vector, as on the scanner. With
    ``golden_reorder=False`` the plain golden order fills the first ``n_pro``
    places (``NPro = NGoldenSteps``, a user value then). ``use_origin``
    overwrites spoke 0 with the zero vector (the count is unchanged).
    """
    n_subsets = int(n_subsets)
    per = int(n_spokes_per_subset)
    n_spokes = n_subsets * per
    n_pro = n_spokes if n_pro is None else int(n_pro)
    out = np.zeros((3, n_pro))
    if golden_reorder:
        if n_pro != n_spokes:
            raise ValueError(f"GoldenReorder=Yes gives {n_spokes} spokes "
                             f"({n_subsets} x {per}), not {n_pro}")
        g, theta, subset, visit = _bucket_order(n_subsets, per, zstack_angle_deg)
        placed = visit >= 0
        # within a subset: z-stack visit order, then azimuth (qsort compare_theta)
        order = np.lexsort((theta, np.where(placed, visit, np.iinfo(np.int64).max), subset))
        g = g[:, order]
        placed = placed[order]
        g[:, ~placed] = 0.0                       # left out: zero at the end of the subset
        out[:, :] = g
    else:
        g, _theta = golden_samples(n_spokes)
        k = min(n_spokes, n_pro)
        out[:, :k] = g[:, :k]
    if use_origin and n_pro:
        out[:, 0] = 0.0
    return out


def unplaced_spokes(n_subsets: int, n_spokes_per_subset: int, zstack_angle_deg: float) -> int:
    """Number of golden spokes the z-stack reordering leaves out (0 when 180/angle is whole)."""
    *_, visit = _bucket_order(int(n_subsets), int(n_spokes_per_subset), zstack_angle_deg)
    return int((visit < 0).sum())


# ---------------------------------------------------------------- Golden Grid
@dataclass(frozen=True)
class SreagGrid:
    """``SreagGrid``: rings of equal-area cells (edges in degrees, north to south)."""

    n_ring: int
    n_cell: int
    lat_edges: Tuple[float, ...]
    n_lon: Tuple[int, ...]
    lon_edges: Tuple[Tuple[float, ...], ...]


def sreag_grid(n_ring: int) -> SreagGrid:
    """``GenerateSreagGrid`` (scalar C arithmetic; the grid is small)."""
    n_ring = int(n_ring)
    if n_ring < 1:
        raise ValueError(f"NGridRing must be at least 1, got {n_ring}")
    d_b = 180.0 / n_ring
    beta0 = []
    for i in range(n_ring):
        t = 0.0 if n_ring == 1 else float(i) / (n_ring - 1)
        beta0.append((90.0 - d_b / 2.0) + t * ((-90.0 + d_b / 2.0) - (90.0 - d_b / 2.0)))
    d_l, n_lon = [], []
    for i in range(n_ring):
        span = d_b / math.cos(beta0[i] * math.pi / 180.0)
        count = int(math.floor(360.0 / span + 0.5))
        n_lon.append(count)
        d_l.append(360.0 / count)
    n_cell = sum(n_lon)
    area = 4.0 * math.pi / n_cell
    lat = [math.pi / 2.0]
    for i in range(n_ring):
        sin_bl = math.sin(lat[i]) - area / (d_l[i] * math.pi / 180.0)
        sin_bl = min(1.0, max(-1.0, sin_bl))
        lat.append(math.asin(sin_bl))
    lat = [v * (180.0 / math.pi) for v in lat]
    lon = tuple(tuple(360.0 * float(j) / n for j in range(n + 1)) for n in n_lon)
    return SreagGrid(n_ring, n_cell, tuple(lat), tuple(n_lon), lon)


def _cell_tables(grid: SreagGrid):
    """Per cell: z_u, Z_cs, alpha0, theta_cs and the 1-based cell id (C order)."""
    z_u, z_cs, alpha0, theta_cs = [], [], [], []
    for i in range(grid.n_ring):
        zu = math.sin(grid.lat_edges[i] * math.pi / 180.0)
        zl = math.sin(grid.lat_edges[i + 1] * math.pi / 180.0)
        lon = grid.lon_edges[i]
        for j in range(grid.n_lon[i]):
            z_u.append(zu)
            z_cs.append(math.fabs(zu - zl))
            alpha0.append(lon[j])
            theta_cs.append(lon[j + 1] - lon[j])
    cell_id = np.arange(1, grid.n_cell + 1, dtype=float)
    return (np.asarray(z_u), np.asarray(z_cs), np.asarray(alpha0), np.asarray(theta_cs),
            np.fmod(cell_id * GOLDEN_CONJ, 1.0), np.fmod(cell_id * SILVER_CONJ, 1.0))


def _cells(tables, m: np.ndarray) -> np.ndarray:
    """``UpdateCellsGoldenPercell`` for frames ``m`` (1-based, shape (F, 1)): (3, F, n_cell)."""
    z_u, z_cs, alpha0, theta_cs, phi, psi = tables
    alpha_m = alpha0 + theta_cs * np.fmod(m * PHI2 + phi, 1.0)
    z_m = np.clip(z_u - z_cs * np.fmod(m * PHI1 + psi, 1.0), -1.0, 1.0)
    r = np.sqrt(1.0 - z_m * z_m)
    alpha_rad = alpha_m * math.pi / 180.0
    return np.stack([r * np.cos(alpha_rad), r * np.sin(alpha_rad), z_m])


def sreag_cells(grid: SreagGrid, m: int) -> np.ndarray:
    """One spoke per cell for frame ``m`` (1-based, as in C): (3, n_cell)."""
    return _cells(_cell_tables(grid), np.asarray([[float(int(m))]]))[:, 0, :]


def sreag_trajectory(n_ring: int, n_frames: int, mirror: bool) -> np.ndarray:
    """``GenerateSreagTrajectory``: frames 1..n_frames, each cells then (mirror) their opposites."""
    grid = sreag_grid(n_ring)
    n_frames = int(n_frames)
    if n_frames < 1:
        raise ValueError(f"NGridFrames must be at least 1, got {n_frames}")
    k = _cells(_cell_tables(grid), np.arange(1, n_frames + 1, dtype=float)[:, None])
    if mirror:
        k = np.concatenate([k, -k], axis=2)          # (3, F, 2 n_cell): cells, then opposites
    return np.ascontiguousarray(k.reshape(3, -1))


# ---------------------------------------------------------------- method parameters
def trajectory_mode(recon_info: Mapping[str, Any]) -> str:
    """``TrajectoryMode`` of the scan; ``Default`` when the method has no such key."""
    value = recon_info.get("TrajectoryMode")
    if value is None:
        return "Default"
    text = str(value).strip()
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1].strip()
    if text == "":
        return "Default"
    if text not in MODES:
        raise ValueError(f"sordino: unknown TrajectoryMode {text!r}; this version reconstructs "
                         f"{', '.join(MODES)}")
    return text


def _need(recon_info: Mapping[str, Any], key: str, mode: str):
    value = recon_info.get(key)
    if value is None:
        raise ValueError(f"sordino: TrajectoryMode {mode} needs the method parameter {key}, "
                         "which this scan does not have")
    return value


def _flag(recon_info: Mapping[str, Any], key: str, mode: str) -> bool:
    value = _need(recon_info, key, mode)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("yes", "true", "on", "1"):
            return True
        if text in ("no", "false", "off", "0"):
            return False
        raise ValueError(f"sordino: {key} must be Yes or No, got {value!r}")
    return bool(value)


def golden_parameters(recon_info: Mapping[str, Any]) -> Dict[str, Any]:
    """The method values that generate the golden list (also its trajectory-cache fields)."""
    mode = trajectory_mode(recon_info)
    use_origin = bool(recon_info.get("UseOrigin") or False)
    if mode == "GoldenSampling":
        reorder = _flag(recon_info, "GoldenReorder", mode)
        params: Dict[str, Any] = {
            "mode": mode,
            "n_subsets": int(_need(recon_info, "NGoldenSubsets", mode)),
            "n_spokes_per_subset": int(_need(recon_info, "NGoldenSpokesPerSubset", mode)),
            # the z-stacks are used only by the reordering
            "zstack_deg": (float(_need(recon_info, "ZStackAngleDeg", mode)) if reorder
                           else None),
            "golden_reorder": reorder,
            "use_origin": use_origin,
        }
        params["n_pro"] = (params["n_subsets"] * params["n_spokes_per_subset"] if reorder
                           else int(_need(recon_info, "NGoldenSteps", mode)))
        return params
    if mode == "GoldenGridSampling":
        n_ring = int(_need(recon_info, "NGridRing", mode))
        n_frames = int(_need(recon_info, "NGridFrames", mode))
        mirror = _flag(recon_info, "GridMirror", mode)
        n_cell = sreag_grid(n_ring).n_cell
        return {"mode": mode, "n_ring": n_ring, "n_frames": n_frames, "mirror": mirror,
                "n_cell": n_cell, "n_pro": n_frames * n_cell * (2 if mirror else 1)}
    raise ValueError(f"sordino: {mode} is not a golden trajectory mode")


def golden_gradients(recon_info: Mapping[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """(3, NPro) golden directions and their parameters; stops when the count is not NPro."""
    params = golden_parameters(recon_info)
    npro = recon_info.get("NPro")
    if npro is None or int(npro) != params["n_pro"]:
        raise ValueError(
            f"sordino: TrajectoryMode {params['mode']} gives {params['n_pro']} spokes per "
            f"repetition from the method parameters, but NPro is {npro}; the trajectory "
            "cannot be matched to the FID")
    if params["mode"] == "GoldenSampling":
        grad = reorder_golden_samples(params["n_subsets"], params["n_spokes_per_subset"],
                                      params["zstack_deg"], use_origin=params["use_origin"],
                                      golden_reorder=params["golden_reorder"], n_pro=params["n_pro"])
    else:
        grad = sreag_trajectory(params["n_ring"], params["n_frames"], params["mirror"])
    return grad, params


def golden_unit(recon_info: Mapping[str, Any]) -> Optional[int]:
    """Spokes in one method subset (the frame unit), or None for the Default trajectory.

    GoldenSampling: ``NGoldenSpokesPerSubset`` (1 with ``GoldenReorder=No``,
    where any spoke count is a contiguous golden segment); GoldenGridSampling:
    one grid frame, the cells (times 2 with ``GridMirror``).
    """
    mode = trajectory_mode(recon_info)
    if mode == "Default":
        return None
    params = golden_parameters(recon_info)
    if mode == "GoldenSampling":
        return params["n_spokes_per_subset"] if params["golden_reorder"] else 1
    return params["n_cell"] * (2 if params["mirror"] else 1)


# ---------------------------------------------------------------- Golden Grid head (WI-0113 run 5)
#: Sequence versions whose Golden Grid scans carry the Default head (RecoRelations.c of
#: sordino_260801: ``radialGrad3D`` fills GradR/P/S whenever GoldenSampTraj is No). Used only
#: when the scan has no ``traj`` file to show which list the reconstruction relation wrote.
GRID_HEAD_METHODS = ("sordino_260801",)
#: Largest difference between ``traj``-file directions and a list that still counts as equal
#: (measured on scans 17 and 21: 1.6e-15 against the Default list, 1.9e-15 against the golden one).
GRID_HEAD_TOL = 1e-9


def default_head_count(matrix: int, under_sampling: float, half_sphere: bool, use_origin: bool) -> int:
    """Number of entries ``radialGrad3D(matrix, ProUnderSampling, ...)`` writes (radial.c)."""
    from .traj import calc_npro

    n = calc_npro(int(matrix), float(under_sampling))          # one hemisphere, the C loop
    return n * (1 if half_sphere else 2) + (1 if use_origin else 0)


def method_name(value: Any) -> Optional[str]:
    """``<User:sordino_260801>`` -> ``sordino_260801``; None when missing."""
    if value is None:
        return None
    text = str(value).strip()
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1].strip()
    if ":" in text:
        text = text.split(":", 1)[1].strip()
    return text or None


def grid_head(recon_info: Mapping[str, Any], *, method: Any = None, golden_samp_traj: Optional[bool] = None,
              traj_dirs: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
    """Whether a Golden Grid scan was played with the Default list at its head (D-0197 decision 1).

    The sequence ``sordino_260801`` fills the gradient lists in its reconstruction
    relation with ``radialGrad3D(matrix, ProUnderSampling, ...)`` whenever
    ``GoldenSampTraj`` is No, which is the case for Golden Grid; the first N entries
    (``default_head_count``) then hold the Default list. The scanner's ``traj`` file
    is written from the same lists, so its first lines (``traj_dirs``, (3, m)) tell
    which list it was: the Default one turns the rule on, the golden one (a fixed
    sequence) off; with no ``traj`` file only the known version ``GRID_HEAD_METHODS``
    gets the rule. That the scanner played these directions is inferred from the
    data (WI-0113 run 4), not documented ParaVision behaviour.

    Returns None for other modes, else a dict with ``applied``, ``n`` (0 when not
    applied), ``source`` (``"traj file"``, ``"sequence version"`` or ``"GoldenSampTraj"``),
    ``method`` and, with a ``traj`` file, ``traj_match`` and the two differences.
    Warns once when the rule is applied.
    """
    if trajectory_mode(recon_info) != "GoldenGridSampling":
        return None
    name = method_name(method)
    out: Dict[str, Any] = {"applied": False, "n": 0, "source": None, "method": name}
    if golden_samp_traj:
        out["source"] = "GoldenSampTraj"
        return out
    npro = int(recon_info["NPro"])
    n = min(default_head_count(int(recon_info["Matrix"][0]), float(recon_info["UnderSampling"]),
                               bool(recon_info["HalfAcquisition"]), bool(recon_info["UseOrigin"])), npro)
    if traj_dirs is not None:
        from .traj import grid_head_list
        from .trajfile import head_match

        dirs = np.asarray(traj_dirs, dtype=float)
        m = min(int(dirs.shape[1]), n)
        golden_list, _ = golden_gradients(recon_info)
        match, d_default, d_golden = head_match(dirs[:, :m], grid_head_list(recon_info)[:, :m],
                                                golden_list[:, :m], GRID_HEAD_TOL)
        out.update(source="traj file", traj_match=match, traj_lines=m,
                   traj_vs_default=d_default, traj_vs_golden=d_golden)
        applied = match == "default"
        if match == "neither":
            logger.warning(
                "sordino: Golden Grid scan: the first %s lines of the traj file match neither the Default nor the "
                "golden list (largest differences %.3g and %.3g); the golden list is used for every spoke.",
                m, d_default, d_golden)
    else:
        out["source"] = "sequence version"
        applied = name in GRID_HEAD_METHODS
    if applied:
        out.update(applied=True, n=n)
        logger.warning(
            "sordino: Golden Grid scan of %s: the first %s spokes of every repetition are reconstructed along the "
            "Default list (ProUnderSampling %s) with their receiver-frequency difference.\n"
            "  The sequence's reconstruction relation wrote that list over the golden one (%s); the data fit "
            "these directions (WI-0113).",
            name or "an unnamed sequence", n, recon_info["UnderSampling"],
            "the traj file shows it" if out["source"] == "traj file"
            else "known for this version; no traj file to check")
    return out


__all__ = [
    "MODES", "GOLDEN_MODES", "golden_samples", "reorder_golden_samples", "unplaced_spokes",
    "SreagGrid", "sreag_grid", "sreag_cells", "sreag_trajectory", "trajectory_mode",
    "golden_parameters", "golden_gradients", "golden_unit",
    "GRID_HEAD_METHODS", "GRID_HEAD_TOL", "default_head_count", "method_name", "grid_head",
]
