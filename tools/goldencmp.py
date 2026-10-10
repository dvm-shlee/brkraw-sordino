"""Image comparison metrics for the golden-trajectory success criteria (WI-0113)."""

import numpy as np
from scipy import ndimage
from typing import Tuple, Any


def otsu_threshold(values: Any, nbins: int = 256) -> float:
    """Compute the Otsu threshold for a given set of values."""
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        raise ValueError("empty")
    v_min, v_max = v.min(), v.max()
    if v_min == v_max:
        raise ValueError("constant")
    counts, edges = np.histogram(v, bins=nbins, range=(v_min, v_max))
    centres = (edges[:-1] + edges[1:]) / 2.0
    best_k = -1
    max_score = -1.0
    for k in range(1, nbins):
        w0 = counts[:k].sum()
        w1 = counts[k:].sum()
        if w0 == 0 or w1 == 0:
            continue
        mu0 = (counts[:k] * centres[:k]).sum() / w0
        mu1 = (counts[k:] * centres[k:]).sum() / w1
        score = float(w0) * float(w1) * (mu0 - mu1) ** 2
        if score > max_score:
            max_score = score
            best_k = k
    if best_k == -1:
        raise ValueError("threshold")
    return float(edges[best_k])


def largest_component(mask: Any) -> np.ndarray:
    """Return the largest 26-connected component of a boolean mask."""
    m = np.asarray(mask, dtype=bool)
    structure = np.ones((3,) * m.ndim, dtype=bool)
    labels, n = ndimage.label(m, structure=structure)
    if n == 0:
        return np.zeros(m.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    keep = int(np.argmax(sizes))
    return labels == keep


def object_mask(image: Any) -> np.ndarray:
    """Create an object mask using Otsu thresholding and the largest component."""
    mag = np.abs(np.asarray(image))
    t = otsu_threshold(mag)
    return largest_component(mag > t)


def dice(a: Any, b: Any) -> float:
    """Compute the Dice overlap coefficient between two boolean masks."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("shape")
    total = int(a.sum()) + int(b.sum())
    if total == 0:
        raise ValueError("empty")
    return float(2.0 * np.logical_and(a, b).sum() / total)


def centroid(mask: Any) -> np.ndarray:
    """Compute the centroid of a boolean mask."""
    m = np.asarray(mask, dtype=bool)
    if m.sum() == 0:
        raise ValueError("empty")
    idx = np.argwhere(m)
    return idx.mean(axis=0).astype(float)


def scaled_nrmse(test: Any, ref: Any, mask: Any) -> Tuple[float, float]:
    """Compute the scaled normalized RMS error and the scaling factor."""
    t = np.abs(np.asarray(test))
    r = np.abs(np.asarray(ref))
    m = np.asarray(mask, dtype=bool)
    if t.shape != r.shape or t.shape != m.shape:
        raise ValueError("shape")
    if m.sum() == 0:
        raise ValueError("empty")
    tv = t[m].astype(float)
    rv = r[m].astype(float)
    den = float((tv * tv).sum())
    if den == 0.0:
        raise ValueError("zero")
    a = float((tv * rv).sum()) / den
    nrmse = float(np.sqrt(np.mean((a * tv - rv) ** 2)) / np.mean(rv))
    return (nrmse, a)


# ---------------------------------------------------------------- added by the parent (Park)
def edge_sides(image: Any, mask: Any, axis: int, low: float = 0.1, high: float = 0.9) -> dict:
    """Boundary width (voxels, 90 % to 10 % of the plateau) on each side of the central profile.

    The profile runs through the rounded centroid of ``mask`` along ``axis``; its
    plateau is the median magnitude at the profile points inside ``mask``. From
    the centre outwards, the first fall below ``high`` and below ``low`` times the
    plateau are located by linear interpolation. Returns ``{"+": side, "-": side}``
    with ``width`` (None when an edge is not inside the image) and
    ``lowest_fraction`` (lowest profile value from the centre to the image end on
    that side, as a fraction of the plateau). The same mask gives the same line for
    every image compared (success criterion 4, WI-0113).
    """
    mag = np.abs(np.asarray(image))
    m = np.asarray(mask, dtype=bool)
    if mag.shape != m.shape:
        raise ValueError("shape of image and mask differ")
    c = np.rint(centroid(m)).astype(int)
    index = list(c)
    index[axis] = slice(None)
    prof = mag[tuple(index)].astype(float)
    inside = m[tuple(index)]
    if not inside.any():
        raise ValueError("the mask does not cross the central line (empty)")
    plateau = float(np.median(prof[inside]))
    out = {}
    for name, step in (("+", 1), ("-", -1)):
        crossings = []
        for level in (high * plateau, low * plateau):
            x = int(c[axis])
            found = None
            while 0 <= x + step < prof.size:
                nxt = x + step
                if prof[nxt] < level:
                    p0, p1 = prof[x], prof[nxt]
                    frac = 0.0 if p0 == p1 else (p0 - level) / (p0 - p1)
                    found = x + step * frac
                    break
                x = nxt
            crossings.append(found)
        side = prof[int(c[axis]):] if step == 1 else prof[:int(c[axis]) + 1]
        width = None if None in crossings else abs(crossings[1] - crossings[0])
        out[name] = {"width": width, "lowest_fraction": float(side.min() / plateau),
                     "plateau": plateau, "centre": [int(v) for v in c]}
    return out


def edge_width(image: Any, mask: Any, axis: int, low: float = 0.1, high: float = 0.9) -> float:
    """Mean of the two ``edge_sides`` widths along ``axis``; ValueError when one is not inside the image."""
    sides = edge_sides(image, mask, axis, low, high)
    for name, side in sides.items():
        if side["width"] is None:
            raise ValueError(f"no edge on the {name} side of axis {axis} inside the image "
                             f"(lowest {side['lowest_fraction']:.3g} of the plateau)")
    return float(np.mean([sides["+"]["width"], sides["-"]["width"]]))


def compare(test: Any, ref: Any) -> dict:
    """Criteria numbers of ``test`` against ``ref``: Dice, centroid shift, NRMSE and scale.

    Masks are ``object_mask`` of each image; the NRMSE is in the reference mask
    after the least-squares scale (``scaled_nrmse``).
    """
    mt = object_mask(test)
    mr = object_mask(ref)
    nrmse, scale = scaled_nrmse(test, ref, mr)
    return {
        "dice": dice(mt, mr),
        "centroid_shift_vox": float(np.linalg.norm(centroid(mt) - centroid(mr))),
        "nrmse": nrmse,
        "scale": scale,
        "mask_voxels_test": int(mt.sum()),
        "mask_voxels_ref": int(mr.sum()),
    }
