import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any, cast
import logging

logger = logging.getLogger("brkraw.sordino")


PLANE_PERM = {
    "axial": (0, 1, 2),
    "coronal": (0, 2, 1),
    "sagittal": (1, 2, 0),
}


def apply_axis_perm_flip(dataobj: NDArray, R: NDArray) -> NDArray[Any]:
    """Apply axis permutation based on a rotation/gradient matrix.

    Args:
        dataobj (NDArray): Input data array with spatial axes first.
        R (NDArray): Orientation matrix used to derive axis permutation.

    Returns:
        NDArray[Any]: Transposed data array with permuted axes.
    """
    perm = np.argmax(np.abs(R), axis=1)
    data_t = np.transpose(dataobj, axes=tuple(perm) + tuple(range(3, dataobj.ndim)))
    return data_t


def apply_plane_fix(
    dataobj: NDArray,
    plane: str,
) -> NDArray[Any]:
    """Reorder axes to match a target anatomical plane.

    Args:
        dataobj (NDArray): Input data array with spatial axes first.
        plane (str): Target plane name ("axial", "coronal", "sagittal").

    Returns:
        NDArray[Any]: Reoriented data array.
    """
    perm = PLANE_PERM[plane]
    perm_full = tuple(perm) + tuple(range(3, dataobj.ndim))
    dataobj = np.transpose(dataobj, perm_full)
    return dataobj


def correct(dataobj: NDArray, recon_info: Dict[str, Any]) -> NDArray[Any]:
    """Correct data orientation based on reconstruction metadata.

    Args:
        dataobj (NDArray): Input data array with spatial axes first.
        recon_info (Dict[str, Any]): Reconstruction metadata containing
            "GradientOrientation" and "SliceOrientation".

    Returns:
        NDArray[Any]: Orientation-corrected data array.
    """
    R = cast(NDArray[Any], recon_info.get('GradientOrientation'))
    plane = cast(str, recon_info.get('SliceOrientation'))
    dataobj = apply_axis_perm_flip(dataobj, R.T)
    dataobj = apply_plane_fix(
        dataobj,
        plane,
    )
    return dataobj


__all__ = [
    "correct",
]
