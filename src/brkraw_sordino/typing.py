from dataclasses import dataclass
from typing import Tuple, Optional, Union
from pathlib import Path


@dataclass
class Options:
    ext_factors: Tuple[float, float, float]
    ignore_samples: int
    offset: int
    num_frames: Optional[int]
    correct_spoketiming: bool
    correct_ramptime: bool
    offreso_freqs: Tuple[Optional[Union[float, int]], ...]
    mem_limit: float
    clear_cache: bool
    split_ch: bool
    cache_dir: Path
    as_complex: bool
    # BRK-0066: algebraic estimate of the k-space centre (kcentre.py), SORDINO 0.1 to 1.x only.
    # The hook's default is on since WI-0113 run 6 (D-0200 2; hook._build_options); an Options
    # built directly keeps False so that internal callers state it.
    estimate_k0: bool = False
    # WI-0113 CP3 (frames.py): golden frames; keyed through the frame plan, not as raw options.
    frame_spokes: Optional[Union[str, int]] = None
    frame_step: Optional[int] = None
    frame_accumulate: bool = False
    # WI-0113 run 6 (D-0199 1): density weights of golden scans (dcf.CHOICES); keyed through
    # the weight rule of golden scans only (hook._build_cache_params), not as a raw option.
    golden_weights: str = "capped"
    # WI-0132 (D-0230): reconstruction matrix at the same FOV (lowres.py), read/phase/slice of
    # PVM_Matrix; None = the acquisition matrix. Keyed only when set (hook._build_cache_params).
    recon_matrix: Optional[Tuple[int, int, int]] = None
    # Peri-event frames (perievent.py): one image per time bin relative to a repeating event,
    # each bin a union of one range per event. Keyed through the frame plan, not as a raw option.
    peri_event: Optional[Union[str, dict]] = None
    # Path to a 0/1 censor file over the frames it was measured on; a
    # peri-event range is used only when all of its spokes are marked usable.
    peri_censor: Optional[str] = None
    # Subtract each spoke position's average over repetitions before gridding
    # (frames.recon_frames); only task-neutral when the repetition length is not
    # a multiple of the event period.
    subtract_rep_mean: bool = False
    # Remove the repetition-locked undersampling pattern in the reconstruction (frames.py):
    # "none", "model" (forward model of the frame's own spokes) or "measured" (repetition mean).
    # Keyed only when not "none" (hook._build_cache_params).
    rep_pattern: str = "none"
    # Anatomical reference of a whole scan as one image (frames.py, frame_spokes="all"): "adjoint"
    # or "cg" (conjugate gradients on the normal operator of all spokes), with its iteration count
    # and Tikhonov weight. Keyed only when the solver is not "adjoint" (hook._build_cache_params).
    anat_solver: str = "adjoint"
    anat_cg_iter: int = 10
    anat_lambda: float = 0.0
    # Static part of rep_pattern="model" frames (frames.py): "adjoint" (the adjoint all-spoke image of
    # x_hat) or "cg" (x_hat itself, with its CG iterations and Tikhonov weight). Keyed only when not
    # "adjoint" (hook._build_cache_params).
    frame_static: str = "adjoint"
    frame_static_iter: int = 10
    frame_static_lambda: float = 0.0


__all__ = [
    'Options'
]
