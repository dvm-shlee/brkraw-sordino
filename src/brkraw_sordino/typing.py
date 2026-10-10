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
    # BRK-0066: algebraic estimate of the k-space centre (kcentre.py), SORDINO v1-v3 only.
    estimate_k0: bool = False
    # WI-0113 CP3 (frames.py): golden frames; keyed through the frame plan, not as raw options.
    frame_spokes: Optional[Union[str, int]] = None
    frame_step: Optional[int] = None
    frame_accumulate: bool = False


__all__ = [
    'Options'
]
