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
    # WI-0056: "integral" (gradient integral, timing.py) or "legacy" (the
    # form before WI-0056, kept for pre/post comparison).
    ramp_model: str = "integral"
    # WI-0056: per-projection accumulated phase correction (integral model only).
    correct_phase: bool = True


__all__ = [
    'Options'
]
