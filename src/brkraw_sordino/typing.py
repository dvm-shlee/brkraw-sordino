from dataclasses import dataclass
from typing import Tuple, Optional
from pathlib import Path


@dataclass
class Options:
    ext_factors: Tuple[float, float, float]
    ignore_samples: int
    offset: int
    num_frames: Optional[int]
    correct_spoketiming: bool
    correct_ramptime: bool
    offreso_ch: Optional[int]
    offreso_freq: float
    mem_limit: float
    clear_cache: bool
    split_ch: bool
    cache_dir: Path
    as_complex: bool


__all__ = [
    'Options'
]
