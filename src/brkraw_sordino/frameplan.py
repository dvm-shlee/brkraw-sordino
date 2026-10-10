"""Spoke ranges of the frames of a golden-angle scan (WI-0113 CP3)."""

def frame_ranges(n_spokes, window, step, accumulate):
    """Returns a list of (lo, hi) tuples of ints for the frame ranges."""
    for name, value in [("n_spokes", n_spokes), ("window", window), ("step", step)]:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    
    if not isinstance(accumulate, bool):
        raise ValueError(f"accumulate must be True or False, got {accumulate!r}")
    
    if window > n_spokes:
        raise ValueError(f"window {window} is larger than the {n_spokes} spokes read")
    
    ranges = []
    k = 0
    while True:
        if not accumulate:
            lo = k * step
            hi = lo + window
        else:
            lo = 0
            hi = window + k * step
        
        if hi > n_spokes:
            break
        
        ranges.append((lo, hi))
        k += 1
    
    return ranges

def crossing_count(ranges, period):
    """Returns the number of ranges that contain a multiple of period strictly inside."""
    count = 0
    for lo, hi in ranges:
        if (lo // period) != ((hi - 1) // period):
            count += 1
    return count

def misaligned_count(ranges, unit):
    """Returns the number of ranges with lo % unit != 0 or hi % unit != 0."""
    count = 0
    for lo, hi in ranges:
        if lo % unit != 0 or hi % unit != 0:
            count += 1
    return count

def frame_scales(ranges, period):
    """Returns a list of floats: period / (hi - lo) for each range."""
    return [float(period) / float(hi - lo) for lo, hi in ranges]
