import math

def stack_layout(row_min: list, row_max: list, gap_frac: float = 0.15) -> dict:
    if not row_min:
        raise ValueError("row_min cannot be empty")
    if len(row_min) != len(row_max):
        raise ValueError("row_min and row_max must have the same length")
    if gap_frac < 0:
        raise ValueError("gap_frac cannot be negative")
    for r_min, r_max in zip(row_min, row_max):
        if r_max < r_min:
            raise ValueError("row_max[k] cannot be less than row_min[k]")

    n = len(row_min)
    lo = float(min(row_min))
    hi = float(max(row_max))
    if hi == lo:
        hi = lo + 1.0
    
    band = hi - lo
    step = band * (1.0 + gap_frac)
    
    offsets = []
    centres = []
    for k in range(n):
        offsets.append((n - 1 - k) * step - lo)
        centres.append((n - 1 - k) * step + band / 2)
        
    return {
        "lo": lo,
        "hi": hi,
        "step": step,
        "offsets": offsets,
        "centres": centres
    }

def wrap_to_pi(x: float) -> float:
    # Wrap to (-pi, pi]
    # x + pi -> (0, 2pi] -> mod 2pi -> (0, 2pi]
    # then subtract pi -> (-pi, pi]
    # Using (x + pi) % (2 * pi) - pi handles the wrapping.
    # To ensure the result is in (-pi, pi], we handle the case where % returns 0.
    res = (x + math.pi) % (2 * math.pi) - math.pi
    if res == -math.pi:
        return math.pi
    return res

def neighbour_diff(phases: list, cyclic: bool = True) -> list:
    if not phases:
        return []
    
    n = len(phases)
    d = [0.0] * n
    
    for i in range(1, n):
        d[i] = wrap_to_pi(phases[i] - phases[i - 1])
        
    if cyclic:
        d[0] = wrap_to_pi(phases[0] - phases[-1])
    else:
        d[0] = 0.0
        
    return d
