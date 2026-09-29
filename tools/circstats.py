import math

def circular_stats(angles: list[float]) -> dict:
    if not angles:
        raise ValueError("angles list cannot be empty")
    
    n = len(angles)
    c = sum(math.cos(a) for a in angles) / n
    s = sum(math.sin(a) for a in angles) / n
    r = math.sqrt(c * c + s * s)
    mean = math.atan2(s, c)
    
    if r >= 1.0:
        std = 0.0
    elif r == 0.0:
        std = math.inf
    else:
        std = math.sqrt(-2.0 * math.log(r))
        
    return {"n": n, "mean": mean, "resultant_length": r, "std": std}

def unwrap(angles: list[float]) -> list[float]:
    if not angles:
        return []
    
    result = [angles[0]]
    offset = 0.0
    
    for i in range(1, len(angles)):
        d = angles[i] - angles[i - 1]
        while d > math.pi:
            d -= 2 * math.pi
            offset -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
            offset += 2 * math.pi
        result.append(angles[i] + offset)
        
    return result
