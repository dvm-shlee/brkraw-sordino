"""Summary statistics for tools/eval_ramp.py (standard library only).

cv, pearson, detrend_linear, oscillation (detrended std of a time course after
dropping leading items) and pattern_stability (correlation of each row with
the mean row). Tested by tests/test_tools_evalstats.py.
"""
import math

def cv(values: list[float]) -> float:
    n = len(values)
    if n == 0:
        raise ValueError("values list is empty")
    mean = sum(values) / n
    if mean == 0:
        raise ValueError("mean is exactly 0")
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)
    return std / mean

def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n != len(b):
        raise ValueError("lengths differ")
    if n < 2:
        raise ValueError("length is less than 2")
    
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    
    num = 0.0
    den_a = 0.0
    den_b = 0.0
    
    for ai, bi in zip(a, b):
        da = ai - mean_a
        db = bi - mean_b
        num += da * db
        den_a += da ** 2
        den_b += db ** 2
        
    if den_a == 0 or den_b == 0:
        raise ValueError("zero variance in one of the lists")
        
    return num / math.sqrt(den_a * den_b)

def detrend_linear(values: list[float]) -> list[float]:
    n = len(values)
    if n < 2:
        raise ValueError("n is less than 2")
    
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    
    num = 0.0
    den = 0.0
    for x in range(n):
        num += (x - x_mean) * (values[x] - y_mean)
        den += (x - x_mean) ** 2
        
    slope = num / den
    intercept = y_mean - slope * x_mean
    
    return [values[x] - (intercept + slope * x) for x in range(n)]

def oscillation(values: list[float], exclude: int = 0) -> dict:
    if exclude < 0:
        raise ValueError("exclude is negative")
    
    kept = values[exclude:]
    n = len(kept)
    if n < 3:
        raise ValueError("fewer than 3 items are kept")
    
    mean = sum(kept) / n
    if mean == 0:
        raise ValueError("mean of kept is exactly 0")
    
    d = detrend_linear(kept)
    
    d_mean = sum(d) / n
    variance = sum((v - d_mean) ** 2 for v in d) / n
    std = math.sqrt(variance)
    
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "rel_std": std / abs(mean),
        "peak_to_peak": max(d) - min(d)
    }

def pattern_stability(rows: list[list[float]], exclude: int = 0) -> list[float]:
    if exclude < 0:
        raise ValueError("exclude is negative")
    
    kept = rows[exclude:]
    n_rows = len(kept)
    if n_rows < 2:
        raise ValueError("fewer than 2 rows are kept")
    
    row_len = len(kept[0])
    for row in kept:
        if len(row) != row_len:
            raise ValueError("kept rows do not all have the same length")
            
    reference = []
    for j in range(row_len):
        col_sum = sum(kept[k][j] for k in range(n_rows))
        reference.append(col_sum / n_rows)
        
    return [pearson(row, reference) for row in kept]
