import math

def refine_peak(xs, ys) -> tuple[float, float]:
    if len(xs) != len(ys) or len(xs) < 3:
        raise ValueError
    
    k = 0
    max_val = ys[0]
    for i in range(1, len(ys)):
        if ys[i] > max_val:
            max_val = ys[i]
            k = i
            
    if k == 0 or k == len(ys) - 1:
        return (float(xs[k]), float(ys[k]))
    
    y0 = ys[k-1]
    y1 = ys[k]
    y2 = ys[k+1]
    h = xs[k+1] - xs[k]
    denom = y0 - 2*y1 + y2
    
    if denom == 0:
        return (float(xs[k]), float(ys[k]))
    
    p = 0.5 * (y0 - y2) / denom
    x_peak = xs[k] + p * h
    y_peak = y1 - 0.25 * (y0 - y2) * p
    return (float(x_peak), float(y_peak))

def weighted_mean_std(values, weights) -> tuple[float, float]:
    if not values or len(values) != len(weights):
        raise ValueError
    
    sum_w = 0.0
    for w in weights:
        if w < 0:
            raise ValueError
        sum_w += w
        
    if sum_w == 0:
        raise ValueError
        
    mean = sum(w * v for w, v in zip(weights, values)) / sum_w
    var = sum(w * (v - mean)**2 for w, v in zip(weights, values)) / sum_w
    return (float(mean), float(math.sqrt(var)))
