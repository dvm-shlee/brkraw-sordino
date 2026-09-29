import math

def shared_intercept_fit(xs: list, ys: list, s: list, w: list = None, drift: bool = True) -> tuple:
    G = len(xs)
    if G == 0:
        raise ValueError("G == 0")
    if len(ys) != G or len(s) != G:
        raise ValueError("Length of xs, ys, and s must be the same")
    
    if w is not None:
        if len(w) != G:
            raise ValueError("Weight list length must match number of groups")

    group_stats = []
    for g in range(G):
        x_g = xs[g]
        y_g = ys[g]
        if len(x_g) != len(y_g):
            raise ValueError("Group x and y lengths must match")
        if len(x_g) < 2:
            raise ValueError("Each group must have at least 2 points")
        
        if w is not None:
            w_g = w[g]
            if len(w_g) != len(x_g):
                raise ValueError("Weight group length must match x group length")
        else:
            w_g = [1.0] * len(x_g)
        
        sw = 0.0
        sx = 0.0
        sxx = 0.0
        sy = 0.0
        sxy = 0.0
        for j in range(len(x_g)):
            weight = w_g[j]
            if weight < 0:
                raise ValueError("Weights cannot be negative")
            sw += weight
            sx += weight * x_g[j]
            sxx += weight * x_g[j] * x_g[j]
            sy += weight * y_g[j]
            sxy += weight * x_g[j] * y_g[j]
        
        if sxx <= 1e-300:
            raise ValueError("group %d has no slope information" % g)
        
        p = sw - sx * sx / sxx
        q = sy - sx * sxy / sxx
        group_stats.append((p, q, sw, sx, sxx, sxy))

    sum_p = sum(stat[0] for stat in group_stats)
    sum_q = sum(stat[1] for stat in group_stats)
    
    if drift:
        sum_ps = sum(stat[0] * s[g] for g, stat in enumerate(group_stats))
        sum_pss = sum(stat[0] * s[g] * s[g] for g, stat in enumerate(group_stats))
        sum_qs = sum(stat[1] * s[g] for g, stat in enumerate(group_stats))
        
        m00 = sum_p
        m01 = sum_ps
        m11 = sum_pss
        det = m00 * m11 - m01 * m01
        if abs(det) <= 1e-12 * max(1e-300, abs(m00 * m11)):
            raise ValueError("singular")
        
        a0 = (sum_q * m11 - sum_qs * m01) / det
        a1 = (m00 * sum_qs - sum_ps * sum_q) / det
    else:
        a1 = 0.0
        if sum_p <= 1e-300:
            raise ValueError("singular")
        a0 = sum_q / sum_p

    b = []
    for g in range(G):
        p, q, sw, sx, sxx, sxy = group_stats[g]
        c = a0 + a1 * s[g]
        b_g = (sxy - c * sx) / sxx
        b.append(b_g)
        
    return (a0, a1, b)

def shared_residual_rms(xs: list, ys: list, s: list, a0: float, a1: float, b: list, w: list = None) -> float:
    G = len(xs)
    if len(ys) != G or len(s) != G or len(b) != G:
        raise ValueError("Shapes of xs, ys, s, and b must match")
    
    if w is not None:
        if len(w) != G:
            raise ValueError("Weight list length must match number of groups")

    total_weighted_sq_res = 0.0
    total_weight = 0.0
    
    for g in range(G):
        x_g = xs[g]
        y_g = ys[g]
        if len(x_g) != len(y_g):
            raise ValueError("Group x and y lengths must match")
        
        if w is not None:
            w_g = w[g]
            if len(w_g) != len(x_g):
                raise ValueError("Weight group length must match x group length")
        else:
            w_g = [1.0] * len(x_g)
            
        for j in range(len(x_g)):
            weight = w_g[j]
            if weight < 0:
                raise ValueError("Weights cannot be negative")
            res = y_g[j] - a0 - a1 * s[g] - b[g] * x_g[j]
            total_weighted_sq_res += weight * res * res
            total_weight += weight
            
    if total_weight <= 0:
        raise ValueError("Total weight must be positive")
        
    return math.sqrt(total_weighted_sq_res / total_weight)
