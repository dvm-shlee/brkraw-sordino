import math

def polyfit(x: list, y: list, degree: int, w: list = None) -> list:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if len(x) < degree + 1:
        raise ValueError("not enough points for the given degree")
    
    weights = w if w is not None else [1.0] * len(x)
    if w is not None and len(w) != len(x):
        raise ValueError("weights must have the same length as x")
    
    for weight in weights:
        if weight < 0:
            raise ValueError("weights must be non-negative")

    n = degree + 1
    M = [[0.0] * n for _ in range(n)]
    b = [0.0] * n

    for p in range(n):
        for q in range(n):
            s = 0.0
            for i in range(len(x)):
                s += weights[i] * (x[i] ** (p + q))
            M[p][q] = s
        
        s_b = 0.0
        for i in range(len(x)):
            s_b += weights[i] * y[i] * (x[i] ** p)
        b[p] = s_b

    # Gaussian elimination with partial pivoting
    # Find max absolute entry of M for singularity check
    max_m_abs = 0.0
    for row in M:
        for val in row:
            max_m_abs = max(max_m_abs, abs(val))
    
    threshold = 1e-12 * max(1.0, max_m_abs)

    for i in range(n):
        # Pivot
        max_row = i
        max_val = abs(M[i][i])
        for k in range(i + 1, n):
            if abs(M[k][i]) > max_val:
                max_val = abs(M[k][i])
                max_row = k
        
        if max_val <= threshold:
            raise ValueError("singular")
        
        M[i], M[max_row] = M[max_row], M[i]
        b[i], b[max_row] = b[max_row], b[i]

        for k in range(i + 1, n):
            factor = M[k][i] / M[i][i]
            b[k] -= factor * b[i]
            for j in range(i, n):
                M[k][j] -= factor * M[i][j]

    # Back substitution
    c = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(M[i][j] * c[j] for j in range(i + 1, n))
        c[i] = (b[i] - s) / M[i][i]
    
    return c

def polyval(c: list, x: float) -> float:
    if not c:
        return 0.0
    # Horner's rule: c[0] + x(c[1] + x(c[2] + ...))
    res = 0.0
    for coeff in reversed(c):
        res = res * x + coeff
    return res

def unwrap(phases: list) -> list:
    if not phases:
        return []
    
    out = [0.0] * len(phases)
    out[0] = float(phases[0])
    
    pi = math.pi
    two_pi = 2 * pi
    
    for i in range(1, len(phases)):
        d = phases[i] - phases[i-1]
        if abs(d) < pi:
            dm = d
        else:
            dm = ((d + pi) % two_pi) - pi
            if dm == -pi and d > 0:
                dm = pi
        out[i] = out[i-1] + dm
        
    return out

def extrapolate_log_magnitude(x_fit: list, mag_fit: list, x_new: list, degree: int = 1, w: list = None) -> list:
    log_mags = []
    for m in mag_fit:
        if m <= 0:
            raise ValueError("magnitude must be positive for log")
        log_mags.append(math.log(m))
    
    c = polyfit(x_fit, log_mags, degree, w)
    return [math.exp(polyval(c, xn)) for xn in x_new]

def extrapolate_phase(x_fit: list, phase_fit: list, x_new: list, degree: int = 1, w: list = None) -> list:
    unwrapped = unwrap(phase_fit)
    c = polyfit(x_fit, unwrapped, degree, w)
    return [polyval(c, xn) for xn in x_new]
