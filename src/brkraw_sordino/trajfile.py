"""Spoke directions of the lines of a ParaVision radial traj file (WI-0113)."""

import numpy as np

def line_directions(data, matrix, n_lines):
    """Calculate the least-squares spoke directions and the maximum misfit for the first n_lines."""
    if not isinstance(matrix, int) or isinstance(matrix, bool) or matrix <= 0:
        raise ValueError(f"matrix must be a positive integer, got {matrix!r}")
    if not isinstance(n_lines, int) or isinstance(n_lines, bool) or n_lines <= 0:
        raise ValueError(f"n_lines must be a positive integer, got {n_lines!r}")

    need = n_lines * matrix * 3 * 8
    if len(data) < need:
        raise ValueError(f"traj data hold {len(data)} bytes, {need} are needed for {n_lines} lines")

    t = np.frombuffer(data, dtype="<f8", count=n_lines * matrix * 3).reshape(n_lines, matrix, 3)
    s = np.arange(matrix, dtype=np.float64) / matrix - 0.5
    
    g = np.einsum("lic,i->lc", t, s) / float(np.sum(s * s))
    fit = float(np.max(np.abs(t - s[None, :, None] * g[:, None, :])))
    
    return (np.ascontiguousarray(g.T, dtype=np.float64), fit)

def head_match(dirs, default_dirs, golden_dirs, tol=1e-9):
    """Compare the extracted directions against default and golden reference sets."""
    a = np.asarray(dirs, dtype=np.float64)
    b = np.asarray(default_dirs, dtype=np.float64)
    c = np.asarray(golden_dirs, dtype=np.float64)
    
    if not (a.shape == b.shape == c.shape):
        raise ValueError(f"shapes differ: {a.shape}, {b.shape}, {c.shape}")
    
    if a.size == 0:
        raise ValueError("no directions to compare")
    
    d_default = float(np.max(np.abs(a - b)))
    d_golden = float(np.max(np.abs(a - c)))
    
    if d_default <= tol:
        name = "default"
    elif d_golden <= tol:
        name = "golden"
    else:
        name = "neither"
        
    return (name, d_default, d_golden)
