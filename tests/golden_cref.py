"""Line-by-line Python copy of the sequence's golden trajectory C code (WI-0113).

Reference for tests only: scalar loops with ``math`` (the C library functions), in
the order of the C statements, so that the vectorised ``brkraw_sordino.golden``
can be checked against it. One departure is named where it is: with
``GoldenReorder=No`` and NPro below the generated count, C writes past the end
of the list; the copy keeps the first NPro values. Source: sequence ``sordino_260801``
``goldensamp.c`` (``GenerateGoldenSamples``, ``ReorderGoldenSamples``),
``goldengrid.c`` (``GenerateSreagGrid``, ``UpdateCellsGoldenPercell``,
``GenerateSreagTrajectory``), ``backbone.c`` (``UpdateGoldenStepsRange``,
``UpdateGridRange``) and ``BaseLevelRelations.c`` (``InitBeforeAcquisition``
fills GradR/GradP/GradS with zeros first; ``SetBeforeAcquisition`` calls the
generators).
"""
import math

M_PI = 3.14159265358979323846


def generate_golden_samples(n_spokes):
    """GenerateGoldenSamples: (Grad1, Grad2, Grad3, theta) lists of length n_spokes."""
    phi1 = 0.465571231876768
    phi2 = 0.6823278038280193
    twopi = 2.0 * M_PI
    g1 = [0.0] * n_spokes
    g2 = [0.0] * n_spokes
    g3 = [0.0] * n_spokes
    theta = [0.0] * n_spokes
    for n in range(n_spokes):
        frac1 = math.fmod(n * phi1, 1.0)
        frac2 = math.fmod(n * phi2, 1.0)
        z = 1.0 - 2.0 * frac2
        r = math.sqrt(1.0 - z * z)
        theta_n = twopi * frac1
        theta[n] = theta_n
        g1[n] = r * math.cos(theta_n)
        g2[n] = r * math.sin(theta_n)
        g3[n] = z
    return g1, g2, g3, theta


def reorder_golden_samples(n_subsets, n_spokes_per_subset, zstack_angle_deg, use_origin,
                           golden_reorder=True, n_pro=None):
    """ReorderGoldenSamples into GradR/GradP/GradS of length NPro (zero-filled first).

    With GoldenReorder=Yes, NPro = NGoldenSteps = n_subsets * n_spokes_per_subset
    (UpdateGoldenStepsRange). With No, NPro = NGoldenSteps (user value) while the
    generator still writes n_subsets * n_spokes_per_subset values; only the first
    NPro of them exist in the scanner list (C writes past the end when NPro is
    smaller; the tail stays zero when NPro is larger).
    Returns (out1, out2, out3, unplaced) where unplaced counts the spokes that
    fell in no z-stack bucket (their places keep the zero vector).
    """
    n_spokes = n_subsets * n_spokes_per_subset
    if n_pro is None:
        n_pro = n_spokes
    out1 = [0.0] * n_pro
    out2 = [0.0] * n_pro
    out3 = [0.0] * n_pro

    zstack_angle = zstack_angle_deg * M_PI / 180.0
    n_half = int(math.floor(M_PI / zstack_angle))
    n_full = 2 * n_half
    lower_edge = [0.0] * n_full
    upper_edge = [0.0] * n_full
    for i in range(1, n_half + 1):
        lower_edge[i - 1] = math.cos(zstack_angle * i)
    upper_edge[0] = 1.0
    for i in range(1, n_half):
        upper_edge[i] = lower_edge[i - 1]
    for i in range(n_half):
        lower_edge[n_half + i] = lower_edge[n_half - 1 - i]
        upper_edge[n_half + i] = upper_edge[n_half - 1 - i]

    unplaced = 0
    if golden_reorder:
        g1, g2, g3, theta = generate_golden_samples(n_spokes)
        subset_n = 0
        for spoke_scan_n in range(0, n_spokes, n_spokes_per_subset):
            subset = []
            for i in range(n_spokes_per_subset):
                subset.append((g1[spoke_scan_n + i], g2[spoke_scan_n + i], g3[spoke_scan_n + i],
                               theta[spoke_scan_n + i]))
            even_subset = (subset_n % 2 == 0)
            shifted_s = [0.0] * n_spokes_per_subset
            for i in range(n_spokes_per_subset):
                frac = 1.0 - subset[i][3] / (2.0 * M_PI)
                shift = frac * zstack_angle
                polar = math.acos(subset[i][2])
                shifted_s[i] = math.cos(polar + shift) if even_subset else math.cos(polar - shift)
            spoke_n = 0
            for spoke_zstack_n in range(n_half):
                bucket = spoke_zstack_n if even_subset else (n_half + spoke_zstack_n)
                lower = lower_edge[bucket]
                upper = upper_edge[bucket]
                zstack_buf = []
                for i in range(n_spokes_per_subset):
                    if shifted_s[i] > lower and shifted_s[i] <= upper:
                        zstack_buf.append(subset[i])
                # qsort by theta (compare_theta); the golden angles of one subset are distinct
                zstack_buf.sort(key=lambda sp: sp[3])
                base = subset_n * n_spokes_per_subset + spoke_n
                for i in range(len(zstack_buf)):
                    out1[base + i] = zstack_buf[i][0]
                    out2[base + i] = zstack_buf[i][1]
                    out3[base + i] = zstack_buf[i][2]
                spoke_n += len(zstack_buf)
            unplaced += n_spokes_per_subset - spoke_n
            subset_n += 1
    else:
        # The one departure from the C statements: C calls
        # GenerateGoldenSamples(nSpokes, Out1, Out2, Out3, theta) on arrays of NPro values,
        # so with NPro < nSpokes it writes past their end (undefined behaviour in C). The
        # copy writes only the first NPro values, which are what the list holds; with
        # NPro > nSpokes the tail keeps InitBeforeAcquisition's zeros, as in C.
        g1, g2, g3, _ = generate_golden_samples(n_spokes)
        for i in range(min(n_spokes, n_pro)):
            out1[i], out2[i], out3[i] = g1[i], g2[i], g3[i]

    if use_origin:
        out1[0] = 0.0
        out2[0] = 0.0
        out3[0] = 0.0
    return out1, out2, out3, unplaced


def generate_sreag_grid(n_ring):
    """GenerateSreagGrid: dict with nRing, nCell, latEdges (deg), nLon, lonEdges (deg)."""
    d_b = 180.0 / n_ring
    beta0 = [0.0] * n_ring
    for i in range(n_ring):
        t = 0.0 if n_ring == 1 else float(i) / (n_ring - 1)
        beta0[i] = (90.0 - d_b / 2.0) + t * ((-90.0 + d_b / 2.0) - (90.0 - d_b / 2.0))
    d_l = [0.0] * n_ring
    n_lon = [0] * n_ring
    for i in range(n_ring):
        d_l[i] = d_b / math.cos(beta0[i] * M_PI / 180.0)
        n_lon[i] = int(math.floor(360.0 / d_l[i] + 0.5))
        d_l[i] = 360.0 / n_lon[i]
    n_cell = 0
    for i in range(n_ring):
        n_cell += n_lon[i]
    area = 4.0 * M_PI / n_cell
    lat_edges = [0.0] * (n_ring + 1)
    lat_edges[0] = M_PI / 2.0
    for i in range(n_ring):
        d_l_rad = d_l[i] * M_PI / 180.0
        bu = lat_edges[i]
        sin_bl = math.sin(bu) - area / d_l_rad
        if sin_bl > 1.0:
            sin_bl = 1.0
        if sin_bl < -1.0:
            sin_bl = -1.0
        lat_edges[i + 1] = math.asin(sin_bl)
    for i in range(n_ring + 1):
        lat_edges[i] *= 180.0 / M_PI
    lon_edges = []
    for i in range(n_ring):
        lon_edges.append([360.0 * float(j) / n_lon[i] for j in range(n_lon[i] + 1)])
    return {"nRing": n_ring, "nCell": n_cell, "latEdges": lat_edges, "nLon": n_lon,
            "lonEdges": lon_edges}


def update_cells_golden_percell(grid, m):
    """UpdateCellsGoldenPercell: (K1, K2, K3) for frame m (1-based)."""
    theta_g1 = 0.465571231876768
    theta_g2 = 0.6823278038280193
    golden_conj = 0.61803398875
    silver_conj = 0.41421356237
    k1, k2, k3 = [], [], []
    cell_id = 0
    for i in range(grid["nRing"]):
        beta_u = grid["latEdges"][i]
        beta_l = grid["latEdges"][i + 1]
        z_u = math.sin(beta_u * M_PI / 180.0)
        z_l = math.sin(beta_l * M_PI / 180.0)
        z_cs = math.fabs(z_u - z_l)
        lon = grid["lonEdges"][i]
        for j in range(grid["nLon"][i]):
            cell_id += 1
            alpha0 = lon[j]
            theta_cs = lon[j + 1] - lon[j]
            phi = math.fmod(cell_id * golden_conj, 1.0)
            psi = math.fmod(cell_id * silver_conj, 1.0)
            alpha_m = alpha0 + theta_cs * math.fmod(m * theta_g2 + phi, 1.0)
            z_m = z_u - z_cs * math.fmod(m * theta_g1 + psi, 1.0)
            if z_m > 1.0:
                z_m = 1.0
            if z_m < -1.0:
                z_m = -1.0
            r = math.sqrt(1.0 - z_m * z_m)
            alpha_rad = alpha_m * M_PI / 180.0
            k1.append(r * math.cos(alpha_rad))
            k2.append(r * math.sin(alpha_rad))
            k3.append(z_m)
    return k1, k2, k3


def generate_sreag_trajectory(n_ring, n_frames, mirror):
    """UpdateGridRange + GenerateSreagTrajectory: (out1, out2, out3), NPro values."""
    grid = generate_sreag_grid(n_ring)
    n_cell = grid["nCell"]
    stride = n_cell * (2 if mirror else 1)
    n_pro = n_frames * stride
    out1 = [0.0] * n_pro
    out2 = [0.0] * n_pro
    out3 = [0.0] * n_pro
    for m in range(1, n_frames + 1):
        k1, k2, k3 = update_cells_golden_percell(grid, m)
        base = (m - 1) * stride
        for c in range(n_cell):
            out1[base + c] = k1[c]
            out2[base + c] = k2[c]
            out3[base + c] = k3[c]
        if mirror:
            for c in range(n_cell):
                out1[base + n_cell + c] = -k1[c]
                out2[base + n_cell + c] = -k2[c]
                out3[base + n_cell + c] = -k3[c]
    return out1, out2, out3
