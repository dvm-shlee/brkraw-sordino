"""The three documents agree with the code (WI-0122).

README.md (GitHub, PyPI) is the short entry; src/brkraw_sordino/docs.md is what
``brkraw hook docs sordino`` shows (a CLI usage page, one line per option, WI-0138);
docs/technical-note.md holds the algorithms with LaTeX equations and, in Appendix A, the full
option tables and practical notes. These tests check that docs.md and the appendix name exactly
the options the code knows, with the code's defaults, that README and docs.md do not repeat each
other, that the math has no syntax slip GitHub would show as an error, and that the equations
of the note give what the code computes (constants, closed forms, small examples).
"""
import dataclasses
import inspect
import math
import re
from pathlib import Path

import numpy as np
import pytest

from brkraw_sordino import dcf, golden, hook, memguard, ramp, recon, traj
from brkraw_sordino.typing import Options

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
DOCS_PATH = ROOT / "src" / "brkraw_sordino" / "docs.md"
DOCS = DOCS_PATH.read_text(encoding="utf-8")
NOTE = (ROOT / "docs" / "technical-note.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------- option tables of docs.md
def _table_rows(text):
    rows = []
    for line in text.splitlines():
        m = re.match(r"^\|\s*`([a-z_0-9]+)`\s*\|\s*(.*?)\s*\|\s*(.*)\|\s*$", line)
        if m:
            rows.append((m.group(1), m.group(2), m.group(3)))
    return rows


#: docs.md is a CLI usage page (WI-0138): one line per option, ``  key=default   text``
_DOCS_OPTION = re.compile(r"^  ([a-z_0-9]+)=(\S*)\s+(\S.*)$")


def _option_rows():
    """The option lines of docs.md (reconstruction, frames, read-time): (name, default, text)."""
    return [m.groups() for m in map(_DOCS_OPTION.match, DOCS.splitlines()) if m]


def _note_option_rows():
    """Rows of the three option tables of the note's Appendix A, not the info-dict table."""
    head = NOTE.split("### A.2 Reconstruction options", 1)[1].split("### A.3 Cautions", 1)[0]
    return _table_rows(head)


def _value(cell):
    text = cell.strip().strip("`").strip()
    if text == "true":
        return True
    if text == "false":
        return False
    if text == "null":
        return None
    for conv in (int, float):
        try:
            return conv(text)
        except ValueError:
            pass
    return text


@pytest.mark.parametrize("rows", [_option_rows, _note_option_rows], ids=["docs_md", "note_appendix"])
def test_option_names_are_the_code_options(rows):
    documented = {name for name, _, _ in rows()}
    code = {f.name for f in dataclasses.fields(Options)} | set(hook.READ_KEYS)
    assert documented == code
    names = [name for name, _, _ in rows()]
    assert len(names) == len(set(names)), "an option is documented twice"


@pytest.mark.parametrize("rows", [_option_rows, _note_option_rows], ids=["docs_md", "note_appendix"])
def test_option_defaults_are_the_code_defaults(tmp_path, rows):
    built = dataclasses.asdict(hook._build_options({"cache_dir": str(tmp_path)}))
    rows = {name: _value(default) for name, default, _ in rows()}
    for name, value in built.items():
        doc = rows[name]
        if name == "cache_dir":
            continue                      # checked below, the default is a path under the config root
        if name == "ext_factors":
            assert doc == 1.0 and value == (1.0, 1.0, 1.0)
        elif name == "offreso_freqs":
            assert doc is None and value == ()
        else:
            assert doc == value, f"{name}: the documents say {doc!r}, the code builds {value!r}"
    assert rows["cache_dir"] == "~/.brkraw/cache/sordino"
    assert '"cache" / "sordino"' in inspect.getsource(hook._get_cache_dir)


def test_read_time_defaults():
    for rows, limit in ((_option_rows, "auto"), (_note_option_rows, "half of physical memory")):
        rows = {name: _value(default) for name, default, _ in rows()}
        assert rows["frames"] is None and rows["axis"] is None
        assert rows["allow_short_fid"] is True
        assert rows["max_memory_gb"] == limit
    assert "auto: half of physical memory" in DOCS
    assert 'kwargs.get("allow_short_fid", True)' in inspect.getsource(hook._plan)
    assert memguard.DEFAULT_FRACTION == 0.5
    assert memguard.FALLBACK_LIMIT_BYTES == 4 * 1024 ** 3


def test_documented_choices_exist():
    assert dcf.CHOICES == ("capped", "pipe", "k2") and dcf.DEFAULT == "capped"
    for word in dcf.CHOICES:
        assert f"`{word}`" in NOTE
    assert "capped, pipe or k2" in DOCS
    assert set(golden.MODES) <= set(re.findall(r"`(\w+Sampling|Default)`", NOTE))


def test_docs_md_is_a_cli_usage_page():
    """WI-0138: what ``brkraw hook docs sordino`` prints is short, 80 columns, usage lines first."""
    lines = DOCS.splitlines()
    assert len(lines) <= 43
    assert max(len(ln) for ln in lines) <= 80, [ln for ln in lines if len(ln) > 80]
    assert lines[2].startswith("usage: brkraw convert STUDY -s SCAN")
    assert "-H sordino:KEY=VALUE" in DOCS and "--hook-args-yaml FILE" in DOCS
    assert "brkraw hook preset sordino" in DOCS
    assert not [ln for ln in lines if ln.startswith(("|", "#", "```"))], "no tables, headings or code blocks"
    assert "docs/technical-note.md, Appendix A" in DOCS and "## Appendix A. Options and practical notes" in NOTE


def test_info_keys_are_documented():
    info_keys = ["shape", "dtype", "count", "nbytes", "frames", "frames_reconstructed", "cached",
                 "cache_path", "cache_dir", "cache_dtype", "cache_nbytes", "peak_nbytes", "recon_nbytes",
                 "recon_chunk_spokes", "recon_chunks", "recon_frames_open", "recon_dcf_nbytes", "recon_k0_method", "recon_k0_reason", "disk_nbytes",
                 "disk_free_nbytes", "limit_nbytes", "limit_source", "frames_planned", "fid_short_nbytes",
                 "sequence_generation", "sequence_method"]
    table = NOTE.split("### A.5 Size estimate without reading data", 1)[1]
    for key in info_keys:
        assert f"`{key}`" in table, key
    source = (inspect.getsource(hook._output_info) + inspect.getsource(hook._plan)
              + inspect.getsource(memguard.memory_limit_bytes))
    for key in info_keys:
        assert f'"{key}"' in source, f"{key} is not an info key of the code"


# ---------------------------------------------------------------- the files and their roles
def test_docs_md_is_what_the_hook_shows():
    manifest = (ROOT / "src" / "brkraw_sordino" / "brkraw_hook.yaml").read_text(encoding="utf-8")
    assert re.search(r"^docs:\s*docs\.md\s*$", manifest, re.M)
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"docs.md"' in pyproject and 'readme = "README.md"' in pyproject
    assert DOCS_PATH.exists()


def test_readme_is_short_and_does_not_repeat_docs_md():
    lines = README.splitlines()
    assert len(lines) <= 60
    assert "brkraw hook docs sordino" in README and "docs/technical-note.md" in README
    assert "| Option |" not in README
    readme_lines = {ln.strip() for ln in lines if len(ln.strip()) >= 40}
    docs_lines = {ln.strip() for ln in DOCS.splitlines()}
    assert not (readme_lines & docs_lines), readme_lines & docs_lines


# ---------------------------------------------------------------- LaTeX of the note
def _math_segments(text):
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", "", text)
    display = re.findall(r"\$\$(.*?)\$\$", text, flags=re.S)
    rest = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)
    inline = []
    for line in rest.splitlines():
        parts = line.split("$")
        assert len(parts) % 2 == 1, f"odd number of $ in: {line[:100]}"
        inline.extend(parts[1::2])
    return display, inline


def test_note_math_syntax():
    display, inline = _math_segments(NOTE)
    assert len(display) >= 15 and len(inline) >= 100
    for seg in display + inline:
        assert seg.strip(), "empty math"
        assert seg.count("{") == seg.count("}") or "\\{" in seg or "\\}" in seg, seg
        stripped = seg.replace("\\{", "").replace("\\}", "")
        assert stripped.count("{") == stripped.count("}"), f"unbalanced braces: {seg}"
        assert len(re.findall(r"\\left\b", seg)) == len(re.findall(r"\\right\b", seg)), seg
        assert len(re.findall(r"\\begin\{", seg)) == len(re.findall(r"\\end\{", seg)), seg
    # a | inside math in a table row would split the cell; the note writes \lvert, \rvert instead
    for line in NOTE.splitlines():
        if line.startswith("|"):
            for seg in re.findall(r"\$(.*?)\$", line):
                assert "|" not in seg, line
    assert NOTE.count("$$") % 2 == 0


def test_note_has_no_unescaped_html_like_text():
    text = re.sub(r"```.*?```", "", NOTE, flags=re.S)
    text = re.sub(r"`[^`\n]*`", "", text)
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)
    text = re.sub(r"\$[^$\n]*\$", "", text)
    assert not re.search(r"<[A-Za-z0-9/!][^>\n]*>", text), re.search(r"<[A-Za-z0-9/!][^>\n]*>", text)


# ---------------------------------------------------------------- equations of the note against the code
def test_note_constants_are_the_code_constants():
    for value in (golden.PHI1, golden.PHI2, golden.GOLDEN_CONJ, golden.SILVER_CONJ):
        assert repr(value) in NOTE or str(value) in NOTE, value
    from brkraw_sordino import kcentre, serial

    assert kcentre.N_ITER == 10 and "10 iterations" in NOTE
    assert dcf.PIPE_ITER == 10 and dcf.PIPE_UPSAMPFAC == 2.0 and "n=0,\\dots,9" in NOTE
    assert serial.NUFFT_EPS == 1e-6 and "10^{-6}" in NOTE
    assert traj.TRAJ_CACHE_VERSION == 2
    assert golden.GRID_HEAD_TOL == 1e-9
    assert traj.O1_ORDER_TOL == 1e-6
    assert recon.GRID_RULE == "affine-fov-centre"
    assert memguard.K0_SAMPLES_MAX_FRACTION == 0.5 and "at most half" in NOTE
    assert memguard.CHUNK_SAMPLES_CAP == 13_000_000 and "13 million" in NOTE
    assert memguard.MIN_CHUNK_SPOKES == 256 and "256-spoke" in NOTE
    assert memguard.SPOKETIMING_FACTOR == 5.0
    assert "65535" in NOTE


def test_default_directions_follow_section_3_1():
    n, under = 16, 1.3
    root_u = math.sqrt(under)
    n_theta = math.ceil(math.pi * n / (2 * root_u))
    vectors = []
    for a in range(n_theta):
        theta = math.pi * (a + 0.5) / n_theta
        n_phi = math.ceil(math.pi * n * math.sin(theta) / (2 * root_u))
        for b in range(n_phi):
            phi = math.pi * (b + 0.5) / n_phi
            vectors.append((math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)))
    assert len(vectors) == traj.calc_npro(n, under)
    code = traj._radial_grad3d(n, root_u, True, False, False, None)        # one hemisphere, ring reversal
    assert code.shape == (3, len(vectors))
    key = lambda v: tuple(np.round(v, 12))                                  # noqa: E731
    assert sorted(map(key, vectors)) == sorted(map(key, code.T))
    full = traj._radial_grad3d(n, root_u, False, True, False, None)         # mirrored, origin first
    half = code.shape[1]
    assert np.all(full[:, 0] == 0) and full.shape[1] == 2 * half + 1
    assert np.allclose(full[:, 1 + half:], -full[:, 1:1 + half])


def test_golden_sampling_follows_section_3_2():
    g, theta = golden.golden_samples(50)
    n = np.arange(50, dtype=float)
    z = 1 - 2 * (n * golden.PHI2 % 1.0)
    azimuth = 2 * np.pi * (n * golden.PHI1 % 1.0)
    expect = np.stack([np.sqrt(1 - z * z) * np.cos(azimuth), np.sqrt(1 - z * z) * np.sin(azimuth), z])
    assert np.allclose(g, expect, atol=1e-12)


@pytest.mark.parametrize("rings", [3, 6, 11, 20])
def test_golden_grid_follows_section_3_3(rings):
    grid = golden.sreag_grid(rings)
    d_beta = 180.0 / rings
    cells = 0
    for l in range(rings):
        beta = 90.0 - d_beta / 2 - l * d_beta
        m_l = math.floor(360.0 * math.cos(math.radians(beta)) / d_beta + 0.5)
        assert grid.n_lon[l] == m_l
        cells += m_l
    assert grid.n_cell == cells
    omega = 4 * math.pi / cells
    sin_b = 1.0
    for l in range(rings):
        assert math.isclose(math.sin(math.radians(grid.lat_edges[l])), sin_b, abs_tol=1e-12)
        sin_b = max(-1.0, min(1.0, sin_b - omega * grid.n_lon[l] / (2 * math.pi)))
    # one spoke of the cell formula, frame m, cell q (1-based)
    m, q = 3, 5
    spokes = golden.sreag_cells(grid, m)
    l, first = 0, 0
    while q - 1 >= first + grid.n_lon[l]:
        first += grid.n_lon[l]
        l += 1
    j = q - 1 - first
    z_u = math.sin(math.radians(grid.lat_edges[l]))
    d_z = abs(z_u - math.sin(math.radians(grid.lat_edges[l + 1])))
    alpha0 = 360.0 * j / grid.n_lon[l]
    d_alpha = 360.0 / grid.n_lon[l]
    frac = lambda x: x % 1.0                                                # noqa: E731
    alpha = alpha0 + d_alpha * frac(m * golden.PHI2 + frac(q * 0.61803398875))
    z = z_u - d_z * frac(m * golden.PHI1 + frac(q * 0.41421356237))
    expect = (math.sqrt(1 - z * z) * math.cos(math.radians(alpha)),
              math.sqrt(1 - z * z) * math.sin(math.radians(alpha)), z)
    assert np.allclose(spokes[:, q - 1], expect, atol=1e-9)
    mirrored = golden.sreag_trajectory(rings, 2, True)
    assert mirrored.shape == (3, 2 * 2 * cells)
    assert np.allclose(mirrored[:, cells:2 * cells], -mirrored[:, :cells])


@pytest.mark.parametrize("start,length", [(5.0, 40.0), (-30.0, 50.0), (0.0, 12.0), (-4.0, 3.0)])
def test_ramp_integral_is_the_integral_of_the_fraction(start, length):
    for t in (0.0, 3.0, 17.5, 60.0, 200.0):
        grid = np.linspace(0.0, t, 200001)
        f = np.clip((grid - start) / length, 0.0, 1.0)
        numeric = float(np.sum((f[1:] + f[:-1]) / 2 * np.diff(grid)))
        assert math.isclose(ramp.ramp_integral(t, start, length), numeric, rel_tol=1e-6, abs_tol=1e-6)
        assert math.isclose(ramp.phase_delay(t, start, length), t - numeric, rel_tol=1e-6, abs_tol=1e-6)


def test_ramp_trajectory_follows_section_4():
    rng = np.random.default_rng(3)
    g = rng.normal(size=(3, 7))
    g /= np.linalg.norm(g, axis=0)
    matrix, over = 16, 2.0
    n = int(matrix / 2 * over)
    dwell = 3.0
    times = [10.0 + j * dwell for j in range(n)]
    start, length = 12.0, 30.0
    integral = [ramp.ramp_integral(t, start, length) for t in times]
    k = traj.calc_radial_traj3d_integral(g, matrix, over, times, integral, dwell)
    u = 1.0 / (2 * (n - 1))
    for i in range(7):
        prev = g[:, i - 1]
        for j in range(n):
            expect = u * (times[j] / dwell * prev + integral[j] / dwell * (g[:, i] - prev))
            assert np.allclose(k[i, j], expect, atol=1e-14)
    # a constant gradient (no ramp) reaches the fixed-vector trajectory with tau0 = T0 / dwell
    const = traj.calc_radial_traj3d_integral(g, matrix, over, times, times, dwell)
    fixed = traj.calc_radial_traj3d(g, matrix, over, 10.0 / dwell)
    assert np.allclose(const, fixed, atol=1e-14)


def test_phase_corrections_follow_section_5():
    d = np.array([5000.0, -3000.0, 0.0])
    tau = np.array([0.0, 1e-5, 2e-5, 4e-5])
    rows = recon.PhaseRows(d, tau)
    expect = np.exp(-2j * np.pi * d[:, None] * tau[None, :])
    assert np.allclose(rows[:], expect, atol=1e-5)
    e = np.array([1000.0, 2000.0, 3000.0])
    t = np.array([1e-5, 2e-5, 3e-5, 4e-5])
    both = recon.PhaseRows(d, tau, e, t)
    assert np.allclose(both[:], np.exp(-2j * np.pi * (d[:, None] * tau[None, :] + e[:, None] * t[None, :])),
                       atol=1e-5)
    # off-resonance: exp(-2 pi i f (j' + 1) dwell), dwell = 1 / (B sigma)
    k = np.ones((2, 5), dtype=complex)
    out = recon.correct_offreso(k, 100.0, eff_bandwidth=50000.0, over_sampling=2.0)
    dwell = 1.0 / (50000.0 * 2.0)
    assert np.allclose(out[0], np.exp(-2j * np.pi * 100.0 * (np.arange(5) + 1) * dwell))


def test_pipe_menon_cap_follows_section_6():
    rng = np.random.default_rng(5)
    g = rng.normal(size=(3, 300))
    g /= np.linalg.norm(g, axis=0)
    samples = np.linspace(0.02, 0.5, 24)
    tr = samples[None, :, None] * g.T[:, None, :]
    shape = (24, 24, 24)
    capped = dcf.pipe_weights(tr, 300, 1, shape, 100).w
    free = dcf.pipe_weights(tr, 300, 1, shape, 100, cap=False).w
    d = np.square(tr[:, 1:, :]).sum(-1)
    d_max = float(np.square(tr[:, 1:, :]).sum(-1).max())
    d = d / d_max
    # uncapped: the sum equals the |k|^2 sum; capped: lambda * min(w_hat, d), lambda >= 1, same sum
    assert math.isclose(float(free.sum()), float(d.sum()), rel_tol=1e-9)
    assert math.isclose(float(capped.sum()), float(d.sum()), rel_tol=1e-9)
    lam = float(d.sum()) / float(np.minimum(free, d).sum())
    assert lam >= 1.0
    assert np.allclose(capped, lam * np.minimum(free, d), rtol=1e-9, atol=1e-12)


def test_grid_shift_and_ext_factor_affine_follow_section_8():
    affine = np.diag([0.2, -0.2, 0.2, 1.0])
    affine[:3, 3] = [-12.0, 11.8, -11.9]
    centre = np.array([0.3, -0.1, 0.0])
    n = [120, 120, 120]
    shift = recon.grid_shift_from_affine(affine, centre, (0, 1, 2), n)
    index = np.linalg.solve(affine[:3, :3], centre - affine[:3, 3])
    assert np.allclose(shift, index - 60, atol=1e-6)
    # the phase origin lands on index N//2 + s on the NUFFT grid: object at d, phase factor exp(-i w.s)
    om = np.array([[0.3, -0.2, 0.7]])
    factor = recon.grid_shift_factor(om / (2 * np.pi), (0.5, 0.0, -0.25))
    assert np.allclose(factor, np.exp(-1j * (0.3 * 0.5 + 0.7 * -0.25)))
    # ext_factors: A' = A [[I, v], [0, 1]] with v = -(M//2 - N//2), the centre still maps to c
    base, scaled = [16, 16, 16], [int(16 * 1.5), int(16 * 1.1), 16]
    v = tuple(-(m // 2 - b // 2) for m, b in zip(scaled, base))
    moved = hook._apply_ext_factor_affine(affine, v)
    expect = affine @ np.block([[np.eye(3), np.array(v, dtype=float)[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])
    assert np.allclose(moved, expect)


def test_uint16_scaling_follows_section_11():
    rng = np.random.default_rng(7)
    x = rng.uniform(-3.0, 9.0, size=(4, 5, 6, 3))
    q, slope, inter = hook._calc_slope_inter(x)
    assert q.dtype == np.uint16 and inter == x.min()
    assert math.isclose(slope, (x.max() - x.min()) / 65535)
    assert int(q.max()) == 65535 and int(q.min()) == 0
    assert np.max(np.abs(inter + slope * q.astype(float) - x)) <= slope / 2 + 1e-12
    _, flat_slope, _ = hook._calc_slope_inter(np.full((2, 2, 2), 4.0))
    assert flat_slope == 1.0
