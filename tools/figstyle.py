"""Shared figure style for the sordino evaluation tools (WI-0058 run 2).

Director's figure standard (2026-09-29, for report and paper figures): one
comparison per figure, figures in stage order, large text, few panels, the
same colour for a condition in every figure, units on every axis, and the
same y scale for figures that are compared. More files are fine.

Use:

    import figstyle
    figstyle.apply()
    fig, ax = figstyle.figure()               # one panel, 6.4 x 4.2 in
    ax.plot(x, y, color=figstyle.color("S3"), label=figstyle.label("S3"))
    figstyle.save(fig, out_dir / "name")      # name.png (300 dpi) and name.pdf

Development tool, not part of the installed package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

#: One colour per condition, used by every figure (Okabe-Ito based, colour-blind safe).
COLORS: Dict[str, str] = {
    "raw": "#7f7f7f",       # measured, no correction
    "S0": "#999999",
    "S1": "#E69F00",
    "S1p": "#B07800",
    "S1h": "#F0E442",
    "S1hp": "#A89C1E",
    "S2": "#56B4E9",
    "S3": "#0072B2",        # product default
    "S3z": "#D55E00",       # S3 + centre from the least-squares image
    "S3c": "#009E73",       # S3 + first samples from a Gaussian FID curve (run 2, replaced)
    "S3e": "#882255",       # S3 + exponential FID envelope, one K0 shared by all spokes (run 3)
    "truth": "#000000",
    "old": "#CC79A7",       # a superseded result shown for comparison
}

LABELS: Dict[str, str] = {
    "raw": "measured",
    "S0": "S0 no ramp", "S1": "S1 legacy code", "S1p": "S1+φ legacy + its phase", "S1h": "S1h legacy, curvature /2",
    "S1hp": "S1h+φ legacy /2 + its phase", "S2": "S2 integral trajectory",
    "S3": "S3 integral trajectory + FID phase",
    "S3z": "S3z algebraic (image least squares)", "S3c": "S3c Gaussian curve (replaced)",
    "S3e": "S3e exponential FID, shared K0", "truth": "truth", "old": "before the fix",
}

FIGSIZE: Tuple[float, float] = (6.4, 4.2)


def color(name: str) -> str:
    return COLORS.get(name, "#333333")


def label(name: str) -> str:
    return LABELS.get(name, name)


def apply() -> None:
    """Set matplotlib defaults: large text, thin frame, no top/right spines."""
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
        "lines.linewidth": 2.0, "lines.markersize": 6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "legend.frameon": False, "savefig.bbox": "tight", "figure.dpi": 100,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def figure(ncols: int = 1, width: float = FIGSIZE[0], height: float = FIGSIZE[1], **kw):
    import matplotlib.pyplot as plt

    return plt.subplots(1, ncols, figsize=(width * ncols, height), **kw)


def save(fig, stem) -> Tuple[Path, Path]:
    """Write stem.png (300 dpi) and stem.pdf; close the figure."""
    import matplotlib.pyplot as plt

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png, pdf = stem.with_suffix(".png"), stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


__all__ = ["COLORS", "LABELS", "FIGSIZE", "color", "label", "apply", "figure", "save"]
