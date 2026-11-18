"""House plotting style for every CbyD figure: colours, fonts, line styles and a few plot helpers. Call ``ft.set_style()`` first.
"""
from __future__ import annotations
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from . import paths

# Palette — CbyD gets the saturated accent; comparisons are muted. -----
COLORS = {
    "CbyD":          "#cf6a1a",   
    "USPTO-190":     "#225560",   
    "PaRoutes":      "#EDF060",   
    "AbSynth":       "#17BECF",   
    "USPTO-Full":    "#3a6ea5",   
    "Pistachio":     "#84a98c",   
    "Pistachio-Reachable": "#64B5CD",
    "Pistachio-Hard":      "#B07AA1",
}
# Order used whenever several datasets appear in the same panel/legend.
DATASET_ORDER = ["CbyD", "USPTO-190", "PaRoutes", "USPTO-Full", "Pistachio"]

# Qualitative palette for non-dataset categories (reaction classes, roles, …).
CATEGORICAL = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]

# Molecule-role colours for the route trees and route-topology cartoons.
ROLE_COLORS = {"target": "#C8102E", "start": "#55A868",
               "intermediate": "#4C72B0", "byproduct": "#DD8452"}

LINESTYLES = {"CbyD": "-", "USPTO-190": "--", "PaRoutes": "-.",
              "AbSynth": (0, (5, 1)),
              "USPTO-Full": (0, (3, 1, 1, 1)), "Pistachio": ":"}


def color(name: str) -> str:
    """Dataset/category colour with a graceful fallback."""
    if name in COLORS:
        return COLORS[name]
    return CATEGORICAL[hash(name) % len(CATEGORICAL)]


# Paper output conventions (see savefig): every figure is an EDITABLE vector PDF and ALL text is 7 pt; multi-panel figures are exactly FIG_WIDTH inches wide.
FONT_PT = 7
FIG_WIDTH = 7


def set_style(context: str = "paper") -> None:
    """Apply the global rcParams: 7 pt text everywhere and editable-PDF fonts (``pdf.fonttype = 42`` embeds TrueType so text stays selectable/editable in Illustrator/Acrobat, not converted to outlines)."""
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": None,          # keep the exact figure size (width stays 7")
        # Default subplot margins near the page edges so a figure that does NOT set its own gridspec left/right still FILLS the FIG_WIDTH page (no side gaps).
        # Figures that pass explicit left/right (or call tight_layout) override these.
        "figure.subplot.left": 0.09,
        "figure.subplot.right": 0.975,
        "pdf.fonttype": 42,            # editable (TrueType) text in the PDF
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FONT_PT,
        "axes.titlesize": FONT_PT,
        "axes.labelsize": FONT_PT,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.fontsize": FONT_PT,
        "legend.frameon": False,
        "xtick.labelsize": FONT_PT,
        "ytick.labelsize": FONT_PT,
        "lines.linewidth": 1.2,
        "figure.autolayout": False,
    })


def despine(ax, top=True, right=True):
    for side, off in (("top", top), ("right", right)):
        ax.spines[side].set_visible(not off)


def savefig(fig, name, subdir=None, formats=("pdf",)):
    """Save a figure into ``data/data_analysis/figures`` (or a sub-folder) as an EDITABLE vector PDF. Multi-panel figures (>1 axes) are forced to exactly ``FIG_WIDTH`` inches wide (paper column), preserving their height."""
    outdir = paths.FIGURE_DIR if subdir is None else os.path.join(paths.FIGURE_DIR, subdir)
    os.makedirs(outdir, exist_ok=True)
    if len([ax for ax in fig.axes]) > 1:
        w, h = fig.get_size_inches()
        if abs(w - FIG_WIDTH) > 1e-6:
            fig.set_size_inches(FIG_WIDTH, h)
    primary = None
    for fmt in formats:
        p = os.path.join(outdir, f"{name}.{fmt}")
        fig.savefig(p, bbox_inches="tight")
        if primary is None:
            primary = p
    print(f"saved -> {paths.rel(primary)}")
    return primary


# High-level plot helpers -----
def overlaid_hist(ax, series_by_dataset, bins=40, range=None, percent=True,
                  order=None, histtype="step", lw=1.8, fill_cbyd=True,
                  logy=False):
    """Overlay one histogram per dataset (``{dataset: values}``) in the house colours, CbyD last so it sits on top. With ``percent=True`` each dataset's bars sum to 100.
    """
    order = order or [d for d in DATASET_ORDER if d in series_by_dataset]
    order = order + [d for d in series_by_dataset if d not in order]
    if range is None:
        allv = np.concatenate([np.asarray(v, float) for v in series_by_dataset.values() if len(v)])
        range = (np.nanmin(allv), np.nanpercentile(allv, 99.5))
    for name in order:
        v = np.asarray(series_by_dataset[name], float)
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        c = color(name)
        wts = np.ones_like(v) * (100.0 / len(v)) if percent else None
        if name == "CbyD" and fill_cbyd:
            ax.hist(v, bins=bins, range=range, weights=wts, color=c,
                    alpha=0.18, histtype="stepfilled", zorder=5)
        ax.hist(v, bins=bins, range=range, weights=wts, color=c,
                histtype=histtype, lw=lw if name == "CbyD" else 1.3,
                ls=LINESTYLES.get(name, "-"), label=name,
                zorder=6 if name == "CbyD" else 3)
    if logy:
        ax.set_yscale("log")
    if percent:
        percent_axis(ax)
    else:
        ax.set_ylabel("count")
    ax.legend()
    return ax


def pct_hist(ax, series_by_dataset, order, bins, alpha=0.7, colors=None):
    """Overlaid filled histograms, y = % of each dataset's values per bin (datasets drawn in ``order``, so put CbyD last to draw it on top; ``colors`` overrides the house colour per name). Returns the tallest bar."""
    top = 0
    for name in order:
        v = pd.to_numeric(pd.Series(series_by_dataset.get(name, [])), errors="coerce").to_numpy(float)
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        w = np.ones_like(v) * (100 / len(v))
        ax.hist(v, bins=bins, weights=w, color=(colors or {}).get(name, color(name)), label=name,
                histtype="stepfilled", alpha=alpha, lw=0)
        top = max(top, np.histogram(v, bins=bins, weights=w)[0].max())
    return top


def nice_max(v):
    """Smallest 'round' axis limit at least 5% above ``v``."""
    return next(s for s in (1, 2, 5, 10, 20, 25, 40, 50, 60, 80, 100) if s >= v * 1.05)


def percent_axis(ax, label="Percentage (%)", top=100, step=20):
    """Standardise a percentage y-axis: label + fixed 0–``top`` range, ticks every ``step``."""
    ax.set_ylabel(label)
    ax.set_ylim(0, top)
    ax.set_yticks(np.arange(0, top + 1, step))


PANEL_LABEL_PT = 9      # (A)/(B)/(C) tag size — small, just above body text (7 pt)


def add_panel_label(ax, label, dx=-0.06, dy=1.02, size=PANEL_LABEL_PT, **kw):
    """Bold (a)/(b)/(c) panel tag in axes-fraction coordinates."""
    ax.text(dx, dy, label, transform=ax.transAxes, fontsize=size,
            fontweight="bold", va="bottom", ha="right", **kw)
