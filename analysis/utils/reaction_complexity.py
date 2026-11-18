"""Section 2 — reaction complexity and diversity: unique CbyD single-step reactions vs USPTO-Full and Pistachio.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

from . import chem, complexity as cx, figure_template as ft
from .data_loaders import opt_csv
from .paths import a

DATASETS = ["CbyD", "USPTO-Full", "Pistachio"]
KEY = {"CbyD": "cbyd", "USPTO-Full": "uspto_full", "Pistachio": "pistachio"}

CLASS_NAMES = {"0": "Unrecognized", "1": "Heteroatom alkylation/arylation",
               "2": "Acylation", "3": "C-C bond formation", "4": "Heterocycle formation",
               "5": "Protection", "6": "Deprotection", "7": "Reduction", "8": "Oxidation",
               "9": "Functional group interconversion", "10": "Functional group addition",
               "11": "Resolution",
               # a reaction whose NameRxn label joins several classes with ';'
               "mix classes": "Mixed first-level classes",
               "mix sub-classes": "Mixed second-level classes"}


# Data -----
def load_reaction_tables():
    """(features, NameRxn classes, unique CbyD reaction ids); CbyD restricted to its unique reactions."""
    feat = {d: opt_csv("reaction_features", f"{KEY[d]}_rxn_features.csv") for d in DATASETS}
    cls = {d: opt_csv("reaction_class", f"{KEY[d]}_namerxn.csv") for d in DATASETS}
    feat = {d: v for d, v in feat.items() if v is not None}
    cls = {d: v for d, v in cls.items() if v is not None}
    uniq_ids = set(pd.read_csv(a("reactions", "cbyd_unique_reactions.csv")).query("first")["id"])
    n_steps = len(feat["CbyD"])
    feat["CbyD"] = feat["CbyD"][feat["CbyD"]["id"].isin(uniq_ids)].reset_index(drop=True)
    cls["CbyD"] = cls["CbyD"][cls["CbyD"]["id"].isin(uniq_ids)].reset_index(drop=True)
    print(f"CbyD: {n_steps:,} route steps -> {len(feat['CbyD']):,} unique reactions")
    return feat, cls, uniq_ids


def class_fractions(cls):
    """(fraction of reactions per first-level class, per dataset; ordered class keys)."""
    fr = {d: cls[d]["class_1"].astype(str).value_counts(normalize=True) for d in cls}
    keys = {k for f in fr.values() for k in f.index}
    order = (sorted((k for k in keys if str(k).isdigit()), key=int)
             + [k for k in ["mix sub-classes", "mix classes"] if k in keys])
    return fr, order


def feat_series(feat, col, lo=0, hi=None):
    """``{dataset: values of an integer feature}`` (missing/failed features are -1)."""
    out = {}
    for d in DATASETS:
        if d not in feat:
            continue
        s = feat[d][col].values.astype(float); s = s[s >= lo]
        if hi is not None:
            s = s[s <= hi]
        out[d] = s
    return out


def stereo_series(feat):
    """Stereochemistry changes per reaction = R/S + E/Z (rows where both are valid)."""
    out = {}
    for d in DATASETS:
        if d not in feat:
            continue
        df = feat[d]
        m = (df["n_atom_stereo_changes"] >= 0) & (df["n_bond_stereo_changes"] >= 0)
        out[d] = (df.loc[m, "n_atom_stereo_changes"] + df.loc[m, "n_bond_stereo_changes"]).values
    return out


def metal_csv(k):
    return "cbyd_unique_metals.csv" if k == "cbyd" else f"{k}_metals.csv"


def load_metal_counts():
    """``{dataset: {metal: count}}`` over the full datasets (``60_metal_counts.py``)."""
    metal = {}
    for d, k in KEY.items():
        mc = opt_csv("reaction_conditions", metal_csv(k))
        if mc is not None and len(mc):
            metal[d] = dict(zip(mc["metal"], mc["count"]))
    return metal


def top_transition_metals(metal, n=10):
    """``{dataset: [(metal, count), ...]}`` — the n most frequent transition metals."""
    return {d: sorted(((m, c) for m, c in metal[d].items() if m in chem.TRANSITION_METALS),
                      key=lambda kv: -kv[1])[:n]
            for d in DATASETS if d in metal}


def template_overlap(match_col="exact_match"):
    """Unique CbyD reactions recovered by USPTO-Full / Pistachio templates: a reaction counts when a template applied to its product reproduces the recorded precursors.
    Returns counts of the four regions (they partition all CbyD reactions)."""
    te = [opt_csv("template_enumeration", f"cbyd_{k}", "template_enumeration_final_results.csv",
                  usecols=["id", match_col]) for k in ("uspto", "pistachio")]
    hit = te[0].merge(te[1], on="id", suffixes=("_u", "_p"), validate="1:1")
    u, p = hit[f"{match_col}_u"].astype(bool), hit[f"{match_col}_p"].astype(bool)
    return {"uspto_only": int((u & ~p).sum()), "both": int((u & p).sum()),
            "pistachio_only": int((~u & p).sum()), "none": int((~u & ~p).sum()),
            "total": len(hit)}


# Plots -----
def cat_bar(ax, series_by_ds, cap, order):
    """Integer feature as grouped bars over 0..cap, the top bar meaning '>= cap'."""
    cats = list(range(cap + 1))
    w = 0.8 / max(len(order), 1)
    for i, d in enumerate(order):
        s = pd.Series(series_by_ds[d]).astype(int).clip(upper=cap)
        vc = s.value_counts(normalize=True) * 100
        ax.bar(np.array(cats) + i * w, [vc.get(c, 0) for c in cats],
               width=w, color=ft.color(d), label=d)
    ax.set_xticks(np.array(cats) + w * (len(order) - 1) / 2)
    ax.set_xticklabels([str(c) for c in cats[:-1]] + [f"{cap}+"])
    ft.percent_axis(ax)


def _human(n):
    """Approximate dataset size, e.g. 59053 -> '~59K', 4326750 -> '~4.3M'."""
    if n >= 1e6:
        return f"~{f'{n / 1e6:.1f}'.rstrip('0').rstrip('.')}M"
    if n >= 1e3:
        return f"~{round(n / 1e3)}K"
    return f"~{n}"


def _panel_classes(ax, fr, classes):
    shown = [k for k in classes if CLASS_NAMES.get(k, k) != "Resolution"]
    order = [d for d in DATASETS if d in fr]
    x = np.arange(len(shown)); w = 0.8 / max(len(order), 1)
    for i, d in enumerate(order):
        ax.bar(x + i * w, [fr[d].get(k, 0) * 100 for k in shown], width=w, color=ft.color(d), label=d)
    ax.set_xticks(x + w * (len(order) - 1) / 2)
    ax.set_xticklabels([CLASS_NAMES.get(k, k) for k in shown], rotation=20, ha="right")
    ft.percent_axis(ax, top=30, step=10)
    ax.set_ylim(0, 30)
    ax.legend(loc="upper right")


def _panel_venn(ax, ov):
    ax.axis("off")
    lab = lambda n: f"{100 * n / ov['total']:.1f}%"
    # circle AREAS are proportional to the counts
    v = venn2(subsets=(ov["uspto_only"], ov["pistachio_only"], ov["both"]),
              set_labels=("USPTO-Full", "Pistachio"),
              set_colors=(ft.color("USPTO-Full"), ft.color("Pistachio")), alpha=0.55, ax=ax)
    for sid, n in [("10", ov["uspto_only"]), ("01", ov["pistachio_only"]), ("11", ov["both"])]:
        t = v.get_label_by_id(sid)
        if t is not None:
            t.set_text(lab(n)); t.set_fontsize(6)
    # the USPTO-only crescent is too thin for its label -> outside, with a leader line
    t = v.get_label_by_id("10")
    if t is not None:
        t.set_visible(False)
        ax.annotate(lab(ov["uspto_only"]), xy=t.get_position(), xytext=(-16, 0),
                    textcoords="offset points", ha="right", va="center", ma="center", fontsize=6,
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="0.35", shrinkA=1, shrinkB=0))
    for sid, col in [("A", ft.color("USPTO-Full")), ("B", ft.color("Pistachio"))]:
        t = v.get_label_by_id(sid)
        if t is not None:
            t.set_fontsize(6.3); t.set_fontweight("bold"); t.set_color(col)
    ax.text(0.5, -0.06, f"not captured by any template: {100 * ov['none'] / ov['total']:.1f}%",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.0, color="0.35")


def _panel_metals(ax, metal, sizes):
    for d, ranked in top_transition_metals(metal).items():
        freqs = [c for _, c in ranked]
        ax.plot(range(1, len(freqs) + 1), freqs, marker="o", ms=3, color=ft.color(d), ls="None",
                label=f"{d} ({_human(sizes.get(d, 0))})")
        for r, (m, c) in enumerate(ranked, start=1):
            ax.annotate(m, (r, c), xytext=(4, 0), textcoords="offset points", va="center",
                        ha="left", fontweight="bold", color=ft.color(d),
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))
    ax.set_yscale("log")
    ax.set_xlim(0.4, 11)
    ax.set_ylim(top=ax.get_ylim()[1] * 3)                  # headroom for the top labels
    ax.set_xlabel("Rank"); ax.set_ylabel("Count")
    ax.legend()


def plot_reaction_complexity(feat, fr, classes, metal, overlap):
    """Main figure ``reaction_complexity``: (A) reaction classes + reacting atoms / changing rings / stereo changes, (B) template-overlap Venn, (C) transition-metal rank-size."""
    order = [d for d in DATASETS if d in feat]
    sizes = {d: len(feat[d]) for d in feat}
    fig = plt.figure(figsize=(7, 5.5))
    # the class panel gets its own band so its rotated labels don't push the rows below
    gsA = fig.add_gridspec(1, 1, left=0.09, right=0.98, top=0.93, bottom=0.78)
    gsL = fig.add_gridspec(2, 6, left=0.09, right=0.98, top=0.65, bottom=0.10,
                           hspace=0.55, wspace=0.42)

    axc = fig.add_subplot(gsA[0]); _panel_classes(axc, fr, classes)
    ft.add_panel_label(axc, "(A)")
    axr = fig.add_subplot(gsL[0, 0:2])
    cat_bar(axr, feat_series(feat, "num_react_atom_retro"), 6, order)
    axr.set_xlabel("# of reacting atoms")
    for c0, c1, series, cap, lab in [(2, 4, feat_series(feat, "num_change_ring"), 4,
                                      "# of changing rings"),
                                     (4, 6, stereo_series(feat), 4,
                                      "# of stereochemistry changes (R/S + E/Z)")]:
        ax = fig.add_subplot(gsL[0, c0:c1])
        cat_bar(ax, series, cap, order)
        ax.set_xlabel(lab)
        ax.set_ylabel(""); ax.set_yticklabels([])          # shares the first panel's scale

    axB = fig.add_subplot(gsL[1, 0:2]); _panel_venn(axB, overlap)
    ft.add_panel_label(axB, "(B)")
    axC = fig.add_subplot(gsL[1, 2:6]); _panel_metals(axC, metal, sizes)
    ft.add_panel_label(axC, "(C)")
    ft.savefig(fig, "reaction_complexity"); plt.show()


def plot_bonds_formed_si(feat):
    """SI ``cc_cx_bonds``: new C-C and C-heteroatom bonds per reaction."""
    order = [d for d in DATASETS if d in feat]
    fig, axes = plt.subplots(1, 2, figsize=(7, 1.5))
    cat_bar(axes[0], feat_series(feat, "n_cc_bond_formed"), 4, order)
    axes[0].set_xlabel("# of C-C bonds formed"); axes[0].legend()
    cat_bar(axes[1], feat_series(feat, "n_cx_bond_formed"), 4, order)
    axes[1].set_xlabel("# of C-X bonds formed")
    fig.tight_layout(); ft.savefig(fig, "cc_cx_bonds", subdir="SI"); plt.show()


# SI: product complexity of individual reactions -----
# (col, label, fixed range, cap): values >= cap pooled into a "cap+" top bin
PRODUCT_PANELS = [("boettcher", "Böttcher's score", None, 1000),
                  ("sascore", "SAScore", None, None),
                  ("sps", "SPS", None, 2000),
                  ("npscore", "NPScore", None, None),
                  ("overall_product_complexity", "Overall product complexity", (0, 1), None),
                  ("n_heavy", "# of heavy atoms", None, 100),
                  ("n_stereo", "# of tetrahedral stereocentres", None, 20)]
CAP_TICKS = {1000: [0, 250, 500, 750, 1000], 2000: [0, 500, 1000, 1500, 2000],
             100: [0, 25, 50, 75, 100], 20: [0, 5, 10, 15, 20]}


def load_product_complexity(uniq_ids):
    """``{dataset: product complexity table}`` (CbyD: unique reactions only)."""
    pc = {d: opt_csv("complexity", f"{KEY[d]}_products_complexity.csv") for d in DATASETS}
    pc = {d: v for d, v in pc.items() if v is not None}
    pc["CbyD"] = pc["CbyD"][pc["CbyD"]["id"].isin(uniq_ids)]
    return pc


def plot_product_complexity_si(pc):
    """SI ``product_complexity``: the raw scores, overall complexity (also pooled over all datasets, grey), heavy atoms and stereocentres of every reaction's product."""
    pooled = cx.pool_and_normalise(pc)
    overall = pooled["overall_product_complexity"].dropna().values

    def series(col):
        if col == "overall_product_complexity":
            return {**cx.by_dataset(pooled, col), "Overall": overall}
        return {d: v[col][v[col] > -9990] for d, v in pc.items()}

    order = ["Overall"] + [d for d in DATASETS if d in pc and d != "CbyD"] + ["CbyD"]
    fig = plt.figure(figsize=(9, 4.2))
    gs = fig.add_gridspec(2, 4, wspace=0.18, hspace=0.40)
    for i, (col, lab, rng, cap) in enumerate(PRODUCT_PANELS):
        s = series(col)
        if cap is not None:
            s = {n: np.minimum(np.asarray(v, float), cap) for n, v in s.items()}
        if rng is not None:
            lo, hi = rng
        elif cap is not None:
            lo, hi = 0, cap
        else:
            allv = np.concatenate([np.asarray(s.get(n, []), float) for n in order])
            allv = allv[np.isfinite(allv)]
            lo, hi = np.nanmin(allv), np.nanmax(allv)
        bins = (np.arange(0, cap + 2) - 0.5 if col == "n_stereo" and cap is not None
                else np.linspace(lo, hi, 30))
        ax = fig.add_subplot(gs[i // 4, i % 4])
        ft.pct_hist(ax, s, order, bins, colors={"Overall": "gray"})
        ax.set_xlabel(lab)
        ft.add_panel_label(ax, f"({chr(65 + i)})")
        ax.set_ylim(0, 50)
        ax.set_yticks(np.arange(0, 51, 10))
        if cap is not None and cap in CAP_TICKS:
            ticks = CAP_TICKS[cap]
            pad = 0.6 if col == "n_stereo" else cap * 0.03
            ax.set_xlim(lo - pad, cap + pad)
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks[:-1]] + [f"{cap}+"])
        if i == 0:
            ax.set_ylabel("Percentage (%)"); ax.legend()
        elif i == 4:
            ax.set_ylabel("Percentage (%)")
        else:
            ax.set_yticklabels([]); ax.set_ylabel("")
    fig.tight_layout(); ft.savefig(fig, "product_complexity", subdir="SI"); plt.show()
