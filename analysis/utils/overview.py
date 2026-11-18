"""Section 1 — dataset overview: Table 1 and the ``dataset_overview`` figure, one row per unique target so resynthesised targets are not counted twice.
"""
from __future__ import annotations
import collections
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms
from matplotlib.ticker import MaxNLocator
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdmd

from . import chem, complexity as cx, draw as dr, figure_template as ft
from .data_loaders import opt_csv
from .paths import a

TARGET_DATASETS = ["CbyD", "USPTO-190", "PaRoutes"]
PLOT_ORDER = ["USPTO-190", "PaRoutes", "CbyD"]          # CbyD drawn last (on top)
ELEMS = ["C", "N", "O", "S", "P", "Halogens"]
_HAL = {"F", "Cl", "Br", "I"}
RING_CATS = [("n_arom", "Aromatic rings"), ("n_hetarom", "Heteroaromatic rings"),
             ("n_aliph_het", "Non-aromatic heterocycles"),
             ("n_aliph_carb", "Non-aromatic carbocycles")]
F_ELEMS = [("O", 10), ("N", 10), ("Halogens", 6), ("S", 6), ("P", 6)]   # panel F: element, cap
COMP_ALPHA = 0.7


# Data -----
def load_cbyd_topology():
    """One row per CbyD route with a parsed reaction network."""
    topo = opt_csv("route_topology", "cbyd_route_topology.csv")
    return topo[topo["error"].isna()].copy()


def scope_table(topo):
    """Table 1. Pharmaceutical = a target for which ANY route names a drug."""
    n_reactions = sum(1 for _ in open(a("reactions", "cbyd_reactions.jsonl")))
    tgt = topo.dropna(subset=["target"]).copy()
    tgt["is_pharma"] = tgt["drug_name"].notna()
    pharma_by_target = tgt.groupby("target")["is_pharma"].any()
    n_unique = pharma_by_target.index.nunique()
    n_pharma = int(pharma_by_target.sum())
    return pd.DataFrame({
        "Number of": ["Multi-step routes", "Reactions (single-step)",
                      "Unique target molecules", "   Natural products / other",
                      "   Pharmaceuticals"],
        "Count": [len(topo), n_reactions, n_unique, n_unique - n_pharma, n_pharma],
    })


def load_target_complexity():
    """``{dataset: target complexity table}``, one row per unique target."""
    tc = {"CbyD":      opt_csv("complexity", "cbyd_targets_complexity.csv"),
          "USPTO-190": opt_csv("complexity", "uspto190_targets_complexity.csv"),
          "PaRoutes":  opt_csv("complexity", "paroutes_targets_complexity.csv")}
    return {k: v.drop_duplicates("smiles").reset_index(drop=True)
            for k, v in tc.items() if v is not None}


def composition(smi):
    """Element counts, ring-type counts and ring junctions of one target.

    Two SSSR rings sharing exactly one bond are FUSED, sharing two or more are BRIDGED. Rings sharing a bond form one ring system, and an atom shared by two different systems is a SPIRO centre.
    """
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return None
    c = dict.fromkeys(ELEMS, 0)
    for at in m.GetAtoms():
        e = at.GetSymbol()
        if e in _HAL:
            c["Halogens"] += 1
        elif e in c:
            c[e] += 1
    rings = [set(r) for r in m.GetRingInfo().AtomRings()]
    c["n_rings"] = len(rings)
    bonds = [set(r) for r in m.GetRingInfo().BondRings()]   # parallel to `rings`
    shared = [len(a & b) for i, a in enumerate(bonds) for b in bonds[i + 1:]]
    c["fused"] = 1 in shared
    c["bridged"] = any(k >= 2 for k in shared)
    root = list(range(len(rings)))                         # union-find: ring -> its ring system

    def _find(i):
        while root[i] != i:
            i = root[i]
        return i
    for i, a in enumerate(bonds):
        for j in range(i + 1, len(bonds)):
            if a & bonds[j]:
                root[_find(j)] = _find(i)
    systems = collections.defaultdict(set)                 # atom -> ring systems it sits in
    for i, r in enumerate(rings):
        for at in r:
            systems[at].add(_find(i))
    c["spiro"] = any(len(s) > 1 for s in systems.values())
    c.update(n_arom=rdmd.CalcNumAromaticRings(m),
             n_hetarom=rdmd.CalcNumAromaticHeterocycles(m),
             n_aliph_het=rdmd.CalcNumAliphaticHeterocycles(m),
             n_aliph_carb=rdmd.CalcNumAliphaticCarbocycles(m))
    return c


def composition_tables(tc):
    """``{dataset: one composition row per target}`` in TARGET_DATASETS order."""
    return {name: pd.DataFrame([r for r in map(composition, tc[name]["smiles"]) if r is not None])
            for name in TARGET_DATASETS if name in tc}


def composition_summary(comp):
    """% of targets with >= 1 of each element / ring type / ring junction."""
    return pd.DataFrame({n: {**{f"% with >=1 {e}": (c[e] > 0).mean() * 100 for e in ELEMS},
                             **{f"% with >=1 {lab.lower()}": (c[col] > 0).mean() * 100
                                for col, lab in RING_CATS},
                             "% acyclic": (c["n_rings"] == 0).mean() * 100,
                             **{f"% with >=1 {k} junction": c[k].mean() * 100
                                for k in ("fused", "bridged", "spiro")},
                             "n": len(c)}
                         for n, c in comp.items()}).round(1)


def publication_years(df):
    y = pd.to_numeric(df["year"], errors="coerce").dropna()
    return y.values


def _target_key(df):
    """Group routes by target NAME with stereodescriptors stripped, so (+)/(-)/(+/-)/ R/S/D/L/dl variants count as ONE target (canonical SMILES when unnamed)."""
    df = df.dropna(subset=["target"]).copy()
    nm = df["target_name"].map(lambda x: chem.strip_stereo_prefix(x).strip().lower()
                               if isinstance(x, str) and x.strip() else "")
    df["tkey"] = nm.where(nm.astype(bool), df["target"])
    return df


def routes_per_target(topo, top=5):
    """(# targets with k routes, the ``top`` most-resynthesised targets)."""
    keyed = _target_key(topo)
    counts = keyed.groupby("tkey").size().value_counts().sort_index()
    top_targets = (keyed.groupby("tkey")
                   .agg(n_routes=("route", "size"),
                        name=("target_name", lambda s: chem.strip_stereo_prefix(
                            next((x for x in s if isinstance(x, str) and x.strip()), ""))))
                   .sort_values("n_routes", ascending=False).head(top))
    return counts, top_targets


# Main figure: dataset_overview -----
# (column, x label, fixed range, cap): values >= cap are pooled into a "cap+" top bin
C_PANELS = [("overall_product_complexity", "Overall target complexity", (0, 1), None),
            ("mw", "Molecular weight", None, 1000),
            ("n_stereo", "# of tetrahedral stereocenters", None, 15)]


def _panel_years(ax, years):
    yr = np.maximum(years, 1940)                   # every route before 1940 in one bin
    ymax = int(years.max())
    ax.hist(yr, bins=np.arange(1940, ymax + 2), color=ft.color("CbyD"), alpha=0.85)
    ax.set_xlabel("Publication year"); ax.set_ylabel("# of routes")
    ax.set_xlim(1936, ymax + 3)
    xt = list(range(1940, ymax + 1, 20))
    ax.set_xticks(xt)
    ax.set_xticklabels(["<1940"] + [str(t) for t in xt[1:]])


def _panel_routes_per_target(ax, routes_counts, top_targets):
    ax.bar(routes_counts.index, routes_counts.values, color=ft.color("CbyD"), alpha=0.85,
           width=0.85, bottom=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("# of routes per target"); ax.set_ylabel("# of targets")
    # each top target named above its bar with a vertical arrow; staggered label heights keep neighbouring bars (e.g. 25 & 26) from colliding
    label_y = [1500, 500, 160, 50, 15]
    for (_, r), yt in zip(top_targets.iterrows(), label_y):
        xb = int(r["n_routes"])
        nm = r["name"]; nm = nm[:1].lower() + nm[1:]
        ax.annotate(nm, xy=(xb, routes_counts.get(xb, 1)), xytext=(xb, yt),
                    ha="center", va="bottom", fontsize=6.5, color="0.15", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"),
                    arrowprops=dict(arrowstyle="->", lw=0.6, color="0.45", shrinkB=12))
    ax.set_xlim(-2, int(routes_counts.index.max()) + 5)


def _panel_target_props(fig, slot, targets):
    """(C) one overlaid histogram per property, own y-scale per plot."""
    gs = slot.subgridspec(1, 6, wspace=0.3, hspace=0.25)
    for i, (col, lab, rng, cap) in enumerate(C_PANELS):
        if col == "n_stereo":                               # integer-aligned bins
            bins = np.arange(0, cap + 2) - 0.5
        else:
            bins = np.linspace(*(rng if rng is not None else (0, cap)), 21)
        series = cx.by_dataset(targets, col)
        if cap is not None:
            series = {n: np.minimum(v, cap) for n, v in series.items()}
        ax = fig.add_subplot(gs[0, i * 2:(i + 1) * 2])
        top = ft.pct_hist(ax, series, PLOT_ORDER, bins)
        ax.set_xlabel(lab)
        ax.set_ylim(0, ft.nice_max(top))
        ax.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 5, 10]))
        if cap == 1000:
            ax.set_xlim(-45, 1045); ax.set_xticks([0, 250, 500, 750, 1000])
            ax.set_xticklabels(["0", "250", "500", "750", "1000+"])
        elif cap == 15:
            ax.set_xlim(-1.2, 16.2); ax.set_xticks([0, 5, 10, 15])
            ax.set_xticklabels(["0", "5", "10", "15+"])
        if i == 0:
            ax.set_ylabel("Percentage (%)")
            ft.add_panel_label(ax, "(C)")
            ax.legend()


def _hist_row(fig, comp, slots, cats, caps, xlabel, tick_sets, label, label_dx):
    """(D)/(F): one small histogram per (column, title) in ``cats``. x = count per target (0 not shown, >= cap pooled), y = % of ALL the dataset's targets."""
    draw = [n for n in PLOT_ORDER if n in comp]
    axes = []
    for k, (slot, (col, title), cap, ticks) in enumerate(zip(slots, cats, caps, tick_sets)):
        bins = np.arange(0.5, cap + 1)
        ax = fig.add_subplot(slot)
        top = 0
        for n in draw:
            v = comp[n][col].values
            v = np.minimum(v[v > 0], cap)
            w = np.full(len(v), 100 / len(comp[n]))
            ax.hist(v, bins=bins, weights=w, color=ft.color(n), alpha=COMP_ALPHA,
                    histtype="stepfilled", lw=0)
            top = max(top, np.histogram(v, bins=bins, weights=w)[0].max())
        ax.set_ylim(0, ft.nice_max(top))
        ax.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 5, 10]))
        ax.set_xlim(0.4, cap + 0.6)
        ax.set_xticks(range(1, cap + 1))
        ax.set_xticklabels([str(t) + ("+" if t == cap else "") if t in ticks else ""
                            for t in range(1, cap + 1)])
        ax.set_title(title, pad=3)
        if k == 0:
            ax.set_ylabel("Percentage of targets (%)")
            ft.add_panel_label(ax, label, dx=label_dx)
        axes.append(ax)
    p0, p1 = axes[0].get_position(), axes[-1].get_position()
    fig.text((p0.x0 + p1.x1) / 2, p0.y0 - 0.032, xlabel, ha="center", va="top")


def _panel_junctions(ax, comp):
    """(E) % of targets with >= 1 fused / bridged / spiro junction (grouped bars)."""
    order = [n for n in PLOT_ORDER if n in comp]
    cats = [("fused", "Fused"), ("bridged", "Bridged"), ("spiro", "Spiro")]
    w = 0.8 / len(order)
    for j, n in enumerate(order):
        col = tuple(COMP_ALPHA * v + 1 - COMP_ALPHA for v in mcolors.to_rgb(ft.color(n)))
        xs = np.arange(len(cats)) + (j - (len(order) - 1) / 2) * w
        ax.bar(xs, [comp[n][k].mean() * 100 for k, _ in cats], width=w, color=col, lw=0, zorder=3)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels([lab for _, lab in cats])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, len(cats) - 0.45)
    ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20))
    ax.grid(False, axis="x")


def plot_overview(years, routes_counts, top_targets, targets, comp):
    """Main figure ``dataset_overview`` (4 rows): A years | B routes/target; C target properties; D ring types | E ring junctions; F heteroatoms."""
    fig = plt.figure(figsize=(8, 10.3))
    gs = fig.add_gridspec(4, 6, hspace=0.55, wspace=1)

    axA = fig.add_subplot(gs[0, 0:3]); _panel_years(axA, years)
    ft.add_panel_label(axA, "(A)")
    axB = fig.add_subplot(gs[0, 3:6]); _panel_routes_per_target(axB, routes_counts, top_targets)
    ft.add_panel_label(axB, "(B)")

    _panel_target_props(fig, gs[1, :], targets)

    gs_DE = gs[2, :].subgridspec(1, 6, wspace=0.3)
    _hist_row(fig, comp, [gs_DE[0, k] for k in range(4)],
              [(col, lab.replace("Non-aromatic ", "Non-aromatic\n")) for col, lab in RING_CATS],
              caps=[6] * 4, xlabel="# of rings per target", tick_sets=[range(1, 7)] * 4,
              label="(D)", label_dx=-0.13)
    # F: bar widths follow the bin count (10 vs 6) so every bar is ~equally wide
    _hist_row(fig, comp, list(gs[3, :].subgridspec(1, 5, wspace=0.3, width_ratios=[9, 9, 6, 6, 6])),
              [(e, "Halogens (F, Cl, Br, I)" if e == "Halogens" else e) for e, _ in F_ELEMS],
              caps=[k for _, k in F_ELEMS], xlabel="# of atoms per target",
              tick_sets=[range(1, k + 1) for _, k in F_ELEMS], label="(F)", label_dx=-0.1)
    axE = fig.add_subplot(gs_DE[0, 4:6]); _panel_junctions(axE, comp)
    ft.add_panel_label(axE, "(E)")

    ft.savefig(fig, "dataset_overview"); plt.show()


# SI: the four raw complexity scores of the targets -----
SI_PANELS = [("sascore", "SAScore", None), ("npscore", "NPScore", None),
             ("sps", "SPS", 2000), ("boettcher", "Böttcher's score", 1000)]   # (col, label, cap)


def plot_target_complexity_si(targets):
    """SI ``target_complexity_metrics``: SAScore, NPScore, SPS, Böttcher (values >= cap pooled into one extra "cap+" bin)."""
    fig = plt.figure(figsize=(8, 2))
    gs = fig.add_gridspec(1, 4, wspace=0.35, hspace=0.3)
    for i, (col, lab, cap) in enumerate(SI_PANELS):
        series = cx.by_dataset(targets, col)
        allv = np.concatenate([series[n] for n in PLOT_ORDER])
        allv = allv[np.isfinite(allv)]
        lo = np.nanmin(allv)
        if cap is not None:
            w = (cap - lo) / 20                                   # 20 bins up to the cap ...
            bins = np.append(np.linspace(lo, cap, 21), cap + w)   # ... + one overflow bin
            series = {n: np.where(v < cap, v, cap + w / 2) for n, v in series.items()}
        else:
            bins = np.linspace(lo, np.nanmax(allv), 21)
        ax = fig.add_subplot(gs[0, i])
        ft.pct_hist(ax, series, PLOT_ORDER, bins)
        ax.set_xlabel(lab)
        if cap is not None:
            ticks = list(np.arange(0, cap, cap / 4)) + [cap + w / 2]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:g}" for t in ticks[:-1]] + [f"{cap:g}+"])
        ft.add_panel_label(ax, f"({'ABCD'[i]})")
        ax.set_ylim(0, 30)
        ax.set_yticks(range(0, 60, 10))
        if i == 0:
            ax.set_ylabel("Percentage (%)")
            ax.legend()
        else:
            ax.set_yticklabels([]); ax.set_ylabel("")
    ft.savefig(fig, "target_complexity_metrics", subdir="SI"); plt.show()


# SI: most frequent N- / O-heterocyclic ring systems in CbyD targets -----
def ring_systems(mol):
    """Atom-index sets of the ring systems (rings sharing >= 2 atoms are merged)."""
    systems = []
    for r in map(set, mol.GetRingInfo().AtomRings()):
        for s in [s for s in systems if len(s & r) >= 2]:
            systems.remove(s); r |= s
        systems.append(r)
    return systems


def ring_system_smiles(mol, atoms):
    """Canonical, stereo-free SMILES of one ring system (+ exocyclic =X atoms); each cut bond is capped with hydrogens."""
    keep = set(atoms)
    for i in atoms:
        for b in mol.GetAtomWithIdx(i).GetBonds():
            if b.GetBondType() == Chem.BondType.DOUBLE:
                keep.add(b.GetOtherAtomIdx(i))
    em = Chem.RWMol(mol)
    for at in em.GetAtoms():
        if at.GetIdx() in keep:
            lost = sum(int(round(b.GetBondTypeAsDouble())) for b in at.GetBonds()
                       if b.GetOtherAtomIdx(at.GetIdx()) not in keep)
            at.SetNumExplicitHs(at.GetNumExplicitHs() + lost)
    for i in sorted((i for i in range(em.GetNumAtoms()) if i not in keep), reverse=True):
        em.RemoveAtom(i)
    m = em.GetMol()
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)


def heterocycle_counts(smiles):
    """Count, over the unique targets in ``smiles``, how many contain each N- and O-heterocyclic ring system (once per target). Returns ({"N": Counter, "O":
    Counter}, number of unique targets)."""
    targets = sorted({Chem.MolToSmiles(m) for s in smiles.dropna()
                      if (m := Chem.MolFromSmiles(s)) is not None})
    counts = {"N": collections.Counter(), "O": collections.Counter()}
    n_fail = 0
    for smi in targets:
        m = Chem.MolFromSmiles(smi)
        seen = {"N": set(), "O": set()}
        for sys_atoms in ring_systems(m):
            els = {m.GetAtomWithIdx(i).GetSymbol() for i in sys_atoms}
            if not els & {"N", "O"}:
                continue
            key = ring_system_smiles(m, sys_atoms)
            if key is None:
                n_fail += 1; continue
            for el in ("N", "O"):
                if el in els:
                    seen[el].add(key)
        for el in ("N", "O"):
            counts[el].update(seen[el])
    print(f"{len(targets):,} unique CbyD targets; {len(counts['N']):,} distinct N- and "
          f"{len(counts['O']):,} distinct O-heterocyclic ring systems ({n_fail} failed)")
    return counts, len(targets)


def plot_top_heterocycles(groups, n_total, top=15, cols_per_group=2, pad=0.07,
                          name="top15_heterocycles"):
    """One panel per ``(label, title, counter)`` in ``groups``; each panel's top-``top`` ranking runs down ``cols_per_group`` columns of horizontal bars, every bar headed by its ring system drawn in the house style. Rows are sized by the structure's ink box;
    the rank split between columns balances column heights. Returns the counts table."""
    cols, dims = [], {}                              # (group index, items of this column)
    for g, (_, _, counter) in enumerate(groups):
        items = counter.most_common(top)
        dims.update({s: dr.mol_vector_paths(s, tight=True)[1:] for s, _ in items})  # ink (w, h), pt
        hs = [dims[s][1] / 72 + pad for s, _ in items]
        cuts = min(itertools.combinations(range(1, len(items)), cols_per_group - 1),
                   key=lambda c: max(sum(hs[a:b]) for a, b in zip((0,) + c, c + (len(items),))))
        for a, b in zip((0,) + cuts, cuts + (len(items),)):
            cols.append((g, items[a:b]))
    row_h = lambda s: dims[s][1] / 72 + pad
    H = max(sum(row_h(s) for s, _ in col) for _, col in cols)
    mol_w = [max(dims[s][0] for s, _ in col) / 72 + 0.06 for _, col in cols]
    top_m, bot_m, left, right = 0.22, 0.34, 0.04, 0.2
    gap, group_gap = 0.2, 0.35                       # between columns / between panels
    W = ft.FIG_WIDTH
    n_group_gaps = len(groups) - 1
    ax_w = (W - sum(mol_w) - (len(cols) - 1 - n_group_gaps) * gap - n_group_gaps * group_gap
            - left - right) / len(cols)
    FH = H + top_m + bot_m
    fig = plt.figure(figsize=(W, FH))
    xmax = {g: max(c for gg, col in cols if gg == g for _, c in col) * 1.7
            for g in range(len(groups))}
    x0, prev_g = left, None
    for (g, col), mw in zip(cols, mol_w):
        if prev_g is not None:
            x0 += group_gap if g != prev_g else gap
        if g != prev_g:                              # panel label + title above its first column
            label, title, _ = groups[g]
            fig.text(x0 / W, 1 - 0.04 / FH, label, fontsize=ft.PANEL_LABEL_PT,
                     fontweight="bold", ha="left", va="top")
            fig.text((x0 + 0.3) / W, 1 - 0.05 / FH, title, fontsize=ft.FONT_PT,
                     ha="left", va="top")
        x0 += mw
        ax = fig.add_axes([x0 / W, bot_m / FH, ax_w / W, H / FH])
        x0 += ax_w
        prev_g = g
        ax.set_ylim(H, 0); ax.set_xlim(0, xmax[g])
        hs = np.array([row_h(s) for s, _ in col])
        ys = np.cumsum(hs) - hs / 2                  # row centres (in)
        ax.barh(ys, [c for _, c in col], height=0.13, color=ft.color("CbyD"), zorder=3)
        tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        for y, (smi, c) in zip(ys, col):
            dr.draw_mol_mpl(ax, smi, (-0.03, y), transform=tr, ha="right", tight=True)
            ax.text(c + xmax[g] * 0.015, y, f"{c} ({c / n_total * 100:.1f}%)",
                    va="center", ha="left", fontsize=ft.FONT_PT)
        ax.set_yticks([]); ax.grid(False, axis="y")
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_xlabel("# of CbyD targets", labelpad=2)
    ft.savefig(fig, name, subdir="SI"); plt.show()
    return pd.concat({title: pd.DataFrame(c.most_common(top), columns=["ring_system", "n_targets"])
                      for _, title, c in groups}).assign(
        pct=lambda d: (d["n_targets"] / n_total * 100).round(2))
