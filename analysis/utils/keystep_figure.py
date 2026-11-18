"""Main-text figure ``sascore_keysteps``: SAScore vs synthetic progress, plus example cards showing one key step drawn in its route.
"""
from __future__ import annotations
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.textpath import TextPath
from rdkit import Chem
from rdkit.Chem import rdAbbreviations

from . import data_loaders as dl, draw as dr, figure_template as ft
from .paths import rel
from .complexity import clean_score
from .complexity_evolution import (RouteScores, TREE_ROLE, TREE_RXN, build_network, lls_path,
                                   plot_evolution, plot_route_sascore)

MAIN_BOND_PT = 0.13 * 72                              # largest bond length tried
MAIN_LINE_PT = dr.LINE_WIDTH_PT
ARROW_PAD, ARROW_MIN = 0.06, 0.36                     # arrow = widest text + pad each side; min length
GAP_IN, PLUS_IN, TXT_GAP = 0.05, 0.12, 0.03           # item gap, "+" width, text-to-arrow gap (in)
LINE_H = ft.FONT_PT * 1.35 / 72                       # one line of conditions text (in)
TXT = FontProperties(family="sans-serif", size=ft.FONT_PT, math_fontfamily="custom")   # = MATHTEXT
JOURNAL = {"J. Am. Chem. Soc.": "JACS", "Nature Chem.": "Nat. Chem.", "Angew. Chem. Int. Ed.": "Angew. Chem."}
INSET_PAD = 0.03                                      # padding inside the target bubble (in)

FW = 7.0
AVAIL = FW - 0.2                                      # usable width (0.1-in side margins)
COL_W = AVAIL / 3                                     # width of the (A) column
YLAB_W = 0.36                                         # room for a plot's y label + ticks
GRID_BONDS = [b / 400 * 72 for b in range(52, 31, -1)]   # 0.13 -> 0.08 in, 0.0025-in steps
G_TITLE_H, G_XLAB_H, G_ROW_GAP, G_LEG_H, G_TOP = 0.27, 0.35, 0.10, 0.16, 0.04
SGAP = 0.08                                           # min gap between neighbouring schemes
MATHTEXT = {"mathtext.fontset": "custom", "mathtext.rm": "Arial",   # $...$ in Arial
            "mathtext.it": "Arial:italic", "mathtext.default": "regular"}
# groups drawn as labels in every structure
GRID_ABBREV = rdAbbreviations.ParseAbbreviations("""OTBS O[Si](C)(C)C(C)(C)C OTBS TBSO
OTMS O[Si](C)(C)C OTMS TMSO
Bn Cc1ccccc1 Bn Bn
CO2Me C(=O)OC CO<sub>2</sub>Me MeO<sub>2</sub>C
CO2Et C(=O)OCC CO<sub>2</sub>Et EtO<sub>2</sub>C
OEt OCC OEt EtO""")
_LEGEND = ([Line2D([], [], color=TREE_RXN[True][0], lw=2), Line2D([], [], color=TREE_RXN[False][0], lw=2)]
           + [Line2D([], [], ls="", marker="o", ms=3.6, mfc="white", mec=TREE_ROLE[r], mew=1.0)
              for r in ("start", "intermediate", "target")])


def route_title(rte):
    bg = dl.load_cbyd_background(rte)
    name, au, yr = bg.get("name"), bg.get("author"), bg.get("year")
    return f"{name} ({au}, {yr})" if name else re.sub(r"(_processed)?\.json$", "", rte)


def _ref_title(rte):
    """'(+)-isomatrine (Reisman,\\n JACS, 2022)' (journal in italics)."""
    bg = dl.load_cbyd_background(rte)
    name = re.sub(r"^((?:\([^)]*\)-)?)([A-Z])(?=[a-z])", lambda m: m.group(1) + m.group(2).lower(),
                  bg.get("name") or rte)
    m = re.match(r"\s*(.*?)\s+\d{4}\b", bg.get("citation") or "")
    jr = JOURNAL.get(m.group(1), m.group(1)) if m else ""
    jr = re.sub(r"([A-Za-z]+)", r"$\\mathit{\1}$", jr) + ", " if jr else ""
    return f"{name} ({bg.get('author')},\n{jr}{bg.get('year')})"


_text_w = lambda s: TextPath((0, 0), s, size=ft.FONT_PT, prop=TXT).get_extents().width / 72


def _mol_wh(s, bond_pt, abbrevs=None):                 # drawn size (in) of one molecule
    return tuple(v / 72 for v in dr.mol_vector_paths(s, bond_pt, line_pt=MAIN_LINE_PT, tight=True,
                                                     abbrevs=abbrevs)[1:])


def key_scheme(rte, syn, rs: RouteScores, pick=None, opts=None, bond_pt=MAIN_BOND_PT, abbrevs=None):
    """Layout of one key step on the route's LLS (the step at LLS position ``pick`` if an int, else the one whose NameRxn name matches regex ``pick``, else the largest-Δ one):
    items [(kind, payload, width_in, height_in)], total width ``w``, room ``up`` / ``dn`` above / below the arrow, LLS position ``pos`` and synthesis index ``rxn`` of the step."""
    opts = opts or {}
    key_indices = rs.keyidx.get(rte, set())
    _wh = lambda s: _mol_wh(s, bond_pt, abbrevs)
    path = lls_path(syn)
    on = [i for i, (ri, _) in enumerate(path) if ri in key_indices]
    if isinstance(pick, int):
        if path[pick][0] not in key_indices:
            print(f"   note: step {pick} of {rte} is not one of its SAScore key steps")
        on = [pick]
    elif pick:
        on = [i for i in on if re.search(pick, rs.name_rxn.get((rte, path[i][0]), ""), re.I)] or on
    i = max(on, key=lambda i: rs.delta_sa.get((rte, path[i][0]), -np.inf))
    net = build_network(syn, align=False); nodes = {n["id"]: n for n in net["nodes"]}
    rx = next(n for n in nodes.values() if n["type"] == "reaction" and int(n["index"]) == path[i][0])
    reacts = [nodes[e["source"]]["smiles"] for e in net["edges"]
              if e["target"] == rx["id"] and nodes[e["source"]]["type"] == "molecule"]
    heavy = lambda s: (Chem.MolFromSmiles(s) or Chem.MolFromSmiles("")).GetNumHeavyAtoms()
    reacts = opts.get("reactants") or [s for s in dict.fromkeys(reacts) if heavy(s) >= 3] or [path[i - 1][1]]
    am = [(s, *_wh(s)) for s in opts.get("above_mols", [])]
    reacts = [s for s in reacts if s not in {s for s, _, _ in am}]
    above, below = opts.get("above", []), opts.get("below", [])
    items = []
    for j, s in enumerate(reacts):
        if j:
            items.append(("plus", None, PLUS_IN, 0))
        items.append(("mol", s, *_wh(s)))
    aw = max([_text_w(t) for t in above + below] + [w for _, w, _ in am]
             + [ARROW_MIN - 2 * ARROW_PAD]) + 2 * ARROW_PAD
    items += [("arrow", dict(mols=am, above=above, below=below), aw, 0),
              ("mol", path[i][1], *_wh(path[i][1]))]
    mol_h = max(h for kind, _, _, h in items if kind == "mol")
    up = TXT_GAP + LINE_H * len(above) + sum(h + TXT_GAP for *_, h in am)
    dn = TXT_GAP + LINE_H * len(below)
    return dict(items=items, w=sum(it[2] for it in items) + GAP_IN * (len(items) - 1),
                up=max(mol_h / 2, up), dn=max(mol_h / 2, dn), pos=i, rxn=path[i][0],
                bond=bond_pt, abbrevs=abbrevs)


def draw_scheme(ax, sch):
    """Draw scheme ``sch`` centred in ``ax``: molecules at absolute size on the arrow line; conditions text (and reagent structures) above / below the arrow."""
    ax.axis("off")
    W = ax.get_position().width * ax.figure.get_size_inches()[0]
    H = ax.get_position().height * ax.figure.get_size_inches()[1]
    if sch["w"] > W + 1e-6:
        print(f"   note: scheme is {sch['w']:.2f} in wide, panel is {W:.2f} in")
    fx, fy = (lambda v: v / W), (lambda v: v / H)
    kw = dict(transform=ax.transAxes, tight=True, bond_len_pt=sch["bond"], line_pt=MAIN_LINE_PT,
              abbrevs=sch["abbrevs"])
    x, y0 = (W - sch["w"]) / 2, sch["dn"] + (H - sch["up"] - sch["dn"]) / 2
    for kind, p, w, h in sch["items"]:
        if kind == "mol":
            dr.draw_mol_mpl(ax, p, (fx(x), fy(y0)), ha="left", va="center", **kw)
        elif kind == "plus":
            ax.text(fx(x + w / 2), fy(y0), "+", transform=ax.transAxes, ha="center", va="center",
                    fontsize=8)
        else:
            ax.annotate("", xy=(fx(x + w), fy(y0)), xytext=(fx(x), fy(y0)), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8,
                                        mutation_scale=7, shrinkA=0, shrinkB=0))
            cx, y = fx(x + w / 2), y0 + TXT_GAP
            for t in reversed(p["above"]):               # bottom line sits just over the arrow
                ax.text(cx, fy(y), t, transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=ft.FONT_PT)
                y += LINE_H
            for s, _, mh in reversed(p["mols"]):
                dr.draw_mol_mpl(ax, s, (cx, fy(y)), ha="center", va="bottom", **kw)
                y += mh + TXT_GAP
            y = y0 - TXT_GAP
            for t in p["below"]:
                ax.text(cx, fy(y), t, transform=ax.transAxes, ha="center", va="top",
                        fontsize=ft.FONT_PT)
                y -= LINE_H
        x += w + GAP_IN


def _target_inset(ax, syn, mol_score, style, abbrevs=None, grows=None):
    """The route's target in a red-edged bubble inside ``ax``, at the free spot nearest the target point (growing the y range if nothing is free), with a thin red line from the target point. ``grows`` = y-range extensions to try (fractions of it)."""
    path = lls_path(syn)
    smi = path[-1][1]
    style = dict(style, tight=True, abbrevs=abbrevs)
    _, w_pt, h_pt = dr.mol_vector_paths(smi, style["bond_len_pt"], style["font_pt"], style["line_pt"],
                                        tight=True, abbrevs=abbrevs)
    fw, fh = ax.figure.get_size_inches()
    W, H = ax.get_position().width * fw, ax.get_position().height * fh
    bw, bh = (w_pt / 72 + 2 * INSET_PAD) / W, (h_pt / 72 + 2 * INSET_PAD) / H
    ys = np.array([clean_score(mol_score.get(s, {}).get("sascore", np.nan)) for _, s in path], float)
    xs, t = np.arange(len(path), dtype=float), np.linspace(0, 1, 15)
    line = np.concatenate([np.c_[xs[i] + t, ys[i] + t * (ys[i + 1] - ys[i])] for i in range(len(xs) - 1)])
    line = line[~np.isnan(line).any(axis=1)]
    y0, y1 = ax.get_ylim()
    ticks = [v for v in ax.get_yticks() if y0 - 1e-9 <= v <= y1 + 1e-9]   # kept if the range grows
    grows = grows or [bh / (1 - bh) * 1.1]
    ranges = [(y0, y1)] + [r for g in grows for r in ((y0, y1 + g * (y1 - y0)), (y0 - g * (y1 - y0), y1))]
    best = None
    for n_exp, (lo, hi) in enumerate(ranges):
        ax.set_ylim(lo, hi)
        pts = ax.transAxes.inverted().transform(ax.transData.transform(line))
        end = ax.transAxes.inverted().transform(ax.transData.transform((xs[-1], ys[-1])))
        away = pts[np.hypot(*(pts - end).T) > 0.05]       # the line, minus the target point
        for bx in np.arange(0.02, 0.98 - bw + 1e-9, 0.02):
            for by in np.arange(0.03, 0.97 - bh + 1e-9, 0.02):
                hit = np.sum((pts[:, 0] > bx - 0.03) & (pts[:, 0] < bx + bw + 0.03)
                             & (pts[:, 1] > by - 0.04) & (pts[:, 1] < by + bh + 0.04))
                cx_, cy_ = min(max(end[0], bx), bx + bw), min(max(end[1], by), by + bh)
                lead = end + np.linspace(0, 1, 12)[1:, None] * (np.array([cx_, cy_]) - end)
                hit += np.sum(np.hypot(away[:, None, 0] - lead[None, :, 0],     # the red line to
                                       away[:, None, 1] - lead[None, :, 1]).min(axis=1) < 0.03)  # the bubble crosses the data
                key = (hit, (n_exp + 1) // 2, np.hypot(bx + bw / 2 - end[0], by + bh / 2 - end[1]))
                if best is None or key < best[0]:
                    best = (key, lo, hi, bx, by)
    (hit, _, _), lo, hi, bx, by = best
    if hit:
        print(f"   note: the target bubble overlaps the line ({hit} sampled points)")
    ax.set_ylim(lo, hi)
    ax.set_yticks(ticks)
    end = ax.transAxes.inverted().transform(ax.transData.transform((xs[-1], ys[-1])))
    ex_, ey_ = min(max(end[0], bx), bx + bw), min(max(end[1], by), by + bh)
    ax.plot([end[0], ex_], [end[1], ey_], transform=ax.transAxes, color=TREE_ROLE["target"],
            lw=0.5, zorder=2.5, clip_on=False)
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=0.025",
                                transform=ax.transAxes, fc="white", ec=TREE_ROLE["target"],
                                lw=0.6, zorder=4, clip_on=False))
    dr.draw_mol_mpl(ax, smi, (bx + bw / 2, by + bh / 2), transform=ax.transAxes, ha="center",
                    va="center", zorder=5, **style)


def _pack(widths, centres, lo, hi):
    """Left edges for schemes of ``widths`` as close as possible to ``centres``, in order, without overlapping (gap SGAP) or leaving [lo, hi]."""
    left = []
    for w, c in zip(widths, centres):
        left.append(max(c - w / 2, lo if not left else left[-1] + widths[len(left) - 1] + SGAP))
    for k in range(len(left) - 1, -1, -1):
        right_lim = hi if k == len(left) - 1 else left[k + 1] - SGAP
        left[k] = min(left[k], right_lim - widths[k])
    return left


def grid_figure(rows, examples, df, rs: RouteScores, out, plot_h=1.3):
    """Draw and save the figure. ``examples`` = [(route file, LLS position or NameRxn regex, scheme options)]; ``rows`` = lists of route files, one per card row (the first row sits next to (A)); ``df`` = the per-reaction table (for panel A)."""
    with plt.rc_context(MATHTEXT):
        _grid_figure(rows, examples, df, rs, out, plot_h)


def _grid_figure(rows, examples, df, rs, out, plot_h):
    ex = [[next(e for e in examples if e[0] == r) for r in row] for row in rows]
    syn = {rte: dl.load_cbyd_synthesis(rte) for row in ex for rte, _, _ in row}
    spans = [(0.1 + COL_W if k == 0 else 0.1, 0.1 + AVAIL) for k in range(len(rows))]   # x range of the cards
    for bond in GRID_BONDS:                            # largest bond at which every row fits
        sch = [[key_scheme(rte, syn[rte], rs, pick, opts, bond_pt=bond, abbrevs=GRID_ABBREV)
                for rte, pick, opts in row] for row in ex]
        if all(hi - lo - sum(s["w"] for s in r) - SGAP * (len(r) - 1) >= 0 for r, (lo, hi) in zip(sch, spans)):
            break
    else:
        raise ValueError(f"the rows do not fit even at {GRID_BONDS[-1] / 72:.4f}-in bonds")
    print(f"all molecules at {bond / 72:.4f}-in bonds")
    sh = [max(s["up"] + s["dn"] + 0.04 for s in r) for r in sch]
    card = [G_TITLE_H + plot_h + G_XLAB_H + h for h in sh]
    FH = G_TOP + G_LEG_H + sum(card) + G_ROW_GAP * (len(rows) - 1) + 0.02
    fig = plt.figure(figsize=(FW, FH))
    axes = lambda x, y_top, w, h: fig.add_axes([x / FW, (FH - y_top - h) / FH, w / FW, h / FH])
    xy = lambda x, y_top: (x / FW, 1 - y_top / FH)     # inches (from top-left) -> figure fraction
    label = lambda x, y_top, t: fig.text(*xy(x, y_top), t, fontsize=ft.PANEL_LABEL_PT,
                                         fontweight="bold", ha="left", va="top")
    fig.legend(_LEGEND, ["key step", "other step", "starting material", "intermediate", "target"],
               loc="upper left", bbox_to_anchor=xy(0.1 + COL_W + 0.34, G_TOP - 0.02), ncol=5,
               frameon=False, fontsize=ft.FONT_PT, handlelength=1.4, handletextpad=0.35,
               columnspacing=1.0, borderaxespad=0, borderpad=0)
    label(0.1, G_TOP - 0.02, "(A)")
    label(0.1 + COL_W + 0.04, G_TOP - 0.02, "(B)")
    style = dict(bond_len_pt=bond, font_pt=dr.MPL_FONT_PT, line_pt=MAIN_LINE_PT)   # = the schemes
    y = G_TOP + G_LEG_H
    for k, (row, srow, (lo, hi)) in enumerate(zip(ex, sch, spans)):
        cw = (hi - lo) / len(row)                      # card width
        lefts = _pack([s["w"] for s in srow], [lo + (j + 0.5) * cw for j in range(len(row))], lo, hi)
        for j, ((rte, _, _), s, sl) in enumerate(zip(row, srow, lefts)):
            ax = axes(lo + j * cw + YLAB_W, y + G_TITLE_H, cw - YLAB_W - 0.06, plot_h)
            plot_route_sascore(ax, syn[rte], {s["rxn"]}, rs.mol_score)   # red = the drawn step only
            ax.set_xlabel("Step", labelpad=1)
            ax.set_title(_ref_title(rte), fontsize=ft.FONT_PT, pad=3, linespacing=1.1)
            _target_inset(ax, syn[rte], rs.mol_score, style, abbrevs=GRID_ABBREV,
                          grows=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0))
            draw_scheme(axes(sl, y + G_TITLE_H + plot_h + G_XLAB_H, s["w"], sh[k]), s)
            print(f"   {route_title(rte)}: step {s['pos']}; scheme {s['w']:.2f} in")
        if k == 0:                                     # (A) next to the first row
            a_bottom = y + card[0]
            axA = axes(0.1 + YLAB_W, G_TOP + 0.16, COL_W - YLAB_W - 0.12, a_bottom - G_XLAB_H - G_TOP - 0.16)
            plot_evolution(axA, df, "sascore", "SAScore")
        y += card[k] + G_ROW_GAP
    xr, yb = 0.1 + COL_W - 0.02, a_bottom + G_ROW_GAP / 2   # light-grey L-shaped rule around (A)
    (x0, y0), (x1, y1), (x2, y2) = xy(xr, 0.02), xy(xr, yb), xy(0.02, yb)
    fig.add_artist(Line2D([x0, x1, x2], [y0, y1, y2], color="#c8c8c8", lw=0.8, transform=fig.transFigure))
    fig.savefig(out)                                   # exact 7-in page
    print(f"saved -> {rel(out)}  ({FW:.2f} x {FH:.2f} in)")
    plt.show()
