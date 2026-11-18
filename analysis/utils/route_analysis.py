"""Section 3 — route-level analysis: CbyD vs PaRoutes and USPTO-190, with the ``convergency_dot`` figure.
"""
from __future__ import annotations
from collections import deque

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle

from . import data_loaders as dl, figure_template as ft, route_topology as rt
from .data_loaders import opt_csv

ORDER = ["USPTO-190", "PaRoutes", "CbyD"]           # legend / table order
ZORDER = {"USPTO-190": 6, "PaRoutes": 5, "CbyD": 4}  # USPTO dots in front


def load_route_topology():
    rtopo = {"CbyD":      opt_csv("route_topology", "cbyd_route_topology.csv"),
             "PaRoutes":  opt_csv("route_topology", "paroutes_route_topology.csv"),
             "USPTO-190": opt_csv("route_topology", "uspto190_route_topology.csv")}
    rtopo = {k: v for k, v in rtopo.items() if v is not None}
    rtopo["CbyD"] = rtopo["CbyD"][rtopo["CbyD"]["error"].isna()]
    return rtopo


def route_length_summary(rtopo):
    """Median / mean reactions per route and the convergent share, per dataset."""
    return pd.DataFrame({k: {"routes": len(rtopo[k]),
                             "convergent": int(rtopo[k]["is_convergent"].astype("boolean").sum()),
                             "median # reactions": np.median(rtopo[k]["n_reactions"].dropna()),
                             "mean # reactions": np.mean(rtopo[k]["n_reactions"].dropna())}
                         for k in ORDER if k in rtopo}).T.round(1)


def example_routes(cb):
    """(convergent example, linear example) rows of the CbyD topology table: the second-largest convergent route with LLS 20-35 and 50-60 reactions, and the hand-picked linear arboduridine route (Ma et al., Angew. Chem. Int. Ed. 2024)."""
    lin = cb.loc[cb["route"] == "Arboduridine_2024_Ma.json"].iloc[0]
    cc = cb[(cb["is_convergent"] == True)
            & cb["longest_linear_sequence"].between(20, 35) & cb["n_reactions"].between(50, 60)]
    conv = cc.sort_values("n_reactions", ascending=False).iloc[1]
    return conv, lin


# Route-tree cartoons -----
def _route_graph(route_file):
    net = rt._get_build_network()(dl.load_cbyd_synthesis(route_file), align=False)
    G = nx.DiGraph(); role = {}
    for n in net["nodes"]:
        G.add_node(n["id"]); role[n["id"]] = "rxn" if n["type"] == "reaction" else n.get("role")
    for e in net["edges"]:
        G.add_edge(e["source"], e["target"])
    tg = [n for n in G.nodes if role.get(n) == "target"]
    return G, role, (tg[0] if tg else list(G.nodes)[0])


def _tidy_xy(G, root):
    """Depth + tidy spread (a node sits at the mean of its children, so a molecule backbone with one child stays a straight line)."""
    UG = G.to_undirected()
    parent = {root: None}; childs = {root: []}; seen = {root}; order = [root]; q = deque([root])
    while q:
        u = q.popleft()
        for w in sorted(UG.neighbors(u)):
            if w not in seen:
                seen.add(w); parent[w] = u; childs.setdefault(u, []).append(w)
                childs.setdefault(w, []); order.append(w); q.append(w)
    depth = {root: 0}
    for u in order:
        if parent[u] is not None:
            depth[u] = depth[parent[u]] + 1
    x = {}; cnt = [0]

    def assign(u):
        ch = childs.get(u, [])
        if not ch:
            x[u] = cnt[0]; cnt[0] += 1
        else:
            for c in ch:
                assign(c)
            x[u] = sum(x[c] for c in ch) / len(ch)
    assign(root)
    return depth, x


def draw_route(route_file, ax, orient="h"):
    """Tidy route tree; 'h' = target on the LEFT growing right, 'v' = target on top.
    Returns the target (red dot) position in data coordinates."""
    G, role, root = _route_graph(route_file)
    depth, spread = _tidy_xy(G, root)
    pos = ({n: (depth[n], spread[n]) for n in G.nodes} if orient == "h"
           else {n: (spread[n], -depth[n]) for n in G.nodes})
    styles = {"rxn": ("#cccccc", "s", 3), "start": (ft.ROLE_COLORS["start"], "o", 7),
              "intermediate": (ft.ROLE_COLORS["intermediate"], "o", 5),
              "target": (ft.ROLE_COLORS["target"], "o", 16),
              "byproduct": (ft.ROLE_COLORS["byproduct"], "o", 5)}
    for r, (col, sh, sz) in styles.items():
        nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes if role.get(n) == r],
                               node_color=col, node_shape=sh, node_size=sz, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=False, width=0.3, edge_color="0.6")
    ax.axis("off"); ax.margins(0.05)
    return pos[root]


# Main figure: convergency_dot -----
def _convergent_table(ax, rtopo, order):
    """Boxed mini-table: [colour dot] dataset | convergent / total routes."""
    bx0, bw, bh = 0.015, 0.27, 0.225
    dot_x = bx0 + 0.022
    tx0, tx1 = bx0 + 0.045, bx0 + bw - 0.012
    ty, dy, fs = 0.965, 0.05, 7
    ax.add_patch(Rectangle((bx0, 0.76), bw, bh, transform=ax.transAxes,
                           fc="white", ec="0.7", lw=0.6, alpha=0.95, zorder=8))
    ax.text(tx1, ty, "convergent / total", transform=ax.transAxes, va="top", ha="right",
            fontweight="bold", fontsize=fs, zorder=9)
    for i, k in enumerate(order):
        yy = ty - (i + 1) * dy
        conv = rtopo[k]["is_convergent"].astype("boolean")
        ax.scatter([dot_x], [yy - 0.013], transform=ax.transAxes, s=14,
                   color=ft.color(k), edgecolors="none", zorder=9)
        ax.text(tx0, yy, k, transform=ax.transAxes, va="top", ha="left", fontsize=fs, zorder=9)
        ax.text(tx1, yy, f"{int(conv.sum())}/{len(rtopo[k])}",
                transform=ax.transAxes, va="top", ha="right", fontsize=fs, zorder=9)


def plot_convergency(rtopo, conv, lin):
    """Total reactions vs longest linear sequence per route, with the convergent / total table and the two example routes as tree insets (arrow from their dot)."""
    order = [k for k in ORDER if k in rtopo]
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(1, 1, 1)
    for k in order:
        v = rtopo[k]
        ax.scatter(v["longest_linear_sequence"], v["n_reactions"], s=7, zorder=ZORDER[k],
                   color=ft.color(k), label=k, edgecolors="none", alpha=0.6)
    ax.plot([0, 140], [0, 140], "k--", lw=0.8, alpha=0.6, zorder=2)
    ax.set_xlim(0, 100); ax.set_ylim(0, 140)
    ax.set_xlabel("longest linear sequence"); ax.set_ylabel("# of total reactions")
    _convergent_table(ax, rtopo, order)

    insets = []
    for row, box, kind in [(conv, [0.4, 0.6, 0.6, 0.35], "convergent"),
                           (lin, [0.33, 0.03, 0.5, 0.15], "linear")]:
        ins = ax.inset_axes(box)
        tpos = draw_route(row["route"], ins, "h")
        ins.text(0.5, 1.06 if kind == "convergent" else 1.05, f"{kind} · {row['target_name']}",
                 transform=ins.transAxes, ha="center", va="bottom", fontsize=5.5)
        insets.append((row, ins, tpos))
    for row, ins, tpos in insets:                    # highlighted dot -> the route's red target
        xx, yy = float(row["longest_linear_sequence"]), float(row["n_reactions"])
        ax.scatter([xx], [yy], s=45, facecolors="none", edgecolors="black", linewidths=1.0, zorder=10)
        fig.add_artist(ConnectionPatch(xyA=(xx, yy), coordsA=ax.transData, xyB=tpos,
                                       coordsB=ins.transData, arrowstyle="->", color="0.35",
                                       lw=0.7, mutation_scale=8, shrinkA=4, shrinkB=7, zorder=9))
    ft.savefig(fig, "convergency_dot"); plt.show()
