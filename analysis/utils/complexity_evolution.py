"""Section 5 — complexity evolution along CbyD routes, and key steps: a reaction whose complexity change is in the dataset-wide top 10%.
"""
from __future__ import annotations
import io
import json
import multiprocessing as mp
import os
import re
import subprocess
import tempfile
import textwrap
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import ScaledTranslation
from rdkit import Chem, RDLogger

from . import data_loaders as dl, figure_template as ft, route_tree_svg as rts
from .complexity import clean_score
from .data_loaders import opt_csv
from .paths import a, CBYD_ROUTES_DIR, FIGURE_DIR
from .route_topology import _get_build_network

RDLogger.DisableLog("rdApp.*")
RAW = ["boettcher", "sascore", "sps", "npscore"]
METRICS = [("boettcher", "Böttcher's score"), ("sascore", "SAScore"),
           ("sps", "SPS"), ("npscore", "NPScore")]
# Böttcher / SPS are additive (grow with size): Δ = C - (A + B);
# SAScore / NPScore are intensive (per molecule): Δ = C - max(A, B)
DELTA_AGG = {"boettcher": sum, "sps": sum, "sascore": max, "npscore": max}
BUILD_PCTL = 0.90                                   # key step = dataset-wide top 10% of Δ
# protecting-group formation NameRxn files outside class 5: never a key step
ACETAL_PG = {"4.2.20",   # Dioxolane synthesis
             "4.2.39",   # 1,3-Dioxane synthesis
             "1.7.30",   # Acetal formation
             "1.7.31",   # Ketal formation
             "4.3.30"}   # 1,3-Dithiane synthesis
EVO_BINS = np.linspace(0, 1, 21)                     # 5% progress bins for the mean line
TREE_ROLE = {"start": "#2ca02c", "intermediate": "#1f77b4", "target": "#d62728"}
TREE_RXN = {True: ("#d62728", "#7f1d1d"), False: ("#ffd400", "#b39400")}   # key step? -> (fill, edge)
SCORE_CACHE = a("complexity", "cbyd_reactant_complexity.csv")    # smiles -> 4 scores
NPROC = min(32, os.cpu_count() or 1)
build_network = _get_build_network()


# Per-reaction table -----
def _can(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m) if m else s


def _can_flat(s):                                   # stereo-free key (2nd matching pass)
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else s


def _depths(net):
    """synthesis index -> reaction depth (reactions from the target, inclusive)."""
    adj = defaultdict(set)
    for e in net["edges"]:
        adj[e["source"]].add(e["target"]); adj[e["target"]].add(e["source"])
    ntype = {n["id"]: n["type"] for n in net["nodes"]}
    rindex = {n["id"]: n.get("index") for n in net["nodes"] if n["type"] == "reaction"}
    depth = {}
    for t in (n["id"] for n in net["nodes"] if n["type"] == "molecule" and n.get("role") == "target"):
        seen, q = {t}, deque([(t, 0)])
        while q:
            node, rx = q.popleft()
            for nb in adj[node]:
                if nb in seen:
                    continue
                seen.add(nb)
                nrx = rx + (ntype.get(nb) == "reaction")
                if ntype.get(nb) == "reaction":
                    depth[rindex[nb]] = nrx
                q.append((nb, nrx))
    return depth


def _route_rows(route_name):
    """One row per in-tree reaction of the current route JSON."""
    try:
        net = build_network(dl.load_cbyd_synthesis(route_name), align=False)
    except Exception:
        return []
    nodes = {n["id"]: n for n in net["nodes"]}
    reacts, prod = defaultdict(list), {}
    for e in net["edges"]:
        s, t = nodes[e["source"]], nodes[e["target"]]
        if s["type"] == "molecule" and t["type"] == "reaction":
            reacts[e["target"]].append(e["source"])
        elif s["type"] == "reaction" and t["type"] == "molecule":
            prod[e["source"]] = t["smiles"]
    depth = _depths(net)
    lls = max(depth.values()) if depth else 0
    rows = {}
    for rxid, p in prod.items():
        idx = nodes[rxid]["index"]
        if idx in rows or idx not in depth:          # tree-duplicated node / outside the target tree
            continue
        rows[idx] = dict(route=route_name, synthesis_index=int(idx), depth=depth[idx], lls=lls,
                         n_reactions=len(depth), smiles=p,
                         reactants=[nodes[m]["smiles"] for m in reacts.get(rxid, [])])
    return list(rows.values())


def _load_snapshot(routes):
    """Precomputed reactions (id -> route, product complexity, NameRxn), with each snapshot route mapped to its current file by (target name, year, author)."""
    snap = (pd.read_json(a("reactions", "cbyd_reactions.jsonl"), lines=True)
              [["id", "route", "target_name", "year", "author"]]
              .merge(opt_csv("complexity", "cbyd_products_complexity.csv")[["id", "smiles"] + RAW], on="id")
              .merge(opt_csv("reaction_class", "cbyd_namerxn.csv"), on="id", how="left"))
    with mp.Pool(NPROC) as pool:
        snap["can"] = pool.map(_can, snap["smiles"]); snap["can_flat"] = pool.map(_can_flat, snap["smiles"])
    key = lambda *v: tuple(re.sub(r"[^a-z0-9]", "", str(x).lower()) for x in v)
    by_key = defaultdict(list)
    for f in routes:
        bg = dl.load_cbyd_background(f)
        by_key[key(bg.get("name"), bg.get("year"), bg.get("author"))].append(f)
    old = snap.drop_duplicates("route")
    old2new = {r: by_key[k][0] for r, k in zip(old["route"], map(key, old["target_name"], old["year"], old["author"]))
               if len(by_key.get(k, ())) == 1}
    snap["route"] = snap["route"].map(old2new).fillna(snap["route"])
    print(f"snapshot routes mapped to current route files: {len(old2new):,} of {len(old):,}")
    # prefer snapshot reactions that were IN the route tree (an out-of-tree duplicate can share the product SMILES, e.g. a step later deleted from the route)
    in_tree = set(opt_csv("route_topology", "cbyd_reaction_depth.csv")["id"])
    return (snap.drop(columns=["target_name", "year", "author"])
                .assign(_o=lambda d: ~d["id"].isin(in_tree)).sort_values("_o", kind="stable")
                .drop(columns="_o"))


def _score_smiles(smiles):
    """``{smiles: {score: value}}`` from the cache, scoring the missing ones (in-process if openbabel is importable, else via the ``cbyd_analysis`` env) and updating the cache."""
    scored = {}
    if os.path.exists(SCORE_CACHE):
        scored = {r["smiles"]: {m: r[m] for m in RAW} for _, r in pd.read_csv(SCORE_CACHE).iterrows()}
    missing = [s for s in smiles if s not in scored]
    if not missing:
        return scored
    cols = ["smiles"] + RAW
    try:
        from complexity_features.score_smiles import score as _score          # needs openbabel
        print(f"scoring {len(missing):,} SMILES in-process ...")
        with mp.Pool(min(16, NPROC)) as pool:
            new = pd.DataFrame(pool.map(_score, missing))[cols]
    except Exception:
        print(f"scoring {len(missing):,} SMILES via the cbyd_analysis env ...")
        script = os.path.join(os.path.dirname(__file__), "..", "complexity_features", "score_smiles.py")
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for s in missing:
                f.write(json.dumps({"smiles": s}) + "\n")
            inp = f.name
        outp = inp.replace(".jsonl", ".csv")
        py = os.path.expanduser("~/miniconda3/envs/cbyd_analysis/bin/python")
        cmd = ([py] if os.path.exists(py)
               else [os.environ.get("CONDA_EXE", "conda"), "run", "-n", "cbyd_analysis", "python"])
        subprocess.run(cmd + [script, "--input", inp, "--output", outp, "--processes", "16"], check=True)
        new = pd.read_csv(outp)[cols]
        for p in (inp, outp):
            if os.path.exists(p):
                os.remove(p)
    if os.path.exists(SCORE_CACHE):
        new = pd.concat([pd.read_csv(SCORE_CACHE)[cols], new], ignore_index=True).drop_duplicates("smiles")
    new.to_csv(SCORE_CACHE, index=False)
    return {r["smiles"]: {m: r[m] for m in RAW} for _, r in new.iterrows()}


def build_reaction_table():
    """(per-reaction table, ``{smiles: scores}`` of reactants / unmatched products)."""
    routes = sorted(f for f in os.listdir(CBYD_ROUTES_DIR) if f.endswith(".json"))
    with mp.Pool(NPROC) as pool:
        cur = pd.DataFrame([r for rows in pool.map(_route_rows, routes) for r in rows])
        cur["can"] = pool.map(_can, cur["smiles"]); cur["can_flat"] = pool.map(_can_flat, cur["smiles"])
    snap = _load_snapshot(routes)

    used, snap_id = set(), [None] * len(cur)
    for key in ("can", "can_flat"):
        lut = defaultdict(list)
        for sid, rte, k in zip(snap["id"], snap["route"], snap[key]):
            lut[(rte, k)].append(sid)
        for i, (rte, k) in enumerate(zip(cur["route"], cur[key])):
            if snap_id[i] is None:
                sid = next((x for x in lut.get((rte, k), []) if x not in used), None)
                if sid is not None:
                    snap_id[i] = sid; used.add(sid)
    cur["snap_id"] = snap_id
    cur["id"] = [sid if sid is not None else f"new-{r}-{i}"
                 for sid, r, i in zip(cur["snap_id"], cur["route"], cur["synthesis_index"])]
    df = cur.merge(snap.drop(columns=["route", "smiles", "can", "can_flat"]).rename(columns={"id": "snap_id"}),
                   on="snap_id", how="left")
    df["class_1"] = df["class_1"].astype("string")
    n_un = int(df["snap_id"].isna().sum())
    print(f"{len(df):,} in-tree reactions in {df['route'].nunique():,} current routes; "
          f"{len(df) - n_un:,} matched to the snapshot, {n_un:,} new/edited (no NameRxn class)")

    scored = _score_smiles(sorted({s for rs in df["reactants"] for s in rs}
                                  | set(df.loc[df["snap_id"].isna(), "smiles"])))
    print(f"reactant/new-product complexity cached -> {os.path.relpath(SCORE_CACHE, a())}")
    for m in RAW:                                   # unmatched products: on-the-fly scores
        fill = df["smiles"].map(lambda s: scored.get(s, {}).get(m, np.nan))
        df[m] = df[m].where(df["snap_id"].notna(), fill)
    df["step_from_start"] = df["lls"] - df["depth"] + 1          # 1 = first synthetic step
    df["frac"] = 1 - (df["depth"] - 1) / df["lls"].clip(lower=1)  # 0 start -> 1 target
    return df, scored


# Key steps -----
def key_step_candidates(df, scored):
    """(candidate steps with Δ per score, ``{score: dataset-wide 90th-percentile Δ}``).
    Excluded: no NameRxn class (added/edited after the snapshot), (de)protection and redox (classes 5-8), cyclic acetal/ketal/dithiane formation (ACETAL_PG)."""
    rows = []
    for r in df.itertuples():
        rec = {"id": r.id, "route": r.route, "synthesis_index": r.synthesis_index,
               "class_1": r.class_1, "name_rxn": r.name_rxn}
        for m in RAW:
            C = clean_score(getattr(r, m))
            vals = [clean_score(scored[s][m]) for s in r.reactants if s in scored]
            ok = vals and len(vals) == len(r.reactants) and not any(pd.isna(v) for v in vals)
            rec["delta_" + m] = C - DELTA_AGG[m](vals) if (ok and not pd.isna(C)) else np.nan
        rows.append(rec)
    steps = pd.DataFrame(rows)
    n_all = len(steps)
    steps = steps[steps["class_1"].notna()]
    steps["class_1"] = steps["class_1"].astype(str)
    steps = steps[~steps["class_1"].isin(["5", "6", "7", "8"])]
    code = steps["name_rxn"].astype(str).str.extract(r"^([\d.]+)\s")[0]
    n_pg = int(code.isin(ACETAL_PG).sum())
    steps = steps[~code.isin(ACETAL_PG)]
    steps["class_2"] = steps["name_rxn"].astype(str).str.extract(r"^(\d+\.\d+)")[0].fillna("0.0")
    thresh = {m: steps["delta_" + m].quantile(BUILD_PCTL) for m in RAW}
    print(f"{len(steps):,} candidate steps (of {n_all:,}) after excluding (de)protection/redox, "
          f"{n_pg:,} cyclic acetal/ketal formations and unclassified new steps")
    return steps, thresh


def keysteps(steps, thresh, metric, drop_class=()):
    """Key steps for one score: Δ >= the dataset-wide threshold; optionally drop NameRxn first-level classes (e.g. {"0"} = unrecognized) afterwards."""
    key = steps[steps["delta_" + metric] >= thresh[metric]]
    return key[~key["class_1"].isin(set(drop_class))]


def keystep_summary(steps, thresh):
    return pd.DataFrame({"Δ threshold": [thresh[m] for m, _ in METRICS],
                         "key steps": [len(keysteps(steps, thresh, m)) for m, _ in METRICS]},
                        index=pd.Index([n for _, n in METRICS], name="score")).round(3)


@dataclass
class RouteScores:
    """Lookups the route-level plots need."""
    mol_score: dict    # smiles -> {score: value} (products from the table, reactants scored)
    keyidx: dict       # route file -> synthesis indices of its SAScore key steps
    delta_sa: dict     # (route, synthesis index) -> ΔSAScore
    name_rxn: dict     # (route, synthesis index) -> NameRxn name


def route_scores(df, scored, steps, thresh):
    mol_score = {s: dict(v) for s, v in scored.items()}
    for r in df.itertuples():
        mol_score[r.smiles] = {m: getattr(r, m) for m in RAW}
    id2idx = dict(zip(df["id"], df["synthesis_index"]))
    ks = keysteps(steps, thresh, "sascore").copy()
    ks["synthesis_index"] = ks["id"].map(id2idx)
    keyidx = (ks.dropna(subset=["synthesis_index"]).groupby("route")["synthesis_index"]
                .apply(lambda s: {int(x) for x in s}).to_dict())
    return RouteScores(
        mol_score, keyidx,
        {(r, int(i)): d for r, i, d in zip(steps["route"], steps["synthesis_index"], steps["delta_sascore"])},
        dict(zip(zip(df["route"], df["synthesis_index"]), df["name_rxn"].astype(str))))


# Dataset-wide evolution + SI figure -----
def plot_evolution(ax, df, metric, met_name, n_routes=400, lw=2.2):
    """Per-route complexity traces (thin, 400 random routes) + mean in 5% progress bins (bold)."""
    v = df[df[metric] > -9990]
    routes = v["route"].drop_duplicates()
    for rf in routes.sample(min(n_routes, len(routes)), random_state=np.random.RandomState(0)):
        g = v[v["route"] == rf].sort_values("step_from_start")
        if len(g) >= 3:
            ax.plot(g["frac"], g[metric], color=ft.color("CbyD"), alpha=0.04, lw=0.6)
    mt = v.groupby(pd.cut(v["frac"], EVO_BINS))[metric].mean()
    ax.plot([(b.left + b.right) / 2 for b in mt.index], mt.values, color=ft.color("CbyD"), lw=lw)
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("Synthetic progress (0→1)")
    ax.set_ylabel(met_name)


def top_reactions(key, n=10):
    """Rank | Reaction | N | % (of ``key``) for the n most common NameRxn reactions."""
    rx = key["name_rxn"].value_counts().head(n)
    return pd.DataFrame({"Reaction": rx.index, "N": rx.values,
                         "%": (rx.values / len(key) * 100).round(1)},
                        index=pd.RangeIndex(1, len(rx) + 1, name="Rank"))


def _clean_name(name):                               # "2.7.2 A;6.3.6 B" -> "A + B"
    name = re.sub(r";\s*[\d.]+\s+", " + ", str(name))
    return re.sub(r"^[\d.]+\s+", "", name).strip()


def _draw_name_list(ax, tab, title=None, fontsize=6.5, wrap=26, gap_pt=2.5, linespacing=1.15):
    """Reaction names (most frequent first) stacked from the top of ``ax`` between two rules; spacing in points, so each name takes only its own lines (+ ``gap_pt``)."""
    ax.axis("off")
    at = lambda y_pt: ax.transAxes + ScaledTranslation(0, -y_pt / 72, ax.figure.dpi_scale_trans)
    rule = dict(color="black", lw=0.6, clip_on=False)
    if title:
        ax.text(0.0, 1, title, transform=at(-2), va="bottom", ha="left", fontsize=fontsize,
                fontweight="bold")
    ax.plot([0, 1], [1, 1], transform=at(0), **rule)
    y = gap_pt
    for _, t in tab.iterrows():
        txt = "\n".join(textwrap.wrap(_clean_name(t["Reaction"]), wrap))
        ax.text(0.03, 1, txt, transform=at(y), va="top", ha="left", fontsize=fontsize,
                linespacing=linespacing)
        y += fontsize * (1 + txt.count("\n") * linespacing) + gap_pt
    ax.plot([0, 1], [1, 1], transform=at(y + 0.5), **rule)


def plot_evolution_keysteps_si(df, steps, thresh, ymax={"boettcher": 1000, "sps": 2000},
                               drop=("0", "11", "12")):
    """SI ``complexity_evolution_keysteps``: one column per score — complexity evolution (top) and the 10 most frequent NameRxn key-step reactions (bottom; unrecognized and classes 11/12 removed). Also writes ``SI/top10_keysteps.csv``; returns ``{(score name, # key steps): top-10 table}``."""
    tabs = {}
    for m, name in METRICS:
        k = keysteps(steps, thresh, m, drop)
        tabs[(name, len(k))] = top_reactions(k)
    fig = plt.figure(figsize=(7, 4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.38, wspace=0.6)
    for j, ((m, name), tab, lab) in enumerate(zip(METRICS, tabs.values(), "ABCD")):
        ax = fig.add_subplot(gs[0, j])
        plot_evolution(ax, df, m, name)
        if m in ymax:
            ax.set_ylim(top=ymax[m])
        _draw_name_list(fig.add_subplot(gs[1, j]), tab, title="Top-10 named key-step reactions")
        ft.add_panel_label(ax, f"({lab})", dx=-0.32, dy=1.04)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.95, bottom=0.02)
    ft.savefig(fig, "complexity_evolution_keysteps", subdir="SI"); plt.show()
    table = pd.concat({f"{name} (n={n:,})": tab for (name, n), tab in tabs.items()}, axis=1)
    table.to_csv(os.path.join(FIGURE_DIR, "SI", "top10_keysteps.csv"))
    return tabs, table


# One route: SAScore along the longest linear sequence + snake route tree -----
def lls_path(syn):
    """Longest linear sequence of the route tree as [(reaction index | None, smiles)]:
    starting material first, target last (ties -> the larger molecule)."""
    net = build_network(syn, align=False)
    nodes = {n["id"]: n for n in net["nodes"]}
    made_by, reacts = {}, defaultdict(list)
    for e in net["edges"]:
        s_, t_ = e["source"], e["target"]
        if nodes[s_]["type"] == "reaction" and nodes[t_]["type"] == "molecule":
            made_by[t_] = s_
        elif nodes[s_]["type"] == "molecule" and nodes[t_]["type"] == "reaction":
            reacts[t_].append(s_)
    depth = {}

    def _d(m):
        if m not in depth:
            rx = made_by.get(m)
            depth[m] = 0 if rx is None else 1 + max((_d(r) for r in reacts[rx]), default=0)
        return depth[m]

    def _heavy(m):
        mol = Chem.MolFromSmiles(nodes[m]["smiles"])
        return mol.GetNumHeavyAtoms() if mol else 0
    used = {m for v in reacts.values() for m in v}
    targets = ([n for n, d in nodes.items() if d.get("role") == "target"]
               or [m for m in made_by if m not in used])
    cur, path = max(targets, key=_d), []
    while True:
        rx = made_by.get(cur)
        path.append((int(nodes[rx]["index"]) if rx else None, nodes[cur]["smiles"]))
        if rx is None or not reacts[rx]:
            break
        cur = max(reacts[rx], key=lambda m: (_d(m), _heavy(m)))
    return path[::-1]


def plot_route_sascore(ax, syn, key_indices, mol_score, metric="sascore", label="SAScore"):
    """A score along the route's LLS, styled like the tree: molecules = open circles in the role colours; each reaction = a segment, RED for a key step, YELLOW otherwise."""
    path = lls_path(syn)
    xs = np.arange(len(path))
    ys = np.array([clean_score(mol_score.get(smi, {}).get(metric, np.nan)) for _, smi in path], float)
    for i in range(1, len(path)):
        ax.plot(xs[i - 1:i + 1], ys[i - 1:i + 1], color=TREE_RXN[path[i][0] in key_indices][0],
                lw=2.0, zorder=2, solid_capstyle="round")
    for i, (x, y) in enumerate(zip(xs, ys)):
        role = "start" if i == 0 else ("target" if i == len(path) - 1 else "intermediate")
        ax.plot(x, y, "o", ms=3.6, mfc="white", mec=TREE_ROLE[role], mew=1.0, zorder=3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(-0.4, len(path) - 0.6)
    ax.set_xlabel("Step (longest linear sequence)")
    ax.set_ylabel(label)
    off = sorted(set(key_indices) - {i for i, _ in path})
    if off:
        print(f"   note: key step(s) {[k + 1 for k in off]} (tree box numbers) sit on a branch arm, not on the plotted LLS")


def save_keystep_figure_svg(route_file, rs: RouteScores, max_width=7.05, plot_w=2.3, plot_h=1.5):
    """Snake route tree (RED box = SAScore key step) ending with the route's SAScore plot, in one SVG: ``SI/keystep_sascore_tree_<route>.svg``. Returns (path, SVG)."""
    syn = dl.load_cbyd_synthesis(route_file)
    keys = rs.keyidx.get(route_file, set())
    fig, ax = plt.subplots(figsize=(plot_w, plot_h))
    plot_route_sascore(ax, syn, keys, rs.mol_score)
    fig.tight_layout(pad=0.3)
    buf = io.StringIO(); fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)

    def _style(idx):
        return ("#d62728", "#7f1d1d", 0.9, "white") if idx in keys else rts.default_rxn_style(idx)
    stem = re.sub(r"(_processed)?\.json$", "", route_file)
    slug = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    out = a("figures", "SI", f"keystep_sascore_tree_{slug}.svg")
    svg = rts.snake_tree_svg(syn, out, max_width_in=max_width, tail_svg=buf.getvalue(), rxn_style=_style)
    W, H = (float(v) for v in re.search(r'<svg width="([\d.]+)in" height="([\d.]+)in"', svg).groups())
    print(f"{stem}: SAScore key step(s) {[k + 1 for k in sorted(keys)]} (tree box numbers); {W:.2f} x {H:.2f} in -> {os.path.relpath(out, a())}")
    return out, svg
