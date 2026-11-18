"""Source data for Table 1, Table 2 and the main figures, plus the Word file of supplementary tables.
"""
from __future__ import annotations
import os

import numpy as np
import pandas as pd

from . import complexity as cx, complexity_evolution as ce, data_loaders as dl
from . import keystep_figure as kf, overview as ov, reaction_complexity as rc
from . import starting_materials as sm
from .paths import rel
from .table_docx import fixed, tables_to_docx

MET_NAME = {"sascore": "SAScore", "npscore": "NPScore", "sps": "SPS (unnormalised)",
            "boettcher": "Böttcher score", "overall_product_complexity": "Overall target complexity",
            "mw": "Molecular weight", "n_stereo": "Tetrahedral stereocentres"}


class SourceData:
    """Registry of the source-data tables and their index. Values are kept unrounded and rounded once, when the Word tables are written (rounding here first would round twice, e.g. 7.6499 -> 7.650 -> 7.7)."""

    def __init__(self):
        self.tables, self.rows = {}, []

    def keep(self, df, name, figure, desc, index=False):
        self.tables[name] = df.reset_index() if index else df
        self.rows.append({"table": name, "figure": figure, "description": desc, "rows": len(df)})
        return df

    def index(self):
        return pd.DataFrame(self.rows)


# Generic summaries -----
def describe(series_by_ds, order):
    """n, mean, sd, median, quartiles, min, max per dataset."""
    rows = []
    for d in order:
        v = pd.Series(series_by_ds[d], dtype=float).dropna()
        rows.append({"dataset": d, "n": len(v), "mean": v.mean(), "sd": v.std(),
                     "median": v.median(), "q1": v.quantile(0.25), "q3": v.quantile(0.75),
                     "min": v.min(), "max": v.max()})
    return pd.DataFrame(rows)


def cat_pct(series_by_ds, cap, order):
    """Integer values -> % per category 0..cap (top category = '>= cap')."""
    cats = list(range(cap + 1))
    out = {d: (pd.Series(series_by_ds[d]).astype(int).clip(upper=cap)
               .value_counts(normalize=True) * 100).reindex(cats, fill_value=0).values
           for d in order}
    return pd.DataFrame({"value": [str(c) for c in cats[:-1]] + [f"{cap}+"], **out})


def hist_pct(series_by_ds, edges, order, cap=None, labels=None):
    """A histogram as plotted: % of the dataset per bin (values >= cap in the top bin)."""
    out = {}
    for d in order:
        v = np.asarray(series_by_ds[d], float); v = v[np.isfinite(v)]
        if cap is not None:
            v = np.minimum(v, cap)
        out[d] = np.histogram(v, bins=edges)[0] / len(v) * 100
    labels = labels or [f"{lo:g}–{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])]
    return pd.DataFrame({"bin": labels, **out})


def fmt_med(t, d, nd=0):
    """'median (q1–q3)' of dataset d in a :func:`describe` table."""
    r = t.set_index("dataset").loc[d]
    f = (lambda x: fixed(x, nd))
    return f"{f(r['median'])} ({f(r['q1'])}–{f(r['q3'])})"


# Per-section tables -----
def overview_tables(book, table1, topo, years, routes_counts, top5, targets, comp):
    ds = [d for d in ov.TARGET_DATASETS if d in comp]
    by = lambda col: cx.by_dataset(targets, col)
    book.keep(table1, "table1_scope", "Table 1",
              "Scope of CbyD: routes, reactions, unique targets (natural products / pharmaceuticals)")
    y = pd.Series(np.maximum(years, 1940).astype(int)).value_counts().sort_index()   # <=1940 pooled, as in the figure
    book.keep(pd.DataFrame({"publication_year": ["≤1940" if v == 1940 else str(v) for v in y.index],
                            "n_routes": y.values}),
              "routes_by_year", "Dataset overview (A)",
              f"Routes per publication year; {len(topo) - len(years)} routes without a publication year not shown")
    book.keep(routes_counts.rename_axis("routes_per_target").reset_index(name="n_targets"),
              "routes_per_target", "Dataset overview (B)",
              "Number of targets with k routes (targets grouped by name)")
    book.keep(top5.reset_index(drop=True)[["name", "n_routes"]], "top5_targets", "Dataset overview (B)",
              "Five most frequently synthesised targets")

    props = pd.concat([describe(by(c), ds).assign(property=MET_NAME[c])
                       for c in ["overall_product_complexity", "mw", "n_stereo",
                                 "sascore", "npscore", "sps", "boettcher"]])
    book.keep(props[["property"] + [c for c in props if c != "property"]],
              "target_property_summary", "Dataset overview (C) / SI",
              "Per-dataset summary of target properties (one row per unique target)")
    hists = [("overall_product_complexity", np.linspace(0, 1, 21), None, None),
             ("mw", np.linspace(0, 1000, 21), 1000,
              [f"{lo:g}–{hi:g}" for lo, hi in zip(np.linspace(0, 950, 20), np.linspace(50, 1000, 20))][:-1]
              + ["950–1000+"]),
             ("n_stereo", np.arange(0, 17) - 0.5, 15, [str(k) for k in range(15)] + ["15+"])]
    book.keep(pd.concat([hist_pct(by(c), e, ds, cap, lab).assign(property=MET_NAME[c])
                         for c, e, cap, lab in hists])[["property", "bin"] + ds],
              "target_property_histograms", "Dataset overview (C)", "Histograms as plotted (% of targets per bin)")

    def count_rows(cols, caps, kind):            # % of ALL targets with exactly k (k+ pooled; 0 not plotted)
        rows = []
        for (col, lab), cap in zip(cols, caps):
            for k in range(1, cap + 1):
                rec = {kind: lab, "count_per_target": str(k) if k < cap else f"{k}+"}
                for d in ds:
                    v = comp[d][col].values
                    rec[d] = ((np.minimum(v, cap) == k) & (v > 0)).sum() / len(v) * 100
                rows.append(rec)
        return pd.DataFrame(rows)
    book.keep(count_rows(ov.RING_CATS, [6] * len(ov.RING_CATS), "ring_type"),
              "ring_types", "Dataset overview (D)", "% of targets with k rings of each type")
    book.keep(count_rows([(e, "Halogens (F, Cl, Br, I)" if e == "Halogens" else e) for e, _ in ov.F_ELEMS],
                         [k for _, k in ov.F_ELEMS], "element"),
              "heteroatoms", "Dataset overview (F)", "% of targets with k atoms of each element")
    junctions = pd.DataFrame({"junction": ["Fused", "Bridged", "Spiro"],
                              **{d: [comp[d][k].mean() * 100 for k in ("fused", "bridged", "spiro")]
                                 for d in ds}})
    book.keep(junctions, "ring_junctions", "Dataset overview (E)",
              "% of targets with >=1 fused / bridged / spiro ring junction (not exclusive)")
    frac = {d: {**{f"≥1 {e}": (c[e] > 0).mean() * 100 for e in ov.ELEMS if e != "C"},
                **{f"≥1 {lab.lower()}": (c[col] > 0).mean() * 100 for col, lab in ov.RING_CATS},
                "acyclic": (c["n_rings"] == 0).mean() * 100, "n targets": len(c)}
            for d, c in comp.items() if d in ds}
    book.keep(pd.DataFrame(frac).rename_axis("measure"), "composition_summary", "Dataset overview (D–F)",
              "% of targets with at least one of each element / ring type", index=True)
    return {"props": props, "junctions": junctions, "n_targets": {d: len(comp[d]) for d in ds},
            "n_year_excluded": len(topo) - len(years), "datasets": ds}


def reaction_tables(book, feat, fr, classes, metal):
    ds = [d for d in rc.DATASETS if d in feat]
    book.keep(pd.DataFrame({"class": classes, "class_name": [rc.CLASS_NAMES.get(k, k) for k in classes],
                            "plotted": [rc.CLASS_NAMES.get(k, k) != "Resolution" for k in classes],
                            **{d: [fr[d].get(k, 0) * 100 for k in classes] for d in ds}}),
              "reaction_classes", "Reaction complexity (A)", "% of reactions per NameRxn superclass")
    descriptors = [("Reacting atoms", rc.feat_series(feat, "num_react_atom_retro"), 6),
                   ("Changing rings", rc.feat_series(feat, "num_change_ring"), 4),
                   ("Stereochemical changes (R/S + E/Z)", rc.stereo_series(feat), 4),
                   ("C–C bonds formed", rc.feat_series(feat, "n_cc_bond_formed"), 4),
                   ("C–X bonds formed", rc.feat_series(feat, "n_cx_bond_formed"), 4)]
    book.keep(pd.concat([cat_pct(s, cap, ds).assign(descriptor=lab) for lab, s, cap in descriptors])
              [["descriptor", "value"] + ds],
              "reaction_descriptors", "Reaction complexity (A) / SI", "% of reactions per descriptor value")
    desc = pd.concat([describe(s, ds).assign(descriptor=lab,
                                             pct_at_least_1=[(np.asarray(s[d]) >= 1).mean() * 100 for d in ds])
                      for lab, s, _ in descriptors])
    book.keep(desc[["descriptor"] + [c for c in desc if c != "descriptor"]],
              "reaction_descriptor_summary", "Reaction complexity (A) / SI",
              "Per-dataset summary of reaction descriptors (reactions that could be parsed)")
    metals = pd.DataFrame([{"dataset": d, "rank": r, "metal": m, "count": c}
                           for d, ranked in rc.top_transition_metals(metal).items()
                           for r, (m, c) in enumerate(ranked, 1)])
    book.keep(metals, "transition_metals", "Reaction complexity (C)",
              "Ten most frequent transition metals per dataset (counts of agent fragments)")
    return {"sizes": {d: len(feat[d]) for d in feat}, "unrecognized": {d: fr[d].get("0", 0) * 100 for d in ds},
            "desc": desc, "metals": metals, "datasets": ds}


def route_tables(book, rtopo):
    ds = [d for d in ["CbyD", "USPTO-190", "PaRoutes"] if d in rtopo]
    rows, per = [], []
    for d in ds:
        t = rtopo[d]
        rec = {"dataset": d, "n_routes": len(t), "n_convergent": int(t["is_convergent"].sum()),
               "pct_convergent": t["is_convergent"].mean() * 100}
        for col, lab in (("n_reactions", "total_reactions"), ("longest_linear_sequence", "lls")):
            v = t[col]
            rec.update({f"{lab}_median": v.median(), f"{lab}_q1": v.quantile(.25),
                        f"{lab}_q3": v.quantile(.75), f"{lab}_mean": v.mean(),
                        f"{lab}_min": v.min(), f"{lab}_max": v.max()})
        rows.append(rec)
        per.append(pd.DataFrame({"dataset": d, "route": t["route"] if "route" in t else t["id"],
                                 "n_reactions": t["n_reactions"],
                                 "longest_linear_sequence": t["longest_linear_sequence"],
                                 "is_convergent": t["is_convergent"]}))
    summary = pd.DataFrame(rows)
    book.keep(summary, "route_summary", "Route analysis",
              "Route length, longest linear sequence and convergence per dataset")
    book.keep(pd.concat(per), "routes", "Route analysis", "Every plotted route: total reactions, LLS, convergent")
    return summary


def starting_material_tables(book, freq_bb, u_bb, p_bb, buy, prov):
    bb = freq_bb.reset_index().rename(columns={"canon": "smiles"})
    bb.insert(0, "rank", range(1, len(bb) + 1))
    book.keep(bb[["rank", "smiles", "n_routes", "n_occurrence"]], "chiral_building_blocks",
              "Starting materials", "All CbyD chiral building blocks, ranked by number of routes")
    for nm, bbx in (("uspto190", u_bb), ("paroutes", p_bb)):
        t = bbx.reset_index().rename(columns={"canon": "smiles"}).head(24)
        book.keep(t[["smiles", "n_routes"]], f"si_{nm}_building_blocks_top24", "SI",
                  f"Top chiral building blocks of {nm} ({len(bbx)} in total)")
    w, nb = buy["n_routes"], ~buy["buyable_norm"]
    isbb = buy["smiles"].isin(set(freq_bb.index))
    book.keep(pd.DataFrame({"measure": ["unique start molecules", "buyable, exact stereoisomer (%)",
                                        "buyable, incl. normalised forms (%)", "not buyable (%)",
                                        "not buyable, weighted by route count (%)",
                                        "chiral building blocks", "chiral building blocks not buyable (%)"],
                            "value": [len(buy), buy["buyable"].mean() * 100, buy["buyable_norm"].mean() * 100,
                                      nb.mean() * 100, w[nb].sum() / w.sum() * 100,
                                      int(isbb.sum()), nb[isbb].mean() * 100]}),
              "purchasability", "Starting materials / SI", "Purchasability of starting materials (eMolecules)")
    book.keep(sm.provenance_summary(prov).reset_index().rename(columns={"index": "class"}),
              "stereocentre_origin", "Starting materials / SI",
              f"Installed vs purchased target stereocentres ({len(prov):,} routes)")
    book.keep(sm.top_donors(prov), "stereocentre_donors", "Starting materials / SI",
              "Chiral starting materials that donate a target stereocentre most often")


def keystep_tables(book, rxn, steps, thresh, si_tabs, examples, rs):
    ev = []
    for m, name in ce.METRICS:                    # pooled statistics per 5% progress bin (bold line = mean)
        v = rxn[rxn[m] > -9990]
        g = v.groupby(pd.cut(v["frac"], ce.EVO_BINS))
        ev.append(pd.DataFrame({"metric": name,
                                "progress_bin": [f"{b.left:.2f}–{b.right:.2f}" for b in g.size().index],
                                "n_reactions": g.size().values, "n_routes": g["route"].nunique().values,
                                "mean": g[m].mean().values, "sd": g[m].std().values,
                                "median": g[m].median().values}))
    book.keep(pd.concat(ev), "complexity_by_progress", "Key steps (A) / SI",
              "Product complexity per synthetic-progress bin (all in-tree reactions)")
    book.keep(pd.DataFrame({"metric": [n for _, n in ce.METRICS],
                            "threshold_delta": [thresh[m] for m, _ in ce.METRICS],
                            "n_key_steps": [len(ce.keysteps(steps, thresh, m)) for m, _ in ce.METRICS],
                            "n_candidates": len(steps)}),
              "key_step_thresholds", "Key steps", "Key step = ΔC >= 90th percentile of candidate steps")
    book.keep(pd.concat([t.reset_index().assign(metric=name, n_key_steps=n)
                         for (name, n), t in si_tabs.items()])[["metric", "n_key_steps", "Rank", "Reaction", "N", "%"]],
              "top10_key_step_reactions", "Key steps / SI",
              "Ten most frequent NameRxn reactions among key steps (classes 0, 11, 12 removed)")
    rows = []
    for rte, *_ in examples:                       # SAScore along each example route's LLS
        keys = rs.keyidx.get(rte, set())
        for pos, (ri, smi) in enumerate(ce.lls_path(dl.load_cbyd_synthesis(rte))):
            rows.append({"route": rte.removesuffix(".json").removesuffix("_processed"),
                         "target": kf.route_title(rte), "lls_position": pos, "synthesis_index": ri,
                         "name_rxn": rs.name_rxn.get((rte, ri), "") if ri is not None else "",
                         "smiles": smi,
                         "sascore": cx.clean_score(rs.mol_score.get(smi, {}).get("sascore", np.nan)),
                         "delta_sascore": rs.delta_sa.get((rte, ri), np.nan) if ri is not None else np.nan,
                         "sascore_key_step": ri in keys})
    book.keep(pd.DataFrame(rows).astype({"synthesis_index": "Int64"}), "example_routes", "Key steps (B)",
              "SAScore of each molecule on the longest linear sequence of the example routes")


# Table 2 and the Word file -----
TABLE2_COLS = ["CbyD", "USPTO-190", "PaRoutes", "USPTO-Full", "Pistachio"]


def table2(s00, s01, s02):
    """CbyD vs the comparison datasets; medians with the IQR in parentheses, "—" = n/a."""
    p, j, d, rt = s00["props"], s00["junctions"].set_index("junction"), s01["desc"], s02.set_index("dataset")
    row = lambda label, f, ds: {"": label, **{c: (f(c) if c in ds else "—") for c in TABLE2_COLS}}
    desc = lambda lab, col, c: d[(d["descriptor"] == lab) & (d["dataset"] == c)].iloc[0][col]
    rtq = lambda c, lab: (f"{rt.loc[c, lab + '_median']:.0f} "
                          f"({rt.loc[c, lab + '_q1']:.0f}–{rt.loc[c, lab + '_q3']:.0f})")
    T, R, X = s00["datasets"], list(rt.index), s01["datasets"]
    return pd.DataFrame([
        row("Unique targets", lambda c: f"{s00['n_targets'][c]:,}", T),
        row("Molecular weight", lambda c: fmt_med(p[p.property == "Molecular weight"], c), T),
        row("Tetrahedral stereocentres", lambda c: fmt_med(p[p.property == "Tetrahedral stereocentres"], c), T),
        row("Overall target complexity", lambda c: fmt_med(p[p.property == "Overall target complexity"], c, 2), T),
        row("Targets with fused rings (%)", lambda c: fixed(j.loc['Fused', c], 1), T),
        row("Targets with bridged rings (%)", lambda c: fixed(j.loc['Bridged', c], 1), T),
        row("Targets with spiro centres (%)", lambda c: fixed(j.loc['Spiro', c], 1), T),
        row("Routes", lambda c: f"{int(rt.loc[c, 'n_routes']):,}", R),
        row("Reactions per route", lambda c: rtq(c, "total_reactions"), R),
        row("Longest linear sequence", lambda c: rtq(c, "lls"), R),
        row("Convergent routes (%)", lambda c: fixed(rt.loc[c, 'pct_convergent'], 1), R),
        row("Reactions", lambda c: f"{s01['sizes'][c]:,}", X),
        row("Unclassified by NameRxn (%)", lambda c: fixed(s01['unrecognized'][c], 1), X),
        row("Reacting atoms", lambda c: (f"{desc('Reacting atoms', 'median', c):.0f} "
                                         f"({desc('Reacting atoms', 'q1', c):.0f}–{desc('Reacting atoms', 'q3', c):.0f})"), X),
        row("Reactions changing a ring (%)", lambda c: fixed(desc('Changing rings', 'pct_at_least_1', c), 1), X),
        row("Reactions changing stereochemistry (%)",
            lambda c: fixed(desc('Stereochemical changes (R/S + E/Z)', 'pct_at_least_1', c), 1), X),
    ])


STATS = {"property": "Property", "descriptor": "Descriptor", "dataset": "Dataset", "n": "n",
         "mean": "Mean", "sd": "SD", "median": "Median", "q1": "Q1", "q3": "Q3",
         "min": "Min", "max": "Max", "pct_at_least_1": "≥1 (%)"}
ROUTE_ROWS = {"n_routes": "Routes", "n_convergent": "Convergent routes", "pct_convergent": "Convergent routes (%)",
              "total_reactions_median": "Total reactions, median", "total_reactions_q1": "Total reactions, Q1",
              "total_reactions_q3": "Total reactions, Q3", "total_reactions_mean": "Total reactions, mean",
              "total_reactions_min": "Total reactions, min", "total_reactions_max": "Total reactions, max",
              "lls_median": "LLS, median", "lls_q1": "LLS, Q1", "lls_q3": "LLS, Q3", "lls_mean": "LLS, mean",
              "lls_min": "LLS, min", "lls_max": "LLS, max"}


def docx_specs(T, n_year_excluded):
    """Table S1, S2, ... specs for :func:`utils.table_docx.tables_to_docx`. Per-item lists (top targets, individual routes, key-step reaction types, example routes, building-block rankings, stereocentre donors) and Tables 1 / 2 are left out."""
    mt = T["transition_metals"].assign(cell=lambda d: d["metal"] + " (" + d["count"].map("{:,}".format) + ")")
    mt = mt.pivot(index="rank", columns="dataset", values="cell")[rc.DATASETS].reset_index()
    rs = T["route_summary"].set_index("dataset").T
    rs = rs.apply(lambda col: [fixed(v, 0) if float(v).is_integer() else fixed(v, 1) for v in col]).reset_index()
    rs["index"] = rs["index"].map(ROUTE_ROWS)
    pu = T["purchasability"].assign(value=lambda d: [fixed(v, 0) if not m.endswith("(%)") else fixed(v, 1)
                                                     for m, v in zip(d["measure"], d["value"])])
    comp = T["composition_summary"].set_index("measure").apply(
        lambda col: [fixed(v, 0) if m == "n targets" else fixed(v, 1) for m, v in col.items()]).reset_index()
    return [
        dict(df=T["routes_by_year"], headers={"publication_year": "Year", "n_routes": "Routes"}, blocks=4,
             caption=f"Number of CbyD routes by publication year (dataset-overview figure, panel A). Routes published in or "
                     f"before 1940 are pooled in the first row."
                     + (f" {n_year_excluded} {'route' if n_year_excluded == 1 else 'routes'} without a publication "
                        f"year {'is' if n_year_excluded == 1 else 'are'} not shown." if n_year_excluded else "")),
        dict(df=T["routes_per_target"], headers={"routes_per_target": "Routes per target", "n_targets": "Targets"},
             caption="Number of CbyD targets synthesised by a given number of routes (dataset-overview figure, panel B). "
                     "Targets are grouped by name after removal of stereodescriptor prefixes."),
        dict(df=T["target_property_summary"], headers=STATS, decimals=2, group="property",
             caption="Summary statistics of target properties of CbyD, USPTO-190 and PaRoutes (dataset-overview figure, panel C), "
                     "one entry per unique target. Overall target complexity is the mean of the min–max-"
                     "normalised SAScore, NPScore, SPS and Böttcher score; SPS is not size-normalised."),
        dict(df=T["target_property_histograms"], headers={"bin": "Bin"}, group="property", blocks=2,
             caption="Distributions of overall target complexity, molecular weight and number of tetrahedral "
                     "stereocentres as plotted in the dataset-overview figure, panel C (% of each dataset's targets per bin). Molecular "
                     "weights of 1000 or more are pooled in the last bin, stereocentre counts of 15 or more in “15+”."),
        dict(df=T["ring_types"], headers={"count_per_target": "Rings per target"}, group="ring_type",
             caption="Ring types per target (dataset-overview figure, panel D): % of each dataset's targets with the given number of "
                     "rings of each type. Targets without a ring of that type are not listed; six or more rings are pooled."),
        dict(df=T["ring_junctions"], headers={"junction": "Ring junction"},
             caption="Ring junctions (dataset-overview figure, panel E): % of targets with at least one fused (two rings sharing one bond), "
                     "bridged (two rings sharing two or more bonds) or spiro (one atom shared by two ring systems) "
                     "junction. A target can belong to several categories."),
        dict(same_page=True, df=comp, headers={"measure": "Targets with"},
             caption="Percentage of targets containing at least one atom of each element or ring of each type, "
                     "and number of targets per dataset (dataset-overview figure, panels D–F)."),
        dict(df=T["heteroatoms"], headers={"count_per_target": "Atoms per target"}, group="element", blocks=2,
             caption="Heteroatoms per target (dataset-overview figure, panel F): % of targets with the given number of atoms of each "
                     "element. Counts of 10 or more (O, N) or 6 or more (halogens, S, P) are pooled."),
        dict(df=T["reaction_classes"].drop(columns="plotted").replace(
                 {"class_name": {"Resolution": "Resolution and separation"}}),
             headers={"class": "Class", "class_name": "NameRxn superclass"},
             caption="NameRxn superclass distribution (reaction-complexity figure, panel A), as % of all reactions of each dataset. Reactions "
                     "with several NameRxn assignments are mixed sub-classes (one superclass) or mixed classes "
                     "(different superclasses). Class 11 (chiral and other separations, isomerizations, racemizations, "
                     "purifications and salt formations) was removed from USPTO-Full and Pistachio during their "
                     "preprocessing in previous work, so it is 0 for both and is not plotted in the figure."),
        dict(same_page=True, df=mt, headers={"rank": "Rank"},
             caption="The ten most frequent transition metals among the reaction agents of each dataset (reaction-complexity figure, panel C), "
                     "with the number of metal-containing agent fragments in parentheses."),
        dict(df=T["reaction_descriptors"], headers={"value": "Value"}, group="descriptor",
             caption="Reaction descriptors (reaction-complexity figure, panel A, and the SI): % of reactions with each number of reacting atoms, "
                     "changing rings, stereochemical changes and C–C or C–X bonds formed. The last value of each "
                     "descriptor pools all higher values."),
        dict(df=T["reaction_descriptor_summary"], headers=STATS, decimals=2, group="descriptor",
             caption="Summary statistics of the reaction descriptors; n is the number of reactions for which the "
                     "descriptor could be computed."),
        dict(df=rs, headers={"index": ""}, decimals=1,
             caption="Route length, longest linear sequence (LLS) and convergence (route-analysis figure). A route is convergent "
                     "when its total number of reactions exceeds its LLS."),
        dict(df=T["complexity_by_progress"], decimals=2, group="metric",
             headers={"metric": "Score", "progress_bin": "Synthetic progress", "n_reactions": "Reactions",
                      "n_routes": "Routes", "mean": "Mean", "sd": "SD", "median": "Median"},
             caption="Product complexity along CbyD routes (key-steps figure, panel A, and the SI): statistics of the product "
                     "score of all in-tree reactions per 5% bin of synthetic progress, 1 − (depth − 1)/LLS. The mean "
                     "is the bold line in the figure."),
        dict(df=T["key_step_thresholds"], decimals={"threshold_delta": 3},
             headers={"metric": "Score", "threshold_delta": "ΔC threshold", "n_key_steps": "Key steps",
                      "n_candidates": "Candidates"},
             caption="Key-step thresholds (key-steps figure): a key step is a candidate reaction whose complexity change ΔC "
                     "is at or above the 90th percentile of all candidates for that score."),
        dict(same_page=True, df=pu, headers={"measure": "", "value": "Value"},
             caption="Purchasability of the CbyD starting materials in the eMolecules catalogue; normalised forms "
                     "include stereochemistry-free and charge- or ylide-normalised representations."),
        dict(same_page=True, df=T["stereocentre_origin"], headers={"class": "", "stereocentres": "Stereocentres"},
             caption="Origin of the assigned tetrahedral stereocentres of CbyD targets: installed during the route "
                     "or carried unchanged from a chiral starting material (purchased)."),
    ]


def write_docx(book, n_year_excluded, path):
    specs = docx_specs(book.tables, n_year_excluded)
    tables_to_docx(specs, path, prefix="S",
                   title="Supplementary Tables: numerical values underlying the main-text figures")
    print(f"{len(specs)} tables (S1–S{len(specs)}) -> {rel(path)}")
