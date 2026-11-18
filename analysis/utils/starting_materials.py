"""Section 4 — starting materials: chiral building blocks, purchasability of CbyD starting materials, and whether target stereocentres were purchased or installed.
"""
from __future__ import annotations
import collections
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import glob

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

from . import buyable as bx, chem, data_loaders as dl, route_topology as rt
from . import route_tree_svg as rts, stereo_provenance as sp
from .data_loaders import opt_csv
from .paths import a, rel, CBYD_ROUTES_DIR, EMOLECULES_SMI

# reagents build_network labels as 'start' (MeOH, TBSCl, NBS, ...) are not building blocks
_REAGENT_SMARTS = [Chem.MolFromSmarts(s) for s in
                   ["[Si]", "[B]", "S(=O)(=O)", "[OX2][OX2]", "[NX3]Br", "[NX3]Cl",
                    "[Cl,Br,I][Cl,Br,I]", "C(=O)O[Cl]"]]


# Chiral building blocks -----
def is_building_block(smiles):
    """A CHIRAL un-produced reactant that is not reagent-like. R/S labels alone are not enough: meso compounds (cis-cyclohexene-1,2-diol, tropinone, ...) equal their mirror image."""
    m = Chem.MolFromSmiles(smiles)
    if m is None or not chem.is_chiral(smiles):
        return False
    return not any(m.HasSubstructMatch(p) for p in _REAGENT_SMARTS if p is not None)


def _frequency(df):
    """(all start molecules, chiral building blocks) ranked by # routes using them."""
    freq = (df.groupby("canon").agg(n_occurrence=("route", "size"), n_routes=("route", "nunique"))
              .sort_values("n_routes", ascending=False))
    return freq, freq[[is_building_block(s) for s in freq.index]]


def cbyd_starting_materials():
    df = opt_csv("starting_materials", "cbyd_starting_materials.csv")
    df["canon"] = df["smiles"]
    freq, freq_bb = _frequency(df)
    print(f"CbyD: {len(freq_bb):,} building blocks of {len(freq):,} un-produced reactants")
    return freq, freq_bb


def _frequency_from_pairs(pairs):
    """(route_id, [smiles]) pairs -> (freq, freq_bb), for datasets without an SM table."""
    rows = [(rid, c) for rid, sms in pairs for smi in sms if (c := chem.canonicalize(smi))]
    return _frequency(pd.DataFrame(rows, columns=["route", "canon"]))


def _uspto190_sm(route):
    """Reactant fragments never formed as a product (steps are RETRO product>>precursors)."""
    prods, reac_frags = set(), []
    for rxn in (r for r in route if r):
        p, _, r = chem.rxn_parts(rxn)
        prods.update(c for f in p.split(".") if (c := chem.canonicalize(f)))
        reac_frags.append(r)
    return [f for r in reac_frags for f in r.split(".")
            if (c := chem.canonicalize(f)) and c not in prods]


def _paroutes_leaves(tree):
    out = []

    def walk(n):
        ch = n.get("children") or []
        if n.get("type") == "mol" and not ch and n.get("smiles"):
            out.append(n["smiles"])
        for c in ch:
            walk(c)
    walk(tree)
    return out


def uspto190_building_blocks():
    _, bb = _frequency_from_pairs((i, _uspto190_sm(r)) for i, r in enumerate(dl.load_uspto190()))
    print(f"USPTO-190: {len(bb):,} building blocks")
    return bb


def paroutes_building_blocks():
    _, bb = _frequency_from_pairs((i, _paroutes_leaves(t)) for i, t in enumerate(dl.load_paroutes()))
    print(f"PaRoutes: {len(bb):,} building blocks")
    return bb


def draw_top(frame, n=24):
    """Grid of the n most-used building blocks (inline display only)."""
    top = frame.head(n).reset_index()
    mols = [Chem.MolFromSmiles(s) for s in top["canon"]]
    keep = [i for i, m in enumerate(mols) if m is not None]
    return Draw.MolsToGridImage([mols[i] for i in keep], molsPerRow=6, subImgSize=(180, 150),
                                legends=[f"{top['n_routes'][i]} routes" for i in keep], useSVG=True)


# Purchasability (eMolecules) -----
BUY_CACHE = a("starting_materials", "cbyd_sm_buyable.csv")


def buyability(freq):
    """Per CbyD start molecule: buyable as the exact stereoisomer / any stereoisomer (flat) / after normalisation (azide <-> HN3, covalent <-> ionic organometallics, ylide drawings, counterion-agnostic ions). Cached; the scan takes ~5 min."""
    if os.path.exists(BUY_CACHE):
        print("loaded cached buyability ->", os.path.relpath(BUY_CACHE, a()))
        return pd.read_csv(BUY_CACHE)
    buy = freq.reset_index().rename(columns={"canon": "smiles"})
    buy["iso"] = [bx.canon_iso(s) for s in buy["smiles"]]
    buy["flat"] = [bx.canon_flat(s) for s in buy["smiles"]]
    exalt = [bx.variant_keys(s) for s in buy["smiles"]]
    frags = [bx.significant_ion_frags(s) for s in buy["smiles"]]
    buy["alts"] = [";".join(sorted(al)) for _, al in exalt]
    buy["frags"] = [";".join(sorted(fr)) for fr in frags]
    iso_t = set(x for x in buy["iso"] if x) | set(x for _, al in exalt for x in al)
    flat_t = set(x for x in buy["flat"] if x)
    frag_t = set(x for fr in frags for x in fr)
    fi, ff, fg = bx.buyable_scan(iso_t, flat_t, EMOLECULES_SMI, n_workers=64, frag_targets=frag_t)
    buy["buyable"] = buy["iso"].isin(fi)
    buy["buyable_flat"] = buy["flat"].isin(ff)
    buy["buyable_norm"] = [(e in fi) or any(x in fi for x in al) or any(x in fg for x in fr)
                           for (e, al), fr in zip(exalt, frags)]
    buy.to_csv(BUY_CACHE, index=False)
    print("scanned eMolecules ->", os.path.relpath(BUY_CACHE, a()))
    return buy


def _has_carbon(s):
    m = Chem.MolFromSmiles(s)
    return bool(m) and any(at.GetSymbol() == "C" for at in m.GetAtoms())


def buyability_summary(buy, freq_bb):
    """Shares of buyable / not-buyable CbyD start molecules, as a table."""
    nb = ~buy["buyable_norm"]
    w = buy["n_routes"]
    has_c = buy.loc[nb, "smiles"].map(_has_carbon)
    bb = buy[buy["smiles"].isin(freq_bb.index)]
    rows = [("unique start molecules", len(buy), 100.0),
            ("buyable, exact stereoisomer", buy["buyable"].sum(), buy["buyable"].mean() * 100),
            ("buyable, incl. normalised forms", buy["buyable_norm"].sum(), buy["buyable_norm"].mean() * 100),
            ("NOT buyable (after normalisation)", nb.sum(), nb.mean() * 100),
            ("   only a different stereoisomer sold", (nb & buy["buyable_flat"]).sum(),
             (nb & buy["buyable_flat"]).mean() * 100),
            ("   scaffold fully absent", (~buy["buyable_flat"]).sum(), (~buy["buyable_flat"]).mean() * 100),
            ("   NOT buyable, weighted by route count", np.nan, w[nb].sum() / w.sum() * 100),
            ("   of NOT buyable: carbon-containing", has_c.sum(), has_c.mean() * 100),
            ("   of NOT buyable: no carbon (H2O, I2, OsO4, ...)", (~has_c).sum(), (~has_c).mean() * 100),
            ("chiral building blocks", len(bb), 100.0),
            ("   chiral building blocks NOT buyable", (~bb["buyable_norm"]).sum(),
             (~bb["buyable_norm"]).mean() * 100)]
    return pd.DataFrame(rows, columns=["", "n", "%"]).set_index("").round(1).astype({"%": float})


def plot_not_buyable(buy, top=25):
    """SI ``starting_materials_not_buyable``: the most-used non-buyable start molecules, carbon-containing vs inorganic."""
    nb = buy[~buy["buyable_norm"]].copy()
    nb["has_C"] = nb["smiles"].map(_has_carbon)
    top_smiles = lambda f: list(f.sort_values("n_routes", ascending=False).head(top)["smiles"])
    sections = {"carbon-containing": top_smiles(nb[nb["has_C"]]),
                "no carbon/inorganic": top_smiles(nb[~nb["has_C"]])}
    out = rts.molecule_sections(sections, a("figures", "SI", "starting_materials_not_buyable"), show=True)
    print(f"saved -> {rel(out[0])}")


# Purchased vs installed target stereocentres -----
STEREO_CACHE = a("starting_materials", "cbyd_stereo_provenance.csv")
_RED, _GREEN = (1, 0.55, 0.55), (0.55, 0.9, 0.55)
_VCOL = {"installed": _RED, "purchased": _GREEN, "unresolved": (0.55, 0.55, 0.6)}


def stereo_provenance():
    """One row per route: target stereocentres installed / purchased / unresolved (``utils.stereo_provenance``; cached)."""
    if os.path.exists(STEREO_CACHE):
        print("loaded cached stereo provenance ->", os.path.relpath(STEREO_CACHE, a()))
        return pd.read_csv(STEREO_CACHE)
    files = sorted(glob.glob(os.path.join(CBYD_ROUTES_DIR, "*.json")))
    with ProcessPoolExecutor(max_workers=32) as ex:
        rows = [r for r in ex.map(sp.classify_file, files, chunksize=8) if r]
    prov = pd.DataFrame(rows); prov.to_csv(STEREO_CACHE, index=False)
    print(f"classified {len(prov):,} routes -> {os.path.relpath(STEREO_CACHE, a())}")
    return prov


def provenance_summary(prov):
    tot_i, tot_p, tot_u = (int(prov[c].sum()) for c in ("n_installed", "n_purchased", "n_unresolved"))
    tot, res = tot_i + tot_p + tot_u, tot_i + tot_p
    print(f"assigned stereocentres in final targets across {len(prov):,} routes:")
    return pd.DataFrame(
        {"stereocentres": [tot_i, tot_p, tot_u, tot],
         "% of all": [tot_i / tot * 100, tot_p / tot * 100, tot_u / tot * 100, 100.0],
         "% of resolved": [tot_i / res * 100, tot_p / res * 100, np.nan, np.nan]},
        index=["installed (made by route)", "purchased (from a chiral SM)",
               "unresolved (mapping gap)", "total"])


def top_donors(prov, n=12):
    """Chiral starting materials that most often donate a target stereocentre."""
    donors = collections.Counter(smi for s in prov["purchased_sms"].dropna()
                                 for smi in str(s).split(";") if smi)
    return pd.DataFrame(donors.most_common(n), columns=["purchased SM (SMILES)", "# routes"])


def provenance_example(prov, pick=17):
    """A compact route (4-8 reactions, <= 4 stereocentres) with both an installed and a purchased centre; ``pick`` = its position when sorted by size."""
    cand = prov[(prov.n_purchased > 0) & (prov.n_installed > 0)]
    mid = cand[cand.n_reactions.between(4, 8) & (cand.n_stereo <= 4)]
    return (mid if len(mid) else cand).sort_values(["n_reactions", "n_stereo"]).iloc[pick]


def provenance_tree_svg(row, max_width_in=7.05):
    """Snake route tree (SMs left -> target right) of a ``stereo_provenance`` row, each molecule's tracked stereocentres circled red (installed) / green (purchased); saved to ``SI/traced_stereo_provtree_<target>.svg``. Returns (path, SVG text)."""
    syn = dl.load_cbyd_synthesis(row["route"])
    ex = sp.explain_route(syn)
    rxn_hl = defaultdict(dict)                      # synthesis reaction idx -> {product map: colour}
    for c in ex["centers"]:
        for si, mp in c.get("path", []):
            rxn_hl[si][mp] = _VCOL[c["verdict"]]
    net = rt._get_build_network()(syn, align=False)
    nodes = {n["id"]: n for n in net["nodes"]}
    rxn_syn = {n["id"]: n["index"] for n in net["nodes"] if n["type"] == "reaction"}
    prod_rxn = {e["target"]: rxn_syn[e["source"]] for e in net["edges"]
                if nodes[e["source"]]["type"] == "reaction" and nodes[e["target"]]["type"] == "molecule"}

    def _draw(n):
        m = rts.node_mol(n) or Chem.MolFromSmiles(n["smiles"])   # highlights in ITS atom order
        if m is None:
            return None
        ic = {}
        if n["id"] in prod_rxn and prod_rxn[n["id"]] in rxn_hl:   # product -> map-based highlight
            mpd = sp.syn_reaction_mapped(syn, prod_rxn[n["id"]])
            mp = Chem.MolFromSmiles(sp._main_product(sp._rxn_parts(mpd)[2])) if mpd else None
            if mp is not None:
                cmap = rxn_hl[prod_rxn[n["id"]]]
                col = {at.GetIdx(): cmap[at.GetAtomMapNum()] for at in mp.GetAtoms()
                       if at.GetAtomMapNum() in cmap}
                for at in mp.GetAtoms():
                    at.SetAtomMapNum(0)
                match = m.GetSubstructMatch(mp, useChirality=True) or m.GetSubstructMatch(mp)
                ic = {match[k]: v for k, v in col.items()} if match else {}
        elif n.get("role") == "start":               # leaf SM -> its stereocentres green
            for idx, _ in Chem.FindMolChiralCenters(m, useLegacyImplementation=False,
                                                    includeUnassigned=False):
                ic[idx] = _GREEN
        return rts.node_svg(n, highlight=ic)

    slug = re.sub(r"[^a-z0-9]+", "_", str(row["target"]).lower()).strip("_")
    out = a("figures", "SI", f"traced_stereo_provtree_{slug}.svg")
    return out, rts.snake_tree_svg(syn, out, max_width_in=max_width_in, mol_drawer=_draw)
