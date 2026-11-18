"""Loaders for every dataset in the analysis; routes are returned as lists, the one-step sets are streamed.
"""
from __future__ import annotations
import json
import os
import pickle
import random
from typing import Iterator, Optional

from . import paths


def opt_csv(*parts, **kw):
    """Load a ``data_analysis`` CSV, or warn + return None if the precompute that produces it has not been run yet (so a notebook still executes top-to-bottom)."""
    import pandas as pd
    p = paths.a(*parts)
    if not os.path.exists(p):
        print(f"[missing] {os.path.relpath(p, paths.a())}  -- run its precompute script first.")
        return None
    return pd.read_csv(p, **kw)


# CbyD (this repo) -----
def cbyd_route_files():
    """Sorted list of absolute paths to every processed CbyD route JSON."""
    d = paths.CBYD_ROUTES_DIR
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".json")]


def load_cbyd_route(path: str) -> dict:
    with open(path) as f:
        route = json.load(f)
    route["_file"] = os.path.basename(path)
    return route


def iter_cbyd_routes() -> Iterator[dict]:
    """Yield every route dict ({'background', 'synthesis', '_file'})."""
    for p in cbyd_route_files():
        try:
            yield load_cbyd_route(p)
        except Exception as e:  # a handful of files may be malformed
            print(f"[cbyd] skipped {os.path.basename(p)}: {e}")


def load_cbyd_routes() -> list:
    return list(iter_cbyd_routes())


def load_cbyd_synthesis(route_file: str) -> list:
    """The ``synthesis`` list of one CbyD route, by file name (e.g. ``Codeine_2009_Magnus.json``)."""
    return load_cbyd_route(os.path.join(paths.CBYD_ROUTES_DIR, route_file))["synthesis"]


def load_cbyd_background(route_file: str) -> dict:
    return load_cbyd_route(os.path.join(paths.CBYD_ROUTES_DIR, route_file)).get("background") or {}


def iter_cbyd_reactions(rxn_key: str = "reaction_smiles_mapped") -> Iterator[dict]:
    """Flatten every route into per-reaction records ``{route, target_name, year, author, reaction_id, rxn_smiles, conditions, mapping_confidence}``; ``rxn_smiles`` is the mapped one.
    """
    for route in iter_cbyd_routes():
        bg = route.get("background", {}) or {}
        for rxn in route.get("synthesis", []) or []:
            smi = (rxn.get(rxn_key) or "").strip()
            if not smi:
                # fall back to the unmapped reaction
                smi = (rxn.get("reaction_smiles") or "").strip()
            yield {
                "route": route.get("_file"),
                "target_name": bg.get("name"),
                "year": bg.get("year"),
                "author": bg.get("author"),
                "drug_name": bg.get("drug name") or bg.get("drug_name"),
                "reaction_id": rxn.get("reaction_id"),
                "rxn_smiles": smi,
                "conditions": rxn.get("conditions", {}),
                "mapping_confidence": rxn.get("mapping_confidence"),
            }


# PaRoutes (AiZynthFinder nested trees) -----
def load_paroutes() -> list:
    """List of nested route trees. Root node has ``smiles`` (target) and ``children`` = [reaction-node, ...] where a reaction node has ``metadata.smiles`` (mapped rxn) and ``children`` = precursor mol nodes."""
    with open(paths.PAROUTES_JSON) as f:
        return json.load(f)


def _walk_paroutes(node, out_reactions, depth, depths):
    """DFS a PaRoutes tree collecting reaction SMILES and leaf depths."""
    children = node.get("children") or []
    if node.get("type") == "mol":
        if not children:                       # leaf molecule
            depths.append(depth)
        for ch in children:
            _walk_paroutes(ch, out_reactions, depth, depths)
    elif node.get("type") == "reaction":
        smi = (node.get("metadata", {}) or {}).get("smiles") or node.get("smiles") or ""
        if smi:
            out_reactions.append(smi)
        for ch in children:                    # precursors are one step deeper
            _walk_paroutes(ch, out_reactions, depth + 1, depths)


def paroutes_route_summary(tree: dict) -> dict:
    """Topology of one PaRoutes tree: target, #reactions, longest-linear-seq (= max leaf depth), #leaves, and how many leaves are in-stock."""
    reactions, depths = [], []
    _walk_paroutes(tree, reactions, 0, depths)
    # count leaves + in-stock leaves
    n_leaves = [0]; n_stock = [0]

    def _leaves(n):
        ch = n.get("children") or []
        if n.get("type") == "mol" and not ch:
            n_leaves[0] += 1
            if n.get("in_stock"):
                n_stock[0] += 1
        for c in ch:
            _leaves(c)
    _leaves(tree)
    return {
        "target": tree.get("smiles"),
        "n_reactions": len(reactions),
        "longest_linear_sequence": max(depths) if depths else 0,
        "n_leaves": n_leaves[0],
        "n_leaves_in_stock": n_stock[0],
        "reactions": reactions,
    }


def iter_paroutes_reactions() -> Iterator[str]:
    for tree in load_paroutes():
        reactions, _ = [], []
        _walk_paroutes(tree, reactions, 0, [])
        yield from reactions


# USPTO-190 (list of linear routes, each a list of reaction SMILES) -----
def load_uspto190() -> list:
    with open(paths.USPTO190_PKL, "rb") as f:
        return pickle.load(f)


def _largest_fragment(smiles: str) -> str:
    """Largest '.'-separated fragment by character length (proxy for size) — used to drop reagent/byproduct fragments from a product string."""
    if not smiles:
        return smiles
    return max(smiles.split("."), key=len)


def iter_absynth_routes():
    """Yield ``(path_id, meta, synthesis)`` per ABSynth route, from the TSV's contiguous ``PathId`` groups, in the shape ``build_network`` and ``route_metrics`` consume.
    """
    import csv
    with open(paths.ABSYNTH_TSV, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cur_id, cur, meta = None, [], {}
        for row in reader:
            pid = row.get("PathId")
            if pid is None:
                continue
            if pid != cur_id:
                if cur:
                    yield cur_id, meta, cur
                cur_id, cur = pid, []
                meta = {"target_name": row.get("TargetName"), "year": row.get("Year"),
                        "doi": row.get("DOI"), "author": row.get("Author")}
            cur.append({"reaction_id": row.get("StepId"),
                        "reaction_smiles": (row.get("RxnSMILES") or "").strip()})
        if cur:
            yield cur_id, meta, cur


def uspto190_to_synthesis(route: list) -> list:
    """USPTO-190 steps are RETRO strings ``product>>precursors`` (first step makes the target). Reverse each to forward ``precursors>>product`` in the ``{reaction_id, reaction_smiles}`` shape ``build_network`` consumes."""
    return [{"reaction_id": i + 1,
             "reaction_smiles": f"{r.partition('>>')[2]}>>{r.partition('>>')[0]}"}
            for i, r in enumerate(x for x in route if x)]


def uspto190_route_summary(route: list) -> dict:
    """Target = product (LEFT side) of the first retro step ``product>>precursors``.
    Topology (n_reactions / LLS / convergence) is computed in 30_route_topology.py via ``build_network`` on ``uspto190_to_synthesis`` - routes are NOT all linear."""
    reactions = [r for r in route if r]
    target = _largest_fragment(reactions[0].split(">")[0]) if reactions else None
    return {"target": target, "n_reactions": len(reactions), "reactions": reactions}


# One-step reaction sets (USPTO-Full, Pistachio) — streamed JSON-lines -----
def iter_onestep(path: str, limit: Optional[int] = None) -> Iterator[dict]:
    """Stream a JSON-lines reaction file. Each yielded dict has at least ``rxn_smiles`` and ``name_rxn`` (already present in the cleaned files)."""
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield rec
            n += 1
            if limit and n >= limit:
                return


def count_lines(path: str) -> int:
    with open(path) as f:
        return sum(1 for _ in f)


def sample_onestep(path: str, k: int, seed: int = 0,
                   total: Optional[int] = None) -> list:
    """Uniform reservoir sample of ``k`` records from a large JSONL file in a single pass (no need to know the length ahead of time)."""
    rng = random.Random(seed)
    reservoir = []
    for i, rec in enumerate(iter_onestep(path)):
        if i < k:
            reservoir.append(rec)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = rec
    return reservoir


def uspto_full(limit=None):
    return iter_onestep(paths.USPTO_FULL_JSONL, limit=limit)


def pistachio(limit=None):
    return iter_onestep(paths.PISTACHIO_JSONL, limit=limit)
