"""Per-route starting materials and reaction depths, from the CbyD reaction network.
"""
from __future__ import annotations
import csv
import os
import sys
from collections import defaultdict, deque
from multiprocessing import Pool

from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import data_loaders as dl, route_topology as rt
from utils.paths import a, ensure_output_dirs


def _reaction_depths(net):
    """Map synthesis-index -> reaction depth (reactions from target, inclusive)."""
    adj = defaultdict(set)
    for e in net["edges"]:
        adj[e["source"]].add(e["target"])
        adj[e["target"]].add(e["source"])
    ntype = {n["id"]: n["type"] for n in net["nodes"]}
    rindex = {n["id"]: n.get("index") for n in net["nodes"] if n["type"] == "reaction"}
    targets = [n["id"] for n in net["nodes"]
               if n["type"] == "molecule" and n.get("role") == "target"]
    depth_by_index = {}
    for t in targets:
        # BFS from target; carry #reaction-nodes seen on the path so far
        seen = {t}
        q = deque([(t, 0)])
        while q:
            node, rx = q.popleft()
            for nb in adj[node]:
                if nb in seen:
                    continue
                seen.add(nb)
                nrx = rx + (1 if ntype.get(nb) == "reaction" else 0)
                if ntype.get(nb) == "reaction":
                    depth_by_index[rindex[nb]] = nrx
                q.append((nb, nrx))
    return depth_by_index


def _cbyd_routes():
    for route in dl.iter_cbyd_routes():
        yield route.get("_file"), (route.get("synthesis", []) or [])


def _absynth_routes():
    for pid, meta, synthesis in dl.iter_absynth_routes():
        yield f"absynth-{pid}", synthesis


def _route_detail(item):
    """Pool worker: (route_name, synthesis) -> (route_name, n_synthesis, starting-material SMILES, {synthesis_index: depth}) or None depths on failure."""
    route_name, synthesis = item
    try:
        net = rt._get_build_network()(synthesis, align=False)
    except Exception:
        return route_name, len(synthesis), [], None
    sms = [n["smiles"] for n in net["nodes"]
           if n["type"] == "molecule" and n.get("role") == "start"]
    return route_name, len(synthesis), sms, _reaction_depths(net)


def process(dataset, routes_iter, total, prefix, processes=32):
    """Emit starting-material + per-reaction-depth tables for a route dataset.
    ``prefix`` is the reaction-id prefix (must match the flattening order used by 00_prepare_tables so ids join with the product-complexity table)."""
    sm_path = a("starting_materials", f"{dataset}_starting_materials.csv")
    dep_path = a("route_topology", f"{dataset}_reaction_depth.csv")
    os.makedirs(os.path.dirname(sm_path), exist_ok=True)
    os.makedirs(os.path.dirname(dep_path), exist_ok=True)

    sm_f = open(sm_path, "w", newline="")
    dep_f = open(dep_path, "w", newline="")
    sm_w = csv.writer(sm_f); sm_w.writerow(["route", "smiles"])
    dep_w = csv.writer(dep_f); dep_w.writerow(
        ["id", "route", "synthesis_index", "depth", "lls", "n_reactions"])

    g = 0  # global reaction counter (matches <prefix>-<i> ids)
    n_sm = n_dep = 0
    # imap (ordered) keeps route order, so the running counter ``g`` still matches the flattening order of 00_prepare_tables.
    pool = Pool(processes=processes)
    for route_name, n_syn, sms, depths in tqdm(
            pool.imap(_route_detail, routes_iter, chunksize=4), total=total,
            desc=f"{dataset}-route-details"):
        if depths is None:
            g += n_syn
            continue
        for smi in sms:
            sm_w.writerow([route_name, smi]); n_sm += 1
        lls = max(depths.values()) if depths else 0
        n_rx = len(depths)
        for k in range(n_syn):
            d = depths.get(k)
            if d is not None:
                dep_w.writerow([f"{prefix}-{g + k}", route_name, k, d, lls, n_rx])
                n_dep += 1
        g += n_syn
    pool.close(); pool.join()

    sm_f.close(); dep_f.close()
    print(f"wrote {n_sm:,} starting-material rows -> {sm_path}")
    print(f"wrote {n_dep:,} reaction-depth rows -> {dep_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cbyd", "absynth"], default="cbyd")
    ap.add_argument("--processes", type=int, default=32)
    args = ap.parse_args()
    ensure_output_dirs()
    if args.dataset == "cbyd":
        process("cbyd", _cbyd_routes(), len(dl.cbyd_route_files()), "cbyd",
                args.processes)
    else:
        routes = list(_absynth_routes())
        process("absynth", routes, len(routes), "absynth", args.processes)


if __name__ == "__main__":
    main()
