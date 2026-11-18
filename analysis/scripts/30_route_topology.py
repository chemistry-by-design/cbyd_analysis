"""
Route-level topology for CbyD, PaRoutes and USPTO-190: size, longest linear sequence, convergency and the connectivity flags.
"""
from __future__ import annotations
import argparse
import csv
import os

from tqdm import tqdm
from multiprocessing import Pool

from _common import write_jsonl
from utils import data_loaders as dl, route_topology as rt
from utils.paths import a, ensure_output_dirs


def _cbyd_row(path):
    try:
        route = dl.load_cbyd_route(path)
    except Exception as e:
        return {"route": os.path.basename(path), "error": f"load: {e}"}
    bg = route.get("background", {}) or {}
    m = rt.route_metrics(route.get("synthesis", []))
    return {
        "route": route.get("_file"),
        "target_name": bg.get("name"),
        "year": bg.get("year"),
        "author": bg.get("author"),
        "drug_name": bg.get("drug name") or bg.get("drug_name"),
        "doi": bg.get("doi"),
        "target": m.get("target"),
        "n_steps": m.get("n_steps"),
        "n_reactions": m.get("n_reactions"),
        "longest_linear_sequence": m.get("longest_linear_sequence"),
        "is_convergent": m.get("is_convergent"),
        "connected": m.get("connected"),
        "n_targets": m.get("n_targets"),
        "n_orphans": m.get("n_orphans"),
        "error": m.get("error"),
    }


def build_cbyd(processes):
    files = dl.cbyd_route_files()
    rows = []
    with Pool(processes=processes) as pool:
        for r in tqdm(pool.imap(_cbyd_row, files, chunksize=8),
                      total=len(files), desc="cbyd-topology"):
            rows.append(r)
    out = a("route_topology", "cbyd_route_topology.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fields = ["route", "target_name", "year", "author", "drug_name", "doi",
              "target", "n_steps", "n_reactions", "longest_linear_sequence",
              "is_convergent", "connected", "n_targets", "n_orphans", "error"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"wrote {len(rows):,} rows -> {out}")
    # targets table for complexity scoring
    write_jsonl(a("molecules", "cbyd_targets.jsonl"),
                ({"id": r["route"], "smiles": r.get("target")}
                 for r in rows if r.get("target")))


def build_paroutes():
    out = a("route_topology", "paroutes_route_topology.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    trees = dl.load_paroutes()
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "target", "n_reactions",
                                          "longest_linear_sequence", "n_leaves",
                                          "n_leaves_in_stock", "is_convergent"])
        w.writeheader()
        for i, t in enumerate(tqdm(trees, desc="paroutes-topology")):
            s = dl.paroutes_route_summary(t)
            s.pop("reactions", None)
            s["id"] = f"paroutes-{i}"
            s["is_convergent"] = s["n_reactions"] > s["longest_linear_sequence"]
            w.writerow(s)
    print(f"wrote {len(trees):,} rows -> {out}")


def _absynth_row(args):
    path_id, meta, synthesis = args
    m = rt.route_metrics(synthesis)
    return {
        "id": f"absynth-{path_id}",
        "target_name": meta.get("target_name"),
        "year": meta.get("year"),
        "author": meta.get("author"),
        "doi": meta.get("doi"),
        "target": m.get("target"),
        "n_steps": m.get("n_steps"),
        "n_reactions": m.get("n_reactions"),
        "longest_linear_sequence": m.get("longest_linear_sequence"),
        "is_convergent": m.get("is_convergent"),
        "connected": m.get("connected"),
        "n_targets": m.get("n_targets"),
        "n_orphans": m.get("n_orphans"),
        "error": m.get("error"),
    }


def build_absynth(processes):
    routes = list(dl.iter_absynth_routes())
    rows = []
    with Pool(processes=processes) as pool:
        for r in tqdm(pool.imap(_absynth_row, routes, chunksize=8),
                      total=len(routes), desc="absynth-topology"):
            rows.append(r)
    out = a("route_topology", "absynth_route_topology.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fields = ["id", "target_name", "year", "author", "doi", "target", "n_steps",
              "n_reactions", "longest_linear_sequence", "is_convergent",
              "connected", "n_targets", "n_orphans", "error"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"wrote {len(rows):,} rows -> {out}")
    write_jsonl(a("molecules", "absynth_targets.jsonl"),
                ({"id": r["id"], "smiles": r.get("target")}
                 for r in rows if r.get("target")))


def build_uspto190():
    out = a("route_topology", "uspto190_route_topology.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    routes = dl.load_uspto190()
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "target", "n_reactions",
                                          "longest_linear_sequence", "is_convergent"])
        w.writeheader()
        for i, r in enumerate(routes):
            # same machinery as CbyD: forward steps -> build_network tree
            m = rt.route_metrics(dl.uspto190_to_synthesis(r))
            w.writerow({"id": f"uspto190-{i}",
                        "target": dl.uspto190_route_summary(r)["target"],
                        **{k: m[k] for k in ("n_reactions", "longest_linear_sequence",
                                             "is_convergent")}})
    print(f"wrote {len(routes):,} rows -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=16)
    ap.add_argument("--only", choices=["cbyd", "paroutes", "uspto190", "absynth"],
                    default=None)
    args = ap.parse_args()
    ensure_output_dirs()
    if args.only in (None, "cbyd"):
        build_cbyd(args.processes)
    if args.only in (None, "paroutes"):
        build_paroutes()
    if args.only in (None, "uspto190"):
        build_uspto190()
    if args.only in (None, "absynth"):
        build_absynth(args.processes)


if __name__ == "__main__":
    main()
