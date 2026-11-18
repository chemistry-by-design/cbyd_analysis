"""
Build the reaction and molecule tables in ``data/data_analysis/`` that the later scripts read.
"""
from __future__ import annotations
import argparse
import os

from _common import write_jsonl
from utils import data_loaders as dl, chem
from utils.paths import (a, ensure_output_dirs, USPTO_FULL_JSONL, PISTACHIO_JSONL)


def _product_mol(rxn_smiles):
    prod = chem.product_smiles(rxn_smiles)
    if not prod:
        return ""
    # largest fragment, atom-maps stripped, canonical
    frag = max(prod.split("."), key=len) if prod else prod
    return chem.canonicalize(frag, remove_atom_maps=True)


def build_cbyd():
    rxn_path = a("reactions", "cbyd_reactions.jsonl")
    prod_path = a("molecules", "cbyd_products.jsonl")

    def rxn_rows():
        for i, rec in enumerate(dl.iter_cbyd_reactions()):
            if not rec["rxn_smiles"]:
                continue
            yield {
                "id": f"cbyd-{i}",
                "rxn_smiles": rec["rxn_smiles"],
                "route": rec["route"],
                "target_name": rec["target_name"],
                "year": rec["year"],
                "author": rec["author"],
                "mapping_confidence": rec["mapping_confidence"],
            }

    rows = list(rxn_rows())
    write_jsonl(rxn_path, rows)
    write_jsonl(prod_path,
                ({"id": r["id"], "smiles": _product_mol(r["rxn_smiles"])} for r in rows))


def build_onestep(name, path, sample, seed=0, full=False):
    """Write ``{name}_reactions.jsonl`` (id, rxn_smiles, name_rxn).

    ``full=True`` streams EVERY reaction (low memory, no product table here — build products with 05_extract_products.py in parallel). Otherwise reservoir-samples ``sample`` reactions and also writes the product table inline."""
    if not os.path.exists(path):
        print(f"[skip] {name}: {path} not found")
        return
    rxn_out = a("reactions", f"{name}_reactions.jsonl")

    if full:
        print(f"[{name}] streaming ALL reactions from {path} ...")
        def rows():
            for i, r in enumerate(dl.iter_onestep(path)):
                yield {"id": f"{name}-{i}", "rxn_smiles": r.get("rxn_smiles", ""),
                       "name_rxn": r.get("name_rxn", "")}
        write_jsonl(rxn_out, rows())
        print(f"[{name}] now run 05_extract_products.py on {rxn_out}")
        return

    print(f"[{name}] reservoir-sampling {sample:,} reactions from {path} ...")
    recs = dl.sample_onestep(path, k=sample, seed=seed)
    prod_out = a("molecules", f"{name}_products.jsonl")
    rows = []
    for i, r in enumerate(recs):
        smi = r.get("rxn_smiles", "")
        rows.append({"id": f"{name}-{i}", "rxn_smiles": smi,
                     "name_rxn": r.get("name_rxn", "")})
    write_jsonl(rxn_out, rows)
    write_jsonl(prod_out,
                ({"id": r["id"], "smiles": _product_mol(r["rxn_smiles"])} for r in rows))


def build_absynth():
    """Flatten ABSynth routes into reaction + product tables. ``id = absynth-<i>`` uses the same global reaction order as 50_route_details.py (so depth joins)."""
    rxn_path = a("reactions", "absynth_reactions.jsonl")
    prod_path = a("molecules", "absynth_products.jsonl")
    rows, g = [], 0
    for pid, meta, synthesis in dl.iter_absynth_routes():
        for rxn in synthesis:
            rows.append({"id": f"absynth-{g}", "route": f"absynth-{pid}",
                         "target_name": meta.get("target_name"), "year": meta.get("year"),
                         "rxn_smiles": rxn.get("reaction_smiles", "")})
            g += 1
    write_jsonl(rxn_path, rows)
    write_jsonl(prod_path,
                ({"id": r["id"], "smiles": _product_mol(r["rxn_smiles"])} for r in rows))


def build_route_targets():
    # PaRoutes: clean root target
    write_jsonl(a("molecules", "paroutes_targets.jsonl"),
                ({"id": f"paroutes-{i}",
                  "smiles": chem.canonicalize(dl.paroutes_route_summary(t)["target"])}
                 for i, t in enumerate(dl.load_paroutes())))
    # USPTO-190: largest fragment of first-reaction product
    write_jsonl(a("molecules", "uspto190_targets.jsonl"),
                ({"id": f"uspto190-{i}",
                  "smiles": chem.canonicalize(dl.uspto190_route_summary(r)["target"])}
                 for i, r in enumerate(dl.load_uspto190())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=60000,
                    help="rows to sample from each large one-step dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full", action="store_true",
                    help="use ALL one-step reactions instead of sampling")
    ap.add_argument("--only", choices=["cbyd", "onestep", "targets", "absynth"],
                    default=None)
    args = ap.parse_args()

    ensure_output_dirs()
    if args.only in (None, "cbyd"):
        build_cbyd()
    if args.only in (None, "onestep"):
        build_onestep("uspto_full", USPTO_FULL_JSONL, args.sample, args.seed, args.full)
        build_onestep("pistachio", PISTACHIO_JSONL, args.sample, args.seed, args.full)
    if args.only in (None, "absynth"):
        build_absynth()
    if args.only in (None, "targets"):
        build_route_targets()


if __name__ == "__main__":
    main()
