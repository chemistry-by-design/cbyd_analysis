"""
Extract the product molecule of every reaction into a ``{id, smiles}`` JSONL.
"""
from __future__ import annotations
import argparse

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from _common import read_jsonl, stream_jsonl_pool, count_lines
from utils import chem


def process_row(row):
    prod = chem.product_smiles(row.get("rxn_smiles", ""))
    if not prod:
        smi = ""
    else:
        frag = max(prod.split("."), key=len)
        smi = chem.canonicalize(frag, remove_atom_maps=True)
    return {"id": row.get("id"), "smiles": smi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--processes", type=int, default=32)
    args = ap.parse_args()
    stream_jsonl_pool(process_row, read_jsonl(args.input), args.output,
                      processes=args.processes, desc="extract-products",
                      total=count_lines(args.input))


if __name__ == "__main__":
    main()
