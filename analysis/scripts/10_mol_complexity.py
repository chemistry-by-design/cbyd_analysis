"""
Complexity scores and RDKit descriptors for a ``{id, smiles}`` JSONL table.
Scorers come from ``complexity_features``, so openbabel is required; a failed score is -9999.
"""
from __future__ import annotations
import argparse

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit import Chem

from _common import read_jsonl, run_pool, count_lines
from complexity_features.product_complexity import (
    get_SAScore, get_NPScore, get_SPScore, get_BoettcherScore)
from utils import chem

FAIL = -9999


def process_row(row):
    smiles = row.get("smiles") or ""
    out = {"id": row.get("id"), "smiles": smiles,
           "sascore": FAIL, "npscore": FAIL, "sps": FAIL, "boettcher": FAIL,
           "mw": None, "n_heavy": None, "n_rings": None, "n_stereo": None,
           "n_aromatic_rings": None, "fsp3": None, "tpsa": None}
    if not smiles:
        return out
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return out
    for key, fn in (("sascore", get_SAScore), ("npscore", get_NPScore),
                    ("sps", get_SPScore)):
        try:
            v = fn(mol)
            if v is not None:
                out[key] = v
        except Exception:
            pass
    try:
        b = get_BoettcherScore(smiles)
        if b is not None:
            out["boettcher"] = b
    except Exception:
        pass
    try:
        out.update(chem.descriptor_row(smiles))
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--processes", type=int, default=32)
    args = ap.parse_args()
    run_pool(process_row, read_jsonl(args.input), args.output,
             processes=args.processes, desc="mol-complexity",
             total=count_lines(args.input))


if __name__ == "__main__":
    main()
