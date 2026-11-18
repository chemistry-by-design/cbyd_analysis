# Scorers from:
# Tran, S. B. A.; Roh, J.; Coley, C. W. Quantifying the Failure Modes of Current One-step Retrosynthesis Models. ChemRxiv 2026. DOI: 10.26434/chemrxiv.15000071/v1
# https://github.com/suongbatran/one_step_retro_failure_mode

from __future__ import annotations
import argparse
import csv
import json
import os
import sys

# make the ``complexity_features`` package importable regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit import Chem

from complexity_features.product_complexity import (
    get_SAScore, get_NPScore, get_SPScore, get_BoettcherScore)

FAIL = -9999
METRICS = ["sascore", "npscore", "sps", "boettcher"]


def score(smiles: str) -> dict:
    """{smiles, sascore, npscore, sps, boettcher}; failed metrics -> FAIL (-9999)."""
    out = {"smiles": smiles, **{m: FAIL for m in METRICS}}
    mol = Chem.MolFromSmiles(smiles) if smiles else None
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
    return out


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL with a 'smiles' field per line")
    ap.add_argument("--output", required=True)
    ap.add_argument("--processes", type=int, default=16)
    args = ap.parse_args()

    smis = [json.loads(l)["smiles"] for l in open(args.input) if l.strip()]
    with mp.Pool(args.processes) as pool:
        rows = pool.map(score, smis)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["smiles"] + METRICS)
        w.writeheader()
        w.writerows(rows)
    print(f"scored {len(rows):,} SMILES -> {args.output}")


if __name__ == "__main__":
    main()
