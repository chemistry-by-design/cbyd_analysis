"""
Deduplicate CbyD reactions on their canonical map-free ``reactants>>products``, as USPTO-Full and Pistachio were, for the one-step comparison.
"""
from __future__ import annotations
import argparse
import collections
import csv
from multiprocessing import Pool

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from tqdm import tqdm

from _common import read_jsonl, count_lines
from utils import chem
from utils.paths import a, ensure_output_dirs


def onestep_key(row):
    r, _, p = chem.rxn_parts(row.get("rxn_smiles", ""))
    rc, pc = chem.canonicalize(r), chem.canonicalize(p)
    return row["id"], (f"{rc}>>{pc}" if rc and pc else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=32)
    args = ap.parse_args()
    ensure_output_dirs()
    inp = a("reactions", "cbyd_reactions.jsonl")
    with Pool(args.processes) as pool:                 # imap keeps route order
        rows = list(tqdm(pool.imap(onestep_key, read_jsonl(inp), chunksize=256),
                         total=count_lines(inp), desc="dedup-key"))
    n_copies = collections.Counter(k or f"unparsed:{i}" for i, k in rows)
    seen, out = set(), a("reactions", "cbyd_unique_reactions.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "key", "first", "n_copies"])
        for i, k in rows:
            kk = k or f"unparsed:{i}"
            w.writerow([i, k, kk not in seen, n_copies[kk]])
            seen.add(kk)
    print(f"{len(rows):,} CbyD reactions -> {len(seen):,} unique reactants>>product "
          f"({sum(not k for _, k in rows)} unparsable) -> {out}")


if __name__ == "__main__":
    main()
