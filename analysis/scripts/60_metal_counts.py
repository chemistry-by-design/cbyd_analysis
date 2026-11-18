"""Metal-reagent counts per dataset, read from the agent channel of every reaction.
"""
from __future__ import annotations
import argparse
import collections
import csv
import os

from tqdm import tqdm
from multiprocessing import Pool

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from _common import read_jsonl, count_lines
from utils import chem
from utils.paths import a, ensure_output_dirs


def metals_of(row):
    _, agents, _ = chem.rxn_parts(row.get("rxn_smiles", ""))
    out = []
    for frag in agents.split("."):
        if frag:
            out.extend(chem.metals_in_smiles(frag))
    return out


def metals_with_id(row):
    return row.get("id"), metals_of(row)


def metals_year_of(row):
    """(year, [metals]) of one reaction, for the metal × year counts (``--by-year``)."""
    y = row.get("year")
    try:
        y = int(float(y))
    except (TypeError, ValueError):
        y = None
    return y, sorted(set(metals_of(row)))


def count_dataset(name, processes):
    inp = a("reactions", f"{name}_reactions.jsonl")
    if not os.path.exists(inp):
        print(f"[skip] {name}: {inp} not found")
        return
    # CbyD also gets a count over its unique reactions (one-step comparison)
    uniq_csv = a("reactions", "cbyd_unique_reactions.csv")
    uniq = None
    if name == "cbyd" and os.path.exists(uniq_csv):
        with open(uniq_csv) as f:
            uniq = {r["id"] for r in csv.DictReader(f) if r["first"] == "True"}
    counter, counter_u = collections.Counter(), collections.Counter()
    total = count_lines(inp)
    with Pool(processes=processes) as pool:
        for rid, metals in tqdm(pool.imap(metals_with_id, read_jsonl(inp), chunksize=256),
                                total=total, desc=f"metals({name})"):
            counter.update(metals)
            if uniq is not None and rid in uniq:
                counter_u.update(metals)
    outs = [(f"{name}_metals.csv", counter)] + ([("cbyd_unique_metals.csv", counter_u)] if uniq else [])
    for fname, cnt in outs:
        out = a("reaction_conditions", fname)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["metal", "count"])
            for m, c in cnt.most_common():
                w.writerow([m, c])
        print(f"wrote {len(cnt)} metals -> {out}")


def count_by_year(name, processes):
    """metal × year usage for a dataset (needs a per-reaction ``year``; CbyD has it).
    Writes ``year, metal, count`` plus a ``__all__`` metal row = #reactions/year (denominator for per-year normalisation)."""
    inp = a("reactions", f"{name}_reactions.jsonl")
    if not os.path.exists(inp):
        print(f"[skip] {name}: {inp} not found")
        return
    my = collections.Counter()      # (year, metal) -> count
    per_year = collections.Counter()  # year -> #reactions
    total = count_lines(inp)
    with Pool(processes=processes) as pool:
        for year, metals in tqdm(pool.imap(metals_year_of, read_jsonl(inp), chunksize=256),
                                 total=total, desc=f"metal-by-year({name})"):
            if year is None:
                continue
            per_year[year] += 1
            for m in metals:
                my[(year, m)] += 1
    out = a("reaction_conditions", f"{name}_metal_by_year.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["year", "metal", "count"])
        for y, n in sorted(per_year.items()):
            w.writerow([y, "__all__", n])
        for (y, m), c in sorted(my.items()):
            w.writerow([y, m, c])
    print(f"wrote metal-by-year ({len(my)} cells) -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=32)
    ap.add_argument("--datasets", nargs="+",
                    default=["cbyd", "uspto_full", "pistachio"])
    ap.add_argument("--by-year", nargs="*", default=["cbyd"],
                    help="datasets to also emit metal×year tables for (need a year field)")
    args = ap.parse_args()
    ensure_output_dirs()
    for name in args.datasets:
        count_dataset(name, args.processes)
    for name in (args.by_year or []):
        count_by_year(name, args.processes)


if __name__ == "__main__":
    main()
