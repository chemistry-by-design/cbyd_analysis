"""Reaction-class labels (NameRxn / RXNO) per dataset, reduced to the first-level class.
"""
from __future__ import annotations
import argparse
import csv
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

from tqdm import tqdm

from _common import read_jsonl
from utils import chem
from utils.paths import NAMERXN_BIN

BATCH = 2000


def first_level_class(name_rxn):
    """First-level RXNO class, or a mixed-class label.

    ``name_rxn`` may join several named reactions with ';', e.g.
    ``'3.5.3 Negishi coupling;6.2.2 CO2H-Me deprotection'``. Convention: a single class gives its first-level digit ('2.7.2 ...' -> '2'); several classes sharing an L1 give 'mix sub-classes'; several with differing L1 give 'mix classes'; empty or unrecognised gives '0'.
    """
    if not name_rxn:
        return "0"
    firsts = []
    for part in name_rxn.split(";"):
        part = part.strip()
        if not part:
            continue
        head = part.split()[0].split(".")[0]   # '3.5.3 Negishi' -> '3'
        firsts.append(head if head.isdigit() else "0")
    if not firsts:
        return "0"
    if len(firsts) > 1:
        return "mix sub-classes" if all(f == firsts[0] for f in firsts) else "mix classes"
    return firsts[0]


def _canon_rxn(rxn_smiles):
    """Canonical, map-stripped ``reactants>agents>products`` for the CLI."""
    parts = rxn_smiles.split(">")
    if len(parts) != 3:
        return ""
    out = []
    for part in parts:
        frags = [chem.canonicalize(f, remove_atom_maps=True) for f in part.split(".") if f]
        out.append(".".join(f for f in frags if f))
    return ">".join(out)


def _parse_rdf(path, n):
    """Parse namerxn RDF. A reaction with several recognised transformations emits a primary ``RXN_NAME`` plus one or more ``AUX_RXN_NAME`` fields; we join them with ';' (primary first) to reproduce the multi-class format the one-step datasets use (e.g. '2.5.7 ...;6.1.28 ...'). ``num`` keeps the primary RXN_CLASS as a fallback."""
    names = [[] for _ in range(n)]     # ordered: primary RXN_NAME, then each AUX
    primary_class = [None] * n
    idx, dtype = -1, None
    try:
        with open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("$RFMT"):
                    idx += 1; dtype = None
                elif line.startswith("$DTYPE") and 0 <= idx < n:
                    dtype = line[6:].strip().lower()
                elif line.startswith("$DATUM") and dtype and 0 <= idx < n:
                    val = line[6:].strip() or None
                    if val and dtype in ("rxn_name", "aux_rxn_name"):
                        names[idx].append(val)
                    elif val and dtype == "rxn_class":
                        primary_class[idx] = val
                    dtype = None
    except FileNotFoundError:
        pass
    return [{"name": ";".join(names[i]) if names[i] else None,
             "num": primary_class[i]} for i in range(n)]


def run_namerxn(smiles_list):
    fd_in, ip = tempfile.mkstemp(suffix=".smi")
    fd_out, op = tempfile.mkstemp(suffix=".rdf")
    os.close(fd_in); os.close(fd_out)
    try:
        with open(ip, "w") as f:
            for s in smiles_list:
                f.write((s or "") + "\n")
        subprocess.run([NAMERXN_BIN, "-ismi", "-ordf", "-matching", "-addrxnname",
                        ip, op], capture_output=True, text=True, timeout=1200)
        return _parse_rdf(op, len(smiles_list))
    except Exception as e:
        print(f"[namerxn] batch failed: {e}")
        return [{"name": None, "num": None} for _ in smiles_list]
    finally:
        for p in (ip, op):
            try: os.unlink(p)
            except OSError: pass


def classify_cbyd(rows, output, processes=32, cli_workers=16):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    # canonicalisation (map-strip + double round-trip per fragment) — parallelise it across all reactions first, then batch through the CLI.
    smis = [r.get("rxn_smiles", "") for r in rows]
    with Pool(processes=processes) as pool:
        canon = list(tqdm(pool.imap(_canon_rxn, smis, chunksize=200),
                          total=len(smis), desc="canon(cbyd)"))
    # each CLI batch is single-threaded (~1 min / 2000 rxns), so run batches concurrently; ``map`` keeps batch order so output rows stay in input order.
    starts = list(range(0, len(rows), BATCH))
    with ThreadPoolExecutor(max_workers=cli_workers) as ex:
        results = list(tqdm(ex.map(lambda i: run_namerxn(canon[i:i + BATCH]), starts),
                            total=len(starts), desc="namerxn(cbyd)"))
    with open(output, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "name_rxn", "class_1"])
        for i, res in zip(starts, results):
            batch = rows[i:i + BATCH]
            for r, rr in zip(batch, res):
                name = rr["name"] or rr["num"] or "0.0 Unrecognized"
                w.writerow([r.get("id"), name, first_level_class(name)])
    print(f"wrote -> {output}")


def classify_onestep(rows, output):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "name_rxn", "class_1"])
        for r in tqdm(rows, desc="namerxn(onestep)"):
            name = r.get("name_rxn") or ""
            w.writerow([r.get("id"), name, first_level_class(name)])
    print(f"wrote -> {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--source", choices=["cbyd", "onestep"], required=True)
    ap.add_argument("--processes", type=int, default=32)
    ap.add_argument("--cli-workers", type=int, default=16,
                    help="namerxn CLI batches to run concurrently")
    args = ap.parse_args()
    if args.source == "cbyd":
        classify_cbyd(list(read_jsonl(args.input)), args.output, args.processes,
                      args.cli_workers)
    else:
        classify_onestep(read_jsonl(args.input), args.output)      # stream (no CLI)


if __name__ == "__main__":
    main()
