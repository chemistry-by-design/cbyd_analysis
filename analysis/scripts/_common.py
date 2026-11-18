"""
Every metric script maps a pure ``process_row(dict) -> dict`` over an input table with a ``multiprocessing.Pool`` and streams the result to CSV.
"""
from __future__ import annotations
import csv
import json
import os
import sys
from multiprocessing import Pool

from tqdm import tqdm

# make ``utils`` importable (analysis/ on the path)
_ANALYSIS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ANALYSIS not in sys.path:
    sys.path.insert(0, _ANALYSIS)


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            n += 1
    print(f"wrote {n:,} rows -> {path}")
    return n


def count_lines(path):
    with open(path) as f:
        return sum(1 for _ in f)


def run_pool(process_row, rows, output_csv, processes=32, desc="processing",
             total=None, chunksize=64):
    """Map ``process_row`` over ``rows`` and stream dict results to ``output_csv``.

    ``rows`` is iterated lazily (a generator is fine and preferred for large inputs — nothing is materialised in memory). Pass ``total`` for a progress bar.
    Returns the number of rows written."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    written = 0
    with open(output_csv, "w", newline="") as outfile:
        writer = None
        with Pool(processes=processes) as pool:
            for res in tqdm(pool.imap(process_row, rows, chunksize=chunksize),
                            total=total, desc=desc):
                if res is None:
                    continue
                if writer is None:
                    writer = csv.DictWriter(outfile, fieldnames=list(res.keys()))
                    writer.writeheader()
                writer.writerow(res)
                written += 1
    print(f"wrote {written:,} rows -> {output_csv}")
    return written


def stream_jsonl_pool(process_row, rows, output_jsonl, processes=32,
                      desc="processing", total=None, chunksize=200):
    """Like ``run_pool`` but streams dict results to a JSONL file."""
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    written = 0
    with open(output_jsonl, "w") as out:
        with Pool(processes=processes) as pool:
            for res in tqdm(pool.imap(process_row, rows, chunksize=chunksize),
                            total=total, desc=desc):
                if res is None:
                    continue
                out.write(json.dumps(res) + "\n")
                written += 1
    print(f"wrote {written:,} rows -> {output_jsonl}")
    return written
