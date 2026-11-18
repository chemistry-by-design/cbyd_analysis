"""Canonical filesystem locations for every dataset and analysis output. ``REPO_ROOT`` is resolved from this file, so the working directory does not matter.
"""
from __future__ import annotations
import os

# analysis/utils/paths.py  ->  repo root is two levels up.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# CbyD (this repo) -----
CBYD_ROUTES_DIR = os.path.join(REPO_ROOT, "data", "routes")

# route-level comparison datasets (shipped in the repo) -----
OTHER_DATA_DIR = os.path.join(REPO_ROOT, "data", "other_data")
PAROUTES_JSON = os.path.join(OTHER_DATA_DIR, "paroutes_n1.json")   # 10k AiZynth trees
USPTO190_PKL = os.path.join(OTHER_DATA_DIR, "uspto190.pkl")        # 190 linear routes
EMOLECULES_SMI = os.path.join(OTHER_DATA_DIR, "emolecules_buyable.smi")  # 33.6M buyable SMILES
# ABSynth: TSV, one reaction (step) per row, grouped by PathId (route-level, 3k routes)
ABSYNTH_TSV = os.path.join(OTHER_DATA_DIR, "ABSynth_Dataset.csv")

# one-step reaction comparison datasets (large; data/ is git-ignored) -----
# JSON-lines, each row: {"id", "canID", "rxn_smiles", "name_rxn"}.
USPTO_FULL_JSONL = os.path.join(OTHER_DATA_DIR, "uspto_clean.json")
PISTACHIO_JSONL = os.path.join(OTHER_DATA_DIR, "pistachio2025Q2_clean.json")

# analysis outputs (precomputed feature tables the notebooks read) -----
DATA_ANALYSIS_DIR = os.path.join(REPO_ROOT, "data", "data_analysis")
FIGURE_DIR = os.path.join(DATA_ANALYSIS_DIR, "figures")

# NameRxn CLI (HazELNut) — used to classify CbyD reactions (USPTO/Pistachio already carry a ``name_rxn`` field, so only CbyD needs classifying).
# Looked up in order: $NAMERXN_BIN, the active conda env's bin/ (install it there with
# ln -s <repo>/HazELNut/namerxn "$CONDA_PREFIX/bin/namerxn"   — not in environment.yml), then the repo's own HazELNut/ (git-ignored: licensed NextMove software, Linux x86-64 build).
def _find_namerxn() -> str:
    import sys
    for p in (os.environ.get("NAMERXN_BIN"),
              os.path.join(sys.prefix, "bin", "namerxn"),
              os.path.join(REPO_ROOT, "HazELNut", "namerxn")):
        if p and os.path.exists(p):
            return p
    return os.path.join(REPO_ROOT, "HazELNut", "namerxn")


NAMERXN_BIN = _find_namerxn()


def ensure_output_dirs() -> None:
    os.makedirs(DATA_ANALYSIS_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)


def rel(path: str) -> str:
    """``path`` relative to the repo root — for printing, so logs and notebook outputs never show a machine-specific absolute path."""
    return os.path.relpath(path, REPO_ROOT)


def a(*parts: str) -> str:
    """Shorthand for a path inside ``data/data_analysis/``."""
    return os.path.join(DATA_ANALYSIS_DIR, *parts)
