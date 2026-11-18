"""
Reaction features from mapped reaction SMILES: reacting atoms, ring and stereo changes, and bonds formed.
"""
from __future__ import annotations
import argparse
import json

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit import Chem

from _common import read_jsonl, run_pool, count_lines
from utils import chem
from complexity_features.rxn_complexity import (
    count_num_react_atom, count_num_react_atom_retro, count_num_change_ring)
from complexity_features.stereochem import check_stereochemistry_changes

_HET = set()  # populated lazily; treat non-C, non-H heavy atoms as X


def _mapped_bond_set(mol):
    """frozenset pairs of atom-map numbers bonded together (mapped atoms only), plus a lookup mapnum->symbol."""
    bonds = set()
    sym = {}
    if mol is None:
        return bonds, sym
    for a in mol.GetAtoms():
        if a.GetAtomMapNum():
            sym[a.GetAtomMapNum()] = a.GetSymbol()
    for b in mol.GetBonds():
        i = b.GetBeginAtom().GetAtomMapNum()
        j = b.GetEndAtom().GetAtomMapNum()
        if i and j:
            bonds.add(frozenset((i, j)))
    return bonds, sym


def count_formed_bonds(rxn_smiles):
    """(#C-C, #C-X) bonds present in product but not in reactants (mapped atoms)."""
    r, _, p = chem.rxn_parts(rxn_smiles)
    if not p:
        return 0, 0
    rmol, pmol = Chem.MolFromSmiles(r), Chem.MolFromSmiles(p)
    rbonds, _ = _mapped_bond_set(rmol)
    pbonds, psym = _mapped_bond_set(pmol)
    formed = pbonds - rbonds
    n_cc = n_cx = 0
    for pair in formed:
        i, j = tuple(pair)
        si, sj = psym.get(i), psym.get(j)
        if si == "C" and sj == "C":
            n_cc += 1
        elif ("C" in (si, sj)) and (si != sj) and si and sj:
            n_cx += 1
    return n_cc, n_cx


def _count_json(s):
    if not s:
        return 0
    try:
        return len(json.loads(s))
    except Exception:
        return 0


def _safe(fn, smi, default=-1):
    """Feature functions assume a well-behaved atom map (every mapped reactant atom recurs in the product); CbyD's mapping does not always guarantee this, so any failure is recorded as ``-1`` (missing) instead of crashing the pool."""
    try:
        return fn(smi)
    except Exception:
        return default


def process_row(row):
    smi = row.get("rxn_smiles") or ""
    out = {"id": row.get("id")}
    out["num_react_atom_forward"] = _safe(count_num_react_atom, smi)
    out["num_react_atom_retro"] = _safe(count_num_react_atom_retro, smi)
    out["num_change_ring"] = _safe(count_num_change_ring, smi)
    try:
        has_a, has_b, atom_changes, bond_changes = check_stereochemistry_changes(smi)
    except Exception:
        has_a, has_b, atom_changes, bond_changes = False, False, "", ""
    out["has_p_atom_stereo"] = bool(has_a)
    out["has_p_bond_stereo"] = bool(has_b)
    out["n_atom_stereo_changes"] = _count_json(atom_changes)
    out["n_bond_stereo_changes"] = _count_json(bond_changes)
    try:
        cc, cx = count_formed_bonds(smi)
    except Exception:
        cc, cx = -1, -1
    out["n_cc_bond_formed"] = cc
    out["n_cx_bond_formed"] = cx
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--processes", type=int, default=32)
    args = ap.parse_args()
    run_pool(process_row, read_jsonl(args.input), args.output,
             processes=args.processes, desc="rxn-features",
             total=count_lines(args.input))


if __name__ == "__main__":
    main()
