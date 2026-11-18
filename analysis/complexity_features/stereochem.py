# Tran, S. B. A.; Roh, J.; Coley, C. W. Quantifying the Failure Modes of Current One-step Retrosynthesis Models. ChemRxiv 2026. DOI: 10.26434/chemrxiv.15000071/v1
# https://github.com/suongbatran/one_step_retro_failure_mode

import sys
sys.path.append("../")


import json
import logging
from typing import Set
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

RS_TAG = ["R", "S", "None"]
RS_TAG_DICT = {rs: i for i, rs in enumerate(RS_TAG)}


def analyze_molecule(smiles: str) -> Set[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Return empty sets and 0 for invalid SMILES
        return set(), set(), 0
        
    atom_types_mapped = set()
    atom_types_unmapped = set()
    num_unmapped_atoms = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum():
            atom_types_mapped.add(atom.GetAtomicNum())
        else:
            atom_types_unmapped.add(atom.GetAtomicNum())
            num_unmapped_atoms += 1
    return atom_types_mapped, atom_types_unmapped, num_unmapped_atoms

unspecified_atom = Chem.ChiralType.CHI_UNSPECIFIED
unspecified_bond = Chem.BondStereo.STEREONONE

def has_atom_stereochemistry(atom):
    return atom.GetChiralTag() != unspecified_atom

def has_bond_stereochemistry(bond):
    return bond.GetStereo() != unspecified_bond

def has_atom_stereo_from_smiles(smiles):

    return "@" in smiles

def has_bond_stereo_from_smiles(smiles):
    return "/" in smiles or "\\" in smiles

def get_rs_tag(atom):

    return atom.GetPropsAsDict().get("_CIPCode", "None")

sp3 = Chem.rdchem.HybridizationType.SP3
double = Chem.rdchem.BondType.DOUBLE


def _unmapped_copy(mol):
    """Copy of ``mol`` with atom-map numbers removed and stereo re-perceived, so only REAL stereo elements keep their tag (RDKit's legacy ranking otherwise breaks ties on map numbers, which keeps tags on non-stereocentres and makes R/S / E/Z labels depend on the arbitrary numbering). Indices are unchanged."""
    m = Chem.Mol(mol)
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    return m


# Stereo changes are compared by CONFIGURATION relative to mapped neighbours, not by CIP label: R/S and E/Z descriptors flip whenever substituent priorities change (epoxide opening, C=O -> C=NOH, ...) although the stereocentre is untouched.
_TETRA = (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
_TRANS = (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS)
_CIS = (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS)


def _perm_parity(a, b):
    """0 if list ``b`` is an even permutation of list ``a``, else 1."""
    b, p = list(b), 0
    for i, x in enumerate(a):
        j = b.index(x, i)
        if j != i:
            b[i], b[j] = b[j], b[i]
            p ^= 1
    return p


def _tetra(un, mapped, idx):
    """(chiral tag, neighbour map numbers in bond order) if atom ``idx`` is a specified tetrahedral stereocentre, else None."""
    a = un.GetAtomWithIdx(idx)
    if a.GetChiralTag() not in _TETRA:
        return None
    return a.GetChiralTag(), [mapped.GetAtomWithIdx(b.GetOtherAtomIdx(idx)).GetAtomMapNum()
                              for b in a.GetBonds()]


def _tetra_change(r, p):
    """created / destroyed / inverted / bond changed at the centre; None if the configuration is retained or cannot be compared (unmapped neighbour)."""
    if r is None and p is None:
        return None
    if r is None:
        return "created"
    if p is None:
        return "destroyed"
    (rt, rn), (pt, pn) = r, p
    if 0 in rn or 0 in pn or len(set(rn)) < len(rn) or len(set(pn)) < len(pn):
        return None
    if set(rn) != set(pn):
        return "bond changed at centre"
    same = (rt == pt) != bool(_perm_parity(rn, pn))
    return None if same else "inverted"


def _double_bond(un, mapped, bidx):
    """None if bond ``bidx`` has no specified E/Z; "undet" if a substituent is unmapped; else (trans?, {end-atom map: substituent maps}) with trans? taken between the LOWEST-mapped substituent at each end (a fixed, label-free reference)."""
    b = un.GetBondWithIdx(bidx)
    st = b.GetStereo()
    if st not in _TRANS + _CIS:
        return None
    trans = st in _TRANS                              # relative to the CIP stereo atoms
    subs = {}
    for at, other, s in ((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetStereoAtoms()[0]),
                         (b.GetEndAtomIdx(), b.GetBeginAtomIdx(), b.GetStereoAtoms()[1])):
        nb = {mapped.GetAtomWithIdx(n.GetIdx()).GetAtomMapNum(): n.GetIdx()
              for n in un.GetAtomWithIdx(at).GetNeighbors() if n.GetIdx() != other}
        if 0 in nb:
            return "undet"
        if nb[min(nb)] != s:                          # reference = the other substituent
            trans = not trans
        subs[mapped.GetAtomWithIdx(at).GetAtomMapNum()] = frozenset(nb)
    return trans, subs


def _bond_change(r, p):
    """created / destroyed / inverted (E<->Z); None if retained or not comparable (unmapped substituent, or the substituents themselves changed)."""
    if "undet" in (r, p) or (r is None and p is None):
        return None
    if r is None:
        return "created"
    if p is None:
        return "destroyed"
    if r[1] != p[1]:
        return None
    return None if r[0] == p[0] else "inverted"


def check_stereochemistry_changes(smiles):

    # Counts mapped stereocentres / stereo double bonds whose configuration changes (created, destroyed, inverted, or - for centres - a bond made/broken at it).

    try:
        reactants, _, products = smiles.split(">")
        r_mol = Chem.MolFromSmiles(reactants)
        p_mol = Chem.MolFromSmiles(products)
        r_un, p_un = _unmapped_copy(r_mol), _unmapped_copy(p_mol)

        has_p_atom_stereo = any(
            has_atom_stereochemistry(atom) 
            for atom in p_mol.GetAtoms()
        )
        has_p_bond_stereo = any(
            has_bond_stereochemistry(bond)
            for bond in p_mol.GetBonds()
        )

        atom_changes = {}
        bond_changes = {}

        # Map atom map numbers to atoms in reactants
        r_atoms = {atom.GetAtomMapNum(): atom for atom in r_mol.GetAtoms() if atom.GetAtomMapNum() > 0}
        p_atoms = {atom.GetAtomMapNum(): atom for atom in p_mol.GetAtoms() if atom.GetAtomMapNum() > 0}

        # Check for tetrahedral configuration changes
        for map_num, p_atom in p_atoms.items():
            r_atom = r_atoms.get(map_num)
            if not r_atom:
                continue
            change = _tetra_change(_tetra(r_un, r_mol, r_atom.GetIdx()),
                                   _tetra(p_un, p_mol, p_atom.GetIdx()))
            if change:
                atom_changes[map_num] = change

        # Create a map from (sorted atom map numbers) to bond in reactants
        def get_bond_key(bond, mol):
            begin_map = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum()
            end_map = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum()
            if begin_map == 0 or end_map == 0:
                return None
            return "-".join(map(str, sorted([begin_map, end_map])))

        r_bonds = {
            get_bond_key(bond, r_mol): bond
            for bond in r_mol.GetBonds()
            if get_bond_key(bond, r_mol)
        }

        # Compare double-bond configuration (bonds formed in this reaction are skipped)
        for p_bond in p_mol.GetBonds():
            bond_key = get_bond_key(p_bond, p_mol)
            if not bond_key or bond_key not in r_bonds:
                continue
            change = _bond_change(_double_bond(r_un, r_mol, r_bonds[bond_key].GetIdx()),
                                  _double_bond(p_un, p_mol, p_bond.GetIdx()))
            if change:
                bond_changes[bond_key] = change
        atom_changes = json.dumps(atom_changes) if atom_changes else ""
        bond_changes = json.dumps(bond_changes) if bond_changes else ""

        return has_p_atom_stereo, has_p_bond_stereo, atom_changes, bond_changes
    except Exception as e:
        

        # Default to checking the smiles if error in creating mol obkect
        r, _, p = smiles.split(">")
        has_p_atom_stereo = has_atom_stereo_from_smiles(p)
        has_p_bond_stereo = has_bond_stereo_from_smiles(p)

        logging.error(f"Error checking stereochemistry inversion: {e} with smiles: {smiles}, "
            f"product check result - atom: {has_p_atom_stereo}, bond: {has_p_bond_stereo}")

        return has_p_atom_stereo, has_p_bond_stereo, "", ""
