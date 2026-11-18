"""Commercial availability of molecules, matched against the eMolecules buyable catalogue by canonical SMILES, both with stereochemistry (``iso``) and without (``flat``).
"""
from __future__ import annotations
import os
from concurrent.futures import ProcessPoolExecutor

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")


def canon_iso(smiles: str):
    """RDKit canonical isomeric SMILES, or ``None`` if unparseable."""
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m is not None else None


def canon_flat(smiles: str):
    """RDKit canonical SMILES with stereochemistry removed, or ``None``."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)


# Representation-robust matching -----
# CbyD and eMolecules disagree on how they draw charged / organometallic reagents: an azide is stored as the anion ``[N-]=[N+]=[N-]`` in one and the neutral form in the other; MeLi / Grignards are covalent ``[Li]C`` / ``C[Mg]Br`` in CbyD but ionic ``[CH3-].[Li+]`` / ``[Br-].[CH3-].[Mg+2]`` in eMolecules. So each query molecule is expanded into a few equivalent keys and counted buyable if ANY of them is in the catalogue (matched against the catalogue's plain RDKit canonical SMILES, so ``[Mg++]`` -> ``[Mg+2]`` etc.).
_METALS = {"Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba", "Zn", "Cu", "Al",
           "Fe", "Mn", "Co", "Ni", "Sn", "Cd", "Ce", "Sm", "Ti", "Cr", "Pd", "Ag"}


def _has_metal(m):
    return any(a.GetSymbol() in _METALS for a in m.GetAtoms())


def neutralize_net(smiles: str):
    """Protonate/deprotonate only enough to zero the NET charge (e.g. azide anion ``[N-]=[N+]=[N-]`` -> neutral ``[N-]=[N+]=N``). Returns canonical SMILES/None."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    net = sum(a.GetFormalCharge() for a in m.GetAtoms())
    if net == 0:
        return Chem.MolToSmiles(m)
    rw = Chem.RWMol(m)
    for a in sorted(rw.GetAtoms(), key=lambda a: -abs(a.GetFormalCharge())):
        if net == 0:
            break
        if a.GetSymbol() in _METALS:
            continue
        ch = a.GetFormalCharge()
        if net < 0 and ch < 0:
            a.SetFormalCharge(ch + 1); a.SetNumExplicitHs(a.GetTotalNumHs() + 1)
            a.SetNoImplicit(True); net += 1
        elif net > 0 and ch > 0 and a.GetTotalNumHs() > 0:
            a.SetFormalCharge(ch - 1); a.SetNumExplicitHs(a.GetTotalNumHs() - 1)
            a.SetNoImplicit(True); net -= 1
    mm = rw.GetMol()
    try:
        Chem.SanitizeMol(mm)
        return Chem.MolToSmiles(mm)
    except Exception:
        return None


def metal_disconnect(smiles: str):
    """Break covalent metal-nonmetal bonds into the ionic form (``[Li]C`` -> ``[CH3-].[Li+]``; ``C[Mg]Br`` -> ``[Br-].[CH3-].[Mg+2]``). None if no metal."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    brk = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()
           if (b.GetBeginAtom().GetSymbol() in _METALS)
           != (b.GetEndAtom().GetSymbol() in _METALS)]
    if not brk:
        return None
    rw = Chem.RWMol(m)
    for i, j in brk:
        rw.RemoveBond(i, j)
        for aa, ot in ((rw.GetAtomWithIdx(i), rw.GetAtomWithIdx(j)),
                       (rw.GetAtomWithIdx(j), rw.GetAtomWithIdx(i))):
            if aa.GetSymbol() in _METALS:
                aa.SetFormalCharge(aa.GetFormalCharge() + 1); aa.SetNoImplicit(True)
                ot.SetFormalCharge(ot.GetFormalCharge() - 1)
    mm = rw.GetMol()
    try:
        Chem.SanitizeMol(mm)
        return Chem.MolToSmiles(mm)
    except Exception:
        return None


def ylide_variants(smiles: str):
    """The other drawing of a P/S ylide: charge-separated ``[P+]-[C-]`` <-> ylene ``P=C``, so a query matches whichever form the catalogue stores. Only P/S=C pairs are touched.
    """
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return set()
    out = set()
    exact = Chem.MolToSmiles(m)

    def _emit(rw):
        mm = rw.GetMol()
        try:
            Chem.SanitizeMol(mm)
            s = Chem.MolToSmiles(mm)
            if s and s != exact:
                out.add(s)
        except Exception:
            pass

    # charge-separated -> double bond: [P+]-[C-] => P=C
    rw = Chem.RWMol(m); changed = False
    for b in list(rw.GetBonds()):
        if b.GetBondType() != Chem.BondType.SINGLE:
            continue
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        pos = a1 if a1.GetFormalCharge() > 0 else (a2 if a2.GetFormalCharge() > 0 else None)
        neg = a2 if pos is a1 else (a1 if pos is a2 else None)
        if pos is None or neg is None or neg.GetFormalCharge() >= 0:
            continue
        if pos.GetSymbol() in ("P", "S") and neg.GetSymbol() == "C":
            b.SetBondType(Chem.BondType.DOUBLE)
            pos.SetFormalCharge(pos.GetFormalCharge() - 1)
            neg.SetFormalCharge(neg.GetFormalCharge() + 1)
            changed = True
    if changed:
        _emit(rw)

    # double bond -> charge-separated: P=C => [P+]-[C-]
    rw = Chem.RWMol(m); changed = False
    for b in list(rw.GetBonds()):
        if b.GetBondType() != Chem.BondType.DOUBLE:
            continue
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        het = a1 if a1.GetSymbol() in ("P", "S") else (a2 if a2.GetSymbol() in ("P", "S") else None)
        c = a2 if het is a1 else (a1 if het is a2 else None)
        if het is None or c is None or c.GetSymbol() != "C":
            continue
        if het.GetFormalCharge() != 0 or c.GetFormalCharge() != 0:
            continue
        b.SetBondType(Chem.BondType.SINGLE)
        het.SetFormalCharge(1)
        c.SetFormalCharge(-1)
        changed = True
    if changed:
        _emit(rw)

    return out


# Single-atom counterions that carry no chemical identity of their own: when a reagent is an ion, the salt it's sold as (which counterion) does not matter, so these are stripped before matching. NH4+ / OH- etc. still count on their own (a lone-ion query keeps its ion when nothing else remains).
_SPECTATOR_RAW = ["[Li+]", "[Na+]", "[K+]", "[Rb+]", "[Cs+]", "[Mg+2]", "[Ca+2]",
                  "[Sr+2]", "[Ba+2]", "[Zn+2]", "[Cu+2]", "[Cu+]", "[Fe+2]", "[Fe+3]",
                  "[Mn+2]", "[Co+2]", "[Ni+2]", "[Cd+2]", "[Al+3]", "[Ce+3]", "[Cr+3]",
                  "[Ag+]", "[NH4+]", "[Cl-]", "[Br-]", "[I-]", "[F-]", "[OH-]"]
_SPECTATOR = {s for s in (canon_iso(x) for x in _SPECTATOR_RAW) if s}


def significant_ion_frags(smiles: str):
    """Canonical SMILES of the query's net-charged fragments with generic counterions removed, so a reagent matches whichever salt it is sold as. Empty for neutral molecules and zwitterions.
    """
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return set()
    forms = [m]
    dm = metal_disconnect(smiles)                 # covalent RM -> ionic, so C[Mg]Br gives [CH3-]
    if dm:
        dmm = Chem.MolFromSmiles(dm)
        if dmm is not None:
            forms.append(dmm)
    ions = []
    for mm in forms:
        for fr in Chem.GetMolFrags(mm, asMols=True, sanitizeFrags=False):
            if sum(a.GetFormalCharge() for a in fr.GetAtoms()) != 0:
                try:
                    ions.append(Chem.MolToSmiles(fr))
                except Exception:
                    pass
    if not ions:
        return set()
    sig = {i for i in ions if i not in _SPECTATOR}
    return sig if sig else set(ions)              # lone spectator ion counts as itself


def variant_keys(smiles: str):
    """``(exact, alts)``: the canonical SMILES plus equivalent whole-molecule keys to look up as well (ionic organometallic, net-neutralised, the other ylide drawing).
    """
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None, set()
    exact = Chem.MolToSmiles(m)
    alts = set()
    dm = metal_disconnect(smiles)                 # covalent organometallic -> ionic
    if dm:
        alts.add(dm)
    if not _has_metal(m):                          # neutralise only non-metal ions
        nn = neutralize_net(smiles)                # (azide anion -> HN3, etc.)
        if nn and nn != exact:
            alts.add(nn)
    alts |= ylide_variants(smiles)                # [P+]-[C-] <-> P=C
    alts.discard(exact)
    return exact, alts


def _scan_chunk(job):
    """Worker: scan one byte range of the catalogue, return the query keys hit."""
    path, start, end, iso_targets, flat_targets, frag_targets = job
    hit_iso, hit_flat, hit_frag = set(), set(), set()
    do_flat = bool(flat_targets)
    do_frag = bool(frag_targets)
    with open(path, "rb") as f:
        f.seek(start)
        if start > 0:
            f.readline()                       # drop the partial line owned by the previous chunk
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            smi = line.split(b" ", 1)[0].decode("ascii", "ignore").strip()
            if not smi:
                continue
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            iso = Chem.MolToSmiles(m)
            if iso in iso_targets:
                hit_iso.add(iso)
            if do_frag and any(a.GetFormalCharge() for a in m.GetAtoms()):
                # counterion-agnostic: match this salt's net-charged fragment(s)
                for fr in Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False):
                    if sum(a.GetFormalCharge() for a in fr.GetAtoms()) != 0:
                        try:
                            fs = Chem.MolToSmiles(fr)
                        except Exception:
                            continue
                        if fs in frag_targets:
                            hit_frag.add(fs)
            if do_flat:
                Chem.RemoveStereochemistry(m)
                flat = Chem.MolToSmiles(m)
                if flat in flat_targets:
                    hit_flat.add(flat)
    return hit_iso, hit_flat, hit_frag


def buyable_scan(iso_targets, flat_targets, emol_path, n_workers: int = 64,
                 frag_targets=None):
    """Return ``(found_iso, found_flat, found_frag)`` — the subsets of the query key sets that occur in the eMolecules catalogue. ``flat_targets`` may be empty to skip the (2x cost) stereo-removed pass; ``frag_targets`` (net-charged reagent fragments) enables counterion-agnostic ion matching."""
    iso_targets = set(iso_targets)
    flat_targets = set(flat_targets)
    frag_targets = set(frag_targets or ())
    size = os.path.getsize(emol_path)
    n = max(1, n_workers)
    step = size // n
    jobs = [(emol_path, i * step, size if i == n - 1 else (i + 1) * step,
             iso_targets, flat_targets, frag_targets) for i in range(n)]
    found_iso, found_flat, found_frag = set(), set(), set()
    with ProcessPoolExecutor(max_workers=n) as ex:
        for hi, hf, hg in ex.map(_scan_chunk, jobs):
            found_iso |= hi
            found_flat |= hf
            found_frag |= hg
    return found_iso, found_flat, found_frag
