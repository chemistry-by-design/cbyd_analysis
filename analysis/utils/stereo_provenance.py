"""Stereocentre provenance for CbyD routes: every assigned centre in the target is ``purchased``, ``installed`` or ``unresolved``, traced back one reaction at a time.
"""
from __future__ import annotations
import json
import os
import re
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")


def _rxn_parts(rxn):
    p = re.split(r"(?<!-)>(?!-)", rxn)
    if len(p) == 3:
        return p[0], p[1], p[2]
    if len(p) == 2:
        return p[0], "", p[1]
    return rxn, "", ""


def _frags(side):
    return [x for x in side.split(".") if x]


def _canon_nomap(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m)


def _main_product(P):
    best, bn = None, -1
    for x in _frags(P):
        m = Chem.MolFromSmiles(x)
        if m and m.GetNumHeavyAtoms() > bn:
            best, bn = x, m.GetNumHeavyAtoms()
    return best


def _assigned_centers(mol):
    """map_num -> CIP for atoms carrying an ASSIGNED tetrahedral stereocentre."""
    out = {}
    for idx, cip in Chem.FindMolChiralCenters(mol, useLegacyImplementation=False,
                                              includeUnassigned=False, force=True):
        mp = mol.GetAtomWithIdx(idx).GetAtomMapNum()
        if mp:
            out[mp] = cip
    return out


def _bridge(frag_mol, prod_mol):
    """map_num(in frag) -> map_num(in prod) for the SAME molecule, aligned by RDKit canonical atom rank with ``breakTies=True``, which is deterministic and also handles symmetric molecules where the equivalent twins give the same verdict either way.
    """
    def ranks(mol):
        m = Chem.Mol(mol)
        for a in m.GetAtoms():
            a.SetAtomMapNum(0)
        return list(Chem.CanonicalRankAtoms(m, breakTies=True))
    rf, rp = ranks(frag_mol), ranks(prod_mol)
    prod_by_rank = {rp[a.GetIdx()]: a.GetAtomMapNum() for a in prod_mol.GetAtoms()}
    out = {}
    for a in frag_mol.GetAtoms():
        r = rf[a.GetIdx()]
        if r in prod_by_rank:
            out[a.GetAtomMapNum()] = prod_by_rank[r]
    return out


def _build(syn):
    rxns = []
    for si, r in enumerate(syn or []):
        mp = (r.get("reaction_smiles_mapped") or "").strip()
        if not mp:
            continue
        R, _A, P = _rxn_parts(mp)
        pmain = _main_product(P)
        if not pmain:
            continue
        rmols = [(x, Chem.MolFromSmiles(x)) for x in _frags(R)]
        rmols = [(x, m) for x, m in rmols if m is not None]
        pmol = Chem.MolFromSmiles(pmain)
        if pmol is None:
            continue
        pcanon = _canon_nomap(pmain)
        rcanon = {_canon_nomap(x) for x, _m in rmols}
        # A step whose product is also one of its reactants is a salt / free-base / protecting-group juggling no-op (mirrors build_network's check_if_reactant_product_same): it must NOT count as a producer, else a centre carried through it would trace to the no-op itself and stall.
        rxns.append(dict(rmols=rmols, pmol=pmol, pcanon=pcanon, rcanon=rcanon,
                         noop=(pcanon in rcanon), syn_idx=si))   # syn_idx = position in synthesis[]
    reactant_canons = set().union(*(rx["rcanon"] for rx in rxns)) if rxns else set()
    prod_canons = {rx["pcanon"] for rx in rxns if not rx["noop"]}
    producers = {}                                             # canon -> [reaction indices]
    for i, rx in enumerate(rxns):
        if not rx["noop"]:
            producers.setdefault(rx["pcanon"], []).append(i)
    return rxns, prod_canons, reactant_canons, producers


def classify_route(syn, max_depth: int = 60):
    """Return a dict for one route, or ``None`` if it has no traceable stereo:
    ``{n, final_heavy, results:[(map_num, cip, verdict, info)]}`` where verdict is 'purchased' | 'installed' | 'unresolved' and info is the leaf SMILES for purchased centres."""
    rxns, prod_canons, reactant_canons, producers = _build(syn)
    if not rxns:
        return None
    terminals = [rx for rx in rxns if rx["pcanon"] not in reactant_canons] or rxns
    final = max(terminals, key=lambda rx: rx["pmol"].GetNumHeavyAtoms())
    final_i = rxns.index(final)

    def reactant_atom(rx, mapnum):
        for x, m in rx["rmols"]:
            for a in m.GetAtoms():
                if a.GetAtomMapNum() == mapnum:
                    return x, m, a
        return None, None, None

    def _next_producer(fragcanon, visited):
        # highest-index producer NOT already on this branch -> skips protect/deprotect cycles (A->BocA->A) that would otherwise loop forever.
        cands = [j for j in producers.get(fragcanon, []) if j not in visited]
        return max(cands) if cands else None

    def classify(ri, mapnum, visited, depth=0):
        if depth > max_depth:
            return ("unresolved", None)
        rx = rxns[ri]
        x, m, a = reactant_atom(rx, mapnum)
        if a is None:
            return ("installed", None)                # atom created in this reaction
        if mapnum not in _assigned_centers(m):
            return ("installed", None)                # atom present, stereo set here
        fragcanon = _canon_nomap(x)
        if fragcanon not in prod_canons:
            return ("purchased", fragcanon)           # carried from a leaf reactant
        rj = _next_producer(fragcanon, visited)
        if rj is None:
            return ("purchased", fragcanon)           # only cyclic producers left -> the leaf form
        nm = _bridge(m, rxns[rj]["pmol"]).get(mapnum)
        if nm is None:
            return ("unresolved", fragcanon)
        return classify(rj, nm, visited | {rj}, depth + 1)

    centers = _assigned_centers(final["pmol"])
    results = [(mp, cip, *(classify(final_i, mp, {final_i}),)) for mp, cip in centers.items()]
    # flatten (verdict, info) tuple
    results = [(mp, cip, v, info) for (mp, cip, (v, info)) in results]
    return dict(n=len(centers), final_heavy=final["pmol"].GetNumHeavyAtoms(),
                n_reactions=len(rxns), results=results)


def explain_route(syn, max_depth: int = 60):
    """Like :func:`classify_route`, but returns per-centre detail for drawing: the mapped target, and for each assigned stereocentre its atom index, verdict and site (the installing reaction, or the source starting material). None if no traceable stereo.
    """
    rxns, prod_canons, reactant_canons, producers = _build(syn)
    if not rxns:
        return None
    terminals = [rx for rx in rxns if rx["pcanon"] not in reactant_canons] or rxns
    final = max(terminals, key=lambda rx: rx["pmol"].GetNumHeavyAtoms())
    final_i = rxns.index(final)

    def reactant_atom(rx, mapnum):
        for x, m in rx["rmols"]:
            for a in m.GetAtoms():
                if a.GetAtomMapNum() == mapnum:
                    return x, m, a
        return None, None, None

    def _next_producer(fragcanon, visited):
        cands = [j for j in producers.get(fragcanon, []) if j not in visited]
        return max(cands) if cands else None

    def classify(ri, mapnum, path, visited, depth=0):
        # `path` records (reaction_index, product-atom map number) for every step this centre passes through, newest (final target) first.
        path.append((ri, mapnum))
        if depth > max_depth:
            return dict(verdict="unresolved")
        rx = rxns[ri]
        x, m, a = reactant_atom(rx, mapnum)
        if a is None:
            return dict(verdict="installed", rxn_idx=ri, install_map=mapnum)   # atom created here
        if mapnum not in _assigned_centers(m):
            return dict(verdict="installed", rxn_idx=ri, install_map=mapnum)   # stereo set here
        fragcanon = _canon_nomap(x)
        if fragcanon not in prod_canons:
            return dict(verdict="purchased", sm=x, sm_map=mapnum)              # leaf starting material
        rj = _next_producer(fragcanon, visited)
        if rj is None:
            return dict(verdict="purchased", sm=x, sm_map=mapnum)              # only cyclic producers left
        nm = _bridge(m, rxns[rj]["pmol"]).get(mapnum)
        if nm is None:
            return dict(verdict="unresolved", sm=x)
        return classify(rj, nm, path, visited | {rj}, depth + 1)

    fmol = final["pmol"]
    idx_by_map = {a.GetAtomMapNum(): a.GetIdx() for a in fmol.GetAtoms()}
    centers = []
    for mp, cip in _assigned_centers(fmol).items():
        path = []
        d = classify(final_i, mp, path, {final_i})
        # report path/rxn_idx as *synthesis* reaction indices (align with build_network)
        d["path"] = [(rxns[ri]["syn_idx"], m) for (ri, m) in path]
        d.update(map_num=mp, cip=cip, final_atom_idx=idx_by_map.get(mp))
        if d["verdict"] == "installed" and "rxn_idx" in d:
            d["rxn_idx"] = rxns[d["rxn_idx"]]["syn_idx"]
            d["rxn_smiles"] = syn_reaction_mapped(syn, d["rxn_idx"])
        centers.append(d)
    return dict(final_mapped=Chem.MolToSmiles(fmol),
                n_reactions=len(rxns), centers=centers)


def syn_reaction_mapped(syn, si):
    """The mapped reaction SMILES for synthesis reaction index ``si`` (or ``None``)."""
    if syn and 0 <= si < len(syn):
        r = syn[si]
        return (r.get("reaction_smiles_mapped") or "").strip() or None
    return None


_OXAZOLIDINONE = Chem.MolFromSmarts("O=C1OC[C]N1")


def _is_chiral_oxazolidinone(smi):
    m = Chem.MolFromSmiles(smi)
    return (m is not None and m.HasSubstructMatch(_OXAZOLIDINONE)
            and len(Chem.FindMolChiralCenters(m, useLegacyImplementation=False,
                                              includeUnassigned=False)) > 0)


def auxiliary_route_row(path):
    """Summary row if the route installs a target stereocentre with a chiral oxazolidinone (Evans) auxiliary that is then removed, else None. Shows that an 'installed' centre can still owe its chirality to the chiral pool.
    """
    try:
        d = json.load(open(path))
    except Exception:
        return None
    syn = d.get("synthesis", [])
    e = explain_route(syn)
    if not e:
        return None
    tmol = Chem.MolFromSmiles(e["final_mapped"])
    if tmol is None or tmol.HasSubstructMatch(_OXAZOLIDINONE):
        return None                                    # auxiliary must be gone from the target
    aux = None
    for c in e["centers"]:
        if c["verdict"] == "installed" and c.get("rxn_smiles"):
            for x in _rxn_parts(c["rxn_smiles"])[0].split("."):
                if _is_chiral_oxazolidinone(x):
                    aux = Chem.MolToSmiles(Chem.MolFromSmiles(x))
                    break
        if aux:
            break
    if not aux:
        return None
    bg = d.get("background", {}) or {}
    verd = [c["verdict"] for c in e["centers"]]
    return dict(route=os.path.basename(path), target=bg.get("name"),
                author=bg.get("author"), year=bg.get("year"),
                n_reactions=e.get("n_reactions"), n_stereo=len(e["centers"]),
                n_installed=verd.count("installed"), n_purchased=verd.count("purchased"),
                aux_smiles=aux)


def classify_file(path):
    """Classify one route JSON -> a flat per-route summary row (or ``None``).
    Module-level so it can be used with ``ProcessPoolExecutor`` from a notebook."""
    try:
        d = json.load(open(path))
    except Exception:
        return None
    res = classify_route(d.get("synthesis", []))
    if not res or res["n"] == 0:
        return None
    verdicts = [v for _, _, v, _ in res["results"]]
    purchased_sms = [info for _, _, v, info in res["results"]
                     if v == "purchased" and info]
    bg = d.get("background", {}) or {}
    return dict(
        route=os.path.basename(path),
        target=bg.get("name"),
        author=bg.get("author"),
        year=bg.get("year"),
        n_reactions=res.get("n_reactions"),
        n_stereo=res["n"],
        n_purchased=verdicts.count("purchased"),
        n_installed=verdicts.count("installed"),
        n_unresolved=verdicts.count("unresolved"),
        purchased_sms=";".join(purchased_sms),
    )
