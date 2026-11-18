"""Horizontal snake route-tree SVG: Graphviz lays the tree out left to right, then the rank columns are cut into page-wide bands and every other band is mirrored.
"""
import html
import math
import os
import re
from collections import defaultdict

import graphviz
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D

from . import draw as dr

from . import chem as _chem
build_network = _chem.build_network
_abbreviate = _chem.abbreviate_mol                                  # protecting-group abbreviations
_MAPPED = re.compile(r":\d+\]")                                    # an atom-map number

ROLE_COLOR = {"target": "#d62728", "start": "#2ca02c", "intermediate": "#1f77b4"}
RXN_BOX_IN = (0.26, 0.11)                            # reaction box (w, h), inches
RANK_GAP_IN = 0.14                                   # graphviz ranksep: gap between columns, inches


def default_rxn_style(index):
    """Plain yellow reaction box, black number."""
    return "#ffd400", "#b39400", 0.5, "black"


def mol_svg(mol, bond_in=dr.BOND_LENGTH_PT / 72, font_pt=dr.LABEL_PT, abbreviate=True,
            highlight=None, highlight_bonds=False, keep_coords=False, wedges=None):
    """House-layout molecule cropped to its ink: ``(svg body, viewBox, width_in, height_in)``.
    ``abbreviate`` condenses common groups, ``highlight`` is {atom index: (r, g, b)}, ``keep_coords`` draws the mol's own conformer and ``wedges`` its wedge plan.
    """
    m = Chem.MolFromSmiles(mol) if isinstance(mol, str) else Chem.Mol(mol)
    # no heavy atoms (H2, hydride, proton -- reagents listed as SMs): RDKit strips the Hs, leaving an empty drawing whose canvas comes out -2147483648 px wide
    if m is None or not m.GetNumHeavyAtoms():
        return None
    if not (keep_coords and m.GetNumConformers()):
        m = dr.house_layout(m)                       # house layout (same as SI/top15_heterocycles)
    else:
        _unit_bonds(m)                               # mixed RDKit (1.5) / CoordGen (1.0) scales
        if wedges:                                   # consistent wedge/dash down the tree
            _chem.apply_wedges(m, [tuple(int(x) for x in w) for w in wedges])
    for at in m.GetAtoms():                          # track atoms through the abbreviation
        at.SetIntProp("_src", at.GetIdx())
    if abbreviate:
        m = _abbreviate(m)[0] or m
    hl = {}
    if highlight:
        hl = {at.GetIdx(): highlight[at.GetIntProp("_src")] for at in m.GetAtoms()
              if at.HasProp("_src") and at.GetIntProp("_src") in highlight}
    d = rdMolDraw2D.MolDraw2DSVG(-1, -1)
    o = dr.apply_house_style(d.drawOptions(), bond_in * 72, font_pt)   # ACS 1996 house style
    o.clearBackground = False                        # no opaque white box behind the molecule
    hb = ({b.GetIdx(): hl[b.GetBeginAtomIdx()] for b in m.GetBonds()
           if b.GetBeginAtomIdx() in hl and b.GetEndAtomIdx() in hl} if highlight_bonds else {})
    if keep_coords and wedges:                       # honour the wedge plan, don't recompute it
        m = rdMolDraw2D.PrepareMolForDrawing(m, wedgeBonds=False)
        d.DrawMolecule(m, highlightAtoms=list(hl), highlightAtomColors=hl,
                       highlightBonds=list(hb), highlightBondColors=hb)
    else:
        rdMolDraw2D.PrepareAndDrawMolecule(d, m, highlightAtoms=list(hl), highlightAtomColors=hl,
                                           highlightBonds=list(hb), highlightBondColors=hb)
    d.FinishDrawing(); s = dr.thin_hashed_bonds(d.GetDrawingText())   # ChemDraw-like 4 dashes per hash
    wpx = float(re.search(r"width=['\"]([\d.]+)px", s).group(1))
    hpx = float(re.search(r"height=['\"]([\d.]+)px", s).group(1))
    body = re.search(r"<svg[^>]*>(.*)</svg>", s, re.S).group(1)
    bls = sorted(math.hypot(d.GetDrawCoords(b.GetEndAtomIdx()).x - d.GetDrawCoords(b.GetBeginAtomIdx()).x,
                            d.GetDrawCoords(b.GetEndAtomIdx()).y - d.GetDrawCoords(b.GetBeginAtomIdx()).y)
                 for b in m.GetBonds())
    # no bonds (H2O, salts, ions): fall back to the SUPERSAMPLED drawn bond length -- without the x RDKIT_SUPERSAMPLE these molecules came out 10x too large
    dbl = bls[len(bls) // 2] if bls else bond_in * 72 * dr.RDKIT_SUPERSAMPLE
    ipp = bond_in / dbl
    # every x,y pair of every path (bonds use "x,y", glyph outlines use "x y" -- read both, so long abbreviation labels like TBDPS / MeO are inside the crop box)
    pairs = []
    for dd in re.findall(r"d=['\"]([^'\"]+)['\"]", body):
        nums = re.findall(r"-?\d+(?:\.\d+)?", dd)
        pairs += list(zip(nums[0::2], nums[1::2]))
    pairs += [(mt.group(1), mt.group(2)) for mt in
              re.finditer(r"<text[^>]*\bx=['\"]([\d.]+)['\"][^>]*\by=['\"]([\d.]+)['\"]", body)]
    if pairs:
        xs = [float(x) for x, _ in pairs]; ys = [float(y) for _, y in pairs]; mrg = 0.25 * dbl
        vbx, vby = min(xs) - mrg, min(ys) - mrg
        iw, ih = (max(xs) - min(xs)) + 2 * mrg, (max(ys) - min(ys)) + 2 * mrg
    else:
        vbx, vby, iw, ih = 0.0, 0.0, wpx, hpx
    return body, f"{vbx:.2f} {vby:.2f} {iw:.2f} {ih:.2f}", iw * ipp, ih * ipp


def _unit_bonds(m):
    """Scale `m`'s 2D conformer in place to a median bond length of 1 (the house layout's):
    the labels are a fixed size, so a structure laid out at another scale (RDKit's 1.5 vs CoordGen's 1.0) would otherwise come out with smaller / larger letters."""
    conf = m.GetConformer()
    ls = sorted((conf.GetAtomPosition(b.GetBeginAtomIdx()) - conf.GetAtomPosition(b.GetEndAtomIdx())).Length()
                for b in m.GetBonds())
    k = 1.0 / ls[len(ls) // 2] if ls and ls[len(ls) // 2] > 1e-6 else 1.0
    if abs(k - 1) > 1e-3:
        for i in range(m.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, (p.x * k, p.y * k, 0.0))


def node_mol(n):
    """A tree node's molecule with its (aligned) 2D coordinates, or None."""
    mb = n.get("molblock")
    return Chem.MolFromMolBlock(mb) if mb else None


def node_svg(n, bond_in=dr.BOND_LENGTH_PT / 72, font_pt=dr.LABEL_PT, **kw):
    """`mol_svg` of a tree node: its aligned molblock + wedge plan when present, else SMILES.
    Bridged cages use the house layout: `build_network` never aligns them, and its flattened 3D layout can put a bridge carbon on a straight line (RDKit then labels it "C")."""
    m = node_mol(n)
    if m is not None and not _chem._is_bridged(m):
        return mol_svg(m, bond_in, font_pt, keep_coords=True, wedges=n.get("wedges"), **kw)
    return mol_svg(n["smiles"], bond_in, font_pt, **kw)


_MCS_CACHE = {}

def _mcs_pairs(child, parent, timeout=2):
    """[(child atom, parent atom), ...] over the maximum common substructure (elements, any bond order -- reactions change bond orders; ring atoms only match ring atoms, complete rings only). Empty when the MCS has < 3 atoms."""
    key = (Chem.MolToSmiles(child), Chem.MolToSmiles(parent))
    if key not in _MCS_CACHE:
        res = rdFMCS.FindMCS([child, parent], timeout=timeout,
                             atomCompare=rdFMCS.AtomCompare.CompareElements,
                             bondCompare=rdFMCS.BondCompare.CompareAny,
                             ringMatchesRingOnly=True, completeRingsOnly=True)
        q = Chem.MolFromSmarts(res.smartsString) if res.numAtoms >= 3 else None
        _MCS_CACHE[key] = q
    q = _MCS_CACHE[key]
    if q is None:
        return []
    a, b = child.GetSubstructMatch(q), parent.GetSubstructMatch(q)
    return list(zip(a, b)) if a and b else []


def mcs_aligned(smiles, parent):
    """`smiles` laid out oriented onto `parent` (a Mol with 2D coords) through their MCS, with `chem._aligned_child` (rigid fit, or pinned shared atoms when the fit is poor; a ring-forming step is laid out as a clean chain). None if no MCS."""
    child = Chem.MolFromSmiles(smiles)
    pairs = _mcs_pairs(child, parent) if child is not None else []
    if len(pairs) < 3:
        return None
    frag = Chem.Mol(child)                           # the MCS as pseudo atom-map numbers
    for k, (c, _) in enumerate(pairs, 1):
        frag.GetAtomWithIdx(c).SetAtomMapNum(k)
    return _chem._aligned_child(smiles, frag, {k: p for k, (_, p) in enumerate(pairs, 1)}, parent)


def realign_net(net, rxn_smiles=None):
    """Finish aligning a ``build_network(align=True)`` tree: reactions with no atom mapping are oriented by MCS instead, top-down from the target, and a defective layout falls back to the house layout. Updates molblocks and wedge plans in place; returns ``net``.
    """
    nodes = {n["id"]: n for n in net["nodes"]}
    made_by, reacts = {}, defaultdict(list)
    for e in net["edges"]:
        s_, t_ = nodes[e["source"]], nodes[e["target"]]
        if s_["type"] == "reaction" and t_["type"] == "molecule":
            made_by[t_["id"]] = s_["id"]
        elif s_["type"] == "molecule" and t_["type"] == "reaction":
            if s_["id"] not in reacts[t_["id"]]:
                reacts[t_["id"]].append(s_["id"])
    mapped = lambda i: bool(rxn_smiles) and i < len(rxn_smiles) and bool(_MAPPED.search(rxn_smiles[i] or ""))
    used = {c for v in reacts.values() for c in v}
    todo = [nid for nid, n in nodes.items() if n["type"] == "molecule" and nid not in used]
    moved = set()
    _score = lambda m: sum(int(x) for x in dr._layout_defects(m))
    while todo:
        pid = todo.pop()
        cur = node_mol(nodes[pid])
        if cur is not None and cur.GetNumBonds() and _score(cur):
            house = dr.house_layout(Chem.MolFromSmiles(nodes[pid]["smiles"]))
            if _score(house) < _score(cur):
                nodes[pid]["molblock"] = Chem.MolToMolBlock(house)
                nodes[pid]["wedges"] = _chem._wedge_plan(house)
                moved.add(pid)
        rx = made_by.get(pid)
        if rx is None:
            continue
        kids = reacts[rx]
        todo += kids
        if mapped(int(nodes[rx]["index"])) and pid not in moved:
            continue                                 # already aligned via the mapping
        parent = node_mol(nodes[pid])
        if parent is None:
            continue
        for cid in kids:
            c = nodes[cid]
            cm = Chem.MolFromSmiles(c["smiles"])
            if cm is None or _chem._is_bridged(cm):
                continue
            if cid not in made_by and cm.GetNumHeavyAtoms() < 10:
                continue                             # small SM: own clean layout
            new = mcs_aligned(c["smiles"], parent)
            if new is not None:
                c["molblock"] = Chem.MolToMolBlock(new)
                c["wedges"] = _chem._wedge_plan(new)
                moved.add(cid)
    return net


def aligned_network(syn=None, rxn_smiles=None):
    """`build_network(syn, align=True)` + the MCS fallback for unmapped reactions."""
    net = build_network(syn, align=True)
    if rxn_smiles is None:
        rxn_smiles = [(r.get("reaction_smiles_mapped") or "") for r in syn]
    return realign_net(net, rxn_smiles)


def svg_box(svg):
    """(inner body, viewBox, width_in, height_in) of a standalone SVG document."""
    svg = re.sub(r"<\?xml.*?\?>|<!DOCTYPE.*?>", "", svg, flags=re.S)
    tag = re.search(r"<svg\b[^>]*>", svg)
    head = tag.group(0)
    def _len(attr):
        v, u = re.search(r"\s" + attr + r'="([\d.]+)(in|pt|px)?"', head).groups()
        return float(v) / (1.0 if u == "in" else 72.0)
    vb = re.search(r'viewBox="([^"]+)"', head).group(1)
    return svg[tag.end():svg.rindex("</svg>")], vb, _len("width"), _len("height")


def snake_tree_svg(syn, out_path=None, max_width_in=7.05, tail_svg=None, rxn_style=None,
                   mol_drawer=None, bond_in=dr.BOND_LENGTH_PT / 72, font_pt=dr.LABEL_PT,
                   margin_in=0.08, band_gap_in=0.12, tail_gap_in=0.15, net=None, frame_style=None,
                   rxn_label=None):
    """Route tree folded into a left-right / right-left snake exactly `max_width_in` wide, each band laid out on its own. `tail_svg` goes at the end of the snake; `frame_style` and `rxn_label` override frames and box numbers. Returns the SVG, and writes `out_path` if given.
    """
    rxn_style = rxn_style or default_rxn_style
    mol_drawer = mol_drawer or (lambda n: node_svg(n, bond_in, font_pt))
    frame_style = frame_style or (lambda n: (ROLE_COLOR.get(n.get("role")), 1.2))
    rxn_label = rxn_label or (lambda i: i + 1)
    net = net if net is not None else aligned_network(syn)
    nodes = {n["id"]: n for n in net["nodes"]}
    mols, size = {}, {}
    for nid, n in nodes.items():
        if n["type"] == "reaction":
            size[nid] = RXN_BOX_IN
        elif n["type"] == "molecule":
            r = mol_drawer(n)
            if r is not None:
                mols[nid] = r; size[nid] = (r[2], r[3])
    edges = [(e["source"], e["target"]) for e in net["edges"]
             if e["source"] in size and e["target"] in size]
    safe = {nid: f"n{k}" for k, nid in enumerate(size)}; inv = {v: k for k, v in safe.items()}

    def _layout(ids):                                            # graphviz LR positions (in, y up)
        ids = set(ids)
        g = graphviz.Digraph(); g.attr(rankdir="LR", ranksep=f"{RANK_GAP_IN}", nodesep="0.06")
        for nid in (n for n in size if n in ids):                # route order, NOT set order:
            # set iteration is hash-randomised per process, which reshuffled the layout per run
            w, h = size[nid]
            g.node(safe[nid], label="", shape="box", fixedsize="true", width=f"{w:.4f}", height=f"{h:.4f}")
        for k, (s_, t_) in enumerate(edges):
            if s_ in ids and t_ in ids:
                g.edge(safe[s_], safe[t_])
            elif (s_ in ids) != (t_ in ids):
                # edge to / from ANOTHER band: an invisible stub keeps its lane (the margin connector's horizontal run) free, so graphviz puts e.g. the SM of a band's first reaction beside the connector instead of on top of it (the connector then runs along the stub's lane and curves into the node)
                stub = f"in{k}" if t_ in ids else f"out{k}"
                g.node(stub, label="", shape="box", fixedsize="true", width="0.05", height="0.12")
                g.edge(stub, safe[t_]) if t_ in ids else g.edge(safe[s_], stub)
        out_ = {}
        for line in g.pipe(format="plain").decode().splitlines():
            p = line.split()
            if p and p[0] == "node":                             # stubs keyed ("in"|"out", edge k)
                m_ = re.fullmatch(r"(in|out)(\d+)", p[1])
                key = (m_.group(1), int(m_.group(2))) if m_ else inv[p[1]]
                out_[key] = (float(p[2]), float(p[3]))
        return out_
    def _span(pos, ids):
        return (min(pos[i][0] - size[i][0] / 2 for i in ids), max(pos[i][0] + size[i][0] / 2 for i in ids))

    # 1. greedy bands: walk the rank columns of the global left->right layout and add
    #    them to the current band while ITS OWN re-layout still fits the width. Starting
    #    materials travel with the reaction they feed (so they never land in another band).
    Wc = max_width_in - 2 * margin_in
    sm_of = defaultdict(list)                                    # reaction -> its SMs
    attached = set()
    for s_, t_ in edges:
        if (nodes[s_].get("role") == "start" and nodes[t_]["type"] == "reaction"
                and s_ not in attached):
            sm_of[t_].append(s_); attached.add(s_)
    with_sm = lambda ids: ids + [m for i in ids for m in sm_of.get(i, [])]
    pos0 = _layout(size)
    cols = defaultdict(list)
    for nid in size:
        if nid not in attached:
            cols[round(pos0[nid][0], 3)].append(nid)
    fitted, cur, cur_pos = [], [], None
    for x in sorted(cols):
        trial = with_sm(cur + cols[x])
        p_ = _layout(trial)
        l, r = _span(p_, trial)
        if cur and r - l > Wc:                                   # close the band before this column
            fitted.append((with_sm(cur), cur_pos))
            cur = list(cols[x]); cur_pos = _layout(with_sm(cur))
        else:
            cur, cur_pos = cur + cols[x], p_
    fitted.append((with_sm(cur), cur_pos))
    bands = [ids for ids, _ in fitted]
    band_of = {nid: b for b, ids in enumerate(bands) for nid in ids}
    pos = {k: v for _, p_ in fitted for k, v in p_.items() if not isinstance(k, tuple)}
    spans = [_span(pos, ids) for ids in bands]
    if max(r - l for l, r in spans) > Wc:
        print(f"   note: a band is {max(r - l for l, r in spans):.2f} in wide (> {Wc:.2f} in)")
    # 2. SKYLINE stacking: each band (in band-local coords, y down from its top) moves up
    #    until its ink -- molecule / reaction boxes, the bezier edges between them and the
    #    margin connectors to the neighbouring bands -- is `band_gap_in` clear of anything
    #    already placed, so a short band tucks under the low parts of the band above it.
    W = max_width_in
    local, ink = {}, defaultdict(list)                           # band -> [(x0, x1, y0, y1)]
    for b, ids in enumerate(bands):
        l, _ = spans[b]
        yhi = max(pos[i][1] + size[i][1] / 2 for i in ids)
        for i in ids:
            xl = pos[i][0] - l
            local[i] = (margin_in + (Wc - xl if b % 2 else xl), yhi - pos[i][1])
            w, h = size[i]
            ink[b].append((local[i][0] - w / 2, local[i][0] + w / 2, local[i][1] - h / 2, local[i][1] + h / 2))
        for key, (sx, sy) in fitted[b][1].items():               # this band's cross-band stubs
            if isinstance(key, tuple):
                xl = sx - l
                local[key] = (margin_in + (Wc - xl if b % 2 else xl), yhi - sy)
    stub_band = {}
    for k, (s_, t_) in enumerate(edges):
        bs, bt = band_of[s_], band_of[t_]
        (xs_, ys_), (xt_, yt_) = local[s_], local[t_]
        if bs == bt:                                             # bezier stays in its endpoints' bbox
            ink[bs].append((min(xs_, xt_), max(xs_, xt_), min(ys_, yt_), max(ys_, yt_)))
        else:                                                    # node -> stub lane -> margin (and back)
            right = bs % 2 == 0                                  # even bands turn at the RIGHT margin
            xo, yo = local.get(("out", k), (xs_, ys_)); xi, yi = local.get(("in", k), (xt_, yt_))
            stub_band[("out", k)], stub_band[("in", k)] = bs, bt
            ink[bs].append((min(xs_, xo), max(xs_, xo), min(ys_, yo), max(ys_, yo)))
            ink[bs].append((xo, W, yo, yo) if right else (0.0, xo, yo, yo))
            ink[bt].append((xi, W, yi, yi) if right else (0.0, xi, yi, yi))
            ink[bt].append((min(xi, xt_), max(xi, xt_), min(yi, yt_), max(yi, yt_)))
    placed, XY = [], {}
    def _drop(rects, x_clear=0.03):
        """Smallest y offset putting `rects` band_gap_in below everything placed."""
        off = margin_in
        for x0, x1, y0, _ in rects:
            for p0, p1, _, q1 in placed:
                if x0 < p1 + x_clear and p0 < x1 + x_clear:
                    off = max(off, q1 + band_gap_in - y0)
        return off
    for b, ids in enumerate(bands):
        off = _drop(ink[b])
        for i in ids + [key for key, sb in stub_band.items() if sb == b and key in local]:
            XY[i] = (local[i][0], local[i][1] + off)
        placed += [(x0, x1, y0 + off, y1 + off) for x0, x1, y0, y1 in ink[b]]

    # 3. tail (plot) at the END of the snake, flush to the flow-end side, as high as it fits
    tail = None
    if tail_svg is not None:
        tb, tvb, tw, th = svg_box(tail_svg)
        b = len(bands) - 1
        tx = margin_in + Wc - tw if b % 2 == 0 else margin_in
        last_top = min(y0 for _, _, y0, _ in placed[-len(ink[b]):])   # never above the last band
        gap = (tail_gap_in, 0.0) if b % 2 == 0 else (0.0, tail_gap_in)  # clearance on the tree side only
        ty = max(last_top, _drop([(tx - gap[0], tx + tw + gap[1], 0.0, th)]))
        placed.append((tx, tx + tw, ty, ty + th))
        tail = (tb, tvb, tx, ty, tw, th)
    H = max(y1 for _, _, _, y1 in placed) + margin_in
    P = 72.0                                                     # inches -> pt
    out = [f'<svg width="{W:.4f}in" height="{H:.4f}in" viewBox="0 0 {W * P:.2f} {H * P:.2f}" '
           'xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1">']

    # 4. edges (under the boxes): bezier inside a band, margin connector between bands
    AH = 0.055                                                   # arrowhead length, in
    def _arrow(x, y, d):
        L, Hh = AH * P, 0.022 * P
        return (f'<path d="M{x:.2f},{y:.2f} L{x - d * L:.2f},{y - Hh:.2f} L{x - d * L:.2f},{y + Hh:.2f} Z" '
                'fill="#888888" stroke="none"/>')
    # flow-side edge of every rank column (graphviz centres a rank's nodes on one x): an edge runs straight out to its column's edge and only bends in the gap beyond it, so a small SM's curve can't cut across a big molecule stacked in the same column
    col_end = {}
    for nid, (x, _) in XY.items():
        if isinstance(nid, tuple):
            continue
        b = band_of[nid]; d_ = -1 if b % 2 else 1; key = (b, round(x, 2))
        e_ = x + d_ * size[nid][0] / 2
        col_end[key] = e_ if key not in col_end else (max if d_ > 0 else min)(col_end[key], e_)
    for k, (s_, t_) in enumerate(edges):
        bs, bt = band_of[s_], band_of[t_]
        ds, dt = (-1 if bs % 2 else 1), (-1 if bt % 2 else 1)
        (xs_, ys_), (xt_, yt_) = XY[s_], XY[t_]
        x1, y1 = (xs_ + ds * size[s_][0] / 2) * P, ys_ * P
        xh, y2 = (xt_ - dt * size[t_][0] / 2) * P, yt_ * P          # arrow tip
        head = nodes[t_]["type"] == "molecule"
        x2 = xh - dt * AH * P if head else xh
        xa = col_end[(bs, round(xs_, 2))] * P                    # straight run to the column edge
        if bs == bt:
            if (xa - x1) * ds < 0 or (x2 - xa) * ds < 0:         # (defensive) stay between ends
                xa = x1
            c = abs(x2 - xa) / 2
            d = (f"M{x1:.2f},{y1:.2f} L{xa:.2f},{y1:.2f} "
                 f"C{xa + ds * c:.2f},{y1:.2f} {x2 - dt * c:.2f},{y2:.2f} {x2:.2f},{y2:.2f}")
        else:                                                    # curve onto the stub lane, down the margin
            xe = (W - margin_in / 2) * P if ds > 0 else (margin_in / 2) * P
            xo, yo = (v * P for v in XY.get(("out", k), (x1 / P, y1 / P)))
            xi, yi = (v * P for v in XY.get(("in", k), (x2 / P, y2 / P)))
            # stay in the lane until the rank gap next to the node, THEN curve, so the bend clears the neighbouring SM box (whose own edge bends in the same gap)
            if (xa - x1) * ds < 0 or (xe - xa) * ds < 0:
                xa = x1
            to = xa + ds * min(RANK_GAP_IN * P, abs(xo - xa), abs(xe - xa))   # end of the outgoing bend
            ti = x2 - dt * min(RANK_GAP_IN * P, abs(x2 - xi), abs(x2 - xe))   # start of the incoming bend
            co, ci = abs(to - xa) / 2, abs(x2 - ti) / 2
            d = (f"M{x1:.2f},{y1:.2f} L{xa:.2f},{y1:.2f} "
                 f"C{xa + ds * co:.2f},{y1:.2f} {to - ds * co:.2f},{yo:.2f} {to:.2f},{yo:.2f} "
                 f"L{xe:.2f},{yo:.2f} L{xe:.2f},{yi:.2f} L{ti:.2f},{yi:.2f} "
                 f"C{ti + dt * ci:.2f},{yi:.2f} {x2 - dt * ci:.2f},{y2:.2f} {x2:.2f},{y2:.2f}")
        out.append(f'<path d="{d}" fill="none" stroke="#888888" stroke-width="0.9"/>')
        if head:
            out.append(_arrow(xh, y2, dt))

    # 5. molecules (role-coloured frames) and numbered reaction boxes
    for nid, (x, y) in XY.items():
        if isinstance(nid, tuple):                               # invisible lane stub
            continue
        w, h = size[nid]; x0, y0 = (x - w / 2) * P, (y - h / 2) * P
        n = nodes[nid]
        if n["type"] == "reaction":
            idx = int(n["index"])
            fill, edge, sw, tcol = rxn_style(idx)
            out.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{w * P:.2f}" height="{h * P:.2f}" '
                       f'fill="{fill}" stroke="{edge}" stroke-width="{sw}"/>')
            out.append(f'<text x="{x * P:.2f}" y="{y * P + 2.1:.2f}" font-family="Arial, Helvetica, sans-serif" '
                       f'font-size="6" font-weight="bold" text-anchor="middle" '
                       f'fill="{tcol}">{rxn_label(idx)}</text>')
        else:
            body, vb, _, _ = mols[nid]
            out.append(f'<svg x="{x0:.2f}" y="{y0:.2f}" width="{w * P:.2f}" height="{h * P:.2f}" '
                       f'viewBox="{vb}" preserveAspectRatio="xMidYMid meet">{body}</svg>')
            col, sw = frame_style(n)
            if col:
                out.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{w * P:.2f}" height="{h * P:.2f}" '
                           f'fill="none" stroke="{col}" stroke-width="{sw}"/>')
    if tail is not None:
        tb, tvb, tx, ty, tw, th = tail
        out.append(f'<svg x="{tx * P:.2f}" y="{ty * P:.2f}" width="{tw * P:.2f}" height="{th * P:.2f}" '
                   f'viewBox="{tvb}" preserveAspectRatio="xMidYMin meet">{tb}</svg>')
    out.append("</svg>")
    svg = "\n".join(out)
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fh:
            fh.write(svg)
    return svg


def stack_svg(parts, out_path=None, width_in=7.05, gap_in=0.12, caption_pt=7, margin_in=0.04):
    """Stack SVGs top -> bottom, each under an optional caption: `parts` = [(svg, caption
    | None, caption colour | None), ...]; a caption's FIRST line is bold, later lines ("\n") regular. Returns the combined SVG; writes `out_path`."""
    P, out, y = 72.0, [], margin_in
    for svg, cap, *col in parts:
        if cap:
            c = (col[0] if col else None) or "black"
            for j, line in enumerate(cap.split("\n")):
                y += caption_pt * 1.25 / P
                out.append(f'<text x="{margin_in * P:.2f}" y="{y * P:.2f}" font-family="Arial, Helvetica, sans-serif" '
                           f'font-size="{caption_pt}" font-weight="{"bold" if j == 0 else "normal"}" '
                           f'fill="{c}">{html.escape(line)}</text>')
            y += 0.05
        body, vb, w, h = svg_box(svg)
        out.append(f'<svg x="0" y="{y * P:.2f}" width="{w * P:.2f}" height="{h * P:.2f}" viewBox="{vb}">{body}</svg>')
        y += h + gap_in
    H = y - gap_in + margin_in
    svg = (f'<svg width="{width_in:.4f}in" height="{H:.4f}in" viewBox="0 0 {width_in * P:.2f} {H * P:.2f}" '
           'xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1">'
           + "".join(out) + "</svg>")
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fh:
            fh.write(svg)
    return svg


def _balanced_rows(widths, avail, gap, max_per_row=None):
    """Split `widths` (in order) into the fewest rows that fit `avail` (incl. `gap` between items), with row widths as even as possible (min-max linear partition)."""
    n = len(widths)
    def fits(lo, hi):                                            # items lo..hi-1 on one row
        return (hi - lo <= (max_per_row or n)
                and sum(widths[lo:hi]) + (hi - lo - 1) * gap <= avail + 1e-9)
    def split(limit):                                            # greedy rows <= limit wide
        rows, lo = [], 0
        while lo < n:
            hi = lo + 1
            while hi < n and fits(lo, hi + 1) and sum(widths[lo:hi + 1]) + (hi - lo) * gap <= limit + 1e-9:
                hi += 1
            rows.append((lo, hi)); lo = hi
        return rows
    best = split(avail)
    nrow, lo_l, hi_l = len(best), max(widths), avail             # shrink the widest row while
    for _ in range(40):                                          # the row count stays the same
        mid = (lo_l + hi_l) / 2
        r = split(mid)
        if len(r) <= nrow:
            best, hi_l = r, mid
        else:
            lo_l = mid
    return best


def _packed_grid_parts(drawn, y, width_in, margin_in, gap_in, row_gap_in, max_per_row=None,
                       nest=True):
    """Lay `drawn` molecules out as packed, justified rows from height `y`, each at its own ink width. With `nest`, a row rises until it is `row_gap_in` below an overlapping molecule above. Returns (svg parts, bottom y, rows, shrink).
    """
    P, avail = 72.0, width_in - 2 * margin_in
    k = min(1.0, avail / max(r[2] for r in drawn))               # shrink only if one won't fit
    ws = [r[2] * k for r in drawn]
    rows = _balanced_rows(ws, avail, gap_in, max_per_row)
    gaps = []
    for lo, hi in rows:
        m = hi - lo
        gaps.append((avail - sum(ws[lo:hi])) / (m - 1) if m > 1 else 0.0)
    full = gaps[:-1]
    last_gap = min(gaps[-1], (sum(full) / len(full)) if full else gap_in)
    parts, placed, row_top = [], [], y                           # placed = (x0, x1, bottom)
    for i, (lo, hi) in enumerate(rows):
        g = gaps[i] if i < len(rows) - 1 else last_gap
        rh = max(drawn[j][3] for j in range(lo, hi)) * k
        cells, x = [], margin_in                                 # (j, x, dy within row, w, h)
        for j in range(lo, hi):
            w, h = drawn[j][2] * k, drawn[j][3] * k
            cells.append((j, x, (rh - h) / 2, w, h)); x += w + g
        if nest and placed:
            pad = gap_in / 2                                     # keep a gap diagonally too
            row_top = max([b + row_gap_in - dy for _, cx, dy, w, _ in cells
                           for x0, x1, b in placed if x0 < cx + w + pad and x1 > cx - pad]
                          + [max(b for *_, b in placed) - rh + row_gap_in, y])
        for j, cx, dy, w, h in cells:
            body, vb = drawn[j][0], drawn[j][1]
            parts.append(f'<svg x="{cx * P:.2f}" y="{(row_top + dy) * P:.2f}" width="{w * P:.2f}" '
                         f'height="{h * P:.2f}" viewBox="{vb}" preserveAspectRatio="xMidYMid meet">{body}</svg>')
            placed.append((cx, cx + w, row_top + dy + h))
        row_top += rh + row_gap_in
    return parts, max(b for *_, b in placed), len(rows), k


def _write_svg_formats(svg, out_stem, formats, dpi):
    import cairosvg
    os.makedirs(os.path.dirname(out_stem) or ".", exist_ok=True)
    out = []
    for fmt in formats:
        p = f"{out_stem}.{fmt}"
        if fmt == "svg":
            open(p, "w").write(svg)
        elif fmt == "pdf":
            cairosvg.svg2pdf(bytestring=svg.encode(), write_to=p)
        elif fmt == "png":
            cairosvg.svg2png(bytestring=svg.encode(), write_to=p, dpi=dpi, background_color="white")
        out.append(p)
    return out


def molecule_grid(smiles, out_stem, per_row=None, width_in=7.5, formats=("pdf",), dpi=600,
                  margin_in=0.05, col_gap_in=0.22, row_gap_in=0.12, abbreviate=False):
    """Molecules packed into justified rows in the house style, each cropped to its ink, in the fewest rows that fit `width_in`. Writes `out_stem`.<fmt> for each of `formats`.
    """
    drawn = [r for r in (mol_svg(s, abbreviate=abbreviate) for s in smiles) if r is not None]
    P = 72.0
    parts, y, nrow, k = _packed_grid_parts(drawn, margin_in, width_in, margin_in, col_gap_in,
                                           row_gap_in, per_row)
    H = y + margin_in
    svg = (f'<svg width="{width_in:.4f}in" height="{H:.4f}in" viewBox="0 0 {width_in * P:.2f} {H * P:.2f}" '
           'xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1">'
           + "".join(parts) + "</svg>")
    out = _write_svg_formats(svg, out_stem, formats, dpi)
    print(f"   grid: {len(drawn)} structures in {nrow} rows, {width_in:.2f} x {H:.2f} in"
          + (f", shrunk to {k:.0%} to fit" if k < 1 else ""))
    return out


def molecule_sections(sections, out_stem, width_in=7.5, formats=("pdf",), dpi=600,
                      title_pt=8, margin_in=0.05, col_gap_in=0.22, row_gap_in=0.12,
                      section_gap_in=0.14, abbreviate=False, show=False):
    """Several titled molecule grids stacked in one figure: `sections` = {title: [smiles]} (or [(title, [smiles])]). Each section is a :func:`molecule_grid`-style packed block under a bold title. Writes `out_stem`.<fmt> for each of `formats`; returns the paths.
    With `show`, also displays the figure inline (Jupyter)."""
    P = 72.0
    items = list(sections.items()) if isinstance(sections, dict) else list(sections)
    parts, y, info = [], margin_in, []
    for title, smis in items:
        drawn = [r for r in (mol_svg(s, abbreviate=abbreviate) for s in smis) if r is not None]
        if not drawn:
            continue
        if parts:
            y += section_gap_in
        y += title_pt / P
        parts.append(f'<text x="{margin_in * P:.2f}" y="{y * P:.2f}" font-family="Arial, Helvetica, sans-serif" '
                     f'font-size="{title_pt}" font-weight="bold" fill="black">{html.escape(title)}</text>')
        y += 0.06
        p_, y, nrow, _ = _packed_grid_parts(drawn, y, width_in, margin_in, col_gap_in, row_gap_in)
        parts += p_
        info.append(f"{title}: {len(drawn)} in {nrow} rows")
    H = y + margin_in
    svg = (f'<svg width="{width_in:.4f}in" height="{H:.4f}in" viewBox="0 0 {width_in * P:.2f} {H * P:.2f}" '
           'xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1">'
           + "".join(parts) + "</svg>")
    out = _write_svg_formats(svg, out_stem, formats, dpi)
    print(f"   sections: {'; '.join(info)}; {width_in:.2f} x {H:.2f} in")
    if show:
        from IPython.display import SVG, display
        display(SVG(data=svg))
    return out


def reaction_svg(left, right, above=None, below=None, out_path=None, margin_in=0.06,
                 plus_in=0.16, arrow_in=0.75, font_pt=dr.LABEL_PT, label_pt=6.5, align=True,
                 above_colour="black", below_colour="black"):
    """One reaction scheme in the house style, molecules centred on the arrow. `left` / `right` items are SMILES or dicts {smiles, highlight, label, label_colour, frame}; `above` / `below` are condition text. With `align`, molecules are oriented onto the first product.
    """
    P = 72.0
    items = [{"smiles": it} if isinstance(it, str) else it for it in list(right) + list(left)]
    ref = dr.house_layout(items[0]["smiles"]) if align and items else None
    def _layout(it, k):
        if ref is None:
            return None
        if k == 0:
            return ref
        return mcs_aligned(it["smiles"], ref)        # None (own house layout) if no MCS
    drawn = []
    for k, it in enumerate(items):
        m = _layout(it, k)
        r = mol_svg(m if m is not None else it["smiles"], abbreviate=False, highlight=it.get("highlight"),
                    highlight_bonds=True, keep_coords=m is not None)
        drawn.append((r, it) if r is not None else None)
    R = [d_ for d_ in drawn[:len(right)] if d_]; L = [d_ for d_ in drawn[len(right):] if d_]
    tw = lambda s, pt: 0.52 * pt * len(s) / P if s else 0.0      # Arial width estimate, in
    up = above.split("\n") if above else []; dn = below.split("\n") if below else []
    lh = (font_pt + 1.5) / P                                       # text line height, in
    arrow = max([arrow_in] + [tw(t, font_pt) + 0.1 for t in up + dn])
    mh = max(r[3] for r, _ in L + R)
    has_lab = any(it.get("label") for _, it in L + R)
    cy = margin_in + max(mh / 2, 0.05 + len(up) * lh)              # room for text over arrow
    top = cy - mh / 2
    parts, x = [], margin_in
    def _text(x_, y_, s, pt, anchor="middle", fill="black"):
        parts.append(f'<text x="{x_ * P:.2f}" y="{y_ * P:.2f}" font-family="Arial, Helvetica, sans-serif" '
                     f'font-size="{pt}" text-anchor="{anchor}" fill="{fill}">{html.escape(s)}</text>')
    def _mols(items):
        nonlocal x
        for k, ((body, vb, w, h), it) in enumerate(items):
            if k:
                _text(x + plus_in / 2, cy + 0.035, "+", font_pt + 2); x += plus_in
            slot = max(w, tw(it.get("label"), label_pt) + 0.04)   # room for a long label
            x += (slot - w) / 2
            parts.append(f'<svg x="{x * P:.2f}" y="{(cy - h / 2) * P:.2f}" width="{w * P:.2f}" '
                         f'height="{h * P:.2f}" viewBox="{vb}" preserveAspectRatio="xMidYMid meet">{body}</svg>')
            if it.get("frame"):                                   # dashed call-out box
                pad = 0.02
                parts.append(f'<rect x="{(x - pad) * P:.2f}" y="{(cy - h / 2 - pad) * P:.2f}" '
                             f'width="{(w + 2 * pad) * P:.2f}" height="{(h + 2 * pad) * P:.2f}" fill="none" '
                             f'stroke="{it["frame"]}" stroke-width="0.8" stroke-dasharray="2.5,1.5"/>')
            if it.get("label"):                                   # right under its molecule
                _text(x + w / 2, cy + h / 2 + label_pt / P + 0.04, it["label"], label_pt,
                      fill=it.get("label_colour", "#555555"))
            x += w + (slot - w) / 2
    _mols(L)
    x += 0.08; x0, x1 = x, x + arrow                              # reaction arrow
    parts.append(f'<path d="M{x0 * P:.2f},{cy * P:.2f} L{(x1 - 0.06) * P:.2f},{cy * P:.2f}" '
                 'stroke="black" stroke-width="0.6" fill="none"/>')
    parts.append(f'<path d="M{x1 * P:.2f},{cy * P:.2f} L{(x1 - 0.08) * P:.2f},{(cy - 0.028) * P:.2f} '
                 f'L{(x1 - 0.08) * P:.2f},{(cy + 0.028) * P:.2f} Z" fill="black"/>')
    for k, t in enumerate(reversed(up)):                         # stacked upwards from the arrow
        _text((x0 + x1) / 2, cy - 0.05 - k * lh, t, font_pt, fill=above_colour)
    for k, t in enumerate(dn):
        _text((x0 + x1) / 2, cy + 0.05 + font_pt / P + k * lh, t, font_pt, fill=below_colour)
    x = x1 + 0.08
    _mols(R)
    W = x + margin_in
    H = max(top + mh + (label_pt / P + 0.08 if has_lab else 0.02),
            cy + 0.05 + len(dn) * lh + 0.03) + margin_in
    svg = (f'<svg width="{W:.4f}in" height="{H:.4f}in" viewBox="0 0 {W * P:.2f} {H * P:.2f}" '
           'xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1">'
           + "".join(parts) + "</svg>")
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fh:
            fh.write(svg)
    return svg


def hcat_svg(parts, gap_in=0.25, align="middle"):
    """Place SVGs side by side (vertically centred); returns the combined SVG."""
    P = 72.0
    boxes = [svg_box(s) for s in parts]
    H = max(b[3] for b in boxes); out, x = [], 0.0
    for body, vb, w, h in boxes:
        y = (H - h) / 2 if align == "middle" else 0.0
        out.append(f'<svg x="{x * P:.2f}" y="{y * P:.2f}" width="{w * P:.2f}" height="{h * P:.2f}" viewBox="{vb}">{body}</svg>')
        x += w + gap_in
    W = x - gap_in
    return (f'<svg width="{W:.4f}in" height="{H:.4f}in" viewBox="0 0 {W * P:.2f} {H * P:.2f}" '
            'xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1">'
            + "".join(out) + "</svg>")


def gutter_label_svg(svg, label, colour="black", gutter_in=0.55, font_pt=7):
    """`svg` with a bold `label` in a left gutter, vertically centred."""
    P = 72.0
    body, vb, w, h = svg_box(svg)
    W = gutter_in + w
    return (f'<svg width="{W:.4f}in" height="{h:.4f}in" viewBox="0 0 {W * P:.2f} {h * P:.2f}" '
            'xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1">'
            f'<text x="2" y="{(h / 2) * P + font_pt * 0.35:.2f}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="{font_pt}" font-weight="bold" fill="{colour}">{html.escape(label)}</text>'
            f'<svg x="{gutter_in * P:.2f}" y="0" width="{w * P:.2f}" height="{h * P:.2f}" viewBox="{vb}">{body}</svg></svg>')
