"""House-style structure and reaction drawing: RDKit for figure PNG/SVG, Indigo for editable ChemDraw ``.cdxml``.
"""
from __future__ import annotations
from typing import Iterable, Optional, Sequence
import xml.etree.ElementTree as ET

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit import Chem
from rdkit.Chem import rdChemReactions, rdCoordGen, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

# Prefer the CoordGen (Schrödinger) 2D layout everywhere: it draws fused/bridged polycyclic ring systems cleanly instead of collapsing them into a big polygon.
rdDepictor.SetPreferCoordGen(True)

from . import chem


# House style — keep every reaction/molecule drawing visually consistent. -----
BOND_LINE_WIDTH = 1.5      # RDKit line width (px at PANEL_PX)
MIN_FONT_SIZE = 12         # atom-label font (px); min == max => fixed size
MAX_FONT_SIZE = 12
PANEL_PX = (1000, 320)     # per-reaction canvas for the RDKit PNG

# ACS Document 1996 — ChemDraw's default publication stylesheet, and the ONE house style for every structure in every figure (ChemDraw > Drawing settings):
#   chain angle 120°, fixed length 0.2 in, bold width 0.0278 in, line width
#   0.0083 in, margin width 0.0222 in, hash spacing 0.0347 in, bond spacing 18 %
#   of length.
# As usual for ACS 1996, the drawing is then SCALED TO 75 % when placed in the figure, so every length below is the drawn value x DRAWING_SCALE (0.15-in bonds, 0.45-pt lines, ...); the % bond spacing and the chain angle are scale-free. Atom labels are 7 pt ON THE PAGE (figure body text).
# The CDXML exports carry all of these exactly (at the scaled size). The RDKit renderers (SVG trees, vector structures in matplotlib) take bond length, line width, label size and double-bond spacing from here; RDKit has no wedge-width / hash-spacing setting.
DRAWING_SCALE = 0.75
_DRAWN_BOND_LENGTH_PT = 14.4   # 0.2 in       (ChemDraw "Fixed Length")
_DRAWN_LINE_WIDTH_PT = 0.6     # 0.0083 in    ("Line Width")
_DRAWN_BOLD_WIDTH_PT = 2.0     # 0.0278 in    ("Bold Width")
_DRAWN_MARGIN_WIDTH_PT = 1.6   # 0.0222 in    ("Margin Width")
_DRAWN_HASH_SPACING_PT = 2.5   # 0.0347 in    ("Hash Spacing")
BOND_LENGTH_PT = _DRAWN_BOND_LENGTH_PT * DRAWING_SCALE      # 10.8 pt = 0.15 in
LINE_WIDTH_PT = _DRAWN_LINE_WIDTH_PT * DRAWING_SCALE        # 0.45 pt
BOLD_WIDTH_PT = _DRAWN_BOLD_WIDTH_PT * DRAWING_SCALE        # 1.5 pt (wedge width; CDXML only)
MARGIN_WIDTH_PT = _DRAWN_MARGIN_WIDTH_PT * DRAWING_SCALE    # 1.2 pt (label-to-bond gap)
HASH_SPACING_PT = _DRAWN_HASH_SPACING_PT * DRAWING_SCALE    # 1.875 pt (CDXML only)
BOND_SPACING = 0.18          # double-bond spacing, fraction of bond length
CHAIN_ANGLE = 120
LABEL_PT = 7                 # atom-label size on the page
RDKIT_SUPERSAMPLE = 10       # RDKit draws at 10x, then is scaled back (see apply_house_style)
HASH_DASHES = 4              # dashes per hashed wedge (RDKit's count grows with the 10x canvas)
HASH_WIDTH_SCALE = 1.5       # RDKit's hashed wedge is narrow (~0.18 bond at the wide end); ChemDraw ~0.27


def apply_house_style(opts, bond_len_pt: float = BOND_LENGTH_PT,
                      font_pt: float = LABEL_PT, line_pt: float = LINE_WIDTH_PT):
    """Set RDKit ``MolDrawOptions`` to the house (ACS 1996) style, in points. The canvas is ``RDKIT_SUPERSAMPLE`` x oversized because RDKit rounds stroke widths, so callers rescale by the measured median bond length.
    """
    k = RDKIT_SUPERSAMPLE
    opts.fixedBondLength = bond_len_pt * k
    opts.scalingFactor = bond_len_pt * k     # flexicanvas caps bonds at scalingFactor (20)
    opts.fixedFontSize = int(round(font_pt * k))   # RDKit wants an int: 0.1-pt steps
    opts.minFontSize = int(round(font_pt * k))
    opts.maxFontSize = int(round(font_pt * k))
    opts.bondLineWidth = line_pt * k
    opts.scaleBondWidth = False
    opts.multipleBondOffset = BOND_SPACING
    # ChemDraw "Margin Width": clear bonds 1.2 pt from atom labels (RDKit's default 0 lets bonds touch the letters). RDKit's padding is a fraction of the bond length, so this stays 1.2 pt at any bond length.
    opts.additionalAtomLabelPadding = MARGIN_WIDTH_PT / bond_len_pt
    opts.padding = 0.0
    opts.useBWAtomPalette()
    return opts


# Indigo lays out at bond length 30, so we scale coordinates by BOND_LENGTH_PT/30 on export and stamp these attributes onto the CDXML root.
ACS1996_BOND_LENGTH = BOND_LENGTH_PT
_ACS1996_ROOT = {
    "Name": "ACS Document 1996",
    "LineWidth": f"{LINE_WIDTH_PT:g}",
    "BoldWidth": f"{BOLD_WIDTH_PT:g}",
    "MarginWidth": f"{MARGIN_WIDTH_PT:g}",
    "HashSpacing": f"{HASH_SPACING_PT:g}",
    "BondSpacing": f"{BOND_SPACING * 100:g}",
    "ChainAngle": f"{CHAIN_ANGLE:g}",
    "LabelFont": "2",          # id 2 == Arial in Indigo's font table
    "LabelSize": f"{LABEL_PT:g}",
    "LabelFace": "96",
    "CaptionFont": "2",
    "CaptionSize": f"{LABEL_PT:g}",
}
# CDXML attributes holding geometry we must scale with the bond length.
_GEOM_ATTRS = ("p", "Head3D", "Tail3D", "BoundingBox")      # space-separated numbers
_ARROW_SIZE_ATTRS = ("HeadSize", "ArrowheadWidth", "ArrowheadCenterSize")


# SMILES prep -----
def clean_rxn_smiles(rxn_smiles: str) -> str:
    """``a>agents>b`` (atom-mapped) -> ``reactants>>products`` with atom maps and reagents removed, each component canonicalised. Returns the input unchanged if it cannot be parsed."""
    reactants, _agents, products = chem.rxn_parts(rxn_smiles)

    def _clean_side(side: str) -> str:
        out = []
        for frag in side.split("."):
            if not frag:
                continue
            out.append(chem.canonicalize(frag, remove_atom_maps=True) or frag)
        return ".".join(out)

    r, p = _clean_side(reactants), _clean_side(products)
    return f"{r}>>{p}" if (r or p) else rxn_smiles


# RDKit — inline PNG (all atoms black) -----
def reaction_png(rxn_smiles: str, size=PANEL_PX, clean: bool = True) -> bytes:
    """Render one reaction to PNG bytes in the house style (black atoms)."""
    smi = clean_rxn_smiles(rxn_smiles) if clean else rxn_smiles
    rxn = rdChemReactions.ReactionFromSmarts(smi, useSmiles=True)
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    opt = d.drawOptions()
    opt.useBWAtomPalette()              # every atom black
    opt.bondLineWidth = BOND_LINE_WIDTH
    opt.minFontSize = MIN_FONT_SIZE
    opt.maxFontSize = MAX_FONT_SIZE
    d.DrawReaction(rxn)
    d.FinishDrawing()
    return d.GetDrawingText()


def show_reaction(rxn_smiles: str, size=PANEL_PX, clean: bool = True):
    """Display one reaction inline in a notebook (returns the IPython image)."""
    from IPython.display import Image
    return Image(data=reaction_png(rxn_smiles, size=size, clean=clean))


def molecules_to_png(smiles, out_path: str, per_row: int = 6,
                     width_in: float = 7.5, dpi: int = 600,
                     cell_aspect: float = 0.85, legends=None) -> str:
    """Write many molecules as one PNG grid in the house style, sized to ``width_in`` at ``dpi``.
    Returns ``out_path``.
    """
    import io, os
    from rdkit.Chem import Draw
    from PIL import Image
    mols = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        mols.append(house_layout(m))

    cell_w = max(1, round(width_in * dpi / per_row))     # px per structure
    cell_h = round(cell_w * cell_aspect)
    scale = cell_w / 250.0                                # vs the ~250 px baseline cell
    opts = rdMolDraw2D.MolDrawOptions()
    opts.useBWAtomPalette()                      # every atom black
    opts.bondLineWidth = max(1, int(round(BOND_LINE_WIDTH * scale)))
    fs = max(6, int(round(MIN_FONT_SIZE * scale)))
    opts.minFontSize = fs
    opts.maxFontSize = fs
    png = Draw.MolsToGridImage(mols, molsPerRow=per_row, subImgSize=(cell_w, cell_h),
                               legends=legends if legends is not None else [""] * len(mols),
                               useSVG=False, drawOptions=opts, returnPNG=True)
    data = png.data if hasattr(png, "data") else bytes(png)   # IPython Image vs raw bytes
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    Image.open(io.BytesIO(data)).save(out_path, dpi=(dpi, dpi))   # stamp physical size
    return out_path


def _grid_png_bytes(smiles, per_row, cell_w, cell_h, legends=None, font_px=None):
    """One MolsToGridImage grid as PNG bytes, in the house style (all atoms black, CoordGen layout). Bond width scales with ``cell_w`` like :func:`molecules_to_png`; the atom-label font is ``font_px`` pixels if given, else it scales with ``cell_w`` too."""
    from rdkit.Chem import Draw
    mols = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        mols.append(house_layout(m))
    scale = cell_w / 250.0
    opts = rdMolDraw2D.MolDrawOptions()
    opts.useBWAtomPalette()
    opts.bondLineWidth = max(1, int(round(BOND_LINE_WIDTH * scale)))
    fs = int(font_px) if font_px else max(6, int(round(MIN_FONT_SIZE * scale)))
    opts.minFontSize = fs
    opts.maxFontSize = fs
    png = Draw.MolsToGridImage(mols, molsPerRow=per_row, subImgSize=(cell_w, cell_h),
                               legends=legends if legends is not None else [""] * len(mols),
                               useSVG=False, drawOptions=opts, returnPNG=True)
    return png.data if hasattr(png, "data") else bytes(png)


def molecule_sections_to_png(sections, out_path: str, per_row: int = 6,
                             width_in: float = 7.5, dpi: int = 600,
                             cell_aspect: float = 0.85, legends=None,
                             label_pt: float = 10) -> str:
    """Write labelled sections of molecules into one PNG, stacked under bold titles. ``sections`` is ``{title: [smiles, ...]}`` or ``(title, [smiles, ...])`` pairs, ``legends`` optional per-molecule captions. Returns ``out_path``.
    """
    import io, os
    from PIL import Image, ImageDraw, ImageFont

    items = list(sections.items()) if isinstance(sections, dict) else list(sections)
    cell_w = max(1, round(width_in * dpi / per_row))
    cell_h = round(cell_w * cell_aspect)
    full_w = cell_w * per_row
    font_px = max(1, round(label_pt / 72.0 * dpi)) if label_pt else None   # true point size

    title_h = max(1, round(0.16 * dpi))                # band above each grid
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(8, round(0.08 * dpi)))
    except Exception:
        font = ImageFont.load_default()

    panels = []                                        # (title, grid PIL image)
    for title, smis in items:
        if not len(smis):
            continue
        legs = (legends or {}).get(title) if isinstance(legends, dict) else None
        grid = Image.open(io.BytesIO(
            _grid_png_bytes(smis, per_row, cell_w, cell_h, legends=legs,
                            font_px=font_px))).convert("RGB")
        panels.append((title, grid))

    total_h = sum(title_h + g.height for _, g in panels)
    canvas = Image.new("RGB", (full_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for title, grid in panels:
        draw.text((round(0.05 * dpi), y + round(title_h * 0.30)), title, fill="black", font=font)
        y_grid = y + title_h
        canvas.paste(grid, (round((full_w - grid.width) / 2), y_grid))
        y = y_grid + grid.height

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path, dpi=(dpi, dpi))
    return out_path


# Indigo/RDKit — editable CDXML, several reactions stacked into one file -----
# Coordinates come from RDKit CoordGen (good ring layout); Indigo is used only to serialise each molecule to CDXML. Reactions are then assembled by hand (fragments + "+" symbols + arrow + scheme) so we control spacing and never emit the "abs"/"Chiral" enhanced-stereo annotations ChemDraw would show.
# Layout constants, in Indigo's ~30-unit (bond-length) space.
_MOL_GAP = 45.0        # horizontal gap around a "+" between co-reactants/products
_ARROW_PAD = 30.0      # gap between structures and the reaction arrow
_ARROW_LEN = 60.0      # reaction-arrow length
_ROW_GAP = 70.0        # vertical gap between stacked reactions


def _shift_elem(el: ET.Element, dx: float, dy: float):
    """Translate every coordinate in an element subtree by (dx, dy)."""
    for e in el.iter():
        for a in ("p", "Head3D", "Tail3D"):
            if a in e.attrib:
                v = e.attrib[a].split()
                v[0] = f"{float(v[0]) + dx:.6f}"
                v[1] = f"{float(v[1]) + dy:.6f}"
                e.set(a, " ".join(v))
        if "BoundingBox" in e.attrib:
            b = e.attrib["BoundingBox"].split()
            b[0] = f"{float(b[0]) + dx:.6f}"; b[1] = f"{float(b[1]) + dy:.6f}"
            b[2] = f"{float(b[2]) + dx:.6f}"; b[3] = f"{float(b[3]) + dy:.6f}"
            e.set("BoundingBox", " ".join(b))


def _elem_y_range(elems):
    ys = []
    for el in elems:
        for e in el.iter():
            if "p" in e.attrib:
                ys.append(float(e.attrib["p"].split()[1]))
            if "BoundingBox" in e.attrib:
                bb = e.attrib["BoundingBox"].split()
                ys += [float(bb[1]), float(bb[3])]
    return (min(ys), max(ys)) if ys else (0.0, 0.0)


def _node_bbox(frag: ET.Element):
    xs, ys = [], []
    for n in frag.iter("n"):
        if "p" in n.attrib:
            px, py = map(float, n.attrib["p"].split()[:2])
            xs.append(px); ys.append(py)
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


def _reindex(frag: ET.Element, offset: int) -> int:
    """Add ``offset`` to every id in a fragment (ids, bond ends, and the BondOrdering lists that reference them) so it is unique in the document.
    Returns the new maximum id."""
    mx = offset
    for e in frag.iter():
        for a in ("id", "B", "E"):
            if a in e.attrib:
                nv = int(e.attrib[a]) + offset
                e.set(a, str(nv)); mx = max(mx, nv)
        if "BondOrdering" in e.attrib:                 # references bond ids
            e.set("BondOrdering",
                  " ".join(str(int(t) + offset) for t in e.attrib["BondOrdering"].split()))
    return mx


def _plus_graphic(x: float, y: float, idn: int) -> ET.Element:
    return ET.Element("graphic", {
        "id": str(idn),
        "BoundingBox": f"{x - 3.75:.6f} {y + 3.75:.6f} {x + 3.75:.6f} {y - 3.75:.6f}",
        "GraphicType": "Symbol", "SymbolType": "Plus"})


def _arrow(x0: float, x1: float, y: float, idn: int) -> ET.Element:
    return ET.Element("arrow", {
        "id": str(idn),
        "BoundingBox": f"{x1:.6f} {y:.6f} {x0:.6f} {y:.6f}",
        "FillType": "None", "ArrowheadType": "Solid", "HeadSize": "2250",
        "ArrowheadWidth": "563", "ArrowheadHead": "Full", "ArrowheadCenterSize": "25",
        "Head3D": f"{x1:.6f} {y:.6f} 0.000000",
        "Tail3D": f"{x0:.6f} {y:.6f} 0.000000"})


def _mol_fragment(ind, smiles: str):
    """``house_layout`` coords -> molblock -> Indigo -> a CDXML ``<fragment>`` element (all-black, Kekulized, wedge stereo kept, enhanced-stereo labels stripped)."""
    if Chem.MolFromSmiles(smiles) is None:
        return None
    m = house_layout(smiles)                            # house ring layout
    try:
        Chem.Kekulize(m, clearAromaticFlags=True)      # explicit double bonds
    except Exception:
        pass
    try:
        Chem.WedgeMolBonds(m, m.GetConformer())        # stereo -> wedge/hash bonds
    except Exception:
        pass
    frag = ET.fromstring(ind.loadMolecule(Chem.MolToMolBlock(m, kekulize=True)).cdxml()
                         ).find(".//fragment")
    if frag is not None:
        for n in frag.iter("n"):                       # drop the "abs" annotations
            n.attrib.pop("EnhancedStereoType", None)
            n.attrib.pop("EnhancedStereoGroupNum", None)
    return frag


def _apply_acs1996(root: ET.Element, label_size: float = LABEL_PT,
                   bond_length: float = ACS1996_BOND_LENGTH):
    """Scale an Indigo CDXML tree to ``bond_length`` points and stamp the ACS Document 1996 style (line widths, black Arial labels) onto the root, in place.
    ``label_size`` is the atom/caption font size in points (house 7; ACS uses 10);
    ``bond_length`` is the drawn bond length in points (house default BOND_LENGTH_PT = 10.8 = 0.2 in at 75 %)."""
    try:
        indigo_bl = float(root.get("BondLength", "30"))
    except ValueError:
        indigo_bl = 30.0
    s = bond_length / indigo_bl

    def _scale(text):
        return " ".join(f"{float(v) * s:.6f}" for v in text.split())

    for el in root.iter():
        for a in _GEOM_ATTRS:
            if a in el.attrib:
                el.set(a, _scale(el.attrib[a]))
        for a in _ARROW_SIZE_ATTRS:                # keep arrowheads proportional
            if a in el.attrib:
                try:
                    el.set(a, f"{float(el.attrib[a]) * s:.6f}")
                except ValueError:
                    pass

    root.set("BondLength", f"{bond_length:.6f}")
    for k, v in _ACS1996_ROOT.items():
        root.set(k, v)
    root.set("LabelSize", f"{label_size:g}")        # override the ACS-default 10 pt
    root.set("CaptionSize", f"{label_size:g}")
    for s_el in root.iter("s"):                    # atom labels + captions -> Arial, black
        s_el.set("font", "2")
        s_el.set("size", f"{label_size:g}")
        s_el.set("color", "0")                     # colour id 0 == black
    return root


def _normalize_page(root: ET.Element, margin: float = 30.0):
    """Place the drawing where ChemDraw expects it: Indigo emits y-up around the origin, CDXML pages are y-down from the top-left, so flip Y, shift to ``margin`` and set the BoundingBox.
    """
    xs, ys = [], []
    for el in root.iter():
        for a in ("p", "Head3D", "Tail3D"):
            if a in el.attrib:
                v = el.attrib[a].split()
                xs.append(float(v[0])); ys.append(float(v[1]))
        if "BoundingBox" in el.attrib:
            b = el.attrib["BoundingBox"].split()
            xs += [float(b[0]), float(b[2])]; ys += [float(b[1]), float(b[3])]
    if not xs:
        return
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)

    def tx(x):
        return margin + (float(x) - min_x)

    def ty(y):
        return margin + (max_y - float(y))          # flip Y and translate

    for el in root.iter():
        for a in ("p", "Head3D", "Tail3D"):
            if a in el.attrib:
                v = el.attrib[a].split()
                v[0], v[1] = f"{tx(v[0]):.6f}", f"{ty(v[1]):.6f}"
                el.set(a, " ".join(v))
        if "BoundingBox" in el.attrib:
            b = el.attrib["BoundingBox"].split()
            b[0], b[1] = f"{tx(b[0]):.6f}", f"{ty(b[1]):.6f}"
            b[2], b[3] = f"{tx(b[2]):.6f}", f"{ty(b[3]):.6f}"
            el.set("BoundingBox", " ".join(b))

    width, height = (max_x - min_x) + 2 * margin, (max_y - min_y) + 2 * margin
    page = root.find("page")
    if page is not None:
        page.set("BoundingBox", f"0 0 {width:.6f} {height:.6f}")


def reactions_to_cdxml(rxn_smiles: Sequence[str], out_path: str,
                       clean: bool = True, style: str = "acs1996") -> str:
    """Write several reactions into one ``.cdxml``, stacked top to bottom, without ChemDraw's ``abs``/``Chiral`` enhanced-stereo labels. Returns ``out_path``. Needs ``indigo`` (``pip install epam.indigo``).
    """
    from indigo import Indigo
    ind = Indigo()
    ind.setOption("ignore-stereochemistry-errors", True)

    # document skeleton (header + font/colour tables) taken from a trivial molecule
    tmpl = ET.fromstring(ind.loadMolecule("C").cdxml())
    root = ET.Element("CDXML", dict(tmpl.attrib))
    for tag in ("fonttable", "colortable"):
        e = tmpl.find(tag)
        if e is not None:
            root.append(e)
    page = ET.SubElement(root, "page", {"HeightPages": "1", "WidthPages": "1"})

    idc = [0]                                   # running id counter (mutable cell)

    def nid():
        idc[0] += 1
        return idc[0]

    running_top = 0.0
    for smi in rxn_smiles:
        if clean:
            smi = clean_rxn_smiles(smi)
        reactants, _agents, products = chem.rxn_parts(smi)
        r_smis = [s for s in reactants.split(".") if s]
        p_smis = [s for s in products.split(".") if s]

        row, react_ids, prod_ids = [], [], []
        x = [0.0]

        def place(smis, bucket):
            for i, s in enumerate(smis):
                frag = _mol_fragment(ind, s)
                if frag is None:
                    continue
                idc[0] = _reindex(frag, idc[0])         # make ids unique
                minx, miny, maxx, maxy = _node_bbox(frag)
                _shift_elem(frag, x[0] - minx, -(miny + maxy) / 2.0)   # left-align, y-centre
                row.append(frag); bucket.append(int(frag.get("id")))
                x[0] += (maxx - minx)
                if i < len(smis) - 1:                   # "+" between co-reactants
                    row.append(_plus_graphic(x[0] + _MOL_GAP / 2.0, 0.0, nid()))
                    x[0] += _MOL_GAP

        place(r_smis, react_ids)
        ax0 = x[0] + _ARROW_PAD
        ax1 = ax0 + _ARROW_LEN
        arrow_id = nid()
        row.append(_arrow(ax0, ax1, 0.0, arrow_id))
        x[0] = ax1 + _ARROW_PAD
        place(p_smis, prod_ids)

        scheme = ET.Element("scheme")
        ET.SubElement(scheme, "step", {
            "ReactionStepReactants": " ".join(map(str, react_ids)),
            "ReactionStepProducts": " ".join(map(str, prod_ids)),
            "ReactionStepArrows": str(arrow_id)})

        # drop this reaction below the previous one
        rmin, rmax = _elem_y_range(row)
        dy = running_top - rmax
        for el in row:
            _shift_elem(el, 0.0, dy)
            page.append(el)
        page.append(scheme)
        running_top = (rmin + dy) - _ROW_GAP

    if style == "acs1996":
        _apply_acs1996(root)
    elif style is not None:
        raise ValueError(f"unknown style {style!r} (use 'acs1996' or None)")

    _normalize_page(root)                          # keep it on-page in ChemDraw
    ET.ElementTree(root).write(out_path, xml_declaration=True, encoding="UTF-8")
    return out_path


# Indigo/RDKit — editable CDXML, many *molecules* laid out in one file -----
# Grid layout constants, in Indigo's ~30-unit (bond-length) space.
_GRID_COL_GAP = 28.0     # horizontal gap between molecules in a row
_GRID_ROW_GAP = 42.0     # vertical gap between rows
_CAP_GAP = 14.0          # gap between a molecule and its caption
_CAP_H = 16.0            # reserved caption height (layout only)
_TITLE_GAP = 46.0        # extra space above a section title
_TITLE_PAD = 24.0        # gap between a section title and its first row


def _caption(text: str, x: float, y: float, idn: int, bold: bool = False) -> ET.Element:
    """A centred CDXML text label at (x, y); size/font/colour are set later by :func:`_apply_acs1996`, so captions share the atom-label house style."""
    t = ET.Element("t", {"id": str(idn), "p": f"{x:.6f} {y:.6f}", "Justification": "Center"})
    s = ET.SubElement(t, "s", {"font": "2"})
    if bold:
        s.set("face", "1")                          # 1 == bold in ChemDraw's face bitmask
    s.text = text
    return t


def _as_pairs(seq):
    """Normalise an item list to [(smiles, caption_or_None), ...]."""
    out = []
    for it in seq:
        if isinstance(it, (tuple, list)):
            out.append((it[0], it[1] if len(it) > 1 else None))
        else:
            out.append((it, None))
    return out


def molecules_to_cdxml(data, out_path: str, per_row: int = 6,
                       style: str = "acs1996", label_size: float = LABEL_PT,
                       bond_length: float = BOND_LENGTH_PT) -> str:
    """Write many molecules into one editable ``.cdxml``, grid-laid-out. ``data`` is SMILES or ``(smiles, caption)`` pairs, or ``{section_title: [...]}`` for titled grids. Returns ``out_path``. Needs ``indigo``.
    """
    from indigo import Indigo
    ind = Indigo()
    ind.setOption("ignore-stereochemistry-errors", True)

    tmpl = ET.fromstring(ind.loadMolecule("C").cdxml())
    root = ET.Element("CDXML", dict(tmpl.attrib))
    for tag in ("fonttable", "colortable"):
        e = tmpl.find(tag)
        if e is not None:
            root.append(e)
    page = ET.SubElement(root, "page", {"HeightPages": "1", "WidthPages": "1"})

    idc = [0]

    def nid():
        idc[0] += 1
        return idc[0]

    groups = ([(k, _as_pairs(v)) for k, v in data.items()]
              if isinstance(data, dict) else [(None, _as_pairs(data))])

    running_top = 0.0                               # y-up; rows march downward (decreasing)
    for title, items in groups:
        if title:
            running_top -= _TITLE_GAP
            page.append(_caption(title, 0.0, running_top, nid(), bold=True))
            running_top -= _TITLE_PAD
        for i in range(0, len(items), per_row):
            row, x, tops, bots = [], 0.0, [], []
            for smi, cap in items[i:i + per_row]:
                frag = _mol_fragment(ind, smi)
                if frag is None:
                    continue
                idc[0] = _reindex(frag, idc[0])
                minx, miny, maxx, maxy = _node_bbox(frag)
                _shift_elem(frag, x - minx, -(miny + maxy) / 2.0)   # left-align, y-centre
                row.append(frag)
                w, h = maxx - minx, maxy - miny
                tops.append(h / 2.0)
                if cap:
                    cap_y = -h / 2.0 - _CAP_GAP
                    row.append(_caption(cap, x + w / 2.0, cap_y, nid()))
                    bots.append(cap_y - _CAP_H)
                else:
                    bots.append(-h / 2.0)
                x += w + _GRID_COL_GAP
            if not row:
                continue
            dy = running_top - max(tops)
            for el in row:
                _shift_elem(el, 0.0, dy)
                page.append(el)
            running_top = min(bots) + dy - _GRID_ROW_GAP

    if style == "acs1996":
        _apply_acs1996(root, label_size=label_size, bond_length=bond_length)
    elif style is not None:
        raise ValueError(f"unknown style {style!r} (use 'acs1996' or None)")

    _normalize_page(root)
    ET.ElementTree(root).write(out_path, xml_declaration=True, encoding="UTF-8")
    return out_path


# Vector molecules inside matplotlib figures -----
# RDKit's SVG is just stroked bond paths (M/L) + filled glyph outlines (M/L/Q/Z), so it converts 1:1 into matplotlib PathPatches: the molecule stays TRUE VECTOR in the saved PDF, next to the plot it annotates (no raster images embedded).
import re as _re

MPL_BOND_LEN_PT = BOND_LENGTH_PT   # house style (ACS 1996, see top of module)
MPL_FONT_PT = LABEL_PT
MPL_LINE_PT = LINE_WIDTH_PT

_SVG_PATH = _re.compile(r"<path\b([^>]*?)/>", _re.S)
_SVG_ATTR = _re.compile(r"(\w[\w-]*)='([^']*)'")
_SVG_TOK = _re.compile(r"[A-Za-z]|-?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _svg_d_to_path(d: str):
    """Absolute M/L/Q/Z path data (all RDKit emits) -> matplotlib Path."""
    from matplotlib.path import Path
    verts, codes, cmd, start, i = [], [], None, (0.0, 0.0), 0
    toks = _SVG_TOK.findall(d)
    while i < len(toks):
        t = toks[i]
        if t.isalpha():
            cmd, i = t, i + 1
            if cmd == "Z":
                verts.append(start); codes.append(Path.CLOSEPOLY)
            elif cmd not in "MLQ":
                raise ValueError(f"unsupported SVG path command {cmd!r}")
            continue
        n = 4 if cmd == "Q" else 2
        xy = [float(v) for v in toks[i:i + n]]
        i += n
        if cmd == "M":
            start = (xy[0], xy[1])
            verts.append(start); codes.append(Path.MOVETO)
            cmd = "L"                          # implicit lineto after a moveto
        elif cmd == "L":
            verts.append((xy[0], xy[1])); codes.append(Path.LINETO)
        else:                                  # quadratic Bezier: control + end
            verts += [(xy[0], xy[1]), (xy[2], xy[3])]
            codes += [Path.CURVE3, Path.CURVE3]
    return Path(verts, codes)


def _min_atom_gap(mol) -> float:
    """Smallest distance between any two atoms of the 2D layout, in bond lengths (median bond = 1). A value well below 1 means two atoms (nearly) coincide."""
    import itertools, statistics
    c = mol.GetConformer()
    pos = [c.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    if mol.GetNumBonds() == 0 or len(pos) < 2:
        return float("inf")
    bl = statistics.median((pos[b.GetBeginAtomIdx()] - pos[b.GetEndAtomIdx()]).Length()
                           for b in mol.GetBonds()) or 1.0
    return min((p - q).Length() for p, q in itertools.combinations(pos, 2)) / bl


def _layout_defects(mol, min_gap: float = 0.35):
    """(# crossing bond pairs, # atoms lying on a bond they are not part of (closer than 0.25 bond lengths; 0.45 for lettered atoms), # straight-line 2-connected ring atoms (RDKit labels such a carbon "C"), atoms closer than ``min_gap`` bond lengths?) — lower is better."""
    import statistics
    c = mol.GetConformer()
    P = [c.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    if not bonds:
        return 0, 0, 0, False
    bl = statistics.median((P[i] - P[j]).Length() for i, j in bonds) or 1.0
    # a lettered atom (heteroatom, charged, ...) needs more room than a bare vertex
    clear = [0.25 if at.GetAtomicNum() == 6 and not at.GetFormalCharge() else 0.45
             for at in mol.GetAtoms()]
    on_bond = set()
    for i, j in bonds:
        A, AB = P[i], P[j] - P[i]
        L2 = AB.LengthSq() or 1.0
        for k, Q in enumerate(P):
            if k not in (i, j):
                t = max(0.0, min(1.0, (Q - A).DotProduct(AB) / L2))
                if (Q - (A + AB * t)).Length() < clear[k] * bl:
                    on_bond.add(k)
    def ccw(a, b, c):
        return (c.y - a.y) * (b.x - a.x) - (b.y - a.y) * (c.x - a.x)
    cross = 0
    for x in range(len(bonds)):
        for y in range(x + 1, len(bonds)):
            (i, j), (k, l) = bonds[x], bonds[y]
            if len({i, j, k, l}) == 4 and \
                    ccw(P[i], P[j], P[k]) * ccw(P[i], P[j], P[l]) < 0 and \
                    ccw(P[k], P[l], P[i]) * ccw(P[k], P[l], P[j]) < 0:
                cross += 1
    linear = 0
    for at in mol.GetAtoms():
        nb = [n.GetIdx() for n in at.GetNeighbors()]
        if len(nb) == 2 and at.IsInRing():
            u, v = P[nb[0]] - P[at.GetIdx()], P[nb[1]] - P[at.GetIdx()]
            # RDKit labels a 2-connected carbon "C" once its bonds are within ~18 deg of straight (cos < -0.95, i.e. > ~162 deg; measured), so flag from there
            if u.Length() and v.Length() and u.DotProduct(v) / (u.Length() * v.Length()) < -0.95:
                linear += 1
    return cross, len(on_bond), linear, _min_atom_gap(mol) < min_gap


def _rdkit_template_coords(mol):
    """RDKit's own depiction with its ring-template library (textbook drawings of bridged / caged ring systems: norbornanes, cedranes, morphinans, ...)."""
    rdDepictor.SetPreferCoordGen(False)
    try:
        rdDepictor.Compute2DCoords(mol, useRingTemplates=True)
    finally:
        rdDepictor.SetPreferCoordGen(True)
    return mol


def _indigo_coords(mol):
    from indigo import Indigo
    im = Indigo().loadMolecule(Chem.MolToMolBlock(mol)); im.layout()
    alt = Chem.MolFromMolBlock(im.molfile(), sanitize=False, removeHs=False)
    if alt is None or alt.GetNumAtoms() != mol.GetNumAtoms():
        raise ValueError("indigo layout failed")
    mol.RemoveAllConformers()                               # same atom order -> copy coords
    mol.AddConformer(Chem.Conformer(alt.GetConformer()), assignId=True)
    return mol


def house_layout(mol, min_gap: float = 0.35):
    """RDKit mol with the house 2D layout; use it for every structure in a figure. Tries CoordGen (ring templates first for bridged molecules), then Indigo, and keeps the first candidate with no drawing defect, else the least-defective one.
    """
    from rdkit.Chem import rdMolDescriptors
    if isinstance(mol, str):
        smiles, mol = mol, Chem.MolFromSmiles(mol)
        if mol is None:
            raise ValueError(f"bad SMILES {smiles!r}")
    else:
        mol = Chem.Mol(mol)
    mol.RemoveAllConformers()
    makers = [rdCoordGen.AddCoords, _rdkit_template_coords, _indigo_coords]
    if rdMolDescriptors.CalcNumBridgeheadAtoms(mol):
        makers[:2] = makers[1::-1]
    best = None
    for make in makers:
        m = Chem.Mol(mol)
        try:
            make(m)
        except Exception:
            continue
        if m.GetNumConformers() == 0:
            continue
        bad = _layout_defects(m, min_gap)
        if best is None or bad < best[0]:
            best = (bad, m)
        if bad == (0, 0, 0, False):
            break
    mol = best[1] if best else mol
    if mol.GetNumConformers() and mol.GetNumBonds():
        # RDKit's fixedBondLength is pixels per coordinate unit (CoordGen bonds = 1);
        # ring templates / Indigo use other units -> normalise the median bond to 1 so the drawn bond equals the requested one and the labels keep their size after the drawing is scaled to the house bond length.
        import statistics
        conf = mol.GetConformer()
        bl = statistics.median((conf.GetAtomPosition(b.GetBeginAtomIdx())
                                - conf.GetAtomPosition(b.GetEndAtomIdx())).Length()
                               for b in mol.GetBonds())
        if bl > 0:
            for i in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(i, conf.GetAtomPosition(i) * (1.0 / bl))
    return mol


_HASH_DASH = _re.compile(r"<path class='(bond-\d+)[^']*' d='M ([-\d.]+),([-\d.]+) L ([-\d.]+),([-\d.]+)' "
                        r"style='([^']*stroke-width:)([\d.]+)(px[^']*)' />\n?")
_STROKE_W = _re.compile(r"<path class='bond-[^']*' d='[^']*' style='[^']*stroke-width:([\d.]+)px")


def thin_hashed_bonds(svg: str, n: int = HASH_DASHES, width_scale: float = HASH_WIDTH_SCALE) -> str:
    """Redraw RDKit's hashed wedges the ChemDraw way: keep ``n`` evenly spaced dashes instead of the ~20 the supersampled canvas produces, at full bond width and lengthened by ``width_scale``. Double bonds are untouched.
    """
    from collections import Counter, defaultdict
    groups = defaultdict(list)
    for m in _HASH_DASH.finditer(svg):
        groups[(m.group(1), m.group(7))].append(m)
    widths = Counter(_STROKE_W.findall(svg))
    edits = []
    for (_, w), ms in groups.items():
        N = len(ms)
        if N < 5 or N <= n:
            continue
        others = [x for x in widths if float(x) > float(w)]
        bond_w = max(others, key=lambda x: widths[x]) if others else w   # the ordinary bond width
        keep = {round((k + 1) * (N - 1) / n) for k in range(n)}
        for i, m in enumerate(ms):
            if i not in keep:
                edits.append((m.span(), ""))
                continue
            x1, y1, x2, y2 = (float(m.group(k)) for k in (2, 3, 4, 5))
            cx, cy, hx, hy = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2 * width_scale, (y2 - y1) / 2 * width_scale
            edits.append((m.span(), f"<path class='{m.group(1)}' d='M {cx - hx:.1f},{cy - hy:.1f} "
                                    f"L {cx + hx:.1f},{cy + hy:.1f}' style='{m.group(6)}{bond_w}{m.group(8)}' />\n"))
    for (a, b), rep in sorted(edits, reverse=True):
        svg = svg[:a] + rep + svg[b:]
    return svg


def mol_vector_paths(smiles: str, bond_len_pt: float = MPL_BOND_LEN_PT,
                     font_pt: float = MPL_FONT_PT, line_pt: float = MPL_LINE_PT,
                     tight: bool = False, abbrevs=None):
    """Draw ``smiles`` and return ``(items, width_pt, height_pt)``, ``items`` being ``(Path, kind, line_width_pt)`` in points with y down from a top-left origin and ``kind`` "stroke" or "fill". ``tight`` crops to the ink box; ``abbrevs`` condenses groups to labels.
    """
    import math
    from matplotlib.transforms import Affine2D
    mol = house_layout(smiles)
    if abbrevs:
        from rdkit.Chem import rdAbbreviations
        mol = rdAbbreviations.CondenseMolAbbreviations(mol, abbrevs, maxCoverage=0.6)
    d = rdMolDraw2D.MolDraw2DSVG(-1, -1)
    apply_house_style(d.drawOptions(), bond_len_pt, font_pt, line_pt)
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    svg = thin_hashed_bonds(d.GetDrawingText())           # ChemDraw-like 4 dashes per hash
    bls = sorted(math.hypot(p.x - q.x, p.y - q.y) for b in mol.GetBonds()
                 for p, q in [(d.GetDrawCoords(b.GetBeginAtomIdx()),
                               d.GetDrawCoords(b.GetEndAtomIdx()))])
    s = bond_len_pt / (bls[len(bls) // 2] if bls else bond_len_pt)   # px -> pt
    scale = Affine2D().scale(s)
    items = []
    for attrs in _SVG_PATH.findall(svg):
        at = dict(_SVG_ATTR.findall(attrs))
        style = at.get("style", "")
        lw = _re.search(r"stroke-width:([\d.]+)px", style)
        items.append((scale.transform_path(_svg_d_to_path(at["d"])),
                      "stroke" if "fill:none" in style else "fill",
                      float(lw.group(1)) * s if lw else 0.0))
    if tight and items:
        boxes = [(path.get_extents(), lw / 2) for path, _, lw in items]
        x0 = min(b.x0 - e for b, e in boxes); y0 = min(b.y0 - e for b, e in boxes)
        x1 = max(b.x1 + e for b, e in boxes); y1 = max(b.y1 + e for b, e in boxes)
        shift = Affine2D().translate(-x0, -y0)
        items = [(shift.transform_path(path), kind, lw) for path, kind, lw in items]
        return items, x1 - x0, y1 - y0
    return items, d.Width() * s, d.Height() * s


def draw_mol_mpl(ax, smiles: str, xy, transform=None, ha: str = "center",
                 va: str = "center", bond_len_pt: float = MPL_BOND_LEN_PT,
                 font_pt: float = MPL_FONT_PT, line_pt: float = MPL_LINE_PT,
                 color: str = "black", zorder: float = 5, tight: bool = False, abbrevs=None):
    """Draw a molecule as vector patches on ``ax`` at ``xy``, sized absolutely in points and so independent of the axes scale. Returns ``(width_pt, height_pt)``.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.transforms import Affine2D, ScaledTranslation
    items, w, h = mol_vector_paths(smiles, bond_len_pt, font_pt, line_pt, tight, abbrevs)
    dx = {"left": 0.0, "center": -w / 2, "right": -w}[ha]
    dy = {"top": 0.0, "center": h / 2, "bottom": h}[va]
    fig = ax.figure
    tr = (Affine2D().scale(1 / 72, -1 / 72).translate(dx / 72, dy / 72)
          + fig.dpi_scale_trans
          + ScaledTranslation(*xy, transform if transform is not None else ax.transData))
    for path, kind, lw in items:
        if kind == "stroke":
            p = PathPatch(path, fill=False, ec=color, lw=lw, capstyle="butt",
                          joinstyle="miter")
        else:
            p = PathPatch(path, fc=color, ec="none", lw=0)
        p.set_transform(tr); p.set_clip_on(False); p.set_zorder(zorder)
        ax.add_patch(p)
    return w, h
