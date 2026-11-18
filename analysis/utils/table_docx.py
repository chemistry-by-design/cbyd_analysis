"""DataFrames -> one editable Word file of numbered tables ("Table S1. caption", ...), written as plain OOXML.
"""
from __future__ import annotations

import zipfile
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from PIL import ImageFont

FONT, MONO = "Times New Roman", "Courier New"
FONT_PT = 12
PAD = 8                     # cell padding (left + right), pt
GAP = 12                    # spacer column before each folded block, pt
PAGE = (12240, 15840)       # US Letter, twips
MARGIN = 1440               # 1 in -> 6.5 in text width
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


@lru_cache(None)
def _ttf(family, bold):
    try:
        f = fm.findfont(fm.FontProperties(family=family, weight="bold" if bold else "normal"),
                        fallback_to_default=False)
        return ImageFont.truetype(f, FONT_PT * 10)
    except Exception:
        return None


def _width(s, family=FONT, bold=False):
    """Width of ``s`` in pt at FONT_PT (rough estimate if the font is missing)."""
    f = _ttf(family, bold)
    return f.getlength(s) / 10 if f else 0.55 * FONT_PT * len(s)


def fixed(v, dec):
    """``v`` with ``dec`` decimals and thousands separators, rounded half-up from its shortest decimal form, so a tie such as 25.95 gives "26.0" rather than the "25.9" that binary-float formatting gives."""
    q = Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-dec), rounding=ROUND_HALF_UP)
    return f"{q:,.{dec}f}"


def _fmt(v, dec):
    """Cell text: thousands separators, fixed decimals, en dash for missing."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return "–"
    if isinstance(v, (bool, np.bool_)):
        return "Yes" if v else "No"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return fixed(v, dec)
    return str(v)


def _column_text(s: pd.Series, dec):
    num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    if num and s.dropna().apply(lambda x: float(x).is_integer()).all():
        s = s.astype("Int64")                    # whole numbers -> no decimals
    return [_fmt(v, dec) for v in s.astype(object)], num


def _fold(lines, blocks, labels):
    """Split the table lines into ``blocks`` side-by-side chunks of near-equal length;
    a chunk never ends on a group label and one that starts inside a group repeats its label as "... (cont.)"."""
    m = -(-len(lines) // blocks)
    chunks, cur = [], []
    for ln in lines:
        if len(cur) >= m and len(chunks) < blocks - 1:
            carry = [cur.pop()] if cur[-1][0] == "label" else []
            chunks.append(cur)
            cur = carry
            if not carry and ln[0] == "row" and labels is not None:
                cur.append(("label", labels[ln[1]] + " (cont.)"))
        cur.append(ln)
    return chunks + [cur]


def _col_widths(heads, texts, numeric, is_smiles, nb, width):
    """Column widths (pt) of ``nb`` repeated blocks: the longest cell or header word;
    if too wide, water-filling -- numbers never wrap and short text columns keep their width, only the widest text columns (SMILES, names) are capped at a common width L and wrap, never narrower than their longest header word."""
    word = [max(_width(x, bold=True) for x in str(h).split() or [""]) + PAD for h in heads]
    nat = [max([_width(x, MONO if sm else FONT) for x in t] + [0]) + PAD
           for t, sm in zip(texts, is_smiles)]
    nat = np.tile(np.maximum(nat, word), nb)
    room = width - GAP * (nb - 1)
    if nat.sum() > room:
        minw, wrap = np.tile(word, nb), ~np.tile(numeric, nb)
        avail = room - nat[~wrap].sum()
        lo, hi = 0.0, nat.max()
        for _ in range(60):                             # bisect for L
            L = (lo + hi) / 2
            if np.maximum(np.minimum(nat[wrap], L), minw[wrap]).sum() > avail:
                hi = L
            else:
                lo = L
        nat[wrap] = np.maximum(np.minimum(nat[wrap], lo), minw[wrap])
        nat = nat * min(1.0, room / nat.sum())
    return nat


# OOXML pieces -----
def _run(text, bold=False, mono=False):
    rpr = (f'<w:rFonts w:ascii="{MONO}" w:hAnsi="{MONO}" w:cs="{MONO}"/>' if mono else "") \
        + ("<w:b/><w:bCs/>" if bold else "")
    return f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def _para(runs, right=False, keep=False, page_break=False, before=0, after=0):
    ppr = ("<w:keepNext/>" if keep else "") + ("<w:pageBreakBefore/>" if page_break else "") \
        + f'<w:spacing w:before="{before}" w:after="{after}"/>' + ('<w:jc w:val="right"/>' if right else "")
    return f"<w:p><w:pPr>{ppr}</w:pPr>{runs}</w:p>"


def _border(top=0, bottom=0):
    line = lambda side, sz: f'<w:{side} w:val="single" w:sz="{sz}" w:space="0" w:color="000000"/>'
    return (line("top", top) if top else "") + (line("bottom", bottom) if bottom else "")


def _cell(width_pt, para, span=1, borders=""):
    tcpr = f'<w:tcW w:w="{round(width_pt * 20)}" w:type="dxa"/>' \
        + (f'<w:gridSpan w:val="{span}"/>' if span > 1 else "") \
        + (f"<w:tcBorders>{borders}</w:tcBorders>" if borders else "")
    return f"<w:tc><w:tcPr>{tcpr}</w:tcPr>{para}</w:tc>"


def _table(df, decimals, headers, width, group=None, blocks=1):
    labels = None
    if group is not None:                               # shown as spanning rows instead
        labels = df[group].astype(str).tolist()
        df = df.drop(columns=group)
    cols, texts, numeric = list(df.columns), [], []
    for c in cols:
        dec = decimals.get(c, decimals.get("*", 1)) if isinstance(decimals, dict) else decimals
        t, num = _column_text(df[c], dec)
        texts.append(t); numeric.append(num)
    heads = [headers.get(c, str(c)) for c in cols]
    is_smiles = ["smiles" in str(c).lower() for c in cols]

    # table lines: a bold group label whenever the group changes, then the data rows;
    # folded into side-by-side blocks that repeat the columns
    lines = []
    for i in range(len(df)):
        if labels is not None and (i == 0 or labels[i] != labels[i - 1]):
            lines.append(("label", labels[i]))
        lines.append(("row", i))
    chunks = _fold(lines, blocks, labels)
    k, nb = len(cols), len(chunks)
    cw = _col_widths(heads, texts, numeric, is_smiles, nb, width)
    nrows = max(map(len, chunks))
    keep = nrows <= 40                                  # keep short tables on one page

    def gap(borders=""):                                # spacer cell between blocks
        return _cell(GAP, _para("", keep=keep), borders=borders)

    hb = _border(top=8, bottom=4)                       # 1 pt above, 0.5 pt below the header
    head = []
    for b in range(nb):
        if b:
            head.append(gap(hb))
        head += [_cell(cw[b * k + j], _para(_run(str(heads[j]), bold=True), right=numeric[j], keep=True),
                       borders=hb) for j in range(k)]
    rows = ["<w:tr><w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>" + "".join(head) + "</w:tr>"]
    for r in range(nrows):
        last = r == nrows - 1
        bb = _border(bottom=8) if last else ""          # 1 pt below the last row
        kp = keep and not last
        cells = []
        for b, ch in enumerate(chunks):
            if b:
                cells.append(gap(bb))
            w = cw[b * k:(b + 1) * k]
            if r >= len(ch):
                cells += [_cell(w[j], _para("", keep=kp), borders=bb) for j in range(k)]
            elif ch[r][0] == "label":                   # group label row, spans the block
                cells.append(_cell(w.sum(), _para(_run(ch[r][1], bold=True), keep=True,
                                                  before=120 if nb == 1 and r else 0),
                                   span=k, borders=bb))
            else:
                i = ch[r][1]
                cells += [_cell(w[j], _para(_run(texts[j][i], mono=is_smiles[j]), right=numeric[j],
                                            keep=kp), borders=bb) for j in range(k)]
        rows.append("<w:tr><w:trPr><w:cantSplit/></w:trPr>" + "".join(cells) + "</w:tr>")

    grid = []
    for b in range(nb):
        if b:
            grid.append(GAP)
        grid += list(cw[b * k:(b + 1) * k])
    return ("<w:tbl><w:tblPr>"
            f'<w:tblW w:w="{round(sum(grid) * 20)}" w:type="dxa"/><w:tblLayout w:type="fixed"/>'
            '<w:tblCellMar><w:left w:w="60" w:type="dxa"/><w:right w:w="60" w:type="dxa"/></w:tblCellMar>'
            "</w:tblPr><w:tblGrid>" + "".join(f'<w:gridCol w:w="{round(g * 20)}"/>' for g in grid)
            + "</w:tblGrid>" + "".join(rows) + "</w:tbl>")


_STYLES = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W}"><w:docDefaults><w:rPrDefault><w:rPr>
<w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}" w:eastAsia="{FONT}" w:cs="{FONT}"/>
<w:sz w:val="{FONT_PT * 2}"/><w:szCs w:val="{FONT_PT * 2}"/><w:lang w:val="en-US"/></w:rPr></w:rPrDefault>
<w:pPrDefault><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr></w:pPrDefault>
</w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>
<w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="caption"/><w:basedOn w:val="Normal"/>
<w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:after="80"/></w:pPr></w:style>
<w:style w:type="table" w:default="1" w:styleId="TableNormal"><w:name w:val="Normal Table"/>
<w:semiHidden/><w:unhideWhenUsed/><w:tblPr><w:tblInd w:w="0" w:type="dxa"/><w:tblCellMar>
<w:top w:w="0" w:type="dxa"/><w:left w:w="108" w:type="dxa"/><w:bottom w:w="0" w:type="dxa"/>
<w:right w:w="108" w:type="dxa"/></w:tblCellMar></w:tblPr></w:style></w:styles>"""

_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

_DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""


def tables_to_docx(specs, path, prefix="S", start=1, title=None):
    """Write ``specs`` to one .docx, numbered ``Table {prefix}{start}`` onwards with Word SEQ fields. Each spec has ``df`` and ``caption``, optionally ``decimals``, ``headers``, ``group`` (a bold spanning row when that column changes), ``blocks`` (fold into side-by-side column blocks) and ``same_page``.
    """
    width = (PAGE[0] - 2 * MARGIN) / 20
    body = [_para(_run(title, bold=True), after=240)] if title else []
    for k, s in enumerate(specs, start=start):
        new_page = k > start and not s.get("same_page")
        num = (f'<w:fldSimple w:instr=" SEQ Table \\* ARABIC "><w:r><w:rPr><w:b/><w:bCs/></w:rPr>'
               f"<w:t>{k}</w:t></w:r></w:fldSimple>")
        body.append('<w:p><w:pPr><w:pStyle w:val="Caption"/>'
                    + ("<w:pageBreakBefore/>" if new_page else "")
                    + (f'<w:spacing w:before="{0 if new_page or k == start else 360}"/></w:pPr>')
                    + _run(f"Table {prefix}", bold=True) + num + _run(".", bold=True)
                    + _run(" " + s["caption"]) + "</w:p>")
        body.append(_table(s["df"].reset_index(drop=True), s.get("decimals", 1), s.get("headers", {}),
                           width, s.get("group"), s.get("blocks", 1)))
    body.append(_para(""))                              # a document must end with a paragraph
    sect = (f'<w:sectPr><w:pgSz w:w="{PAGE[0]}" w:h="{PAGE[1]}"/><w:pgMar w:top="{MARGIN}" '
            f'w:right="{MARGIN}" w:bottom="{MARGIN}" w:left="{MARGIN}" w:header="720" '
            f'w:footer="720" w:gutter="0"/></w:sectPr>')
    doc = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<w:document xmlns:w="{W}">'
           f"<w:body>{''.join(body)}{sect}</w:body></w:document>")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _TYPES)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("word/_rels/document.xml.rels", _DOC_RELS)
        z.writestr("word/styles.xml", _STYLES)
        z.writestr("word/document.xml", doc)
    return path
