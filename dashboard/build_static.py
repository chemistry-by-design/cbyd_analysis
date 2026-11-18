"""Bake the route viewer into ONE self-contained HTML file; no server needed.

``chem.build_network`` runs over every route once and the result ships inside the page, each route gzipped + base64'd separately and unpacked only when opened. Molecule nodes are CXSMILES carrying tree-aligned coordinates, wedge/dash choices (``wU``/``wD``) and group labels (``$OMe$``), which RDKit.js (RDKit compiled to WebAssembly) draws in the browser.

The page is a snapshot: served over http(s) the viewer also reads each route's JSON (``--live-data-url``, by default the routes folder next to the page) and marks what changed since the build (static/livecheck.js); opened from disk it shows the snapshot.
Issue / PR / e-mail links come from --repo and --email, or the CONFIG block in the generated HTML.

    python dashboard/build_static.py                     # -> dashboard.html
    python dashboard/build_static.py --dev               # one route -> dashboard.dev.html
    python dashboard/build_static.py -o docs/index.html  # e.g. for GitHub Pages
    python dashboard/build_static.py --limit 20          # quick test build
    python dashboard/build_static.py --cdn               # RDKit.js from a CDN
"""

import argparse
import base64
import contextlib
import gzip
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
STATIC_DIR = os.path.join(HERE, "static")
VENDOR_DIR = os.path.join(STATIC_DIR, "vendor")
sys.path.insert(0, REPO_ROOT)

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import RDConfig, rdDepictor  # noqa: E402

from dashboard import chem  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# Complexity metrics for the route-info chart: SAScore, Böttcher, SPS and NPScore.
# Each one is optional: a metric whose scorer cannot be imported is left out of the chart (Böttcher needs openbabel; SAScore / NPScore need RDKit's Contrib folder).
# The Böttcher scorer (dashboard/boettcher.py) is from the molcomplex package:
# Wright, B. A., et al. (2024). J. Am. Chem. Soc., 146 (48), 33130–33148. DOI: 10.1021/jacs.4c11252
def _load_scorers():
    scorers = {}
    try:
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer
        scorers["sascore"] = lambda smi, mol: sascorer.calculateScore(mol)
    except Exception:
        pass
    try:
        from openbabel import openbabel
        from dashboard.boettcher import BottchScorer
        openbabel.obErrorLog.SetOutputLevel(0)

        def _boettcher(smi, mol):
            obmol, conv = openbabel.OBMol(), openbabel.OBConversion()
            conv.SetInAndOutFormats("smi", "smi")
            conv.ReadString(obmol, smi)
            return BottchScorer(obmol, verbose=False).score(obmol)
        scorers["boettcher"] = _boettcher
    except Exception:
        pass
    try:
        from rdkit.Chem import SpacialScore
        scorers["sps"] = lambda smi, mol: SpacialScore.SPS(mol, normalize=False)
    except Exception:
        pass
    try:
        sys.path.append(os.path.join(RDConfig.RDContribDir, "NP_Score"))
        import npscorer
        model = []                           # ~20 MB; read on first use, per worker

        def _np(smi, mol):
            if not model:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    model.append(npscorer.readNPModel())
            return npscorer.scoreMol(mol, model[0])
        scorers["npscore"] = _np
    except Exception:
        pass
    return scorers


METRICS = ["sascore", "boettcher", "sps", "npscore"]      # order of the chart's dropdown
SCORERS = _load_scorers()
_SCORE_CACHE = {}

RDKIT_JS_VERSION = "2025.3.4-1.0.0"
RDKIT_JS_CDN = f"https://cdn.jsdelivr.net/npm/@rdkit/rdkit@{RDKIT_JS_VERSION}/dist/"

# parsed_reaction keys the viewer shows (names + the paired SMILES definitions).
PARSED_KEYS = [
    "GENERAL_REAGENTS", "CATALYSTS", "LIGANDS", "SOLVENTS", "MODIFIERS",
    "TEMPERATURE", "YIELD",
    "GENERAL_REAGENTS_DEFINITIONS", "CATALYSTS_DEFINITIONS",
    "LIGANDS_DEFINITIONS", "SOLVENTS_DEFINITIONS",
]

# CXSMILES extensions we need: 2D coords, atom labels (abbreviations), wedging.
_CX_FLAGS = (Chem.CXSmilesFields.CX_COORDS
             | Chem.CXSmilesFields.CX_ATOM_LABELS
             | Chem.CXSmilesFields.CX_BOND_CFG)


_CX_COORDS_RE = re.compile(r"\|\(([^)]*)\)")
_NUM_RE = re.compile(r"-?\d+\.\d+")


def _round_coord(m):
    s = f"{float(m.group(0)):.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _cx(mol):
    """Mol -> CXSMILES carrying its current 2D coords, atom labels and wedges.
    Coords are cut to 2 decimals (sub-pixel at any size we draw) — they are most of the file size."""
    ps = Chem.SmilesWriteParams()
    cx = Chem.MolToCXSmiles(mol, ps, int(_CX_FLAGS))
    return _CX_COORDS_RE.sub(
        lambda m: "|(" + _NUM_RE.sub(_round_coord, m.group(1)) + ")", cx, count=1)


def _wedged(molblock, wedges):
    """Rebuild a node's mol from its aligned molblock with OUR wedge plan applied so the CXSMILES carries those wedges."""
    mol = Chem.MolFromMolBlock(molblock)
    if mol is None:
        return None
    for b in mol.GetBonds():                 # drop the molblock's default wedging
        if b.HasProp("_MolFileBondCfg"):
            b.ClearProp("_MolFileBondCfg")
        if b.HasProp("_MolFileBondStereo"):
            b.ClearProp("_MolFileBondStereo")
    chem.apply_wedges(mol, wedges)
    return mol


def _mol_node(n):
    out = {"id": n["id"], "t": "m", "s": n["smiles"], "r": n["role"]}
    mb = n.get("molblock")
    if mb:
        wedges = [tuple(int(x) for x in p) for p in (n.get("wedges") or [])]
        mol = _wedged(mb, wedges) if wedges else Chem.MolFromMolBlock(mb)
        if mol is not None:
            conf = mol.GetConformer()
            if any(not math.isfinite(v) for pos in conf.GetPositions() for v in pos):
                rdDepictor.Compute2DCoords(mol, forceRDKit=True)
            out["cx"] = _cx(mol)
            if wedges:
                out["wd"] = 1                # draw with OUR wedges, don't recompute
            if n.get("ext_w_a") is not None:
                amol, _ = chem.abbreviate_mol(mol)
                if amol is not None:
                    out["cxa"] = _cx(amol)
                    out["ewa"], out["eha"] = n["ext_w_a"], n["ext_h_a"]
        out["ew"], out["eh"] = n.get("ext_w"), n.get("ext_h")
        out["na"] = n.get("n_atoms")
    # Non-finite extents are unusable; omit them so the viewer uses its defaults.
    for key in ("ew", "eh", "ewa", "eha"):
        if out.get(key) is not None and not math.isfinite(out[key]):
            del out[key]
    return out


def _scores(smiles):
    """{metric: score | None} for one molecule (cached per worker)."""
    if smiles not in _SCORE_CACHE:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        out = {}
        for m, fn in SCORERS.items():
            try:
                v = float(fn(smiles, mol)) if mol is not None else None
                out[m] = round(v, 2) if v is not None and math.isfinite(v) else None
            except Exception:
                out[m] = None
        _SCORE_CACHE[smiles] = out
    return _SCORE_CACHE[smiles]


def complexity_evolution(graph):
    """Complexity along the route's longest linear sequence, starting material first, target last; where two reactants are equally deep, the larger one is followed.

    Returns ``{"p": [[reaction node id | None, molecule node id], ...], "y": {metric: [score | None, ...]}}`` -- node ids, so the viewer can tie each point / line to its node in the tree -- or None when there is nothing to plot.
    """
    if not SCORERS:
        return None
    nodes = {n["id"]: n for n in graph["nodes"]}
    made_by, reacts = {}, {}
    for e in graph["edges"]:
        s, t = nodes[e["source"]], nodes[e["target"]]
        if s["type"] == "reaction" and t["type"] == "molecule":
            made_by[t["id"]] = s["id"]
        elif s["type"] == "molecule" and t["type"] == "reaction":
            reacts.setdefault(t["id"], []).append(s["id"])

    depth = {}

    def _d(m):
        if m not in depth:
            rx = made_by.get(m)
            depth[m] = 0 if rx is None else 1 + max((_d(r) for r in reacts.get(rx, [])), default=0)
        return depth[m]

    def _heavy(m):
        mol = Chem.MolFromSmiles(nodes[m]["smiles"])
        return mol.GetNumHeavyAtoms() if mol else 0

    used = {m for v in reacts.values() for m in v}
    targets = ([i for i, n in nodes.items() if n["type"] == "molecule" and n.get("role") == "target"]
               or [m for m in made_by if m not in used])
    if not targets:
        return None
    cur, path = max(targets, key=_d), []
    while True:
        rx = made_by.get(cur)
        path.append([rx, cur])
        if rx is None or not reacts.get(rx):
            break
        cur = max(reacts[rx], key=lambda m: (_d(m), _heavy(m)))
    if len(path) < 2:
        return None
    path.reverse()
    scores = [_scores(nodes[m]["smiles"]) for _, m in path]
    return {"p": path, "y": {m: [sc[m] for sc in scores] for m in METRICS if m in SCORERS}}


def export_route(path):
    """One route file -> (index row, gzipped+base64 payload) or (name, error)."""
    name = os.path.basename(path)[: -len(".json")]
    try:
        with open(path) as f:
            data = chem.normalize_route(json.load(f))
        bg = data.get("background", {}) or {}
        synthesis = data.get("synthesis", []) or []
        graph = chem.build_network(synthesis)

        reactions = []
        for i, rxn in enumerate(synthesis):
            parsed = rxn.get("parsed_reaction", {}) or {}
            row = {
                "rid": str(rxn.get("reaction_id", i)),
                "d": rxn.get("raw_description", "") or "",
                "m": rxn.get("full_reaction_smiles_mapped", "") or "",
                "p": {k: parsed[k] for k in PARSED_KEYS if parsed.get(k)},
            }
            # What the tree was built from, so the viewer can compare this build with the route file as it is now (see static/livecheck.js):
            #   s = source reaction SMILES (only when it isn't the mapped one)
            #   c = [reactants, products] molecule-node SMILES, or null if skipped
            #   a = reagents written in the reaction SMILES itself
            _, source, agents, cleaned, _ = chem.reaction_parts(rxn)
            if source != row["m"].strip():
                row["s"] = source
            row["c"] = [list(cleaned[0]), list(cleaned[1])] if cleaned else None
            if agents:
                row["a"] = agents
            reactions.append(row)

        nodes = []
        for n in graph["nodes"]:
            if n["type"] == "molecule":
                nodes.append(_mol_node(n))
            else:
                rn = {"id": n["id"], "t": "r", "i": n["index"]}
                if n.get("invalid"):
                    rn["inv"] = 1
                nodes.append(rn)

        payload = {
            "bg": bg,
            "rx": reactions,
            "nodes": nodes,
            "edges": [[e["source"], e["target"]] for e in graph["edges"]],
            "warn": graph.get("warnings", []),
        }
        try:
            cv = complexity_evolution(graph)
        except Exception:                      # e.g. a cyclic route: no chart
            cv = None
        if cv:
            payload["cv"] = cv
        blob = gzip.compress(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(), 9)
        compound = " | ".join(str(v) for v in (bg.get("name"), bg.get("drug name")) if v)
        row = {
            "name": name,
            "title": bg.get("name") or name,
            "author": bg.get("author"),
            "year": bg.get("year"),
            "n_steps": len(synthesis),
            "q_compound": compound.lower(),
            "q_doi": str(bg.get("doi") or "").lower(),
        }
        return row, base64.b64encode(blob).decode()
    except Exception as e:  # keep going; report at the end
        return name, f"{type(e).__name__}: {e}"


def _fetch(url, dest):
    if not os.path.exists(dest):
        print(f"  downloading {url}")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    with open(dest, "rb") as f:
        return f.read()


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _inline_js(src):
    # a literal "</script" inside inlined JS would end the <script> element early
    return src.replace("</script", "<\\/script")


DEFAULT_REPO = "chemistry-by-design/cbyd"


def _default_repo():
    """owner/name from the git remote if it points at GitHub, else DEFAULT_REPO."""
    try:
        url = subprocess.run(["git", "-C", REPO_ROOT, "remote", "get-url", "origin"],
                             capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return DEFAULT_REPO
    m = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
    return m.group(1) if m else DEFAULT_REPO


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--routes", default=os.path.join(REPO_ROOT, "data", "routes"),
                    help="folder of route JSON files (default: data/routes)")
    ap.add_argument("-o", "--out", default=None,
                    help="output HTML file (default: dashboard.html; --dev: dashboard.dev.html)")
    ap.add_argument("--repo", default=_default_repo(),
                    help="GitHub owner/name for the issue + PR links (default: the git remote, or " + DEFAULT_REPO + ")")
    ap.add_argument("--branch", default="main", help="branch the PR links edit (default: main)")
    ap.add_argument("--email", default="cbc-cbyd@arizona.edu",
                    help="address for the 'Email' link (default: cbc-cbyd@arizona.edu; \"\" for none)")
    ap.add_argument("--data-path", default="data/routes",
                    help="path of the route files inside the GitHub repo (for PR links)")
    ap.add_argument("--live-data-url", default=None,
                    help="where the viewer looks for the current route files, to show what "
                         "changed since this build (default: the --routes folder, relative "
                         "to the output file; may be an absolute URL)")
    ap.add_argument("--no-live-check", action="store_true",
                    help="never look for the route files; always show the built routes")
    ap.add_argument("--cdn", action="store_true",
                    help="load RDKit.js from a CDN instead of inlining it (~9 MB smaller)")
    selection = ap.add_mutually_exclusive_group()
    selection.add_argument("--dev", action="store_true",
                           help="build only the first route, default to one worker, and write "
                                "dashboard.dev.html unless -o is set")
    selection.add_argument("--limit", type=int, default=0,
                           help="only the first N routes (testing)")
    ap.add_argument("-j", "--jobs", type=int, default=None,
                    help="worker count (default: CPU count; --dev: 1)")
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(REPO_ROOT, "dashboard.dev.html" if args.dev else "dashboard.html")
    if args.jobs is None:
        args.jobs = 1 if args.dev else (os.cpu_count() or 1)
    if args.dev:
        args.limit = 1

    files = sorted((f for f in os.listdir(args.routes) if f.endswith(".json")),
                   key=lambda f: (f.lower(), f))
    if args.limit:
        files = files[: args.limit]
    paths = [os.path.join(args.routes, f) for f in files]
    missing = [m for m in METRICS if m not in SCORERS]
    if missing:
        print(f"  ! no scorer for {', '.join(missing)}: left out of the complexity chart"
              + (" (Böttcher needs openbabel: pip install openbabel-wheel)" if "boettcher" in missing else ""))
    print(f"Exporting {len(paths)} routes with {args.jobs} workers…")

    t0 = time.time()
    index, blobs, failed = [], {}, []
    with Pool(args.jobs) as pool:
        for k, (row, blob) in enumerate(pool.imap(export_route, paths, chunksize=4), 1):
            if isinstance(row, dict):
                row["number"] = k
                index.append(row)
                blobs[row["name"]] = blob
            else:
                failed.append((row, blob))
            if k % 250 == 0 or k == len(paths):
                print(f"  {k}/{len(paths)}  ({time.time() - t0:.0f}s)")
    for name, err in failed:
        print(f"  ! skipped {name}: {err}")

    if args.cdn:
        rdkit_js = f'<script src="{RDKIT_JS_CDN}RDKit_minimal.js"></script>'
        rdkit_wasm = ""
    else:
        js = _fetch(RDKIT_JS_CDN + "RDKit_minimal.js",
                    os.path.join(VENDOR_DIR, f"rdkit-{RDKIT_JS_VERSION}", "RDKit_minimal.js"))
        wasm = _fetch(RDKIT_JS_CDN + "RDKit_minimal.wasm",
                      os.path.join(VENDOR_DIR, f"rdkit-{RDKIT_JS_VERSION}", "RDKit_minimal.wasm"))
        rdkit_js = f"<script>{_inline_js(js.decode())}</script>"
        rdkit_wasm = base64.b64encode(wasm).decode()

    live_url = args.live_data_url
    if live_url is None:
        live_url = os.path.relpath(os.path.abspath(args.routes),
                                   os.path.dirname(os.path.abspath(args.out))).replace(os.sep, "/")
    config = {
        "repo": args.repo, "branch": args.branch, "email": args.email,
        "dataPath": args.data_path.strip("/"),
        "liveCheck": not args.no_live_check,
        "liveDataUrl": live_url.rstrip("/"),
        "rdkitWasmUrl": RDKIT_JS_CDN + "RDKit_minimal.wasm" if args.cdn else "",
        "built": time.strftime("%Y-%m-%d"),
    }
    vendor = "\n".join(
        f"<script>{_inline_js(_read(os.path.join(VENDOR_DIR, f)))}</script>"
        for f in ("dagre.min.js", "cytoscape.min.js", "cytoscape-dagre.js"))
    as_json = lambda o: json.dumps(o, separators=(",", ":")).replace("</", "<\\/")  # noqa: E731

    html = _read(os.path.join(STATIC_DIR, "standalone.html"))
    for key, val in {
        "/*__CONFIG__*/": json.dumps(config, indent=2),
        "/*__CSS__*/": _read(os.path.join(STATIC_DIR, "style.css")),
        "<!--__VENDOR__-->": vendor,
        "<!--__RDKIT_JS__-->": rdkit_js,
        "__RDKIT_WASM__": rdkit_wasm,
        "__INDEX__": as_json(index),
        "__ROUTES__": as_json(blobs),
        "/*__APP_JS__*/": _inline_js(_read(os.path.join(STATIC_DIR, "copy.js")) + "\n" +
                                     _read(os.path.join(STATIC_DIR, "route-list.js")) + "\n" +
                                     _read(os.path.join(STATIC_DIR, "route-filters.js")) + "\n" +
                                     _read(os.path.join(STATIC_DIR, "livecheck.js")) + "\n" +
                                     _read(os.path.join(STATIC_DIR, "standalone.js"))),
    }.items():
        assert key in html, f"template placeholder missing: {key}"
        html = html.replace(key, val, 1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    mb = os.path.getsize(args.out) / 1e6
    print(f"Wrote {args.out}  ({len(index)} routes, {mb:.1f} MB, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
