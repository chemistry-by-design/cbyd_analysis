"use strict";

// The read-only route viewer (built into one HTML file by dashboard/build_static.py).
// Route data is embedded in the page and every structure is drawn in the browser
// by RDKit.js. "Report a mistake" links (e-mail / GitHub issue / pull request) are
// configured by window.DASH_CONFIG.
//
// The embedded routes are a snapshot. When a route is opened the viewer also looks
// for that route's JSON file and, if it has changed since the build, shows the
// file's version with the changes marked in purple (see livecheck.js, which is
// inlined just above this script). No file reachable -> the snapshot.

cytoscape.use(cytoscapeDagre);

const CONFIG = Object.assign(
  { repo: "", branch: "main", email: "", dataPath: "data/routes", rdkitWasmUrl: "", built: "",
    liveCheck: true, liveDataUrl: "" },
  window.DASH_CONFIG || {});

const ROLE_COLOR = {
  start: "#48c479",
  intermediate: "#4b8fd4",
  target: "#d66756",
  byproduct: "#9aa0ac",
};
const RXN_YELLOW = "#fff2ca";
const RXN_BORDER = "#eece3f";
const LIVE_PURPLE = "#a55ac5";   // "changed in the route file since this page was built"

// Reagent/condition categories. The first four carry a paired *_DEFINITIONS
// SMILES list (DEF_FOR); the rest are name-only.
const REAGENT_FIELDS = [
  ["GENERAL_REAGENTS", "Reagents"],
  ["CATALYSTS", "Catalysts"],
  ["LIGANDS", "Ligands"],
  ["SOLVENTS", "Solvents"],
  ["MODIFIERS", "Modifiers"],
  ["TEMPERATURE", "Temperature"],
  ["YIELD", "Yield"],
];
const DEF_FOR = {
  GENERAL_REAGENTS: "GENERAL_REAGENTS_DEFINITIONS",
  CATALYSTS: "CATALYSTS_DEFINITIONS",
  LIGANDS: "LIGANDS_DEFINITIONS",
  SOLVENTS: "SOLVENTS_DEFINITIONS",
};
const BG_FIELDS = [
  ["name", "name"], ["author", "author"], ["year", "year"],
  ["drug name", "drug name"], ["manufacturer", "manufacturer"],
  ["citation", "citation"], ["doi", "doi"],
];

let RDKit = null;
let LIVE = null;        // LiveCheck bound to RDKit (set once the engine is up)
let CY = null;
let CURRENT = null;
let LAST_MOL = null;    // last molecule node shown (so a toggle can refresh the panel)
let PANEL = null;       // bottom panel: {kind:'reaction'|'molecule'|'report', ...} — for refresh

// ---- embedded data ---------------------------------------------------------
const INDEX = JSON.parse(document.getElementById("route-index").textContent);
const ROUTE_BLOBS = JSON.parse(document.getElementById("route-data").textContent);
const BY_NAME = new Map(INDEX.map((r) => [r.name, r]));

function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
async function gunzipJson(b64) {
  const stream = new Blob([b64ToBytes(b64)]).stream()
    .pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).json();
}

// Unpack one route payload into the shape the viewer works with. `d` is either the embedded payload or that payload patched with the
// current route file (LiveCheck.diffRoute), which adds the change marks: `diff` on
// a reaction, `chg` / `was` on a molecule node.
function unpackRoute(name, d) {
  const reactions = d.rx.map((r, i) => ({
    index: i, number: i + 1, reaction_id: r.rid,
    raw_description: r.d, full_reaction_smiles_mapped: r.m, parsed_reaction: r.p || {},
    diff: r.diff || null,
  }));
  const nodes = d.nodes.map((n) => n.t === "m"
    ? { id: n.id, type: "molecule", smiles: n.s, role: n.r, cx: n.cx, cxa: n.cxa, wd: !!n.wd,
        ext_w: n.ew, ext_h: n.eh, ext_w_a: n.ewa, ext_h_a: n.eha, n_atoms: n.na,
        changed: !!n.chg, was: n.was || null }
    : { id: n.id, type: "reaction", index: n.i, label: `${n.i + 1}`,
        raw_description: reactions[n.i].raw_description, invalid: !!n.inv,
        changed: !!reactions[n.i].diff });
  return {
    name,
    background: d.bg || {},
    cv: d.cv || null,       // SAScore along the longest linear sequence (build_static.py)
    reactions,
    graph: { nodes, edges: d.edges.map(([s, t]) => ({ source: s, target: t })), warnings: d.warn || [] },
  };
}

// ---- drawing (RDKit.js) ----------------------------------------------------
// Whether to draw heteroatoms in their element colours (O red, N blue, …). Default
// off = monochrome (all black).
let COLOR = false;
// Whether reaction drawings show atom-map numbers. Default off: they often make the
// drawing unreadable.
let SHOW_MAPS = false;
// Drop atom-map numbers ([CH3:4] -> [CH3]) so the reaction is drawn without them.
const stripMaps = (smiles) => smiles.replace(/\[([^\]]*?):\d+\]/g, "[$1]");
// Whether to condense groups (Bn, Boc, OMe, PEG6, …). Default on.
let ABBREV = true;
const BW_PALETTE = (() => { const p = {}; for (let z = -1; z <= 118; z++) p[z] = [0, 0, 0]; return p; })();
const paletteOpts = () => (COLOR ? {} : { atomColourPalette: BW_PALETTE });

const IMG_CACHE = new Map();
const IMG_CACHE_MAX = 6000;
function cached(key, make) {
  let v = IMG_CACHE.get(key);
  if (v === undefined) {
    v = make();
    if (IMG_CACHE.size >= IMG_CACHE_MAX) IMG_CACHE.delete(IMG_CACHE.keys().next().value);
    IMG_CACHE.set(key, v);
  }
  return v;
}
const svgUri = (svg) => "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
function placeholderSvg(text, w, h) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">` +
    `<rect x="0.5" y="0.5" width="${w - 1}" height="${h - 1}" fill="#f5f5f5" stroke="#c8c8c8"/>` +
    `<text x="50%" y="50%" fill="#969696" font-family="sans-serif" font-size="${Math.max(10, Math.round(h / 12))}" ` +
    `text-anchor="middle" dominant-baseline="middle">${escapeHtml(text)}</text></svg>`;
}

// Draw a molecule from SMILES / CXSMILES. A CXSMILES from the build carries the
// tree-aligned 2D coords, so RDKit.js keeps that orientation. `wedged` = honour
// the stored wU/wD wedges instead of recomputing them.
function molSvg(input, w, h, wedged = false, maxBond = 0) {
  return cached(`m|${input}|${w}|${h}|${COLOR}|${wedged}|${maxBond}`, () => {
    let mol = null;
    try {
      mol = input ? RDKit.get_mol(input) : null;
      if (!mol || !mol.is_valid()) return placeholderSvg("invalid", w, h);
      const details = Object.assign({ width: w, height: h, padding: 0.08 }, paletteOpts(),
        wedged ? { useMolBlockWedging: true, addChiralHs: false } : {},
        maxBond ? { fixedBondLength: maxBond } : {});
      return mol.get_svg_with_highlights(JSON.stringify(details));
    } catch (e) {
      return placeholderSvg("invalid", w, h);
    } finally {
      if (mol) mol.delete();
    }
  });
}
const molUrl = (smiles, size = 200) => svgUri(molSvg(smiles, size, size));
// Reagent thumbnails: RDKit zooms each drawing to fill its box, which gives small
// species (NaOH, H2O, ions) huge labels. Capping the bond length keeps them at the
// same scale as larger reagents; bigger molecules still shrink to fit.
const reagentUrl = (smiles, size = 80) => svgUri(molSvg(smiles, size, size, false, Math.round(size * 25 / 80)));
// Tree node depiction: aligned coords; the abbreviated form when ABBREV is on.
const nodeUrl = (d, w, h, abbr) =>
  d.cx ? svgUri(molSvg(abbr && d.cxa ? d.cxa : d.cx, w, h, d.wd)) : molUrl(d.smiles, Math.max(w, h));

const useAbbr = (d) => ABBREV && d && d.ext_w_a != null && !!d.cxa;
const NODE_SS = Math.min(3, Math.max(2, Math.ceil(window.devicePixelRatio || 1) + 1));
const NODE_MIN_PX = 260;
const nodeBgImg = (d) => {
  let w = (d.w || 76) * NODE_SS, h = (d.h || 76) * NODE_SS;
  const long = Math.max(w, h);
  if (long < NODE_MIN_PX) { const k = NODE_MIN_PX / long; w *= k; h *= k; }
  return nodeUrl(d, Math.round(w), Math.round(h), useAbbr(d));
};
// Big, aspect-preserving render for the panel / lightbox. Always the FULL structure.
const mbBig = (d, long) => {
  const ew = d.ext_w || 1, eh = d.ext_h || 1;
  if (d.cx && (ew || eh)) {
    return ew >= eh ? nodeUrl(d, long, Math.max(60, Math.round(long * eh / ew)), false)
                    : nodeUrl(d, Math.max(60, Math.round(long * ew / eh)), long, false);
  }
  return nodeUrl(d, long, long, false);
};

// Reaction image drawn from the reaction SMILES (used when the tree layouts can't be).
function rxnSvg(smiles, sub = 320) {
  if (smiles && !SHOW_MAPS) smiles = stripMaps(smiles);
  const subW = Math.round(sub * 1.25), subH = sub;
  if (!smiles) return placeholderSvg("no reaction smiles", 700, 240);
  return cached(`r|${smiles}|${sub}|${COLOR}`, () => {
    let rxn = null;
    try {
      const parts = smiles.split(/(?<!-)>(?!-)/);
      const nMain = parts.length >= 2
        ? parts[0].split(".").filter(Boolean).length + parts[parts.length - 1].split(".").filter(Boolean).length
        : 2;
      const width = Math.min(4000, Math.max(2, nMain + 1) * subW);
      rxn = RDKit.get_rxn(smiles, JSON.stringify({ useSmiles: true }));
      if (!rxn) throw new Error("could not parse reaction");
      // CoordGen leaves some reaction drawings blank; use the classic depictor here.
      if (RDKit.prefer_coordgen) RDKit.prefer_coordgen(false);
      try {
        // fixedBondLength caps the zoom: without it a bond-less agent such as NaOH
        // ([Na+].[OH-]) is blown up to giant labels over the arrow.
        return rxn.get_svg_with_highlights(JSON.stringify(Object.assign(
          { width, height: subH, fixedBondLength: Math.round(sub / 6) }, paletteOpts())));
      } finally {
        if (RDKit.prefer_coordgen) RDKit.prefer_coordgen(true);
      }
    } catch (e) {
      return placeholderSvg("invalid reaction", Math.max(subW, 400), subH);
    } finally {
      if (rxn) rxn.delete();
    }
  });
}

// The reaction drawn from the tree's own molecule layouts, so the panel shows each
// reactant in the same orientation as in the tree (aligned to the product by the
// build). `rxnNodeId` picks which copy of the reaction node, if the tree repeats it.
// Returns null when that is not possible, and the caller draws it from SMILES:
// the step changed since the build, or a reactant/product of this step is not a
// drawn molecule in the tree (e.g. a salt formation).
function alignedRxnParts(index, rxnNodeId) {
  if (!CURRENT) return null;
  const rxn = CURRENT.reactions[index];
  if (!rxn || rxn.diff) return null;
  const { nodes, edges } = CURRENT.graph;
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const node = (rxnNodeId && byId.get(rxnNodeId)) ||
    nodes.find((n) => n.type === "reaction" && n.index === index);
  if (!node || node.invalid) return null;
  const reactants = new Map(), products = [];
  for (const e of edges) {
    if (e.target === node.id) {
      const m = byId.get(e.source);
      if (!m || m.type !== "molecule" || !m.cx) return null;
      const r = reactants.get(m.id);
      if (r) r.mult++; else reactants.set(m.id, { mol: m, mult: 1 });
    } else if (e.source === node.id) {
      const m = byId.get(e.target);
      if (!m || m.type !== "molecule" || !m.cx) return null;
      products.push({ mol: m, mult: 1 });
    }
  }
  if (!reactants.size || !products.length) return null;
  // The mapped SMILES: reactants > agents > products. Its fragments give the atom
  // maps for the tree molecules and the agents (reagents/solvents) for over the arrow.
  const sides = (rxn.full_reaction_smiles_mapped || "").split(/(?<!-)>(?!-)/);
  const frags = (i) => (sides.length === 3 ? sides[i].split(".").filter(Boolean) : []);
  return { key: `${CURRENT.name}|${node.id}`, reactants: [...reactants.values()], products,
           reactantFrags: frags(0), agents: [...new Set(frags(1).map(stripMaps))], productFrags: frags(2) };
}

// Atom-map number of each atom of a mapped SMILES, in atom order (0 = unmapped).
const SMILES_ATOM_RE = /\[[^\]]*\]|Br|Cl|[BCNOPSFI]|[bcnops]|\*/g;
const smilesMapNums = (smi) => (smi.match(SMILES_ATOM_RE) || []).map((t) => {
  const m = t.match(/^\[.*:(\d+)\]$/);
  return m ? Number(m[1]) : 0;
});
const CANON = new Map();
function canonSmiles(smi) {
  if (!CANON.has(smi)) {
    let mol = null, out = null;
    try { mol = RDKit.get_mol(smi); if (mol && mol.is_valid()) out = mol.get_smiles(); }
    catch (e) {} finally { if (mol) mol.delete(); }
    CANON.set(smi, out);
  }
  return CANON.get(smi);
}
const noStereo = (smi) => smi.replace(/[@\/\\]/g, "");

// A tree molecule's CXSMILES (aligned coords) with the atom-map numbers of its
// fragment in the mapped reaction added, so RDKit draws them on the aligned layout.
// Unchanged if the fragment cannot be found or matched.
function cxWithMaps(mol, frags) {
  const want = canonSmiles(mol.smiles), wantFlat = canonSmiles(noStereo(mol.smiles));
  const frag = frags.find((f) => canonSmiles(stripMaps(f)) === want) ||
    frags.find((f) => canonSmiles(noStereo(stripMaps(f))) === wantFlat);
  if (!frag) return mol.cx;
  let nodeMol = null, query = null;
  try {
    nodeMol = RDKit.get_mol(mol.cx);
    query = RDKit.get_mol(stripMaps(frag));
    if (!nodeMol || !query) return mol.cx;
    const hit = JSON.parse(nodeMol.get_substruct_match(query) || "{}").atoms;
    const nums = smilesMapNums(frag);
    if (!hit || hit.length !== nums.length) return mol.cx;
    const props = hit.map((atom, i) => (nums[i] ? `${atom}.molAtomMapNumber.${nums[i]}` : null))
      .filter(Boolean).join(":");
    if (!props) return mol.cx;
    const cx = mol.cx.trim();
    return /\|$/.test(cx) ? cx.replace(/\|$/, `,atomProp:${props}|`) : `${cx} |atomProp:${props}|`;
  } catch (e) {
    return mol.cx;
  } finally {
    if (nodeMol) nodeMol.delete();
    if (query) query.delete();
  }
}

// An agent (reagent, catalyst, solvent) in its own clean 2D layout: a molblock plus
// its extent in coordinate units, for sizing. Null if it cannot be parsed.
function agentLayout(smi) {
  let mol = null;
  try {
    mol = RDKit.get_mol(smi);
    if (!mol || !mol.is_valid()) return null;
    const mb = mol.get_molblock();
    const lines = mb.split("\n"), n = parseInt(lines[3].slice(0, 3), 10);
    const xs = [], ys = [];
    for (let i = 0; i < n; i++) {
      xs.push(parseFloat(lines[4 + i].slice(0, 10)));
      ys.push(parseFloat(lines[4 + i].slice(10, 20)));
    }
    if (!n || xs.some(isNaN)) return null;
    return { mb, ew: Math.max(...xs) - Math.min(...xs), eh: Math.max(...ys) - Math.min(...ys) };
  } catch (e) {
    return null;
  } finally {
    if (mol) mol.delete();
  }
}

// One SVG: reactants + … —(agents)→ products. Every reactant/product is drawn from
// the tree's aligned layout at the same bond scale; agents are smaller, over the
// arrow. With SHOW_MAPS the atom-map numbers are drawn on the same layouts.
function alignedRxnSvg(parts, sub) {
  return cached(`a|${parts.key}|${sub}|${COLOR}|${SHOW_MAPS}`, () => {
    const k = Math.max(12, Math.round(sub / 13));   // px per coordinate unit (bond = 1.5)
    const ka = Math.max(8, Math.round(k * 0.7));    // agents: smaller
    const GAP = Math.round(k * 0.6), PLUS = Math.round(k * 1.4), ARROW = Math.round(k * 3.2);
    const font = Math.round(k * 0.9);
    // Box big enough that RDKit's fit-to-box never zooms past the scale (fixedBondLength
    // caps it), so all molecules share one scale.
    const box = (ew, eh, s) => ({ w: Math.round((ew + 1.6) * s / 0.84), h: Math.round((eh + 1.6) * s / 0.84) });
    const strip = (svg) => svg.replace(/^<\?xml[^>]*>\s*/, "");
    const draw = (frags) => ({ mol, mult }) => {
      const { w, h } = box(mol.ext_w || 1, mol.ext_h || 1, k);
      const cx = SHOW_MAPS ? cxWithMaps(mol, frags) : mol.cx;
      return { svg: strip(molSvg(cx, w, h, mol.wd, k)), w, h, mult };
    };
    const left = parts.reactants.map(draw(parts.reactantFrags));
    const right = parts.products.map(draw(parts.productFrags));
    const agents = parts.agents.map(agentLayout).filter(Boolean).map((a) => {
      const { w, h } = box(a.ew, a.eh, ka);
      return { svg: strip(molSvg(a.mb, w, h, false, ka)), w, h };
    });

    const agentsW = agents.reduce((t, a) => t + a.w, 0);
    const agentsH = Math.max(0, ...agents.map((a) => a.h));
    const arrowLen = Math.max(ARROW, agentsW + 2 * GAP);
    const mainH = Math.max(...left.map((d) => d.h), ...right.map((d) => d.h));
    const top = Math.max(0, agentsH - mainH / 2 + 4);   // room for agents over the arrow
    const H = Math.round(top + mainH), y = top + mainH / 2;
    let x = 0, body = "";
    const text = (t, cx) =>
      `<text x="${cx}" y="${y}" font-family="sans-serif" font-size="${font}" fill="#333" ` +
      `text-anchor="middle" dominant-baseline="central">${t}</text>`;
    const place = (list) => list.forEach((d, i) => {
      if (i) { body += text("+", x + PLUS / 2); x += PLUS; }
      if (d.mult > 1) { body += text(`${d.mult}`, x + font * 0.6); x += Math.round(font * 1.2); }
      body += `<g transform="translate(${x},${Math.round(y - d.h / 2)})">${d.svg}</g>`;
      x += d.w;
    });
    place(left);
    x += GAP;
    let ax = x + (arrowLen - agentsW) / 2;
    for (const a of agents) {
      body += `<g transform="translate(${Math.round(ax)},${Math.round(y - 4 - a.h)})">${a.svg}</g>`;
      ax += a.w;
    }
    const ah = Math.round(k * 0.35);
    body += `<line x1="${x}" y1="${y}" x2="${x + arrowLen}" y2="${y}" stroke="#333" stroke-width="${Math.max(1.5, k / 12)}"/>` +
      `<polygon points="${x + arrowLen},${y} ${x + arrowLen - ah * 1.6},${y - ah} ${x + arrowLen - ah * 1.6},${y + ah}" fill="#333"/>`;
    x += arrowLen + GAP;
    place(right);
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${x}" height="${H}" viewBox="0 0 ${x} ${H}">` +
      `<rect width="100%" height="100%" fill="#fff"/>${body}</svg>`;
  });
}

// The panel's reaction image: aligned from the tree where possible, else from SMILES.
function reactionSvg(rxn, rxnNodeId, sub) {
  const parts = alignedRxnParts(rxn.index, rxnNodeId);
  return parts ? alignedRxnSvg(parts, sub) : rxnSvg(rxn.full_reaction_smiles_mapped || "", sub);
}
let _fullSizeBlobUrl = null;
function svgBlobUrl(svg) {
  if (_fullSizeBlobUrl) URL.revokeObjectURL(_fullSizeBlobUrl);
  _fullSizeBlobUrl = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
  return _fullSizeBlobUrl;
}

// ---- route list ------------------------------------------------------------
const routePager = createRoutePager(selectRoute, () => CURRENT && CURRENT.name);
const routeFilters = createRouteFilters((items) => routePager.setItems(items));
function markActive(name) {
  document.querySelectorAll("#route-list li").forEach((li) =>
    li.classList.toggle("active", li.dataset.name === name));
}

// ---- route load ------------------------------------------------------------
let _routeSeq = 0;
async function selectRoute(name) {
  if (!ROUTE_BLOBS[name]) return;
  const seq = ++_routeSeq;
  routePager.reveal(name);
  markActive(name);
  const built = await gunzipJson(ROUTE_BLOBS[name]);
  if (seq !== _routeSeq) return;   // a newer click won
  CURRENT = unpackRoute(name, built);
  try { history.replaceState(null, "", "#route=" + encodeURIComponent(name)); } catch (e) {}
  renderHeader();
  showSingle();
  clearPanel();
  showRouteInfo();
  checkLive(name, built, seq);     // not awaited: the built route is already on screen
}

// ---- live cross-check with the route file ----------------------------------
function liveRouteUrl(name) {
  const base = String(CONFIG.liveDataUrl || CONFIG.dataPath || "").replace(/\/+$/, "");
  return new URL((base ? base + "/" : "") + encodeURIComponent(name) + ".json", document.baseURI).href;
}
// The route's JSON file as it is now, or null if there is none to be had (page
// opened from disk, file missing, offline, not a route file…).
async function fetchLiveRoute(name) {
  if (!CONFIG.liveCheck || !LIVE) return null;
  let url;
  try { url = liveRouteUrl(name); } catch (e) { return null; }
  if (!/^https?:/.test(url)) return null;          // browsers do not let file:// pages read files
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), 8000);
  try {
    const res = await fetch(url, { cache: "no-store", signal: ctl.signal });
    if (!res.ok) return null;
    const data = await res.json();
    return data && Array.isArray(data.synthesis) ? data : null;
  } catch (e) {
    return null;
  } finally {
    clearTimeout(timer);
  }
}
// Compare the built route with its file; if substrates or reagents differ, show
// the file's version with the differences marked. Otherwise leave the page alone.
async function checkLive(name, built, seq) {
  const live = await fetchLiveRoute(name);
  if (!live || seq !== _routeSeq) return;
  let result;
  try {
    result = LIVE.diffRoute(built, live);
  } catch (e) {
    console.warn("Live check failed; showing the built route.", e);
    return;
  }
  if (result.status === "changed") {
    CURRENT = unpackRoute(name, result.route);
    CURRENT.live = result;
    // The complexity chart was computed at build time. diffRoute hands back the very
    // same `nodes` array unless it had to rebuild the tree; a rebuilt tree has new
    // molecules that were never scored, so the chart is dropped.
    CURRENT.cv = result.route.nodes === built.nodes ? built.cv || null : null;
    CURRENT.cvStale = result.route.nodes !== built.nodes;
    renderHeader();
    showSingle();
    clearPanel();           // the tree was redrawn: nothing is selected any more
    showRouteInfo();
  } else {
    CURRENT.live = result;
    renderLiveNote();
  }
}
const plural = (n, one, many) => `${n} ${n === 1 ? one : many || one + "s"}`;
function renderLiveNote() {
  const el = document.getElementById("live-note");
  const live = CURRENT && CURRENT.live;
  el.className = "live-note";
  el.innerHTML = "";
  if (!live) { el.classList.add("hidden"); return; }
  const file = `<code>${escapeHtml(CURRENT.name)}.json</code>`;
  if (live.status === "same") {
    el.classList.add("same");
    el.innerHTML = `✓ Same as the current ${file}`;
    return;
  }
  const built = CONFIG.built ? ` (${escapeHtml(CONFIG.built)})` : "";
  if (live.status === "unverified") {
    el.classList.add("warn");
    el.innerHTML = `<b>${file} has changed since this page was built${built}</b>, but the changes ` +
      `could not be applied here. This is the built version — rebuild the page to see them.`;
    return;
  }
  const s = live.summary, bits = [];
  if (s.changed.length) bits.push(plural(s.changed.length, "reaction") + " changed");
  if (s.added.length) bits.push(plural(s.added.length, "step") + " added");
  if (s.removed.length) bits.push(plural(s.removed.length, "step") + " removed");
  if (s.molecules) bits.push(plural(s.molecules, "molecule") + " redrawn");
  if (s.background.length) bits.push("route info edited");
  const chips = [...s.changed, ...s.added].sort((a, b) => a - b).map((j) =>
    `<button class="live-chip" data-rxn="${j}">Rxn ${j + 1}${s.added.includes(j) ? " · new" : ""}</button>`).join("");
  const gone = s.removed.map((r) =>
    `Rxn ${r.number}${r.d ? " (" + escapeHtml(r.d.length > 60 ? r.d.slice(0, 57) + "…" : r.d) + ")" : ""}`).join("; ");
  el.innerHTML =
    `<b>Showing ${file} as it is now</b> — it has changed since this page was built${built}: ` +
    `${bits.join(" · ")}. Changes are marked in <b class="live-ink">purple</b>.` +
    (chips ? `<div class="live-chips">${chips}</div>` : "") +
    (gone ? `<div class="live-gone">No longer in the file: ${gone}</div>` : "");
  el.querySelectorAll("[data-rxn]").forEach((b) =>
    b.addEventListener("click", () => focusReaction(Number(b.dataset.rxn))));
}
function focusReaction(index) {
  showReaction(index);
  if (!CY) return;
  const nodes = CY.nodes('[type="reaction"]').filter((n) => n.data("index") === index);
  if (!nodes.length) return;
  CY.nodes().unselect();
  nodes.select();
  CY.animate({ center: { eles: nodes } }, { duration: 250 });
}

function renderHeader() {
  renderAbbrevUI();
  renderColorUI();
  const bg = CURRENT.background || {};
  document.getElementById("route-title").textContent = `${CURRENT.name}.json`;
  document.getElementById("route-meta").textContent =
    [bg.name, bg.author, bg.year].filter(Boolean).join(" · ");
  document.getElementById("route-tools").classList.remove("hidden");

  const warnEl = document.getElementById("warnings");
  const warns = (CURRENT.graph && CURRENT.graph.warnings) || [];
  if (warns.length) {
    warnEl.innerHTML = `<b>${warns.length} note(s):</b><ul>` +
      warns.map((w) => `<li>${escapeHtml(w)}</li>`).join("") + "</ul>";
    warnEl.classList.remove("hidden");
  } else warnEl.classList.add("hidden");
  renderLiveNote();
}

// ---- graph rendering -------------------------------------------------------
function graphStyle() {
  return [
    {
      selector: 'node[type="molecule"]',
      style: {
        "shape": "round-rectangle",
        "width": (ele) => ele.data("w") || 76,
        "height": (ele) => ele.data("h") || 76,
        "background-color": "#fff",
        "background-image": (ele) => nodeBgImg(ele.data()),
        "background-fit": "contain", "background-clip": "node",
        "border-width": 3, "border-color": (ele) => ROLE_COLOR[ele.data("role")] || "#999",
      },
    },
    {
      selector: 'node[type="reaction"]',
      style: {
        "shape": "circle",
        "label": (ele) => (ele.data("invalid") ? "⚠ " : "") + ele.data("label"),
        "width": (ele) => Math.max(RXN_NODE_H, 7 * String(ele.data("label")).length + 12),
        "height": RXN_NODE_H,
        "background-color": (ele) => ele.data("invalid") ? "#ffd9d2" : RXN_YELLOW,
        "color": "#222",
        "font-size": 11, "font-weight": 700,
        "text-valign": "center", "text-halign": "center",
        "border-width": (ele) => ele.data("invalid") ? 3 : 2,
        "border-style": (ele) => ele.data("invalid") ? "dashed" : "solid",
        "border-color": (ele) => ele.data("invalid") ? "#e0533d" : RXN_BORDER,
      },
    },
    // Changed in the route file since the build: purple halo; molecules also get a
    // dashed border (their role colour stays) and an "updated" tag.
    {
      selector: 'node[type="molecule"][?changed]',
      style: {
        "border-style": "dashed", "border-width": 4,
        "underlay-color": LIVE_PURPLE, "underlay-opacity": 0.25, "underlay-padding": 8,
        "label": "updated", "color": LIVE_PURPLE, "font-size": 11, "font-weight": 700,
        "text-valign": "top", "text-halign": "center", "text-margin-y": -10,
      },
    },
    {
      selector: 'node[type="reaction"][?changed]',
      style: {
        "background-color": "#ead9f7", "border-color": LIVE_PURPLE, "border-width": 3,
        "border-style": "solid",
        "underlay-color": LIVE_PURPLE, "underlay-opacity": 0.25, "underlay-padding": 6,
      },
    },
    // the step under the cursor in the complexity chart (or its own node, hovered)
    { selector: "node.cv-hl",
      style: { "underlay-color": CV_HL, "underlay-opacity": 0.35, "underlay-padding": 9 } },
    { selector: "node:selected", style: { "border-color": "#111", "border-width": 4 } },
    {
      selector: "edge",
      style: {
        "width": 2, "line-color": "#b9bdc7", "target-arrow-color": "#b9bdc7",
        "target-arrow-shape": "triangle", "curve-style": "bezier", "arrow-scale": 0.9,
      },
    },
    {
      selector: 'edge[kind="rxn2rxn"]',
      style: { "line-color": "#8a8f9c", "target-arrow-color": "#8a8f9c", "width": 3 },
    },
  ];
}

function molNodeDims(n) {
  const SCALE = 16, MIN = 46, DEFAULT = 76;
  const abbr = useAbbr(n);
  const ew = abbr ? n.ext_w_a : n.ext_w, eh = abbr ? n.ext_h_a : n.ext_h;
  if ((ew || eh) && ew != null) {
    return { w: Math.max(MIN, Math.round(ew * SCALE)),
             h: Math.max(MIN, Math.round(eh * SCALE)) };
  }
  const s = n.n_atoms ? Math.max(MIN, Math.round(19 * Math.sqrt(n.n_atoms))) : DEFAULT;
  return { w: s, h: s };
}

// A reaction's number is the tail of its reaction_id ("Name_2004_Smith_3" -> "3").
function rxnNum(index) {
  const rid = (CURRENT.reactions[index] || {}).reaction_id;
  if (rid == null || rid === "") return String(index + 1);
  const m = String(rid).match(/_([^_]+)$/);
  return m ? m[1] : String(rid);
}
const rxnName = (index) => `RXN ${rxnNum(index)}`;

function toElements(graph) {
  const els = [];
  const rxnIds = new Set(graph.nodes.filter(n => n.type === "reaction").map(n => n.id));
  for (const n of graph.nodes) {
    if (n.type === "molecule") {
      const dim = molNodeDims(n);
      els.push({ data: { id: n.id, type: "molecule", smiles: n.smiles, role: n.role,
                         cx: n.cx, cxa: n.cxa, wd: n.wd, w: dim.w, h: dim.h,
                         ext_w: n.ext_w, ext_h: n.ext_h, ext_w_a: n.ext_w_a, ext_h_a: n.ext_h_a,
                         changed: n.changed, was: n.was } });
    } else {
      els.push({ data: {
        id: n.id, type: "reaction", index: n.index,
        label: rxnNum(n.index),
        raw: n.raw_description, invalid: !!n.invalid, changed: n.changed,
      }});
    }
  }
  graph.edges.forEach((e, i) => {
    const kind = (rxnIds.has(e.source) && rxnIds.has(e.target)) ? "rxn2rxn" : "mol";
    els.push({ data: { id: `e${i}__${e.source}__${e.target}`, source: e.source, target: e.target, kind } });
  });
  return els;
}

function bindWheelScroll(container) {
  if (container.dataset.wheelBound) return;
  container.dataset.wheelBound = "1";
  container.addEventListener("wheel", (e) => {
    const cy = container.__cy;
    if (!cy) return;
    e.preventDefault();
    const level = Math.min(Math.max(cy.zoom() * Math.pow(1.01, -e.deltaY),
                                    cy.minZoom()), cy.maxZoom());
    const r = container.getBoundingClientRect();
    cy.zoom({ level, renderedPosition: { x: e.clientX - r.left, y: e.clientY - r.top } });
  }, { passive: false });
}

function spaceComponents(cy) {
  const comps = cy.elements().components();
  if (comps.length <= 1) return;
  const items = comps.map((c) => ({ c, bb: c.boundingBox() })).sort((a, b) => a.bb.x1 - b.bb.x1);
  const maxW = Math.max(...items.map((it) => it.bb.w));
  const gap = Math.min(Math.max(maxW * 0.7, 260), 900);
  let right = null;
  for (const { c, bb } of items) {
    if (right === null) { right = bb.x2; continue; }
    const dx = (right + gap) - bb.x1;
    if (dx) c.nodes().positions((n) => ({ x: n.position("x") + dx, y: n.position("y") }));
    right = bb.x2 + dx;
  }
}

// Initial view: every route opens at the same scale — a reaction node (62 px wide in
// the tree) spans RXN_VIEW_FRACTION of the canvas width — centred on Rxn 1.
const RXN_NODE_W = 62, RXN_NODE_H = 30;
const RXN_VIEW_FRACTION = 0.10;
function setDefaultViewport(cy) {
  const nodes = cy.nodes();
  if (!nodes.length || cy.width() <= 0 || cy.height() <= 0) return;
  const zoom = cy.width() * RXN_VIEW_FRACTION / RXN_NODE_W;
  const bounds = cy.elements().boundingBox();
  const margin = 50;
  // Keep the requested initial scale even on a very small or very large canvas.
  cy.minZoom(Math.min(cy.minZoom(), zoom));
  cy.maxZoom(Math.max(cy.maxZoom(), zoom));
  // Start centred on Rxn 1; a route without it starts at its top-left corner.
  const first = cy.nodes('[type="reaction"]').filter((n) => n.data("index") === 0);
  const at = first.length ? first[0].position() : null;
  cy.viewport({ zoom, pan: at
    ? { x: cy.width() / 2 - at.x * zoom, y: cy.height() / 2 - at.y * zoom }
    : { x: margin - bounds.x1 * zoom, y: margin - bounds.y1 * zoom } });
}

function makeGraph(containerId, graph) {
  const container = document.getElementById(containerId);
  const cy = cytoscape({
    container,
    elements: toElements(graph),
    minZoom: 0.06, maxZoom: 3,
    userZoomingEnabled: false,
    style: graphStyle(),
    layout: { name: "dagre", rankDir: "LR", nodeSep: 24, rankSep: 28, edgeSep: 10, fit: false, padding: 20 },
  });
  spaceComponents(cy);
  setDefaultViewport(cy);
  container.__cy = cy;
  bindWheelScroll(container);
  cy.on("tap", 'node[type="reaction"]', (evt) => showReaction(evt.target.data("index"), evt.target.id()));
  cy.on("tap", 'node[type="molecule"]', (evt) => showMolecule(evt.target.data()));
  cy.on("mouseover", "node", (evt) => {
    const d = evt.target.data();
    if (d.type === "molecule") { hideTooltip(); return; }
    showTooltip((d.changed ? "Changed in the route file — " : "") + (d.raw || d.label),
      evt.renderedPosition, containerId);
  });
  cy.on("mousemove", 'node[type="reaction"]', (evt) => moveTooltip(evt.renderedPosition, containerId));
  cy.on("mouseout", "node", hideTooltip);
  // keep the complexity chart (route info panel) in step with the tree
  cy.on("mouseover", "node", (evt) => {
    if (CV_NODES.has(evt.target.id())) cvHover(evt.target.id());
  });
  cy.on("mouseout", "node", () => cvHover(null));
  cy.on("select unselect", "node", cvSyncSelection);
  return cy;
}

function renderAbbrevUI() {
  const btn = document.getElementById("btn-abbr");
  if (btn) {
    btn.textContent = "Abbrev: " + (ABBREV ? "on" : "off");
    btn.classList.toggle("btn-on", ABBREV);
    btn.setAttribute("aria-pressed", String(ABBREV));
  }
}
function renderColorUI() {
  const btn = document.getElementById("btn-color");
  if (btn) {
    btn.textContent = "Colored atoms: " + (COLOR ? "on" : "off");
    btn.classList.toggle("btn-on", COLOR);
    btn.setAttribute("aria-pressed", String(COLOR));
  }
}
function renderMapUI() {
  const btn = document.getElementById("btn-map");
  if (btn) {
    btn.textContent = "Atom maps: " + (SHOW_MAPS ? "on" : "off");
    btn.classList.toggle("btn-on", SHOW_MAPS);
    btn.setAttribute("aria-pressed", String(SHOW_MAPS));
  }
}
// Re-render whichever panel is open, so its image reflects a toggle.
function refreshPanel() {
  if (!PANEL) return;
  if (PANEL.kind === "reaction") showReaction(PANEL.index, PANEL.nodeId);
  else if (PANEL.kind === "molecule" && LAST_MOL) showMolecule(LAST_MOL);
}
function rerenderGraph() {
  if (!CURRENT || !CY) return;
  CY.batch(() => {
    for (const node of CURRENT.graph.nodes) {
      if (node.type === "molecule") {
        CY.getElementById(node.id).data(molNodeDims(node));
      }
    }
    CY.style(graphStyle());
  });
}

function showSingle() {
  if (CY) CY.destroy();
  CY = makeGraph("cy", CURRENT.graph);
}

// ---- panel -----------------------------------------------------------------
function clearPanel() {
  PANEL = null;
  document.getElementById("panel-empty").classList.remove("hidden");
  document.getElementById("panel-content").classList.add("hidden");
}
function openPanel(html) {
  const el = document.getElementById("panel-content");
  el.innerHTML = html;
  bindCopyBoxes(el);
  document.getElementById("panel-empty").classList.add("hidden");
  el.classList.remove("hidden");
  bindReportBox(el);
  return el;
}

// ---- panel: marks for what changed in the route file -------------------------
const liveBadge = (text) => `<span class="live-badge">${escapeHtml(text)}</span>`;
const liveThumbs = (list, cls) => list.map((s) =>
  `<img class="reagent-thumb ${cls}" src="${reagentUrl(s, 80)}" data-mol="${escapeHtml(s)}" title="${escapeHtml(s)}" alt="${escapeHtml(s)}"/>`).join("");
const liveBefore = (text) =>
  `<div class="live-before"><span>before</span>${escapeHtml(text) || "∅"}</div>`;
// "now" / "before" structure rows for one reaction's substrates and reagents.
function liveDiffBox(diff) {
  let rows = "";
  const part = (label, d) => {
    if (!d || (!d.added.length && !d.removed.length)) return;
    rows += `<div class="live-row"><span class="fld-label">${label}</span>` +
      (d.added.length ? `<div class="thumb-row"><span class="live-tag">now</span>${liveThumbs(d.added, "live-add")}</div>` : "") +
      (d.removed.length ? `<div class="thumb-row"><span class="live-tag was">before</span>${liveThumbs(d.removed, "live-del")}</div>` : "") +
      `</div>`;
  };
  const added = diff.status === "added";
  part(added ? "Reactants new to the route" : "Reactants", diff.reactants);
  part(added ? "Products new to the route" : "Products", diff.products);
  part("Reagents written in the reaction SMILES", diff.agents);
  if (diff.kinds.mapping) {
    rows += `<div class="live-row">The reaction SMILES was rewritten with the same molecules ` +
      `(for example, a different atom mapping).</div>`;
  }
  const fields = Object.keys(diff.fields).filter((k) => k !== "m").length;
  return `<div class="live-box"><div class="live-title">` +
    (added ? "This step is in the route file but not in the built page."
           : "The route file has changed for this step since the page was built.") +
    `</div>` + rows +
    (fields ? `<div class="live-row muted">${rows ? "Also edited" : "Edited"}: the fields marked below.</div>` : "") +
    `</div>`;
}

function showMolecule(d) {
  LAST_MOL = d;
  PANEL = { kind: "molecule" };
  const role = d.role;
  const roleLabel = { target: "Target", intermediate: "Intermediate", start: "Start" }[role] || role;
  const was = d.changed && d.was && d.was.length
    ? `<div class="section-title">Before, in the built page</div>` +
      `<div class="thumb-row">${liveThumbs(d.was, "live-del")}</div>`
    : "";
  openPanel(
    `<div class="panel-step" style="color:${ROLE_COLOR[role] || "#666"}">${escapeHtml(roleLabel)}` +
      (d.changed ? liveBadge("updated from the route file") : "") + `</div>` +
    `<img class="panel-img molecule-img zoomable" src="${mbBig(d, 520)}" alt="molecule" title="click to enlarge" />` +
    (d.changed ? `<div class="live-box"><div class="live-title">This molecule is new at this position ` +
      `since the page was built.</div></div>` : "") +
    `<div class="section-title">SMILES</div>` +
    `<div class="smiles-box" style="min-height:auto">${escapeHtml(d.smiles || "∅")}</div>` + was +
    reportBox({ kind: "molecule", role, smiles: d.smiles }, "Report a mistake in this molecule"));
  document.querySelector("#panel-content .panel-img")
    .addEventListener("click", () => openLightbox(mbBig(d, 2000)));
}

function showReaction(index, rxnNodeId = null) {
  const rxn = CURRENT.reactions.find((r) => r.index === index);
  if (!rxn) return;
  PANEL = { kind: "reaction", index, nodeId: rxnNodeId };
  const mapped = rxn.full_reaction_smiles_mapped || "";
  const parsed = rxn.parsed_reaction || {};
  const diff = rxn.diff;                       // set if the route file changed this step
  const edited = (diff && diff.fields) || {};
  const joined = (v) => (Array.isArray(v) ? v.join("; ") : String(v == null ? "" : v));

  // Reagents/conditions: names with structure thumbnails and SMILES values.
  // Empty categories are skipped (unless the route file just emptied them).
  let reagentSection = "";
  for (const [key, label] of REAGENT_FIELDS) {
    const names = parsed[key] || [];
    const defKey = DEF_FOR[key];
    const list = defKey ? (parsed[defKey] || []) : [];
    const nameEdit = edited[key], defEdit = defKey && edited[defKey];
    if (!names.length && !list.length && !nameEdit && !defEdit) continue;
    let card = `<div class="reagent-card${nameEdit || defEdit ? " live-changed" : ""}">` +
      `<div class="fld"><span class="fld-label">${label}${nameEdit ? liveBadge("edited") : ""}</span>` +
      `<div class="smiles-box" style="min-height:auto">${escapeHtml(names.join("; ")) || "∅"}</div>` +
      (nameEdit ? liveBefore(joined(nameEdit.old)) : "") + `</div>`;
    if (defKey) {
      const isNew = (s) => defEdit && defEdit.added.includes(LIVE.keyOf(String(s)));
      const thumbs = list.map((s) =>
        `<img class="reagent-thumb${isNew(s) ? " live-add" : ""}" src="${reagentUrl(s, 80)}" data-mol="${escapeHtml(s)}" title="${escapeHtml(s)}" alt="${escapeHtml(s)}"/>`).join("");
      card += `<div class="fld fld-smiles">` +
        (defEdit ? liveBadge("edited") : "") +
        (thumbs ? `<div class="thumb-row">${thumbs}</div>` : "") +
        `<div class="smiles-box" style="min-height:auto">${escapeHtml(list.join("; ")) || "∅"}</div>` +
        (defEdit && defEdit.removed.length
          ? `<div class="thumb-row" style="margin-top:4px"><span class="live-tag was">before</span>${liveThumbs(defEdit.removed, "live-del")}</div>` : "") +
        `</div>`;
    }
    reagentSection += card + `</div>`;
  }
  if (!reagentSection) reagentSection = `<div class="muted">No reagent information.</div>`;

  const roBox = (label, v) =>
    `<div class="fld"><span class="fld-label">${label}</span>` +
    `<div class="smiles-box">${escapeHtml(v || "∅")}</div></div>`;

  openPanel(
    `<div class="panel-step reaction-heading" style="color:${RXN_BORDER}">RXN ${escapeHtml(rxn.reaction_id ?? rxn.number)}</div>` +
    `<img class="panel-img zoomable" id="rxn-img" src="${svgUri(reactionSvg(rxn, rxnNodeId, 360))}" alt="reaction" title="click to enlarge" />` +
    `<div class="img-actions">` +
      `<a id="rxn-full" href="#" target="_blank" rel="noopener">Open full size ↗</a>` +
    `</div>` +
    (diff ? liveDiffBox(diff) : "") +
    `<div class="section-title">Description${edited.d ? liveBadge("edited") : ""}</div>` +
    `<div class="smiles-box" style="min-height:auto">${escapeHtml(rxn.raw_description || "∅")}</div>` +
    (edited.d ? liveBefore(edited.d.old) : "") +
    `<div class="section-title">Reagents &amp; conditions</div>` + reagentSection +
    `<div class="section-title">Reaction SMILES</div>` +
    roBox("full_reaction_smiles_mapped" + (edited.m ? liveBadge("edited") : ""), mapped) +
    (edited.m ? liveBefore(edited.m.old) : "") +
    reportBox({ kind: "reaction", rxn }, "Report a mistake in this reaction"));

  document.getElementById("rxn-img").addEventListener("click", () => openLightbox(svgUri(reactionSvg(rxn, rxnNodeId, 820))));
  document.getElementById("rxn-full").addEventListener("click", (e) => {
    e.currentTarget.href = svgBlobUrl(reactionSvg(rxn, rxnNodeId, 760));
  });
}

// ---- complexity evolution chart (route info) --------------------------------
// Complexity of each molecule along the route's longest linear sequence: a
// molecule is an open circle in its role colour, a reaction is the line into its product. Every circle and every
// line stands for one node of the tree; hovering or clicking either one is mirrored
// on the other (the tree's selection is the single source of truth for "selected").
const CV_HL = "#cf6a1a";
const CV_METRICS = [["sascore", "SAScore"], ["boettcher", "Böttcher"], ["sps", "SPScore"], ["npscore", "NPScore"]];
let CV_METRIC = "sascore";      // the dropdown's choice; kept from route to route
let CV_STEPS = [];              // [{rxn, mol, y}] for the chart on screen
let CV_NODES = new Map();       // tree node id -> its step in CV_STEPS

const cvMetricsHere = () => CV_METRICS.filter(([m]) => CURRENT.cv && CURRENT.cv.y && CURRENT.cv.y[m]);
const cvLabel = () => (CV_METRICS.find(([m]) => m === CV_METRIC) || ["", ""])[1];
const cvNum = (v) => (v == null ? "n/a" : v.toFixed(Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 10 ? 1 : 2));

function complexitySection() {
  CV_STEPS = [];
  CV_NODES = new Map();
  const here = cvMetricsHere();
  if (!here.length) {
    return `<div class="section-title">Complexity Evolution</div><div class="muted">${CURRENT.cvStale
      ? "Not available: the route file changed the tree since this page was built."
      : "Not available for this route."}</div>`;
  }
  if (!here.some(([m]) => m === CV_METRIC)) CV_METRIC = here[0][0];
  return `<div class="section-title">Complexity Evolution</div>` +
    `<select id="cv-metric" title="Complexity metric">` +
    here.map(([m, label]) => `<option value="${m}"${m === CV_METRIC ? " selected" : ""}>${label}</option>`).join("") +
    `</select><div id="cv-box"></div>`
}

function renderComplexityChart() {
  const box = document.getElementById("cv-box");
  if (!box) return;
  CV_STEPS = [];
  CV_NODES = new Map();
  const byId = new Map(CURRENT.graph.nodes.map((n) => [n.id, n]));
  const yy = CURRENT.cv.y[CV_METRIC];
  const steps = CURRENT.cv.p.map(([rid, mid], i) =>
    ({ rxn: rid == null ? null : byId.get(rid), mol: byId.get(mid), y: yy[i] }));
  const ys = steps.filter((s) => s.y != null).map((s) => s.y);
  if (steps.some((s) => !s.mol) || ys.length < 2) {
    box.innerHTML = `<div class="muted">${cvLabel()} is not available for this route.</div>`;
    return;
  }
  CV_STEPS = steps;
  steps.forEach((s) => {
    CV_NODES.set(s.mol.id, s);
    if (s.rxn) CV_NODES.set(s.rxn.id, s);
  });

  const W = 420, H = 260, L = 50, R = 14, T = 14, B = 40, pw = W - L - R, ph = H - T - B;
  const n = steps.length;
  // "nice" y ticks: 1, 2, 2.5 or 5 x 10^k, at most ~6 of them
  const span = Math.max(...ys) - Math.min(...ys) || Math.abs(ys[0]) || 1;
  const mag = Math.pow(10, Math.floor(Math.log10(span / 5)));
  const tick = mag * ([1, 2, 2.5, 5].find((t) => span / (t * mag) <= 5) || 10);
  const lo = Math.floor(Math.min(...ys) / tick + 1e-9) * tick;
  const hi = Math.max(Math.ceil(Math.max(...ys) / tick - 1e-9) * tick, lo + tick);
  const X = (i) => L + (n === 1 ? 0.5 : (i + 0.5) / n) * pw;
  const Y = (v) => T + (hi - v) / (hi - lo) * ph;
  const f = (v) => v.toFixed(1);

  let grid = "", segs = "", pts = "", hits = "";
  for (let k = 0; lo + k * tick <= hi + tick * 1e-6; k++) {
    const v = lo + k * tick, y = f(Y(v));
    grid += `<line class="cv-grid" x1="${L}" x2="${W - R}" y1="${y}" y2="${y}"/>` +
      `<text x="${L - 6}" y="${y}" text-anchor="end" dominant-baseline="middle">${+v.toPrecision(6)}</text>`;
  }
  const every = Math.ceil(n / 10);
  let ptHits = "";
  steps.forEach((s, i) => {
    if (i % every === 0) grid += `<text x="${f(X(i))}" y="${H - B + 14}" text-anchor="middle">${i}</text>`;
    const prev = steps[i - 1];
    if (s.rxn && prev && prev.y != null && s.y != null) {
      const at = `data-node="${escapeHtml(s.rxn.id)}" x1="${f(X(i - 1))}" y1="${f(Y(prev.y))}" x2="${f(X(i))}" y2="${f(Y(s.y))}"`;
      segs += `<line class="cv-seg" ${at}/>`;
      hits += `<line class="cv-hit" ${at}/>`;
    }
    if (s.y != null) {
      const at = `data-node="${escapeHtml(s.mol.id)}" cx="${f(X(i))}" cy="${f(Y(s.y))}"`;
      pts += `<circle class="cv-pt" ${at} r="4" stroke="${ROLE_COLOR[s.mol.role] || "#999"}"/>`;
      ptHits += `<circle class="cv-hit" ${at} r="${Math.min(9, Math.max(5, pw / n / 2.4)).toFixed(1)}"/>`;
    }
  });
  // paint order: lines, circles, then the (invisible) hit areas — circles on top
  box.innerHTML =
    `<svg id="cv-chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${cvLabel()} along the longest linear sequence">` +
    `<line class="cv-axis" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/>` +
    `<line class="cv-axis" x1="${L}" x2="${L}" y1="${T}" y2="${H - B}"/>` + grid +
    `<text x="${L + pw / 2}" y="${H - 6}" text-anchor="middle">Step (longest linear sequence)</text>` +
    segs + pts + hits + ptHits + `</svg>`;

  const svg = box.firstChild;
  const nodeAt = (e) => (e.target.dataset && e.target.dataset.node) || null;
  svg.addEventListener("mousemove", (e) => {
    const id = nodeAt(e);
    if (svg.dataset.over !== (id || "")) { svg.dataset.over = id || ""; cvHover(id); }
    if (!id) { hideTooltip(); return; }
    tip.textContent = cvTipText(id);
    tip.classList.remove("img-tip", "hidden");
    tip.style.left = e.clientX + 14 + "px";
    tip.style.top = e.clientY + 14 + "px";
  });
  svg.addEventListener("mouseleave", () => { svg.dataset.over = ""; cvHover(null); hideTooltip(); });
  svg.addEventListener("click", (e) => { const id = nodeAt(e); if (id) cvSelect(id); });
  cvSyncSelection();
}

function cvMark(cls, ids) {
  const svg = document.getElementById("cv-chart");
  if (!svg) return;
  svg.querySelectorAll("[data-node]").forEach((el) => el.classList.toggle(cls, ids.has(el.dataset.node)));
}
// Highlight one molecule / reaction in the chart and in the tree (null clears).
function cvHover(id) {
  cvMark("on", new Set(id == null ? [] : [id]));
  if (!CY) return;
  CY.nodes(".cv-hl").removeClass("cv-hl");
  if (id != null && document.getElementById("cv-chart")) CY.getElementById(id).addClass("cv-hl");
}
function cvSyncSelection() {
  cvMark("sel", new Set(CY ? CY.nodes(":selected").map((n) => n.id()) : []));
}
// Click in the chart = click on that node in the tree: select it, pan to it, and
// open it in the bottom panel.
function cvSelect(id) {
  if (!CY) return;
  const ele = CY.getElementById(id);
  if (!ele.length) return;
  CY.nodes().unselect();
  ele.select();
  CY.animate({ center: { eles: ele } }, { duration: 250 });
  if (ele.data("type") === "reaction") showReaction(ele.data("index"), ele.id());
  else showMolecule(ele.data());
}
function cvTipText(id) {
  const s = CV_NODES.get(id), i = CV_STEPS.indexOf(s), prev = CV_STEPS[i - 1];
  if (s.mol.id === id) {
    const what = { start: "Starting material", target: "Target" }[s.mol.role] || "Intermediate";
    return `${what} (step ${i}) · ${cvLabel()} ${cvNum(s.y)}`;
  }
  const d = prev && prev.y != null && s.y != null ? s.y - prev.y : null;
  return `${rxnName(s.rxn.index)} (step ${i}) · ${cvLabel()} ${cvNum(prev && prev.y)} → ${cvNum(s.y)}` +
    (d == null ? "" : ` (${d >= 0 ? "+" : "−"}${cvNum(Math.abs(d))})`);
}

// ---- route info (right-hand panel; the "Route info" button shows / hides it) ----
function showRouteInfo() {
  if (!CURRENT) return;
  const bg = CURRENT.background || {};
  const targets = ((CURRENT.graph && CURRENT.graph.nodes) || [])
    .filter((n) => n.type === "molecule" && n.role === "target");
  const targetImages = targets.length
    ? `<div class="route-targets">` + targets.map((n, i) =>
      `<img class="panel-img zoomable" data-target-index="${i}" src="${mbBig(n, 520)}" ` +
      `alt="Target molecule ${i + 1} for ${escapeHtml(bg.name || CURRENT.name)}" title="click to enlarge" />`).join("") + `</div>`
    : `<div class="muted">No target molecule available for this route.</div>`;
  const bgEdited = (CURRENT.live && CURRENT.live.summary && CURRENT.live.summary.background) || [];
  let rows = "";
  for (const [key, label] of BG_FIELDS) {
    const v = bg[key] == null ? "" : String(bg[key]);
    rows += `<div class="fld"><span class="fld-label">${label}` +
      `${bgEdited.includes(key) ? liveBadge("edited in the route file") : ""}</span>` +
      `<div class="smiles-box" style="min-height:auto">${escapeHtml(v || "∅")}</div></div>`;
  }
  const el = document.getElementById("info-content");
  el.innerHTML =
    `<div class="panel-step">Route Info</div>` +
    `<div class="section-title">Target${targets.length > 1 ? "s" : ""}</div>` + targetImages +
    complexitySection() + `<div class="section-title">Details</div>` + rows +
    reportBox({ kind: "route" }, "Report a mistake in this route");
  bindCopyBoxes(el);
  bindReportBox(el);
  el.querySelectorAll("[data-target-index]").forEach((img) => {
    const target = targets[Number(img.dataset.targetIndex)];
    img.addEventListener("click", () => openLightbox(mbBig(target, 2000)));
  });
  const sel = document.getElementById("cv-metric");
  if (sel) sel.addEventListener("change", () => { CV_METRIC = sel.value; renderComplexityChart(); });
  renderComplexityChart();
}
function setInfoOpen(open) {
  document.getElementById("app").classList.toggle("info-closed", !open);
  document.getElementById("btn-info").classList.toggle("btn-on", open);
  try { localStorage.setItem("dash-info", open ? "1" : "0"); } catch (e) {}
  resizeGraphs();
}

function showReport() {
  if (!CURRENT) return;
  PANEL = { kind: "report" };
  openPanel(
    `<div class="panel-step">Report a mistake — ${escapeHtml(CURRENT.name)}.json</div>` +
    `<div class="panel-desc">Spotted a wrong structure, reagent, condition, or reference? ` +
    `Tip: click the reaction or molecule first, then use the report box in its panel ` +
    `so the details are filled in for you.</div>` +
    reportBox({ kind: "route" }, null));
}

// ---- reporting (e-mail / GitHub issue / pull request) ----------------------
// Every link is pre-filled with the route, file, and (if any) reaction/molecule
// details plus the reporter's note. GitHub's file editor turns an edit into a PR
// (forking first for people without write access).
const REPORTS = new Map();
let _reportSeq = 0;

const routeFilePath = () => `${CONFIG.dataPath}/${CURRENT.name}.json`;
const ghFileUrl = (mode) =>
  `https://github.com/${CONFIG.repo}/${mode}/${encodeURIComponent(CONFIG.branch)}/` +
  routeFilePath().split("/").map(encodeURIComponent).join("/");

function reportText(scope, note) {
  const bg = CURRENT.background || {};
  const num = (BY_NAME.get(CURRENT.name) || {}).number;
  let subject = `[CbyD] Mistake in ${CURRENT.name}`;
  const lines = [
    "**What is wrong / suggested fix:**",
    note.trim() || "_(please describe the problem and, if you can, the correct value)_",
    "",
    "---",
    `- Route: \`${CURRENT.name}.json\`` + (num ? ` (#${num})` : ""),
    `- File: \`${routeFilePath()}\``,
  ];
  if (bg.name) lines.push(`- Compound: ${bg.name}`);
  const ref = [bg.author, bg.year, bg.citation, bg.doi ? `doi:${bg.doi}` : null].filter(Boolean).join(", ");
  if (ref) lines.push(`- Reference: ${ref}`);
  if (scope.kind === "reaction") {
    const r = scope.rxn;
    subject += ` · Rxn ${r.number}`;
    lines.push(`- Reaction: Rxn ${r.number} (reaction_id ${r.reaction_id}, index ${r.index})`);
    if (r.raw_description) lines.push(`- Description: ${r.raw_description}`);
    if (r.full_reaction_smiles_mapped) lines.push(`- Mapped SMILES: \`${r.full_reaction_smiles_mapped}\``);
  } else if (scope.kind === "molecule") {
    subject += ` · ${scope.role} molecule`;
    lines.push(`- Molecule (${scope.role}): \`${scope.smiles}\``);
  }
  if (CONFIG.built) lines.push(`- Viewer build: ${CONFIG.built}`);
  return { subject, body: lines.join("\n") };
}

function reportLinks(scope, note) {
  const { subject, body } = reportText(scope, note);
  const enc = encodeURIComponent;
  const links = {};
  if (CONFIG.email) {
    links.email = `mailto:${CONFIG.email}?subject=${enc(subject)}&body=${enc(body)}`;
  }
  if (CONFIG.repo) {
    // Browsers / GitHub cap URL length; trim the body rather than fail.
    let b = body;
    const base = `https://github.com/${CONFIG.repo}/issues/new?title=${enc(subject)}&body=`;
    while (base.length + enc(b).length > 7500 && b.length > 200) b = b.slice(0, Math.floor(b.length * 0.8)) + "…";
    links.issue = base + enc(b);
    links.pr = ghFileUrl("edit");
    links.view = ghFileUrl("blob");
  }
  return links;
}

function reportBox(scope, summary) {
  const id = `report-${++_reportSeq}`;
  REPORTS.set(id, scope);
  const configured = CONFIG.email || CONFIG.repo;
  const inner = configured
    ? `<textarea placeholder="What's wrong, and what should it be? (optional — you can also write it in the e-mail / issue)"></textarea>` +
      `<div class="report-actions">` +
        (CONFIG.email ? `<a class="btn btn-primary" data-link="email" href="#">✉ Email</a>` : "") +
        (CONFIG.repo ? `<a class="btn btn-primary" data-link="issue" href="#" target="_blank" rel="noopener">Open GitHub issue ↗</a>` +
                       `<a class="btn" data-link="pr" href="#" target="_blank" rel="noopener" ` +
                       `title="Edit ${escapeHtml(CURRENT.name)}.json on GitHub and open a pull request">Suggest a fix (PR) ↗</a>` +
                       `<a class="btn" data-link="view" href="#" target="_blank" rel="noopener">View JSON ↗</a>` : "") +
      `</div>`
    : `<div class="muted">Reporting links are not configured for this copy of the viewer.</div>`;
  const box = `<div class="report-box" data-report="${id}">${inner}</div>`;
  return summary
    ? `<details class="report-details" style="margin-top:16px"><summary class="section-title" style="cursor:pointer">${escapeHtml(summary)}</summary>${box}</details>`
    : box;
}

function bindReportBox(root) {
  root.querySelectorAll(".report-box[data-report]").forEach((box) => {
    const scope = REPORTS.get(box.dataset.report);
    const ta = box.querySelector("textarea");
    const update = () => {
      const links = reportLinks(scope, ta ? ta.value : "");
      box.querySelectorAll("a[data-link]").forEach((a) => { a.href = links[a.dataset.link] || "#"; });
    };
    if (ta) ta.addEventListener("input", update);
    update();
  });
}

// ---- lightbox --------------------------------------------------------------
function openLightbox(src) {
  document.getElementById("lightbox-img").src = src;
  document.getElementById("lightbox").classList.remove("hidden");
}
function closeLightbox() {
  document.getElementById("lightbox").classList.add("hidden");
  document.getElementById("lightbox-img").src = "";
}

// ---- tooltip ---------------------------------------------------------------
const tip = document.getElementById("tooltip");
function showTooltip(content, pos, containerId) {
  if (!content) return;
  tip.textContent = content;
  tip.classList.remove("img-tip");
  tip.classList.remove("hidden");
  moveTooltip(pos, containerId);
}
function moveTooltip(pos, containerId) {
  const rect = document.getElementById(containerId || "cy").getBoundingClientRect();
  tip.style.left = rect.left + pos.x + 14 + "px";
  tip.style.top = rect.top + pos.y + 14 + "px";
}
function hideTooltip() { tip.classList.add("hidden"); }

// Hover a reagent thumbnail (info panel) -> larger structure preview near cursor.
function _placeMolHover(x, y) {
  const w = 312;
  let left = x + 18, top = y + 18;
  if (left + w > window.innerWidth) left = x - w - 18;
  if (top + w > window.innerHeight) top = Math.max(8, window.innerHeight - w - 8);
  tip.style.left = left + "px";
  tip.style.top = top + "px";
}
const _panelEl = document.getElementById("panel");
if (_panelEl) {
  _panelEl.addEventListener("mouseover", (e) => {
    const t = e.target.closest && e.target.closest(".reagent-thumb");
    if (!t || !t.dataset.mol) return;
    tip.innerHTML = `<img src="${reagentUrl(t.dataset.mol, 320)}" alt="reagent" />`;
    tip.classList.add("img-tip");
    tip.classList.remove("hidden");
    _placeMolHover(e.clientX, e.clientY);
  });
  _panelEl.addEventListener("mousemove", (e) => {
    if (tip.classList.contains("hidden")) return;
    if (e.target.closest && e.target.closest(".reagent-thumb")) _placeMolHover(e.clientX, e.clientY);
  });
  _panelEl.addEventListener("mouseout", (e) => {
    if (e.target.closest && e.target.closest(".reagent-thumb")) hideTooltip();
  });
}

// ---- utils -----------------------------------------------------------------
function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

// ---- resizable panes -----------------------------------------------------
function resizeGraphs() { if (CY) CY.resize(); }
function initResizers() {
  const app = document.getElementById("app");
  const getVar = (n, d) => parseFloat(getComputedStyle(app).getPropertyValue(n)) || d;
  try {
    const s = JSON.parse(localStorage.getItem("dash-cols") || "{}");
    if (s.left) app.style.setProperty("--left", s.left + "px");
    if (s.bottom) app.style.setProperty("--bottom", s.bottom + "px");
    if (s.right) app.style.setProperty("--right", s.right + "px");
  } catch (e) {}

  let drag = null;
  document.querySelectorAll(".gutter").forEach((g) => {
    g.addEventListener("mousedown", (e) => {
      drag = { side: g.dataset.side, x: e.clientX, y: e.clientY, el: g,
               left: getVar("--left", 280), right: getVar("--right", 440),
               bottom: getVar("--bottom", window.innerHeight * 0.34) };
      g.classList.add("dragging");
      document.body.style.cursor = drag.side === "bottom" ? "row-resize" : "col-resize";
      document.body.style.userSelect = "none";
      e.preventDefault();
    });
  });
  window.addEventListener("mousemove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x;
    const maxW = window.innerWidth - 360;
    if (drag.side === "left") {
      app.style.setProperty("--left", Math.min(Math.max(drag.left + dx, 180), maxW) + "px");
    } else if (drag.side === "right") {
      app.style.setProperty("--right",
        Math.min(Math.max(drag.right - dx, 260), window.innerWidth * 0.6) + "px");
    } else {
      const dy = e.clientY - drag.y;
      app.style.setProperty("--bottom",
        Math.min(Math.max(drag.bottom - dy, 120), window.innerHeight * 0.6) + "px");
    }
    resizeGraphs();
  });
  window.addEventListener("mouseup", () => {
    if (!drag) return;
    drag.el.classList.remove("dragging");
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    try {
      localStorage.setItem("dash-cols",
        JSON.stringify({ left: getVar("--left", 280), right: getVar("--right", 440),
                         bottom: getVar("--bottom", window.innerHeight * 0.34) }));
    } catch (e) {}
    drag = null;
    resizeGraphs();
  });
}

// ---- init ------------------------------------------------------------------
async function initRDKit() {
  const wasmEl = document.getElementById("rdkit-wasm");
  const b64 = wasmEl ? wasmEl.textContent.trim() : "";
  const opts = b64 ? { wasmBinary: b64ToBytes(b64) } : { locateFile: () => CONFIG.rdkitWasmUrl };
  if (wasmEl) wasmEl.remove();   // free the base64 text
  RDKit = await window.initRDKitModule(opts);
  if (RDKit.prefer_coordgen) RDKit.prefer_coordgen(true);   // cleaner 2D for SMILES-only drawings
  LIVE = LiveCheck.create(RDKit);
}

function renderAbout() {
  const bits = [];
  if (CONFIG.built) bits.push(`Built ${escapeHtml(CONFIG.built)}`);
  if (CONFIG.repo) bits.push(`<a href="https://github.com/${escapeHtml(CONFIG.repo)}" target="_blank" rel="noopener">GitHub</a>`);
  document.getElementById("about").innerHTML = bits.join(" · ");
}

document.getElementById("btn-info").addEventListener("click", () =>
  setInfoOpen(document.getElementById("app").classList.contains("info-closed")));
document.getElementById("btn-report").addEventListener("click", showReport);
document.getElementById("btn-abbr").addEventListener("click", () => {
  ABBREV = !ABBREV;
  renderAbbrevUI();
  rerenderGraph();
});
document.getElementById("btn-color").addEventListener("click", () => {
  COLOR = !COLOR;
  renderColorUI();
  rerenderGraph();
  refreshPanel();
  showRouteInfo();          // the target structure is drawn with the same colours
});
document.getElementById("btn-map").addEventListener("click", () => {
  SHOW_MAPS = !SHOW_MAPS;
  renderMapUI();
  refreshPanel();           // only reaction drawings carry atom maps
});
document.getElementById("lightbox").addEventListener("click", closeLightbox);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeLightbox(); });

(async function main() {
  initResizers();
  let infoOpen = true;
  try { infoOpen = localStorage.getItem("dash-info") !== "0"; } catch (e) {}
  setInfoOpen(infoOpen);
  renderAbout();
  routeFilters.setRoutes(INDEX);
  try {
    await initRDKit();
  } catch (e) {
    const el = document.getElementById("loading");
    el.classList.add("error");
    document.getElementById("loading-msg").textContent =
      "Could not start the chemistry engine (RDKit.js): " + e.message +
      (CONFIG.rdkitWasmUrl ? " — this copy loads it from the internet; check your connection." : "");
    return;
  }
  document.getElementById("loading").remove();
  // Deep link: #route=<name> opens that route (handy in bug reports).
  const m = location.hash.match(/route=([^&]+)/);
  if (m) {
    const name = decodeURIComponent(m[1]);
    if (ROUTE_BLOBS[name]) selectRoute(name);
  }
})();
