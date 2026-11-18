"use strict";

// livecheck.js — compare a route as it was BUILT into dashboard.html with the
// route's JSON file as it is NOW, and patch the built route with what changed.
//
// build_static.py bakes every route into the page once. If a route file is edited
// afterwards, the page is stale until someone rebuilds it. The viewer therefore fetches the route file when a route is opened and hands both versions
// to LiveCheck.create(RDKit).diffRoute(built, live):
//
//   { status: "same" }                      nothing to do — show the built route
//   { status: "changed", route, summary }   `route` is a payload in the build's own
//                                           format, with the changes applied + marked
//   { status: "unverified", summary }       the file differs, but this code could not
//                                           reproduce the built tree, so it will not
//                                           guess — show the built route
//
// Reactions that are textually unchanged reuse what the Python build computed (it
// ships its inputs with each reaction: `s`, `c`, `a` — see build_static.py). Only
// reactions that really changed go through the JS port of chem.py's cleaning chain
// below, and molecules that are new to the route are laid out here with RDKit.js.
//
// No DOM access in this file, so it also runs under Node.

const LiveCheck = (() => {
  // Same list, same order, as PARSED_KEYS in build_static.py.
  const PARSED_KEYS = [
    "GENERAL_REAGENTS", "CATALYSTS", "LIGANDS", "SOLVENTS", "MODIFIERS",
    "TEMPERATURE", "YIELD",
    "GENERAL_REAGENTS_DEFINITIONS", "CATALYSTS_DEFINITIONS",
    "LIGANDS_DEFINITIONS", "SOLVENTS_DEFINITIONS",
  ];
  const NORMALIZE_KEYS = ["GENERAL_REAGENTS", "CATALYSTS", "LIGANDS", "SOLVENTS", "MODIFIERS"];
  const REAGENT_NAME_KEYS = new Set(NORMALIZE_KEYS);
  const BG_KEYS = ["name", "author", "year", "drug name", "manufacturer", "citation", "doi"];
  const NODE_CAP = 2500;          // as in chem.build_network

  // Python truthiness for the JSON values we meet ([] and {} are falsy).
  const truthy = (v) => Array.isArray(v) ? v.length > 0
    : (v && typeof v === "object") ? Object.keys(v).length > 0 : !!v;
  const same = (a, b) => JSON.stringify(a === undefined ? null : a) === JSON.stringify(b === undefined ? null : b);

  // ---- chem.normalize_reaction / normalize_route ---------------------------
  function normalizeReaction(reaction) {
    const conditions = truthy(reaction.conditions) ? reaction.conditions : {};
    const parsed = Object.assign({}, conditions, truthy(reaction.parsed_reaction) ? reaction.parsed_reaction : {});
    for (const key of NORMALIZE_KEYS) {
      if (!truthy(parsed[key + "_DEFINITIONS"]) && truthy(parsed[key + "_SMILES"])) {
        parsed[key + "_DEFINITIONS"] = parsed[key + "_SMILES"];
      }
    }
    return Object.assign({}, reaction, {
      parsed_reaction: parsed,
      raw_description: reaction.raw_description || conditions.raw_description || "",
      full_reaction_smiles_mapped: reaction.full_reaction_smiles_mapped || reaction.reaction_smiles_mapped || "",
    });
  }
  function normalizeRoute(route) {
    const background = Object.assign({}, truthy(route.background) ? route.background : {});
    if (!truthy(background["drug name"])) {
      background["drug name"] = background.drug_name === undefined ? null : background.drug_name;
    }
    const synthesis = Array.isArray(route.synthesis) ? route.synthesis : [];
    return { background, synthesis: synthesis.map((r) => normalizeReaction(r || {})) };
  }

  const splitRxn = (s) => s.split(/(?<!-)>(?!-)/);
  const str = (v) => (typeof v === "string" ? v : v == null ? "" : String(v));

  // One reaction of the route file -> the row build_static.export_route would write.
  function liveRow(rxn, index) {
    const parsed = rxn.parsed_reaction || {};
    const p = {};
    for (const k of PARSED_KEYS) if (truthy(parsed[k])) p[k] = parsed[k];
    const m = str(rxn.full_reaction_smiles_mapped);
    return {
      rid: "reaction_id" in rxn ? str(rxn.reaction_id) : String(index),
      d: str(rxn.raw_description), m, p,
      source: m.trim() || str(rxn.full_reaction_smiles) || str(rxn.reaction_smiles),
    };
  }
  // One reaction as stored in the page.
  function builtRow(r) {
    const p = {};
    for (const k of PARSED_KEYS) if (r.p && truthy(r.p[k])) p[k] = r.p[k];
    return {
      rid: str(r.rid), d: str(r.d), m: str(r.m), p,
      source: "s" in r ? str(r.s) : str(r.m).trim(),
      hasParts: "c" in r, c: r.c, a: r.a || [],
    };
  }
  const rowSig = (row) => JSON.stringify([row.source, row.d, row.p]);

  // ---- pairing old and new reactions ---------------------------------------
  // Identical reactions anchor the alignment (longest common subsequence), so an
  // inserted or deleted step does not make every later step look "changed". What
  // lies between two anchors is paired by reaction_id, then by position.
  function alignRows(oldRows, newRows) {
    const n = oldRows.length, m = newRows.length;
    const os = oldRows.map(rowSig), ns = newRows.map(rowSig);
    const pairs = [];
    const gap = (oi, oj, ni, nj) => {       // old[oi:oj) vs new[ni:nj)
      const oldIdx = [], newIdx = [];
      for (let i = oi; i < oj; i++) oldIdx.push(i);
      for (let j = ni; j < nj; j++) newIdx.push(j);
      const partner = new Map();            // new index -> old index
      const used = new Set();
      let last = -1;
      for (const j of newIdx) {
        const i = oldIdx.find((x) => x > last && !used.has(x) && oldRows[x].rid === newRows[j].rid);
        if (i !== undefined) { partner.set(j, i); used.add(i); last = i; }
      }
      const restOld = oldIdx.filter((i) => !used.has(i));
      const restNew = newIdx.filter((j) => !partner.has(j));
      for (let k = 0; k < Math.min(restOld.length, restNew.length); k++) {
        partner.set(restNew[k], restOld[k]); used.add(restOld[k]);
      }
      for (const i of oldIdx) if (!used.has(i)) pairs.push({ o: i, n: null });
      for (const j of newIdx) pairs.push({ o: partner.has(j) ? partner.get(j) : null, n: j });
    };
    if (n * m > 1e6) {                      // absurdly long route: plain positional
      gap(0, n, 0, m);
      return pairs;
    }
    const w = m + 1;
    const dp = new Uint32Array((n + 1) * w);
    for (let i = n - 1; i >= 0; i--) {
      for (let j = m - 1; j >= 0; j--) {
        dp[i * w + j] = os[i] === ns[j] ? dp[(i + 1) * w + j + 1] + 1
          : Math.max(dp[(i + 1) * w + j], dp[i * w + j + 1]);
      }
    }
    let i = 0, j = 0, gi = 0, gj = 0;
    while (i < n && j < m) {
      if (os[i] === ns[j]) {
        gap(gi, i, gj, j);
        pairs.push({ o: i, n: j });
        i++; j++; gi = i; gj = j;
      } else if (dp[(i + 1) * w + j] >= dp[i * w + j + 1]) i++;
      else j++;
    }
    gap(gi, n, gj, m);
    return pairs;
  }

  // Multiset difference of two SMILES lists: what `b` has that `a` lacks, and back.
  function listDiff(a, b) {
    const count = new Map();
    for (const x of a) count.set(x, (count.get(x) || 0) + 1);
    const added = [];
    for (const x of b) {
      const c = count.get(x) || 0;
      if (c > 0) count.set(x, c - 1); else added.push(x);
    }
    const removed = [];
    for (const [x, c] of count) for (let k = 0; k < c; k++) removed.push(x);
    return { added, removed };
  }
  const moved = (d) => d.added.length > 0 || d.removed.length > 0;

  function create(RDKit) {
    // ---- SMILES helpers (ports of chem.py) ----------------------------------
    function molSmiles(smiles) {
      let mol = null;
      try {
        mol = RDKit.get_mol(smiles);
        if (!mol || !mol.is_valid()) return null;
        return mol.get_smiles();
      } catch (e) {
        return null;
      } finally {
        if (mol) mol.delete();
      }
    }
    const canonCache = new Map();
    // chem.safe_canonicalize: canonical SMILES without atom-map numbers, or null.
    function canon(smiles) {
      if (typeof smiles !== "string") return null;
      if (smiles === "") return "";
      let out = canonCache.get(smiles);
      if (out !== undefined) return out;
      out = null;
      const first = molSmiles(smiles.replace(/:\d+\]/g, "]"));
      if (first !== null) {
        const second = molSmiles(first);
        out = second !== null ? second : first;
      }
      canonCache.set(smiles, out);
      return out;
    }
    // The key a molecule is known by in here. SMILES written by the Python build
    // are re-canonicalised by RDKit.js too, so both sides are compared like for like.
    const keyOf = (smiles) => canon(smiles) || smiles;

    let salts = null;
    function saltSet() {
      if (salts) return salts;
      salts = new Set();
      for (const s of [
        "Cl", "Cl-", "[Cl-]", "Cl[H]", "ClH", "Cl.H", "HCl", "[H][Cl]",
        "Br", "Br-", "[Br-]", "Br[H]", "BrH", "Br.H", "HBr", "[H][Br]",
        "I", "I-", "[I-]", "IH", "I.H", "HI", "[H][I]",
        "F", "F-", "[F-]", "FH", "F.H", "HF", "[H][F]",
        "OP(=O)(O)O", "OP(=O)(O)[O-]", "OP(=O)([O-])[O-]", "[O-]P(=O)([O-])[O-]",
        "OS(=O)(=O)O", "OS(=O)(=O)[O-]", "[O-]S(=O)(=O)[O-]",
        "Cc1ccc(S(=O)(=O)O)cc1", "Cc1ccc(S(=O)(=O)[O-])cc1",
        "OS(C)(=O)=O", "[O-]S(C)(=O)=O",
        "OS(=O)(=O)C(F)(F)F", "[O-]S(=O)(=O)C(F)(F)F",
        "OC(=O)C(F)(F)F", "[O-]C(=O)C(F)(F)F",
      ]) {
        const c = canon(s);
        if (c) salts.add(c);
      }
      return salts;
    }
    const isSalt = (c) => c != null && c !== "" && saltSet().has(c);

    let carbonQuery = null;
    function countCarbons(smiles) {
      let mol = null;
      try {
        mol = RDKit.get_mol(smiles);
        if (!mol || !mol.is_valid()) return 0;
        if (!carbonQuery) carbonQuery = RDKit.get_qmol("[#6]");
        const hits = JSON.parse(mol.get_substruct_matches(carbonQuery));
        return Array.isArray(hits) ? hits.length : (hits && hits.atoms ? 1 : 0);
      } catch (e) {
        return 0;
      } finally {
        if (mol) mol.delete();
      }
    }
    function mapNums(component) {
      const out = new Set();
      for (const m of component.matchAll(/:(\d+)\]/g)) if (Number(m[1])) out.add(Number(m[1]));
      return out;
    }

    // chem._protonation_key. RDKit.js has no Uncharger, so this covers what that
    // step is for here: a counter-ion vs its acid (Cl-/HCl, TFA-/TFA, R-NH3+/R-NH2).
    function protonationKey(smiles) {
      const c = canon(smiles);
      if (!c) return null;
      let neutral = c;
      if (c.includes("-]") && !c.includes("+]")) {
        neutral = c.replace(/\[(O|S|F|Cl|Br|I)-\]/g, "$1");
      } else if (c.includes("+]") && !c.includes("-]")) {
        neutral = c.replace(/\[NH3?\+\]/g, "N").replace(/\[NH2\+\]/g, "N").replace(/\[nH\+\]/g, "n");
      }
      return (neutral !== c && canon(neutral)) || c;
    }

    function checkSame(rxn) {                // chem.check_if_reactant_product_same
      const parts = splitRxn(rxn);
      if (parts.length < 3) return false;
      const splitSalt = (side) => {
        const comps = side.split(".").filter(Boolean);
        return [comps.filter((c) => !isSalt(canon(c))), comps.filter((c) => isSalt(canon(c)))];
      };
      const [nhR, hR] = splitSalt(parts[0]);
      const [nhP, hP] = splitSalt(parts[2]);
      let rc = parts[0], pc = parts[2];
      if (hR.length && hP.length && nhR.length && nhP.length) { rc = nhR.join("."); pc = nhP.join("."); }
      const a = canon(rc), b = canon(pc);
      return a !== null && a === b;
    }

    function cleanDuplicates(rxn) {          // chem.clean_rxn_duplicates
      const parts = splitRxn(rxn);
      if (parts.length < 3) return rxn;
      const reactants = parts[0].split("."), products = parts[parts.length - 1].split(".");
      const rc = reactants.map(canon), pc = products.map(canon);
      const common = new Set(rc.filter((x) => x !== null && pc.includes(x)));
      const count = (list, x) => list.filter((y) => y === x).length;
      let rNew = reactants.filter((r, i) => !common.has(rc[i]) || count(rc, rc[i]) > count(pc, rc[i]));
      let pNew = products.filter((p, i) => !common.has(pc[i]) || count(pc, pc[i]) > count(rc, pc[i]));
      if (rNew.length > 1 && pNew.length > 1) {
        const rk = rNew.map(protonationKey), pk = pNew.map(protonationKey);
        const shared = new Set(rk.filter((k) => k && pk.includes(k)));
        if (shared.size) {
          const r2 = rNew.filter((r, i) => !shared.has(rk[i]));
          const p2 = pNew.filter((p, i) => !shared.has(pk[i]));
          if (r2.length && p2.length) { rNew = r2; pNew = p2; }
        }
      }
      return rNew.join(".") + ">" + parts[1] + ">" + pNew.join(".");
    }

    function removeSpectatorHalides(rxn) {   // chem.remove_spectator_halides
      const parts = splitRxn(rxn);
      if (parts.length < 3) return rxn;
      const organic = new Set();
      for (const c of parts[parts.length - 1].split(".")) {
        if (!c || isSalt(canon(c))) continue;
        for (const n of mapNums(c)) organic.add(n);
      }
      const stripSide = (side) => {
        const comps = side.split(".").filter(Boolean);
        const keep = [], halide = [];
        for (const c of comps) {
          const incorporated = [...mapNums(c)].some((n) => organic.has(n));
          (isSalt(canon(c)) && !incorporated ? halide : keep).push(c);
        }
        return halide.length && keep.some((c) => countCarbons(c) > 1) ? keep : comps;
      };
      return stripSide(parts[0]).join(".") + ">" + parts[1] + ">" + stripSide(parts[parts.length - 1]).join(".");
    }

    function cleanForNodes(rxn) {            // chem._clean_reaction_for_nodes
      if (checkSame(rxn)) return null;
      const parts = splitRxn(removeSpectatorHalides(cleanDuplicates(rxn)));
      if (parts.length < 3) return null;
      const side = (s) => s.split(".").filter(Boolean).map(canon).filter(Boolean);
      const reactants = side(parts[0]), products = side(parts[parts.length - 1]);
      return reactants.length && products.length ? [reactants, products] : null;
    }

    // chem.reaction_parts, from a source reaction SMILES.
    function reactionParts(source) {
      const parts = source ? splitRxn(source) : [];
      const agents = parts.length >= 3 ? parts[1].split(".").map(canon).filter(Boolean) : [];
      let cleaned = null;
      if (source) {
        try { cleaned = cleanForNodes(source); } catch (e) { cleaned = null; }
      }
      return { source, agents, cleaned };
    }

    // ---- the tree (port of pass 2 of chem.build_network) --------------------
    // parts[i] = {source, agents, cleaned}; sortName orders the roots the way the
    // Python build did. Returns nodes in the build's order, with the bookkeeping
    // (consumer / producer / parent) needed to place depictions and change marks.
    function buildTree(parts, sortName) {
      const agents = new Set();
      const produced = new Set(), consumed = new Set();
      const producedBy = new Map();
      parts.forEach((p, index) => {
        for (const a of p.agents) agents.add(a);
        if (!p.cleaned) return;
        for (const r of p.cleaned[0]) consumed.add(r);
        for (const s of p.cleaned[1]) {
          produced.add(s);
          if (!producedBy.has(s)) producedBy.set(s, []);
          producedBy.get(s).push(index);
        }
      });
      const nodes = [], edges = [], used = new Set();
      let counter = 0, capped = false;
      const nid = (prefix) => `${prefix}::${++counter}`;

      function expand(key, path, isRoot, consumer, parentKey) {
        const prods = (producedBy.get(key) || []).filter((j) => !path.has(j));
        const role = isRoot ? (agents.has(key) ? "byproduct" : "target")
          : prods.length ? "intermediate" : "start";
        const node = { id: nid("m"), type: "molecule", key, role, consumer, parentKey, producer: null };
        nodes.push(node);
        if (!prods.length || nodes.length > NODE_CAP) {
          capped = capped || nodes.length > NODE_CAP;
          return node.id;
        }
        const producer = Math.max(...prods);
        node.producer = producer;
        const rxid = nid("rx");
        nodes.push({ id: rxid, type: "reaction", index: producer });
        used.add(producer);
        edges.push([rxid, node.id]);
        const counts = new Map();
        for (const r of parts[producer].cleaned[0]) counts.set(r, (counts.get(r) || 0) + 1);
        const next = new Set(path);
        next.add(producer);
        for (const [r, mult] of counts) {
          const cid = expand(r, next, false, producer, key);
          for (let k = 0; k < mult; k++) edges.push([cid, rxid]);
        }
        return node.id;
      }

      const roots = [...produced].filter((p) => !consumed.has(p));
      roots.sort((a, b) => { const x = sortName(a), y = sortName(b); return x < y ? -1 : x > y ? 1 : 0; });
      for (const root of roots) expand(root, new Set(), true, null, null);

      const warnings = [];
      if (capped) warnings.push(`Tree truncated at ${NODE_CAP} nodes (highly convergent route).`);
      parts.forEach((p, index) => {
        if (used.has(index)) return;
        const node = { id: `rxn::${index}`, type: "reaction", index, orphan: true };
        if (!p.cleaned) {
          if (p.source && checkSame(p.source)) {
            warnings.push(`Rxn ${index + 1}: reactants == products (salt/complex); shown with no molecule edges.`);
          } else if (!p.source) {
            warnings.push(`Rxn ${index + 1}: no reaction SMILES available.`);
          } else {
            node.invalid = true;
            warnings.push(`Rxn ${index + 1}: SMILES could not be parsed — shown as invalid (red).`);
          }
        }
        nodes.push(node);
      });
      return { nodes, edges, warnings };
    }

    // Does our tree for the BUILT inputs equal the tree in the page, node for node?
    function matchesBuilt(tree, built) {
      const bn = built.nodes || [], be = built.edges || [];
      if (tree.nodes.length !== bn.length || tree.edges.length !== be.length) return false;
      for (let k = 0; k < bn.length; k++) {
        const a = tree.nodes[k], b = bn[k];
        if (a.id !== b.id) return false;
        if (b.t === "m") {
          if (a.type !== "molecule" || a.role !== b.r || a.key !== keyOf(b.s)) return false;
        } else if (a.type !== "reaction" || a.index !== b.i) return false;
      }
      for (let k = 0; k < be.length; k++) {
        if (tree.edges[k][0] !== be[k][0] || tree.edges[k][1] !== be[k][1]) return false;
      }
      return true;
    }

    // ---- a fresh 2D layout for a molecule the build never saw ----------------
    // Mirrors chem._plain_coords: CoordGen layout, long axis horizontal. Returns
    // the node fields the viewer sizes and draws from ({cx, ew, eh, na}).
    function depict(smiles) {
      let mol = null;
      try {
        mol = RDKit.get_mol(smiles);
        if (!mol || !mol.is_valid()) return {};
        const out = { na: mol.get_num_atoms() };
        const read = () => {
          const cx = mol.get_cxsmiles();
          const m = cx.match(/^(\S+ \|[^(]*)\(([^)]*)\)(.*)$/);
          if (!m) return null;
          const pts = m[2].split(";").map((t) => t.split(",").slice(0, 2).map(Number));
          return pts.length && pts.every((p) => p.length === 2 && p.every(Number.isFinite))
            ? { head: m[1], pts, tail: m[3] } : null;
        };
        mol.set_new_coords(mol.get_num_bonds() > 0);     // CoordGen; plain RDKit for lone atoms
        let got = read();
        if (!got) { mol.set_new_coords(false); got = read(); }
        if (!got) return out;
        const { pts } = got;
        const n = pts.length;
        if (n > 1) {                                     // chem._horizontalize
          const cx0 = pts.reduce((s, p) => s + p[0], 0) / n, cy0 = pts.reduce((s, p) => s + p[1], 0) / n;
          let sxx = 0, syy = 0, sxy = 0;
          for (const [x, y] of pts) { sxx += (x - cx0) ** 2; syy += (y - cy0) ** 2; sxy += (x - cx0) * (y - cy0); }
          const theta = 0.5 * Math.atan2(2 * sxy, sxx - syy);
          const c = Math.cos(theta), s = Math.sin(theta);
          for (const p of pts) {
            const x = p[0] - cx0, y = p[1] - cy0;
            p[0] = x * c + y * s + cx0;
            p[1] = -x * s + y * c + cy0;
          }
        }
        const num = (v) => { const t = v.toFixed(2).replace(/\.?0+$/, ""); return t === "-0" || t === "" ? "0" : t; };
        out.cx = got.head + "(" + pts.map((p) => `${num(p[0])},${num(p[1])},`).join(";") + ")" + got.tail;
        const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
        out.ew = Math.round((Math.max(...xs) - Math.min(...xs)) * 1000) / 1000;
        out.eh = Math.round((Math.max(...ys) - Math.min(...ys)) * 1000) / 1000;
        return out;
      } catch (e) {
        return {};
      } finally {
        if (mol) mol.delete();
      }
    }

    // ---- the comparison -------------------------------------------------------
    function diffRoute(built, liveRaw) {
      const live = normalizeRoute(liveRaw || {});
      const oldRows = (built.rx || []).map(builtRow);
      const newRows = live.synthesis.map(liveRow);
      const bgOld = built.bg || {};
      const bgChanged = BG_KEYS.filter((k) => !same(bgOld[k], live.background[k]));

      const textSame = oldRows.length === newRows.length &&
        oldRows.every((r, i) => rowSig(r) === rowSig(newRows[i]));
      if (textSame && !bgChanged.length) return { status: "same" };

      // What each reaction contributes to the tree. Unchanged reactions keep the
      // Python build's answer; only changed ones are cleaned here.
      const pyName = new Map();           // key -> SMILES as the build wrote it
      const key = (s) => { const k = keyOf(s); if (!pyName.has(k)) pyName.set(k, s); return k; };
      const oldParts = oldRows.map((r) => r.hasParts
        ? { source: r.source, agents: r.a.map(key), cleaned: r.c ? [r.c[0].map(key), r.c[1].map(key)] : null }
        : reactionParts(r.source));
      for (const n of built.nodes || []) if (n.t === "m") key(n.s);
      const sortName = (k) => (pyName.has(k) ? pyName.get(k) : k);

      const pairs = alignRows(oldRows, newRows);
      const oldOfNew = new Array(newRows.length).fill(null);
      const newOfOld = new Array(oldRows.length).fill(null);
      for (const p of pairs) if (p.o !== null && p.n !== null) { oldOfNew[p.n] = p.o; newOfOld[p.o] = p.n; }
      const newParts = newRows.map((r, j) => {
        const i = oldOfNew[j];
        return i !== null && oldRows[i].source === r.source ? oldParts[i] : reactionParts(r.source);
      });

      const oldKeys = new Set();
      for (const p of oldParts) if (p.cleaned) for (const side of p.cleaned) for (const k of side) oldKeys.add(k);

      // Per-reaction differences.
      const defKeys = (list) => (Array.isArray(list) ? list : []).map((s) => keyOf(str(s)));
      const diffs = newRows.map((row, j) => {
        const i = oldOfNew[j], parts = newParts[j];
        const sides = parts.cleaned || [[], []];
        if (i === null) {
          return {
            status: "added", kinds: { substrates: true, reagents: false, details: false },
            reactants: { added: sides[0].filter((k) => !oldKeys.has(k)), removed: [] },
            products: { added: sides[1].filter((k) => !oldKeys.has(k)), removed: [] },
            agents: { added: [], removed: [] }, fields: {},
          };
        }
        const old = oldRows[i], oparts = oldParts[i];
        const osides = oparts.cleaned || [[], []];
        const d = {
          status: "changed", kinds: { substrates: false, reagents: false, details: false, mapping: false },
          reactants: listDiff(osides[0], sides[0]), products: listDiff(osides[1], sides[1]),
          agents: listDiff(oparts.agents, parts.agents), fields: {}, oldNumber: i + 1,
        };
        d.kinds.substrates = moved(d.reactants) || moved(d.products) || !oparts.cleaned !== !parts.cleaned;
        d.kinds.reagents = moved(d.agents);
        // Same molecules, but the reaction SMILES itself was rewritten — typically
        // the atom mapping. Flagged, nothing to highlight.
        if (old.source !== row.source && !d.kinds.substrates && !d.kinds.reagents) {
          d.fields.m = { old: old.source, new: row.source };
          d.kinds.mapping = true;
        }
        if (old.d !== row.d) { d.fields.d = { old: old.d, new: row.d }; d.kinds.details = true; }
        for (const k of PARSED_KEYS) {
          if (same(old.p[k], row.p[k])) continue;
          if (k.endsWith("_DEFINITIONS")) {
            const dd = listDiff(defKeys(old.p[k]), defKeys(row.p[k]));
            if (!moved(dd)) continue;               // same molecules, written differently
            d.fields[k] = { old: old.p[k] || [], new: row.p[k] || [], added: dd.added, removed: dd.removed };
            d.kinds.reagents = true;
          } else {
            d.fields[k] = { old: old.p[k] || [], new: row.p[k] || [] };
            if (REAGENT_NAME_KEYS.has(k)) d.kinds.reagents = true; else d.kinds.details = true;
          }
        }
        return d.kinds.substrates || d.kinds.reagents || d.kinds.details || d.kinds.mapping ? d : null;
      });
      const removed = pairs.filter((p) => p.n === null).map((p) => ({
        number: p.o + 1, rid: oldRows[p.o].rid, d: oldRows[p.o].d, m: oldRows[p.o].m,
      }));

      const summary = {
        changed: diffs.map((d, j) => (d && d.status === "changed" ? j : -1)).filter((j) => j >= 0),
        added: diffs.map((d, j) => (d && d.status === "added" ? j : -1)).filter((j) => j >= 0),
        removed, background: bgChanged, molecules: 0,
      };
      if (!summary.changed.length && !summary.added.length && !removed.length && !bgChanged.length) {
        return { status: "same" };                 // e.g. a reagent SMILES written differently
      }

      const rx = newRows.map((row, j) => {
        const out = { rid: row.rid, d: row.d, m: row.m, p: row.p };
        if (diffs[j]) out.diff = diffs[j];
        return out;
      });
      const route = { bg: live.background, rx, nodes: built.nodes, edges: built.edges, warn: built.warn || [] };

      const rebuild = removed.length > 0 || diffs.some((d) => d && (d.status === "added" || d.kinds.substrates));
      if (!rebuild) return { status: "changed", route, summary };

      // The molecules changed, so the tree has to be rebuilt. First prove this code
      // reproduces the tree that is in the page from the build's own inputs.
      const oldTree = buildTree(oldParts, sortName);
      if (!matchesBuilt(oldTree, built)) return { status: "unverified", summary };

      // Built depictions, to reuse wherever the same molecule is still in the tree.
      const exact = new Map(), any = new Map();
      const slot = (k, consumer, parentKey) => `${k}\n${consumer}\n${parentKey}`;
      oldTree.nodes.forEach((n, k) => {
        if (n.type !== "molecule") return;
        const b = built.nodes[k];
        const consumer = n.consumer === null ? null : newOfOld[n.consumer];
        if (!exact.has(slot(n.key, consumer, n.parentKey))) exact.set(slot(n.key, consumer, n.parentKey), b);
        if (!any.has(n.key) || (!any.get(n.key).cx && b.cx)) any.set(n.key, b);
      });

      const tree = buildTree(newParts, sortName);
      const fresh = new Map();
      route.nodes = tree.nodes.map((n) => {
        if (n.type === "reaction") {
          const out = { id: n.id, t: "r", i: n.index };
          if (n.invalid) out.inv = 1;
          return out;
        }
        const b = exact.get(slot(n.key, n.consumer, n.parentKey)) || any.get(n.key);
        let out;
        if (b) out = Object.assign({}, b, { id: n.id, r: n.role });
        else {
          if (!fresh.has(n.key)) fresh.set(n.key, depict(n.key));
          out = Object.assign({ id: n.id, t: "m", s: n.key, r: n.role }, fresh.get(n.key));
        }
        // Marked when its reaction gained it, or when it had to be drawn from scratch.
        const asReactant = n.consumer !== null && diffs[n.consumer] && diffs[n.consumer].reactants;
        const asProduct = n.producer !== null && diffs[n.producer] && diffs[n.producer].products;
        const was = [];
        let marked = !b;
        if (asReactant && asReactant.added.includes(n.key)) { marked = true; was.push(...asReactant.removed); }
        if (asProduct && asProduct.added.includes(n.key)) { marked = true; was.push(...asProduct.removed); }
        if (marked) {
          out.chg = 1;
          if (was.length) out.was = [...new Set(was)];
          summary.molecules++;
        }
        return out;
      });
      route.edges = tree.edges;
      route.warn = tree.warnings;
      return { status: "changed", route, summary };
    }

    return { diffRoute, canon, cleanForNodes, reactionParts, buildTree, matchesBuilt, depict, keyOf };
  }

  return { create, normalizeRoute, normalizeReaction, alignRows, listDiff, PARSED_KEYS };
})();

if (typeof module !== "undefined" && module.exports) module.exports = LiveCheck;
