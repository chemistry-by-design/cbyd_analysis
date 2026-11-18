"""Route-level topology for CbyD from ``utils.chem.build_network``: n_reactions, longest_linear_sequence, is_convergent and the connectivity flags.
"""
from __future__ import annotations
import sys
from collections import defaultdict

from .chem import build_network as _build_network

sys.setrecursionlimit(100000)


def _get_build_network():
    """Return ``build_network`` (from ``utils.chem``) — the single source of truth for the CbyD reaction network."""
    return _build_network


def _adjacency(net):
    adj = defaultdict(set)
    for e in net["edges"]:
        adj[e["source"]].add(e["target"])
        adj[e["target"]].add(e["source"])
    return adj


def _lls_and_size(root_id, adj, ntype):
    """Iterative post-order over the tree rooted at ``root_id``.

    Returns (longest #reaction-nodes on any root→leaf path, #reaction nodes in the whole sub-tree)."""
    best_path = defaultdict(int)   # node -> max reaction count on a downward path
    subtree_rx = defaultdict(int)  # node -> reaction nodes in its subtree
    # explicit DFS stack with a "post" flag
    stack = [(root_id, None, False)]
    order = []
    parent = {root_id: None}
    # first pass: establish parent + emit post-order
    while stack:
        node, par, processed = stack.pop()
        if processed:
            order.append((node, par))
            continue
        parent[node] = par
        stack.append((node, par, True))
        for nb in adj[node]:
            if nb != par:
                stack.append((nb, node, False))
    # post-order accumulation
    for node, par in order:
        inc = 1 if ntype.get(node) == "reaction" else 0
        child_best = 0
        child_sum = 0
        for nb in adj[node]:
            if nb == parent.get(node):
                continue
            child_best = max(child_best, best_path[nb])
            child_sum += subtree_rx[nb]
        best_path[node] = child_best + inc
        subtree_rx[node] = child_sum + inc
    return best_path[root_id], subtree_rx[root_id]


def route_metrics(synthesis: list) -> dict:
    """Topology metrics for one CbyD route's ``synthesis`` list."""
    build_network = _get_build_network()
    out = {
        "n_steps": len(synthesis or []),
        "n_targets": None, "connected": False, "n_orphans": None,
        "target": None, "n_reactions": None,
        "longest_linear_sequence": None, "is_convergent": None,
        "error": None,
    }
    try:
        net = build_network(synthesis, align=False)
    except Exception as e:
        out["error"] = f"build_network: {e}"
        return out

    out["n_steps"] = net.get("n_steps", out["n_steps"])
    out["n_targets"] = net.get("n_targets")
    out["n_orphans"] = len(net.get("orphan_reactions", []) or [])
    out["connected"] = bool(net.get("connected"))

    ntype = {n["id"]: n["type"] for n in net["nodes"]}
    targets = [n for n in net["nodes"]
               if n["type"] == "molecule" and n.get("role") == "target"]
    if not targets:
        out["error"] = "no target node"
        return out

    adj = _adjacency(net)
    # If several targets (disconnected route), score the largest target tree.
    best = None
    for t in targets:
        lls, size = _lls_and_size(t["id"], adj, ntype)
        if best is None or size > best[1]:
            best = (lls, size, t["smiles"])
    lls, size, tsmiles = best
    out["target"] = tsmiles
    out["n_reactions"] = size
    out["longest_linear_sequence"] = lls
    out["is_convergent"] = size > lls if size is not None else None
    return out


def target_smiles(synthesis: list):
    """Just the target SMILES for a route (or None)."""
    return route_metrics(synthesis).get("target")
