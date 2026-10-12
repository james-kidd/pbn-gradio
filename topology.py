"""
Discrete structure of the segmentation: planar graphs, colouring, Steiner trees.

A segmentation of an image into connected regions is a **planar subdivision**,
and its region adjacency graph is the dual of that subdivision — therefore
planar. That is not a decorative observation. Planarity is a strong structural
constraint, and it buys three concrete things here.

**1. Invariants that catch bugs.** A simple planar graph on V >= 3 vertices
satisfies E <= 3V - 6, and its average degree is strictly below 6. These are
necessary conditions, so a violation is proof that something upstream is wrong
— most likely a "region" that is actually two disconnected blobs sharing a
label, which breaks the duality the whole pipeline assumes. Used as a test
oracle rather than a fact to recite.

**2. A complexity guarantee.** E <= 3V - 6 means the adjacency graph has O(V)
edges, not O(V^2). Every graph algorithm downstream — adjacency scanning,
colouring, routing over parcels — is therefore linear in the number of regions.
This is why the pipeline stays fast as segmentation gets finer, and it is a
guarantee rather than an observation.

**3. Bounded scheduling.** By the Four Colour Theorem, four colours suffice to
colour any planar graph so that no two adjacent regions share one. Read as a
work schedule — adjacent parcels cannot be worked simultaneously, because one
crew and its spoil occupy both — this says any site plan can be executed in at
most four parallel rounds, no matter how many parcels there are. Greedy
colouring in degeneracy order does not always hit four, but planar graphs are
5-degenerate, so it never needs more than six.

Separately, naming a problem correctly is worth as much as solving it: the
siting loop's "connect a set of chosen sites to an existing network as cheaply
as possible" is the **Steiner tree problem** in a metric space, which is
NP-hard. The standard 2-approximation — a minimum spanning tree over the metric
closure of the terminals — is computed here as a benchmark, so the cost of
choosing sites by desirability rather than by pure cheapness becomes a number
rather than a hand-wave.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from skimage import graph as skgraph
from skimage.measure import label as cc_label
from skimage.measure import regionprops


# ── The region adjacency graph as a discrete object ───────────

@dataclass
class RegionGraph:
    """Undirected planar graph: vertices are regions, edges are shared borders."""
    vertices: list[int]
    edges: set[tuple[int, int]]
    adjacency: dict[int, set[int]] = field(default_factory=dict)
    border_length: dict[tuple[int, int], int] = field(default_factory=dict)

    @property
    def V(self) -> int:
        return len(self.vertices)

    @property
    def E(self) -> int:
        return len(self.edges)

    @property
    def mean_degree(self) -> float:
        return 2.0 * self.E / self.V if self.V else 0.0

    @property
    def faces(self) -> int:
        """Faces of the planar embedding, via Euler's formula V - E + F = 2.

        Valid only for a connected graph; `components` below is what says
        whether that precondition actually holds.
        """
        return self.E - self.V + 2

    @property
    def components(self) -> int:
        seen: set[int] = set()
        count = 0
        for v in self.vertices:
            if v in seen:
                continue
            count += 1
            stack = [v]
            seen.add(v)
            while stack:
                u = stack.pop()
                for nb in self.adjacency.get(u, ()):
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        return count


def build_region_graph(labels: np.ndarray) -> RegionGraph:
    """Extract the adjacency graph from a labelled image in one pass.

    Linear in pixels. The border length of each edge is recorded because it is
    the natural edge weight for anything that cares how *much* two parcels
    touch rather than merely whether they do.
    """
    ids = np.unique(labels)
    ids = ids[ids > 0]
    graph_obj = RegionGraph(vertices=ids.tolist(), edges=set())
    graph_obj.adjacency = {int(i): set() for i in ids}

    for axis in (0, 1):
        a = labels.take(np.arange(labels.shape[axis] - 1), axis=axis)
        b = labels.take(np.arange(1, labels.shape[axis]), axis=axis)
        differs = a != b
        for u, v in zip(a[differs].tolist(), b[differs].tolist()):
            if u <= 0 or v <= 0:
                continue
            key = (u, v) if u < v else (v, u)
            graph_obj.edges.add(key)
            graph_obj.border_length[key] = graph_obj.border_length.get(key, 0) + 1
            graph_obj.adjacency[u].add(v)
            graph_obj.adjacency[v].add(u)

    return graph_obj


# ── 1. Planarity as a test oracle ─────────────────────────────

def check_planarity(graph_obj: RegionGraph, labels: np.ndarray) -> dict:
    """Verify the structural invariants a region adjacency graph must satisfy.

    These are *necessary* conditions for planarity, not sufficient ones — a
    graph can satisfy the edge bound and still be non-planar. That is fine for
    the purpose. The graph is planar by construction if the segmentation is
    well-formed, so these checks are not asking "is this planar?" but "is the
    segmentation well-formed?", and they answer it cheaply.

    The disconnection check is the one that has actually caught things: if a
    single label covers two separate blobs, the graph is no longer the dual of
    a planar subdivision and every adjacency-derived feature silently means
    something different from what it claims.
    """
    V, E = graph_obj.V, graph_obj.E
    edge_bound = 3 * V - 6 if V >= 3 else None

    disconnected = []
    for i in graph_obj.vertices:
        if cc_label(labels == i, connectivity=2).max() > 1:
            disconnected.append(int(i))

    return {
        "V": V,
        "E": E,
        "faces": graph_obj.faces,
        "components": graph_obj.components,
        "edge_bound": edge_bound,
        "edge_bound_holds": edge_bound is None or E <= edge_bound,
        "edge_headroom": (edge_bound - E) if edge_bound is not None else None,
        "mean_degree": graph_obj.mean_degree,
        "degree_bound_holds": graph_obj.mean_degree < 6.0,
        "disconnected_regions": disconnected,
        "well_formed": (
            (edge_bound is None or E <= edge_bound)
            and graph_obj.mean_degree < 6.0
            and not disconnected
        ),
    }


def region_holes(labels: np.ndarray) -> dict[int, int]:
    """Number of holes in each region, from its Euler number.

    For a single connected component the Euler number is 1 - (number of holes),
    so a parcel wrapped around a pond reports 1. This is a genuine planning
    fact rather than a curiosity: a holed parcel has interior boundary that
    cannot be built on and must be worked around, and its usable area is
    already correct while its *shape* statistics are not.
    """
    holes: dict[int, int] = {}
    for prop in regionprops(labels):
        if prop.label == 0:
            continue
        holes[int(prop.label)] = int(max(0, 1 - int(prop.euler_number)))
    return holes


# ── 2. Degeneracy and colouring ───────────────────────────────

def degeneracy_ordering(graph_obj: RegionGraph) -> tuple[list[int], int]:
    """Order vertices by repeatedly removing one of minimum remaining degree.

    The largest minimum-degree encountered is the graph's degeneracy, and
    greedy colouring in the reverse of this order needs at most degeneracy + 1
    colours. Planar graphs are 5-degenerate — every planar graph has a vertex
    of degree at most 5 — so this bounds the schedule at six rounds before the
    Four Colour Theorem is invoked at all.
    """
    remaining = {v: set(nbrs) for v, nbrs in graph_obj.adjacency.items()}
    order: list[int] = []
    degeneracy = 0

    while remaining:
        v = min(remaining, key=lambda x: len(remaining[x]))
        degeneracy = max(degeneracy, len(remaining[v]))
        order.append(v)
        for nb in remaining[v]:
            remaining[nb].discard(v)
        del remaining[v]

    return order, degeneracy


def colour_graph(graph_obj: RegionGraph) -> dict[int, int]:
    """Greedy colouring in reverse degeneracy order.

    No two adjacent regions receive the same colour. The Four Colour Theorem
    guarantees four suffice for any planar graph, but constructive four-colouring
    is famously involved; greedy is used here because the bound it gives
    (degeneracy + 1, so at most 6) is close enough to be practical and the
    result is verified rather than assumed.
    """
    order, _ = degeneracy_ordering(graph_obj)
    colouring: dict[int, int] = {}
    for v in reversed(order):
        used = {colouring[nb] for nb in graph_obj.adjacency[v] if nb in colouring}
        colour = 0
        while colour in used:
            colour += 1
        colouring[v] = colour
    return colouring


def verify_colouring(graph_obj: RegionGraph, colouring: dict[int, int]) -> bool:
    """Confirm no edge joins two vertices of the same colour."""
    return all(colouring[u] != colouring[v] for u, v in graph_obj.edges)


def work_schedule(
    graph_obj: RegionGraph,
    parcels: list[int],
) -> tuple[dict[int, int], int]:
    """Group parcels into rounds where no two in a round are adjacent.

    The practical reading of graph colouring: a crew working a parcel occupies
    its boundary — access, spoil, equipment swing — so two touching parcels
    cannot be worked at once. Each colour class is therefore a set of parcels
    that *can* run in parallel, and the number of colours is the number of
    sequential rounds the whole plan needs.
    """
    colouring = colour_graph(graph_obj)
    subset = {p: colouring[p] for p in parcels if p in colouring}
    if not subset:
        return {}, 0

    # Re-index so rounds are numbered 1..k with no gaps, since the colours a
    # parcel subset happens to use are rarely contiguous.
    used = sorted(set(subset.values()))
    remap = {c: i + 1 for i, c in enumerate(used)}
    return {p: remap[c] for p, c in subset.items()}, len(used)


# ── 3. Steiner tree: naming what the siting loop approximates ──

def steiner_benchmark(
    cost_surface: np.ndarray,
    network: np.ndarray,
    terminals: list[tuple[int, int]],
) -> dict:
    """Compare the achieved connection cost against the MST-based benchmark.

    Connecting a set of chosen sites to an existing network at minimum total
    cost is the Steiner tree problem in a metric space — NP-hard. The classic
    2-approximation takes the minimum spanning tree over the *metric closure*
    of the terminals: the complete graph whose edge weights are the true
    least-cost path distances between every pair.

    Computing it here is not an optimisation; it is a yardstick. It answers a
    question the siting plan cannot otherwise answer honestly: how much of the
    connection bill is inherent to reaching these locations, and how much is
    the premium paid for choosing sites by desirability rather than cheapness?
    """
    if not terminals:
        return {}

    rows, cols = np.nonzero(network)
    network_seed = list(zip(rows.tolist(), cols.tolist()))
    nodes = ["network"] + list(range(len(terminals)))

    # Pairwise least-cost distances. One MCP solve per terminal plus one from
    # the network gives every pair, since find_costs is single-source-to-all.
    dist: dict[tuple, float] = {}
    mcp = skgraph.MCP_Geometric(cost_surface)
    cum, _ = mcp.find_costs(network_seed)
    for i, t in enumerate(terminals):
        dist[("network", i)] = float(cum[t])

    for i, t in enumerate(terminals):
        mcp_i = skgraph.MCP_Geometric(cost_surface)
        cum_i, _ = mcp_i.find_costs([t])
        for j in range(i + 1, len(terminals)):
            dist[(i, j)] = float(cum_i[terminals[j]])

    def weight(a, b) -> float:
        if (a, b) in dist:
            return dist[(a, b)]
        if (b, a) in dist:
            return dist[(b, a)]
        return float("inf")

    # Prim's algorithm over the metric closure, rooted at the existing network.
    in_tree = {"network"}
    total = 0.0
    edges_used: list[tuple] = []
    while len(in_tree) < len(nodes):
        best = None
        best_w = float("inf")
        for a in in_tree:
            for b in nodes:
                if b in in_tree:
                    continue
                w = weight(a, b)
                if w < best_w:
                    best_w, best = w, (a, b)
        if best is None or not np.isfinite(best_w):
            break
        in_tree.add(best[1])
        edges_used.append(best)
        total += best_w

    return {
        "mst_cost": total,
        "edges": edges_used,
        "terminals": len(terminals),
        "connected": len(in_tree) - 1,
    }


# ── Reporting ─────────────────────────────────────────────────

def format_topology_report(
    planarity: dict,
    degeneracy: int,
    n_colours: int,
    colouring_valid: bool,
    holed_parcels: int,
    steiner: dict | None = None,
    achieved_cost: float | None = None,
) -> str:
    """The discrete structure, stated as checks that passed or failed."""
    p = planarity
    ok = "✅" if p["well_formed"] else "❌"

    lines = [
        "### Discrete structure",
        "",
        f"{ok} **Region adjacency graph: V = {p['V']}, E = {p['E']}, "
        f"F = {p['faces']}** (faces via Euler's formula V − E + F = 2)",
        "",
        "| Invariant | Required | Actual | |",
        "|---|---|---|---|",
    ]

    if p["edge_bound"] is not None:
        mark = "✅" if p["edge_bound_holds"] else "❌"
        lines.append(
            f"| Planar edge bound | E ≤ 3V − 6 = {p['edge_bound']} | "
            f"E = {p['E']} | {mark} |"
        )
    mark = "✅" if p["degree_bound_holds"] else "❌"
    lines.append(
        f"| Mean degree | < 6 | {p['mean_degree']:.2f} | {mark} |"
    )
    mark = "✅" if not p["disconnected_regions"] else "❌"
    lines.append(
        f"| Regions singly connected | all | "
        f"{len(p['disconnected_regions'])} violations | {mark} |"
    )
    mark = "✅" if colouring_valid else "❌"
    lines.append(
        f"| Colouring valid | no adjacent pair shares a colour | verified | {mark} |"
    )

    lines += [
        "",
        "These are **necessary conditions for planarity**, used as a test "
        "oracle rather than a fact to recite. The adjacency graph is planar by "
        "construction *if* the segmentation is well-formed, so a violation is "
        "proof of an upstream bug — most often a single label covering two "
        "disconnected blobs, which quietly invalidates every adjacency-derived "
        "feature.",
        "",
        f"**Degeneracy {degeneracy}** — planar graphs are 5-degenerate, so greedy "
        f"colouring in degeneracy order needs at most {degeneracy + 1} colours, "
        "and the Four Colour Theorem guarantees 4 are enough in principle.",
        "",
        f"**Work schedule: {n_colours} parallel round"
        f"{'s' if n_colours != 1 else ''}.** Adjacent parcels cannot be worked "
        "simultaneously — one crew's access and spoil occupy both — so each "
        "colour class is a set of parcels that can run in parallel. Four Colour "
        "bounds this at 4 rounds regardless of how many parcels there are.",
    ]

    if holed_parcels:
        noun = "parcel contains" if holed_parcels == 1 else "parcels contain"
        lines += [
            "",
            f"**{holed_parcels} {noun} holes** (Euler number < 1) — interior "
            "boundary that cannot be built on and must be worked around.",
        ]

    lines += [
        "",
        "**Complexity note.** E ≤ 3V − 6 means the graph has O(V) edges, not "
        "O(V²), so every downstream graph algorithm is linear in the number of "
        "regions. That is a guarantee from planarity, not an empirical "
        "observation about these particular images.",
    ]

    if steiner and achieved_cost is not None and steiner.get("mst_cost"):
        mst = steiner["mst_cost"]
        premium = achieved_cost - mst
        lines += [
            "",
            "### Steiner tree benchmark",
            "",
            "Connecting chosen sites to an existing network at minimum total "
            "cost is the **Steiner tree problem** in a metric space — NP-hard. "
            "The standard 2-approximation is a minimum spanning tree over the "
            "metric closure of the terminals (every pairwise least-cost path).",
            "",
            f"- MST over the metric closure: **${mst:,.0f}**",
            f"- This plan's connection cost: **${achieved_cost:,.0f}**",
            f"- Premium for choosing by desirability rather than cheapness: "
            f"**${premium:,.0f}**"
            + (f" ({100 * premium / mst:+.0f}%)" if mst else ""),
            "",
            "The MST is a *yardstick, not a target*. It connects the same "
            "terminals as cheaply as the geometry allows while ignoring whether "
            "those parcels are worth building on — which is exactly the tradeoff "
            "a planner should be able to see priced.",
            "",
            "*One caveat on comparability: the MST is computed on the original "
            "cost surface, where reaching another terminal costs the full path "
            "to it, while the plan discounts travel along corridors it has "
            "already built. The two models of corridor reuse are close but not "
            "identical, so read the premium as an indication of scale rather "
            "than an exact figure.*",
        ]

    return "\n".join(lines)
