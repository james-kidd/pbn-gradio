"""
Incremental siting: given what already exists, where do we add next?

This is the SimCity question, and it is a better fit for this pipeline than
anything else built on it so far. SimCity's actual mechanic is not "classify
the land" — it is "the network you have already built determines what is cheap
to build next." Every placement rewrites the cost of the next placement.

That mechanic maps onto machinery this project already has:

    regions          → the planning unit (a parcel, not a pixel)
    the RAG          → the adjacency structure the plan reasons over
    distance transform → the maximum-setback point inside a parcel
    land classes     → a traversal cost surface

and it adds the one genuinely standard piece that was missing: least-cost path
routing (`skimage.graph.MCP_Geometric`), which is what GIS practice actually
uses for corridor siting. That matters because it makes this the first part of
the project whose core algorithm is not invented here.

Three things are worth being clear about up front:

1. **The routing is exact; the layer beneath it is not.** Given a cost surface,
   the cheapest corridor is computed correctly and is verifiable by eye. Where
   that surface comes from — our unvalidated classifier — is the weak link, and
   it is *swappable*: hand this module a real published road centreline layer
   and the routing is production-grade without touching a line of it.
2. **Slope is missing and slope dominates trenching cost.** A single RGB frame
   has no elevation. This cannot see a ravine between a site and the network.
3. **So this is a screening tool, not a design tool.** It narrows many candidate
   parcels down to a few worth a human's time with real data. That is a real
   job, and it is a smaller claim than "it plans the development."
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage import graph

from land_use import (
    CLASS_INDEX,
    LAND_CLASSES,
    RegionFeatures,
    _ramp,
    classify_regions,
    compute_adjacency,
    compute_topology,
    extract_features,
    fill_unlabelled,
)
import segmenters
from geo import GeoContext
from optimize import budgeted_selection, format_certificate
from pipeline import load_and_resize, preprocess_rgb
from topology import (
    build_region_graph,
    check_planarity,
    colour_graph,
    degeneracy_ordering,
    format_topology_report,
    region_holes,
    steiner_benchmark,
    verify_colouring,
    work_schedule,
)


# ── Traversal cost surface ────────────────────────────────────

# Dollars per metre to run a service corridor (trench, spur road, cable) across
# each class. These are the shape of real numbers, not quotes — the ratios are
# what drive routing, and the ratios are defensible: an existing right-of-way is
# an order of magnitude cheaper than clearing mature canopy, and a water
# crossing needs a structure.
TRAVERSAL_COST: dict[str, float] = {
    "road":         12.0,    # existing right-of-way, no acquisition, no clearing
    "bare_soil":    45.0,    # open trenching
    "sparse_veg":   62.0,
    "stressed_veg": 62.0,
    "dense_veg":   145.0,    # clearing plus haul-off before trenching
    "unclear":      90.0,    # unknown ground priced pessimistically
    "water":       420.0,    # crossing structure
    "built":       900.0,    # only if unavoidable: demolition and reinstatement
}

# Deliberately finite rather than infinite. A building is not a wall of physics
# — it is expensive, and the estimate should say how expensive rather than
# pretending no route exists. It does mean a route will tunnel through
# structures when every alternative is worse, which is worth knowing about.


@dataclass
class Site:
    """A candidate parcel, with the reasoning behind its score kept attached."""
    label: int
    class_key: str
    point: tuple[int, int]        # (row, col) of maximum setback
    area_ha: float
    inscribed_radius_px: float
    connection_cost: float = 0.0   # working value; re-scored for the certificate
    planned_cost: float = 0.0      # cost at the moment it was placed
    baseline_cost: float = 0.0     # cost against the *original* network
    route: list[tuple[int, int]] = field(default_factory=list)
    route_length_m: float = 0.0
    score: float = 0.0
    drivers: list[tuple[str, float]] = field(default_factory=list)
    build_order: int = 0
    work_round: int = 0          # colour class: parcels sharing one can run in parallel
    holes: int = 0               # interior voids that cannot be built on


def build_cost_surface(
    labels: np.ndarray,
    features: list[RegionFeatures],
    m_per_px: float,
    rates: dict[str, float] | None = None,
) -> np.ndarray:
    """Per-pixel corridor cost in dollars per pixel of travel.

    `MCP_Geometric` multiplies these by true geometric distance, so diagonal
    steps are charged sqrt(2) rather than 1 — which is why routes come out as
    plausible corridors instead of staircases.
    """
    rates = rates or TRAVERSAL_COST
    n = int(labels.max()) + 1
    lut = np.full(n, rates["unclear"], dtype=np.float64)
    for f in features:
        lut[f.label] = rates.get(f.class_key, rates["unclear"])
    return lut[labels] * m_per_px


def seed_network(
    labels: np.ndarray,
    features: list[RegionFeatures],
) -> tuple[np.ndarray, str]:
    """Pixels representing infrastructure that already exists.

    Roads are the network. If none were detected we fall back to the frame edge,
    on the assumption that the grid is reachable somewhere off-frame — a
    deliberately weak assumption, reported rather than hidden, because it makes
    every connection cost a lower bound instead of an estimate.
    """
    network = np.zeros(labels.shape, dtype=bool)
    for f in features:
        if f.class_key == "road":
            network |= labels == f.label

    if network.any():
        return network, "detected road network"

    network[0, :] = network[-1, :] = True
    network[:, 0] = network[:, -1] = True
    return network, "no road detected — assuming grid connection off-frame"


# ── Candidate parcels ─────────────────────────────────────────

def candidate_sites(
    labels: np.ndarray,
    features: list[RegionFeatures],
    m_per_px: float,
    min_area_ha: float,
    target_area_ha: float,
) -> list[Site]:
    """Buildable parcels, each anchored at its point of maximum setback.

    The anchor reuses the paint-by-number number-placement trick unchanged: the
    maximum of the distance transform is the centre of the largest circle that
    fits inside the region. There it was the most readable place to print a
    digit; here it is the point furthest from every boundary — which is exactly
    where you would put a structure to maximise setback. Same primitive, and it
    happens to answer a real siting question.
    """
    ha_per_px = (m_per_px ** 2) / 10_000.0
    sites: list[Site] = []

    for f in features:
        cls = LAND_CLASSES[CLASS_INDEX[f.class_key]]
        if not cls.developable or f.class_key in ("road", "built"):
            continue
        area_ha = f.area_px * ha_per_px
        if area_ha < min_area_ha:
            continue

        mask = (labels == f.label).astype(np.uint8)
        # Pad before the transform. Without it OpenCV treats the frame edge as
        # open ground, so any parcel touching the border reports its maximum
        # setback *on* the border at roughly twice the true radius — the one
        # place a facility demonstrably should not go, since nothing is known
        # about what lies beyond the survey.
        padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
        _, max_val, _, max_loc = cv2.minMaxLoc(dist)
        sites.append(Site(
            label=f.label,
            class_key=f.class_key,
            point=(int(max_loc[1]), int(max_loc[0])),
            area_ha=area_ha,
            inscribed_radius_px=float(max_val),
        ))

    return sites


def score_sites(
    sites: list[Site],
    features: list[RegionFeatures],
    target_area_ha: float,
    connection_budget: float,
) -> None:
    """Rank candidates, keeping each contributing term visible.

    Deliberately the same shape as the labelling evidence elsewhere in the
    project: a small number of named terms that sum, so a planner can disagree
    with one of them specifically rather than with the ranking as a whole.
    """
    by_label = {f.label: f for f in features}

    for s in sites:
        f = by_label[s.label]
        terms = [
            ("usable size", 1.4 * _ramp(s.area_ha, 0.0, target_area_ha)),
            ("regular shape (buildable footprint)",
             1.0 * (1.0 - _ramp(f.shape_index, 1.15, 2.6))),
            ("cheap to connect",
             1.6 * (1.0 - _ramp(s.connection_cost, 0.0, connection_budget))),
            ("light site preparation",
             0.8 * (1.0 - _ramp(LAND_CLASSES[CLASS_INDEX[s.class_key]].base_rate,
                                900.0, 4200.0))),
            ("unobstructed approach",
             0.6 * (1.0 - _ramp(f.obstructed_frac, 0.1, 0.6))),
        ]
        s.drivers = [(name, round(v, 3)) for name, v in terms if abs(v) > 0.01]
        s.score = float(sum(v for _, v in terms))


# ── The SimCity loop: each build changes what is cheap next ────

def plan_build_order(
    cost_surface: np.ndarray,
    network: np.ndarray,
    sites: list[Site],
    features: list[RegionFeatures],
    m_per_px: float,
    rounds: int,
    target_area_ha: float,
    connection_budget: float,
    corridor_cost: float,
) -> tuple[list[Site], np.ndarray]:
    """Greedily place `rounds` sites, growing the network after each one.

    This is the whole point of the exercise. After each placement the new
    corridor becomes part of the network at a reduced traversal cost — it is a
    right-of-way now — so the cost surface for every subsequent candidate
    changes. Sequencing is not cosmetic: a parcel that was uneconomic in round
    one can become the obvious choice in round three, purely because of what got
    built in between.

    Each site also records `baseline_cost`, its connection cost against the
    *original* network. The gap between the two is the measurable value of
    having sequenced the build rather than costing each parcel independently.
    """
    working_cost = cost_surface.copy()
    working_network = network.copy()
    remaining = list(sites)
    placed: list[Site] = []

    # Round-zero costs against the untouched network, kept as the counterfactual.
    baseline_mcp = graph.MCP_Geometric(working_cost)
    baseline_cum, _ = baseline_mcp.find_costs(_seed_indices(working_network))
    for s in remaining:
        s.baseline_cost = _finite(baseline_cum[s.point])

    for round_no in range(1, rounds + 1):
        if not remaining:
            break

        mcp = graph.MCP_Geometric(working_cost)
        cum, _ = mcp.find_costs(_seed_indices(working_network))

        for s in remaining:
            s.connection_cost = _finite(cum[s.point])
        score_sites(remaining, features, target_area_ha, connection_budget)

        best = max(remaining, key=lambda s: s.score)
        try:
            route = mcp.traceback(best.point)
        except ValueError:
            route = [best.point]

        best.route = [(int(r), int(c)) for r, c in route]
        best.route_length_m = _route_length_m(best.route, m_per_px)
        best.build_order = round_no
        best.planned_cost = best.connection_cost
        placed.append(best)
        remaining.remove(best)

        # The corridor is now infrastructure: cheap to follow, cheap to tap.
        for r, c in best.route:
            working_network[r, c] = True
            working_cost[r, c] = min(working_cost[r, c], corridor_cost * m_per_px)

    return placed, working_network


def _seed_indices(network: np.ndarray) -> list[tuple[int, int]]:
    rows, cols = np.nonzero(network)
    return list(zip(rows.tolist(), cols.tolist()))


def _finite(value: float) -> float:
    v = float(value)
    return v if np.isfinite(v) else float("inf")


def _route_length_m(route: list[tuple[int, int]], m_per_px: float) -> float:
    if len(route) < 2:
        return 0.0
    pts = np.asarray(route, dtype=np.float64)
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return float(steps.sum() * m_per_px)


# ── Rendering ─────────────────────────────────────────────────

def render_connection_map(
    img_rgb: np.ndarray,
    cum_costs: np.ndarray,
    network: np.ndarray,
) -> np.ndarray:
    """Heat map of what it would cost to connect at every point on the frame.

    This is the single most useful output here, and notably it needs no
    classification to be read: bright means far from infrastructure in
    cost terms, dark means cheap to reach. A planner can sanity-check it
    against the image directly.
    """
    finite = np.isfinite(cum_costs)
    display = np.zeros(cum_costs.shape, dtype=np.float32)
    if finite.any():
        hi = float(np.percentile(cum_costs[finite], 97)) or 1.0
        display[finite] = np.clip(cum_costs[finite] / hi, 0, 1)
        display[~finite] = 1.0

    heat = cv2.applyColorMap((display * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    blended = (img_rgb * 0.35 + heat * 0.65).astype(np.uint8)
    blended[network] = (90, 220, 255)
    return blended


def render_plan(
    img_rgb: np.ndarray,
    labels: np.ndarray,
    network: np.ndarray,
    placed: list[Site],
) -> np.ndarray:
    """The build plan: existing network, proposed corridors, numbered sites."""
    canvas = (img_rgb * 0.55 + 60).clip(0, 255).astype(np.uint8)

    boundaries = np.zeros(labels.shape, dtype=bool)
    boundaries[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundaries[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    canvas[boundaries] = (40, 40, 40)
    canvas[network] = (90, 220, 255)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for s in placed:
        if len(s.route) > 1:
            pts = np.array([[c, r] for r, c in s.route], dtype=np.int32)
            cv2.polylines(canvas, [pts], False, (255, 210, 60), 3, cv2.LINE_AA)

        r, c = s.point
        radius = int(max(6.0, min(s.inscribed_radius_px, 46.0)))
        cv2.circle(canvas, (c, r), radius, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (c, r), 13, (30, 30, 30), -1, cv2.LINE_AA)
        text = str(s.build_order)
        (tw, th), _ = cv2.getTextSize(text, font, 0.55, 2)
        cv2.putText(canvas, text, (c - tw // 2, r + th // 2),
                    font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas


# ── Reporting ─────────────────────────────────────────────────

PLAN_TABLE_HEADERS = [
    "Build", "Parcel", "Class", "Area (ha)", "Holes", "Work round",
    "Corridor (m)", "Connect cost", "If built first", "Saved by sequencing",
    "Top reason",
]


def build_plan_table(placed: list[Site]) -> list[list]:
    rows = []
    for s in placed:
        saved = s.baseline_cost - s.connection_cost
        top = max(s.drivers, key=lambda d: d[1])[0] if s.drivers else "—"
        rows.append([
            s.build_order,
            s.label,
            LAND_CLASSES[CLASS_INDEX[s.class_key]].name,
            f"{s.area_ha:.2f}",
            s.holes or "—",
            s.work_round or "—",
            f"{s.route_length_m:,.0f}",
            f"${s.connection_cost:,.0f}",
            f"${s.baseline_cost:,.0f}",
            f"${saved:,.0f}" if saved > 1 else "—",
            top,
        ])
    return rows


def build_siting_report(
    placed: list[Site],
    n_candidates: int,
    network_note: str,
    m_per_px: float,
) -> str:
    """What the plan costs, and — the actual finding — what sequencing saved."""
    if not placed:
        return (
            "### No viable sites\n\n"
            "No parcel cleared the minimum area on developable land. Lower the "
            "minimum size, or coarsen the segmentation so parcels merge into "
            "larger units."
        )

    total_connect = sum(s.connection_cost for s in placed)
    total_baseline = sum(s.baseline_cost for s in placed)
    saved = total_baseline - total_connect
    total_corridor = sum(s.route_length_m for s in placed)
    total_area = sum(s.area_ha for s in placed)

    lines = [
        "### Build plan",
        "",
        f"**{len(placed)} sites** selected from {n_candidates} candidate parcels · "
        f"{total_area:.2f} ha · {total_corridor:,.0f} m of new corridor · "
        f"**${total_connect:,.0f}** to connect",
        "",
        "### What sequencing is worth",
        "",
    ]

    if saved > 1:
        lines += [
            f"Costed independently — every parcel connecting to the network as it "
            f"stands today — these same sites total **${total_baseline:,.0f}**. Built "
            f"in the order above, where each corridor becomes a right-of-way the "
            f"next build can tap, they total **${total_connect:,.0f}**.",
            "",
            f"**Sequencing saves ${saved:,.0f}** "
            f"({100 * saved / total_baseline if total_baseline else 0:.0f}%). That "
            "difference is the entire argument for planning the order rather than "
            "quoting each parcel on its own.",
        ]
    else:
        lines += [
            "On this frame, sequencing saves nothing measurable: the chosen sites "
            "do not lie along each other's corridors, so no build makes the next "
            "one cheaper. That is a real result rather than a failure — it says "
            "these parcels are independent, and the build order can be driven by "
            "whatever else matters (cash flow, permitting) without a routing "
            "penalty.",
        ]

    lines += [
        "",
        "### Reading this honestly",
        "",
        f"- Network seed: **{network_note}**",
        f"- Ground resolution: **{m_per_px:.3f} m/px** — corridor lengths and "
        "therefore costs scale linearly with it",
        "- **Routing is exact; the surface under it is not.** The cheapest "
        "corridor across a given cost surface is computed correctly and you can "
        "check it by eye. Where that surface comes from is our unvalidated "
        "classifier — which is the swappable part: feed in a published road "
        "centreline layer and the routing is production-grade unchanged.",
        "- **No elevation data.** A single RGB frame cannot see slope, and slope "
        "dominates real trenching cost. A route that looks cheap here may cross "
        "a ravine.",
        "- **No ownership, zoning, easements or grid capacity.** All four can "
        "veto a site this tool ranks first.",
        "",
        "Use it to narrow many parcels to a few worth real diligence — not to "
        "choose one.",
    ]
    return "\n".join(lines)


# ── Full pipeline ─────────────────────────────────────────────

def run_siting(
    image: np.ndarray,
    working_size: int,
    ground_resolution: float,
    n_superpixels: int,
    rag_merge_thresh: float,
    method: str,
    rounds: int,
    min_area_ha: float,
    target_area_ha: float,
    connection_budget: float,
    corridor_cost: float,
) -> tuple:
    """Segment, label, cost the ground, then place sites one at a time."""
    import gradio as gr

    if image is None:
        raise gr.Error("Upload an image to begin.")

    original_max_dim = max(image.shape[:2])
    img_rgb = load_and_resize(image, int(working_size))
    h, w = img_rgb.shape[:2]

    # One place converts pixels to ground truth, including the resize factor.
    # Swap this constructor for `GeoContext.from_affine` and every area, length
    # and dollar below becomes real without touching the code that uses them.
    ctx = GeoContext.from_scale(ground_resolution, original_max_dim / max(h, w))
    m_per_px = ctx.m_per_px

    img_lab = preprocess_rgb(img_rgb)
    seg = segmenters.segment(
        img_lab, backend=method,
        merge_threshold=rag_merge_thresh,
        min_area=int(h * w * 0.001),
        fill=True,
        n_segments=int(n_superpixels), compactness=10.0, sigma=2.0,
    )
    labels = seg.labels

    features = extract_features(img_rgb, img_lab, labels)
    if not features:
        raise gr.Error(
            "Segmentation produced no regions. The frame may be too uniform to "
            "segment — try a lower merge threshold, or a larger working size."
        )
    compute_topology(labels, features)
    classify_regions(features)
    compute_adjacency(labels, features)
    classify_regions(features)

    cost_surface = build_cost_surface(labels, features, m_per_px)
    network, network_note = seed_network(labels, features)

    sites = candidate_sites(
        labels, features, m_per_px, float(min_area_ha), float(target_area_ha),
    )
    placed, grown_network = plan_build_order(
        cost_surface, network, sites, features, m_per_px, int(rounds),
        float(target_area_ha), float(connection_budget), float(corridor_cost),
    )

    # ── Discrete structure ────────────────────────────────────
    # The adjacency graph is the dual of a planar subdivision, so it must
    # satisfy the planar invariants. Checking them here is a regression test on
    # the segmentation, not a formality: a violation means an adjacency-derived
    # number elsewhere in this module means something other than it claims.
    region_graph = build_region_graph(labels)
    planarity = check_planarity(region_graph, labels)
    _, degeneracy = degeneracy_ordering(region_graph)
    colouring = colour_graph(region_graph)
    colouring_valid = verify_colouring(region_graph, colouring)

    holes = region_holes(labels)
    schedule, n_rounds = work_schedule(region_graph, [s.label for s in placed])
    for s in placed:
        s.work_round = schedule.get(s.label, 0)
        s.holes = holes.get(s.label, 0)

    achieved = sum(s.connection_cost for s in placed)
    steiner = steiner_benchmark(
        cost_surface, network, [s.point for s in placed],
    )

    # ── Selection certificate ─────────────────────────────────
    # Re-score every candidate on the *same* basis — connection cost against
    # the original network — so the LP compares like with like. Scores taken
    # mid-plan would each reflect a different network state.
    for s in sites:
        s.connection_cost = s.baseline_cost
    score_sites(sites, features, float(target_area_ha), float(connection_budget))

    site_costs = [
        s.baseline_cost
        + LAND_CLASSES[CLASS_INDEX[s.class_key]].base_rate * s.area_ha
        for s in sites
    ]
    placed_ids = {id(s) for s in placed}
    chosen_idx = [i for i, s in enumerate(sites) if id(s) in placed_ids]

    # The budget is what the plan actually spent, which makes the question
    # "did greedy spend this money well?" rather than the vaguer "was this a
    # good plan?" — a question the LP can answer exactly.
    budget = sum(site_costs[i] for i in chosen_idx)
    certificate = budgeted_selection(
        [max(0.0, s.score) for s in sites], site_costs, budget, chosen_idx,
    )
    certificate_report = format_certificate(
        certificate, [s.label for s in sites], chosen_idx,
    )

    # Restore the planned costs, which the table and report describe.
    for s in placed:
        s.connection_cost = s.planned_cost

    mcp = graph.MCP_Geometric(cost_surface)
    cum, _ = mcp.find_costs(_seed_indices(network))

    connection_map = render_connection_map(img_rgb, cum, network)
    plan_image = render_plan(img_rgb, labels, grown_network, placed)
    report = build_siting_report(placed, len(sites), network_note, m_per_px)
    table = build_plan_table(placed)
    export = _write_plan_export(placed, m_per_px)
    topology_report = format_topology_report(
        planarity, degeneracy, n_rounds, colouring_valid,
        sum(1 for s in placed if s.holes), steiner, achieved,
    )
    provenance = (
        f"*Segmentation: {seg.describe()} · Ground truth: {ctx.describe()}*"
    )

    return (
        connection_map, plan_image, report, table, export,
        topology_report, certificate_report, provenance,
    )


def _write_plan_export(placed: list[Site], m_per_px: float) -> list[str]:
    """Plan as CSV, plus corridors and sites as GeoJSON in pixel coordinates.

    Pixel coordinates, not real ones — there is no georeference in a bare image.
    Written as GeoJSON anyway because the moment a frame *does* carry a geo
    transform, this becomes the format a GIS can open, and the shape of the
    output should not have to change for that.
    """
    import csv
    import json
    import os

    directory = tempfile.mkdtemp(prefix="siting_plan_")
    csv_path = os.path.join(directory, "build_plan.csv")
    geo_path = os.path.join(directory, "build_plan.geojson")

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "build_order", "parcel_id", "class", "area_ha", "holes",
            "work_round", "corridor_m",
            "connection_cost", "cost_if_built_first", "saved_by_sequencing",
            "site_row_px", "site_col_px",
        ])
        for s in placed:
            writer.writerow([
                s.build_order, s.label, s.class_key, round(s.area_ha, 4),
                s.holes, s.work_round,
                round(s.route_length_m, 1), round(s.connection_cost, 2),
                round(s.baseline_cost, 2),
                round(s.baseline_cost - s.connection_cost, 2),
                s.point[0], s.point[1],
            ])

    features_out = []
    for s in placed:
        features_out.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [s.point[1], s.point[0]]},
            "properties": {
                "build_order": s.build_order, "parcel_id": s.label,
                "class": s.class_key, "area_ha": round(s.area_ha, 4),
                "connection_cost": round(s.connection_cost, 2),
                "reasons": [{"driver": d, "weight": v} for d, v in s.drivers],
            },
        })
        if len(s.route) > 1:
            features_out.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[c, r] for r, c in s.route],
                },
                "properties": {
                    "build_order": s.build_order, "parcel_id": s.label,
                    "corridor_m": round(s.route_length_m, 1),
                },
            })

    with open(geo_path, "w") as fh:
        json.dump({
            "type": "FeatureCollection",
            "crs": None,
            "note": (
                "Coordinates are image pixels, not geographic. "
                f"One pixel is {m_per_px:.4f} m on the ground."
            ),
            "features": features_out,
        }, fh, indent=2)

    return [csv_path, geo_path]
