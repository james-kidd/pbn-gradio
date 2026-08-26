"""
Region labelling and cost estimation on top of the paint-by-number segmentation.

The paint-by-number pipeline answers "what colour is this region?". This module
asks the follow-up question: "what *is* this region, and what would it cost to
do something with it?"

It reuses the same preprocessing and SLIC/RAG segmentation, then adds:

    1. Feature extraction  — per-region colour, texture, shape and adjacency
    2. Labelling           — transparent scored rules mapping features to a class
    3. Cost estimation     — base rate per class x a workability multiplier

The point of stage 3 is that colour alone does not determine cost. Two parcels
of identical vegetation cost different amounts to clear if one is a single
square block and the other is fourteen ragged slivers wedged against water.
Those *structural* features — size, shape regularity, internal consistency,
what a parcel borders — are what a human doing the work actually prices, and
they are all recoverable from the segmentation we already computed.

Every number this module produces is traceable back to a named feature. Nothing
is learned, so nothing is hidden; see CAVEATS at the bottom of the module for
what that costs us in accuracy.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage
from skimage.measure import regionprops

from pipeline import load_and_resize, preprocess_rgb, segment_image


# ── Class definitions ─────────────────────────────────────────

@dataclass(frozen=True)
class LandClass:
    key: str
    name: str
    colour: tuple[int, int, int]      # RGB for the label map
    base_rate: float                  # $ per hectare, default (user-editable)
    basis: str                        # what the rate is meant to represent
    developable: bool = True          # counted toward workable area?


LAND_CLASSES: tuple[LandClass, ...] = (
    LandClass("dense_veg", "Dense vegetation", (34, 120, 52), 4200,
              "Clearing, stump removal, biomass haul-off"),
    LandClass("sparse_veg", "Sparse vegetation", (140, 190, 80), 1800,
              "Light clearing, grading of thin cover"),
    LandClass("stressed_veg", "Stressed / senescent vegetation", (214, 168, 60), 2400,
              "Soil remediation before the ground is usable"),
    LandClass("bare_soil", "Bare soil", (162, 116, 78), 900,
              "Ready ground; light prep and compaction only"),
    LandClass("water", "Water", (48, 112, 176), 0,
              "Not developable; excluded from workable area", developable=False),
    LandClass("built", "Built structure", (188, 74, 74), 8500,
              "Demolition, disposal, site reinstatement"),
    LandClass("road", "Road / track", (128, 128, 136), 0,
              "Existing access; an asset, not a cost"),
    LandClass("unclear", "Shadow / unclear", (86, 86, 106), 1200,
              "Cannot be read from this imagery; nominal contingency"),
)

CLASS_INDEX = {c.key: i for i, c in enumerate(LAND_CLASSES)}
N_CLASSES = len(LAND_CLASSES)

# Classes that make a neighbouring parcel harder to work (no machine access,
# hard boundary to respect, spoil that cannot be pushed sideways).
OBSTRUCTING = ("water", "built")


# ── Small interpretable response curves ───────────────────────

def _ramp(x: float, lo: float, hi: float) -> float:
    """0 below lo, 1 above hi, linear between. Clipped."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _band(x: float, lo: float, plateau_lo: float, plateau_hi: float, hi: float) -> float:
    """Trapezoid: 1 across [plateau_lo, plateau_hi], falling to 0 at lo and hi."""
    if x <= lo or x >= hi:
        return 0.0
    if x < plateau_lo:
        return _ramp(x, lo, plateau_lo)
    if x > plateau_hi:
        return 1.0 - _ramp(x, plateau_hi, hi)
    return 1.0


# ── 0. Gap filling ────────────────────────────────────────────

def fill_unlabelled(labels: np.ndarray) -> np.ndarray:
    """Absorb unlabelled pixels into their nearest labelled region.

    The shared segmentation drops sub-threshold fragments to label 0. For a
    paint-by-number template that is harmless — they read as white space. For a
    cost estimate it is not: unlabelled pixels are area that exists on the
    ground but appears in no parcel, so the total silently under-reports, and a
    total that under-reports without saying so is worse than a crude one that
    doesn't. Every pixel in the frame has to land in exactly one parcel.
    """
    holes = labels == 0
    if not holes.any() or holes.all():
        return labels
    _, indices = ndimage.distance_transform_edt(
        holes, return_distances=True, return_indices=True,
    )
    return labels[tuple(indices)]


# ── 1. Per-region feature extraction ──────────────────────────

@dataclass
class RegionFeatures:
    """Everything the labeller and the cost model are allowed to look at."""
    label: int
    area_px: int
    centroid: tuple[float, float]

    # Colour (perceptual)
    lab_l: float
    lab_a: float
    lab_b: float
    hue_deg: float
    saturation: float
    exg: float               # excess green, visible-band vegetation proxy

    # Texture
    l_std: float             # internal lightness variation
    edge_density: float      # mean gradient magnitude inside the region

    # Shape
    shape_index: float       # perimeter / (2*sqrt(pi*area)); 1.0 == perfect circle
    elongation: float        # major axis / minor axis
    solidity: float          # area / convex hull area

    # Context — topology is class-independent, so it is available *before*
    # labelling and may inform it. The class fractions below necessarily come
    # after, since they describe what the neighbours turned out to be.
    neighbour_count: int = 0
    obstructed_frac: float = 0.0
    road_frac: float = 0.0

    # Results
    class_key: str = "unclear"
    confidence: float = 0.0
    evidence: list[tuple[str, str, float]] = field(default_factory=list)


def _grouped_stats(labels: np.ndarray, values: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of `values` grouped by integer `labels`."""
    flat_labels = labels.ravel()
    flat_values = values.ravel().astype(np.float64)
    counts = np.bincount(flat_labels, minlength=n).astype(np.float64)
    counts_safe = np.maximum(counts, 1.0)
    sums = np.bincount(flat_labels, weights=flat_values, minlength=n)
    sq_sums = np.bincount(flat_labels, weights=flat_values ** 2, minlength=n)
    means = sums / counts_safe
    variance = np.maximum(sq_sums / counts_safe - means ** 2, 0.0)
    return means, np.sqrt(variance)


def extract_features(
    img_rgb: np.ndarray,
    img_lab: np.ndarray,
    labels: np.ndarray,
) -> list[RegionFeatures]:
    """Compute colour, texture and shape descriptors for every labelled region.

    All the per-pixel aggregation is done with bincount rather than per-region
    masking, so cost is linear in pixels rather than pixels x regions.
    """
    n = int(labels.max()) + 1

    rgb = img_rgb.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # Excess green (2G - R - B) is the standard visible-band vegetation index
    # when no near-infrared channel is available. Normalised to roughly [-1, 1].
    exg_map = (2.0 * g - r - b) / 255.0

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hue_map = hsv[..., 0].astype(np.float32) * 2.0   # OpenCV packs hue into 0-179
    sat_map = hsv[..., 1].astype(np.float32) / 255.0

    grey = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    edge_map = np.sqrt(grad_x ** 2 + grad_y ** 2) / 255.0

    l_mean, l_std = _grouped_stats(labels, img_lab[..., 0], n)
    a_mean, _ = _grouped_stats(labels, img_lab[..., 1], n)
    b_mean, _ = _grouped_stats(labels, img_lab[..., 2], n)
    exg_mean, _ = _grouped_stats(labels, exg_map, n)
    sat_mean, _ = _grouped_stats(labels, sat_map, n)
    edge_mean, _ = _grouped_stats(labels, edge_map, n)

    # Hue is circular, so averaging it directly wraps badly at red. Average the
    # unit vectors instead and take the resulting angle.
    hue_rad = np.deg2rad(hue_map)
    cos_mean, _ = _grouped_stats(labels, np.cos(hue_rad), n)
    sin_mean, _ = _grouped_stats(labels, np.sin(hue_rad), n)
    hue_mean = np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360.0

    features: list[RegionFeatures] = []
    for prop in regionprops(labels):
        i = prop.label
        if i == 0:
            continue
        area = int(prop.area)
        perimeter = float(prop.perimeter) or 1.0
        shape_index = perimeter / (2.0 * np.sqrt(np.pi * max(area, 1)))
        minor = float(getattr(prop, "axis_minor_length", 0.0) or 0.0)
        major = float(getattr(prop, "axis_major_length", 0.0) or 0.0)
        elongation = major / minor if minor > 1e-6 else 1.0

        features.append(RegionFeatures(
            label=i,
            area_px=area,
            centroid=(float(prop.centroid[1]), float(prop.centroid[0])),
            lab_l=float(l_mean[i]),
            lab_a=float(a_mean[i]),
            lab_b=float(b_mean[i]),
            hue_deg=float(hue_mean[i]),
            saturation=float(sat_mean[i]),
            exg=float(exg_mean[i]),
            l_std=float(l_std[i]),
            edge_density=float(edge_mean[i]),
            shape_index=float(shape_index),
            elongation=float(elongation),
            solidity=float(prop.solidity),
        ))
    return features


# ── 2. Labelling ──────────────────────────────────────────────

# Every piece of evidence is tagged with the family of features it draws on.
# This is what makes the contribution of shape and context *measurable* rather
# than asserted: switch a family off, re-run, and read the difference.
FEATURE_FAMILIES: tuple[str, ...] = ("colour", "texture", "shape", "context")
ALL_FAMILIES = frozenset(FEATURE_FAMILIES)


def score_region(
    f: RegionFeatures,
    families: frozenset[str] = ALL_FAMILIES,
) -> dict[str, list[tuple[str, str, float]]]:
    """Score each land class against a region, keeping the evidence itemised.

    Each entry is a (family, human-readable reason, weight) triple. The class
    score is the sum of its weights; keeping them separate is what lets the UI
    explain *why* a parcel was labelled the way it was rather than asserting it,
    and what lets a downstream model consume the reasoning rather than just the
    verdict.

    `families` restricts which feature families may contribute — the mechanism
    behind the ablation below.
    """
    scores: dict[str, list[tuple[str, str, float]]] = {c.key: [] for c in LAND_CLASSES}

    def add(key: str, family: str, reason: str, value: float) -> None:
        if family in families and abs(value) > 1e-3:
            scores[key].append((family, reason, value))

    green_hue = _band(f.hue_deg, 68, 88, 165, 190)
    yellow_hue = _band(f.hue_deg, 30, 42, 68, 85)
    brown_hue = _band(f.hue_deg, 0, 12, 42, 60)
    blue_hue = _band(f.hue_deg, 165, 190, 250, 275)
    warm = _ramp(f.lab_b, 2, 22)
    low_sat = 1.0 - _ramp(f.saturation, 0.10, 0.32)
    smooth = 1.0 - _ramp(f.edge_density, 0.05, 0.22)
    uniform = 1.0 - _ramp(f.l_std, 3, 16)

    # Dense vegetation: strongly green, saturated, textured canopy.
    add("dense_veg", "colour", "strong excess-green response", 2.4 * _ramp(f.exg, 0.03, 0.16))
    add("dense_veg", "colour", "green hue", 1.0 * green_hue)
    add("dense_veg", "colour", "saturated colour", 0.6 * _ramp(f.saturation, 0.18, 0.45))
    add("dense_veg", "texture", "canopy texture", 0.5 * _ramp(f.edge_density, 0.08, 0.25))
    # Excess-green cannot tell healthy green from senescent yellow — both are
    # green-dominant in RGB. Hue can, so it has to carry the discrimination.
    add("dense_veg", "colour", "counts against: yellow cast", -2.0 * yellow_hue)

    # Sparse vegetation: green but weakly, usually with soil showing through.
    add("sparse_veg", "colour", "weak excess-green response", 2.0 * _band(f.exg, -0.02, 0.01, 0.06, 0.13))
    add("sparse_veg", "colour", "green hue", 0.7 * green_hue)
    add("sparse_veg", "texture", "mixed cover (high internal variation)", 0.6 * _ramp(f.l_std, 5, 18))

    # Stressed / senescent: yellowing, warm, still some chlorophyll signal.
    add("stressed_veg", "colour", "yellow hue", 1.8 * yellow_hue)
    add("stressed_veg", "colour", "warm (high LAB b*)", 1.0 * _ramp(f.lab_b, 12, 34))
    add("stressed_veg", "colour", "chlorophyll fading but present", 0.8 * _ramp(f.exg, -0.06, 0.04))
    add("stressed_veg", "colour", "bright", 0.4 * _ramp(f.lab_l, 45, 78))

    # Bare soil: warm, red-brown, low green, comparatively smooth.
    add("bare_soil", "colour", "brown/red hue", 1.6 * brown_hue)
    add("bare_soil", "colour", "warm and red (positive a* and b*)", 1.0 * min(_ramp(f.lab_a, 2, 16), warm))
    add("bare_soil", "colour", "no vegetation signal", 1.2 * _ramp(-f.exg, 0.0, 0.08))
    add("bare_soil", "texture", "smooth surface", 0.5 * smooth)

    # Water: dark, flat, unsaturated or blue. Flatness is the strongest cue —
    # but flatness alone also describes a ploughed field, so it needs a colour
    # precondition. Warm chroma rules water out however smooth the surface is.
    warm_chroma = min(_ramp(f.lab_b, 6, 24), _ramp(f.saturation, 0.15, 0.40))
    add("water", "texture", "very low internal variation", 1.6 * uniform)
    add("water", "colour", "dark", 1.0 * _ramp(-f.lab_l, -42, -18))
    add("water", "colour", "blue hue", 1.0 * blue_hue)
    add("water", "texture", "smooth surface", 0.8 * smooth)
    add("water", "colour", "no vegetation signal", 0.6 * _ramp(-f.exg, 0.0, 0.05))
    add("water", "colour", "counts against: warm colour cast", -2.2 * warm_chroma)

    # Built structure: grey, bright, hard edges, compact and convex.
    add("built", "colour", "desaturated (grey)", 1.4 * low_sat)
    add("built", "texture", "hard edges", 1.2 * _ramp(f.edge_density, 0.12, 0.35))
    add("built", "colour", "bright", 0.8 * _ramp(f.lab_l, 48, 85))
    add("built", "shape", "compact, convex footprint", 0.8 * _ramp(f.solidity, 0.82, 0.97))

    # Road / track: the class that colour alone cannot separate from "built".
    # Elongation is what distinguishes it, which is exactly the point.
    add("road", "shape", "elongated, ribbon-like shape", 2.0 * _ramp(f.elongation, 2.6, 7.0))
    add("road", "colour", "desaturated (grey)", 1.0 * low_sat)
    add("road", "colour", "mid lightness", 0.6 * _band(f.lab_l, 22, 34, 68, 84))
    add("road", "texture", "smooth surface", 0.4 * smooth)
    # A track is a corridor: it runs between things rather than sitting inside
    # one thing. High contact with many distinct neighbours is corroboration
    # that colour and shape cannot supply on their own.
    add("road", "context", "borders many distinct parcels (corridor)",
        0.8 * _ramp(f.neighbour_count, 3.0, 9.0))

    # Unclear: deep shadow, where no other class can be trusted.
    add("unclear", "colour", "deep shadow", 2.0 * _ramp(-f.lab_l, -26, -8))
    add("unclear", "colour", "no usable colour information", 0.8 * low_sat * _ramp(-f.lab_l, -34, -14))

    return scores


def classify_regions(
    features: list[RegionFeatures],
    families: frozenset[str] = ALL_FAMILIES,
) -> None:
    """Assign a class and a confidence to every region, in place."""
    for f in features:
        scores = score_region(f, families)
        totals = {key: sum(v for _, _, v in items) for key, items in scores.items()}
        ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
        best_key, best_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

        # Confidence combines two honest signals: how strong the winning
        # evidence is in absolute terms, and how far clear of second place it
        # is. A region can only be called confident if both hold.
        strength = _ramp(best_score, 0.6, 2.4)
        margin = _ramp(best_score - runner_up, 0.1, 1.2)
        f.confidence = float(np.sqrt(max(strength, 1e-6) * max(margin, 1e-6)))

        if best_score < 0.5:
            f.class_key = "unclear"
            f.evidence = [("colour", "no class scored above the evidence floor", best_score)]
        else:
            f.class_key = best_key
            f.evidence = sorted(scores[best_key], key=lambda e: e[2], reverse=True)[:4]


# ── Adjacency: what each parcel borders ───────────────────────

def _boundary_pairs(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every 4-connected pair of touching region labels, both directions."""
    a_parts, b_parts = [], []
    for axis in (0, 1):
        a = labels.take(np.arange(labels.shape[axis] - 1), axis=axis)
        b = labels.take(np.arange(1, labels.shape[axis]), axis=axis)
        differs = a != b
        a_parts.append(a[differs])
        b_parts.append(b[differs])
    left = np.concatenate(a_parts)
    right = np.concatenate(b_parts)
    return np.concatenate([left, right]), np.concatenate([right, left])


def compute_topology(labels: np.ndarray, features: list[RegionFeatures]) -> None:
    """Count how many distinct regions each region touches.

    This is pure graph structure — it says nothing about what the neighbours
    *are*, so unlike the class fractions below it is available before labelling
    and is allowed to inform it. A corridor touches many parcels; a field
    interior touches few. Colour cannot see that distinction at all.
    """
    src, dst = _boundary_pairs(labels)
    n = int(labels.max()) + 1
    seen = np.zeros(n, dtype=np.int64)
    order = np.lexsort((dst, src))
    src_s, dst_s = src[order], dst[order]
    if src_s.size:
        new_pair = np.ones(src_s.size, dtype=bool)
        new_pair[1:] = (src_s[1:] != src_s[:-1]) | (dst_s[1:] != dst_s[:-1])
        np.add.at(seen, src_s[new_pair], 1)

    for f in features:
        f.neighbour_count = int(seen[f.label])


def compute_adjacency(
    labels: np.ndarray,
    features: list[RegionFeatures],
) -> None:
    """Record, per region, what fraction of its border touches each class.

    Bordering water or a standing structure means no machine access from that
    side and a hard edge to work to. Bordering an existing road means the
    opposite. Neither is visible in the region's own colour — it only exists in
    the relationship between regions.
    """
    n = int(labels.max()) + 1

    class_of = np.full(n, CLASS_INDEX["unclear"], dtype=np.int32)
    for f in features:
        class_of[f.label] = CLASS_INDEX[f.class_key]

    src, dst = _boundary_pairs(labels)
    contacts = np.zeros((n, N_CLASSES), dtype=np.float64)
    np.add.at(contacts, (src, class_of[dst]), 1.0)

    obstructing_cols = [CLASS_INDEX[k] for k in OBSTRUCTING]
    road_col = CLASS_INDEX["road"]

    for f in features:
        total = contacts[f.label].sum()
        if total < 1.0:
            continue
        f.obstructed_frac = float(contacts[f.label, obstructing_cols].sum() / total)
        f.road_frac = float(contacts[f.label, road_col] / total)


# ── 3. Cost estimation ────────────────────────────────────────

# Weights for the workability multiplier. Each one answers "how much harder is
# this parcel to work than an idealised square hectare of the same material?"
FRAGMENTATION_WEIGHT = 0.60
SHAPE_WEIGHT = 0.45
HETEROGENEITY_WEIGHT = 0.30
ACCESS_WEIGHT = 0.40
ROAD_CREDIT = 0.25
MULTIPLIER_CAP = 2.5


@dataclass
class RegionCost:
    label: int
    class_key: str
    area_ha: float
    base_cost: float
    multiplier: float
    total_cost: float
    drivers: list[tuple[str, str, float]]


def workability_multiplier(
    f: RegionFeatures,
    area_ha: float,
    families: frozenset[str] = ALL_FAMILIES,
) -> tuple[float, list[tuple[str, str, float]]]:
    """How much more than the base rate this parcel costs, and why.

    Returns the multiplier and the itemised surcharges that produced it, each
    tagged with the feature family it came from. This is the part of the
    estimate that has nothing to do with what the parcel is made of and
    everything to do with how awkward it is to actually work — which is to say,
    it is almost entirely shape and context.
    """
    drivers: list[tuple[str, str, float]] = []

    def add(family: str, reason: str, value: float) -> None:
        if family in families and abs(value) > 0.01:
            drivers.append((family, reason, value))

    # Small parcels lose to fixed costs: mobilising equipment to a tenth of a
    # hectare costs nearly what mobilising to a full one does.
    fragmentation = 1.0 - _ramp(area_ha, 0.05, 1.20)
    add("shape", "small parcel (fixed costs spread thin)",
        FRAGMENTATION_WEIGHT * fragmentation)

    # Ragged boundaries mean more edge work per unit of area cleared.
    raggedness = _ramp(f.shape_index, 1.15, 2.60)
    add("shape", "irregular boundary (more edge work)", SHAPE_WEIGHT * raggedness)

    # Internally inconsistent parcels cannot be worked with one method.
    mixed = _ramp(f.l_std, 4.0, 18.0)
    add("texture", "inconsistent surface (mixed method)", HETEROGENEITY_WEIGHT * mixed)

    # Hemmed in by water or structures: no approach from those sides.
    add("context", "restricted access (water/structure border)",
        ACCESS_WEIGHT * f.obstructed_frac)

    # Fronting an existing road is worth real money.
    add("context", "road frontage (existing access)", -ROAD_CREDIT * f.road_frac)

    multiplier = 1.0 + sum(v for _, _, v in drivers)
    return float(np.clip(multiplier, 1.0, MULTIPLIER_CAP)), drivers


def estimate_costs(
    features: list[RegionFeatures],
    m_per_px: float,
    rates: dict[str, float],
    families: frozenset[str] = ALL_FAMILIES,
) -> list[RegionCost]:
    """Price every region: area x class base rate x workability multiplier."""
    ha_per_px = (m_per_px ** 2) / 10_000.0
    costs: list[RegionCost] = []
    for f in features:
        area_ha = f.area_px * ha_per_px
        multiplier, drivers = workability_multiplier(f, area_ha, families)
        base_rate = rates.get(f.class_key, 0.0)
        base_cost = area_ha * base_rate
        costs.append(RegionCost(
            label=f.label,
            class_key=f.class_key,
            area_ha=area_ha,
            base_cost=base_cost,
            multiplier=multiplier,
            total_cost=base_cost * multiplier,
            drivers=drivers,
        ))
    return costs


# ── Rendering ─────────────────────────────────────────────────

def render_label_map(
    img_rgb: np.ndarray,
    labels: np.ndarray,
    features: list[RegionFeatures],
    opacity: float = 0.62,
) -> np.ndarray:
    """Paint each region in its class colour, blended over the original image."""
    n = int(labels.max()) + 1
    lut = np.zeros((n, 3), dtype=np.uint8)
    for f in features:
        lut[f.label] = LAND_CLASSES[CLASS_INDEX[f.class_key]].colour

    class_img = lut[labels]
    blended = (img_rgb * (1.0 - opacity) + class_img * opacity).astype(np.uint8)

    boundaries = np.zeros(labels.shape, dtype=bool)
    boundaries[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundaries[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    blended[boundaries] = (30, 30, 30)
    return blended


def render_confidence_map(
    labels: np.ndarray,
    features: list[RegionFeatures],
) -> np.ndarray:
    """Greyscale map of labelling confidence — dark regions are guesses.

    This is deliberately a separate output. An estimate that hides its own weak
    spots is worse than no estimate, because it invites trust it hasn't earned.
    """
    n = int(labels.max()) + 1
    lut = np.zeros(n, dtype=np.float32)
    for f in features:
        lut[f.label] = f.confidence

    conf = lut[labels]
    heat = cv2.applyColorMap((conf * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def render_legend(
    features: list[RegionFeatures],
    costs: list[RegionCost],
    total_area_ha: float,
) -> np.ndarray:
    """Legend strip: class colour, name, and the share of area it holds."""
    present = {}
    for f, c in zip(features, costs):
        entry = present.setdefault(f.class_key, [0.0, 0])
        entry[0] += c.area_ha
        entry[1] += 1

    ordered = [c for c in LAND_CLASSES if c.key in present]
    row_h, width = 34, 640
    img = np.full((max(len(ordered), 1) * row_h + 8, width, 3), 250, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, cls in enumerate(ordered):
        area_ha, count = present[cls.key]
        share = 100.0 * area_ha / total_area_ha if total_area_ha > 0 else 0.0
        y0 = 4 + i * row_h
        cv2.rectangle(img, (8, y0 + 6), (34, y0 + 26), cls.colour, -1)
        cv2.rectangle(img, (8, y0 + 6), (34, y0 + 26), (120, 120, 120), 1)
        cv2.putText(img, cls.name, (46, y0 + 21), font, 0.48, (25, 25, 25), 1, cv2.LINE_AA)
        summary = f"{share:5.1f}%   {area_ha:7.2f} ha   {count:4d} parcels"
        cv2.putText(img, summary, (360, y0 + 21), font, 0.45, (70, 70, 70), 1, cv2.LINE_AA)

    return img


# ── Reporting ─────────────────────────────────────────────────

def build_cost_report(
    features: list[RegionFeatures],
    costs: list[RegionCost],
    m_per_px: float,
    rates: dict[str, float],
    frame_area_ha: float,
) -> str:
    """Markdown breakdown of where the money goes and how much to trust it."""
    total_area = sum(c.area_ha for c in costs)
    workable = sum(
        c.area_ha for c in costs
        if LAND_CLASSES[CLASS_INDEX[c.class_key]].developable
    )
    total_base = sum(c.base_cost for c in costs)
    total_cost = sum(c.total_cost for c in costs)
    surcharge = total_cost - total_base

    by_class: dict[str, dict[str, float]] = {}
    for c in costs:
        entry = by_class.setdefault(c.class_key, {"area": 0.0, "base": 0.0, "total": 0.0, "n": 0})
        entry["area"] += c.area_ha
        entry["base"] += c.base_cost
        entry["total"] += c.total_cost
        entry["n"] += 1

    lines = [
        "### Cost estimate",
        "",
        f"**Total: ${total_cost:,.0f}**  ·  {total_area:.2f} ha surveyed  ·  "
        f"{workable:.2f} ha workable",
        "",
        f"Of that total, **${total_base:,.0f}** is the material itself and "
        f"**${surcharge:,.0f}** ({100 * surcharge / total_cost if total_cost else 0:.0f}%) "
        "is the surcharge for how awkward the parcels are to work.",
        "",
        "| Class | Parcels | Area (ha) | Rate ($/ha) | Base | Workability | Total |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for cls in LAND_CLASSES:
        if cls.key not in by_class:
            continue
        e = by_class[cls.key]
        mult = e["total"] / e["base"] if e["base"] > 0 else 1.0
        lines.append(
            f"| {cls.name} | {int(e['n'])} | {e['area']:.2f} | "
            f"{rates.get(cls.key, 0):,.0f} | ${e['base']:,.0f} | "
            f"x{mult:.2f} | **${e['total']:,.0f}** |"
        )

    mean_conf = float(np.mean([f.confidence for f in features])) if features else 0.0
    low_conf = sum(1 for f in features if f.confidence < 0.45)
    low_conf_cost = sum(
        c.total_cost for f, c in zip(features, costs) if f.confidence < 0.45
    )
    # Area matters here as much as cost. A parcel wrongly called water carries
    # $0, so a cost-only trust figure would report it as nothing at risk — when
    # what actually happened is that workable land fell out of the estimate.
    low_conf_area = sum(
        c.area_ha for f, c in zip(features, costs) if f.confidence < 0.45
    )

    lines += [
        "",
        "### How much to trust this",
        "",
        f"- Mean labelling confidence: **{mean_conf:.0%}**",
        f"- Low-confidence parcels (< 45%): **{low_conf}** of {len(features)}, "
        f"carrying **${low_conf_cost:,.0f}** ({100 * low_conf_cost / total_cost if total_cost else 0:.0f}%) "
        f"of the estimate across **{low_conf_area:.2f} ha** "
        f"({100 * low_conf_area / total_area if total_area else 0:.0f}% of surveyed area)",
        f"- Ground resolution assumed: **{m_per_px:.3f} m/px** — area, and therefore "
        "every dollar figure, scales with the square of this number",
        f"- Frame coverage: **{100 * total_area / frame_area_ha if frame_area_ha else 0:.1f}%** "
        f"of the {frame_area_ha:.2f} ha in view is accounted for by a parcel",
        "",
        "Treat the low-confidence share as the part of the estimate that needs a "
        "human to look at the imagery before anyone commits to it.",
    ]
    return "\n".join(lines)


def build_region_table(
    features: list[RegionFeatures],
    costs: list[RegionCost],
    limit: int = 25,
) -> list[list]:
    """The most expensive parcels, with the reasoning attached to each row."""
    paired = sorted(zip(features, costs), key=lambda fc: fc[1].total_cost, reverse=True)
    rows = []
    for f, c in paired[:limit]:
        cls = LAND_CLASSES[CLASS_INDEX[f.class_key]]
        why = "; ".join(
            f"{reason} [{family}]" for family, reason, _ in f.evidence[:2]
        ) or "—"
        drivers = "; ".join(
            f"{reason} ({'+' if v > 0 else ''}{v:.0%}) [{family}]"
            for family, reason, v in c.drivers[:2]
        ) or "none — clean parcel"
        rows.append([
            f.label,
            cls.name,
            f"{f.confidence:.0%}",
            f"{c.area_ha:.3f}",
            f"x{c.multiplier:.2f}",
            f"${c.total_cost:,.0f}",
            why,
            drivers,
        ])
    return rows


REGION_TABLE_HEADERS = [
    "Region", "Label", "Confidence", "Area (ha)",
    "Workability", "Cost", "Why this label", "Cost drivers",
]


# ── Feature export for downstream models ──────────────────────

# The families each exported column belongs to. A model consuming this table
# knows not just the value but what kind of evidence it is, which is what lets
# an attribution over these columns be read as an explanation rather than a
# ranking of opaque indices.
FEATURE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("area_px",        "shape",   "region size in pixels"),
    ("area_ha",        "shape",   "region size on the ground"),
    ("lab_l",          "colour",  "CIELAB lightness"),
    ("lab_a",          "colour",  "CIELAB green-red axis"),
    ("lab_b",          "colour",  "CIELAB blue-yellow axis"),
    ("hue_deg",        "colour",  "circular-mean hue"),
    ("saturation",     "colour",  "HSV saturation"),
    ("exg",            "colour",  "excess green (visible-band vegetation proxy)"),
    ("l_std",          "texture", "internal lightness variation"),
    ("edge_density",   "texture", "mean gradient magnitude"),
    ("shape_index",    "shape",   "perimeter / (2*sqrt(pi*area)); 1.0 is a circle"),
    ("elongation",     "shape",   "major axis / minor axis"),
    ("solidity",       "shape",   "area / convex hull area"),
    ("neighbour_count", "context", "number of distinct regions touched"),
    ("obstructed_frac", "context", "share of border against water or structure"),
    ("road_frac",      "context",  "share of border against road"),
)


def export_features(
    features: list[RegionFeatures],
    costs: list[RegionCost],
) -> list[dict]:
    """Per-region feature records for a downstream model to train on.

    Segmentation is what makes this possible: a region is a unit big enough to
    carry shape and context, which a pixel is not. A per-pixel model can only
    ever see colour and a local neighbourhood; once pixels are grouped, "how big
    is this thing, what shape is it, what does it border" become askable — and
    those are the questions whose answers a human can check.
    """
    records = []
    for f, c in zip(features, costs):
        values = {name: getattr(f, name, None) for name, _, _ in FEATURE_COLUMNS}
        values["area_ha"] = c.area_ha
        records.append({
            "region_id": f.label,
            "centroid_x": round(f.centroid[0], 1),
            "centroid_y": round(f.centroid[1], 1),
            **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in values.items()},
            "predicted_class": f.class_key,
            "confidence": round(f.confidence, 4),
            "evidence": [
                {"family": fam, "reason": reason, "weight": round(w, 4)}
                for fam, reason, w in f.evidence
            ],
            "estimated_cost": round(c.total_cost, 2),
            "workability_multiplier": round(c.multiplier, 4),
            "cost_drivers": [
                {"family": fam, "reason": reason, "delta": round(v, 4)}
                for fam, reason, v in c.drivers
            ],
        })
    return records


def write_feature_export(records: list[dict], directory: str) -> tuple[str, str]:
    """Write the feature table as CSV (flat) and JSON (with reasoning)."""
    import csv
    import json
    import os

    os.makedirs(directory, exist_ok=True)
    csv_path = os.path.join(directory, "region_features.csv")
    json_path = os.path.join(directory, "region_features.json")

    flat_keys = [
        "region_id", "centroid_x", "centroid_y",
        *[name for name, _, _ in FEATURE_COLUMNS],
        "predicted_class", "confidence",
        "workability_multiplier", "estimated_cost",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=flat_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    with open(json_path, "w") as fh:
        json.dump({
            "schema": [
                {"column": name, "family": family, "description": desc}
                for name, family, desc in FEATURE_COLUMNS
            ],
            "regions": records,
        }, fh, indent=2)

    return csv_path, json_path


# ── Ablation: does shape and context actually add anything? ────

ABLATIONS: tuple[tuple[str, frozenset[str]], ...] = (
    ("Colour only", frozenset({"colour"})),
    ("+ texture", frozenset({"colour", "texture"})),
    ("+ shape", frozenset({"colour", "texture", "shape"})),
    ("+ context (full)", ALL_FAMILIES),
)


def run_ablation(
    labels: np.ndarray,
    features: list[RegionFeatures],
    m_per_px: float,
    rates: dict[str, float],
) -> list[dict]:
    """Re-run labelling and costing with feature families progressively enabled.

    The claim "shape and context add information" is cheap to assert and easy to
    test, so it should be tested. Each row re-runs the full pipeline with one
    more family available and reports what changed. Note this measures
    *information contribution*, not accuracy — without ground truth we can show
    that the extra families change the answer substantially, not that they
    change it correctly.
    """
    import copy

    results = []
    baseline_classes: dict[int, str] | None = None

    for name, families in ABLATIONS:
        trial = copy.deepcopy(features)
        classify_regions(trial, families)
        if "context" in families:
            compute_adjacency(labels, trial)
        costs = estimate_costs(trial, m_per_px, rates, families)

        classes = {f.label: f.class_key for f in trial}
        total_area = sum(c.area_ha for c in costs)
        changed_area = 0.0
        if baseline_classes is not None:
            changed_area = sum(
                c.area_ha for f, c in zip(trial, costs)
                if baseline_classes.get(f.label) != f.class_key
            )

        results.append({
            "stage": name,
            "total_cost": sum(c.total_cost for c in costs),
            "mean_confidence": float(np.mean([f.confidence for f in trial])),
            "distinct_classes": len({f.class_key for f in trial}),
            "changed_area_pct": 100.0 * changed_area / total_area if total_area else 0.0,
        })
        baseline_classes = classes

    return results


def format_ablation(results: list[dict]) -> str:
    """Markdown table of the ablation, with the headline delta called out."""
    if not results:
        return ""
    lines = [
        "### Does shape and context actually add anything?",
        "",
        "Each row re-runs labelling and costing with one more feature family "
        "available. *Area relabelled* is the share of ground that changed class "
        "when that family was switched on.",
        "",
        "| Features available | Total cost | Mean confidence | Classes found | Area relabelled |",
        "|---|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(results):
        relabelled = "—" if i == 0 else f"{r['changed_area_pct']:.1f}%"
        lines.append(
            f"| {r['stage']} | ${r['total_cost']:,.0f} | {r['mean_confidence']:.0%} | "
            f"{r['distinct_classes']} | {relabelled} |"
        )

    first, last = results[0], results[-1]
    delta = last["total_cost"] - first["total_cost"]
    pct = 100.0 * delta / first["total_cost"] if first["total_cost"] else 0.0

    # Which family actually moved the number? Averaging "shape and context"
    # into one claim would hide the case where one of them does nothing, and
    # that case is worth knowing about — it tells you where to spend effort.
    steps = [
        (results[i]["stage"], results[i]["total_cost"] - results[i - 1]["total_cost"])
        for i in range(1, len(results))
    ]
    biggest, biggest_delta = max(steps, key=lambda s: abs(s[1]))
    smallest, smallest_delta = min(steps, key=lambda s: abs(s[1]))

    lines += [
        "",
        f"Colour alone puts the estimate at **${first['total_cost']:,.0f}**; the full "
        f"feature set puts it at **${last['total_cost']:,.0f}** — a **{pct:+.0f}%** "
        "difference, on identical pixels and identical base rates.",
        "",
        f"The single largest contribution is **{biggest}** at "
        f"${abs(biggest_delta):,.0f}. The smallest is **{smallest}** at "
        f"${abs(smallest_delta):,.0f}"
        + (
            " — near-zero here, which is a fact about this scene rather than a "
            "defect. Context earns its keep on fragmented ground with real water "
            "and structure boundaries; a scene without much of that has little "
            "for it to find."
            if "context" in smallest and abs(smallest_delta) < 0.02 * first["total_cost"]
            else "."
        ),
        "",
        "This measures how much the extra families *change* the answer, not "
        "whether they make it correct. Without ground truth the honest claim is "
        "that these features carry decision-relevant information a colour-only "
        "model structurally cannot access — not that this number is right.",
    ]
    return "\n".join(lines)


# ── Rate table plumbing ───────────────────────────────────────

RATE_TABLE_HEADERS = ["Class", "Base rate ($/ha)", "What the rate covers"]


def default_rate_table() -> list[list]:
    return [[c.name, c.base_rate, c.basis] for c in LAND_CLASSES]


def parse_rate_table(table) -> dict[str, float]:
    """Read the user-edited rate table back into a {class_key: rate} mapping.

    Falls back to the default for any row that has been mangled, so a bad edit
    degrades one rate rather than breaking the run.
    """
    rates = {c.key: float(c.base_rate) for c in LAND_CLASSES}
    if table is None:
        return rates

    rows = table.values.tolist() if hasattr(table, "values") else list(table)
    by_name = {c.name: c.key for c in LAND_CLASSES}
    for row in rows:
        if len(row) < 2:
            continue
        key = by_name.get(str(row[0]).strip())
        if key is None:
            continue
        try:
            rates[key] = max(0.0, float(row[1]))
        except (TypeError, ValueError):
            continue
    return rates


# ── Full pipeline ─────────────────────────────────────────────

def run_land_use(
    image: np.ndarray,
    working_size: int,
    ground_resolution: float,
    n_superpixels: int,
    compactness: float,
    sigma: float,
    rag_merge_thresh: float,
    method: str,
    rate_table,
) -> tuple:
    """Segment, label, price, and explain. Returns the UI outputs."""
    import gradio as gr

    if image is None:
        raise gr.Error("Upload an image to begin.")

    original_max_dim = max(image.shape[:2])
    img_rgb = load_and_resize(image, int(working_size))
    h, w = img_rgb.shape[:2]

    # Resizing changes what a pixel covers on the ground. Without this the area
    # figures — and every cost derived from them — would be silently wrong.
    scale = original_max_dim / max(h, w)
    effective_m_per_px = float(ground_resolution) * scale

    img_lab = preprocess_rgb(img_rgb)
    min_area = int(h * w * 0.001)
    labels, _ = segment_image(
        img_lab, int(n_superpixels), compactness, sigma,
        rag_merge_thresh, min_area, method,
    )
    labels = fill_unlabelled(labels)

    features = extract_features(img_rgb, img_lab, labels)
    if not features:
        raise gr.Error("Segmentation produced no regions — try a larger working size.")

    compute_topology(labels, features)
    classify_regions(features)
    compute_adjacency(labels, features)
    # Neighbour classes are now known, so the context-dependent evidence can
    # finally be scored. One refinement pass only: iterating to convergence
    # would be a conditional random field, which is a different project.
    classify_regions(features)

    rates = parse_rate_table(rate_table)
    costs = estimate_costs(features, effective_m_per_px, rates)

    total_area_ha = sum(c.area_ha for c in costs)
    label_map = render_label_map(img_rgb, labels, features)
    confidence_map = render_confidence_map(labels, features)
    legend = render_legend(features, costs, total_area_ha)
    frame_area_ha = h * w * (effective_m_per_px ** 2) / 10_000.0
    report = build_cost_report(features, costs, effective_m_per_px, rates, frame_area_ha)
    table = build_region_table(features, costs)

    ablation = format_ablation(
        run_ablation(labels, features, effective_m_per_px, rates)
    )

    export_dir = tempfile.mkdtemp(prefix="region_features_")
    csv_path, json_path = write_feature_export(
        export_features(features, costs), export_dir,
    )

    return (
        label_map, confidence_map, legend, report, table,
        ablation, [csv_path, json_path],
    )


# ── CAVEATS ───────────────────────────────────────────────────
#
# This is a demonstration of a method, not a validated model. Specifically:
#
# 1. No near-infrared band. Real vegetation analysis uses NDVI, which needs NIR.
#    Excess-green is a visible-band substitute that confuses green paint, algae
#    and shadowed grass with healthy crop.
# 2. No ground truth. The class rules were written by hand from what the
#    features ought to mean, not fitted to labelled imagery. Nothing here has
#    been measured against reality.
# 3. Illustrative rates. The default $/ha figures are plausible placeholders,
#    not quotes. They are editable precisely because they are meant to be
#    replaced with real ones.
# 4. Flat-earth area. Ground resolution is assumed constant across the frame,
#    which holds for nadir satellite imagery and fails for oblique photographs.
#
# The workability multiplier is the piece most likely to survive contact with
# real data, because it depends on geometry rather than on spectra — a small
# ragged parcel hemmed in by water is expensive to work regardless of what the
# labelling says is growing on it.
