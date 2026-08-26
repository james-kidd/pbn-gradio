"""
Composable segmentation: backends, and the stages that run after them.

The most important thing this module fixes is a category error baked into the
original code. SLIC, Felzenszwalb and watershed are **segmenters** — they turn
an image into regions, and they are alternatives to one another. RAG merging is
**not** one of them: it is a post-process that takes regions and returns fewer
regions, and it composes with any backend.

Conflating the two had a visible consequence: the old `segment_image` applied
RAG merging only on the SLIC path, so the "RAG Merge Threshold" control did
nothing at all when Felzenszwalb was selected. A control that silently does
nothing is worse than a missing one.

The pipeline is therefore expressed as an explicit sequence:

    backend  →  merge  →  drop small  →  fill gaps  →  relabel

Every stage is optional and independently parameterised, and the result records
which stages actually ran so downstream code can report its own provenance.

**Choosing a backend.** They are not interchangeable in practice — each encodes
a different assumption about what a region *is*:

    slic          Regions should be compact and similarly sized. Enforces that
                  through a spatial term in a 5-D (L,a,b,x,y) clustering, so you
                  get roughly the count you ask for. Best when uniformity is
                  the point — paint-by-number wants evenly paintable areas — and
                  worst for thin or elongated features, which compactness
                  actively fights.

    felzenszwalb  Regions should follow real boundaries, whatever shape that
                  makes them. Graph-based, minimum-spanning-tree formulation
                  with an adaptive predicate, so region sizes vary enormously.
                  Best for land parcels and natural boundaries; the cost is that
                  `scale` controls granularity only indirectly — you cannot ask
                  for a specific number of regions.

    watershed     Regions are catchment basins of the intensity gradient.
                  Strongest boundary adherence of the three, and the natural
                  choice when you have markers (known seed points) to grow from.
                  Grid markers are used here when none are supplied.

    precomputed   Do not segment at all — the regions are already known. This is
                  the backend that matters once real geodata is attached: a
                  cadastral parcel layer *is* the segmentation, and it is
                  authoritative in a way no image-derived guess can be.

**Choosing a merge threshold.** RAG merging collapses adjacent regions whose
mean colours are closer than the threshold. It is how deliberate
over-segmentation becomes a usable number of regions, and the over-segment-then-
merge pattern beats single-pass segmentation at almost any target granularity —
which is why it is available to every backend rather than to one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import ndimage
from skimage import graph, morphology, segmentation
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, watershed


# ── Result ────────────────────────────────────────────────────

@dataclass
class SegmentationResult:
    """Regions plus the provenance of how they were produced."""
    labels: np.ndarray
    backend: str
    stages: list[str] = field(default_factory=list)

    @property
    def n_regions(self) -> int:
        return int(len(np.unique(self.labels[self.labels > 0])))

    def describe(self) -> str:
        return f"{self.backend} → {' → '.join(self.stages)} ({self.n_regions} regions)"


# ── Backends ──────────────────────────────────────────────────

def _slic(img_lab: np.ndarray, *, n_segments: int = 800, compactness: float = 10.0,
          sigma: float = 2.0, **_) -> np.ndarray:
    """Compact superpixels via 5-D (L,a,b,x,y) clustering."""
    return segmentation.slic(
        img_lab, n_segments=int(n_segments), compactness=compactness,
        sigma=sigma, start_label=1,
    )


def _felzenszwalb(img_lab: np.ndarray, *, n_segments: int = 800,
                  min_area: int = 64, sigma: float = 0.5, **_) -> np.ndarray:
    """Graph-based segmentation following real boundaries.

    `scale` is derived from the requested region count only as a convenience —
    the mapping is heuristic, and Felzenszwalb genuinely cannot be asked for an
    exact number of regions. Callers who need a specific count should use SLIC.
    """
    scale = max(50.0, 400.0 - n_segments / 10.0)
    labels = felzenszwalb(img_lab, scale=scale, sigma=sigma, min_size=int(min_area))
    return labels + 1          # felzenszwalb is 0-based; keep 0 free for "unlabelled"


def _watershed(img_lab: np.ndarray, *, n_segments: int = 800,
               compactness: float = 0.001, markers=None, **_) -> np.ndarray:
    """Catchment basins of the gradient, grown from grid markers by default."""
    elevation = sobel(img_lab[..., 0].astype(np.float64))

    if markers is None:
        h, w = elevation.shape
        step = max(4, int(np.sqrt(h * w / max(int(n_segments), 1))))
        marker_img = np.zeros((h, w), dtype=np.int32)
        idx = 1
        for r in range(step // 2, h, step):
            for c in range(step // 2, w, step):
                marker_img[r, c] = idx
                idx += 1
        markers = marker_img

    return watershed(elevation, markers, compactness=compactness)


def _precomputed(img_lab: np.ndarray, *, labels: np.ndarray | None = None, **_) -> np.ndarray:
    """Regions supplied from outside — a rasterised parcel layer, typically.

    The seam where authoritative geodata replaces inference. Everything
    downstream consumes a label array and does not care whether it came from
    clustering an image or from a municipal cadastre; the latter is simply
    correct in a way the former can never be.
    """
    if labels is None:
        raise ValueError(
            "The 'precomputed' backend requires a `labels` array — "
            "typically a rasterised parcel or zoning layer."
        )
    if labels.shape != img_lab.shape[:2]:
        raise ValueError(
            f"Precomputed labels {labels.shape} do not match the image "
            f"{img_lab.shape[:2]}. Rasterise the vector layer to the image grid first."
        )
    return labels.astype(np.int32)


BACKENDS: dict[str, Callable[..., np.ndarray]] = {
    "slic": _slic,
    "felzenszwalb": _felzenszwalb,
    "watershed": _watershed,
    "precomputed": _precomputed,
}

# What each backend assumes a region is, for UI help text and reports.
BACKEND_GUIDANCE: dict[str, str] = {
    "slic": "Compact, similarly-sized regions; honours a requested count. "
            "Best for paint-by-number. Fights thin and elongated features.",
    "felzenszwalb": "Follows real boundaries; region sizes vary widely. "
                    "Best for land parcels. Cannot target an exact count.",
    "watershed": "Strongest boundary adherence; grows from markers. "
                 "Best when seed points are known.",
    "precomputed": "No segmentation — regions supplied from an authoritative "
                   "layer such as a cadastral parcel map.",
}

INTERCHANGEABLE = ("slic", "felzenszwalb", "watershed")


# ── Post-processing stages, each independent of the backend ───

def merge_similar(img_lab: np.ndarray, labels: np.ndarray, threshold: float) -> np.ndarray:
    """Collapse adjacent regions whose mean LAB colours are closer than `threshold`.

    This is the stage the original code applied only after SLIC. It has nothing
    to do with SLIC — it consumes a label array and returns a label array, so it
    composes with every backend, and now does.
    """
    if threshold is None or threshold <= 0:
        return labels
    rag = graph.rag_mean_color(img_lab, labels, mode="distance")
    return graph.cut_threshold(labels, rag, thresh=threshold)


def drop_small(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Remove regions below `min_area` pixels, leaving them unlabelled (0)."""
    if min_area <= 0:
        return labels
    try:
        # scikit-image >= 0.26 renamed the parameter; both mean "remove objects
        # up to this size".
        return morphology.remove_small_objects(labels, max_size=min_area)
    except TypeError:
        return morphology.remove_small_objects(labels, min_size=min_area)


def fill_gaps(labels: np.ndarray) -> np.ndarray:
    """Absorb unlabelled pixels into the nearest labelled region.

    Whether this stage is wanted depends entirely on the application. A
    paint-by-number template can leave dropped fragments as white space. An area
    or cost calculation cannot: unlabelled pixels are ground that exists but
    belongs to no region, so totals under-report silently. It is also required
    for the planar-dual invariants to mean anything, since a label array with
    holes is not a subdivision of the plane.
    """
    holes = labels == 0
    if not holes.any() or holes.all():
        return labels
    _, indices = ndimage.distance_transform_edt(
        holes, return_distances=True, return_indices=True,
    )
    return labels[tuple(indices)]


# ── The composed pipeline ─────────────────────────────────────

def segment(
    img_lab: np.ndarray,
    backend: str = "slic",
    *,
    merge_threshold: float | None = None,
    min_area: int = 0,
    fill: bool = False,
    **params,
) -> SegmentationResult:
    """Run a backend, then whichever post-processing stages are requested.

    Stages are explicit and independently controllable rather than bundled into
    the backend choice, which is what makes the backends genuinely
    interchangeable: swapping `slic` for `felzenszwalb` changes only how regions
    are found, not what happens to them afterwards.
    """
    if backend not in BACKENDS:
        raise ValueError(
            f"Unknown backend {backend!r}. Available: {', '.join(BACKENDS)}"
        )

    labels = BACKENDS[backend](img_lab, min_area=min_area, **params)
    stages: list[str] = []

    if merge_threshold and backend != "precomputed":
        labels = merge_similar(img_lab, labels, merge_threshold)
        stages.append(f"merge<{merge_threshold:g}")

    if min_area > 0 and backend != "precomputed":
        labels = drop_small(labels, min_area)
        stages.append(f"drop<{min_area}px")

    labels, _, _ = segmentation.relabel_sequential(labels)

    if fill:
        labels = fill_gaps(labels)
        stages.append("fill-gaps")

    if not stages:
        stages.append("no post-processing")

    return SegmentationResult(labels=labels, backend=backend, stages=stages)
