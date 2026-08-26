---
title: Region Segmentation Toolkit
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.14.0"
app_file: app.py
pinned: false
---

# Region Segmentation Toolkit

A deterministic computer vision pipeline that segments images into coherent regions using classical methods — no neural networks required. Two applications share the same segmentation core:

1. **Paint-by-Number** — quantize regions to a palette and render a numbered template.
2. **Land Use & Cost** — label those same regions by what they are, and estimate what they would cost to work.

Built by [James Kidd](https://jameskidd.ca) — B.Sc. Mathematics & Computer Science, McGill University.

---

## Pipeline

### 1. Preprocessing
The input image is resized to a configurable maximum dimension and converted from RGB to **CIELAB color space**. CIELAB is perceptually uniform — equal Euclidean distances in LAB correspond to approximately equal perceived color differences — making it far better suited for segmentation than RGB. A **bilateral filter** then smooths noise while preserving edges, a critical property for clean region boundaries.

### 2. Segmentation — SLIC Superpixels
The core segmentation uses **SLIC (Simple Linear Iterative Clustering)**, introduced by Achanta et al. (2012). SLIC clusters pixels in a joint 5-dimensional space: three LAB color channels plus normalized XY spatial coordinates. By enforcing locality through a spatial weighting term (compactness *m*), it produces compact, uniform superpixels in O(N) time — linear in the number of pixels — making it substantially faster than graph-based alternatives at scale.

> **Why SLIC?** Traditional clustering (e.g., k-means over full images) has no spatial constraint, producing fragmented and non-contiguous regions. SLIC's compactness parameter explicitly controls the trade-off between color adherence and shape regularity, giving us predictable, paintable regions.
>
> See also: [SuperPixelSegmentation_using_SLIC](https://github.com/meysam-safarzadeh/SuperPixelSegmentation_using_SLIC) — a clean implementation demonstrating how SLIC's iterative cluster center updates converge to stable, perceptually coherent superpixels with controllable granularity. The key insight is that initializing cluster centers on a grid and confining search to a 2S×2S neighborhood (where S = sqrt(N/k)) achieves near-global optimality without the computational cost of exhaustive search.

A **Region Adjacency Graph (RAG)** is then constructed over the superpixel labels, and adjacent regions whose mean LAB colors fall within a merge threshold are collapsed. This two-stage approach (over-segment then merge) consistently outperforms single-pass methods for producing clean, artifact-free boundaries.

An alternative **Felzenszwalb** graph-based segmentation is also available, which uses a minimum spanning tree formulation with a scale parameter controlling region granularity.

### 3. Palette Extraction
Region mean colors (in LAB) are fed into **k-means clustering** to extract a palette of *k* representative colors. All regions are then assigned to their nearest palette cluster, reducing the image to a small set of discrete, paintable colors. Clustering in LAB rather than RGB ensures the palette is perceptually balanced — colors that look similar are grouped together, not colors that happen to share raw channel values.

### 4. Region Finalization
A **connected component analysis** (8-connectivity) is applied to the indexed image. This guarantees that each numbered paint region is spatially contiguous — a requirement for physical paint-by-number usability. Fragments below a minimum area threshold are discarded.

### 5. Rendering
Contours are extracted via OpenCV and drawn onto a white canvas. Region numbers are placed at the **distance transform maximum** of each region — the point geometrically farthest from all boundaries, ensuring numbers are readable and not clipped by edges.

---

## Land Use & Cost Estimation

The paint-by-number mode asks *what colour is this region?* This mode asks the follow-up: *what **is** it, and what would it cost to do something with it?*

Same segmentation, two extra stages.

### Labelling

Every region gets a descriptor vector spanning three families of evidence:

| Family | Features |
|---|---|
| Colour | LAB means, circular-mean hue, saturation, excess-green index |
| Texture | internal lightness variance, mean Sobel gradient magnitude |
| Shape | isoperimetric shape index, axis elongation, solidity |

These are scored against eight land classes — dense / sparse / stressed vegetation, bare soil, water, built structure, road, and shadow — using explicit trapezoidal response curves. Nothing is learned, so nothing is hidden: each region reports the named evidence that produced its label, plus a confidence derived from both the absolute strength of the winning evidence and its margin over second place.

Shape is what makes this more than colour thresholding. A road and a rooftop are the same grey; only **elongation** separates them.

### Cost

```
cost = area × base_rate[class] × workability_multiplier
```

The base rate is what the material costs. The multiplier is everything about the *parcel* that colour cannot see:

| Driver | Reasoning |
|---|---|
| Fragmentation | Mobilising equipment to 0.1 ha costs nearly what mobilising to 1 ha does |
| Boundary raggedness | Irregular perimeters mean more edge work per unit area |
| Internal heterogeneity | An inconsistent surface can't be worked with a single method |
| Restricted access | A border of water or structures means no approach from that side |
| Road frontage | Existing access is a credit, not a cost |

Access and frontage are computed from a **region adjacency scan** — they exist only in the relationship *between* regions, not in any region's own pixels. This is the direct answer to "does data beyond colour help optimize cost?" — and it is measurable. On the synthetic test scene the workability surcharge accounts for roughly **30% of the total estimate** (28–36% across segmentation settings), and it is entirely invisible to a colour-only model. That figure is from one constructed scene, not a survey of real imagery; the point it supports is that the effect is large enough to matter, not that 30% is the number.

### Honesty features

An estimate that hides its weak spots is worse than no estimate, because it invites trust it hasn't earned. So the tool reports:

- a **confidence map** showing where the labelling is guessing,
- low-confidence share by **both cost and area** — a parcel wrongly called water carries $0, so a cost-only figure would report nothing at risk while workable land quietly fell out of the total,
- **frame coverage**, confirming every pixel in view landed in exactly one parcel,
- the split between base cost and workability surcharge.

Base rates are editable in the UI, because they are the one unvalidated input a user can actually fix.

### Limitations

This is a method demonstration, not a validated model:

1. **No near-infrared band.** Real vegetation analysis uses NDVI. Excess-green is a visible-band substitute that confuses green paint, algae and shadowed grass with healthy crop.
2. **No ground truth.** The rules were reasoned out from what the features ought to mean, not fitted to labelled imagery.
3. **Illustrative rates.** The defaults are plausible placeholders, not quotes.
4. **Flat-earth area.** Ground resolution is assumed constant across the frame — true for nadir satellite imagery, false for oblique photographs.

The workability multiplier is the piece most likely to survive contact with real data, because it depends on geometry rather than spectra. A small ragged parcel hemmed in by water is expensive to work regardless of what is growing on it.

### Where this goes

The natural extension is time series: vegetation follows predictable colour progressions as it matures, so tracking class distributions across repeat passes of the same field turns this from a snapshot into a maturity signal — the crop-progression idea this project was originally aiming at, with the labelling and area accounting now actually built.

---

## Local Development

```bash
pip install -r requirements.txt
python app.py
```

Launches a Gradio interface on `http://localhost:7860`.

---

## Tech Stack

Python · OpenCV · scikit-image · scikit-learn · NumPy · Gradio

---

## References

- Achanta, R., et al. (2012). *SLIC Superpixels Compared to State-of-the-Art Superpixel Methods*. IEEE TPAMI.
- Felzenszwalb, P., & Huttenlocher, D. (2004). *Efficient Graph-Based Image Segmentation*. IJCV.
