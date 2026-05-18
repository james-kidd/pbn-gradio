---
title: Paint-by-Number Generator
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.14.0"
app_file: app.py
pinned: false
---

# Paint-by-Number Generator

A deterministic computer vision pipeline that converts photographs into paint-by-number templates using classical image segmentation and color quantization — no neural networks required.

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

## Future Direction — Crop Progression Detection

This pipeline is a stepping stone toward a more applied model: **automated crop health and maturity detection from satellite or drone imagery**, for use in commodity market analysis.

The intuition is direct: vegetation undergoes predictable color progressions as it matures — from deep green through yellowing to harvest-ready brown. SLIC superpixels provide a principled way to segment field regions by spectral similarity, and the LAB-space palette extraction can track these progressions across time-series imagery. With labeled ground truth, a downstream classifier could map palette distributions to growth stages, linking field-level color statistics to commodity supply forecasts.

The current paint-by-number application is a controlled environment for validating the segmentation and quantization pipeline before applying it to multispectral agricultural imagery.

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
