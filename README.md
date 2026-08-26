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

## Toolkit architecture

The modules are layered so that applications are swappable and the bottom layer can be replaced with real geodata without touching anything above it.

```
geo.py          pixel ↔ world      the seam where real georeferencing attaches
segmenters.py   image → regions    interchangeable backends + composable stages
topology.py     regions → graph    planar invariants, colouring, Steiner benchmark
optimize.py     graph → decisions  LP relaxation, duality certificates
land_use.py     ─┐
siting.py        ├─ applications, none of which the layers below know about
pipeline.py     ─┘
```

### SLIC vs Felzenszwalb vs RAG — they are not three of the same thing

This was a category error in the original code, with a visible consequence.

**SLIC, Felzenszwalb and watershed are *segmenters*.** They turn an image into regions and are alternatives to one another.

**RAG merging is a *post-process*.** It takes regions and returns fewer regions. It composes with *any* backend.

The old `segment_image` bundled RAG merging into the SLIC branch only, so the "RAG Merge Threshold" slider silently did nothing when Felzenszwalb was selected — a control that lies. The pipeline is now an explicit sequence, every stage optional and independent:

```
backend → merge → drop small → fill gaps → relabel
```

With the fix, Felzenszwalb responds to the merge threshold as it always should have: **228 regions unmerged → 8 regions at threshold 14.**

### Which backend, and why

Each encodes a different assumption about what a region *is*:

| Backend | Assumes a region is… | Best for | Weakness |
|---|---|---|---|
| **SLIC** | compact and similarly sized | paint-by-number — uniformity *is* the goal | compactness actively fights thin/elongated features |
| **Felzenszwalb** | whatever shape the real boundaries make | land parcels, natural boundaries | cannot target an exact region count |
| **Watershed** | a catchment basin of the gradient | strongest boundary adherence; when seeds are known | needs good markers |
| **Precomputed** | already known — supplied externally | **real cadastral parcels** | requires that layer to exist |

Siting now defaults to Felzenszwalb rather than SLIC. Uniform region size is a virtue for a paintable template and a liability for a parcel map.

### The `precomputed` backend is the point

Once a real parcel layer is attached, **the correct move is to stop segmenting.** A municipal cadastre *is* the segmentation, and it is authoritative in a way no image-derived guess can be. Everything downstream consumes a label array and does not care where it came from, so this is a backend swap, not a rewrite.

### `geo.py` — where the world attaches

Every hectare and dollar derives from a pixel count times an assumed ground resolution. That conversion was previously scattered inline across modules, each free to get it wrong independently — and one did. Centralising it makes it testable once and creates a single attachment point:

| Source | Constructor | Result |
|---|---|---|
| Uploaded image | `GeoContext.from_scale(m_per_px, resize_scale)` | not georeferenced; coordinates are pixels, and exports say so |
| GeoTIFF / imagery service | `GeoContext.from_affine(transform, crs)` | real world coordinates, everything downstream unchanged |

The resize factor is a constructor argument precisely because forgetting it silently rescaled every cost — a bug this repo already had once. A geographic CRS (degrees) is rejected rather than mis-measured as metres.

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

### Segmentation as an interpretability layer

The heuristic labeller is not really the point. The point is what SLIC makes *askable*.

A per-pixel model sees colour and a local neighbourhood. That is all a pixel has. Once pixels are grouped into regions, an entirely different class of question becomes answerable — *how big is this thing, what shape is it, how many distinct things does it border, is it walled in* — and those questions have two properties the pixel-level ones lack:

1. **They carry decision-relevant information.** The ablation below shows they move the cost estimate by 33–70% on identical pixels.
2. **A human can check the answers.** "This is a road because it is twelve times longer than it is wide and touches four different parcels" is a claim you can verify by looking. "This is a road because channel 47 activated" is not.

So the pipeline exports a **per-region feature table** rather than just a picture: a flat CSV for training, and a JSON carrying the same rows plus the scored evidence behind every label and every cost driver, with each column tagged by feature family. A downstream model consuming this gets features whose attributions are readable as explanations, because each input already means something in the world.

This is what shape and context buy: not better numbers necessarily, but numbers whose reasoning survives being questioned.

### Ablation — testing the claim

"Shape and context add information" is cheap to assert, so the tool tests it. Each stage re-runs labelling and costing with one more feature family available:

| Features available | Total cost | Mean confidence | Area relabelled |
|---|---:|---:|---:|
| Colour only | $44,170 | 76% | — |
| + texture | $44,438 | 89% | 3.4% |
| + shape | $69,442 | 93% | 0.4% |
| + context (full) | $75,234 | 93% | 0.0% |

*(fragmented synthetic test scene, 1200 superpixels; the simpler scene shows nearer +33%. Exact figures move with segmentation settings — the stable finding is the ordering, not the dollar amounts.)*

Two honest readings of this:

- **Shape dominates.** On both test scenes it is by far the largest single contributor. Fragmentation and boundary raggedness drive cost harder than anything colour reports.
- **Context is scene-dependent.** It contributes thousands of dollars on the fragmented scene and a few hundred on the simple one, and it acts almost entirely through the cost channel (access and frontage) rather than through labelling. That is expected — there is nothing for it to find in a scene without much water or structure boundary — but it means context is not a general-purpose win, and claiming otherwise would be overselling it.

The ablation measures how much the extra families *change* the answer, not whether they make it *correct*. Without ground truth that is the strongest claim available.

---

## Incremental Siting — the SimCity question

Given the roads and structures that already exist, **where do we add next?**

SimCity's real mechanic was never "classify the land." It was that the network you have already built determines what is cheap to build next. Every placement rewrites the cost of the next one. That mechanic maps onto machinery already here:

| Existing piece | Role in siting |
|---|---|
| Regions | The planning unit — a parcel, not a pixel |
| Land classes | A **traversal cost surface** ($/m to run a corridor) |
| Region adjacency | What each parcel borders, and what blocks approach |
| Distance transform | The parcel's point of **maximum setback** |

That last one is the neatest reuse. In paint-by-number, the distance-transform maximum is where a digit is most readable — the point furthest from every boundary. That is the same point where you would site a structure for maximum setback. Identical computation, and it happens to answer a real question.

The one genuinely new piece is **least-cost path routing** (`skimage.graph.MCP_Geometric`) — standard GIS corridor practice, and the first core algorithm in this project not invented here. That matters: it is the part of the stack that needs no defending.

### The loop

1. Build a cost surface from land classes — right-of-way is cheap to follow, canopy needs clearing, water needs a structure.
2. Seed the network from detected roads.
3. Compute cost-to-connect for **every pixel at once** (`find_costs` from all network pixels).
4. Score candidate parcels: usable size, shape regularity, connection cost, preparation cost, unobstructed approach.
5. Place the best one, route its corridor, **add that corridor to the network** at reduced traversal cost.
6. Repeat. The surface has changed, so the ranking has changed.

Each site also records what it *would* have cost built first. The gap between the two is the measurable value of sequencing — on the test scene, 8–12% of total connection cost, and visibly so: later corridors join earlier ones instead of running their own line to the road.

### A bug worth recording

Site anchors initially landed **on the frame edge** with roughly double their true inscribed radius. `cv2.distanceTransform` treats the image border as open ground, so any parcel touching it measures setback only against interior boundaries. Padding the mask by one zero pixel before the transform fixes it.

This was not cosmetic. It moved total connection cost from $37,457 to $21,284 and corridor length from 489 m to 283 m on the same frame — the anchors were being pushed to the survey edge, the one place a facility demonstrably should not go, since nothing is known about what lies beyond. Fully interior parcels were unaffected, which is what made it easy to confirm.

*(The same subtlety exists in the paint-by-number number placement, where the consequence is only a digit printed near the edge. That output is left unchanged.)*

---

## Discrete structure — planarity, colouring, Steiner trees

A segmentation into connected regions is a **planar subdivision**, so its region adjacency graph is the dual of that subdivision and is therefore **planar**. That is a strong structural constraint, and it buys three concrete things.

### 1. Invariants that catch bugs

A simple planar graph on *V* ≥ 3 vertices satisfies **E ≤ 3V − 6**, and its mean degree is strictly below 6. These are *necessary* conditions, so a violation is proof that something upstream is broken. The graph is planar by construction **if** the segmentation is well-formed — so the check is not asking "is this planar?" but "is the segmentation well-formed?", which is the question worth asking.

Measured on real segmentations, with Euler's formula V − E + F = 2 giving the face count:

| Scene | V | E | 3V − 6 | Mean degree | Disconnected regions |
|---|---:|---:|---:|---:|---:|
| Simple | 22 | 41 | 60 | 3.73 | 0 |
| Fragmented | 80 | 176 | 234 | 4.40 | 0 |

An invariant that can never fail is decoration, so it was tested against deliberately broken input:

| Input | Result |
|---|---|
| Well-formed 3×3 grid | passes |
| One label split into two non-touching blobs | **caught** — `well_formed=False` |
| K₅, the canonical non-planar graph | **caught** — E=10 > 3V−6=9 |

The disconnected-label case is the one that matters in practice: if a single label covers two separate blobs, the graph stops being the dual of a planar subdivision, and every adjacency-derived feature quietly means something other than what it claims. This is exactly the failure mode the `fill_unlabelled` fix could have introduced — and the invariant confirms it did not.

### 2. A complexity guarantee, not an observation

E ≤ 3V − 6 means the adjacency graph has **O(V) edges, not O(V²)**. Every graph algorithm downstream — adjacency scanning, colouring, routing over parcels — is therefore linear in the number of regions. That is a guarantee that follows from planarity, not a lucky property of these particular images.

### 3. Bounded scheduling via the Four Colour Theorem

Adjacent parcels cannot be worked simultaneously — one crew's access and spoil occupy both. That constraint is exactly graph colouring, and since the graph is planar, the **Four Colour Theorem** guarantees any site plan executes in **at most four parallel rounds, regardless of how many parcels there are.**

Constructive four-colouring is famously involved, so the implementation colours greedily in degeneracy order and *verifies* the result rather than assuming it. Planar graphs are 5-degenerate, bounding greedy at six colours; measured degeneracy on both test scenes is 3, so greedy finds 3 rounds.

### 4. Naming what the siting loop approximates

Connecting a set of chosen sites to an existing network at minimum total cost is the **Steiner tree problem** in a metric space — NP-hard. The incremental siting loop is a greedy heuristic for it, and the standard 2-approximation (a minimum spanning tree over the **metric closure** of the terminals, using true pairwise least-cost path distances) makes a useful yardstick:

| | Simple scene | Fragmented scene |
|---|---:|---:|
| MST over metric closure | $43,694 | $22,185 |
| This plan's connection cost | $47,394 | $22,185 |
| Premium for desirability over cheapness | $3,700 (+8%) | $0 |

The MST is a *yardstick, not a target* — it connects the same terminals as cheaply as geometry allows while ignoring whether those parcels are worth building on. Pricing that tradeoff is the point. On the fragmented scene the premium is zero: the score-driven choice happened to coincide with the cheapest structure.

*Caveat on comparability: the MST runs on the original cost surface, where reaching another terminal costs the full path to it, while the plan discounts travel along corridors it has already built. Close but not identical models of corridor reuse — read the premium as scale, not an exact figure.*

### 5. Linear programming as a certificate, not an optimiser

The siting plan picks greedily. Greedy is fast and explainable and *unbounded in how bad it might be* — nothing in the plan says whether it found something near-optimal or left half the value behind. LP answers that.

**Primal.** Choosing parcels to maximise suitability under a budget is a 0-1 knapsack — NP-hard. Relax `xᵢ ∈ {0,1}` to `0 ≤ xᵢ ≤ 1` and it becomes an LP, solvable exactly. Since every integral solution is also a feasible fractional one, the LP optimum **upper-bounds** the integer optimum. So `greedy / LP` is a *proof*, not an estimate:

> LP bound **17.41** · greedy achieved **16.36** · **certified ≥ 93.9% optimal**

**Dual.** The budget constraint carries one dual variable λ, which by strong duality is the exact **shadow price of the budget** — marginal suitability per additional dollar at the optimum. It answers "should I raise the budget?" without re-solving anything.

**Complementary slackness** makes it diagnostic. Parcel *i* belongs in the plan only if its reduced cost `vᵢ − λcᵢ ≥ 0`, so λ is precisely the value-per-dollar threshold. On the test scene this pinpointed greedy's specific mistake:

| Parcel | Reduced cost | LP | In greedy's plan |
|---|---:|---:|:--:|
| 9 | **+0.647** | 1.00 | — ← should have been |
| 5 | +0.000 | 0.42 | — ← the critical ratio |
| 3 | **−0.409** | 0.00 | ✅ ← should not have been |

Parcel 5 sitting at reduced cost exactly zero with the only fractional value is the textbook knapsack-LP structure — a basic optimum has **at most one** fractional variable. That was property-tested over 400 random instances: LP bound never violated, shadow price never negative, and **zero instances with more than one fractional variable**, matching theory exactly.

Testing also caught a real bug: the certificate did not check that the heuristic's own selection was *budget-feasible*, so an over-budget set could be reported as "≥140% optimal". Certifying a solution that violates the constraint is meaningless; it now refuses to claim a ratio.

**Scope.** This certifies **selection** under independent costs. It does not certify **routing** — shared corridors couple the costs and break the knapsack structure. Routing has its own benchmark (the MST above). Two guarantees over two separable decisions is the honest division; one number pretending to cover both would not be.

### Multi-dimensionality, stated precisely

The pipeline already operates in several distinct spaces, and it is worth separating them:

| Space | Dimensions | Where |
|---|---|---|
| SLIC clustering | 5-D — (L, a, b, x, y) | Segmentation; compactness sets the colour/space trade |
| Palette quantization | 3-D — CIELAB | Perceptually uniform, unlike RGB |
| Region descriptors | 16-D across 4 families | Labelling, cost, and the ML export |
| Adjacency | **0-D geometrically, but a graph** | The discrete layer above all of them |

The last row is the substantive one. Colour, texture and shape are all *continuous* measurements on a region. Adjacency is not a measurement at all — it is combinatorial structure, and it is the only layer where planarity, colouring and connectivity live. That is why it supports guarantees the others cannot: no amount of spectral precision tells you a plan needs at most four work rounds.

**Not yet built:** the honest extension to a genuine extra dimension is time. Repeat passes over the same ground make segmentation 3-D in (x, y, t), where regions become tubes and a parcel's *trajectory* through feature space becomes the signal — the crop-progression idea, with the topological machinery already in place to keep region identity consistent across frames.

---

## Critical assessment — where the abstraction actually holds

Being honest about the three applications, because they are not equally strong.

### What the pipeline is genuinely good at

**RAG merging is the strongest component.** SLIC over-segments; the region adjacency graph is what turns thousands of superpixels into a few dozen coherent, contiguous regions with smooth boundaries. Almost everything useful downstream depends on that merge, not on SLIC itself.

**The RAG is a graph, and that was underused.** Paint-by-number used it to merge. Land-use used it for adjacency features. Siting uses it as the structure a plan reasons over — which is the most complete use of what the pipeline already builds.

**Deterministic and fast.** No training, no GPU, no weights. The same input gives the same answer, which is what makes an auditable quote or a reproducible plan possible at all.

### Ranking the three applications honestly

| Application | Ground truth available? | Verdict |
|---|---|---|
| Paint by Number | Yes — the original image | **Strongest.** Well-posed, output checkable by eye. |
| Incremental Siting | Partially — routing is exact | **Useful as screening.** Exact algorithm on an unvalidated surface. |
| Land Use & Cost | No | **Weakest.** Interesting as a feature layer; unvalidated as a classifier. |

The ordering tracks one thing: **whether the application has an objective quality metric.** Paint-by-number can be scored against the original image. Routing can be verified against a known cost surface. Land-use classification cannot be scored at all here — the rules were reasoned out, never fitted, and there is no labelled imagery to check them against. That is why it stays framed as an interpretable feature layer rather than a classifier.

### Where SLIC is the wrong tool

- **SLIC's compactness constraint fights real parcel boundaries.** It is designed to produce compact, similarly-sized superpixels. Land parcels are neither. For planning specifically, Felzenszwalb or watershed — which follow image structure rather than imposing regularity — are arguably the better primitive, which is why both segmentation methods stay exposed.
- **Thin structures get absorbed.** A two-pixel track disappears into its surroundings under any compactness setting worth using. Roads survive here only because they are drawn wide.
- **Gradients become bands.** On smoothly-varying imagery, quantization produces contour banding rather than meaningful regions.

### Alternatives, and when they win

| Alternative | When it beats this |
|---|---|
| Plain k-means / median-cut quantization | Never for regions — no spatial coherence, so output is speckled and non-contiguous. Fine for palettes alone. |
| Watershed / Felzenszwalb | Parcel-like segmentation where following real boundaries matters more than uniform region size. |
| SAM or a trained segmentation net | When you need *semantic* objects and have a GPU. Overkill here, non-deterministic, and it answers "what object is this" rather than "what contiguous area shares a property". |
| A published GIS layer | **Beats our classifier outright for siting.** Municipal road centrelines and parcel boundaries are authoritative, free, and already vector. |

That last row is the important one and worth stating plainly: for the siting application, **the correct move in production is to not classify at all.** Take the roads layer from open data, take parcels from the assessor, and use this pipeline only for what no public layer provides — the traversal cost surface over undeveloped ground. The architecture already supports that, because routing consumes a cost surface without caring where it came from.

### The honest summary

This is a **screening and feature-extraction layer**, not a decision system. It is at its best when it narrows a large set of candidates to a small set worth a human's attention with real data, and at its worst when its unvalidated middle layer is mistaken for a measurement.

### Where this goes

Two directions, one better supported than the other.

**Supported.** Site suitability for infrastructure — data centres, solar, industrial siting — is largely a question of *shape and context*: is there a large contiguous parcel, is it regular enough to build on, does it have road frontage, is it hemmed in. Those are exactly the features this already computes, and the workability multiplier is close to a suitability score with the sign flipped. Time series is the other: vegetation follows predictable colour progressions as it matures, so tracking class distributions across repeat passes turns a snapshot into a maturity signal.

**Not supported, and worth stating plainly.** Mineral prospectivity cannot be done from RGB. Identifying mineralisation needs hyperspectral or multispectral sensing across bands this pipeline does not have — three visible channels cannot distinguish ore-bearing rock from ordinary rock of the same colour. The region-level framework here would generalise to more bands unchanged (the features are computed per region, not per channel), but the framework is not the missing piece; the sensor is. Similarly, energy infrastructure can only be inferred here through proxies like structure and road adjacency, not sensed directly.

The honest summary: this makes a good **feature and interpretability layer** for a siting or land-preparation model, and it is not a substitute for the spectral data that a minerals or energy question actually requires.

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
