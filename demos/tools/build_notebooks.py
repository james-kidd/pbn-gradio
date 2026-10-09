"""
Regenerate every notebook from this one file.

The notebooks are the artefact people open, but they are generated here. Two
reasons that is worth the indirection: a six-notebook change to a shared
bootstrap cell is one edit rather than six, and a `.ipynb` diff is unreadable
in review while this file is not.

Run it, then execute the notebooks to embed outputs:

    python tools/build_notebooks.py
    ./run_all.sh
"""

import json
import pathlib
import textwrap

OUT = pathlib.Path(__file__).resolve().parent.parent / "notebooks"
OUT.mkdir(parents=True, exist_ok=True)

REPO = "james-kidd/pbn-gradio"
BRANCH = "claude/notebook-demos"

# The first cell of every notebook. In Colab nothing is on disk, so it clones
# the toolkit repo — these notebooks live inside it, under `demos/`, so that
# single clone supplies both `nbtools` and the toolkit itself. Locally it walks
# up from the notebook and finds the checkout it is already sitting in.
BOOT = '''\
# Bootstrap: clones what it needs in Colab, finds what exists locally.
import pathlib, subprocess, sys

REPO = "https://github.com/''' + REPO + '''.git"
BRANCH = "''' + BRANCH + '''"

def _demos_root():
    for base in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:
        if (base / "nbtools" / "bootstrap.py").exists():
            return base
    dest = (pathlib.Path("/content/pbn-gradio") if pathlib.Path("/content").is_dir()
            else pathlib.Path.cwd() / "pbn-gradio")
    if not (dest / "demos" / "nbtools").exists():
        subprocess.run(["git", "clone", "--depth", "1", "--quiet",
                        "--branch", BRANCH, REPO, str(dest)], check=True)
    return dest / "demos"

sys.path.insert(0, str(_demos_root()))
from nbtools.bootstrap import setup

tk = setup()                           # installs anything missing, first
from nbtools import scenes, display    # these need cv2/matplotlib, so: after'''

BADGE = ("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
         "(https://colab.research.google.com/github/" + REPO + "/blob/"
         + BRANCH + "/demos/notebooks/{f})")


def md(s):
    return {"cell_type": "markdown", "metadata": {},
            "source": textwrap.dedent(s).strip().splitlines(True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": textwrap.dedent(s).strip().splitlines(True)}


def write(name, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"name": name, "provenance": []},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }
    (OUT / name).write_text(json.dumps(nb, indent=1) + "\n")
    print("wrote", name)


# ── 01 quickstart ────────────────────────────────────────────
write("01_quickstart.ipynb", [
    md(f"""
    # 01 · Quickstart

    {BADGE.format(f='01_quickstart.ipynb')}

    A sixty-second tour: take an image, cut it into regions, and check the
    result is structurally sound.

    Everything downstream — paint-by-number templates, land-use costing,
    infrastructure siting — is built on the two steps in this notebook.

    **Running this:** in Colab, press the badge and run all. Locally,
    `jupyter lab` from the repo root. Either way the first cell sorts out
    dependencies and the toolkit checkout.
    """),
    code(BOOT),
    md("""
    ## A scene to work on

    Synthetic, because the ground truth is then known by construction — when
    the labeller later claims it found a pond, you can check that against the
    code that drew the pond.

    `scenes.swap_in_real_imagery("photo.jpg")` substitutes a photograph.
    """),
    code("""
    image = scenes.simple_farm()
    display.show(image, f"simple_farm  ·  {image.shape[1]}x{image.shape[0]} px")
    """),
    md("""
    ## Two steps to regions

    Segmentation happens in the perceptually-uniform CIELAB space rather than
    RGB, so "similar colour" means similar *to a viewer*. Then one call cuts
    the image into regions.

    The stages after the backend — merge, drop, fill — are each optional and
    each independent of which backend ran. `describe()` reports what actually
    happened, which is worth reading rather than assuming.
    """),
    code("""
    working = tk.pipeline.load_and_resize(image, 800)
    lab = tk.pipeline.preprocess_rgb(working)

    result = tk.segmenters.segment(
        lab, backend="felzenszwalb",
        merge_threshold=8.0,   # collapse adjacent regions closer than this in LAB
        min_area=480,          # drop specks
        fill=True,             # absorb the dropped specks into their neighbours
        n_segments=900,
        sigma=2.0,             # see notebook 06 on why this matters
    )
    print(result.describe())
    """),
    code("""
    overlay = tk.pipeline.render_segmentation_overlay(working, result.labels)
    display.grid([working, overlay], ["input", f"{result.n_regions} regions"], columns=2)
    """),
    md("""
    ## Is the result sound?

    A segmentation into connected regions is a *planar subdivision*, so its
    adjacency graph must be planar. That gives hard invariants — `E ≤ 3V − 6`,
    mean degree below 6, every label a single connected blob — which hold for
    any correct segmentation and fail for a broken one.

    Checking them is not ceremony. A label covering two disconnected blobs
    silently invalidates every adjacency-derived feature computed later, and
    this catches it immediately. Notebook 05 feeds the checks deliberately
    broken input to prove they can fail.
    """),
    code("""
    graph = tk.topology.build_region_graph(result.labels)
    checks = tk.topology.check_planarity(graph, result.labels)

    display.table(
        [
            ["Vertices (regions)", checks["V"], ""],
            ["Edges (shared borders)", checks["E"], f"bound 3V-6 = {checks['edge_bound']}"],
            ["Faces (Euler V-E+F=2)", checks["faces"], ""],
            ["Mean degree", f"{checks['mean_degree']:.2f}", "must be < 6"],
            ["Disconnected regions", len(checks["disconnected_regions"]), "must be 0"],
            ["**Well formed**", "**yes**" if checks["well_formed"] else "**NO**", ""],
        ],
        ["Property", "Value", "Constraint"],
    )
    """),
    md("""
    ## Where next

    | Notebook | What it covers |
    |---|---|
    | `02_segmentation_backends` | SLIC vs Felzenszwalb vs watershed, and why RAG merging is not one of them |
    | `03_paint_by_number` | The original application, end to end |
    | `04_land_use_and_features` | Labelling regions, costing them, exporting features for a model |
    | `05_topology_invariants` | The planar-graph checks, tested against deliberately broken input |
    | `06_siting_and_lp` | Least-cost corridor routing and linear-programming certificates |

    Each is standalone — none depends on having run the others.
    """),
])

# ── 02 backends ──────────────────────────────────────────────
write("02_segmentation_backends.ipynb", [
    md(f"""
    # 02 · Segmentation backends

    {BADGE.format(f='02_segmentation_backends.ipynb')}

    **SLIC, Felzenszwalb and watershed are segmenters** — image in, regions
    out. They are alternatives to one another.

    **RAG merging is not one of them.** It takes regions and returns fewer
    regions, so it composes with *any* backend.

    Conflating the two had a visible consequence in the original code, shown
    partway down this notebook.
    """),
    code(BOOT),
    code("""
    image = scenes.simple_farm()
    working = tk.pipeline.load_and_resize(image, 800)
    lab = tk.pipeline.preprocess_rgb(working)
    MIN_AREA = int(working.shape[0] * working.shape[1] * 0.001)
    print(f"working {working.shape[1]}x{working.shape[0]}, min region {MIN_AREA}px")
    """),
    md("""
    ## Each backend assumes something different about what a region *is*

    That assumption is the whole basis for choosing between them:

    | Backend | Assumes a region is… | Best for | Weakness |
    |---|---|---|---|
    | `slic` | compact and similarly sized | paint-by-number, where uniformity **is** the goal | fights thin and elongated features |
    | `felzenszwalb` | whatever shape real boundaries make | land parcels | cannot target an exact region count |
    | `watershed` | a catchment basin of the gradient | strongest boundary adherence | needs good markers |
    | `precomputed` | already known, supplied externally | **real cadastral parcels** | requires that layer to exist |
    """),
    code("""
    variants, titles = [], []
    for backend in tk.segmenters.INTERCHANGEABLE:
        r = tk.segmenters.segment(
            lab, backend, merge_threshold=8.0, min_area=MIN_AREA, fill=True,
            n_segments=900, compactness=10.0, sigma=2.0,
        )
        variants.append(tk.pipeline.render_segmentation_overlay(working, r.labels))
        titles.append(f"{backend} - {r.n_regions} regions")
        print(f"{backend:14s} {r.describe()}")

    display.grid(variants, titles, columns=3, width=14)
    """),
    md("""
    ## Merging composes with every backend

    The same post-process, applied after each of them. Note how differently
    each responds: SLIC deliberately over-segments and collapses hard, while
    Felzenszwalb starts closer to the target.
    """),
    code("""
    rows = []
    for backend in tk.segmenters.INTERCHANGEABLE:
        counts = {}
        for thresh in (None, 4.0, 8.0, 14.0):
            r = tk.segmenters.segment(
                lab, backend, merge_threshold=thresh, min_area=MIN_AREA,
                n_segments=900, compactness=10.0, sigma=2.0,
            )
            counts[thresh] = r.n_regions
        rows.append([backend, counts[None], counts[4.0], counts[8.0], counts[14.0]])

    display.table(rows, ["Backend", "no merge", "merge@4", "merge@8", "merge@14"])
    """),
    md("""
    ### The bug this structure fixed

    The original `segment_image` bundled RAG merging into the SLIC branch
    only. Selecting Felzenszwalb left the "RAG Merge Threshold" slider wired
    to nothing — a control that lies.

    The `felzenszwalb` row above is the fix: its counts change with the
    threshold, as they always should have.
    """),
    md("""
    ## The backend that matters once real data arrives

    `precomputed` does not segment at all. A municipal cadastre **is** the
    segmentation, and it is authoritative in a way no image-derived guess can
    be.

    Everything downstream consumes a label array and does not care where it
    came from, so attaching real parcel data is a backend swap rather than a
    rewrite. Here a fake "parcel layer" stands in for the rasterised vector
    layer you would really pass.
    """),
    code("""
    import numpy as np

    parcels = np.zeros(working.shape[:2], np.int32)
    h, w = parcels.shape
    for i in range(3):
        for j in range(3):
            parcels[i * h // 3:(i + 1) * h // 3, j * w // 3:(j + 1) * w // 3] = i * 3 + j + 1

    r = tk.segmenters.segment(lab, "precomputed", labels=parcels)
    print(r.describe())
    display.show(tk.pipeline.render_segmentation_overlay(working, r.labels),
                 "precomputed - regions supplied, not inferred", width=6)

    try:
        tk.segmenters.segment(lab, "precomputed")
    except ValueError as exc:
        print(f"\\nGuard: {exc}")
    """),
    md("""
    ## Choosing, in one line

    Uniform region size is a virtue for a paintable template and a liability
    for a parcel map. That is the whole heuristic — pick the backend whose
    assumption about regions matches what you are going to do with them.
    """),
])

# ── 03 paint by number ───────────────────────────────────────
write("03_paint_by_number.ipynb", [
    md(f"""
    # 03 · Paint by number

    {BADGE.format(f='03_paint_by_number.ipynb')}

    The original application, assembled from its parts rather than called as
    one function — so each stage's contribution is visible.

    This is the strongest of the three applications in the toolkit, for one
    reason worth naming: **it has an objective quality metric.** The original
    image is right there to compare against. The land-use labelling has no
    such thing, which is why it stays framed as a feature layer rather than a
    classifier.

    The scene here is `scenes.landscape()` rather than the flat parcel scenes
    the other notebooks use — quantising an image that is already twelve flat
    colours proves nothing.
    """),
    code(BOOT),
    md("""
    ## Stages 1–2 · Preprocess and segment

    RGB to CIELAB, bilateral filter (smooths noise, keeps edges), then
    segment.
    """),
    code("""
    image = scenes.landscape()
    working = tk.pipeline.load_and_resize(image, 800)
    lab = tk.pipeline.preprocess_rgb(working)
    MIN_AREA = int(working.shape[0] * working.shape[1] * 0.001)

    seg = tk.segmenters.segment(lab, "slic", merge_threshold=None, min_area=MIN_AREA,
                                n_segments=1000, compactness=12.0, sigma=2.0)
    print(seg.describe())
    display.show(working, "a scene with tone to reduce", width=7)
    """),
    md("""
    ### Why merging is switched off here

    Every other notebook merges aggressively. This one passes
    `merge_threshold=None`, and the reason is worth knowing.

    RAG merging collapses adjacent regions closer than the threshold — and it
    **cascades**. On a smooth gradient, superpixel A is within threshold of B,
    their merged region is within threshold of C, and so on until the entire
    sky is one region. Merging this scene at a threshold of 4 takes it from
    751 regions to 8.

    Correct behaviour for parcels, where you *want* a field to be one region.
    Fatal for a paint-by-number template, whose whole job is to reduce
    continuous tone into bands.
    """),
    md("""
    ## Stage 3 · Palette

    k-means over the *region mean colours*, not over pixels. Clustering in
    LAB rather than RGB keeps the palette perceptually balanced: colours that
    look similar group together, rather than colours that happen to share raw
    channel values.
    """),
    code("""
    PALETTE_SIZE = 12
    indexed, palette_lab = tk.pipeline.quantize_palette(lab, seg.labels, PALETTE_SIZE)
    print(f"{len(palette_lab)} palette colours from {seg.n_regions} regions")
    """),
    md("""
    ## Stage 4 · Contiguity

    Connected-component analysis over the indexed image. This is what
    guarantees each numbered area is a single piece — a requirement for a
    physically paintable template, and not something the palette step
    provides on its own.
    """),
    code("""
    final_labels, region_to_colour = tk.pipeline.final_connectivity(indexed, MIN_AREA)
    print(f"{len(region_to_colour)} contiguous paint regions")
    """),
    md("""
    ## Stage 5 · Render

    Numbers are placed at each region's **distance-transform maximum** — the
    point furthest from every boundary, so the digit is never clipped by an
    edge.

    That same primitive reappears in notebook 06 as the point of maximum
    setback for siting a building. Identical computation, different question.
    """),
    code("""
    fill_img, outline_img, numbers, palette_rgb = tk.pipeline.render_outputs(
        final_labels, palette_lab, region_to_colour)
    numbered = tk.pipeline.draw_numbers_on_outline(outline_img, numbers)
    swatch = tk.pipeline.draw_palette_swatch(palette_rgb, PALETTE_SIZE)

    display.grid([working, fill_img, numbered],
                 ["original", "quantised preview", "template to paint"],
                 columns=3, width=15)
    display.show(swatch, "palette legend", width=7)
    """),
    md("""
    ## The parameter that matters

    Palette size is the real dial: it trades fidelity against how much work
    the template is to paint. More colours track the original more closely
    and take longer; fewer read as posterised.
    """),
    code("""
    previews, captions = [], []
    for k in (4, 8, 16):
        idx, pal = tk.pipeline.quantize_palette(lab, seg.labels, k)
        fl, r2c = tk.pipeline.final_connectivity(idx, MIN_AREA)
        preview, _, _, _ = tk.pipeline.render_outputs(fl, pal, r2c)
        previews.append(preview)
        captions.append(f"{k} colours - {len(r2c)} regions")

    display.grid(previews, captions, columns=3, width=15)
    """),
    md("""
    Region count rises with palette size because contiguity is enforced
    *after* quantisation: a finer palette splits areas that a coarser one
    merged, and each split piece becomes its own numbered region.
    """),
])

# ── 04 land use ──────────────────────────────────────────────
write("04_land_use_and_features.ipynb", [
    md(f"""
    # 04 · Land use, cost, and features for a model

    {BADGE.format(f='04_land_use_and_features.ipynb')}

    Paint-by-number asks *what colour is this region?* This asks the
    follow-up: *what is it, and what would it cost to do something with it?*

    **Read the limitations first.** The labeller is unfitted rules over
    visible-band colour with no near-infrared and no ground truth. It is a
    feature layer, not a validated classifier, and the notebook surfaces its
    own confidence for exactly that reason.
    """),
    code(BOOT),
    code("""
    image = scenes.simple_farm()
    working = tk.pipeline.load_and_resize(image, 800)
    lab = tk.pipeline.preprocess_rgb(working)
    MIN_AREA = int(working.shape[0] * working.shape[1] * 0.001)

    seg = tk.segmenters.segment(lab, "felzenszwalb", merge_threshold=8.0,
                                min_area=MIN_AREA, fill=True, n_segments=900,
                                sigma=2.0)   # keeps the thin road alive - see nb 06
    labels = seg.labels
    print(seg.describe())
    """),
    md("""
    ## Ground truth lives in one place

    Every hectare and every dollar comes from a pixel count times an assumed
    ground resolution. `GeoContext` owns that conversion so it cannot be got
    wrong in two places differently — and it is the seam where a real
    georeferenced raster attaches later.
    """),
    code("""
    ctx = tk.geo.GeoContext.from_scale(
        0.5, resize_scale=max(image.shape[:2]) / max(working.shape[:2]))
    print(ctx.describe())
    print(ctx.export_note())
    """),
    md("""
    ## Features, then labels, then labels again

    Adjacency splits into two passes for a real reason. *Neighbour counts* are
    pure graph structure — independent of class — so they are available before
    labelling and may inform it. The *class fractions* necessarily come after,
    since they describe what the neighbours turned out to be. One refinement
    pass, then stop: iterating to convergence would be a conditional random
    field, which is a different project.
    """),
    code("""
    features = tk.land_use.extract_features(working, lab, labels)
    tk.land_use.compute_topology(labels, features)     # class-independent
    tk.land_use.classify_regions(features)             # first pass
    tk.land_use.compute_adjacency(labels, features)    # now classes are known
    tk.land_use.classify_regions(features)             # refine with context

    print(f"{len(features)} regions, {len({f.class_key for f in features})} distinct classes")
    """),
    code("""
    display.grid(
        [tk.land_use.render_label_map(working, labels, features),
         tk.land_use.render_confidence_map(labels, features)],
        ["land use", "confidence - dark means guessing"], columns=2, width=13)
    """),
    md("""
    The confidence map is a deliberate output, not a diagnostic. An estimate
    that hides its weak spots invites trust it has not earned.
    """),
    md("""
    ## Cost

    `area × base_rate[class] × workability_multiplier`.

    The multiplier is the interesting half. Colour tells you a parcel is
    scrub; it does not tell you the parcel is a tenth of a hectare,
    ragged-edged and walled in by water on three sides — which is what
    actually makes it expensive. That information is structural, and it was
    already sitting in the segmentation.
    """),
    code("""
    rates = tk.land_use.parse_rate_table(tk.land_use.default_rate_table())
    costs = tk.land_use.estimate_costs(features, ctx.m_per_px, rates)
    frame_ha = ctx.area_ha(working.shape[0] * working.shape[1])

    display.markdown(
        tk.land_use.build_cost_report(features, costs, ctx.m_per_px, rates, frame_ha))
    """),
    md("""
    ## Does shape actually add anything, or does colour do the work?

    Cheap to assert, so it gets tested. Each row re-runs labelling and costing
    with one more feature family available.
    """),
    code("""
    ablation = tk.land_use.run_ablation(labels, features, ctx.m_per_px, rates)
    display.markdown(tk.land_use.format_ablation(ablation))
    """),
    md("""
    Two honest readings, and the second matters more than the headline:

    - **Shape dominates.** Fragmentation and boundary raggedness move the
      estimate harder than anything colour reports.
    - **Context is scene-dependent.** It contributes meaningfully on
      fragmented ground and close to nothing on a simple scene, acting almost
      entirely through cost rather than labelling. Not a general-purpose win.

    Change the first cell to `scenes.fragmented()` to see the second point
    directly.

    The ablation measures how much the extra families *change* the answer,
    not whether they make it *correct*. Without ground truth that is the
    strongest claim available.
    """),
    md("""
    ## Export for a downstream model

    The flat CSV is for training. The JSON carries the same rows plus the
    scored evidence behind each label and the cost drivers behind each
    estimate, with every column tagged by feature family.

    That tagging is the point: an attribution over these inputs reads as an
    explanation, because each input already means something in the world.
    """),
    code("""
    import csv, json, tempfile

    records = tk.land_use.export_features(features, costs)
    csv_path, json_path = tk.land_use.write_feature_export(records, tempfile.mkdtemp())

    with open(csv_path) as fh:
        rows = list(csv.reader(fh))
    display.table([r[:7] for r in rows[1:6]], rows[0][:7])

    schema = json.load(open(json_path))["schema"]
    display.table([[s["column"], s["family"], s["description"]] for s in schema],
                  ["column", "family", "description"])
    """),
    code("""
    example = next(r for r in records if r["evidence"])
    print(f"region {example['region_id']}  ->  {example['predicted_class']} "
          f"({example['confidence']:.0%})\\n")
    for item in example["evidence"]:
        print(f"  {item['weight']:+.2f}  [{item['family']:7s}] {item['reason']}")
    """),
    md("""
    That block is the whole argument for region-level features. *"A road
    because it is twelve times longer than wide and touches four parcels"* is
    a claim you can verify by looking at the picture. *"A road because channel
    47 activated"* is not.
    """),
])

# ── 05 topology ──────────────────────────────────────────────
write("05_topology_invariants.ipynb", [
    md(f"""
    # 05 · Planar structure, used as a test oracle

    {BADGE.format(f='05_topology_invariants.ipynb')}

    A segmentation into connected regions is a **planar subdivision**, so its
    adjacency graph is the dual of that subdivision and therefore planar.

    That is not decoration. Planarity is a strong constraint, and it buys
    three concrete things: invariants that catch bugs, a complexity
    guarantee, and a bound on how many parallel work rounds any plan needs.

    The middle section deliberately feeds the checks broken input, because
    **an invariant that can never fail is decoration.**
    """),
    code(BOOT),
    code("""
    image = scenes.fragmented()
    working = tk.pipeline.load_and_resize(image, 800)
    lab = tk.pipeline.preprocess_rgb(working)
    MIN_AREA = int(working.shape[0] * working.shape[1] * 0.001)

    seg = tk.segmenters.segment(lab, "slic", merge_threshold=8.0, min_area=MIN_AREA,
                                fill=True, n_segments=1200, compactness=10.0, sigma=2.0)
    labels = seg.labels
    graph = tk.topology.build_region_graph(labels)
    print(seg.describe())
    """),
    md("""
    ## 1 · The invariants

    A simple planar graph on V ≥ 3 vertices satisfies **E ≤ 3V − 6**, and its
    mean degree is strictly below 6. These are *necessary* conditions, so a
    violation proves something upstream is broken.

    The graph is planar by construction **if** the segmentation is well
    formed — which means the check is really asking whether the segmentation
    is well formed, and that is the question worth asking.
    """),
    code("""
    checks = tk.topology.check_planarity(graph, labels)
    display.table(
        [["E <= 3V - 6", f"<= {checks['edge_bound']}", checks["E"],
          "pass" if checks["edge_bound_holds"] else "FAIL"],
         ["Mean degree", "< 6", f"{checks['mean_degree']:.2f}",
          "pass" if checks["degree_bound_holds"] else "FAIL"],
         ["Regions connected", "all", f"{len(checks['disconnected_regions'])} violations",
          "pass" if not checks["disconnected_regions"] else "FAIL"]],
        ["Invariant", "Required", "Actual", "Result"])

    print(f"V = {checks['V']}   E = {checks['E']}   "
          f"F = {checks['faces']}   (Euler: V - E + F = 2)")
    """),
    md("""
    ## 2 · Do the checks have teeth?

    Two deliberately broken inputs. If these passed, the checks above would
    mean nothing.

    **Case A — a label covering two disconnected blobs.** This is the failure
    that matters in practice: the graph stops being a planar dual, and every
    adjacency-derived feature quietly means something other than it claims.
    """),
    code("""
    import numpy as np

    good = np.zeros((90, 90), np.int32)
    for i in range(3):
        for j in range(3):
            good[i * 30:(i + 1) * 30, j * 30:(j + 1) * 30] = i * 3 + j + 1

    broken = good.copy()
    broken[broken == 9] = 1      # label 1 now covers two separate corners

    for name, arr in (("well-formed grid", good), ("split label", broken)):
        g = tk.topology.build_region_graph(arr)
        c = tk.topology.check_planarity(g, arr)
        verdict = ("well formed" if c["well_formed"]
                   else f"CAUGHT - disconnected: {c['disconnected_regions']}")
        print(f"{name:18s} V={c['V']:2d} E={c['E']:2d}  ->  {verdict}")
    """),
    md("""
    **Case B — K₅, the canonical non-planar graph.** Five mutually adjacent
    regions cannot be drawn in the plane, so this is impossible to produce
    from a real image. Constructing the graph directly proves the edge bound
    actually bites.
    """),
    code("""
    k5 = tk.topology.RegionGraph(
        vertices=[1, 2, 3, 4, 5],
        edges={(a, b) for a in range(1, 6) for b in range(a + 1, 6)})
    k5.adjacency = {v: {u for u in range(1, 6) if u != v} for v in range(1, 6)}

    c = tk.topology.check_planarity(k5, np.ones((4, 4), np.int32))
    print(f"K5:  V={c['V']}  E={c['E']}  bound 3V-6={c['edge_bound']}  ->  "
          f"{'CAUGHT' if not c['edge_bound_holds'] else 'passed (wrong!)'}")
    """),
    md("""
    Both broken inputs are caught. The checks are an oracle, not ceremony.
    """),
    md("""
    ## 3 · A complexity guarantee, not an observation

    `E ≤ 3V − 6` means the adjacency graph has **O(V) edges, not O(V²)**.
    Every graph algorithm downstream is therefore linear in the number of
    regions — by consequence of planarity, not by luck of these particular
    images.
    """),
    code("""
    display.table(
        [[checks["V"], checks["E"], checks["V"] ** 2,
          f"{checks['E'] / checks['V']:.2f}", f"{checks['E'] / checks['V']**2:.4f}"]],
        ["V", "actual E", "E if dense (V^2)", "E/V", "E/V^2"])
    """),
    md("""
    ## 4 · Bounded scheduling, via the Four Colour Theorem

    Adjacent parcels cannot be worked at the same time — one crew's access
    and spoil occupy both. That constraint *is* graph colouring, and since
    the graph is planar, **four colours always suffice**: any site plan
    executes in at most four parallel rounds, however many parcels it has.

    Constructive four-colouring is famously involved, so this colours greedily
    in degeneracy order and then *verifies* the result rather than assuming
    it. Planar graphs are 5-degenerate, which bounds greedy at six.
    """),
    code("""
    order, degeneracy = tk.topology.degeneracy_ordering(graph)
    colouring = tk.topology.colour_graph(graph)
    valid = tk.topology.verify_colouring(graph, colouring)
    used = len(set(colouring.values()))

    display.table(
        [["Degeneracy", degeneracy, "planar graphs are <= 5"],
         ["Greedy bound", degeneracy + 1, "degeneracy + 1"],
         ["Colours used", used, "Four Colour Theorem guarantees 4 suffice"],
         ["Colouring valid", "yes" if valid else "NO", "verified, not assumed"]],
        ["Property", "Value", "Note"])
    """),
    code("""
    parcels = sorted(graph.vertices)[:8]
    schedule, rounds = tk.topology.work_schedule(graph, parcels)
    print(f"{len(parcels)} parcels -> {rounds} parallel round(s)\\n")
    for r in range(1, rounds + 1):
        members = sorted(p for p, c in schedule.items() if c == r)
        print(f"  round {r}: parcels {members}")
    """),
    md("""
    ## 5 · Holes

    A region's Euler number is `components − holes`, so a parcel wrapped
    around a pond reports one hole. That is a planning fact, not a curiosity:
    interior boundary cannot be built on and has to be worked around.
    """),
    code("""
    holes = tk.topology.region_holes(labels)
    holed = {k: v for k, v in holes.items() if v}
    print(f"{len(holed)} of {len(holes)} regions contain holes")
    if holed:
        display.table(sorted(holed.items())[:10], ["region", "holes"])
    """),
])

# ── 06 siting + LP ───────────────────────────────────────────
write("06_siting_and_lp.ipynb", [
    md(f"""
    # 06 · Incremental siting, and proving the greedy plan is good

    {BADGE.format(f='06_siting_and_lp.ipynb')}

    The SimCity question: given the roads and structures that already exist,
    **where do we add next?**

    The mechanic is that the network already built determines what is cheap
    to build next — every placement rewrites the cost of the next one.

    Then the part that makes it trustworthy: greedy is fast, explainable, and
    unbounded in how bad it might be. Linear programming closes that gap with
    a proof rather than an assurance.
    """),
    code(BOOT),
    code("""
    import numpy as np
    from skimage import graph as skgraph

    image = scenes.simple_farm()
    working = tk.pipeline.load_and_resize(image, 800)
    lab = tk.pipeline.preprocess_rgb(working)
    MIN_AREA = int(working.shape[0] * working.shape[1] * 0.001)
    ctx = tk.geo.GeoContext.from_scale(
        0.5, resize_scale=max(image.shape[:2]) / max(working.shape[:2]))

    seg = tk.segmenters.segment(lab, "felzenszwalb", merge_threshold=8.0,
                                min_area=MIN_AREA, fill=True, n_segments=900,
                                sigma=2.0)   # see note below on thin features
    labels = seg.labels

    features = tk.land_use.extract_features(working, lab, labels)
    tk.land_use.compute_topology(labels, features)
    tk.land_use.classify_regions(features)
    tk.land_use.compute_adjacency(labels, features)
    tk.land_use.classify_regions(features)
    print(f"{seg.describe()}  |  {ctx.describe()}")
    """),
    md("""
    ## The cost surface

    Land classes become dollars per metre of corridor. The ratios are what
    drive routing and they are defensible: an existing right-of-way is an
    order of magnitude cheaper to follow than mature canopy, and water needs a
    structure.

    Buildings are expensive but **finite** rather than infinite — a building
    is not a wall of physics, and the estimate should say how expensive going
    through it would be rather than pretending no route exists.
    """),
    code("""
    display.table(sorted(tk.siting.TRAVERSAL_COST.items(), key=lambda kv: kv[1]),
                  ["class", "$ per metre"])

    cost_surface = tk.siting.build_cost_surface(labels, features, ctx.m_per_px)
    network, note = tk.siting.seed_network(labels, features)
    print(f"network seed: {note}  ({int(network.sum())} px)")
    """),
    md("""
    ### A gotcha worth keeping

    That `sigma=2.0` in the segmentation cell is load-bearing. Felzenszwalb's
    own default is **0.5**, and at that setting the road — twenty-two pixels
    wide against a large green field — gets smoothed away before segmentation
    even starts. No road means no network, and the whole analysis silently
    falls back to "assume the grid is somewhere off-frame".

    It is the concrete version of the warning that segmenters fight thin
    features. The fallback reports itself rather than hiding, which is how it
    is catchable at all, but the honest fix is to keep the feature.
    """),
    md("""
    ## What would it cost to connect *anywhere*?

    One least-cost-path solve from every network pixel at once gives the
    cost-to-connect for every pixel on the frame. This is the single most
    useful output here, and notably it needs no classification to read: dark
    is cheap to reach, bright is expensive.
    """),
    code("""
    rows, cols = np.nonzero(network)
    seeds = list(zip(rows.tolist(), cols.tolist()))
    mcp = skgraph.MCP_Geometric(cost_surface)
    cum, _ = mcp.find_costs(seeds)

    display.grid(
        [working, tk.siting.render_connection_map(working, cum, network)],
        ["input", "cost to connect (cyan = existing network)"], columns=2, width=13)
    """),
    md("""
    ## The loop

    Place the best site, route its corridor, **add that corridor to the
    network at reduced cost**, recompute, repeat. The surface has changed, so
    the ranking has changed.

    Each site also records what it *would* have cost built first. The gap
    between the two is the measurable value of having sequenced the build
    rather than costing each parcel independently.
    """),
    code("""
    sites = tk.siting.candidate_sites(labels, features, ctx.m_per_px,
                                      min_area_ha=0.05, target_area_ha=2.0)
    placed, grown = tk.siting.plan_build_order(
        cost_surface, network, sites, features, ctx.m_per_px,
        rounds=4, target_area_ha=2.0, connection_budget=40_000, corridor_cost=15.0)

    display.show(tk.siting.render_plan(working, labels, grown, placed),
                 "build order - yellow corridors, cyan existing network", width=9)
    """),
    code("""
    display.table(tk.siting.build_plan_table(placed), tk.siting.PLAN_TABLE_HEADERS)
    """),
    code("""
    display.markdown(tk.siting.build_siting_report(placed, len(sites), note, ctx.m_per_px))
    """),
    md("""
    Look for a later corridor *joining* an earlier one rather than running its
    own line to the road. That join is the saving, made visible.
    """),
    md("""
    ## Naming what the loop approximates

    Connecting a set of chosen sites to an existing network at minimum total
    cost is the **Steiner tree problem** — NP-hard. The greedy loop is a
    heuristic for it, and the standard 2-approximation (a minimum spanning
    tree over the *metric closure* of the terminals) makes a yardstick.

    It is a yardstick, not a target: it connects the same terminals as cheaply
    as geometry allows while ignoring whether those parcels are worth building
    on. Pricing that trade-off is the point.
    """),
    code("""
    steiner = tk.topology.steiner_benchmark(cost_surface, network,
                                            [s.point for s in placed])
    achieved = sum(s.connection_cost for s in placed)
    display.table(
        [["MST over metric closure", f"${steiner['mst_cost']:,.0f}"],
         ["This plan", f"${achieved:,.0f}"],
         ["Premium for desirability over cheapness",
          f"${achieved - steiner['mst_cost']:,.0f}"]],
        ["Measure", "Cost"])
    """),
    md("""
    ## Proving the selection was good

    Choosing parcels to maximise suitability under a budget is a **0-1
    knapsack** — NP-hard. Relax the integrality and it becomes a linear
    program, solvable exactly. Since every integral solution is also a
    feasible fractional one, **the LP optimum upper-bounds the integer
    optimum**.

    So `greedy / LP` is not an estimate. It is a proof of how close greedy
    came to the best achievable.
    """),
    code("""
    for s in sites:
        s.connection_cost = s.baseline_cost
    tk.siting.score_sites(sites, features, 2.0, 40_000)

    site_costs = [
        s.baseline_cost
        + tk.land_use.LAND_CLASSES[tk.land_use.CLASS_INDEX[s.class_key]].base_rate * s.area_ha
        for s in sites]
    placed_ids = {id(s) for s in placed}
    chosen = [i for i, s in enumerate(sites) if id(s) in placed_ids]
    budget = sum(site_costs[i] for i in chosen)

    cert = tk.optimize.budgeted_selection(
        [max(0.0, s.score) for s in sites], site_costs, budget, chosen)
    display.markdown(
        tk.optimize.format_certificate(cert, [s.label for s in sites], chosen))
    """),
    md("""
    ### Reading the dual

    The budget constraint carries one dual variable λ. By strong duality it is
    the **exact** marginal suitability bought by one more dollar — so it
    answers *"should I raise the budget?"* without re-solving anything.

    Complementary slackness then makes the reduced cost `vᵢ − λcᵢ`
    diagnostic: λ is precisely the value-per-dollar threshold a parcel must
    clear. A parcel in the plan with a **negative** reduced cost is one greedy
    should not have taken; a parcel outside it with a **positive** one is a
    parcel greedy should have taken.

    That is not a vague "greedy could be better" — it names the specific swap.
    """),
    md("""
    ## Scope, kept honest

    - The LP certifies **selection** under independent costs. It does not
      certify **routing** — shared corridors couple the costs and break the
      knapsack structure. Routing has its own MST benchmark above. Two
      guarantees over two separable decisions.
    - **Routing is exact; the surface under it is not.** The cheapest corridor
      across a given cost surface is computed correctly and you can check it by
      eye. Where that surface comes from is the unvalidated classifier — and
      that is the swappable part. Feed in a published road centreline layer and
      the routing is production-grade unchanged.
    - **No elevation.** A single RGB frame cannot see slope, and slope
      dominates real trenching cost.

    Use this to narrow many parcels to a few worth real diligence, not to
    choose one.
    """),
])
