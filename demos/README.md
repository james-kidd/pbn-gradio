# Region Toolkit — Notebooks

Runnable demonstrations of the region-segmentation toolkit that lives one
directory up: take an image, cut it into regions, then do something with them
— render a paint-by-number template, price land, or plan where to build next.

These notebooks ship *inside* the toolkit repository, under `demos/`. That is
the arrangement that makes the Colab path simple: one clone supplies both the
notebooks and the code they exercise, so there is no second repository to keep
in step.

Every notebook runs **in Colab with one click** or **locally from a terminal**,
and calls the toolkit's real functions rather than a simplified retelling of
them. Outputs are committed, so GitHub renders the results without you running
anything.

---

## The notebooks

<!-- notebooks:start -->
<!-- generated: edit NOTEBOOKS in tools/build_notebooks.py, not here -->
| | Notebook | What it demonstrates | |
|---|---|---|---|
| 01 | [Quickstart](notebooks/01_quickstart.ipynb) | Image → regions in two calls, then check the result is structurally sound | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/james-kidd/pbn-gradio/blob/main/demos/notebooks/01_quickstart.ipynb) |
| 02 | [Segmentation backends](notebooks/02_segmentation_backends.ipynb) | SLIC vs Felzenszwalb vs watershed — and why RAG merging is **not** one of them | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/james-kidd/pbn-gradio/blob/main/demos/notebooks/02_segmentation_backends.ipynb) |
| 03 | [Paint by number](notebooks/03_paint_by_number.ipynb) | The original application, stage by stage, with the palette-size trade-off | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/james-kidd/pbn-gradio/blob/main/demos/notebooks/03_paint_by_number.ipynb) |
| 04 | [Land use and features](notebooks/04_land_use_and_features.ipynb) | Labelling, costing, the feature ablation, and the export a model would train on | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/james-kidd/pbn-gradio/blob/main/demos/notebooks/04_land_use_and_features.ipynb) |
| 05 | [Topology invariants](notebooks/05_topology_invariants.ipynb) | Planar-graph checks used as a **test oracle** — fed deliberately broken input | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/james-kidd/pbn-gradio/blob/main/demos/notebooks/05_topology_invariants.ipynb) |
| 06 | [Siting and LP](notebooks/06_siting_and_lp.ipynb) | Least-cost corridor routing, sequencing value, and a proof the greedy plan is near-optimal | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/james-kidd/pbn-gradio/blob/main/demos/notebooks/06_siting_and_lp.ipynb) |
<!-- notebooks:end -->

Each is standalone. None depends on having run the others.

---

## Running them

### Colab

Click a badge. The first cell clones this repo and the toolkit, installs
anything missing, and reports what it found. Nothing else to set up.

### Terminal

```bash
git clone https://github.com/james-kidd/pbn-gradio.git
cd pbn-gradio/demos
pip install -r requirements.txt
jupyter lab                 # then open notebooks/01_quickstart.ipynb
```

The bootstrap walks up from the notebook to find the toolkit, so it picks up
whatever is in the checkout you are sitting in — edit `../segmenters.py` and
the next cell run exercises your change.

To run them all headlessly and confirm nothing is broken:

```bash
./run_all.sh
```

---

## How the bootstrap works

Colab opens a single `.ipynb` straight from GitHub: none of the repository is
on disk and nothing is installed. A terminal has the opposite problem —
everything is already there and cloning again would be wrong.
`nbtools.bootstrap.setup()` handles both by looking before it fetches:

1. **Install only what is missing.** Nothing is upgraded — silently changing a
   working environment's package versions is the kind of help nobody asked for.
2. **Find the toolkit.** Checks the working directory, its parents and its
   siblings. Since the notebooks live inside the toolkit repo, this normally
   finds it one level up and clones nothing.
3. **Stub Gradio if absent** (see below).
4. **Import the modules** and return them on one object, so a notebook says
   `tk.segmenters.segment(...)`.

Order matters, and `nbtools/__init__.py` is deliberately empty of imports
because of it. Re-exporting `scenes` there would mean `from nbtools.bootstrap
import setup` imported `scenes`, which imports `cv2` — failing on exactly the
bare environment `setup()` exists to fix. Notebooks therefore call `setup()`
first and import `scenes` and `display` after.

### The Gradio stub

`pipeline.py` imports Gradio at module scope, so importing *any* of the compute
layer drags in the whole web-UI dependency — about a hundred megabytes a
notebook has no use for. `land_use.py` and `siting.py` already defer their
Gradio import into the function bodies that raise `gr.Error`; `pipeline.py`
does not, and the inconsistency looks like an oversight rather than a decision.

Rather than install Gradio to satisfy one exception class, the bootstrap
registers a six-line stub exposing `Error`. It is used **only** when Gradio is
genuinely absent, so a machine that has it keeps the real thing.

The upstream fix is one line — move `import gradio as gr` inside the two
functions in `pipeline.py` that use it, matching what the other modules already
do. Then the stub never fires and the compute layer imports clean anywhere.

---

## Two gotchas the notebooks surface rather than hide

Both were found while building these demos, and both are properties of the
method rather than bugs.

**Felzenszwalb's `sigma` default silently eats thin features.** Its default is
`0.5`; at that setting a 22-pixel road against a large field is smoothed away
before segmentation starts. No road means no network, and the siting analysis
falls back to "assume the grid is off-frame" — correctly reporting the
fallback, but analysing a different problem than you asked for. Notebooks 04
and 06 pass `sigma=2.0` and say why.

**RAG threshold merging cascades across gradients.** Region A is within
threshold of B, their merge is within threshold of C, and so on until an entire
graded sky is one region — 751 regions down to 8 at a threshold of 4. Correct
for parcels, fatal for a paint-by-number template. Notebook 03 disables merging
and explains the reason.

---

## Layout

```
pbn-gradio/
  segmenters.py geo.py topology.py optimize.py …   the toolkit
  demos/
    nbtools/
      bootstrap.py    find the toolkit, install deps, stub gradio
      scenes.py       synthetic demo scenes with known ground truth
      display.py      image grids and Markdown tables, no chart layer
    notebooks/        the six notebooks, with outputs committed
    tools/
      build_notebooks.py   regenerates every notebook from one file
    run_all.sh        execute every notebook headlessly
```

The notebooks are generated rather than hand-edited. Two reasons that is worth
the indirection: a change to the shared bootstrap cell is one edit rather than
six, and a `.ipynb` diff is unreadable in review while the generator is not.
After changing `tools/build_notebooks.py`, run it and then `./run_all.sh` to
re-embed outputs.

### Why synthetic scenes

Real imagery would be better, and its absence is the honest gap in all of this
— **nothing here has been validated against a photograph.** But synthetic
scenes have one property that makes them the right *demo* input: the ground
truth is known by construction, so when a notebook claims the labeller found
the pond, you can check that against the code that drew the pond.

Every notebook takes its input from one variable, so substituting a real image
is a one-line change:

```python
image = scenes.swap_in_real_imagery("aerial.jpg")
```

Two things to keep in mind when you do. The ground-resolution figure must match
the new image, since area and every cost scale with its square. And the
labelling has no ground truth behind it, so read the confidence map before
trusting any class.

---

## A note on the branch

Everything here lives on `main`, and the Colab badges point there, so a badge
keeps working after the branch that introduced a change is deleted.

Two constants name it:

| File | Constant |
|---|---|
| `tools/build_notebooks.py` | `BRANCH` — drives every Colab badge |
| `nbtools/bootstrap.py` | `SOURCE_BRANCH` — the fallback clone |

To point the notebooks at a feature branch while reviewing one, change
`BRANCH` and re-run `python tools/build_notebooks.py`: one edit retargets all
six notebook badges **and the table above**, which is most of why the
generator exists. Change it back before merging — a badge aimed at a branch
dies when the branch does.

The table above used to hold its own hand-written copies of those URLs, which
meant changing `BRANCH` fixed the badges inside the notebooks and left this
table pointing at the old branch — a trap that caught the first retarget, one
grep after it looked finished. It is generated now, from `NOTEBOOKS` in the
same file.

Neither constant is read on the ordinary path. The badge only has to name a
ref that still exists on GitHub, and `SOURCE_BRANCH` is reached only if the
toolkit is somehow not on disk, which cannot happen when the notebook was
opened from a clone of this repository. They matter when something has already
gone sideways, which is exactly when a stale branch name is least welcome.
