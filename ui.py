"""Gradio interface: paint-by-number templates and land-use cost estimation."""

import gradio as gr
from pathlib import Path

from pipeline import run_pipeline
from land_use import (
    REGION_TABLE_HEADERS,
    RATE_TABLE_HEADERS,
    default_rate_table,
    run_land_use,
)

CSS = (Path(__file__).parent / "style.css").read_text()

DESCRIPTION = """
# Region Segmentation Toolkit

Two things built on the same segmentation core: a paint-by-number generator,
and an experiment in labelling those same regions by what they are and what
they would cost to work.

*Built by [James Kidd](https://jameskidd.ca)*
"""

LAND_USE_INTRO = """
### What this is

The paint-by-number mode asks *what colour is this region?* This mode asks the
follow-up: *what is it, and what would it cost to do something with it?*

Same segmentation, two extra stages. Each region gets a land-use label from
transparent scored rules over colour, texture and shape — then a cost, built as
**area x base rate x workability multiplier**.

The multiplier is the interesting half. Colour tells you a parcel is scrub;
it doesn't tell you the parcel is a tenth of a hectare, ragged-edged, and
walled in by water on three sides — which is what actually makes it expensive.
That information is structural, and it was already sitting in the segmentation.

The labeller isn't really the point — what SLIC makes *askable* is. A pixel has
only colour and a neighbourhood. Group pixels into regions and you can ask how
big a thing is, what shape it is, how many distinct things it borders. Those
answers move the estimate, and a human can check them: *"a road because it's
twelve times longer than wide and touches four parcels"* is verifiable by
looking, in a way that *"channel 47 activated"* is not.

So this exports a per-region feature table for downstream models, tagged by
feature family, with the reasoning attached. The ablation below tests whether
the extra families actually earn their place.
"""

CAVEATS = """
**This is a method demonstration, not a validated model.** No near-infrared
band (so excess-green stands in for NDVI, and it confuses green paint with
healthy crop); no ground truth (rules were reasoned out, not fitted); rates are
plausible placeholders, not quotes. The workability multiplier is the piece
most likely to survive contact with real data — geometry doesn't need a
spectral calibration.
"""


def _paint_by_number_tab() -> None:
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Image", type="numpy")

            working_size = gr.Slider(
                400, 1200, value=800, step=100, label="Working Size (px)",
                info="Resize longest edge before processing",
            )
            palette_size = gr.Slider(
                4, 20, value=12, step=1, label="Palette Size",
                info="Number of paint colors (KMeans clusters)",
            )
            n_superpixels = gr.Slider(
                200, 3000, value=1000, step=100, label="Superpixels",
                info="Initial over-segmentation granularity",
            )
            method = gr.Radio(
                ["slic", "felzenszwalb"], value="slic",
                label="Segmentation Method",
            )

            with gr.Accordion("Advanced Parameters", open=False):
                compactness = gr.Slider(
                    1, 30, value=12, step=1, label="SLIC Compactness",
                    info="Higher = more regular shapes",
                )
                sigma_slider = gr.Slider(
                    0.5, 5.0, value=2.0, step=0.5, label="SLIC Sigma",
                    info="Gaussian smoothing before segmentation",
                )
                rag_merge_thresh = gr.Slider(
                    2.0, 30.0, value=10.0, step=1.0,
                    label="RAG Merge Threshold",
                    info="LAB color distance for merging adjacent regions",
                )

            run_btn = gr.Button(
                "Generate Paint-by-Number", variant="primary", size="lg",
            )

        with gr.Column(scale=2):
            stats_output = gr.Markdown(label="Pipeline Stats")

            with gr.Tabs():
                with gr.Tab("Numbered Outline"):
                    outline_output = gr.Image(label="Paint-by-Number Template")
                with gr.Tab("Color Fill"):
                    fill_output = gr.Image(label="Filled Preview")
                with gr.Tab("Segmentation Map"):
                    seg_output = gr.Image(label="Region Boundaries")

            palette_output = gr.Image(label="Color Palette Legend")

    run_btn.click(
        fn=run_pipeline,
        inputs=[
            image_input, working_size, palette_size, n_superpixels,
            compactness, sigma_slider, rag_merge_thresh, method,
        ],
        outputs=[
            seg_output, fill_output, outline_output,
            palette_output, stats_output,
        ],
    )


def _land_use_tab() -> None:
    gr.Markdown(LAND_USE_INTRO)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Image", type="numpy")

            ground_resolution = gr.Number(
                value=0.5, minimum=0.001, label="Ground Resolution (m/px)",
                info="What one pixel of the original image covers on the ground. "
                     "Area — and every dollar figure — scales with its square.",
            )
            working_size = gr.Slider(
                400, 1200, value=800, step=100, label="Working Size (px)",
                info="Resize longest edge before processing (area is corrected for this)",
            )
            n_superpixels = gr.Slider(
                200, 3000, value=800, step=100, label="Superpixels",
                info="Lower values give larger, more parcel-like regions",
            )
            method = gr.Radio(
                ["slic", "felzenszwalb"], value="slic",
                label="Segmentation Method",
            )

            with gr.Accordion("Advanced Parameters", open=False):
                compactness = gr.Slider(
                    1, 30, value=10, step=1, label="SLIC Compactness",
                    info="Higher = more regular shapes",
                )
                sigma_slider = gr.Slider(
                    0.5, 5.0, value=2.0, step=0.5, label="SLIC Sigma",
                    info="Gaussian smoothing before segmentation",
                )
                rag_merge_thresh = gr.Slider(
                    2.0, 30.0, value=8.0, step=1.0,
                    label="RAG Merge Threshold",
                    info="Higher merges more aggressively into contiguous parcels",
                )

            run_btn = gr.Button(
                "Label Regions & Estimate Cost", variant="primary", size="lg",
            )

            gr.Markdown("#### Base rates")
            gr.Markdown(
                "Edit these to match real numbers. They are the only unvalidated "
                "input the estimate depends on that you can fix yourself.",
                elem_classes=["rate-hint"],
            )
            rate_table = gr.Dataframe(
                value=default_rate_table(),
                headers=RATE_TABLE_HEADERS,
                datatype=["str", "number", "str"],
                interactive=True,
                label=None,
            )

        with gr.Column(scale=2):
            report_output = gr.Markdown()

            with gr.Tabs():
                with gr.Tab("Land Use Map"):
                    label_output = gr.Image(label="Labelled Regions")
                with gr.Tab("Confidence"):
                    confidence_output = gr.Image(
                        label="Labelling Confidence — dark regions are guesses",
                    )

            legend_output = gr.Image(label="Legend")

            gr.Markdown("#### Most expensive parcels")
            region_table = gr.Dataframe(
                headers=REGION_TABLE_HEADERS,
                wrap=True,
                label=None,
            )

            ablation_output = gr.Markdown()

            gr.Markdown(
                "#### Feature export\n"
                "Per-region features for a downstream model. The CSV is the flat "
                "table; the JSON carries the same rows plus the scored evidence "
                "and cost drivers behind each one, and a schema tagging every "
                "column with its feature family."
            )
            export_output = gr.File(label=None, file_count="multiple")

            gr.Markdown(CAVEATS)

    run_btn.click(
        fn=run_land_use,
        inputs=[
            image_input, working_size, ground_resolution, n_superpixels,
            compactness, sigma_slider, rag_merge_thresh, method, rate_table,
        ],
        outputs=[
            label_output, confidence_output, legend_output,
            report_output, region_table, ablation_output, export_output,
        ],
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Region Segmentation Toolkit", css=CSS) as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Tabs():
            with gr.Tab("Paint by Number"):
                _paint_by_number_tab()
            with gr.Tab("Land Use & Cost"):
                _land_use_tab()

    return demo
