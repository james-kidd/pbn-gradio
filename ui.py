"""Gradio interface for the paint-by-number pipeline."""

import gradio as gr
from pathlib import Path
from pipeline import run_pipeline

CSS = (Path(__file__).parent / "style.css").read_text()

DESCRIPTION = """
# Paint-by-Number Generator

Upload a photo and generate a paint-by-number template.

*Built by [James Kidd](https://jameskidd.ca)*
"""

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Paint-by-Number Generator", css=CSS) as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Upload Image", type="numpy")
                gr.Examples(
                    examples=[
                        "images/raw/bird.jpeg",
                        "images/raw/vancouver.jpg",
                        "images/raw/monalisa.png",
                        "images/raw/Tiger_Berlin.jpg",
                    ],
                    inputs=image_input,
                    label="Try a sample",
                )

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
                        outline_output = gr.Image(
                            label="Paint-by-Number Template",
                        )
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

    return demo
