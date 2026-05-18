import numpy as np
import cv2
import gradio as gr
from skimage import color, segmentation, graph, morphology
from skimage.segmentation import felzenszwalb
from skimage.measure import label as cc_label, regionprops
from sklearn.cluster import KMeans


# ── Preprocessing ─────────────────────────────────────────────

def load_and_resize(img_rgb, max_dim=800):
    h, w = img_rgb.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_rgb


def preprocess_rgb(img_rgb):
    img_float = img_rgb.astype("float32") / 255.0
    img_lab = color.rgb2lab(img_float)
    l, a, b = cv2.split(img_lab.astype("float32"))
    l = cv2.normalize(l, None, 0, 100, cv2.NORM_MINMAX)
    img_lab = cv2.merge([l, a, b])
    img_lab = cv2.bilateralFilter(img_lab, d=7, sigmaColor=5, sigmaSpace=7)
    return img_lab


# ── Segmentation ──────────────────────────────────────────────

def segment_image(img_lab, n_superpixels, compactness, sigma,
                  rag_merge_thresh, min_area, method):
    if method == "felzenszwalb":
        scale = max(50, 400 - n_superpixels / 10)
        labels = felzenszwalb(img_lab, scale=scale, sigma=0.5, min_size=min_area)
    else:
        labels = segmentation.slic(
            img_lab, n_segments=n_superpixels,
            compactness=compactness, sigma=sigma, start_label=1,
        )
        g = graph.rag_mean_color(img_lab, labels, mode="distance")
        labels = graph.cut_threshold(labels, g, thresh=rag_merge_thresh)

    try:
        labels = morphology.remove_small_objects(labels, max_size=min_area)
    except TypeError:
        labels = morphology.remove_small_objects(labels, min_size=min_area)

    labels, _, _ = segmentation.relabel_sequential(labels)
    return labels, min_area


# ── Palette Extraction ────────────────────────────────────────

def quantize_palette(img_lab, labels, k):
    region_ids = np.unique(labels)
    region_means = np.array(
        [img_lab[labels == rid].mean(axis=0) for rid in region_ids]
    )
    k = min(k, len(region_means))
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    palette_ids = km.fit_predict(region_means)
    label_map = dict(zip(region_ids, palette_ids))
    indexed = np.zeros_like(labels, dtype=np.int32)
    for rid, pid in label_map.items():
        indexed[labels == rid] = pid
    return indexed, km.cluster_centers_


# ── Region Finalization ───────────────────────────────────────

def final_connectivity(img_indexed, min_area):
    final = np.zeros_like(img_indexed, dtype=np.int32)
    region_to_color = {}
    rid = 1
    for k in np.unique(img_indexed):
        cc = cc_label(img_indexed == k, connectivity=2)
        for cid in np.unique(cc):
            if cid == 0:
                continue
            mask = cc == cid
            if mask.sum() < min_area // 2:
                continue
            final[mask] = rid
            region_to_color[rid] = k
            rid += 1
    return final, region_to_color


# ── Rendering ─────────────────────────────────────────────────

def lab_to_rgb_uint8(lab_center):
    lab_pixel = np.array(
        [[[lab_center[0], lab_center[1], lab_center[2]]]], dtype="float32"
    )
    rgb = color.lab2rgb(lab_pixel)[0, 0]
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def render_outputs(final_labels, palette_lab, region_to_color):
    h, w = final_labels.shape
    fill_img = np.zeros((h, w, 3), dtype=np.uint8)
    outline_img = np.full((h, w, 3), 255, dtype=np.uint8)
    number_data = []

    for region in regionprops(final_labels):
        label_id = region.label
        if label_id not in region_to_color:
            continue
        color_index = region_to_color[label_id]
        rgb_color = lab_to_rgb_uint8(palette_lab[color_index])

        coords = region.coords
        fill_img[coords[:, 0], coords[:, 1]] = rgb_color

        mask = (final_labels == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(outline_img, contours, -1, (0, 0, 0), thickness=1)

        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist_map)
        if max_val > 6:
            number_data.append((max_loc[0], max_loc[1], color_index + 1))

    palette_rgb = np.array([lab_to_rgb_uint8(c) for c in palette_lab])
    return fill_img, outline_img, number_data, palette_rgb


def draw_numbers_on_outline(outline_img, number_data):
    result = outline_img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for x, y, n in number_data:
        text = str(int(n))
        scale, thickness = 0.35, 1
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.putText(
            result, text, (x - tw // 2, y + th // 2),
            font, scale, (180, 0, 0), thickness, cv2.LINE_AA,
        )
    return result


def draw_palette_swatch(palette_rgb, palette_size):
    sw, sh, pad = 60, 60, 4
    cols = min(palette_size, 12)
    rows = (palette_size + cols - 1) // cols
    img = np.full((rows * (sh + pad) + pad, cols * (sw + pad) + pad, 3), 40, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, rgb in enumerate(palette_rgb[:palette_size]):
        row, col = divmod(i, cols)
        x0 = pad + col * (sw + pad)
        y0 = pad + row * (sh + pad)
        cv2.rectangle(img, (x0, y0), (x0 + sw, y0 + sh), rgb.tolist(), -1)
        cv2.rectangle(img, (x0, y0), (x0 + sw, y0 + sh), (80, 80, 80), 1)
        text = str(i + 1)
        (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)
        brightness = int(rgb[0]) * 0.299 + int(rgb[1]) * 0.587 + int(rgb[2]) * 0.114
        text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)
        cv2.putText(
            img, text, (x0 + (sw - tw) // 2, y0 + (sh + th) // 2),
            font, 0.5, text_color, 1, cv2.LINE_AA,
        )
    return img


def render_segmentation_overlay(img_rgb, labels):
    boundaries = segmentation.mark_boundaries(
        img_rgb.astype("float32") / 255.0, labels, color=(1, 0.2, 0.2), mode="thick"
    )
    return (boundaries * 255).astype(np.uint8)


# ── Full Pipeline ─────────────────────────────────────────────

def run_pipeline(image, working_size, palette_size, n_superpixels,
                 compactness, sigma, rag_merge_thresh, method):
    if image is None:
        raise gr.Error("Upload an image to begin.")

    img_rgb = load_and_resize(image, int(working_size))
    h, w = img_rgb.shape[:2]
    min_area = int(h * w * 0.001)

    img_lab = preprocess_rgb(img_rgb)

    labels, min_area = segment_image(
        img_lab, int(n_superpixels), compactness, sigma,
        rag_merge_thresh, min_area, method,
    )
    n_regions_after_seg = len(np.unique(labels))
    seg_overlay = render_segmentation_overlay(img_rgb, labels)

    indexed, palette_lab = quantize_palette(img_lab, labels, int(palette_size))
    final_labels, region_to_color = final_connectivity(indexed, min_area)
    n_final_regions = len(np.unique(final_labels)) - 1

    fill_img, outline_img, number_data, palette_rgb = render_outputs(
        final_labels, palette_lab, region_to_color,
    )
    numbered_outline = draw_numbers_on_outline(outline_img, number_data)
    palette_swatch = draw_palette_swatch(palette_rgb, int(palette_size))

    stats = (
        f"**Pipeline Stats**\n\n"
        f"- Input resized to: {w} x {h}\n"
        f"- Superpixel regions: {n_regions_after_seg}\n"
        f"- Palette colors: {int(palette_size)}\n"
        f"- Final paint regions: {n_final_regions}\n"
        f"- Segmentation method: {method}"
    )

    return seg_overlay, fill_img, numbered_outline, palette_swatch, stats


# ── Gradio UI ─────────────────────────────────────────────────

DESCRIPTION = """
# Paint-by-Number Generator

**A data science approach to converting photographs into paint-by-number templates.**

### Pipeline
1. **Preprocessing** — CIELAB color space + bilateral filtering
2. **Segmentation** — SLIC superpixels or Felzenszwalb + RAG merging
3. **Palette Extraction** — KMeans clustering on region means in LAB space
4. **Region Finalization** — Connected component analysis
5. **Rendering** — Contour extraction + distance transform number placement

*Built by [James Kidd](https://jameskidd.ca) — adjust parameters below to explore the pipeline.*
"""

with gr.Blocks(title="Paint-by-Number Generator") as demo:
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

            gr.Markdown("### Parameters")
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
                ["slic", "felzenszwalb"], value="slic", label="Segmentation Method",
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
                    2.0, 30.0, value=10.0, step=1.0, label="RAG Merge Threshold",
                    info="Color distance threshold for merging adjacent regions",
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
        outputs=[seg_output, fill_output, outline_output, palette_output, stats_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
