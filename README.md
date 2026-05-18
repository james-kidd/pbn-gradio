---
title: Paint-by-Number Generator
emoji: "\U0001F3A8"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.0"
app_file: app.py
pinned: false
---

# Paint-by-Number Generator

Data science pipeline that converts photographs into paint-by-number templates.

## Pipeline

1. **Preprocessing** — RGB to CIELAB color space, bilateral filtering for noise reduction
2. **Segmentation** — SLIC superpixels or Felzenszwalb, then Region Adjacency Graph merging
3. **Palette Extraction** — KMeans clustering on region mean colors in LAB space
4. **Region Finalization** — Connected component analysis for clean, paintable regions
5. **Rendering** — Contour extraction, number placement via distance transform

## Local Development

```bash
pip install -r requirements.txt
python app.py
```

## Tech Stack

Python, OpenCV, scikit-image, scikit-learn, NumPy, Gradio

## Author

[James Kidd](https://jameskidd.ca) — B.Sc. Mathematics & Computer Science, McGill University
