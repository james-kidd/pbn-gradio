"""
Display helpers, so notebooks spend their cells on the toolkit rather than on
matplotlib boilerplate.

Everything here shows images or tables. There are deliberately no charts: the
toolkit already emits its quantitative results as Markdown tables with the
reasoning attached, and re-plotting those numbers would add a layer that can
disagree with the source without adding information.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

# Readable defaults for notebook output at typical Colab widths.
plt.rcParams.update({
    "figure.dpi": 100,
    "figure.facecolor": "white",
    "axes.titlesize": 10,
})


def show(image: np.ndarray, title: str = "", width: float = 7.0) -> None:
    """Display one image at a sensible size."""
    h, w = image.shape[:2]
    fig, ax = plt.subplots(figsize=(width, width * h / w))
    ax.imshow(image)
    if title:
        ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    plt.show()


def grid(
    images: Sequence[np.ndarray],
    titles: Sequence[str] | None = None,
    columns: int = 2,
    width: float = 12.0,
) -> None:
    """Display several images side by side for comparison.

    Comparison is the point of most cells here — one backend against another,
    before a fix against after — and a grid makes the difference visible in a
    way that printing two numbers does not.
    """
    images = list(images)
    titles = list(titles) if titles is not None else [""] * len(images)
    if len(titles) != len(images):
        raise ValueError(
            f"Got {len(images)} images but {len(titles)} titles — they must match."
        )

    columns = max(1, min(columns, len(images)))
    rows = (len(images) + columns - 1) // columns
    aspect = images[0].shape[0] / images[0].shape[1]
    fig, axes = plt.subplots(
        rows, columns, figsize=(width, width * aspect * rows / columns)
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap=None if img.ndim == 3 else "viridis")
        if title:
            ax.set_title(title)
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()


def table(rows: Iterable[Sequence], headers: Sequence[str]) -> None:
    """Render rows as a Markdown table in the notebook output."""
    from IPython.display import Markdown, display

    header_line = "| " + " | ".join(str(h) for h in headers) + " |"
    rule = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    display(Markdown("\n".join([header_line, rule, *body])))


def markdown(text: str) -> None:
    """Render a Markdown string — used to show the toolkit's own reports."""
    from IPython.display import Markdown, display

    display(Markdown(text))
