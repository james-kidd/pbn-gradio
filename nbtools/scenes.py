"""
Synthetic aerial scenes, so every notebook has something to run on.

Real imagery would be better and is the honest gap in all of this — nothing
here has been validated against a photograph. But synthetic scenes have one
property that makes them the right *demo* input: the ground truth is known by
construction, so when a notebook claims the labeller found the pond, you can
check that against the code that drew the pond.

Colours are chosen to sit where the toolkit's response curves expect them:
canopy green, senescent-wheat yellow, red-brown soil, blue water, neutral grey
for structures. Noise is added with a fixed seed so every run of every notebook
produces byte-identical inputs, which is what makes the numbers in the prose
worth printing.

`swap_in_real_imagery()` at the bottom is the one-liner for replacing all of
this with a photograph.
"""

from __future__ import annotations

import cv2
import numpy as np

# Rough RGB values for each material, before noise.
CANOPY = (58, 112, 52)
SENESCENT = (198, 178, 108)
SOIL = (142, 106, 74)
WATER = (44, 92, 158)
ROAD = (150, 150, 152)
STRUCTURE = (188, 184, 180)


def _speckle(img: np.ndarray, seed: int, amount: int = 12) -> np.ndarray:
    """Add reproducible sensor-like noise so regions are not perfectly flat."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(-amount, amount, img.shape)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def simple_farm(seed: int = 7) -> np.ndarray:
    """One field of each material, cleanly separated.

    The scene to reach for first: every land class appears exactly once, the
    parcels are large and convex, and the road runs edge to edge so the siting
    demos have a network to connect to.
    """
    h, w = 600, 800
    img = np.full((h, w, 3), CANOPY, np.uint8)
    img[40:270, 430:775] = SENESCENT
    img[300:545, 55:345] = SOIL
    cv2.ellipse(img, (620, 435), (130, 88), 0, 0, 360, WATER, -1)
    cv2.line(img, (0, 285), (800, 300), ROAD, 22)
    cv2.rectangle(img, (378, 58), (456, 142), STRUCTURE, -1)
    # Two slivers by the water: small, ragged, awkward to work — the parcels
    # the workability multiplier exists to price.
    cv2.circle(img, (500, 522), 27, SENESCENT, -1)
    cv2.circle(img, (762, 502), 21, SOIL, -1)
    return _speckle(img, seed)


def fragmented(seed: int = 3) -> np.ndarray:
    """Braided waterway and scattered structures.

    Deliberately hostile ground: many small parcels, long water boundaries and
    obstructions everywhere. This is where adjacency-derived features earn
    their keep, and running the same analysis on both scenes is how you tell a
    general result from an artefact of one picture.
    """
    h, w = 600, 800
    img = np.full((h, w, 3), CANOPY, np.uint8)
    for i in range(5):
        pts = np.array(
            [[[x, int(120 + i * 95 + 45 * np.sin(x / 70.0 + i))] for x in range(0, 800, 20)]],
            np.int32,
        )
        cv2.polylines(img, pts, False, WATER, 18)
    for cx, cy in [(140, 180), (360, 300), (600, 150), (700, 430), (250, 500), (500, 520)]:
        cv2.rectangle(img, (cx - 26, cy - 22), (cx + 26, cy + 22), STRUCTURE, -1)
    for cx, cy in [(200, 260), (420, 380), (640, 240), (120, 420), (560, 460), (720, 330)]:
        cv2.circle(img, (cx, cy), 24, SOIL, -1)
    return _speckle(img, seed, amount=10)


def no_road(seed: int = 11) -> np.ndarray:
    """Fields with no detectable road.

    Exercises the siting fallback: with no network to connect to, the tool
    assumes the grid is reachable off-frame and says so, which makes every
    connection cost a lower bound rather than an estimate.
    """
    h, w = 500, 650
    img = np.full((h, w, 3), CANOPY, np.uint8)
    img[60:240, 60:340] = SOIL
    cv2.ellipse(img, (470, 330), (110, 80), 0, 0, 360, WATER, -1)
    img[300:460, 80:300] = SENESCENT
    return _speckle(img, seed, amount=10)


def landscape(seed: int = 5) -> np.ndarray:
    """A stylised landscape with smooth tonal gradients.

    The other scenes are flat colour blocks, which is what parcel analysis
    wants and what paint-by-number does not: quantising an image that is
    already twelve flat colours produces twelve regions and proves nothing.

    A paint-by-number template exists to reduce *continuous* tone to a small
    palette, so this scene has gradients to reduce — a graded sky, hills that
    fall off with depth, and water with a horizon glow.
    """
    h, w = 600, 800
    img = np.zeros((h, w, 3), np.float32)

    # Sky: warm near the horizon, cool overhead.
    horizon = int(h * 0.52)
    top, bottom = np.array([70, 110, 185], np.float32), np.array([225, 190, 150], np.float32)
    for y in range(horizon):
        t = (y / max(horizon - 1, 1)) ** 1.6
        img[y, :] = top * (1 - t) + bottom * t

    # Sun with a soft halo.
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - 600) ** 2 + (yy - 150) ** 2)
    halo = np.clip(1.0 - dist / 190.0, 0, 1) ** 2.2
    img += halo[..., None] * np.array([90, 70, 30], np.float32)
    img[dist < 42] = np.array([255, 241, 206], np.float32)

    # Water below the horizon, darkening with distance from it.
    for y in range(horizon, h):
        t = (y - horizon) / max(h - horizon - 1, 1)
        img[y, :] = np.array([92, 126, 150], np.float32) * (1 - t) + \
                    np.array([28, 52, 78], np.float32) * t

    # Hill bands, each darker and nearer than the last.
    bands = [(horizon - 8, 0.22, (96, 118, 92)),
             (horizon + 30, 0.30, (64, 88, 68)),
             (horizon + 92, 0.40, (38, 58, 48))]
    for base, amp, colour in bands:
        ridge = base - (amp * 120 * np.sin(np.linspace(0, 3.1, w)
                                           + base)).astype(np.int32)
        col = np.array(colour, np.float32)
        for x in range(w):
            y0 = int(np.clip(ridge[x], 0, h - 1))
            shade = np.linspace(1.12, 0.82, h - y0)[:, None]
            img[y0:, x] = col * shade

    img = np.clip(img, 0, 255).astype(np.uint8)
    return _speckle(img, seed, amount=6)


SCENES = {
    "simple_farm": simple_farm,
    "fragmented": fragmented,
    "no_road": no_road,
    "landscape": landscape,
}


def load(name: str = "simple_farm", seed: int | None = None) -> np.ndarray:
    """Fetch a scene by name, for notebooks that parameterise over scenes."""
    if name not in SCENES:
        raise KeyError(f"Unknown scene {name!r}. Available: {', '.join(SCENES)}")
    builder = SCENES[name]
    return builder() if seed is None else builder(seed)


def swap_in_real_imagery(path: str) -> np.ndarray:
    """Read a real image as RGB, to replace a synthetic scene.

    Every notebook takes its input from one variable, so substituting a
    photograph is a one-line change. Two things to keep in mind when you do:
    the ground-resolution figure must match the new image, since area and every
    cost scale with its square; and the labelling has no ground truth behind
    it, so read the confidence map before trusting any class.
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read an image at {path!r}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
