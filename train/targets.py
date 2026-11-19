"""What the network is taught to paint: Kraken's training targets, rendered the same way.

Four planes at the page's size: where a line starts, where it ends, the
baseline, the text region. The tracing in `src/` was written against a network
taught on exactly these — the baseline a band `LINE_WIDTH` thick, a marker a
square about each end with the baseline cut out of it — so a network taught
anything else would be painting for a different tracer. This follows
`kraken.lib.dataset.segmentation.BaselineSet.transform` step by step.
"""

from __future__ import annotations

from itertools import groupby

import numpy as np
from shapely import geometry as geom
from shapely.ops import split

from fill import polygon

# The planes, as the fine-tuned Kraken model numbers them.
START, END, BASELINE, REGION = 0, 1, 2, 3
PLANES = 4
CLASSES = {"aux": {"_start_separator": START, "_end_separator": END}, "baselines": {"default": BASELINE}, "regions": {"text": REGION}}

# Kraken's `line_width`: the band's thickness on a page 1800 pixels tall.
LINE_WIDTH = 8.0


def _filled(target: np.ndarray, plane: int, shape, value: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Fills `shape`'s outline — each of them, where a buffer came apart — and says which pixels."""
    rows, cols = [], []
    for part in getattr(shape, "geoms", [shape]):
        if part.is_empty or part.geom_type != "Polygon":
            continue
        points = np.array(part.exterior.coords, dtype=int)
        rr, cc = polygon(points[:, 1], points[:, 0], shape=target.shape[1:])
        target[plane, rr, cc] = value
        rows.append(rr)
        cols.append(cc)
    if not rows:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    return np.concatenate(rows), np.concatenate(cols)


def render(size: tuple[int, int], baselines: list, regions: list, width: float = LINE_WIDTH) -> np.ndarray:
    """The four planes for a page `size` = (w, h): `baselines` and `regions`
    are lists of (x, y) points in its pixels, a baseline from its start."""
    w, h = size
    target = np.zeros((PLANES, h, w), dtype=np.uint8)
    for line in baselines:
        # Kraken drops a point repeated at once, and a line that is then no line.
        line = [point for point, _ in groupby(tuple(p) for p in line)]
        if len(line) < 2:
            continue
        shape = geom.LineString(line)
        if shape.length <= 0:
            continue
        rr, cc = _filled(target, BASELINE, shape.buffer(width / 2, cap_style=2))
        # A marker is the line's first — or last — few pixels, made a square.
        offset = min(5.0, shape.length / 2)
        for plane, at, which in ((START, offset, 0), (END, -offset, -1)):
            parts = split(shape, shape.interpolate(at).buffer(0.001)).geoms
            _filled(target, plane, parts[which].buffer(width, cap_style=3))
            target[plane, rr, cc] = 0
    for region in regions:
        if len(region) >= 3:
            points = np.clip(np.array(region, dtype=np.float64), 0, None).astype(int)
            rr, cc = polygon(points[:, 1], points[:, 0], shape=target.shape[1:])
            target[REGION, rr, cc] = 1
    return target
