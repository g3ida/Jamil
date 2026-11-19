"""The pixels of a polygon: `skimage.draw.polygon`'s answer, worked out a row at a time.

scikit-image asks of every pixel in a polygon's box whether it is inside, and
keeps each that is as two Python numbers in two lists. A text region is most
of a training square: a million pixels, two million numbers, 170 ms — nearly
half of what a loading worker did, and where the workers of a long run died,
one every quarter of an hour, with a row no polygon has or with nothing said
at all. Here a row is filled between the places its edges cross it, so the
work goes with the polygon's outline and not with its area, and nothing is
kept a pixel at a time: 27 ms for that region, and a square drawn in 190 ms
where it took 320.

Which pixels, exactly, is scikit-image's rule and is kept to the pixel — the
targets were checked against Kraken's with it. A pixel is in when an odd
number of edges pass it on its right, or an odd number on its left (the two
differ only on an edge, so a pixel on the outline is in), or when it is a
corner. An edge counts on the right for the rows from its lower end up to but
not its upper, on the left for those above its lower end up to its upper.
"""

from __future__ import annotations

import numpy as np

# scikit-image's tolerance for "this pixel is a corner".
CORNER = 1e-12


def _passing(xa, ya, xb, yb, x, y):
    """How far right of pixel (x, y) the edge a→b passes its row, in scikit-image's own arithmetic:
    which side a pixel is on is the sign of this, and for a pixel nearly on an edge the sign is the arithmetic's."""
    x0, y0, x1, y1 = xb - x, yb - y, xa - x, ya - y
    return (x0 * y1 - x1 * y0) / (y1 - y0)


def _rows(first: np.ndarray, last: np.ndarray, r0: int, r1: int) -> tuple[np.ndarray, np.ndarray]:
    """Every (edge, row) with the row from `first` to `last` of that edge and inside r0…r1."""
    first, last = np.maximum(first, r0).astype(np.int64), np.minimum(last, r1).astype(np.int64)
    many = np.maximum(last - first + 1, 0)
    edge = np.repeat(np.arange(len(many)), many)
    row = np.arange(many.sum()) - np.repeat(np.cumsum(many) - many, many) + first[edge]
    return edge, row


def _odd(rows: np.ndarray, places: np.ndarray, high: int, wide: int) -> np.ndarray:
    """For each row of `high`, whether an odd number of its `places` lie at or before each of `wide` + 1 columns."""
    marks = np.bincount(rows * (wide + 1) + places, minlength=high * (wide + 1)).reshape(high, wide + 1)
    return np.cumsum(marks.astype(np.uint8), axis=1, dtype=np.uint8) & 1


def polygon(rows, cols, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The (rows, columns) of the pixels in the polygon with corners at `rows`, `cols`
    that lie inside `shape`, row by row: what `skimage.draw.polygon` returns."""
    r, c = np.asarray(rows, dtype=np.float64), np.asarray(cols, dtype=np.float64)
    none = np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    if len(r) == 0:
        return none
    # The polygon's box, as much of it as is on the page.
    r0, r1 = max(0, int(np.floor(r.min()))), min(shape[0] - 1, int(np.ceil(r.max())))
    c0, c1 = max(0, int(np.floor(c.min()))), min(shape[1] - 1, int(np.ceil(c.max())))
    if r0 > r1 or c0 > c1:
        return none
    high, wide = r1 - r0 + 1, c1 - c0 + 1
    # Each edge runs from the corner before to the corner itself, the last corner before the first.
    xa, ya, xb, yb = np.roll(c, 1), np.roll(r, 1), c, r
    low, top = np.minimum(ya, yb), np.maximum(ya, yb)

    def crossed(edge: np.ndarray, row: np.ndarray, right: bool) -> np.ndarray:
        """The first column of each row that the edge does not pass on the right of — or, for `right` false, the first it passes on the left of."""
        a, b, y = (xa[edge], ya[edge]), (xb[edge], yb[edge]), row.astype(np.float64)
        about = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
        at = np.ceil(about) if right else np.floor(about) + 1
        # The division above is not scikit-image's: where the two fall on different sides of a pixel, its side stands.
        for _ in range(2):
            here, before = _passing(*a, *b, at, y), _passing(*a, *b, at - 1, y)
            at += (here > 0) if right else (here >= 0)
            at -= (before <= 0) if right else (before < 0)
        return np.clip(at - c0, 0, wide).astype(np.int64)

    # Passed on the right: the columns before the crossing, of the rows from an edge's lower end up to but not its upper.
    edge, row = _rows(np.ceil(low), np.ceil(top) - 1, r0, r1)
    before = _odd(row - r0, crossed(edge, row, True), high, wide)
    on_right = before[:, -1:] ^ before[:, :-1]
    # Passed on the left: the columns from the crossing on, of the rows above an edge's lower end up to its upper.
    edge, row = _rows(np.floor(low) + 1, np.floor(top), r0, r1)
    on_left = _odd(row - r0, crossed(edge, row, False), high, wide)[:, :-1]
    inside = (on_right | on_left).astype(bool)
    # A corner on a pixel is in whatever the count says.
    nearest_r, nearest_c = np.rint(r), np.rint(c)
    on = (np.abs(r - nearest_r) < CORNER) & (np.abs(c - nearest_c) < CORNER) & (nearest_r >= r0) & (nearest_r <= r1) & (nearest_c >= c0) & (nearest_c <= c1)
    inside[nearest_r[on].astype(np.int64) - r0, nearest_c[on].astype(np.int64) - c0] = True
    rr, cc = np.nonzero(inside)
    return (rr + r0).astype(np.intp), (cc + c0).astype(np.intp)
