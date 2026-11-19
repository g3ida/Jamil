"""`fill.polygon` against the routine it stands in for.

The targets were checked against Kraken's while scikit-image filled them, so
the filling here has to be scikit-image's to the pixel and in its order —
not nearly: a baseline is a band eight pixels thick, and a pixel of its edge
is an eighth of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from fill import polygon

draw = pytest.importorskip("skimage.draw")


def _same(rows, cols, shape) -> None:
    ours, theirs = polygon(rows, cols, shape), draw.polygon(rows, cols, shape=shape)

    assert np.array_equal(ours[0], theirs[0]) and np.array_equal(ours[1], theirs[1]), (list(rows), list(cols), shape)
    assert ours[0].dtype == theirs[0].dtype and ours[1].dtype == theirs[1].dtype


def test_corners_on_pixels_fill_as_scikit_image_fills_them() -> None:
    """Corners at whole pixels put pixels exactly on edges, which is where two rules for "inside" part."""
    rng = np.random.default_rng(1)

    for _ in range(3000):
        shape = (int(rng.integers(1, 90)), int(rng.integers(1, 90)))
        corners = int(rng.integers(1, 12))

        # Corners past every side of the page too, and close together, so that edges cross and repeat.
        _same(rng.integers(-20, shape[0] + 20, corners), rng.integers(-20, shape[1] + 20, corners), shape)


def test_corners_between_pixels_fill_as_scikit_image_fills_them() -> None:
    rng = np.random.default_rng(2)

    for _ in range(3000):
        shape = (int(rng.integers(1, 90)), int(rng.integers(1, 90)))
        corners = int(rng.integers(3, 12))

        _same(rng.uniform(-20, shape[0] + 20, corners), rng.uniform(-20, shape[1] + 20, corners), shape)


def test_corners_at_halves_and_quarters_fill_as_scikit_image_fills_them() -> None:
    """Edges that pass exactly through pixels without ending on them."""
    rng = np.random.default_rng(3)

    for _ in range(3000):
        shape = (int(rng.integers(1, 60)), int(rng.integers(1, 60)))
        corners = int(rng.integers(3, 9))

        _same(rng.integers(-40, 4 * shape[0] + 40, corners) / 4, rng.integers(-40, 4 * shape[1] + 40, corners) / 4, shape)


def test_a_rectangle_is_filled_with_its_outline() -> None:
    rr, cc = polygon([2, 2, 5, 5], [3, 9, 9, 3], (20, 20))

    assert len(rr) == 4 * 7
    assert (rr.min(), rr.max(), cc.min(), cc.max()) == (2, 5, 3, 9)
    _same([2, 2, 5, 5], [3, 9, 9, 3], (20, 20))


def test_level_edges_upright_edges_and_corners_given_twice_fill_as_scikit_image_fills_them() -> None:
    rng = np.random.default_rng(4)

    for _ in range(2000):
        shape = (int(rng.integers(4, 50)), int(rng.integers(4, 50)))
        rows, cols = [int(rng.integers(-5, shape[0] + 5))], [int(rng.integers(-5, shape[1] + 5))]
        # A walk that changes one coordinate at a time, or neither.
        for _ in range(int(rng.integers(2, 14))):
            step = int(rng.integers(0, 3))
            rows.append(rows[-1] if step != 0 else int(rng.integers(-5, shape[0] + 5)))
            cols.append(cols[-1] if step != 1 else int(rng.integers(-5, shape[1] + 5)))

        _same(rows, cols, shape)


def test_a_polygon_off_the_page_has_no_pixels() -> None:
    rr, cc = polygon([-9, -9, -2], [3, 9, 6], (20, 20))

    assert len(rr) == 0 and len(cc) == 0
    _same([30, 30, 44], [3, 9, 6], (20, 20))
    _same([3, 9, 6], [30, 30, 44], (20, 20))


def test_a_region_as_large_as_the_page_fills_as_scikit_image_fills_it() -> None:
    rng = np.random.default_rng(5)

    for _ in range(4):
        corners = int(rng.integers(4, 40))

        _same(rng.integers(-100, 700, corners), rng.integers(-100, 700, corners), (512, 512))
