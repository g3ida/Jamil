"""dump_outlines.py GOLDEN_DIR [STEM…]: what Kraken's polygonizer makes of each line, step by step.

Needs no model: it starts from the grey page and the baselines `dump_golden.py`
saved, and runs `calculate_polygonal_environment` line by line as `vec_lines`
does, with every inner function wrapped to keep what it returned.

  <stem>.feats.npy       float64 (H, W): the seam-carving energy, `gaussian_filter(sobel(page), 0.5)`
  <stem>.outlines.json   per baseline: the region of interest (`_calc_roi`), the
                         offset baseline, each seam with the numbers that decide
                         it (patch corner, transform, mask checksums), the seams
                         simplified and offset, the region polygon, the outline
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import shapely.geometry as geom
from kraken.lib import segmentation as S
from scipy.ndimage import gaussian_filter
from shapely.ops import unary_union
from skimage.filters import sobel

warnings.filterwarnings("ignore")


def checksum(mask: np.ndarray) -> list[int]:
    """How many pixels are set and where, enough to tell two masks apart."""
    at = np.flatnonzero(mask)
    return [int(len(at)), int((at % 1000003).sum() % 2147483647)]


def coords(g) -> list | None:
    return None if g is None else np.asarray(g, dtype=float).tolist()


def line_of(seam: np.ndarray, side: str) -> dict:
    """The seam as `_extract_patch` goes on to treat it."""
    out: dict = {}
    try:
        simple = geom.LineString(seam).simplify(5)
        out["simplified"] = coords(simple.coords)
        moved = simple.parallel_offset(4, side=side)
        out["offset_type"] = moved.geom_type
        if moved.geom_type == "LineString":
            out["offset"] = coords(moved.coords)
    except Exception as exc:  # noqa: BLE001 - recorded, as Kraken would fail here too
        out["error"] = str(exc)[:200]
    return out


def main() -> int:
    out = Path(sys.argv[1])
    stems = sys.argv[2:] or sorted(p.name[: -len(".scal.npy")] for p in out.glob("*.scal.npy"))
    inner = {name: getattr(S, name) for name in ("_calc_roi", "_calc_seam", "_extract_patch", "_rotate", "make_polygonal_mask", "binary_erosion")}
    for stem in stems:
        meta = json.loads((out / f"{stem}.json").read_text())
        scal = np.load(out / f"{stem}.scal.npy")
        feats = gaussian_filter(sobel(scal), 0.5)
        np.save(out / f"{stem}.feats.npy", feats.astype(np.float64))
        baselines, regions = meta["baselines"], meta["regions"]
        reg_pols = [geom.Polygon(r) for r in regions]
        lines = []
        for i, bl in enumerate(baselines):
            rec: dict = {"seams": []}
            call: dict = {}

            def roi(*a, **k):
                up, bottom = inner["_calc_roi"](*a, **k)
                rec["env_up"], rec["env_bottom"] = up.tolist(), bottom.tolist()
                return up, bottom

            def rotate(*a, **k):
                tform, rotated = inner["_rotate"](*a, **k)
                call["tform"] = tform.params.ravel().tolist()
                call["rotated"] = list(rotated.shape)
                return tform, rotated

            def poly_mask(polygon, shape):
                mask = inner["make_polygonal_mask"](polygon, shape)
                call["drawn"] = checksum(np.array(mask) > 128)
                return mask

            def erosion(*a, **k):
                mask = inner["binary_erosion"](*a, **k)
                call["masked"] = checksum(mask)
                return mask

            def seam(baseline, polygon, angle, im_feats, bias=150):
                call.clear()
                call["corner"] = [int(polygon[:, 0].min()), int(polygon[:, 1].min())]
                call["angle"] = float(angle)
                try:
                    got = inner["_calc_seam"](baseline, polygon, angle, im_feats, bias)
                    call["seam"] = got.tolist()
                    return got
                finally:
                    rec["seams"].append(dict(call))

            def patch(env_up, env_bottom, baseline, offset_baseline, *a, **k):
                rec["offset_line"] = offset_baseline.tolist()
                try:
                    rec["roi"] = coords(unary_union([geom.Polygon(np.concatenate((baseline, env_up[::-1]))),
                                                     geom.Polygon(np.concatenate((baseline, env_bottom[::-1])))]).exterior.coords)
                except Exception as exc:  # noqa: BLE001
                    rec["roi_error"] = str(exc)[:200]
                return inner["_extract_patch"](env_up, env_bottom, baseline, offset_baseline, *a, **k)

            S._calc_roi, S._rotate, S.make_polygonal_mask, S.binary_erosion, S._calc_seam, S._extract_patch = roi, rotate, poly_mask, erosion, seam, patch
            ls = geom.LineString(bl)
            others = baselines[:i] + baselines[i + 1 :]
            for reg, pol in zip(regions, reg_pols, strict=True):
                if S.is_in_region(ls, pol):
                    others.append(reg)
            try:
                pol = S.calculate_polygonal_environment(baselines=[bl], im_feats=feats, suppl_obj=others, topline=meta["topline"], raise_on_error=True)[0]
                rec["polygon"] = np.asarray(pol).tolist()
            except Exception as exc:  # noqa: BLE001 - Kraken drops the line; say why
                rec["polygon"], rec["error"] = None, str(exc)[:300]
            if len(rec["seams"]) == 2 and all("seam" in s for s in rec["seams"]):
                rec["upper"] = line_of(np.array(rec["seams"][0]["seam"]), "right")
                rec["bottom"] = line_of(np.array(rec["seams"][1]["seam"]), "left")
            lines.append(rec)
        for name, fn in inner.items():
            setattr(S, name, fn)
        (out / f"{stem}.outlines.json").write_text(json.dumps(lines))
        failed = sum(r["polygon"] is None for r in lines)
        same = sum(r["polygon"] == p for r, p in zip(lines, meta["polygons"], strict=True))
        print(f"{stem}: {len(lines)} lines, {failed} without an outline, {same} outlines as dump_golden saved them", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
