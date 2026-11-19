"""Kraken's segmenter taken apart, so each stage can be timed and its output kept.

`stages(net, im)` follows `VGSLSegmentationInference._segmentation_pred` step by
step and returns everything a port has to reproduce: the heatmap the network
paints, the scaled grey page, the baselines and outlines in the heatmap's
pixels, the regions, and the lines as Kraken hands them back.

Run under an environment with Kraken 7 installed.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import shapely.geometry as geom
import torch
from kraken.configs import SegmentationInferenceConfig
from kraken.ketos.util import to_ptl_device
from kraken.lib import segmentation as S
from kraken.lib.vgsl import spred
from kraken.models import load_models
from scipy.ndimage import gaussian_filter
from skimage.filters import sobel

TIMES: dict[str, float] = defaultdict(float)
CALLS: dict[str, int] = defaultdict(int)


@contextmanager
def clock(name: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        TIMES[name] += time.perf_counter() - started
        CALLS[name] += 1


def _timed(holder, attr: str, name: str) -> None:
    inner = getattr(holder, attr)

    def wrapper(*args, **kwargs):
        with clock(name):
            return inner(*args, **kwargs)

    setattr(holder, attr, wrapper)


def instrument() -> None:
    """Clocks round the functions inside `kraken.lib.segmentation` worth telling apart."""
    _timed(S.filters, "sato", "lines.sato")
    _timed(S, "skeletonize", "lines.skeletonize")
    _timed(S, "convolve2d", "lines.endpoints")
    _timed(S.LineMCP, "find_costs", "lines.mcp")
    _timed(S, "_extend_boundaries", "lines.extend")
    _timed(S, "maximum_filter", "lines.maxfilter")
    _timed(S, "_calc_roi", "poly.roi")
    _timed(S, "_calc_seam", "poly.seam")
    _timed(S, "_extract_patch", "poly.patch(with seam)")


def load(model: str, device: str = "cuda:0"):
    net = load_models(model)[0]
    accelerator, devices = to_ptl_device(device)
    config = SegmentationInferenceConfig(text_direction="horizontal-rl", accelerator=accelerator, device=devices)
    net.prepare_for_inference(config)
    return net


def stages(net, im) -> dict:
    """One page through every stage; what each produced."""
    with clock("network"):
        # `_segmentation_pred` runs under inference mode; called from outside it,
        # the heatmap still carries its graph and cannot be turned into an array.
        with torch.inference_mode(), net._fabric.init_tensor():
            rets = net._compute_segmentation_map(im)
    heatmap, cls_map, scale, scal_im = rets["heatmap"], rets["cls_map"], rets["scale"], rets["scal_im"]

    with clock("regions"):
        raw_regions = [r for idx in cls_map["regions"].values() for r in S.vectorize_regions(heatmap[idx])]
        found = spred.vec_regions(**rets)
    regions = [r for regs in found.values() for r in regs]
    # `_segmentation_pred` hands the polygonizer its regions scaled to the page
    # and back again, cut to whole numbers each way: a pixel off the traced ones.
    line_regs = S.scale_regions([r.boundary for r in regions], 1 / scale)

    st, end = cls_map["aux"]["_start_separator"], cls_map["aux"]["_end_separator"]
    with clock("lines"):
        baselines = []
        for name, idx in cls_map["baselines"].items():
            baselines += [(name, bl) for bl in S.vectorize_lines(heatmap[(st, end, idx), :, :], text_direction="horizontal")]

    with clock("poly.features"):
        im_feats = gaussian_filter(sobel(scal_im), 0.5)
    topline = bool(net.user_metadata.get("topline", False))
    polygons = []
    with clock("poly"):
        reg_pols = [geom.Polygon(r) for r in line_regs]
        for i, (_, bl) in enumerate(baselines):
            ls = geom.LineString(bl)
            others = [b for _, b in baselines[:i] + baselines[i + 1 :]]
            for reg, pol in zip(line_regs, reg_pols, strict=True):
                if S.is_in_region(ls, pol):
                    others.append(reg)
            polygons.append(
                S.calculate_polygonal_environment(baselines=[bl], im_feats=im_feats, suppl_obj=others, topline=topline)[0]
            )

    with clock("order"):
        kept = [(bl, pol) for (_, bl), pol in zip(baselines, polygons, strict=True) if pol is not None]
        scaled = S.scale_polygonal_lines(kept, scale) if kept else []
        blls = []
        for k, (bl, pol) in enumerate(scaled):
            ls = geom.LineString(bl)
            inside = [r.id for r in regions if S.is_in_region(ls, geom.Polygon(r.boundary))]
            blls.append(spred.BaselineLine(id=f"l{k}", baseline=bl, boundary=pol, tags=None, regions=inside))
        order = S.polygonal_reading_order(lines=blls, regions=regions, text_direction="rl") if blls else []

    return {
        "heatmap": heatmap,
        "scal_im": scal_im,
        "scale": [float(v) for v in scale],
        "cls_map": cls_map,
        "topline": topline,
        "regions_traced": [np.asarray(r).tolist() for r in raw_regions],
        "regions_page": [np.asarray(r.boundary).tolist() for r in regions],
        "regions": [np.asarray(r).tolist() for r in line_regs],
        "baselines": [np.asarray(bl).tolist() for _, bl in baselines],
        "polygons": [np.asarray(p).tolist() if p is not None else None for p in polygons],
        "order": [int(i) for i in order],
        "lines": [{"baseline": blls[i].baseline, "boundary": blls[i].boundary} for i in order],
    }
