"""dump_golden.py MODEL OUT DEVICE IMAGE…: what Kraken makes of each page, stage by stage, as files.

A port is checked against these, one stage at a time, with no Python in the loop:

  <stem>.heatmap.npy   float32 (classes, H, W): the network's output after the sigmoid
  <stem>.scal.npy      uint8 (H, W): the page in grey at the network's size
  <stem>.sato.npy      float32 (H, W): the ridge filter over the baseline map
  <stem>.skel.npy      uint8 (H, W): the skeleton of the thresholded ridges
  <stem>.json          scale, class map, regions (as traced, in page pixels, and
                       as the polygonizer is handed them), baselines and
                       outlines in heatmap pixels, reading order, and the lines
                       as Kraken returns them — checked against `predict`
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).parent))
import kraken_ref as R  # noqa: E402

warnings.filterwarnings("ignore")


def main() -> int:
    model, out, device, images = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4:]
    out.mkdir(parents=True, exist_ok=True)
    net = R.load(model, device)
    for path in images:
        stem = Path(path).stem
        if (out / f"{stem}.json").exists():
            continue
        im = Image.open(path).convert("RGB")
        got = R.stages(net, im)
        heatmap = got.pop("heatmap")
        np.save(out / f"{stem}.heatmap.npy", heatmap.astype(np.float32))
        np.save(out / f"{stem}.scal.npy", got.pop("scal_im").astype(np.uint8))
        baseline_class = next(iter(got["cls_map"]["baselines"].values()))
        ridges = R.S.filters.sato(heatmap[baseline_class], black_ridges=False, mode="constant")
        np.save(out / f"{stem}.sato.npy", ridges.astype(np.float32))
        np.save(out / f"{stem}.skel.npy", skeletonize(ridges > 0.17).astype(np.uint8))
        got["image"] = str(path)
        got["size"] = list(im.size)
        (out / f"{stem}.json").write_text(json.dumps(got))
        # The stages above are Kraken's own inference taken apart: put together
        # again they must give what Kraken gives.
        whole = [{"baseline": line.baseline, "boundary": line.boundary} for line in net.predict(im=im).lines]
        faithful = json.loads(json.dumps(whole)) == json.loads(json.dumps(got["lines"]))
        print(stem, heatmap.shape, len(got["baselines"]), "lines;", "as Kraken's predict" if faithful else "NOT AS KRAKEN'S PREDICT", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
