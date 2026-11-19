"""profile_kraken.py MODEL DEVICE IMAGE…: where Kraken spends a page's time, stage by stage.

Each page is run twice and the second run is the one reported: the first pays
for CUDA start-up and lazy imports. Stages nest — `lines` holds `lines.*`,
`poly` holds `poly.*` — so only the top-level rows add up to the total.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import kraken_ref as R  # noqa: E402

warnings.filterwarnings("ignore")
TOP = ("network", "regions", "lines", "poly.features", "poly", "order")


def main() -> int:
    model, device, images = sys.argv[1], sys.argv[2], sys.argv[3:]
    net = R.load(model, device)
    R.instrument()
    for path in images:
        im = Image.open(path).convert("RGB")
        R.stages(net, im)
        R.TIMES.clear()
        R.CALLS.clear()
        started = time.perf_counter()
        out = R.stages(net, im)
        total = time.perf_counter() - started
        h, w = out["heatmap"].shape[1:]
        print(f"\n{Path(path).name}: page {im.size[0]}x{im.size[1]}, heatmap {w}x{h}, "
              f"{len(out['baselines'])} lines, {len(out['regions'])} regions, total {total:.2f}s")
        for name in sorted(R.TIMES, key=lambda n: (n.split(".")[0] not in TOP, TOP.index(n) if n in TOP else 99, n)):
            pad = "  " if name in TOP else "      "
            print(f"{pad}{name:26s} {R.TIMES[name] * 1000:8.0f} ms  {100 * R.TIMES[name] / total:5.1f}%  x{R.CALLS[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
