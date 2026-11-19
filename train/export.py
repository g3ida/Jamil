"""export.py CHECKPOINT OUT.onnx: a trained painter as the file the detector loads.

The graph takes what `jamil` hands a network — the page's bytes, (1, h, w, 3) —
and gives the four planes' logits at a quarter of its size. What the detector
needs to know besides, which plane is which and the height to read at, rides
in the file's own metadata under `jamil`.
"""

from __future__ import annotations

import json
import sys

import torch

from model import FromBytes, Painter


def main() -> None:
    checkpoint, out = sys.argv[1], sys.argv[2]
    kept = torch.load(checkpoint, map_location="cpu", weights_only=False)
    planes = sum(len(kind) for kind in kept["about"]["classes"].values())
    painter = Painter(kept["about"]["encoder"], planes)
    painter.load_state_dict(kept["averaged"])
    whole = FromBytes(painter).eval()
    page = torch.zeros(1, 256, 192, 3, dtype=torch.uint8)
    torch.onnx.export(
        whole, page, out, input_names=["page"], output_names=["maps"], opset_version=17, dynamo=False,
        dynamic_axes={"page": {1: "height", 2: "width"}, "maps": {2: "rows", 3: "columns"}},
    )
    import onnx

    model = onnx.load(out)
    entry = model.metadata_props.add()
    entry.key, entry.value = "jamil", json.dumps({"height": kept["about"]["height"], "classes": kept["about"]["classes"]})
    onnx.save(model, out)
    print(f"{out}: {kept['about']['encoder']}, epoch {kept.get('epoch')}, validation F1 {kept.get('f1')}")


if __name__ == "__main__":
    main()
