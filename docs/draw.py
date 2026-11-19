"""draw.py PAGE LINES.json OUT [--height N]: a page with the lines Jamil found on it.

Each line's outline is shaded and its baseline drawn over it, a colour a line in
the order the lines are read, so that a line run across a column, or a line
left in pieces, shows at a glance. LINES.json is what `jamil --out` writes for
the page. Needs Pillow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

COLOURS = [(214, 39, 40), (31, 119, 180), (44, 160, 44), (148, 103, 189), (255, 127, 14), (23, 190, 207)]


def drawn(page: Path, lines: list[dict]) -> Image.Image:
    image = Image.open(page).convert("RGBA")
    shade = Image.new("RGBA", image.size, (0, 0, 0, 0))
    pen = ImageDraw.Draw(shade)
    # A baseline as wide as a stroke of print on the page, whatever its size.
    width = max(2, round(image.height / 450))
    for i, line in enumerate(lines):
        colour = COLOURS[i % len(COLOURS)]
        if len(line["boundary"]) >= 3:
            pen.polygon([tuple(p) for p in line["boundary"]], fill=colour + (46,), outline=colour + (150,))
        pen.line([tuple(p) for p in line["baseline"]], fill=colour + (255,), width=width)
    return Image.alpha_composite(image, shade).convert("RGB")


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("page", type=Path)
    a.add_argument("lines", type=Path)
    a.add_argument("out", type=Path)
    a.add_argument("--height", type=int, default=1400, help="the picture's height in pixels")
    args = a.parse_args()
    picture = drawn(args.page, json.loads(args.lines.read_text())["lines"])
    k = args.height / picture.height
    picture.resize((round(picture.width * k), args.height), Image.LANCZOS).save(args.out, quality=80, optimize=True)


if __name__ == "__main__":
    main()
