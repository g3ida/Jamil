"""Pages for training: an image beside a PAGE XML file, cut and varied.

Every source ends in the one shape — per line a baseline, a region round the
lines — whatever it started as. A page is brought to the height the network
reads at, give or take: writing is not all one size, and a network shown every
page at 1800 pixels learns that size and no other. Then it is turned a little,
a square is cut from it, and its tone is varied, the targets drawn after the
turning and scaling so that a baseline is the same band at every size.

`<UnknownRegion>` in a page's XML marks ground nobody vouches for: pages
labelled by a machine and checked only in part. The painting is not judged
there, which lets such a page teach what is sure of it.
"""

from __future__ import annotations

import math
import os
import random
import re
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset

from fill import polygon
from model import MEAN, STD
from targets import LINE_WIDTH, render

# The height Kraken reads a page at, and the detector after it.
HEIGHT = 1800


def _points(text: str) -> list[tuple[float, float]]:
    values = [float(v) for v in re.split(r"[\s,]+", text.strip()) if v]
    return list(zip(values[0::2], values[1::2]))


def read_page(xml: Path) -> dict:
    """A page's image path, baselines, text regions and unvouched ground."""
    text = xml.read_text(encoding="utf-8")
    image = xml.parent / re.search(r'imageFilename="([^"]+)"', text).group(1)
    baselines = [_points(b) for b in re.findall(r'<Baseline points="([^"]+)"', text)]
    regions = [_points(c) for c in re.findall(r'<TextRegion\b[^>]*>\s*<Coords points="([^"]+)"', text)]
    unknown = [_points(c) for c in re.findall(r'<UnknownRegion\b[^>]*>\s*<Coords points="([^"]+)"', text)]
    # A page marked `lines-only` has lines nobody marked: only the ground its own lines stand on is vouched for.
    outlines = None
    if re.search(r'<Page\b[^>]*custom="lines-only"', text):
        outlines = [_points(c) for c in re.findall(r'<TextLine\b[^>]*>\s*<Coords points="([^"]+)"', text)]
    return {"image": image, "baselines": [b for b in baselines if len(b) >= 2], "regions": regions, "unknown": unknown, "outlines": outlines}


def _moved(points, k: float, turn: float, centre, shift) -> list[tuple[float, float]]:
    """Points scaled by `k`, turned by `turn` radians about `centre` (as the image is), then shifted."""
    c, s = math.cos(turn), math.sin(turn)
    out = []
    for x, y in points:
        x, y = x * k - centre[0], y * k - centre[1]
        out.append((c * x + s * y + centre[0] - shift[0], -s * x + c * y + centre[1] - shift[1]))
    return out


def _grown(outline, by: float) -> list[tuple[float, float]]:
    """An outline pushed out from its own middle, by about `by` pixels along its
    longer side and a sixth of that across: a line's markers stand past its ends,
    and above and below it stands the next line, which nobody may have marked."""
    points = np.array(outline, dtype=np.float64)
    low, high = points.min(axis=0), points.max(axis=0)
    centre, half = (low + high) / 2, np.maximum((high - low) / 2, 1.0)
    reach = np.where(half >= half.max(), by, by / 6)
    return [tuple(p) for p in centre + (points - centre) * (1 + reach / half)]


class Pages(Dataset):
    """`listing`: PAGE XML paths. In training a varied square of each page; otherwise the whole page as the detector would see it."""

    def __init__(self, listing: list[Path], crop: int = 1024, train: bool = True, scales: tuple[float, float] = (0.65, 1.4), trail: Path | None = None):
        self.listing, self.crop, self.train, self.scales = listing, crop, train, scales
        # Where each worker notes the page it is on. A worker that dies in a library's own code dies
        # without a word; the note is how the page that killed it is known afterwards.
        self.trail = trail

    def __len__(self) -> int:
        return len(self.listing)

    def __getitem__(self, i: int):
        # Drawn again where the drawing went wrong. A run died twice of a pixel's row that was the
        # right row with one bit set far above it — 2⁴⁴ + 354 in a square of 1024 — which nothing
        # here can work out, and the machine it ran on can get wrong. The same page drawn again
        # is another draw; a page that fails three times is failing for a reason of its own.
        for go in range(3):
            try:
                return self._drawn(i)
            except IndexError:
                if go == 2:
                    raise
                if self.trail is not None:
                    with (self.trail / "redrawn.txt").open("a") as noted:
                        noted.write(f"{self.listing[i]}\n{traceback.format_exc()}\n")

    def _drawn(self, i: int):
        if self.trail is not None:
            (self.trail / f"{os.getpid()}.at").write_text(f"{self.listing[i]}\n")
        page = read_page(self.listing[i])
        image = ImageOps.exif_transpose(Image.open(page["image"])).convert("RGB")
        k = HEIGHT / image.height
        turn = 0.0
        if self.train:
            k *= math.exp(random.uniform(math.log(self.scales[0]), math.log(self.scales[1])))
            if random.random() < 0.3:
                turn = math.radians(random.uniform(-3.0, 3.0))
        w, h = max(8, round(image.width * k)), max(8, round(image.height * k))
        image = image.resize((w, h), Image.BILINEAR)
        paper = tuple(int(v) for v in np.median(np.asarray(image.resize((32, 32))).reshape(-1, 3), axis=0))
        if turn:
            image = image.rotate(math.degrees(turn), resample=Image.BILINEAR, fillcolor=paper)
        shift, size = (0, 0), (w, h)
        if self.train:
            # A square of the page, from anywhere a square fits; a page smaller than the square stands on paper.
            size = (self.crop, self.crop)
            shift = (random.randint(0, max(0, w - self.crop)), random.randint(0, max(0, h - self.crop)))
            cut = Image.new("RGB", size, paper)
            cut.paste(image.crop((shift[0], shift[1], min(w, shift[0] + self.crop), min(h, shift[1] + self.crop))), (0, 0))
            image = self._toned(cut)
        moved = lambda group: [_moved(points, k, turn, (w / 2, h / 2), shift) for points in group]  # noqa: E731
        target = render(size, moved(page["baselines"]), moved(page["regions"]))
        counted = np.ones((1, size[1], size[0]), dtype=np.float32)

        def marked(grounds, value):
            for ground in grounds:
                if len(ground) >= 3:
                    points = np.array(ground)
                    rr, cc = polygon(points[:, 1], points[:, 0], shape=counted.shape[1:])
                    counted[0, rr, cc] = value

        if page["outlines"] is not None:
            # Only the lines given, each with a margin about it for its markers, which stand past its ends.
            counted[:] = 0.0
            marked([_grown(outline, 1.5 * LINE_WIDTH) for outline in moved(page["outlines"])], 1.0)
        marked(moved(page["unknown"]), 0.0)
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
        pixels = (pixels - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)
        return pixels, torch.from_numpy(target).float(), torch.from_numpy(counted)

    @staticmethod
    def _toned(image: Image.Image) -> Image.Image:
        """The same page as another scanner might have made it."""
        if random.random() < 0.25:
            image = ImageOps.grayscale(image).convert("RGB")
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        gain, lift, gamma = random.uniform(0.8, 1.2), random.uniform(-0.08, 0.08), math.exp(random.uniform(-0.3, 0.3))
        pixels = np.clip((np.clip(pixels, 0, 1) ** gamma - 0.5) * gain + 0.5 + lift, 0, 1)
        image = Image.fromarray((pixels * 255).astype(np.uint8))
        if random.random() < 0.15:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 1.2)))
        return image


def listed(spec: str) -> tuple[list[Path], list[float]]:
    """`a.txt b.txt@3`: the pages of each listing, and how heavily each is drawn on."""
    pages, weights = [], []
    for part in spec.split():
        name, _, weight = part.partition("@")
        found = [Path(line) for line in Path(name).read_text().split() if line]
        pages += found
        weights += [float(weight or 1.0)] * len(found)
    return pages, weights
