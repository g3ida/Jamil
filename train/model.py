"""The painter: a pretrained encoder, a small pyramid back up to a quarter of the page, four planes out.

Kraken's network is five convolutions and four LSTMs, and the LSTMs are where a
GPU waits: a step a pixel of the page's width and height. This has none. The
encoder is whichever `timm` offers by the name given — EfficientViT-B1 unless
told otherwise, whose attention sees the whole page at a cost that grows with
its area and not its square — and what it makes at each scale is added back
together top down, as a feature pyramid does, to the scale Kraken paints at.
"""

from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import nn

# ImageNet's, which the encoders were pretrained under.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class Painter(nn.Module):
    def __init__(self, encoder: str = "efficientvit_b1", planes: int = 4, width: int = 96, weights: str | None = None):
        super().__init__()
        overlay = {"file": weights} if weights else None
        self.encoder = timm.create_model(encoder, pretrained=weights is not None, pretrained_cfg_overlay=overlay, features_only=True)
        info = self.encoder.feature_info
        # From a quarter of the page down: a half is too fine to be worth its cost.
        self.used = [i for i, by in enumerate(info.reduction()) if by >= 4]
        if info.reduction()[self.used[0]] != 4:
            raise ValueError(f"{encoder} has no features at a quarter of the page")
        self.lateral = nn.ModuleList(nn.Conv2d(info.channels()[i], width, 1) for i in self.used)
        self.smooth = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(width, planes, 1)
        # Lines are a few pixels in a hundred: starting every plane at "nothing here" spares the first epoch.
        nn.init.constant_(self.head.bias, -4.0)

    def forward(self, page: torch.Tensor) -> torch.Tensor:
        """Logits at a quarter of `page`, which is normalised RGB, (n, 3, h, w)."""
        features = self.encoder(page)
        features = [features[i] for i in self.used]
        top = self.lateral[-1](features[-1])
        for feature, lateral in zip(features[-2::-1], list(self.lateral)[-2::-1]):
            top = lateral(feature) + F.interpolate(top, size=feature.shape[-2:], mode="nearest")
        return self.head(self.smooth(top))


class FromBytes(nn.Module):
    """The painter as the detector calls it: the page's bytes in, (1, h, w, 3), logits out."""

    def __init__(self, painter: Painter):
        super().__init__()
        self.painter = painter
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) * 255.0)

    def forward(self, page: torch.Tensor) -> torch.Tensor:
        planes = page.permute(0, 3, 1, 2).float()
        return self.painter((planes - self.mean) / self.std)
