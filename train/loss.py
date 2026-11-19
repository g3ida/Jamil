"""What a painting is held against: every pixel, and whether the lines hold together.

Pixel by pixel (cross-entropy, as Kraken trains) a baseline broken for the
width of a word costs almost nothing: a dozen pixels of thousands. To the
tracing it costs a line. clDice (Shit et al., 2021) counts what cross-entropy
cannot see — how much of the *skeleton* of what should be there was painted,
and how much of the skeleton of what was painted should be there — so a gap
in a line and a bridge between two lines are each paid for in full.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _thinned(x: torch.Tensor, rounds: int) -> torch.Tensor:
    """A soft skeleton: what an opening takes away, round after round of wearing down."""

    def worn(v):
        return torch.minimum(-F.max_pool2d(-v, (3, 1), 1, (1, 0)), -F.max_pool2d(-v, (1, 3), 1, (0, 1)))

    def opened(v):
        return F.max_pool2d(worn(v), 3, 1, 1)

    skeleton = F.relu(x - opened(x))
    for _ in range(rounds):
        x = worn(x)
        remains = F.relu(x - opened(x))
        skeleton = skeleton + F.relu(remains - skeleton * remains)
    return skeleton


def cl_dice(painted: torch.Tensor, wanted: torch.Tensor, rounds: int = 3, eps: float = 1.0) -> torch.Tensor:
    """1 − clDice for a batch of single planes in 0–1, (n, 1, h, w)."""
    skeleton_painted, skeleton_wanted = _thinned(painted, rounds), _thinned(wanted, rounds)
    dims = (1, 2, 3)
    on_the_lines = ((skeleton_painted * wanted).sum(dims) + eps) / (skeleton_painted.sum(dims) + eps)
    lines_found = ((skeleton_wanted * painted).sum(dims) + eps) / (skeleton_wanted.sum(dims) + eps)
    return (1.0 - 2.0 * on_the_lines * lines_found / (on_the_lines + lines_found)).mean()


def held_against(logits: torch.Tensor, wanted: torch.Tensor, counted: torch.Tensor, baseline: int, connected: float) -> dict:
    """The loss for logits (n, planes, h, w) against targets in 0–1 of the same
    shape; `counted` (n, 1, h, w) is 0 where nobody vouches for the target and
    the painting is not to be judged. `connected` weighs clDice on the baseline."""
    each = F.binary_cross_entropy_with_logits(logits, wanted, reduction="none") * counted
    pixels = each.sum() / (counted.sum() * logits.shape[1]).clamp(min=1.0)
    out = {"pixels": pixels, "loss": pixels}
    if connected > 0:
        # Where nothing is vouched for, the target stands in for the painting: no cost either way.
        painted = torch.sigmoid(logits[:, baseline : baseline + 1].float()) * counted + wanted[:, baseline : baseline + 1] * (1 - counted)
        out["connected"] = cl_dice(painted, wanted[:, baseline : baseline + 1].float())
        out["loss"] = pixels + connected * out["connected"]
    return out
