"""train.py --train LISTS --val LIST --out DIR: teach a painter the lines of the pages listed.

    python train.py --train "train.txt print.txt@3" --val val.txt --out runs/b1 \\
        --weights weights/efficientvit_b1.r288_in1k/model.safetensors

Each listing is a file of PAGE XML paths; `@3` draws on one three times as
often as its size alone would. What is saved is `last.pt`, after every epoch
and every `SAVED_EVERY` steps within one, and `best.pt` when the validation
pages' baselines are painted better than before, both holding the averaged
weights `export.py` writes out. `--resume` goes on from `last.pt`, in the
middle of an epoch if that is where it was written.
"""

from __future__ import annotations

import argparse
import copy
from functools import partial
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import Pages, listed
from loss import held_against
from model import Painter
from targets import BASELINE, CLASSES, PLANES, REGION


# Steps between two savings within an epoch. A run on a machine that kills a loading worker every
# quarter of an hour lost half an epoch each time; saved this often it loses half a minute.
SAVED_EVERY = 100


# Steps at the end of an epoch in which the averaged weights' batch norms are measured.
MEASURED = 100


def measuring(painter: torch.nn.Module) -> None:
    """Has `painter`'s batch norms forget what they hold and take the plain mean of what they are shown next.

    A batch norm's statistics are of what its own weights pass to it. The averaged weights were kept
    with the statistics of the weights being trained, copied over at every step, and the two are not
    the same network: at epoch 8 the trained weights painted the validation pages at a loss of 0.072,
    their average with those borrowed statistics at 0.213 — worse every epoch while training went
    well — and the average with statistics of its own at 0.069."""
    painter.eval()
    for part in painter.modules():
        if isinstance(part, torch.nn.modules.batchnorm._BatchNorm):
            part.reset_running_stats()
            part.momentum = None
            # Only these are put to learning: whatever else behaves differently in training is left as it will be used.
            part.train()


def saved(what: dict, to: Path) -> None:
    """Written beside `to` and moved over it: a run that dies while writing leaves the last one whole."""
    beside = to.with_name(to.name + ".part")
    torch.save(what, beside)
    os.replace(beside, to)


def traced(trail: Path, _worker: int) -> None:
    """Has a loading worker write where it stood if a library kills it: the run's log says only that one died."""
    import faulthandler
    import os

    faulthandler.enable(file=open(trail / f"{os.getpid()}.trace", "w"), all_threads=True)


def quartered(target: torch.Tensor, counted: torch.Tensor, size) -> tuple[torch.Tensor, torch.Tensor]:
    """Targets at the size the network paints at. Kraken enlarges the painting
    to the target and compares pixel by pixel; the mean of a target's pixels
    under each painted one is the same loss, without the enlarging."""
    return F.adaptive_avg_pool2d(target, size), (F.adaptive_avg_pool2d(counted, size) > 0.999).float()


@torch.no_grad()
def validated(model, pages: DataLoader, device) -> dict:
    """How the validation pages' baselines are painted: pixels right of those painted, and of those wanted."""
    model.eval()
    # Every page here is another size, and cuDNN's trying-out of each convolution is per size: off while checking.
    torch.backends.cudnn.benchmark = False
    hit = painted = wanted = 0.0
    loss = n = 0.0
    for pixels, target, counted in pages:
        pixels, target, counted = pixels.to(device).contiguous(memory_format=torch.channels_last), target.to(device), counted.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(pixels)
        target, counted = quartered(target, counted, logits.shape[-2:])
        loss += held_against(logits.float(), target, counted, BASELINE, 0.0)["loss"].item()
        n += 1
        on, want = (logits[:, BASELINE] > 0).float(), (target[:, BASELINE] > 0.5).float()
        hit, painted, wanted = hit + (on * want).sum().item(), painted + on.sum().item(), wanted + want.sum().item()
    precision, recall = hit / max(painted, 1.0), hit / max(wanted, 1.0)
    return {"val_loss": loss / max(n, 1.0), "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-9)}


def initial(path: Path) -> dict:
    """Initial weights for fine-tuning, from a `.safetensors` file or from a checkpoint saved by this script."""
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    return torch.load(path, map_location="cpu", weights_only=False)["averaged"]


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--train", required=True)
    a.add_argument("--val", required=True)
    a.add_argument("--out", required=True, type=Path)
    a.add_argument("--encoder", default="efficientvit_b1")
    a.add_argument("--weights", default=None, help="the encoder's pretrained weights, a .safetensors file")
    a.add_argument("--epochs", type=int, default=40)
    a.add_argument("--pages", type=int, default=0, help="pages an epoch; 0 for as many as are listed")
    a.add_argument("--batch", type=int, default=8)
    a.add_argument("--crop", type=int, default=1024)
    a.add_argument("--lr", type=float, default=5e-4)
    a.add_argument("--connected", type=float, default=0.3, help="weight of clDice on the baseline; 0 for cross-entropy alone")
    a.add_argument("--workers", type=int, default=12)
    a.add_argument("--resume", action="store_true")
    a.add_argument("--init", type=Path, default=None, help="initial weights for fine-tuning (.safetensors or a checkpoint .pt)")
    args = a.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    train_pages, weights = listed(args.train)
    val_pages, _ = listed(args.val)
    per_epoch = args.pages or len(train_pages)
    sampler = WeightedRandomSampler(weights, num_samples=per_epoch, replacement=True)
    trail = args.out / "faults"
    trail.mkdir(exist_ok=True)
    loader = DataLoader(
        Pages(train_pages, args.crop, trail=trail), batch_size=args.batch, sampler=sampler, num_workers=args.workers, drop_last=True,
        persistent_workers=args.workers > 0, worker_init_fn=partial(traced, trail),
    )
    val_loader = DataLoader(Pages(val_pages, train=False), batch_size=1, num_workers=min(4, args.workers))

    model = Painter(args.encoder, PLANES, weights=None if args.init else args.weights)
    if args.init:
        model.load_state_dict(initial(args.init))
    else:
        # Every plane starts at "nothing here" but the region's, which is most of a page.
        with torch.no_grad():
            model.head.bias[REGION] = 0.0
    model = model.to(device).to(memory_format=torch.channels_last)
    averaged = copy.deepcopy(model).eval()
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * len(loader)
    warm = min(500, steps // 20 + 1)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    # `begun`: the steps of the first epoch here that an earlier go already took.
    start, begun, best = 0, 0, -1.0
    if args.resume and (args.out / "last.pt").exists():
        kept = torch.load(args.out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(kept["model"])
        averaged.load_state_dict(kept["averaged"])
        optimiser.load_state_dict(kept["optimiser"])
        schedule.load_state_dict(kept["schedule"])
        start, begun, best = kept["epoch"] + 1, 0, kept["best"]
        if kept.get("step") is not None:
            start, begun = kept["epoch"], kept["step"] + 1
    about = {"encoder": args.encoder, "classes": CLASSES, "height": 1800, "connected": args.connected}
    print(f"{len(train_pages)} pages to learn from, {len(val_pages)} to check on; {per_epoch} an epoch, {args.epochs} epochs; {args.encoder} on {device}", flush=True)

    def state() -> dict:
        return {"model": model.state_dict(), "averaged": averaged.state_dict(), "optimiser": optimiser.state_dict(), "schedule": schedule.state_dict(), "about": about}

    for epoch in range(start, args.epochs):
        model.train()
        # Every square in training is one size: worth cuDNN's finding the quickest way once.
        torch.backends.cudnn.benchmark = True
        clock, seen, sums = time.time(), 0, {}
        # The averaged weights' batch norms are measured on the epoch's last squares, as the weights then are.
        measured_from = max(begun, len(loader) - MEASURED)
        for step, (pixels, target, counted) in enumerate(loader, start=begun):
            if step >= len(loader):
                break
            pixels = pixels.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            target, counted = target.to(device, non_blocking=True), counted.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(pixels)
            target, counted = quartered(target, counted, logits.shape[-2:])
            parts = held_against(logits.float(), target, counted, BASELINE, args.connected)
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(parts["loss"]).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            # The weights that are kept: a running mean of where training has been, steadier than where it is.
            with torch.no_grad():
                keep = min(0.999, (1 + epoch * len(loader) + step) / (10 + epoch * len(loader) + step))
                for mean, now in zip(averaged.parameters(), model.parameters()):
                    mean.lerp_(now, 1 - keep)
                if step == measured_from:
                    measuring(averaged)
                if step >= measured_from:
                    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                        averaged(pixels)
            seen += pixels.shape[0]
            for name, value in parts.items():
                sums[name] = sums.get(name, 0.0) + value.item()
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(loader)} " + " ".join(f"{k} {v / (step + 1 - begun):.4f}" for k, v in sums.items()), flush=True)
            if step % SAVED_EVERY == SAVED_EVERY - 1 and step + 1 < len(loader):
                saved({**state(), "epoch": epoch, "step": step, "best": best}, args.out / "last.pt")
        averaged.eval()
        # The means are of the steps taken in this go: an epoch gone on with says nothing of its beginning.
        taken = max(1, len(loader) - begun)
        begun = 0
        row = {"epoch": epoch, "pages_a_second": seen / (time.time() - clock), **{k: v / taken for k, v in sums.items()}, **validated(averaged, val_loader, device)}
        with (args.out / "log.jsonl").open("a") as log:
            log.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        saved({**state(), "epoch": epoch, "best": max(best, row["f1"])}, args.out / "last.pt")
        if row["f1"] > best:
            best = row["f1"]
            saved({"averaged": averaged.state_dict(), "about": about, "epoch": epoch, "f1": best}, args.out / "best.pt")


if __name__ == "__main__":
    main()
