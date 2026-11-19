# Training and fine-tuning

The scripts here train the network Jamil uses: an EfficientViT-B1 encoder
(from `timm`) with a light decoder that paints the four heatmaps Jamil's
tracing reads — line starts, line ends, baselines and text regions — at a
quarter of the page's resolution. The built-in model's weights are in
[`models/jamil-b1.safetensors`](../models/jamil-b1.safetensors); fine-tuning
starts from them.

## Requirements

A CUDA GPU, and:

```bash
pip install torch timm shapely pillow safetensors onnx
```

## 1. Prepare pages

Each page is an image with a PAGE XML file beside it (`imageFilename` is read
relative to the XML file). What is used from the XML:

- `TextLine` → `Baseline points` — the lines to learn.
- `TextLine` → `Coords points` — the line outlines.
- `TextRegion` → `Coords points` — the text regions.

Two markers let a page be labelled in part. The loss ignores what they cover:

- `custom="lines-only"` on `<Page>`: only the lines given are labelled; the
  rest of the page is not used as background.
- `<UnknownRegion>`: an area to leave out (a stamp over the text, a margin
  that is not labelled).

## 2. List them

A list is a text file with one PAGE XML path per line. Keep a separate
validation list of pages that are not trained on. `--train` takes several
lists; `list.txt@3` draws pages from a list three times as often as its size
alone would.

## 3. Fine-tune

```bash
python train.py \
    --init ../models/jamil-b1.safetensors \
    --train "lists/train.txt" --val lists/val.txt --out runs/finetune \
    --lr 1e-4 --epochs 10 --pages 2000 --batch 8
```

- `--init` starts from the built-in weights (or from a checkpoint of an
  earlier run).
- `--lr 1e-4` suits fine-tuning; the built-in model was trained from ImageNet
  weights at `5e-4`.
- `--pages` is the number of pages drawn per epoch; `--epochs` how many.
  With few pages, raise `--pages` rather than repeating short epochs.
- `--batch` and `--crop` (the square cut from each page, 1024 by default)
  set the GPU memory used.
- `--connected 0.3` (the default) adds clDice on the baseline, which keeps
  lines from breaking; `0` trains on cross-entropy alone.
- `--resume` continues an interrupted run from `last.pt`.

A run writes `last.pt` (every epoch, and periodically within one), `best.pt`
(the epoch with the best baseline F1 on the validation pages) and
`log.jsonl` (losses and validation scores per epoch). Pixel F1 is only a
guide: check the lines themselves on validation pages before choosing between
the two checkpoints.

## 4. Export and use

```bash
python export.py runs/finetune/best.pt finetune.onnx
```

```bash
jamil finetune.onnx page.png
```

From Python, `jamil.Model("finetune.onnx")`.
