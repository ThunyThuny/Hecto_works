"""Save four-panel comparisons for all 50 held-out test pairs."""

import argparse
import csv
import gc
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from evaluate_adapters import DEFAULT_BASELINE, DEFAULT_CANDIDATE
from train_adapter import DATA_ROOT, DEMO_ROOT, ROOT

TEST_MANIFEST = ROOT / "splits" / "test.csv"


def open_image(path, mode="RGB"):
    with Image.open(path) as image:
        return image.convert(mode)


def fit_preview(image, panel_size):
    longest = max(image.size)
    if longest <= panel_size:
        return image
    size = tuple(max(1, round(side * panel_size / longest)) for side in image.size)
    return image.resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0)


def prediction_preview(model, row, source, infer, panel_size):
    source_path = DATA_ROOT / row["source_file"]
    mask_path = DATA_ROOT / row["mask_file"]
    mask = open_image(mask_path, "L").point(lambda value: 255 if value >= 128 else 0)
    raw = infer(model, source_path, mask_path, row["prompt"], seed=0)
    # Match the target's edit boundary before downsampling the preview.
    return fit_preview(Image.composite(raw, source, mask), panel_size)


def comparison_image(panels, labels):
    font = ImageFont.load_default(size=20)
    label_width = max(int(font.getlength(label)) + 24 for label in labels)
    panel_width = max(label_width, *(panel.width for panel in panels))
    panel_height = max(panel.height for panel in panels)
    gap, header = 8, 40
    width = panel_width * len(panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (width, header + panel_height), "#202124")
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels, strict=True)):
        left = index * (panel_width + gap)
        draw.text((left + panel_width // 2, header // 2), label,
                  font=font, fill="white", anchor="mm")
        canvas.paste(panel, (left + (panel_width - panel.width) // 2, header))
    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Save labeled 4-panel PNGs for the 50 held-out test pairs."
    )
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--panel-size", type=int, default=768,
                        help="Longest side of each preview, in pixels (default: 768)")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    args = parser.parse_args()
    if args.panel_size <= 0:
        parser.error("--panel-size must be positive")
    if args.output.exists():
        parser.error(f"Output directory already exists: {args.output}")
    for checkpoint in (args.baseline, args.candidate):
        if not checkpoint.is_file():
            parser.error(f"Checkpoint not found: {checkpoint}")

    with TEST_MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 50 or len({row["image_id"] for row in rows}) != 50:
        raise ValueError("Test manifest must contain 50 unique source IDs")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required")

    sys.path.insert(0, str(DEMO_ROOT / "code"))
    from opt_runtime import infer, load

    base = DEMO_ROOT / "models" / "sd-turbo"
    print(f"Baseline: {args.baseline.resolve()}", flush=True)
    model = load(base, args.baseline, device_mode="cuda")
    crop = (model.crop_size, model.context)
    baseline_previews = []
    for index, row in enumerate(rows, start=1):
        source = open_image(DATA_ROOT / row["source_file"])
        baseline_previews.append(prediction_preview(model, row, source, infer, args.panel_size))
        if index % 10 == 0:
            print(f"Baseline: {index}/{len(rows)}", flush=True)
    del model
    gc.collect()

    print(f"Candidate: {args.candidate.resolve()}", flush=True)
    model = load(base, args.candidate, device_mode="cuda")
    if crop != (model.crop_size, model.context):
        raise ValueError("Checkpoint crop settings differ")
    args.output.mkdir(parents=True, exist_ok=False)
    labels = ("Original", "Ground truth", args.baseline.name, args.candidate.name)
    for index, (row, baseline) in enumerate(zip(rows, baseline_previews, strict=True), start=1):
        source = open_image(DATA_ROOT / row["source_file"])
        target = open_image(DATA_ROOT / row["target_file"])
        candidate = prediction_preview(model, row, source, infer, args.panel_size)
        panels = (fit_preview(source, args.panel_size), fit_preview(target, args.panel_size),
                  baseline, candidate)
        filename = f"{index:02d}_{row['direction']}_{row['image_id']}.png"
        comparison_image(panels, labels).save(args.output / filename)
        if index % 10 == 0:
            print(f"Saved: {index}/{len(rows)}", flush=True)
    print(f"Saved {len(rows)} PNGs to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

