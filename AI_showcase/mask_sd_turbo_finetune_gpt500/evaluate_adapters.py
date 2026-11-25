"""Compare two adapters on the held-out GPT Image test split."""

import argparse
import csv
import gc
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
from torch.utils.data import DataLoader

from train_adapter import DATA_ROOT, DEMO_ROOT, ROOT, PairedDataset, masked_l1

DEFAULT_BASELINE = DEMO_ROOT / "weights" / "adapter_A.pt"
DEFAULT_CANDIDATE = ROOT / "runs" / "gpt500_from_A_20261120_015515" / "adapter_best.pt"


@torch.no_grad()
def score_pairs(model, loader):
    model.eval()
    scores = []
    for batch in loader:
        scores.extend(masked_l1(model, batch, deterministic=True).tolist())
    if len(scores) != len(loader.dataset) or not all(map(math.isfinite, scores)):
        raise RuntimeError("Evaluation did not produce one finite L1 score per pair")
    return scores


def print_comparison(rows, baseline, candidate):
    directions = defaultdict(list)
    for index, row in enumerate(rows):
        directions[row["direction"]].append(index)
    groups = [("overall", range(len(rows))), *sorted(directions.items())]

    print(f"{'direction':<26} {'pairs':>5} {'baseline L1':>12} {'candidate L1':>12} {'L1 reduction':>14}")
    for direction, indices in groups:
        baseline_mean = mean(baseline[index] for index in indices)
        candidate_mean = mean(candidate[index] for index in indices)
        reduction = (
            f"{(baseline_mean - candidate_mean) / baseline_mean:+.2%}"
            if baseline_mean else "n/a"
        )
        print(
            f"{direction:<26} {len(indices):>5} {baseline_mean:>12.6f} "
            f"{candidate_mean:>12.6f} {reduction:>14}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compare adapters on 50 held-out pairs using masked L1 (lower is better)."
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE,
                        help="Baseline checkpoint (default: adapter_A.pt)")
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE,
                        help="Candidate checkpoint (default: fine-tuned adapter_best.pt)")
    parser.add_argument("--workers", type=int, default=2, help="Data loading workers (default: 2)")
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    for path in (args.baseline, args.candidate):
        if not path.is_file():
            parser.error(f"Checkpoint not found: {path}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required")

    sys.path.insert(0, str(DEMO_ROOT / "code"))
    from opt_runtime import load, prepare

    base = DEMO_ROOT / "models" / "sd-turbo"
    splits = ROOT / "splits"
    print(f"Baseline: {args.baseline.resolve()}", flush=True)
    model = load(base, args.baseline, device_mode="cuda")
    crop = (model.crop_size, model.context)
    dataset = PairedDataset(splits / "test.csv", 50, DATA_ROOT, *crop, prepare)
    for name in ("train", "val"):
        with (splits / f"{name}.csv").open(encoding="utf-8-sig", newline="") as handle:
            other_ids = {row["image_id"] for row in csv.DictReader(handle)}
        if dataset.image_ids & other_ids:
            raise ValueError(f"Test source IDs overlap with {name}.csv")
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True
    )

    baseline = score_pairs(model, loader)
    del model
    gc.collect()
    print(f"Candidate: {args.candidate.resolve()}", flush=True)
    model = load(base, args.candidate, device_mode="cuda")
    if crop != (model.crop_size, model.context):
        raise ValueError("Checkpoint crop settings differ")
    candidate = score_pairs(model, loader)

    print_comparison(dataset.rows, baseline, candidate)


if __name__ == "__main__":
    main()
