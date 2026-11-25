import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
DEMO_ROOT = Path("/workspace/mask_sd_turbo_demo_20261008/mask_sd_turbo_demo_20261008")
DATA_ROOT = Path("/workspace/nfs/datasets/Medical/gpt_image_replacement_500")


class PairedDataset:
    def __init__(self, manifest, count, data_root, crop_size, context, prepare):
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if len(self.rows) != count or len(self.image_ids) != count:
            raise ValueError(f"{manifest} must contain {count} unique source IDs")
        self.data_root = data_root
        self.crop_size = crop_size
        self.context = context
        self.prepare = prepare

    @property
    def image_ids(self):
        return {row["image_id"] for row in self.rows}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.data_root / row["source_file"]) as image:
            source = image.convert("RGB")
        with Image.open(self.data_root / row["mask_file"]) as image:
            mask = image.convert("L")
        with Image.open(self.data_root / row["target_file"]) as image:
            target = image.convert("RGB")
        if source.size != mask.size or source.size != target.size:
            raise ValueError(f"Mismatched image and mask sizes: {row['image_id']}")
        (source, mask, target), _ = self.prepare(
            source, mask, target, size=self.crop_size, context=self.context
        )
        mask_tensor = torch.from_numpy(np.asarray(mask) >= 128).unsqueeze(0).float()
        if not mask_tensor.any():
            raise ValueError(f"Empty resized edit mask: {row['image_id']}")
        return rgb_tensor(source), mask_tensor, rgb_tensor(target), row["prompt"]


def rgb_tensor(image):
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1


def masked_l1(model, batch, deterministic):
    source, mask, target, prompts = batch
    source = source.cuda(non_blocking=True)
    mask = mask.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model(source, mask, list(prompts), deterministic=deterministic, return_raw=True)
    error = (prediction.float() - target).abs() * mask
    return error.sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) * target.shape[1])


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total = sum(masked_l1(model, batch, deterministic=True).sum().item() for batch in loader)
    score = total / len(loader.dataset)
    if not math.isfinite(score):
        raise RuntimeError("Non-finite validation loss")
    return score


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the mask SD-Turbo v5 A adapter.")
    parser.add_argument("--output", required=True, type=Path, help="New checkpoint directory")
    parser.add_argument("--demo-dir", type=Path, default=DEMO_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "splits")
    parser.add_argument("--checkpoint", type=Path, help="Default: DEMO_DIR/weights/adapter_A.pt")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--val-every", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    if min(args.steps, args.val_every, args.batch_size) < 1 or args.workers < 0:
        parser.error("steps, val-every and batch-size must be positive, workers nonnegative")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr must be finite and positive")
    if args.output.exists():
        parser.error(f"output directory already exists: {args.output}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required")

    sys.path.insert(0, str(args.demo_dir / "code"))
    from opt_runtime import load, prepare

    torch.manual_seed(args.seed)
    checkpoint = args.checkpoint or args.demo_dir / "weights" / "adapter_A.pt"
    model = load(args.demo_dir / "models" / "sd-turbo", checkpoint, device_mode="cuda")
    train_csv = args.split_dir / "train.csv"
    val_csv = args.split_dir / "val.csv"
    train = PairedDataset(train_csv, 400, args.data_root, model.crop_size, model.context, prepare)
    val = PairedDataset(val_csv, 50, args.data_root, model.crop_size, model.context, prepare)
    if train.image_ids & val.image_ids:
        raise ValueError("Train and validation have overlapping source IDs")
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train, shuffle=True, generator=torch.Generator().manual_seed(args.seed), **loader_options
    )
    val_loader = DataLoader(val, **loader_options)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("The adapter has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "initial_checkpoint": str(checkpoint.resolve()),
        "initial_checkpoint_sha256": sha256(checkpoint),
        "initial_steps": model.checkpoint_steps,
        "train_manifest_sha256": sha256(train_csv),
        "val_manifest_sha256": sha256(val_csv),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "loss": "mask_normalized_l1",
    }
    baseline_score = evaluate(model, val_loader)
    print(json.dumps({"step": 0, "val_mask_l1": baseline_score}), flush=True)
    model.train()
    step = 0
    best_score = math.inf
    while step < args.steps:
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = masked_l1(model, batch, deterministic=False).mean()
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                raise RuntimeError(f"Non-finite training loss at step {step + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1

            if step == 1 or step % 25 == 0:
                print(json.dumps({"step": step, "train_mask_l1": loss_value}), flush=True)
            if step % args.val_every == 0 or step == args.steps:
                score = evaluate(model, val_loader)
                print(json.dumps({"step": step, "val_mask_l1": score}), flush=True)
                info = {
                    **provenance,
                    "fine_tune_steps": step,
                    "val_mask_l1": score,
                    "initial_val_mask_l1": baseline_score,
                }
                total_steps = model.checkpoint_steps + step
                model.save(args.output / "adapter_last.pt", total_steps, info)
                if score < best_score:
                    model.save(args.output / "adapter_best.pt", total_steps, info)
                    best_score = score
                model.train()
            if step == args.steps:
                break


if __name__ == "__main__":
    main()

