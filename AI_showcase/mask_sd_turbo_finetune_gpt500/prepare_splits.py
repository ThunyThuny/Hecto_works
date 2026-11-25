import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

DATA_ROOT = Path("/workspace/nfs/datasets/Medical/gpt_image_replacement_500")
SPLIT_NAMES = ("train", "val", "test")
FILE_COLUMNS = ("source_file", "mask_file", "target_file", "metadata_file")


def main():
    parser = argparse.ArgumentParser(description="Split the GPT Image 500-pair dataset.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=Path("splits"))
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()

    manifest = args.data_root / "manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = [*reader.fieldnames, "prompt"]

    if len(rows) != 500:
        raise ValueError(f"Expected 500 rows, found {len(rows)}")
    image_ids = set()
    source_digests = set()
    by_direction = defaultdict(lambda: defaultdict(list))
    for row in rows:
        image_id = row["image_id"]
        if image_id in image_ids:
            raise ValueError(f"Duplicate source image ID: {image_id}")
        image_ids.add(image_id)
        if row["edit_status"] != "completed":
            raise ValueError(f"Incomplete edit: {image_id}")
        for column in FILE_COLUMNS:
            if not row[column] or not (args.data_root / row[column]).is_file():
                raise FileNotFoundError(f"{image_id}: {column} {row[column]}")
        metadata_path = args.data_root / row["metadata_file"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        digest = metadata["source_png_sha256"]
        if digest in source_digests:
            raise ValueError(f"Duplicate source PNG fingerprint: {image_id}")
        source_digests.add(digest)
        if metadata["mask_pixels"] <= 0:
            raise ValueError(f"Empty edit mask: {image_id}")
        row["prompt"] = (
            f"a realistic photo of a {row['target']}, in the same pose and location, "
            "consistent lighting and perspective"
        )
        by_direction[row["direction"]][row["review_status"]].append(row)

    if len(by_direction) != 5:
        raise ValueError(f"Expected five edit directions, found {len(by_direction)}")
    rng = random.Random(args.seed)
    splits = {name: [] for name in SPLIT_NAMES}
    for direction, group in sorted(by_direction.items()):
        if set(group) != {"pass", "boundary_flagged"} or sum(map(len, group.values())) != 100:
            raise ValueError(f"Unexpected distribution for {direction}")
        pass_per_holdout = (len(group["pass"]) + 5) // 10
        for status, holdout_size in (
            ("pass", pass_per_holdout), ("boundary_flagged", 10 - pass_per_holdout)
        ):
            candidates = sorted(group[status], key=lambda row: row["image_id"])
            rng.shuffle(candidates)
            splits["val"].extend(candidates[:holdout_size])
            splits["test"].extend(candidates[holdout_size:2 * holdout_size])
            splits["train"].extend(candidates[2 * holdout_size:])

    split_ids = [{row["image_id"] for row in splits[name]} for name in SPLIT_NAMES]
    if [len(ids) for ids in split_ids] != [400, 50, 50] or set.union(*split_ids) != image_ids:
        raise ValueError("Splits must contain all 500 source IDs exactly once")

    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        "data_root": str(args.data_root.resolve()),
        "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "seed": args.seed,
        "splits": {},
    }
    for name, subset in splits.items():
        rng.shuffle(subset)
        split_path = args.output / f"{name}.csv"
        with split_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(subset)
        summary["splits"][name] = {
            "count": len(subset),
            "sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "directions": dict(Counter(row["direction"] for row in subset)),
            "review_status": dict(Counter(row["review_status"] for row in subset)),
        }
    report = json.dumps(summary, indent=2)
    (args.output / "split.json").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
