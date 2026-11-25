"""Select small native train/dev splits that retain aligned supervision for every train type.

This is for execution checks, not model performance measurement. No test file is opened.
"""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/processed/kdpii-v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name-or-path", default="skt/A.X-Encoder-base")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--dev-documents", type=int, default=8)
    args = parser.parse_args()
    if args.dev_documents <= 0:
        parser.error("--dev-documents must be positive")
    from transformers import AutoTokenizer
    from kpii_mvp.data import read_jsonl, write_jsonl
    from kpii_mvp.stage_a import build_label_inventory, prepare_features
    inputs, output = Path(args.input_dir), Path(args.output_dir)
    if any((output / name).exists() for name in ("train.jsonl", "valid.jsonl", "manifest.json")):
        raise ValueError("Smoke output already exists; choose a new directory")
    train, dev = read_jsonl(inputs / "train.jsonl"), read_jsonl(inputs / "valid.jsonl")
    inventory = build_label_inventory(train)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    remaining = {label[2:] for label in inventory if label.startswith("B-")}
    chosen = []
    for document in train:
        if not remaining:
            break
        if not remaining.intersection(mention.entity_type for mention in document.mentions):
            continue
        _, report = prepare_features([document], tokenizer, inventory, args.max_length, args.stride,
                                     require_all_type_supervision=False)
        covered = {kind for kind, count in report["positive_tokens_by_type"].items() if count}
        if covered & remaining:
            chosen.append(document)
            remaining -= covered
    if remaining:
        raise ValueError(f"Cannot retain positive supervision for {sorted(remaining)}")
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train.jsonl", chosen)
    write_jsonl(output / "valid.jsonl", dev[:args.dev_documents])
    manifest = {
        "purpose": "execution-smoke-only", "model": args.model_name_or_path,
        "train_documents": len(chosen), "valid_documents": min(len(dev), args.dev_documents),
        "source_sha256": {name: hashlib.sha256((inputs / name).read_bytes()).hexdigest()
                          for name in ("train.jsonl", "valid.jsonl")},
        "max_length": args.max_length, "stride": args.stride,
        "native_types": len(inventory) // 2, "test_accessed": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
