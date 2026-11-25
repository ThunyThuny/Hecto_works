"""Freeze A and package human-reviewed synthetic B data; never starts training."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .b_data import load_reviews, validate_reviewed_documents, validate_split_separation
from .b_inputs import FIXED_POLICY_ID, allowed_to_keep, keep_eligible
from .data import read_jsonl, write_jsonl
from .stage_a import _prediction_checkpoint_artifacts

FIXED_A_DIR = "/workspace/korean-pii-mvp/runs/a-native-gpu5-20261019_020722/best"
FIXED_A_MANIFEST_SHA256 = "87fd30cf17ee1afe5ea1d995c9f511bc28d11e5935efdc80750c0a30dd56d591"
LOCK_SCHEMA = "kpii-fixed-a-lock-1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_a_lock(model_dir, expected_manifest_sha256=None):
    """Snapshot actual A assets without loading a model or writing its files."""
    directory = Path(model_dir).resolve()
    manifest_path = directory / "training_manifest.json"
    metadata = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if metadata.get("stage") != "A":
        raise ValueError("Only a trained Stage A checkpoint can be locked")
    manifest_digest = file_sha256(manifest_path)
    if expected_manifest_sha256 is not None and manifest_digest != expected_manifest_sha256:
        raise ValueError("A training manifest differs from the explicitly fixed checkpoint")
    artifacts = _prediction_checkpoint_artifacts(directory)
    if "config.json" not in artifacts or not any(
            name.endswith((".safetensors", ".bin")) for name in artifacts):
        raise ValueError("A checkpoint requires config and model weights")
    if not any(Path(name).name in {"tokenizer.json", "vocab.txt", "vocab.json"}
               or name.endswith(".model") for name in artifacts):
        raise ValueError("A checkpoint requires saved tokenizer assets")
    return {"schema_version": LOCK_SCHEMA, "stage": "A", "a_model_dir": str(directory),
            "training_manifest_sha256": manifest_digest, "checkpoint_artifacts": artifacts,
            "purpose": "fixed_A_read_only_inference_and_B_encoder_initialization"}


def verify_a_lock(lock, model_dir):
    """Reject changed, added or removed A assets; a lock is never refreshed here."""
    if not isinstance(lock, dict):
        lock = json.loads(Path(lock).read_text(encoding="utf-8-sig"))
    if lock.get("schema_version") != LOCK_SCHEMA or lock.get("stage") != "A":
        raise ValueError("Unsupported A checkpoint lock")
    current = create_a_lock(model_dir, lock.get("training_manifest_sha256"))
    if current != lock:
        raise ValueError("A path, weights, tokenizer or configuration changed after locking")
    return {"status": "verified", "a_model_dir": current["a_model_dir"],
            "training_manifest_sha256": current["training_manifest_sha256"],
            "artifacts": len(current["checkpoint_artifacts"])}


def _synthetic_source(value):
    return value == "synthetic" or value.startswith(("synthetic_", "synthetic:", "synthetic/"))


def prepare_reviewed(documents_path, reviews_path, output_dir, a_model_dir, a_lock_path):
    """Split approved final content; do not fabricate approval or alter fingerprints."""
    output = Path(output_dir).resolve()
    inputs = [Path(path).resolve() for path in (documents_path, reviews_path, a_lock_path)]
    if output.exists() or any(path.is_relative_to(output) for path in inputs):
        raise ValueError("Use a new output directory outside all source files")
    if output.is_relative_to(Path(a_model_dir).resolve()):
        raise ValueError("Reviewed outputs cannot be inside the fixed A checkpoint")
    verify_a_lock(a_lock_path, a_model_dir)
    documents = read_jsonl(documents_path)
    reviews = load_reviews(reviews_path)
    validate_reviewed_documents(documents, reviews, policy_id=FIXED_POLICY_ID)
    splits = {name: [] for name in ("train", "valid", "test")}
    for document in documents:
        if not _synthetic_source(document.source):
            raise ValueError("Synthetic preparation requires explicitly synthetic source tags")
        split = "valid" if document.split in {"valid", "dev", "validation"} else document.split
        if split not in splits:
            raise ValueError("Every document needs its preassigned train/valid/test split")
        if any(mention.action == "KEEP" and not allowed_to_keep(document, mention)
               for mention in document.mentions):
            raise ValueError("KEEP gold conflicts with the fixed policy or trusted metadata")
        splits[split].append(document)
    validate_split_separation(splits)
    from .b_model import validate_train_dev
    validate_train_dev(splits["train"], splits["valid"])
    for split, records in splits.items():
        if {m.action for d in records for m in d.mentions if keep_eligible(m.entity_type)} != {"MASK", "KEEP"}:
            raise ValueError(f"{split} needs contextual MASK and KEEP examples")
    # All semantic, approval and lock checks happen before creating any output.
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + output.name + ".preparing-", dir=output.parent)).resolve()
    try:
        for split, records in splits.items():
            write_jsonl(staging / (split + ".jsonl"), records)
            declarations = [reviews[document.doc_id] for document in records]
            (staging / (split + ".reviews.jsonl")).write_text("".join(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in declarations), encoding="utf-8")
        shutil.copyfile(a_lock_path, staging / "a_checkpoint.lock.json")
        result = {"schema_version": "kpii-reviewed-synthetic-1", "documents": len(documents),
                  "splits": {name: len(records) for name, records in splits.items()},
                  "source_counts": dict(Counter(d.source for d in documents)),
                  "data_kind": "synthetic_human_reviewed", "training_performed": False,
                  "inputs_sha256": {name: file_sha256(path) for name, path in zip(
                      ("documents", "reviews", "a_lock"), inputs)},
                  "output_dir": str(output)}
        (staging / "dataset_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            raise ValueError("Output appeared during preparation; refusing overwrite")
        staging.rename(output)
        return result
    finally:
        if staging.exists():
            assert staging.parent == output.parent and staging.name.startswith("." + output.name + ".preparing-")
            shutil.rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    lock = commands.add_parser("lock-a")
    lock.add_argument("--a-model-dir", default=FIXED_A_DIR)
    lock.add_argument("--output", required=True)
    lock.add_argument("--expected-manifest-sha256", default=FIXED_A_MANIFEST_SHA256)
    verify = commands.add_parser("verify-a-lock")
    verify.add_argument("--a-model-dir", default=FIXED_A_DIR)
    verify.add_argument("--lock", required=True)
    prepare = commands.add_parser("prepare-reviewed")
    for name in ("documents", "reviews", "output-dir", "a-lock"):
        prepare.add_argument("--" + name, required=True)
    prepare.add_argument("--a-model-dir", default=FIXED_A_DIR)
    args = parser.parse_args(argv)
    if args.command == "lock-a":
        target = Path(args.output)
        if target.exists() or target.resolve().is_relative_to(Path(args.a_model_dir).resolve()):
            raise ValueError("A lock output must be new and outside the checkpoint")
        result = create_a_lock(args.a_model_dir, args.expected_manifest_sha256)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    elif args.command == "verify-a-lock":
        result = verify_a_lock(args.lock, args.a_model_dir)
    else:
        result = prepare_reviewed(args.documents, args.reviews, args.output_dir, args.a_model_dir, args.a_lock)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
