"""Label-free A inputs and reviewed or explicit synthetic-experiment preflight."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from .b_inputs import FIXED_POLICY_ID, allowed_to_keep, keep_eligible
from .data import read_jsonl, write_jsonl


def a_inventory(model_dir):
    path = Path(model_dir) / "training_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    labels = manifest.get("label2id")
    if manifest.get("stage") != "A" or not isinstance(labels, dict) or labels.get("O") != 0:
        raise ValueError("A checkpoint requires a valid BIO training manifest")
    types = sorted(name[2:] for name in labels if name.startswith("B-"))
    if not types or not all(types) or set(labels) != {"O"} | {prefix + kind for kind in types for prefix in ("B-", "I-")}:
        raise ValueError("A manifest has an incomplete BIO label inventory")
    if any(type(v) is not int for v in labels.values()) or sorted(labels.values()) != list(range(len(labels))):
        raise ValueError("A label IDs must be unique contiguous integers")
    return types


def prepare_a_documents(documents, inventory):
    """Remove gold candidates/labels; widen only this scan copy to A's types.

    annotation_scope here describes the supported prediction inventory. The
    copy is never suitable as a fully annotated training/evaluation gold file.
    Offsets, source text, metadata, policy and split remain unchanged.
    """
    result, seen = [], set()
    for document in documents:
        document.validate()
        if document.doc_id in seen:
            raise ValueError("Duplicate document ID")
        seen.add(document.doc_id)
        if document.policy_id != FIXED_POLICY_ID:
            raise ValueError("Explicit career_external_share_v1 policy is required")
        result.append(replace(document, mentions=[], annotation_scope=list(inventory)))
    if not result:
        raise ValueError("Input documents cannot be empty")
    return result


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    prepare = commands.add_parser("prepare-a-input")
    for name in ("input", "model-dir", "output"):
        prepare.add_argument("--" + name, required=True)
    preflight = commands.add_parser("preflight", help="Validate all reviewed splits before fitting B")
    for split in ("train", "dev", "test"):
        preflight.add_argument("--" + split, required=True)
        preflight.add_argument("--" + split + "-reviews", required=True)
    preflight.add_argument("--synthetic-experiment-dir",
                           help="Explicitly use the immutable PENDING synthetic packet for a research experiment")
    args = parser.parse_args(argv)
    if args.operation == "prepare-a-input":
        source, target, model_dir = Path(args.input), Path(args.output), Path(args.model_dir)
        sidecar = Path(str(target) + ".preparation.json")
        for path in (target, sidecar):
            if path.resolve() == source.resolve() or path.resolve().is_relative_to(model_dir.resolve()) or path.exists():
                raise ValueError("Preparation outputs must be new and outside inputs/checkpoint")
        inventory = a_inventory(model_dir)
        documents = prepare_a_documents(read_jsonl(source), inventory)
        write_jsonl(target, documents)
        result = {"stage": "B_A_SCAN_INPUT_V1", "gold_labels_removed": True,
                  "prediction_inventory_only": True, "documents": len(documents),
                  "input_sha256": _sha(source), "output_sha256": _sha(target),
                  "a_training_manifest_sha256": _sha(model_dir / "training_manifest.json"),
                  "annotation_scope": inventory}
        sidecar.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        from .b_data import load_reviews, validate_split_separation
        from .b_experiment import validate_b_evidence
        from .b_model import validate_train_dev
        splits, reports = {}, {}
        for split in ("train", "dev", "test"):
            documents = read_jsonl(getattr(args, split))
            reports[split] = validate_b_evidence(documents,
                load_reviews(getattr(args, split + "_reviews")), expected_split=split,
                synthetic_experiment_dir=args.synthetic_experiment_dir)
            for document in documents:
                if any(m.action == "KEEP" and not allowed_to_keep(document, m) for m in document.mentions):
                    raise ValueError("KEEP gold conflicts with fixed policy/trusted metadata")
            splits[split] = documents
        validate_split_separation(splits)
        validate_train_dev(splits["train"], splits["dev"])
        for split in ("dev", "test"):
            if {m.action for d in splits[split] for m in d.mentions if keep_eligible(m.entity_type)} != {"MASK", "KEEP"}:
                raise ValueError(f"{split} requires contextual MASK and KEEP gold for meaningful B evaluation")
        result = {"status": "ready_for_reviewed_data_training", "splits": reports, "training_performed": False}
        if args.synthetic_experiment_dir:
            digests = {report["packet_manifest_sha256"] for report in reports.values()}
            if len(digests) != 1:
                raise ValueError("All experiment splits must belong to the same immutable synthetic packet")
            result.update(status="ready_for_unreviewed_synthetic_experiment", human_reviewed=False,
                          experiment_mode="unreviewed_synthetic_experiment",
                          packet_manifest_sha256=digests.pop())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
