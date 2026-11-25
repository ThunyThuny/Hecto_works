"""Explicit, hash-bound use of unreviewed fictional V3 labels for B experiments.

The default human-review gate is unchanged. This separate opt-in accepts only
an intact PENDING bulk-generator packet, records its provenance, and never
changes review declarations or certifies the labels as human-reviewed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping

from .b_data import (document_fingerprint, load_reviews, pending_review,
                     validate_reviewed_documents, validate_split_separation)
from .b_inputs import FIXED_POLICY_ID, policy_sha256
from .data import read_jsonl
from .schema import Document

EXPERIMENT_KIND = "unreviewed_synthetic_experiment"
GENERATOR_VERSION = "synthetic_rule_based_v3_bulk"
SUPPORTED_GENERATORS = frozenset({GENERATOR_VERSION, "synthetic_rule_based_v4_bulk"})
_SPLITS = {"train": 2000, "valid": 500, "test": 500}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key in synthetic experiment packet")
        result[key] = value
    return result


def _json(path):
    result = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_object)
    if not isinstance(result, dict):
        raise ValueError("Synthetic experiment metadata must be JSON objects")
    return result


def _canonical_split(value):
    return "dev" if value in {"dev", "valid", "validation"} else value


def _fingerprints(documents):
    result = {}
    for document in documents:
        if document.doc_id in result:
            raise ValueError("Duplicate document ID in synthetic experiment data")
        result[document.doc_id] = document_fingerprint(document)
    return result


def _pending(document, record):
    # Notes do not supply approval, but binding the exact original declaration
    # also avoids silently discarding subsequent human annotation work.
    if dict(record) != pending_review(document):
        raise ValueError("Synthetic experiment requires unchanged original PENDING reviews")


def _checked_packet(packet_dir):
    root = Path(packet_dir).absolute()
    for part in (root, *root.parents):
        if part.is_symlink() or getattr(part, "is_junction", lambda: False)():
            raise ValueError("Synthetic experiment packet paths cannot contain symlinks or junctions")
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("Synthetic experiment packet directory does not exist")
    manifest_path = root / "generation_manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Synthetic experiment manifest cannot be a symlink")
    digest = _sha(manifest_path)
    manifest = _json(manifest_path)
    generator = manifest.get("generator_version")
    if (generator not in SUPPORTED_GENERATORS
            or manifest.get("policy_id") != FIXED_POLICY_ID
            or manifest.get("policy_sha256") != policy_sha256()):
        raise ValueError("Experiment requires a fixed-policy V3 or V4 bulk synthetic packet")
    if (manifest.get("review_status") != "PENDING"
            or manifest.get("training_executed") is not False):
        raise ValueError("Experiment packet must retain original PENDING/no-training declarations")
    required = {"proposed_documents.jsonl", "generation_plan.json", "automated_audit.json",
                "review/documents.jsonl", "review/reviews.jsonl", "review/mentions.csv",
                *(f"proposals/{split}.jsonl" for split in _SPLITS)}
    declared = manifest.get("files_sha256")
    if not isinstance(declared, dict) or not required.issubset(declared):
        raise ValueError("Experiment manifest must bind the full proposal, review, split and audit packet")
    for name, expected in declared.items():
        if (not isinstance(name, str) or "\\" in name or ":" in name
                or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                or not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected)):
            raise ValueError("Invalid synthetic experiment manifest path or SHA256")
        target = root / name
        for part in (target, *target.parents):
            if part == root:
                break
            if part.is_symlink() or getattr(part, "is_junction", lambda: False)():
                raise ValueError("Synthetic experiment packet contains a symlink or junction")
        if not target.resolve().is_relative_to(root) or not target.is_file() or _sha(target) != expected:
            raise ValueError("Synthetic experiment file hash mismatch: " + name)
    documents = read_jsonl(root / "proposed_documents.jsonl")
    fingerprints = _fingerprints(documents)
    if (len(documents) != 3000 or manifest.get("documents") != len(documents)
            or manifest.get("split_counts") != _SPLITS):
        raise ValueError("Experiment packet must contain the complete original 3000-document allocation")
    actions = Counter()
    for document in documents:
        if document.source != generator or document.policy_id != FIXED_POLICY_ID:
            raise ValueError("Only the packet's declared synthetic source and fixed policy are allowed")
        for mention in document.mentions:
            if mention.action not in {"MASK", "KEEP"}:
                raise ValueError("Synthetic experiment labels must be explicit MASK/KEEP")
            actions[mention.action] += 1
    if (manifest.get("mentions") != sum(actions.values())
            or manifest.get("proposed_action_counts") != dict(actions)
            or manifest.get("contextual_candidates") != sum(
                m.entity_type != "NAME" for d in documents for m in d.mentions)):
        raise ValueError("Synthetic experiment manifest candidate counts do not match documents")
    splits = {split: [doc for doc in documents if doc.split == split] for split in _SPLITS}
    if {split: len(docs) for split, docs in splits.items()} != _SPLITS:
        raise ValueError("Synthetic experiment split counts do not match original allocation")
    validate_split_separation(splits)
    for split in _SPLITS:
        if _fingerprints(read_jsonl(root / "proposals" / f"{split}.jsonl")) != _fingerprints(splits[split]):
            raise ValueError("Synthetic experiment split proposal file differs from original documents")
    if _fingerprints(read_jsonl(root / "review" / "documents.jsonl")) != fingerprints:
        raise ValueError("Synthetic experiment review documents differ from original proposals")
    reviews = load_reviews(root / "review" / "reviews.jsonl")
    if set(reviews) != set(fingerprints):
        raise ValueError("Synthetic experiment review inventory differs from original proposals")
    for document in documents:
        _pending(document, reviews[document.doc_id])
    stored_audit = _json(root / "automated_audit.json")
    if (stored_audit.get("passed") is not True or stored_audit.get("blockers") != []
            or stored_audit.get("warnings") != []):
        raise ValueError("Synthetic experiment packet must have a passing original audit")
    if generator == GENERATOR_VERSION:
        from .synthetic_bulk_audit import audit_bulk
        fresh = audit_bulk(documents, _json(root / "generation_plan.json"))
    else:
        from .synthetic_v4 import audit_v4
        fresh = audit_v4(documents, _json(root / "generation_plan.json"))
    if fresh.get("passed") is not True or fresh.get("blockers") != [] or fresh.get("warnings") != []:
        raise ValueError("Synthetic experiment packet failed recomputed bulk audit")
    if _sha(manifest_path) != digest:
        raise ValueError("Synthetic experiment manifest changed during validation")
    lock = root / "a_checkpoint.lock.json"
    report = {"data_kind": EXPERIMENT_KIND, "experiment_mode": EXPERIMENT_KIND,
              "review_evidence": EXPERIMENT_KIND, "human_reviewed": False,
              "packet_manifest_sha256": digest, "synthetic_experiment_dir": str(root),
              "sourcepath": str(root), "policy_id": FIXED_POLICY_ID,
              "documents": len(documents), "mentions": sum(actions.values()),
              "contextual_candidates": manifest["contextual_candidates"],
              "split_counts": dict(_SPLITS), "actions": dict(sorted(actions.items())),
              "review_status": "PENDING", "audit_blockers": 0, "audit_warnings": 0,
              "source_provenance": {"classification": "synthetic", "source_counts": {generator: len(documents)},
                  "declared_category_counts": {"synthetic": len(documents)},
                  "basis": "Hash-bound fictional bulk packet; no human annotation approval"},
              "a_lock_sha256": _sha(lock) if lock.is_file() else None,
              "a_lock_verified": False,
              "limitation": "Synthetic rule labels are unreviewed; metrics do not establish real-world performance"}
    return report, documents, reviews


def validate_experiment_packet(packet_dir: str | Path) -> dict[str, Any]:
    """Check packet bytes, current audit and PENDING evidence without ML imports.

    An optional A-lock digest is recorded, but the execution wrapper must verify
    the actual fixed A artifacts separately before and after its work.
    """
    return _checked_packet(packet_dir)[0]


def validate_b_evidence(
    documents: Iterable[Document], reviews: Mapping[str, Mapping[str, Any]], *,
    expected_split: str | None = None, policy_id: str = FIXED_POLICY_ID,
    synthetic_experiment_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Keep the ordinary approval gate, or explicitly bind unreviewed synthetic labels.

    Experiment subsets must match the original packet documents byte-equivalent
    at the canonical document level, retain their original split, and use the
    exact original PENDING declarations. Approval is never fabricated.
    """
    if synthetic_experiment_dir is None:
        return validate_reviewed_documents(documents, reviews, expected_split=expected_split, policy_id=policy_id)
    docs = list(documents)
    if not docs or policy_id != FIXED_POLICY_ID:
        raise ValueError("Synthetic experiment requires nonempty fixed-policy data")
    if expected_split is not None and expected_split not in {"train", "dev", "test"}:
        raise ValueError("expected_split must be train, dev, or test")
    submitted = _fingerprints(docs)
    if set(reviews) != set(submitted):
        raise ValueError("Experiment reviews must match submitted documents exactly")
    for doc in docs:
        if doc.source not in SUPPORTED_GENERATORS or doc.policy_id != FIXED_POLICY_ID:
            raise ValueError("Experiment mode cannot relax real, mixed, unknown or other synthetic sources")
        if expected_split is not None and _canonical_split(doc.split) != expected_split:
            raise ValueError("Experiment document does not match expected split")
        if any(m.action not in {"MASK", "KEEP"} for m in doc.mentions):
            raise ValueError("Experiment action labels must be explicit MASK/KEEP")
        _pending(doc, reviews[doc.doc_id])
    packet, originals, original_reviews = _checked_packet(synthetic_experiment_dir)
    original_fingerprints = _fingerprints(originals)
    for doc_id, fingerprint in submitted.items():
        if original_fingerprints.get(doc_id) != fingerprint or dict(reviews[doc_id]) != original_reviews.get(doc_id):
            raise ValueError("Experiment documents and reviews must match the original hash-bound packet")
    actions = Counter(m.action for d in docs for m in d.mentions)
    types = Counter(m.entity_type for d in docs for m in d.mentions)
    report = dict(packet)
    report.update({"documents": len(docs), "mentions": sum(actions.values()),
                   "contextual_candidates": sum(m.entity_type != "NAME" for d in docs for m in d.mentions),
                   "actions": dict(sorted(actions.items())), "entity_types": dict(sorted(types.items())),
                   "split": expected_split,
                   "source_provenance": {**packet["source_provenance"],
                       "source_counts": dict(Counter(d.source for d in docs)),
                       "declared_category_counts": {"synthetic": len(docs)}}})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("--packet-dir", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(validate_experiment_packet(args.packet_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
