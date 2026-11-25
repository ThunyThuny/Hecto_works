"""Actual-data B annotation exchange and explicit review-declaration validation.

Exported A candidates are proposals, not a complete gold inventory. No command
in this module supplies labels, approves a review, or verifies who reviewed it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from .data import read_jsonl, write_jsonl
from .schema import Document, Mention

DEFAULT_POLICY_ID = "career_external_share_v1"
CSV_FIELDS = ("doc_id", "mention_id", "start", "end", "text", "entity_type", "action", "reason_code")
REVIEW_FIELDS = {
    "doc_id", "document_sha256", "review_status", "reviewer_id", "reviewed_at",
    "candidate_inventory_complete", "action_labels_complete", "notes",
}
REVIEW_REQUIRED = REVIEW_FIELDS - {"notes"}
DEV_SPLITS = {"dev", "valid", "validation"}


def _canonical_split(split: str | None) -> str | None:
    return "dev" if split in DEV_SPLITS else split


def document_fingerprint(document: Document) -> str:
    """Hash the full canonical content, including actions and trusted metadata."""
    document.validate()
    serialized = json.dumps(document.to_dict(), ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _normalized_text_hash(text: str) -> str:
    # This normalization is ONLY for duplicate detection; it never changes
    # source text or annotation offsets. NFKC catches fullwidth-copy variants.
    normalized = " ".join(unicodedata.normalize("NFKC", text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _json_string_hash(value: str) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def document_identities(documents: Iterable[Document]) -> dict[str, list[str]]:
    """Privacy-preserving identity bindings shared by train/calibration/test.

    Text/group/id exact hashes include canonical JSON string quoting. The
    normalized-text hash uses raw UTF-8 after NFKC and whitespace collapse.
    Full-document hashes also bind actions, metadata, offsets and split. These
    are duplicate-detection bindings, not irreversible anonymization of short
    or guessable source strings. Nothing here outputs raw source values.
    """
    docs = _unique_documents(documents)
    return {
        "document_sha256": sorted({document_fingerprint(doc) for doc in docs}),
        "text_sha256": sorted({_json_string_hash(doc.text) for doc in docs}),
        "normalized_text_sha256": sorted({_normalized_text_hash(doc.text) for doc in docs}),
        "source_group_sha256": sorted({_json_string_hash(doc.source_group) for doc in docs}),
        "doc_id_sha256": sorted({_json_string_hash(doc.doc_id) for doc in docs}),
    }


def _unique_documents(documents: Iterable[Document]) -> list[Document]:
    docs = list(documents)
    seen: set[str] = set()
    for doc in docs:
        if not isinstance(doc, Document):
            raise ValueError("expected canonical Document instances")
        doc.validate()
        if doc.doc_id in seen:
            raise ValueError(f"duplicate doc_id: {doc.doc_id}")
        seen.add(doc.doc_id)
    return docs


def pending_review(document: Document) -> dict[str, Any]:
    return {
        "doc_id": document.doc_id,
        "document_sha256": document_fingerprint(document),
        "review_status": "PENDING",
        "reviewer_id": None,
        "reviewed_at": None,
        "candidate_inventory_complete": False,
        "action_labels_complete": False,
        "notes": "",
    }


def _validate_review(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict) or not all(isinstance(key, str) for key in record):
        raise ValueError("review must be a JSON object")
    if REVIEW_REQUIRED - record.keys() or record.keys() - REVIEW_FIELDS:
        raise ValueError("review has missing or unsupported fields")
    if not isinstance(record["doc_id"], str) or not record["doc_id"] or record["doc_id"] != record["doc_id"].strip():
        raise ValueError("review doc_id must be a nonempty string without outer whitespace")
    if not isinstance(record["document_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", record["document_sha256"]):
        raise ValueError("document_sha256 must be a lowercase SHA256 digest")
    if record["review_status"] not in ("PENDING", "APPROVED"):
        raise ValueError("review_status must be PENDING or APPROVED")
    for name in ("candidate_inventory_complete", "action_labels_complete"):
        if type(record[name]) is not bool:
            raise ValueError(f"{name} must be a boolean")
    for name in ("reviewer_id", "reviewed_at"):
        if record[name] is not None and not isinstance(record[name], str):
            raise ValueError(f"{name} must be a string or null")
    if "notes" in record and not isinstance(record["notes"], str):
        raise ValueError("review notes must be a string")
    if record["review_status"] == "APPROVED":
        reviewer = record["reviewer_id"]
        if not reviewer or reviewer != reviewer.strip():
            raise ValueError("APPROVED review requires a nonempty reviewer_id")
        timestamp = record["reviewed_at"]
        if not timestamp or timestamp != timestamp.strip():
            raise ValueError("APPROVED review requires an ISO8601 reviewed_at with timezone")
        try:
            parsed = datetime.fromisoformat(timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp)
        except ValueError as exc:
            raise ValueError("reviewed_at must be an ISO8601 timestamp with timezone") from exc
        if "T" not in timestamp or parsed.utcoffset() is None:
            raise ValueError("reviewed_at must include time and timezone")
        if not record["candidate_inventory_complete"] or not record["action_labels_complete"]:
            raise ValueError("APPROVED requires complete candidate inventory and action labels")
    return dict(record)


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key in review record")
        result[key] = value
    return result


def load_reviews(path: str | Path) -> dict[str, dict[str, Any]]:
    reviews = {}
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = _validate_review(json.loads(line, object_pairs_hook=_no_duplicate_keys))
                if record["doc_id"] in reviews:
                    raise ValueError("duplicate review doc_id")
                reviews[record["doc_id"]] = record
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid review on JSONL line {line_number}: {exc}") from exc
    return reviews


def validate_reviewed_documents(
    documents: Iterable[Document], reviews: Mapping[str, Mapping[str, Any]], *,
    expected_split: str | None = None, policy_id: str = DEFAULT_POLICY_ID,
    require_complete_inventory: bool = True,
) -> dict[str, Any]:
    """Require matching explicit approval declarations; does not certify provenance.

    Native KDPII action=null is not a B label and fails this gate. The optional
    completeness switch never relaxes APPROVED record validity.
    """
    docs = _unique_documents(documents)
    if not docs:
        raise ValueError("B reviewed data cannot be empty")
    if not isinstance(policy_id, str) or not policy_id or policy_id != policy_id.strip():
        raise ValueError("policy_id must be explicit and nonempty")
    if expected_split is not None and expected_split not in ("train", "dev", "test"):
        raise ValueError("expected_split must be train, dev, or test")
    if set(reviews) != {doc.doc_id for doc in docs}:
        raise ValueError("reviews must match documents exactly; missing or extra doc_id")
    actions: Counter[str] = Counter()
    types: Counter[str] = Counter()
    for doc in docs:
        review = _validate_review(reviews[doc.doc_id])
        if review["doc_id"] != doc.doc_id:
            raise ValueError("review key and doc_id disagree")
        if doc.policy_id != policy_id:
            raise ValueError(f"document {doc.doc_id} has a missing or mismatched policy_id")
        if expected_split is not None and _canonical_split(doc.split) != expected_split:
            raise ValueError(f"document {doc.doc_id} does not match expected split {expected_split}")
        if review["review_status"] != "APPROVED":
            raise ValueError(f"document {doc.doc_id} review is PENDING; explicit approval is required")
        if review["document_sha256"] != document_fingerprint(doc):
            raise ValueError(f"document {doc.doc_id} has a stale review fingerprint; re-review edited content")
        if require_complete_inventory and not review["candidate_inventory_complete"]:
            raise ValueError(f"document {doc.doc_id} candidate inventory is incomplete")
        if not review["action_labels_complete"]:
            raise ValueError(f"document {doc.doc_id} action labels are incomplete")
        for mention in doc.mentions:
            if mention.action not in ("MASK", "KEEP"):
                raise ValueError(f"document {doc.doc_id} has unlabelled action; null never means KEEP")
            actions[mention.action] += 1
            types[mention.entity_type] += 1
    return {
        "documents": len(docs), "mentions": sum(actions.values()),
        "actions": dict(sorted(actions.items())), "entity_types": dict(sorted(types.items())),
        "policy_id": policy_id, "split": expected_split,
        "review_evidence": "explicit_reviewer_declarations_not_independently_verified",
    }


def validate_split_separation(splits: Mapping[str, Iterable[Document]]) -> dict[str, int]:
    """Reject id/text duplicates and group leakage before B train/dev/test use."""
    if not splits or any(_canonical_split(key) not in ("train", "dev", "test") for key in splits):
        raise ValueError("split keys must be train, dev, or test")
    if len({_canonical_split(key) for key in splits}) != len(splits):
        raise ValueError("provide only one dev/valid/validation split")
    owners: dict[str, dict[str, str]] = {"id": {}, "text": {}, "group": {}}
    counts = {}
    for supplied_split, records in splits.items():
        split = _canonical_split(supplied_split)
        docs = _unique_documents(records)
        if not docs:
            raise ValueError(f"split {split} cannot be empty")
        counts[split] = len(docs)
        for doc in docs:
            if _canonical_split(doc.split) != split:
                raise ValueError(f"document split does not match supplied split {split}")
            text_hash = _normalized_text_hash(doc.text)
            for key, value in (("id", doc.doc_id), ("text", text_hash), ("group", doc.source_group)):
                previous = owners[key].get(value)
                if previous is not None and (key != "group" or previous != split):
                    raise ValueError(f"duplicate {key} or source-group leakage in B splits ({previous}, {split})")
                owners[key][value] = split
    return counts


def _write_records(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records]
    target.write_text("".join(lines), encoding="utf-8")


def _check_new_outputs(inputs: Iterable[str | Path], outputs: Iterable[str | Path]) -> None:
    sources = {Path(value).resolve() for value in inputs}
    targets = [Path(value).resolve() for value in outputs]
    if len(targets) != len(set(targets)) or sources & set(targets):
        raise ValueError("outputs must be distinct and must not overwrite inputs")
    if any(path.exists() for path in targets):
        raise ValueError("review output already exists; use a new path to preserve prior review evidence")


def export_review(documents: Iterable[Document], output_dir: str | Path, *,
                  policy_id: str | None = None) -> dict[str, Any]:
    docs = _unique_documents(documents)
    if not docs:
        raise ValueError("review export requires at least one actual document")
    if policy_id is not None:
        docs = [replace(doc, policy_id=policy_id) for doc in docs]
    destination = Path(output_dir)
    targets = [destination / name for name in ("documents.jsonl", "mentions.csv", "reviews.jsonl")]
    _check_new_outputs([], targets)
    # All validation/fingerprinting precedes file writes.
    pending = [pending_review(doc) for doc in docs]
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(targets[0], docs)
    with targets[1].open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for doc in docs:
            if not doc.mentions:
                # Every document must have a CSV row so missing rows cannot
                # silently turn a lost inventory into a negative document.
                writer.writerow({"doc_id": doc.doc_id})
            for mention in doc.mentions:
                writer.writerow({"doc_id": doc.doc_id, **mention.to_dict()})
    _write_records(targets[2], pending)
    return {"documents": len(docs), "mentions": sum(len(doc.mentions) for doc in docs),
            "review_status": "PENDING", "output_dir": str(destination)}


def import_review(documents: Iterable[Document], mentions_csv: str | Path) -> list[Document]:
    """Replace candidate inventory with the corrected CSV, preserving raw text.

    Deleting a row deletes that candidate; it does not label it KEEP. Every
    document requires at least one row. An explicitly empty candidate inventory
    is represented by a row containing only doc_id, all other fields blank.
    """
    docs = _unique_documents(documents)
    by_id = {doc.doc_id: doc for doc in docs}
    inventories: dict[str, list[Mention]] = {doc.doc_id: [] for doc in docs}
    seen_documents: set[str] = set()
    empty_inventories: set[str] = set()
    with Path(mentions_csv).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames is None or len(reader.fieldnames) != len(CSV_FIELDS) or set(reader.fieldnames) != set(CSV_FIELDS):
            raise ValueError("mentions CSV must contain each supported column exactly once")
        try:
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("CSV row contains missing or extra columns")
                if row["doc_id"] not in by_id:
                    raise ValueError("CSV contains unknown doc_id")
                doc_id = row["doc_id"]
                if all(row[field] == "" for field in CSV_FIELDS if field != "doc_id"):
                    if doc_id in seen_documents:
                        raise ValueError("empty-inventory row cannot coexist with any other document row")
                    seen_documents.add(doc_id)
                    empty_inventories.add(doc_id)
                    continue
                if doc_id in empty_inventories:
                    raise ValueError("empty-inventory row cannot coexist with any other document row")
                seen_documents.add(doc_id)
                for field in ("start", "end"):
                    if not re.fullmatch(r"0|[1-9][0-9]*", row[field]):
                        raise ValueError("CSV span offsets must be nonnegative decimal integers")
                mention = Mention(
                    row["mention_id"], int(row["start"]), int(row["end"]), row["text"],
                    row["entity_type"], row["action"] or None, row["reason_code"] or None,
                )
                mention.validate(by_id[row["doc_id"]].text)
                inventories[row["doc_id"]].append(mention)
        except (csv.Error, TypeError, ValueError) as exc:
            raise ValueError(f"invalid mentions CSV near physical line {reader.line_num}: {exc}") from exc
    if seen_documents != set(by_id):
        raise ValueError("CSV lost all rows for a document; an empty inventory requires an explicit doc_id-only row")
    return [replace(doc, mentions=sorted(inventories[doc.doc_id], key=lambda m: (m.start, m.end, m.mention_id))) for doc in docs]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export-review", help="Export proposal candidates and PENDING review forms")
    export.add_argument("--input", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--policy-id", help="Explicitly set policy_id on the exported review copy")
    ingest = sub.add_parser("import-review", help="Import corrected CSV without inventing labels or approval")
    ingest.add_argument("--documents", required=True)
    ingest.add_argument("--mentions", required=True)
    ingest.add_argument("--output", required=True)
    group = ingest.add_mutually_exclusive_group()
    group.add_argument("--prepare-reviews", metavar="PATH", help="Write fresh PENDING fingerprints for corrected docs")
    group.add_argument("--reviews", help="Require approved declarations matching corrected content")
    ingest.add_argument("--policy-id", default=DEFAULT_POLICY_ID)
    ingest.add_argument("--split", choices=("train", "dev", "test"))
    validate = sub.add_parser("validate", help="Validate explicit approvals for fully labelled real data")
    validate.add_argument("--input", required=True)
    validate.add_argument("--reviews", required=True)
    validate.add_argument("--policy-id", default=DEFAULT_POLICY_ID)
    validate.add_argument("--split", choices=("train", "dev", "test"))
    args = parser.parse_args(argv)
    if args.command == "export-review":
        outputs = [Path(args.output_dir) / name for name in ("documents.jsonl", "mentions.csv", "reviews.jsonl")]
        _check_new_outputs([args.input], outputs)
        result = export_review(read_jsonl(args.input), args.output_dir, policy_id=args.policy_id)
    elif args.command == "import-review":
        inputs = [args.documents, args.mentions] + ([args.reviews] if args.reviews else [])
        outputs = [args.output] + ([args.prepare_reviews] if args.prepare_reviews else [])
        _check_new_outputs(inputs, outputs)
        docs = import_review(read_jsonl(args.documents), args.mentions)
        result = {"documents": len(docs), "mentions": sum(len(doc.mentions) for doc in docs), "review_status": "UNVERIFIED"}
        if args.reviews:
            result = validate_reviewed_documents(docs, load_reviews(args.reviews), expected_split=args.split, policy_id=args.policy_id)
        pending = [pending_review(doc) for doc in docs] if args.prepare_reviews else []
        write_jsonl(args.output, docs)
        if args.prepare_reviews:
            _write_records(args.prepare_reviews, pending)
            result["review_status"] = "PENDING"
    else:
        result = validate_reviewed_documents(read_jsonl(args.input), load_reviews(args.reviews), expected_split=args.split, policy_id=args.policy_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
