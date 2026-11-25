"""Archive an existing review packet and publish a validated PENDING replacement.

This command moves directories on one filesystem, never deletes a review packet,
never updates the fixed A lock and never starts training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .b_data import document_fingerprint, load_reviews
from .data import read_jsonl


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _safe_path(value):
    path = Path(value).absolute()
    for part in (path, *path.parents):
        if part.is_symlink() or getattr(part, "is_junction", lambda: False)():
            raise ValueError("Publication paths cannot contain symlinks or junctions")
    return path.resolve()


def _snapshot(directory):
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise ValueError("Review packets cannot contain symlinks or junctions")
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = _sha(path)
        elif not path.is_dir():
            raise ValueError("Review packets must contain only files and directories")
    return files


def _validate_staged(directory):
    manifest = _json(directory / "generation_manifest.json")
    if manifest.get("generator_version") != "synthetic_rule_based_v2":
        raise ValueError("Only a V2 generated packet can replace the current version")
    if manifest.get("review_status") != "PENDING" or manifest.get("training_executed") is not False:
        raise ValueError("New packet must be PENDING with no training performed")
    docs = read_jsonl(directory / "proposed_documents.jsonl")
    if not docs or manifest.get("documents") != len(docs):
        raise ValueError("Generation manifest document count does not match packet")
    expected = {document.doc_id: document_fingerprint(document) for document in docs}
    if len(expected) != len(docs):
        raise ValueError("Duplicate document IDs in new packet")
    review_docs = read_jsonl(directory / "review" / "documents.jsonl")
    if {doc.doc_id: document_fingerprint(doc) for doc in review_docs} != expected or len(review_docs) != len(docs):
        raise ValueError("Review documents differ from generated proposals")
    reviews = load_reviews(directory / "review" / "reviews.jsonl")
    if set(reviews) != set(expected):
        raise ValueError("Review inventory differs from generated documents")
    for doc_id, review in reviews.items():
        if review["review_status"] != "PENDING" or review["document_sha256"] != expected[doc_id]:
            raise ValueError("New review records must be PENDING with matching fingerprints")
        if (review["reviewer_id"] is not None or review["reviewed_at"] is not None
                or review["candidate_inventory_complete"] or review["action_labels_complete"]):
            raise ValueError("New review records cannot carry approval declarations")
    plan = _json(directory / "generation_plan.json")
    records = plan.get("documents", [])
    if len(records) != len(docs) or {row.get("doc_id") for row in records} != set(expected):
        raise ValueError("Generation plan document inventory mismatch")
    audit = _json(directory / "automated_audit.json")
    if (audit.get("passed") is not True or audit.get("blockers") != []
            or audit.get("documents") != len(docs) or audit.get("validated_documents") != len(docs)):
        raise ValueError("New packet must have a passing audit with no blockers")
    declared = manifest.get("files_sha256")
    required = {"proposed_documents.jsonl", "generation_plan.json", "automated_audit.json",
                "review/documents.jsonl", "review/reviews.jsonl", "review/mentions.csv"}
    if not isinstance(declared, dict) or not required.issubset(declared):
        raise ValueError("Generation manifest must bind generated review and audit files")
    for name, digest in declared.items():
        target = (directory / name).resolve()
        if not target.is_relative_to(directory) or not target.is_file() or _sha(target) != digest:
            raise ValueError("Generated file hash mismatch or invalid manifest path: " + name)
    # A stored passing report can become stale after a corpus/plan edit. Check
    # actual current content rather than treating its declared hash as proof.
    from .synthetic_audit import audit_corpus
    fresh_audit = audit_corpus(docs, plan)
    if (fresh_audit.get("passed") is not True or fresh_audit.get("blockers") != []
            or fresh_audit.get("documents") != len(docs)
            or fresh_audit.get("validated_documents") != len(docs)):
        raise ValueError("Current packet failed recomputed semantic audit")
    return manifest


def publish_version(staged_dir, current_dir, archive_dir, legacy_source=None):
    """Replace current only after preserving every existing file byte-for-byte.

    All packet and archive paths must stay under current_dir.parent. The old
    packet may contain human edits; its actual files are snapshotted rather
    than compared against the original generation manifest.
    """
    staged, current, archive = map(_safe_path, (staged_dir, current_dir, archive_dir))
    common = current.parent
    if not staged.is_dir() or not current.is_dir():
        raise ValueError("Staged and current directories must both exist")
    if archive.exists():
        raise ValueError("Archive destination already exists; refusing overwrite")
    paths = (staged, current, archive)
    if any(path == common or not path.is_relative_to(common) for path in paths):
        raise ValueError("All directories must stay inside the resolved current parent")
    for left in paths:
        for right in paths:
            if left != right and (left.is_relative_to(right) or right.is_relative_to(left)):
                raise ValueError("Publication directories cannot be nested")
    if len(set(paths)) != 3:
        raise ValueError("Publication directories must be distinct")
    old_files, new_files = _snapshot(current), _snapshot(staged)
    if any((current / name).exists() for name in ("archive_record.json", "generator_source_v1.py")):
        raise ValueError("Archive record names already exist in old packet")
    manifest = _validate_staged(staged)
    old_lock, new_lock = _json(current / "a_checkpoint.lock.json"), _json(staged / "a_checkpoint.lock.json")
    if old_lock != new_lock:
        raise ValueError("Fixed A checkpoint lock differs between old and staged packets")
    source = _safe_path(legacy_source) if legacy_source is not None else None
    source_digest = None
    source_bytes = None
    if source is not None:
        source_bytes = source.read_bytes()
        source_digest = hashlib.sha256(source_bytes).hexdigest()
        if source_digest != _json(current / "generation_manifest.json").get("generator_code_sha256"):
            raise ValueError("Legacy generator source does not match old generation manifest")
    # Verify all resolved move targets before creating a directory or moving it.
    existing_parent = archive.parent
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    if len({current.stat().st_dev, staged.stat().st_dev, existing_parent.stat().st_dev}) != 1:
        raise ValueError("Publish and archive must remain on the same filesystem")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not _safe_path(archive.parent).is_relative_to(common):
        raise ValueError("Archive parent escaped the intended workspace")
    if archive.exists() or _snapshot(current) != old_files or _snapshot(staged) != new_files:
        raise ValueError("Packet changed during validation; refusing publication")
    current.rename(archive)
    published = False
    created_archive_files = []
    try:
        staged.rename(current)
        published = True
        if _snapshot(current) != new_files:
            raise RuntimeError("Published replacement files changed during publication")
        if _snapshot(archive) != old_files:
            raise RuntimeError("Archived original files failed byte-for-byte verification")
        if source_bytes is not None:
            with (archive / "generator_source_v1.py").open("xb") as outgoing:
                created_archive_files.append(archive / "generator_source_v1.py")
                outgoing.write(source_bytes)
            if _sha(archive / "generator_source_v1.py") != source_digest:
                raise RuntimeError("Archived legacy generator hash mismatch")
        record = {"schema_version": "kpii-synthetic-archive-1",
                  "archived_at_utc": datetime.now(timezone.utc).isoformat(),
                  "previous_path": str(current), "archive_path": str(archive),
                  "original_files_sha256": old_files,
                  "legacy_generator_sha256": source_digest,
                  "replacement_generator_version": manifest["generator_version"],
                  "replacement_manifest_sha256": _sha(current / "generation_manifest.json"),
                  "training_executed": False}
        with (archive / "archive_record.json").open("x", encoding="utf-8") as stream:
            created_archive_files.append(archive / "archive_record.json")
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    except BaseException:
        # Reserved names were absent in the original snapshot and were opened
        # exclusively by this invocation. Remove only our own archive metadata,
        # including a partial write, before restoring the original tree.
        for path in reversed(created_archive_files):
            if path.parent != archive or path.name not in {"archive_record.json", "generator_source_v1.py"}:
                raise RuntimeError("Unexpected rollback metadata path")
            path.unlink(missing_ok=True)
        if published:
            current.rename(staged)
        archive.rename(current)
        raise
    return {"current_dir": str(current), "archive_dir": str(archive),
            "archived_files": len(old_files), "documents": manifest["documents"],
            "review_status": "PENDING", "training_executed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish")
    for name in ("staged-dir", "current-dir", "archive-dir"):
        publish.add_argument("--" + name, required=True)
    publish.add_argument("--legacy-source")
    args = parser.parse_args(argv)
    print(json.dumps(publish_version(args.staged_dir, args.current_dir, args.archive_dir,
                                     args.legacy_source), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
