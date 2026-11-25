"""Conservative, label-free ADDRESS/POSITION expansion after the frozen A.

Only original text, A offsets and canonical entity types determine a change.
No gold action, reason, review state or approved public value is consulted.
This deterministic rule layer is distinct from the A checkpoint and writes a
new, hash-bound candidate file. It never shrinks a detected span.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .b_inputs import canonical_type
from .data import read_jsonl, write_jsonl
from .schema import Document, Mention


REFINEMENT_VERSION = "address-position-v1"
# A full road address with an optional province/city and one or two districts.
# Deliberately excludes prose, newlines, phone numbers and unrelated fields.
ADDRESS_PATTERN = (
    r"(?<![가-힣A-Za-z0-9])"
    r"(?:[가-힣]{1,12}(?:특별자치시|특별자치도|특별시|광역시|도|시)[ \t]{1,3})?"
    r"(?:[가-힣]{1,12}(?:시|군|구)[ \t]{1,3}){1,2}"
    r"[가-힣0-9·]{1,20}(?:대로|로|길)[ \t]{1,3}"
    # A phone/account-like longer hyphenated number is not a house number.
    r"[0-9]{1,5}(?:-[0-9]{1,4})?(?![0-9-])"
    # Bounded floor/room tokens, never free-form building/ownership prose.
    r"(?:[ \t]{1,3}[0-9]{1,3}층(?![0-9]))?"
    r"(?:[ \t]{1,3}[0-9]{1,5}호(?![0-9]))?"
)
POSITIONS = (
    "운영책임자", "사업책임자", "총괄책임자", "실무책임자", "책임연구원",
    "선임연구원", "수석연구원", "전임연구원", "연구원", "대표이사", "부대표",
    "대표", "본부장", "센터장", "부장", "차장", "과장", "대리", "주임",
    "사원", "팀장", "실장", "이사", "상무", "전무", "연구교수", "조교수",
    "부교수", "교수", "매니저", "연구개발팀장", "서비스매니저", "프로젝트매니저",
    "기술연구원", "기획실장", "책임매니저", "사업실장", "교육센터장",
    "기술책임자", "기획연구원", "서비스팀장",
)
POSITION_PATTERN = (
    r"(?<![가-힣A-Za-z0-9])(?:" + "|".join(sorted(POSITIONS, key=lambda x: (-len(x), x))) +
    r")(?=$|[\s.,:;!?…·/()\[\]〈〉<>\"'“”‘’]|"
    r"(?:으로|에서|에게|이라|라고|입니다|이었다|였다|은|는|이|가|을|를|의|로|와|과|에))"
)
REFINEMENT_CONFIG = {
    "version": REFINEMENT_VERSION,
    "address_pattern": ADDRESS_PATTERN,
    "position_pattern": POSITION_PATTERN,
    "max_address_characters": 80,
    "max_position_characters": 16,
    "minimum_anchor_characters": 2,
    "cross_type_overlap": "reject",
    "shrink_original_candidates": False,
    "same_line_required": True,
    "address_floor_room": "optional same-line 1..3-digit floor and 1..5-digit room; no free-form suffix",
    "approved_metadata_used": False,
    "gold_labels_used": False,
}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def refine_document(document: Document) -> tuple[Document, dict[str, Any]]:
    """Expand only unambiguous complete forms containing an existing A span.

    A same-type fragmented detection may merge; a different type blocks that
    proposal, including a mandatory-MASK type. The new interval must contain
    *every* intersecting candidate completely. Labels are never copied into
    the candidate output and changing them cannot influence its intervals.
    Diagnostics contain offsets and identifiers, never the original text.
    """
    document.validate()
    original = sorted(document.mentions, key=lambda m: (m.start, m.end, m.mention_id))
    proposals: list[tuple[int, int, str, list[Mention]]] = []
    rejected = Counter()
    for kind, pattern, limit in (("ADDRESS", ADDRESS_PATTERN, 80),
                                  ("POSITION", POSITION_PATTERN, 16)):
        for match in re.finditer(pattern, document.text):
            start, end = match.span()
            if end - start > limit or "\n" in match.group() or "\r" in match.group():
                continue
            intersecting = [m for m in original if _overlap((start, end), (m.start, m.end))]
            if not intersecting or not any(canonical_type(m.entity_type) == kind and
                    m.end - m.start >= 2 for m in intersecting):
                continue
            if any(canonical_type(m.entity_type) != kind for m in intersecting):
                rejected["cross_type_overlap"] += 1
                continue
            if any(m.start < start or m.end > end for m in intersecting):
                rejected["would_shrink_original_span"] += 1
                continue
            if len(intersecting) == 1 and (intersecting[0].start, intersecting[0].end) == (start, end):
                continue
            proposals.append((start, end, kind, intersecting))
    # Reject ambiguous overlapping proposals rather than choosing a wider one.
    ambiguous = {index for index, proposal in enumerate(proposals)
        if any(index != other and _overlap(proposal[:2], candidate[:2])
               for other, candidate in enumerate(proposals))}
    rejected["ambiguous_overlapping_rules"] += len(ambiguous)
    replacements: dict[str, Mention] = {}
    consumed: set[str] = set()
    events = []
    for index, (start, end, kind, anchors) in enumerate(proposals):
        if index in ambiguous:
            continue
        first = anchors[0]
        replacements[first.mention_id] = Mention(first.mention_id, start, end,
            document.text[start:end], first.entity_type)
        consumed.update(m.mention_id for m in anchors[1:])
        events.append({"mention_id": first.mention_id, "canonical_type": kind,
            "rule": "complete_road_address" if kind == "ADDRESS" else "complete_position_lexicon",
            "start": start, "end": end,
            "original_spans": [{"mention_id": m.mention_id, "start": m.start,
                                "end": m.end, "entity_type": m.entity_type} for m in anchors],
            "merged_mention_ids": [m.mention_id for m in anchors[1:]]})
    refined = [replacements.get(m.mention_id, Mention(m.mention_id, m.start, m.end,
        m.text, m.entity_type)) for m in original if m.mention_id not in consumed]
    result = replace(document, mentions=refined)
    report = {"doc_id": document.doc_id, "input_candidates": len(original),
        "output_candidates": len(refined), "changed_candidates": len(events),
        "merged_fragments": len(consumed), "events": events,
        "rejected_proposals": {key: value for key, value in sorted(rejected.items()) if value}}
    return result, report


def _validate_parent(candidate_path: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("stage") != "A-prediction":
        raise ValueError("Refinement input requires an original A-prediction manifest")
    if manifest.get("output", {}).get("sha256") != _sha(candidate_path):
        raise ValueError("Original A manifest does not match the raw candidate file")
    digest = manifest.get("training_manifest_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Original A manifest requires training_manifest_sha256")
    if (type(manifest.get("max_length")) is not int or manifest["max_length"] < 1 or
            type(manifest.get("stride")) is not int or manifest["stride"] < 0):
        raise ValueError("Original A manifest requires explicit window settings")


def refine_candidates_file(input_path: str | Path, output_path: str | Path,
                           report_path: str | Path | None = None) -> dict[str, Any]:
    """Write a new candidate file, diagnostic report and chained manifest.

    This CPU-only command neither loads a model nor changes its checkpoint.
    A raw predict-a sidecar is required so scoring can bind both A and rules.
    """
    source, target = Path(input_path), Path(output_path)
    parent_path = Path(str(source) + ".manifest.json")
    sidecar = Path(str(target) + ".manifest.json")
    report = Path(report_path) if report_path else Path(str(target) + ".refinement.json")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    _validate_parent(source, parent)
    original_source_sha, original_parent_sha = _sha(source), _sha(parent_path)
    outputs = [target.resolve(), sidecar.resolve(), report.resolve()]
    inputs = {source.resolve(), parent_path.resolve()}
    if len(set(outputs)) != len(outputs) or inputs.intersection(outputs):
        raise ValueError("Refinement outputs must be distinct and cannot overwrite inputs")
    if any(path.exists() for path in outputs):
        raise ValueError("Refinement outputs already exist; preserve evidence and use new paths")
    if isinstance(parent.get("model_dir"), str) and any(
            path.is_relative_to(Path(parent["model_dir"]).resolve()) for path in outputs):
        raise ValueError("Refinement outputs cannot be placed inside the frozen A checkpoint")
    documents = read_jsonl(source)
    if not documents:
        raise ValueError("Refinement input cannot be empty")
    refined, reports = zip(*(refine_document(doc) for doc in documents))
    aggregate = {"stage": "A-candidate-refinement-diagnostics", "version": REFINEMENT_VERSION,
        "documents": len(documents), "input_candidates": sum(row["input_candidates"] for row in reports),
        "output_candidates": sum(row["output_candidates"] for row in reports),
        "changed_candidates": sum(row["changed_candidates"] for row in reports),
        "merged_fragments": sum(row["merged_fragments"] for row in reports),
        "documents_changed": sum(bool(row["changed_candidates"]) for row in reports),
        "gold_labels_used": False, "approved_metadata_used": False, "reports": list(reports)}
    if _sha(source) != original_source_sha or _sha(parent_path) != original_parent_sha:
        raise ValueError("Raw A evidence changed during refinement; rerun from an immutable input")
    write_jsonl(target, refined)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"schema_version": "1.0", "stage": "A-candidate-refinement",
        "training_manifest_sha256": parent["training_manifest_sha256"],
        "max_length": parent["max_length"], "stride": parent["stride"],
        "parent": {"candidates_path": str(source.resolve()), "candidates_sha256": original_source_sha,
            "manifest_path": str(parent_path.resolve()), "manifest_sha256": original_parent_sha},
        "output": {"path": str(target.resolve()), "sha256": _sha(target)},
        "diagnostics": {"path": str(report.resolve()), "sha256": _sha(report)},
        "refinement": {"version": REFINEMENT_VERSION, "config_sha256": _digest(REFINEMENT_CONFIG),
                       "config": REFINEMENT_CONFIG},
        "documents": len(documents), "model_execution_performed": False,
        "a_weights_changed": False}
    if "checkpoint_artifacts" in parent:
        manifest["checkpoint_artifacts"] = parent["checkpoint_artifacts"]
    if _sha(source) != original_source_sha or _sha(parent_path) != original_parent_sha:
        raise ValueError("Raw A evidence changed while writing refinement; use new output paths")
    sidecar.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def validate_refinement_manifest(input_path: str | Path) -> dict[str, Any]:
    """Validate the immutable raw A → deterministic refinement chain."""
    source = Path(input_path)
    manifest = json.loads(Path(str(source) + ".manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("stage") != "A-candidate-refinement" or
            manifest.get("output", {}).get("sha256") != _sha(source)):
        raise ValueError("Refined candidate manifest does not match its output")
    config = manifest.get("refinement", {})
    if config != {"version": REFINEMENT_VERSION, "config_sha256": _digest(REFINEMENT_CONFIG),
                  "config": REFINEMENT_CONFIG}:
        raise ValueError("Candidate refinement version/configuration changed; regenerate candidates")
    parent_info = manifest.get("parent", {})
    try:
        raw_path, raw_manifest_path = Path(parent_info["candidates_path"]), Path(parent_info["manifest_path"])
        if raw_manifest_path.resolve() != Path(str(raw_path) + ".manifest.json").resolve():
            raise ValueError("Refinement parent manifest is not adjacent to raw A candidates")
        if (_sha(raw_path) != parent_info["candidates_sha256"] or
                _sha(raw_manifest_path) != parent_info["manifest_sha256"]):
            raise ValueError("Refinement raw A parent evidence changed")
        parent = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
        _validate_parent(raw_path, parent)
        if any(manifest.get(key) != parent.get(key) for key in
               ("training_manifest_sha256", "max_length", "stride", "checkpoint_artifacts")):
            raise ValueError("Refinement changed its original A provenance")
        diagnostic = manifest["diagnostics"]
        if _sha(diagnostic["path"]) != diagnostic["sha256"]:
            raise ValueError("Refinement diagnostic evidence changed")
    except (KeyError, TypeError, OSError) as exc:
        raise ValueError("Refinement requires intact raw A parent and diagnostic evidence") from exc
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    refine = commands.add_parser("refine", help="Refine frozen A candidates using label-free text rules")
    refine.add_argument("--input", required=True)
    refine.add_argument("--output", required=True)
    refine.add_argument("--report")
    args = parser.parse_args(argv)
    result = refine_candidates_file(args.input, args.output, args.report)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
