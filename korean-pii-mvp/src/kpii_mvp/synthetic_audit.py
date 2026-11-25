"""Read-only structural, policy and known-template audit of fictional proposals.

This audit checks labels against the explicit relation plan, not a language
model's judgement. A clean report never approves review records or establishes
that all personal information has been found. Human reading remains necessary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .b_data import document_fingerprint, load_reviews, validate_split_separation
from .b_inputs import CONTACT_TYPES, FIXED_POLICY_ID, allowed_to_keep, canonical_type
from .data import read_jsonl
from .schema import Document


ROLE_ACTIONS = {
    "private": "MASK", "personal": "MASK", "name": "MASK",
    "public_approved": "KEEP", "public_unapproved": "MASK",
}
AWKWARD_PHRASES = ("졸업한 학력을 적", "졸업한 학력을 기재")


def _normal_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _contact_key(kind: str, value: str) -> tuple[str, str]:
    normalized = _normal_text(value)
    if kind == "PHONE":
        normalized = "".join(char for char in normalized if char.isdecimal())
    elif kind == "EMAIL":
        normalized = normalized.casefold()
    return kind, normalized


def _core_clauses(document: Document) -> set[str]:
    """Value-elided span-bearing lines detect identical sentence templates.

    Non-span headers are excluded. This detects exact normalized reuse, not
    paraphrases or semantic equivalence. Offsets in the document are untouched.
    """
    signatures: set[str] = set()
    start = 0
    for line in document.text.splitlines(keepends=True):
        end = start + len(line)
        mentions = [mention for mention in document.mentions
                    if start <= mention.start and mention.end <= end]
        if any(canonical_type(mention.entity_type) != "NAME" for mention in mentions):
            parts, previous = [], start
            for mention in sorted(mentions, key=lambda item: item.start):
                parts.extend((document.text[previous:mention.start],
                              "[" + canonical_type(mention.entity_type) + "]"))
                previous = mention.end
            parts.append(document.text[previous:end])
            signatures.add(_normal_text("".join(parts)))
        start = end
    return signatures


def audit_corpus(documents: Iterable[Document], plan: dict[str, Any]) -> dict[str, Any]:
    """Audit proposals without mutating documents, plans, reviews or models."""
    docs = list(documents)
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    doc_findings: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def add(code: str, *, warning: bool = False, doc_ids: Iterable[str] = (),
            **details: Any) -> None:
        ids = sorted(set(doc_ids))
        item = {"code": code, "doc_ids": ids, **details}
        (warnings if warning else blockers).append(item)
        for doc_id in ids:
            doc_findings[doc_id].append(item)

    if not docs:
        add("empty_corpus")
    ids = [document.doc_id for document in docs]
    if len(ids) != len(set(ids)):
        add("duplicate_document_ids", doc_ids=[key for key, count in Counter(ids).items() if count > 1])
    plan_records = plan.get("documents") if isinstance(plan, dict) else None
    if not isinstance(plan_records, list):
        add("missing_plan_documents")
        plan_records = []
    plan_by_id: dict[str, dict[str, Any]] = {}
    for record in plan_records:
        if not isinstance(record, dict) or not isinstance(record.get("doc_id"), str):
            add("invalid_plan_document")
            continue
        doc_id = record["doc_id"]
        if doc_id in plan_by_id:
            add("duplicate_plan_document", doc_ids=[doc_id])
        plan_by_id[doc_id] = record
    if set(plan_by_id) != set(ids):
        add("plan_document_inventory_mismatch", missing=sorted(set(ids) - set(plan_by_id)),
            extra=sorted(set(plan_by_id) - set(ids)))

    splits: dict[str, list[Document]] = defaultdict(list)
    contacts: dict[tuple[str, str], dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: {"private": [], "keep": []})
    clause_owners: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    clause_ids: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    labels: Counter[str] = Counter()
    types: Counter[str] = Counter()
    count_patterns: Counter[str] = Counter()
    action_patterns: Counter[str] = Counter()
    type_patterns: Counter[str] = Counter()
    approved_by_label: Counter[str] = Counter()
    per_split: dict[str, dict[str, Any]] = {}
    valid_docs: list[Document] = []

    for document in docs:
        doc_id = document.doc_id
        try:
            document.validate()
        except (TypeError, ValueError) as exc:
            add("invalid_document_schema_or_spans", doc_ids=[doc_id], message=str(exc))
            continue
        valid_docs.append(document)
        split = document.split or "unspecified"
        splits[split].append(document)
        if document.policy_id != FIXED_POLICY_ID:
            add("wrong_policy", doc_ids=[doc_id])
        record = plan_by_id.get(doc_id, {})
        if record and record.get("split") != document.split:
            add("plan_split_mismatch", doc_ids=[doc_id])
        relations = record.get("relations", [])
        if not isinstance(relations, list):
            add("invalid_plan_relations", doc_ids=[doc_id])
            relations = []
        roles: dict[str, dict[str, Any]] = {}
        for relation in relations:
            if not isinstance(relation, dict) or not isinstance(relation.get("mention_id"), str):
                add("invalid_plan_relation", doc_ids=[doc_id])
                continue
            mention_id = relation["mention_id"]
            if mention_id in roles:
                add("duplicate_plan_relation", doc_ids=[doc_id], mention_id=mention_id)
            roles[mention_id] = relation
        mention_ids = {mention.mention_id for mention in document.mentions}
        if set(roles) != mention_ids:
            add("plan_mention_inventory_mismatch", doc_ids=[doc_id],
                missing=sorted(mention_ids - set(roles)), extra=sorted(set(roles) - mention_ids))
        for mention in document.mentions:
            kind = canonical_type(mention.entity_type)
            action = mention.action or "UNLABELLED"
            labels[action] += 1
            types[kind] += 1
            relation = roles.get(mention.mention_id, {})
            role = relation.get("relation")
            expected = ROLE_ACTIONS.get(role)
            if expected is None:
                add("unknown_semantic_role", doc_ids=[doc_id], mention_id=mention.mention_id, role=role)
            else:
                expected = "MASK" if kind == "NAME" else expected
                if relation.get("proposed_action", expected) != expected:
                    add("plan_action_disagrees_with_semantic_role", doc_ids=[doc_id],
                        mention_id=mention.mention_id, expected=expected,
                        actual=relation.get("proposed_action"))
                if mention.action != expected:
                    add("semantic_label_mismatch", doc_ids=[doc_id], mention_id=mention.mention_id,
                        role=role, expected=expected, actual=mention.action)
            if mention.action not in {"MASK", "KEEP"}:
                add("unlabelled_candidate", doc_ids=[doc_id], mention_id=mention.mention_id)
            try:
                approved = allowed_to_keep(document, mention)
            except (TypeError, ValueError) as exc:
                approved = False
                add("invalid_trusted_metadata", doc_ids=[doc_id], message=str(exc))
            approved_by_label[f"{action}:approved={int(approved)}"] += 1
            if mention.action == "KEEP" and not approved:
                add("keep_fails_policy_gate", doc_ids=[doc_id], mention_id=mention.mention_id)
            if role == "public_approved" and not approved:
                add("approved_role_missing_approval", doc_ids=[doc_id], mention_id=mention.mention_id)
            if role == "public_unapproved" and approved:
                add("unapproved_role_has_approval", doc_ids=[doc_id], mention_id=mention.mention_id)
            if role == "name" and kind != "NAME":
                add("name_role_has_wrong_type", doc_ids=[doc_id], mention_id=mention.mention_id)
            if kind in CONTACT_TYPES:
                occurrence = {"doc_id": doc_id, "mention_id": mention.mention_id}
                key = _contact_key(kind, mention.text)
                if role in {"private", "personal"}:
                    contacts[key]["private"].append(occurrence)
                if mention.action == "KEEP":
                    contacts[key]["keep"].append(occurrence)
            if kind == "POSITION" and re.search(r"가상.*제\d+", mention.text):
                add("serial_fictional_prefix_inside_position", doc_ids=[doc_id], mention_id=mention.mention_id)
        # Detect omitted repetitions of a known candidate, without pretending
        # that a string matcher discovers every unknown entity in a document.
        for value in {mention.text for mention in document.mentions}:
            cursor = 0
            while True:
                occurrence = document.text.find(value, cursor)
                if occurrence < 0:
                    break
                end = occurrence + len(value)
                if not any(item.start <= occurrence and end <= item.end for item in document.mentions):
                    add("unannotated_known_candidate_occurrence", doc_ids=[doc_id],
                        start=occurrence, end=end)
                cursor = occurrence + 1
        for phrase in AWKWARD_PHRASES:
            if phrase in document.text:
                add("known_awkward_phrase", doc_ids=[doc_id], phrase=phrase)
        for mention in document.mentions:
            if canonical_type(mention.entity_type) == "NAME" and "가" <= mention.text[-1] <= "힣":
                has_final = (ord(mention.text[-1]) - ord("가")) % 28 != 0
                tail = document.text[mention.end:mention.end + 4]
                if (has_final and re.match(r"가(?:\s|$)", tail)) or (
                        not has_final and re.match(r"이(?:\s|$)", tail)):
                    add("suspected_subject_particle_error", warning=True, doc_ids=[doc_id],
                        mention_id=mention.mention_id)
        for clause in _core_clauses(document):
            clause_owners[clause][split].add(doc_id)
        planned_clause_ids = record.get("clause_ids", [])
        if not isinstance(planned_clause_ids, list) or any(not isinstance(value, str) for value in planned_clause_ids):
            add("invalid_clause_ids", doc_ids=[doc_id])
        else:
            for clause_id in planned_clause_ids:
                clause_ids[clause_id][split].add(doc_id)
        count_patterns[str(len(document.mentions))] += 1
        actions = Counter(item.action for item in document.mentions)
        action_patterns[";".join(f"{key}:{actions[key]}" for key in sorted(actions, key=str))] += 1
        type_patterns[",".join(sorted({canonical_type(item.entity_type) for item in document.mentions}))] += 1

    if splits:
        try:
            validate_split_separation(splits)
        except ValueError as exc:
            add("split_identity_or_group_overlap", message=str(exc))
    for key, occurrences in contacts.items():
        if occurrences["private"] and occurrences["keep"]:
            add("private_contact_released_elsewhere", doc_ids=[
                item["doc_id"] for side in occurrences.values() for item in side],
                entity_type=key[0], normalized_value_sha256=hashlib.sha256(key[1].encode()).hexdigest(),
                private_occurrences=occurrences["private"], keep_occurrences=occurrences["keep"])
    for owners in clause_ids.values():
        if len(owners) > 1:
            add("clause_id_crosses_splits", doc_ids=[doc_id for ids_ in owners.values() for doc_id in ids_])
    shared_cores = [owners for owners in clause_owners.values() if len(owners) > 1]
    for owners in shared_cores:
        add("identical_core_clause_crosses_splits", doc_ids=[doc_id for ids_ in owners.values() for doc_id in ids_],
            splits=sorted(owners))
    if len(docs) >= 3 and len(count_patterns) <= 1:
        add("uniform_candidate_count", warning=True)
    if len(docs) >= 3 and len(action_patterns) <= 1:
        add("uniform_document_label_composition", warning=True)
    if len(docs) >= 3 and len(type_patterns) <= 1:
        add("uniform_document_type_composition", warning=True)
    if not approved_by_label["MASK:approved=1"]:
        add("no_approved_mask_context_counterexamples", warning=True)
    for split, partition in sorted(splits.items()):
        per_split[split] = {
            "documents": len(partition),
            "mentions": sum(len(item.mentions) for item in partition),
            "actions": dict(Counter(item.action for doc in partition for item in doc.mentions)),
            "types": dict(Counter(canonical_type(item.entity_type) for doc in partition for item in doc.mentions)),
        }
    return {
        "audit_version": "synthetic_audit_v2", "passed": not blockers,
        "documents": len(docs), "validated_documents": len(valid_docs),
        "mentions": sum(len(item.mentions) for item in valid_docs),
        "actions": dict(labels), "types": dict(types), "per_split": per_split,
        "candidate_count_distribution": dict(count_patterns),
        "label_composition_distribution": dict(action_patterns),
        "type_composition_distribution": dict(type_patterns),
        "approval_by_action": dict(approved_by_label),
        "unique_value_elided_clause_count": len(clause_owners),
        "cross_split_identical_core_clause_count": len(shared_cores),
        "blockers": blockers, "warnings": warnings,
        "document_findings": {doc_id: doc_findings.get(doc_id, []) for doc_id in ids},
        "review_status": "PENDING", "training_executed": False,
        "limitations": [
            "Roles are checked against a generated plan; independent human semantic review is still required.",
            "Known phrase and exact normalized clause checks do not certify Korean fluency or paraphrase independence.",
            "Known candidate repetitions are checked; completeness for unknown entities requires human review.",
            "Simulated approval metadata is not evidence of real-world public status.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("audit", help="Read-only proposal audit; never approve or train")
    command.add_argument("--input", required=True)
    command.add_argument("--plan", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--reviews", help="Optionally verify that every review record remains PENDING")
    args = parser.parse_args(argv)
    sources = {Path(value).resolve() for value in (args.input, args.plan, args.reviews) if value}
    output = Path(args.output).resolve()
    if output in sources or output.exists():
        raise ValueError("audit output must be a fresh path distinct from every input")
    documents = read_jsonl(args.input)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8-sig"))
    report = audit_corpus(documents, plan)
    if args.reviews:
        reviews = load_reviews(args.reviews)
        expected = {document.doc_id for document in documents}
        if set(reviews) != expected or any(item.get("review_status") != "PENDING" for item in reviews.values()):
            report["blockers"].append({"code": "reviews_not_all_pending_or_inventory_mismatch", "doc_ids": []})
            report["passed"] = False
        stale = [document.doc_id for document in documents if document.doc_id in reviews
                 and reviews[document.doc_id].get("document_sha256") != document_fingerprint(document)]
        if stale:
            report["blockers"].append({"code": "review_document_fingerprint_mismatch", "doc_ids": stale})
            report["passed"] = False
        report["review_records_checked"] = len(reviews)
    report["input_sha256"] = hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
    report["plan_sha256"] = hashlib.sha256(Path(args.plan).read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "passed": report["passed"],
                      "blockers": len(report["blockers"]), "warnings": len(report["warnings"]),
                      "review_status": "PENDING", "training_executed": False}, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
