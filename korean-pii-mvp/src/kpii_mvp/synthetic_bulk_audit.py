"""Read-only checks for the quota-controlled, fictional B proposal corpus.

The inherited audit checks generated roles, exact spans and known leakage
patterns. Extra checks verify the declared allocation and expose a deliberately
simple approval-only baseline. Passing these checks neither approves a document
nor certifies linguistic quality or real-world protection performance.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
import unicodedata
from typing import Any, Iterable

from .b_inputs import CONTEXTUAL_TYPES, allowed_to_keep, canonical_type
from .schema import Document
from .synthetic_audit import audit_corpus


GENERATOR_VERSION = "synthetic_rule_based_v3_bulk"
MODES = ("mixed", "all_mask", "all_keep")
SUBROLES = ("private_approved", "private_unapproved", "public_unapproved", "public_approved")
TYPES = tuple(sorted(CONTEXTUAL_TYPES))
METADATA_ONLY_ACCURACY_WARNING_THRESHOLD = 0.90
MINIMUM_APPROVED_PRIVATE_MASK_FRACTION = 0.20


def _count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _approval_metrics(counts: Counter[str]) -> dict[str, Any]:
    total = counts["candidates"]
    correct = counts["correct"]
    counterexamples = counts["approved_private_mask"]
    return {
        "candidates": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "approved_private_mask": counterexamples,
        "approved_private_mask_fraction": counterexamples / total if total else None,
        "confusion": {key: counts[key] for key in (
            "gold_MASK_predicted_MASK", "gold_MASK_predicted_KEEP",
            "gold_KEEP_predicted_MASK", "gold_KEEP_predicted_KEEP")},
    }


def audit_bulk(documents: Iterable[Document], plan: dict[str, Any]) -> dict[str, Any]:
    """Validate quotas and scenario independence without approving or fitting.

    NAME candidates are deliberately excluded from B quotas, modes and baseline
    metrics because the fixed policy always masks them. All mentions remain
    covered by the inherited structural and role audit.
    """
    docs = list(documents)
    report = audit_corpus(docs, plan)
    report["base_audit_version"] = report["audit_version"]
    report["audit_version"] = "synthetic_bulk_audit_v3"
    blockers, warnings = report["blockers"], report["warnings"]

    def add(code: str, *, warning: bool = False, doc_ids: Iterable[str] = (),
            **details: Any) -> None:
        ids = sorted(set(doc_ids))
        item = {"code": code, "doc_ids": ids, **details}
        (warnings if warning else blockers).append(item)
        for doc_id in ids:
            report["document_findings"].setdefault(doc_id, []).append(item)

    if not isinstance(plan, dict):
        add("bulk_plan_not_object")
        plan = {}
    if plan.get("generator_version") != GENERATOR_VERSION:
        add("wrong_bulk_generator_version", expected=GENERATOR_VERSION,
            actual=plan.get("generator_version"))
    records = plan.get("documents", [])
    records = records if isinstance(records, list) else []
    by_id = {record["doc_id"]: record for record in records
             if isinstance(record, dict) and isinstance(record.get("doc_id"), str)}
    family_count = plan.get("family_count_per_split", 10)
    if type(family_count) is not int or family_count < 1:
        add("invalid_family_count_per_split")
        family_count = 10

    actual_quotas: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter))
    actual_modes: dict[str, Counter[str]] = defaultdict(Counter)
    families: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    family_splits: dict[str, set[str]] = defaultdict(set)
    per_type: dict[str, Counter[str]] = defaultdict(Counter)
    per_split_type: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    actual_subroles: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    name_counts: Counter[str] = Counter()

    for document in docs:
        # The inherited report already records malformed spans. Avoid using
        # their contents for this audit's quota or approval statistics.
        try:
            document.validate()
        except (TypeError, ValueError):
            continue
        split = document.split or "unspecified"
        record = by_id.get(document.doc_id, {})
        planned_clause_ids = record.get("clause_ids", [])
        if (isinstance(planned_clause_ids, list)
                and all(isinstance(value, str) for value in planned_clause_ids)):
            repeated = sorted(value for value, count in Counter(planned_clause_ids).items()
                              if count > 1)
            if repeated:
                add("repeated_clause_id_within_document", doc_ids=[document.doc_id],
                    clause_ids=repeated)
        # This is a generator-quality rule for V3 proposals, not a claim that
        # repeating a sentence makes a real source document invalid. Unlike
        # the cross-split template check, candidate values remain in the key.
        line_occurrences: dict[str, list[int]] = defaultdict(list)
        cursor = 0
        for line_number, line in enumerate(document.text.splitlines(keepends=True), 1):
            end = cursor + len(line)
            if any(cursor <= mention.start and mention.end <= end
                   and canonical_type(mention.entity_type) in CONTEXTUAL_TYPES
                   for mention in document.mentions):
                signature = " ".join(unicodedata.normalize("NFKC", line).split())
                line_occurrences[signature].append(line_number)
            cursor = end
        for lines in line_occurrences.values():
            if len(lines) > 1:
                add("repeated_span_bearing_line_within_document", doc_ids=[document.doc_id],
                    line_numbers=lines)
        relation_rows = record.get("relations", [])
        relations = {item["mention_id"]: item for item in relation_rows
                     if isinstance(item, dict) and isinstance(item.get("mention_id"), str)} \
            if isinstance(relation_rows, list) else {}
        candidates = [mention for mention in document.mentions
                      if canonical_type(mention.entity_type) in CONTEXTUAL_TYPES]
        if not candidates:
            add("document_has_no_b_candidates", doc_ids=[document.doc_id])
            actual_mode = "empty"
        else:
            actions = {mention.action for mention in candidates}
            actual_mode = ("all_mask" if actions == {"MASK"} else "all_keep"
                           if actions == {"KEEP"} else "mixed")
        actual_modes[split][actual_mode] += 1
        if record.get("mode") != actual_mode:
            add("document_mode_mismatch", doc_ids=[document.doc_id],
                expected=record.get("mode"), actual=actual_mode)
        family = record.get("family_id")
        if not isinstance(family, str) or not family:
            add("missing_document_family", doc_ids=[document.doc_id])
        else:
            families[split][family][actual_mode] += 1
            family_splits[family].add(split)
        for mention in document.mentions:
            kind = canonical_type(mention.entity_type)
            if kind == "NAME":
                name_counts[split] += 1
            if kind not in CONTEXTUAL_TYPES:
                continue
            action = mention.action or "UNLABELLED"
            actual_quotas[split][kind][action] += 1
            try:
                approved = allowed_to_keep(document, mention)
            except (TypeError, ValueError):
                # The base audit records invalid trusted metadata as a blocker.
                continue
            predicted = "KEEP" if approved else "MASK"
            relation_record = relations.get(mention.mention_id, {})
            role = relation_record.get("relation")
            subrole = (("private_approved" if approved else "private_unapproved")
                       if role in {"private", "personal"} else
                       ("public_approved" if approved else "public_unapproved")
                       if role in {"public_approved", "public_unapproved"} else None)
            if subrole is not None:
                actual_subroles[split][kind][subrole] += 1
            if ("quota_subrole" in relation_record
                    and relation_record["quota_subrole"] != subrole):
                add("declared_quota_subrole_mismatch", doc_ids=[document.doc_id],
                    mention_id=mention.mention_id, actual=subrole,
                    declared=relation_record["quota_subrole"])
            if ("simulated_approved_value" in relation_record
                    and (type(relation_record["simulated_approved_value"]) is not bool
                         or relation_record["simulated_approved_value"] != approved)):
                add("declared_simulated_approval_mismatch", doc_ids=[document.doc_id],
                    mention_id=mention.mention_id, actual=approved,
                    declared=relation_record["simulated_approved_value"])
            for counts in (per_type[kind], per_split_type[split][kind]):
                counts["candidates"] += 1
                counts["correct"] += int(action == predicted)
                counts[f"gold_{action}_predicted_{predicted}"] += 1
                counts["approved_private_mask"] += int(
                    approved and action == "MASK" and role in {"private", "personal"})

    quota_targets = plan.get("quota_targets")
    mode_targets = plan.get("mode_targets")
    if not isinstance(quota_targets, dict) or not quota_targets:
        add("missing_or_invalid_quota_targets")
        quota_targets = {}
    if not isinstance(mode_targets, dict) or not mode_targets:
        add("missing_or_invalid_mode_targets")
        mode_targets = {}
    actual_splits = set(actual_modes)
    if set(quota_targets) != actual_splits:
        add("quota_split_inventory_mismatch", actual=sorted(actual_splits),
            expected=sorted(quota_targets))
    if set(mode_targets) != actual_splits:
        add("mode_split_inventory_mismatch", actual=sorted(actual_splits),
            expected=sorted(mode_targets))

    for split, targets in quota_targets.items():
        if not isinstance(targets, dict) or set(targets) != set(TYPES):
            add("invalid_type_quota_inventory", split=split, expected=list(TYPES))
            continue
        for kind, actions in targets.items():
            if (not isinstance(actions, dict) or set(actions) != {"MASK", "KEEP"}
                    or any(not _count(value) for value in actions.values())):
                add("invalid_type_action_quota", split=split, entity_type=kind)
                continue
            for action, expected in actions.items():
                actual = actual_quotas[split][kind][action]
                if actual != expected:
                    add("type_action_quota_mismatch", split=split, entity_type=kind,
                        action=action, expected=expected, actual=actual)

    # These optional targets refine MASK into approved personal context,
    # unapproved personal context and unapproved public context. Derivation
    # uses the actual policy gate, never a generated approval assertion.
    subrole_targets = plan.get("subrole_targets_per_type")
    if "subrole_targets_per_type" in plan:
        if not isinstance(subrole_targets, dict) or not subrole_targets:
            add("missing_or_invalid_subrole_targets")
            subrole_targets = {}
        if set(subrole_targets) != actual_splits:
            add("subrole_split_inventory_mismatch", actual=sorted(actual_splits),
                expected=sorted(subrole_targets))
        for split, targets in subrole_targets.items():
            if (not isinstance(targets, dict) or set(targets) != set(SUBROLES)
                    or any(not _count(value) for value in targets.values())):
                add("invalid_subrole_targets", split=split)
                continue
            for kind in TYPES:
                for subrole, expected in targets.items():
                    actual = actual_subroles[split][kind][subrole]
                    if actual != expected:
                        add("type_subrole_quota_mismatch", split=split, entity_type=kind,
                            subrole=subrole, expected=expected, actual=actual)

    family_contingency: dict[str, dict[str, Any]] = {}
    maximum_gap = 0.0
    for split, targets in mode_targets.items():
        if (not isinstance(targets, dict) or set(targets) != set(MODES)
                or any(not _count(value) for value in targets.values())
                or sum(targets.values()) == 0):
            add("invalid_mode_targets", split=split)
            continue
        for mode, expected in targets.items():
            actual = actual_modes[split][mode]
            if actual != expected:
                add("document_mode_quota_mismatch", split=split, mode=mode,
                    expected=expected, actual=actual)
        family_rows = families[split]
        if len(family_rows) != family_count:
            add("family_count_mismatch", split=split,
                expected=family_count, actual=len(family_rows))
        target_documents = sum(targets.values())
        family_contingency[split] = {}
        for family, counts in sorted(family_rows.items()):
            size = sum(counts.values())
            expected_row = {}
            for mode in MODES:
                expected = Fraction(targets[mode] * size, target_documents)
                expected_row[mode] = (expected.numerator if expected.denominator == 1
                                      else float(expected))
                if expected.denominator != 1 or counts[mode] != expected:
                    add("family_mode_contingency_mismatch", split=split,
                        family_id=family, mode=mode, expected=expected_row[mode],
                        actual=counts[mode])
                maximum_gap = max(maximum_gap, abs(
                    counts[mode] / size - targets[mode] / target_documents))
            family_contingency[split][family] = {
                "documents": size, "modes": {mode: counts[mode] for mode in MODES},
                "expected_modes": expected_row,
            }
    for family, owners in sorted(family_splits.items()):
        if len(owners) > 1:
            add("family_id_crosses_splits", family_id=family, splits=sorted(owners))

    metrics = {kind: _approval_metrics(per_type[kind]) for kind in TYPES}
    split_metrics = {split: {kind: _approval_metrics(counts[kind]) for kind in TYPES}
                     for split, counts in sorted(per_split_type.items())}
    for split, rows in [("all", metrics), *split_metrics.items()]:
        for kind, values in rows.items():
            accuracy = values["accuracy"]
            counterexample_fraction = values["approved_private_mask_fraction"]
            if accuracy is not None and accuracy > METADATA_ONLY_ACCURACY_WARNING_THRESHOLD:
                add("high_metadata_approval_only_accuracy", warning=True, split=split,
                    entity_type=kind, accuracy=accuracy,
                    warning_above=METADATA_ONLY_ACCURACY_WARNING_THRESHOLD)
            if counterexample_fraction is not None and counterexample_fraction < MINIMUM_APPROVED_PRIVATE_MASK_FRACTION:
                add("insufficient_approved_private_mask_counterexamples", warning=True,
                    split=split, entity_type=kind, fraction=counterexample_fraction,
                    recommended_minimum=MINIMUM_APPROVED_PRIVATE_MASK_FRACTION)

    report.update({
        "passed": not blockers,
        "b_candidates": sum(counts["candidates"] for counts in per_type.values()),
        "b_type_action_counts": {
            split: {kind: {action: actions[action] for action in ("MASK", "KEEP")}
                    for kind, actions in sorted(rows.items())}
            for split, rows in sorted(actual_quotas.items())},
        "name_candidates_excluded_from_b": dict(sorted(name_counts.items())),
        "b_type_subrole_counts": {
            split: {kind: {subrole: counts[subrole] for subrole in SUBROLES}
                    for kind, counts in sorted(rows.items())}
            for split, rows in sorted(actual_subroles.items())},
        "document_mode_counts": {split: dict(sorted(counts.items()))
                                 for split, counts in sorted(actual_modes.items())},
        "family_mode_contingency": family_contingency,
        "family_mode_max_absolute_proportion_gap": maximum_gap,
        "metadata_approval_only_accuracy": metrics,
        "metadata_approval_only_accuracy_per_split": split_metrics,
        "metadata_baseline_definition": "KEEP iff the exact-value approval policy gate permits KEEP; otherwise MASK",
        "review_status": "PENDING", "human_review_required": True,
        "naturalness_certification": "not_established_by_this_automated_audit",
        "training_executed": False,
    })
    report["limitations"].extend([
        "Matching allocation quotas does not demonstrate population representativeness.",
        "Approval-only accuracy is a fixed heuristic diagnostic, not trained B performance.",
        "Balanced family/mode counts do not rule out other text or metadata shortcuts.",
    ])
    return report
