"""Detection and end-to-end masking metrics with explicit gold denominators.

Masking scores use FINAL mask intervals in original coordinates, not the subset
of candidates produced by stage A. Gold action=None is never treated as KEEP.
"""

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from .masking import merge_intervals


def _field(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _index_documents(documents: Iterable[Any], label: str) -> dict[str, Any]:
    indexed = {}
    for document in documents:
        doc_id = _field(document, "doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError(f"{label} records need a nonempty doc_id")
        if doc_id in indexed:
            raise ValueError(f"Duplicate {label} doc_id: {doc_id}")
        if not isinstance(_field(document, "text"), str):
            raise ValueError(f"{label} document {doc_id} needs original text")
        indexed[doc_id] = document
    return indexed


def _paired_documents(
    gold_documents: Iterable[Any], predicted_documents: Iterable[Any]
) -> list[tuple[str, Any, Any]]:
    gold = _index_documents(gold_documents, "gold")
    predicted = _index_documents(predicted_documents, "predicted")
    if set(gold) != set(predicted):
        missing = sorted(set(gold) - set(predicted))
        extra = sorted(set(predicted) - set(gold))
        raise ValueError(f"Document IDs differ: missing={missing}, extra={extra}")
    pairs = []
    for doc_id in sorted(gold):
        if _field(gold[doc_id], "text") != _field(predicted[doc_id], "text"):
            raise ValueError(f"Original text mismatch for doc_id: {doc_id}")
        pairs.append((doc_id, gold[doc_id], predicted[doc_id]))
    return pairs


def _validated_mentions(document: Any, require_actions: bool = False) -> list[Any]:
    text = _field(document, "text")
    mentions = _field(document, "mentions")
    if not isinstance(mentions, (list, tuple)):
        raise ValueError("A document must provide a mentions list or tuple")
    ids = set()
    for mention in mentions:
        start, end = _field(mention, "start"), _field(mention, "end")
        merge_intervals(text, [(start, end)])
        if _field(mention, "text") != text[start:end]:
            raise ValueError("Mention text does not match its original-coordinate span")
        entity_type = _field(mention, "entity_type")
        if not isinstance(entity_type, str) or not entity_type:
            raise ValueError("Each mention needs a nonempty entity_type")
        mention_id = _field(mention, "mention_id")
        if not isinstance(mention_id, str) or not mention_id or mention_id in ids:
            raise ValueError("Mention IDs must be nonempty and unique within a document")
        ids.add(mention_id)
        if require_actions and _field(mention, "action") not in ("MASK", "KEEP"):
            raise ValueError(
                "Integrated masking evaluation requires explicit MASK/KEEP gold "
                "for every annotated mention; null is not KEEP"
            )
    return list(mentions)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _detection_metrics(tp: int, gold: int, predicted: int) -> dict[str, Any]:
    precision = _ratio(tp, predicted)
    recall = _ratio(tp, gold)
    return {
        "true_positive": tp,
        "false_positive": predicted - tp,
        "false_negative": gold - tp,
        "gold_mentions": gold,
        "predicted_mentions": predicted,
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
    }


def _mention_key(mention: Any) -> tuple[int, int, str]:
    return (_field(mention, "start"), _field(mention, "end"), _field(mention, "entity_type"))


def _coverage_counts(
    gold_mentions: list[Any], predicted_intervals: list[tuple[int, int]]
) -> list[int]:
    """Count full gold-span coverage by an already merged prediction union.

    Full coverage of a contiguous gold span must lie inside one merged interval.
    Predicted labels and actions do not affect candidate coverage. Protection is
    counted only for explicitly annotated gold MASK mentions.
    """
    covered = protected_covered = protected_gold = 0
    for mention in gold_mentions:
        start, end = _field(mention, "start"), _field(mention, "end")
        fully_covered = any(left <= start and end <= right for left, right in predicted_intervals)
        covered += int(fully_covered)
        if _field(mention, "action") == "MASK":
            protected_gold += 1
            protected_covered += int(fully_covered)
    return [covered, len(gold_mentions), protected_covered, protected_gold]


def _coverage_metrics(counts: list[int]) -> dict[str, Any]:
    covered, gold, protected_covered, protected_gold = counts
    return {
        "candidate_span_coverage_recall": _ratio(covered, gold),
        "candidate_counts": {"covered_mentions": covered, "gold_mentions": gold},
        "protected_span_coverage_recall": (
            protected_covered / protected_gold if protected_gold else None
        ),
        "protected_counts": {
            "covered_mask_mentions": protected_covered,
            "gold_mask_mentions": protected_gold,
        },
    }


def evaluate_detection(
    gold_documents: Iterable[Any], predicted_documents: Iterable[Any]
) -> dict[str, Any]:
    """Exact span+type micro P/R/F1, matching separately inside each document.

    Duplicate predicted spans with distinct IDs count as extra predictions rather
    than being silently deduplicated. Candidate coverage ignores predicted type
    and requires every character of a gold span to be covered by the prediction
    union. Protected coverage uses only explicit gold MASK annotations and is
    null when none exist. Other zero-denominator metrics return 0.0.
    """
    total = [0, 0, 0]
    coverage_total = [0, 0, 0, 0]
    by_type: dict[str, list[int]] = {}
    by_group: dict[str, list[int]] = {}
    coverage_by_type: dict[str, list[int]] = {}
    coverage_by_group: dict[str, list[int]] = {}
    document_results = []
    for doc_id, gold_document, predicted_document in _paired_documents(gold_documents, predicted_documents):
        gold_mentions = _validated_mentions(gold_document)
        predicted_mentions = _validated_mentions(predicted_document)
        gold = Counter(_mention_key(mention) for mention in gold_mentions)
        predicted = Counter(_mention_key(mention) for mention in predicted_mentions)
        matches = gold & predicted
        counts = [sum(matches.values()), sum(gold.values()), sum(predicted.values())]
        predicted_intervals = merge_intervals(
            _field(gold_document, "text"),
            [(_field(mention, "start"), _field(mention, "end")) for mention in predicted_mentions],
        )
        coverage = _coverage_counts(gold_mentions, predicted_intervals)
        for index, value in enumerate(counts):
            total[index] += value
        for index, value in enumerate(coverage):
            coverage_total[index] += value
        group = str(_field(gold_document, "source_group", "unknown"))
        group_counts = by_group.setdefault(group, [0, 0, 0])
        for index, value in enumerate(counts):
            group_counts[index] += value
        group_coverage = coverage_by_group.setdefault(group, [0, 0, 0, 0])
        for index, value in enumerate(coverage):
            group_coverage[index] += value
        for entity_type in {key[2] for key in gold.keys() | predicted.keys()}:
            type_counts = by_type.setdefault(entity_type, [0, 0, 0])
            type_counts[0] += sum(value for key, value in matches.items() if key[2] == entity_type)
            type_counts[1] += sum(value for key, value in gold.items() if key[2] == entity_type)
            type_counts[2] += sum(value for key, value in predicted.items() if key[2] == entity_type)
            type_coverage = _coverage_counts(
                [mention for mention in gold_mentions if _field(mention, "entity_type") == entity_type],
                predicted_intervals,
            )
            target_coverage = coverage_by_type.setdefault(entity_type, [0, 0, 0, 0])
            for index, value in enumerate(type_coverage):
                target_coverage[index] += value
        document_results.append({
            "doc_id": doc_id,
            **_detection_metrics(*counts),
            **_coverage_metrics(coverage),
        })
    return {
        "evaluation": "exact_span_and_type_detection",
        "rate_scale": "0_to_1",
        "zero_division": 0.0,
        "coverage_definitions": {
            "candidate_span_coverage_recall": "Gold candidate mentions fully covered by the union of predicted spans, ignoring predicted type / all gold candidate mentions",
            "protected_span_coverage_recall": "Explicit gold MASK mentions fully covered by the union of predicted spans / explicit gold MASK mentions; null if denominator is zero",
        },
        "document_count": len(document_results),
        "overall": {**_detection_metrics(*total), **_coverage_metrics(coverage_total)},
        "by_type": {
            key: {**_detection_metrics(*values), **_coverage_metrics(coverage_by_type[key])}
            for key, values in sorted(by_type.items())
        },
        "by_source_group": {
            key: {**_detection_metrics(*values), **_coverage_metrics(coverage_by_group[key])}
            for key, values in sorted(by_group.items())
        },
        "documents": document_results,
    }


def _empty_mask_counts() -> dict[str, int]:
    return {
        "gold_mask_mentions": 0,
        "gold_keep_mentions": 0,
        "leaked_mask_mentions": 0,
        "overmasked_keep_mentions": 0,
        "documents": 0,
        "documents_with_gold_mask": 0,
        "documents_with_leak": 0,
    }


def _mask_metrics(counts: dict[str, int]) -> dict[str, Any]:
    return {
        **counts,
        "leak_rate": _ratio(counts["leaked_mask_mentions"], counts["gold_mask_mentions"]),
        "overmask_rate": _ratio(counts["overmasked_keep_mentions"], counts["gold_keep_mentions"]),
        "document_leak_rate": _ratio(counts["documents_with_leak"], counts["documents_with_gold_mask"]),
    }


def _add_counts(target: dict[str, int], values: dict[str, int]) -> None:
    for key in target:
        target[key] += values[key]


def _mention_mask_result(mention: Any, intervals: list[tuple[int, int]]) -> tuple[bool, bool]:
    start, end = _field(mention, "start"), _field(mention, "end")
    covered = sum(max(0, min(end, mask_end) - max(start, mask_start)) for mask_start, mask_end in intervals)
    if _field(mention, "action") == "MASK":
        return covered < end - start, False
    return False, covered > 0


def _count_mask_document(mentions: list[Any], intervals: list[tuple[int, int]]) -> dict[str, int]:
    counts = _empty_mask_counts()
    counts["documents"] = 1
    for mention in mentions:
        leak, overmask = _mention_mask_result(mention, intervals)
        if _field(mention, "action") == "MASK":
            counts["gold_mask_mentions"] += 1
        else:
            counts["gold_keep_mentions"] += 1
        counts["leaked_mask_mentions"] += int(leak)
        counts["overmasked_keep_mentions"] += int(overmask)
    counts["documents_with_gold_mask"] = int(counts["gold_mask_mentions"] > 0)
    counts["documents_with_leak"] = int(counts["leaked_mask_mentions"] > 0)
    return counts


def evaluate_masking(gold_documents: Iterable[Any], mask_documents: Iterable[Any]) -> dict[str, Any]:
    """Score final mask intervals against all gold MASK and KEEP mentions.

    A mask record must contain doc_id, unmodified original text, and intervals
    ([start, end] pairs or {start, end} objects). An absent prediction document is
    an error; provide intervals=[] to explicitly represent an unmasked document.
    This evaluator cannot establish whether the human/generator gold is complete.
    """
    total = _empty_mask_counts()
    by_type: dict[str, dict[str, int]] = {}
    by_group: dict[str, dict[str, int]] = {}
    document_results = []
    for doc_id, gold_document, mask_document in _paired_documents(gold_documents, mask_documents):
        mentions = _validated_mentions(gold_document, require_actions=True)
        raw_intervals = _field(mask_document, "intervals")
        if not isinstance(raw_intervals, (list, tuple)):
            raise ValueError("Each mask document needs an explicit intervals list")
        intervals = merge_intervals(_field(gold_document, "text"), raw_intervals)
        counts = _count_mask_document(mentions, intervals)
        _add_counts(total, counts)
        group = str(_field(gold_document, "source_group", "unknown"))
        _add_counts(by_group.setdefault(group, _empty_mask_counts()), counts)
        for entity_type in {_field(mention, "entity_type") for mention in mentions}:
            type_mentions = [mention for mention in mentions if _field(mention, "entity_type") == entity_type]
            type_counts = _count_mask_document(type_mentions, intervals)
            _add_counts(by_type.setdefault(entity_type, _empty_mask_counts()), type_counts)
        document_results.append({
            "doc_id": doc_id,
            **_mask_metrics(counts),
            "masked_characters": sum(end - start for start, end in intervals),
            "mask_interval_count": len(intervals),
        })
    return {
        "evaluation": "final_original_coordinate_masking",
        "rate_scale": "0_to_1",
        "zero_division": 0.0,
        "unit": "gold_mention",
        "definitions": {
            "leak_rate": "Gold MASK mentions with at least one uncovered character / all gold MASK mentions",
            "overmask_rate": "Gold KEEP mentions with at least one covered character / all gold KEEP mentions",
            "document_leak_rate": "Documents with a leaked gold MASK mention / documents containing gold MASK mentions",
        },
        "overall": _mask_metrics(total),
        "by_type": {key: _mask_metrics(values) for key, values in sorted(by_type.items())},
        "by_source_group": {key: _mask_metrics(values) for key, values in sorted(by_group.items())},
        "documents": document_results,
    }
