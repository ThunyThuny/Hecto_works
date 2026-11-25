"""Offset-preserving, offline-testable token/window alignment for Stage A.

Offsets are Python string indices into the *unchanged* source text.  Gold spans
are never expanded to token boundaries.  Tokens touching an unrepresentable or
window-clipped annotation receive -100, not the misleading negative label O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

IGNORE_INDEX = -100


@dataclass(frozen=True)
class TokenWindow:
    input_ids: list[int]
    attention_mask: list[int]
    offsets: list[tuple[int, int]]
    special_tokens_mask: list[int]
    token_type_ids: list[int] | None = None
    index: int = 0

    @property
    def content_indices(self) -> list[int]:
        return [i for i, ((s, e), special, attention) in enumerate(zip(
            self.offsets, self.special_tokens_mask, self.attention_mask
        )) if attention and not special and e > s]

    @property
    def bounds(self) -> tuple[int, int]:
        indices = self.content_indices
        return (self.offsets[indices[0]][0], self.offsets[indices[-1]][1]) if indices else (0, 0)

    def model_inputs(self) -> dict[str, list[int]]:
        result = {"input_ids": self.input_ids, "attention_mask": self.attention_mask}
        if self.token_type_ids is not None:
            result["token_type_ids"] = self.token_type_ids
        return result


@dataclass
class AlignmentReport:
    complete_mentions: int = 0
    clipped_mentions: int = 0
    unaligned_mentions: int = 0
    ignored_tokens: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete_mentions": self.complete_mentions,
            "clipped_mentions": self.clipped_mentions,
            "unaligned_mentions": self.unaligned_mentions,
            "ignored_tokens": self.ignored_tokens,
            "details": self.details,
        }


def _field(mention: Any, name: str) -> Any:
    return mention[name] if isinstance(mention, Mapping) else getattr(mention, name)


def tokenize_document(tokenizer: Any, text: str, max_length: int = 512,
                      stride: int = 128) -> list[TokenWindow]:
    """Use every overflow window; require a fast tokenizer with global offsets."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Stage A requires a fast tokenizer with original character offsets")
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    content_capacity = max_length - special_count
    if content_capacity < 1 or not 0 <= stride < content_capacity:
        raise ValueError("max_length must fit content and 0 <= stride < content capacity")
    encoded = tokenizer(
        text, truncation=True, max_length=max_length, stride=stride,
        return_overflowing_tokens=True, return_offsets_mapping=True,
        return_special_tokens_mask=True, return_token_type_ids=False, padding=False,
    )
    rows = encoded["input_ids"]
    # Some lightweight tokenizers return a flat row for a single sequence.
    nested = bool(rows) and isinstance(rows[0], (list, tuple))
    if not nested:
        encoded = {k: [v] for k, v in encoded.items() if k != "overflow_to_sample_mapping"}
        rows = encoded["input_ids"]
    windows: list[TokenWindow] = []
    last_start = -1
    for n, ids in enumerate(rows):
        offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"][n]]
        attention = list(encoded.get("attention_mask", [[1] * len(r) for r in rows])[n])
        specials = list(encoded["special_tokens_mask"][n])
        token_types = encoded.get("token_type_ids")
        if not len(ids) == len(offsets) == len(attention) == len(specials):
            raise ValueError("Tokenizer returned inconsistent window lengths")
        if len(ids) > max_length:
            raise ValueError("Tokenizer did not honor max_length")
        for start, end in offsets:
            if not 0 <= start <= end <= len(text):
                raise ValueError("Tokenizer offsets are not indices into the original text")
        window = TokenWindow(list(ids), attention, offsets, specials,
                             list(token_types[n]) if token_types is not None else None, n)
        content = [offsets[i] for i in window.content_indices]
        if any(a[0] > b[0] or a[1] > b[1] for a, b in zip(content, content[1:])):
            raise ValueError("Tokenizer content offsets are not monotonic")
        if content and content[0][0] < last_start:
            raise ValueError("Overflow windows do not use global original-text offsets")
        if content:
            last_start = content[0][0]
        windows.append(window)
    if not windows:
        raise ValueError("Tokenizer returned no windows")
    # Prove coverage of all meaningful original characters, including interior
    # overflow gaps.  Whitespace may legitimately have no tokenizer offset.
    covered_end = 0
    all_offsets = sorted({w.offsets[i] for w in windows for i in w.content_indices})
    for start, end in all_offsets:
        if start > covered_end and text[covered_end:start].strip():
            raise ValueError("Tokenizer omitted a non-whitespace original-text interval")
        covered_end = max(covered_end, end)
    if text[covered_end:].strip():
        raise ValueError("Tokenizer omitted a non-whitespace document suffix")
    return windows


def align_mentions(window: TokenWindow, mentions: Sequence[Any],
                   label2id: Mapping[str, int], boundary_policy: str = "ignore"
                   ) -> tuple[list[int], AlignmentReport]:
    """Align complete exact-boundary BIO spans, conservatively ignore others.

    ``reject`` rejects non-token-aligned complete annotations.  Window-clipped
    annotations are always ignored in that window; another overlap window may
    contain their complete span.  Callers must supply full annotation coverage.
    """
    if boundary_policy not in {"ignore", "reject"}:
        raise ValueError("boundary_policy must be ignore or reject")
    if "O" not in label2id:
        raise ValueError("label inventory is missing O")
    labels = [IGNORE_INDEX] * len(window.input_ids)
    content = window.content_indices
    for i in content:
        labels[i] = label2id["O"]
    report = AlignmentReport()
    if not content:
        return labels, report
    window_start, window_end = window.bounds
    occupied: set[int] = set()
    ignored: set[int] = set()
    for mention in sorted(mentions, key=lambda m: (_field(m, "start"), _field(m, "end"))):
        start, end = int(_field(mention, "start")), int(_field(mention, "end"))
        kind = str(_field(mention, "entity_type"))
        if start >= end:
            raise ValueError("Mention offsets must describe a nonempty span")
        if f"B-{kind}" not in label2id or f"I-{kind}" not in label2id:
            raise ValueError(f"Unknown entity type {kind!r} in annotation")
        touched = [i for i in content if window.offsets[i][0] < end and window.offsets[i][1] > start]
        if not touched:
            continue
        clipped = start < window_start or end > window_end
        exact = window.offsets[touched[0]][0] == start and window.offsets[touched[-1]][1] == end
        if clipped or not exact:
            reason = "window_clipped" if clipped else "not_token_aligned"
            if not clipped and boundary_policy == "reject":
                raise ValueError(f"Gold span [{start}, {end}) is not exactly token-aligned")
            ignored.update(touched)
            if clipped:
                report.clipped_mentions += 1
            else:
                report.unaligned_mentions += 1
            report.details.append({"start": start, "end": end, "entity_type": kind, "reason": reason})
            continue
        if occupied.intersection(touched):
            raise ValueError("Overlapping gold entities cannot be represented by a single BIO head")
        occupied.update(touched)
        labels[touched[0]] = label2id[f"B-{kind}"]
        for i in touched[1:]:
            labels[i] = label2id[f"I-{kind}"]
        report.complete_mentions += 1
    for i in ignored:
        labels[i] = IGNORE_INDEX
    report.ignored_tokens = len(ignored)
    return labels, report


@dataclass(frozen=True)
class SpanPrediction:
    start: int
    end: int
    entity_type: str
    score: float = 1.0
    left_edge: bool = False
    right_edge: bool = False


def resolve_spans(candidates: Sequence[SpanPrediction]) -> tuple[list[SpanPrediction], dict[str, int]]:
    """Stitch matching overflow-edge fragments and resolve remaining overlaps.

    The deterministic preference is: complete-window evidence, confidence,
    length, then source position and label.  No token boundaries are fabricated.
    """
    spans = sorted(candidates, key=lambda p: (p.start, p.end, p.entity_type, -p.score))
    stitched = 0
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(spans):
            for j in range(i + 1, len(spans)):
                b = spans[j]
                if b.start >= a.end:
                    break
                if (a.entity_type == b.entity_type and a.start <= b.start and a.end < b.end
                        and a.right_edge and b.left_edge):
                    spans[i] = SpanPrediction(a.start, b.end, a.entity_type,
                        min(a.score, b.score), a.left_edge, b.right_edge)
                    spans.pop(j)
                    spans.sort(key=lambda p: (p.start, p.end, p.entity_type, -p.score))
                    stitched += 1
                    changed = True
                    break
            if changed:
                break
    quality = lambda p: (int(p.left_edge) + int(p.right_edge), -p.score,
                         -(p.end - p.start), p.start, p.end, p.entity_type)
    selected: list[SpanPrediction] = []
    duplicates = conflicts = 0
    for candidate in sorted(spans, key=quality):
        overlap = [s for s in selected if candidate.start < s.end and candidate.end > s.start]
        if overlap:
            if any((s.start, s.end, s.entity_type) == (candidate.start, candidate.end, candidate.entity_type) for s in overlap):
                duplicates += 1
            else:
                conflicts += 1
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda p: (p.start, p.end, p.entity_type)), {
        "raw_candidates": len(candidates), "stitched_fragments": stitched,
        "duplicates_removed": duplicates, "overlap_conflicts_removed": conflicts,
        "remaining_edge_candidates": sum(p.left_edge or p.right_edge for p in selected),
    }


def decode_windows(windows: Sequence[TokenWindow], predictions: Sequence[Sequence[int]],
                   id2label: Mapping[int, str],
                   confidences: Sequence[Sequence[float]] | None = None
                   ) -> tuple[list[SpanPrediction], dict[str, int]]:
    """Decode window BIO predictions into deterministic global character spans."""
    if len(windows) != len(predictions) or (confidences is not None and len(confidences) != len(windows)):
        raise ValueError("One prediction/confidence row is required for every window")
    candidates: list[SpanPrediction] = []
    for wi, (window, row) in enumerate(zip(windows, predictions)):
        if len(row) != len(window.input_ids):
            raise ValueError("Prediction lengths must match unpadded window lengths")
        scores = confidences[wi] if confidences is not None else [1.0] * len(row)
        if len(scores) != len(row):
            raise ValueError("Confidence lengths must match prediction lengths")
        content = window.content_indices
        current: list[int] = []
        kind: str | None = None

        def flush() -> None:
            nonlocal current, kind
            if current and kind:
                candidates.append(SpanPrediction(
                    window.offsets[current[0]][0], window.offsets[current[-1]][1], kind,
                    sum(float(scores[i]) for i in current) / len(current),
                    wi > 0 and current[0] == content[0],
                    wi < len(windows) - 1 and current[-1] == content[-1],
                ))
            current, kind = [], None

        for i in content:
            label_id = int(row[i])
            if label_id not in id2label:
                raise ValueError(f"Unknown prediction label id {label_id}")
            label = id2label[label_id]
            if label == "O":
                flush()
                continue
            if not (label.startswith("B-") or label.startswith("I-")) or len(label) < 3:
                raise ValueError(f"Invalid BIO label {label!r}")
            prefix, next_kind = label[:1], label[2:]
            if prefix == "B" or kind != next_kind:
                flush()
                kind = next_kind
            current.append(i)
        flush()
    return resolve_spans(candidates)
