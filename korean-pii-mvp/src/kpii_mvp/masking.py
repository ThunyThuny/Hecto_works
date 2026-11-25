"""Deterministic masking using half-open offsets into the original Python string.

Offsets count Unicode code points, not UTF-8 bytes or UTF-16 code units. No
normalization or substring replacement is performed.
"""

from collections.abc import Iterable, Mapping
from typing import Any


def _interval_pair(interval: Any) -> tuple[int, int]:
    if isinstance(interval, Mapping):
        if "start" not in interval or "end" not in interval:
            raise ValueError("Each interval must contain start and end")
        start, end = interval["start"], interval["end"]
    elif isinstance(interval, (tuple, list)) and len(interval) == 2:
        start, end = interval
    elif hasattr(interval, "start") and hasattr(interval, "end"):
        start, end = interval.start, interval.end
    else:
        raise ValueError("An interval must be a start/end pair or object")
    if type(start) is not int or type(end) is not int:
        raise ValueError("Interval offsets must be integers, not booleans")
    return start, end


def merge_intervals(text: str, intervals: Iterable[Any]) -> list[tuple[int, int]]:
    """Validate and union overlapping or touching original-coordinate spans."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    pairs = []
    for interval in intervals:
        start, end = _interval_pair(interval)
        if not 0 <= start < end <= len(text):
            raise ValueError(
                f"Invalid half-open interval ({start}, {end}) for text length {len(text)}"
            )
        pairs.append((start, end))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(pairs):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def render_masked_text(
    text: str, intervals: Iterable[Any], placeholder: str = "[MASK]"
) -> str:
    """Render one fixed placeholder per union span, preserving all other text."""
    if not isinstance(placeholder, str) or not placeholder:
        raise ValueError("placeholder must be a nonempty string")
    merged = merge_intervals(text, intervals)
    parts: list[str] = []
    cursor = 0
    for start, end in merged:
        parts.extend((text[cursor:start], placeholder))
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def intervals_from_mentions(text: str, mentions: Iterable[Any]) -> list[tuple[int, int]]:
    """Return final MASK spans; undecided candidates are an explicit error."""
    intervals = []
    for mention in mentions:
        if isinstance(mention, Mapping):
            action = mention.get("action")
            start, end = _interval_pair(mention)
            mention_text = mention.get("text")
        else:
            action = getattr(mention, "action", None)
            start, end = _interval_pair(mention)
            mention_text = getattr(mention, "text", None)
        merge_intervals(text, [(start, end)])
        if mention_text is not None and mention_text != text[start:end]:
            raise ValueError("Mention text does not match its original-coordinate span")
        if action not in ("MASK", "KEEP"):
            raise ValueError("Every candidate needs an explicit MASK or KEEP decision")
        if action == "MASK":
            intervals.append((start, end))
    return merge_intervals(text, intervals)
