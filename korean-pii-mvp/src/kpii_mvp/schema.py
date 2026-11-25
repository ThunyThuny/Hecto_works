"""Validated, Unicode-code-point span records used throughout the MVP.

This is a project schema, not a claim about the native KDPII file format.
An absent action is an unlabelled decision; it never means KEEP.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from typing import Any, Mapping


def _nonempty_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty string without outer whitespace")


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return value


def _check_fields(record: Mapping[str, Any], cls: type, required: set[str]) -> None:
    _object(record, cls.__name__)
    if required - record.keys():
        missing = ", ".join(sorted(required - record.keys()))
        raise ValueError(f"{cls.__name__} is missing required fields: {missing}")
    if record.keys() - {item.name for item in fields(cls)}:
        raise ValueError(f"{cls.__name__} contains unsupported fields")


@dataclass(frozen=True)
class Mention:
    mention_id: str
    start: int
    end: int
    text: str
    entity_type: str
    action: str | None = None
    reason_code: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self, document_text: str | None = None) -> "Mention":
        _nonempty_string(self.mention_id, "mention_id")
        _nonempty_string(self.entity_type, "entity_type")
        if type(self.start) is not int or type(self.end) is not int:
            raise ValueError("span offsets must be integers, not booleans")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("a mention requires 0 <= start < end")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("mention text must be a nonempty string")
        if len(self.text) != self.end - self.start:
            raise ValueError("mention length disagrees with Unicode [start, end) offsets")
        if self.action not in (None, "MASK", "KEEP"):
            raise ValueError("action must be MASK, KEEP, or null")
        if self.reason_code is not None:
            _nonempty_string(self.reason_code, "reason_code")
        if document_text is not None:
            if not isinstance(document_text, str):
                raise ValueError("document text must be a string")
            if self.end > len(document_text) or document_text[self.start:self.end] != self.text:
                raise ValueError("mention text does not equal the original document slice")
        return self

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Mention":
        _check_fields(record, cls, {"mention_id", "start", "end", "text", "entity_type"})
        return cls(**dict(record))

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    mentions: list[Mention]
    source_group: str
    annotation_scope: list[str]
    source: str = "unknown"
    split: str | None = None
    policy_id: str | None = None
    trusted_context: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        # Do not retain the caller's top-level mutable containers.
        if isinstance(self.mentions, list):
            object.__setattr__(self, "mentions", list(self.mentions))
        if isinstance(self.annotation_scope, list):
            object.__setattr__(self, "annotation_scope", list(self.annotation_scope))
        if isinstance(self.trusted_context, dict):
            object.__setattr__(self, "trusted_context", dict(self.trusted_context))
        self.validate()

    def validate(self) -> "Document":
        _nonempty_string(self.doc_id, "doc_id")
        _nonempty_string(self.source_group, "source_group")
        _nonempty_string(self.source, "source")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("document text must be a nonempty string")
        if self.schema_version != "1.0":
            raise ValueError("unsupported schema_version")
        for name in ("split", "policy_id"):
            value = getattr(self, name)
            if value is not None:
                _nonempty_string(value, name)
        if not isinstance(self.annotation_scope, list) or not self.annotation_scope:
            raise ValueError("annotation_scope must explicitly list fully annotated types")
        for label in self.annotation_scope:
            _nonempty_string(label, "annotation_scope entry")
        if len(self.annotation_scope) != len(set(self.annotation_scope)):
            raise ValueError("annotation_scope contains duplicate types")
        if not isinstance(self.mentions, list):
            raise ValueError("mentions must be an explicit list, including for negative documents")
        ids: set[str] = set()
        for mention in self.mentions:
            if not isinstance(mention, Mention):
                raise ValueError("mentions must contain Mention instances")
            mention.validate(self.text)
            if mention.mention_id in ids:
                raise ValueError("mention_id values must be unique within a document")
            ids.add(mention.mention_id)
            if mention.entity_type not in self.annotation_scope:
                raise ValueError("mention type is outside the declared annotation_scope")
        ordered = sorted(self.mentions, key=lambda item: (item.start, item.end))
        if any(left.end > right.start for left, right in zip(ordered, ordered[1:])):
            raise ValueError("overlapping annotated mentions cannot be represented by BIO")
        _object(self.trusted_context, "trusted_context")
        try:
            json.dumps(self.trusted_context, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("trusted_context must contain finite JSON values") from exc
        return self

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Document":
        _check_fields(record, cls, {"doc_id", "text", "mentions", "source_group", "annotation_scope"})
        data = dict(record)
        if not isinstance(data["mentions"], list):
            raise ValueError("mentions must be an explicit list")
        data["mentions"] = [Mention.from_dict(item) for item in data["mentions"]]
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = {item.name: getattr(self, item.name) for item in fields(self)}
        result["mentions"] = [mention.to_dict() for mention in self.mentions]
        result["annotation_scope"] = list(self.annotation_scope)
        # Round-tripping gives the caller an independent JSON container.
        result["trusted_context"] = json.loads(json.dumps(self.trusted_context, allow_nan=False))
        return result
