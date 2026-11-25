"""Inspect and explicitly adapt local KDPII exports without guessing their format.

``kdpii-v1`` is an explicit preset for the original source files verified by
the integrating agent: root array of sentence records with ``sent_idx``,
``sentence``, ``PII_set`` (id/form/label/begin/end), ``sent_seq``, and
``labelling_seq``. Native spans use Unicode [begin, end). It verifies the spans
against both the original text and the character BIO sequence. It groups on
the part of sent_idx before the first underscore and, by default, aggregates
sentences in their original file order using a newline separator. Its only
adapter-config option is ``aggregate_conversations`` (boolean, default true).
The 33 native types are retained; no privacy action is inferred. An invalid
record causes an error by default. Explicit ``invalid_policy='quarantine'``
excludes its whole conversation and reports value-free provenance and reasons.

The generic adapters do not infer any native layout. They accept only:
* ``spans``: records with exact Unicode-code-point offsets and surface strings;
* ``bio``: BIO-tagged tokens with exact offsets, or tokens whose concatenation
  exactly equals the original text (including whitespace).

Common adapter-config keys (a JSON object, supplied inline or by file):
``records_key`` is an optional dotted path to an array inside a JSON object;
``single_record: true`` explicitly permits a single-record JSON object;
``id_key``/``text_key`` default to ``doc_id``/``text``;
``group_key`` optionally names a required source-group field. Without it,
``source_group`` is used if present, otherwise exact-text SHA-256 is used.
``annotation_scope`` must explicitly list the fully annotated native types, or
``annotation_scope_key`` (default ``annotation_scope``) must identify that list
in every record. Missing annotation lists are errors, never negative examples.
``allowed_labels`` optionally restricts the declared native label inventory.
``split_key`` defaults to ``split``; if present it must equal the input split.

Span-specific keys: ``mentions_key`` (mentions), ``mention_id_key`` (mention_id),
``start_key`` (start), ``end_key`` (end), ``mention_text_key`` (text), ``label_key``
(entity_type). Missing mention IDs are generated deterministically within the
record. ``end_inclusive`` can explicitly convert inclusive ends; otherwise ends
are exclusive. ``offset_unit`` only accepts ``unicode_codepoint``.

BIO-specific keys: ``tokens_key`` (tokens), ``tags_key`` (bio_tags),
``offsets_key`` (offsets). Offsets are a list of [start, end] pairs. If the offsets
field is absent, exact token concatenation is required. No searching, token
normalization, whitespace insertion, or BIO repair is performed.

An optional label-map JSON explicitly maps every declared native type to a
nonempty output type. Many-to-one type merges are permitted only through this
explicit map; unknown or omitted source labels and duplicate JSON keys fail.
All converted actions remain null; the adapter does not fabricate MASK/KEEP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .data import write_jsonl
from .schema import Document, Mention


_MISSING = object()
_COMMON_KEYS = {
    "records_key", "single_record", "id_key", "text_key", "group_key",
    "annotation_scope", "annotation_scope_key", "allowed_labels", "split_key",
    "offset_unit",
}
_SPAN_KEYS = {
    "mentions_key", "mention_id_key", "start_key", "end_key", "mention_text_key",
    "label_key", "end_inclusive",
}
_BIO_KEYS = {"tokens_key", "tags_key", "offsets_key"}
KDPII_V1_LABELS = (
    "CV_MILITARY_CAMP", "CV_POSITION", "CV_SEX", "DT_BIRTH", "FD_MAJOR",
    "LCP_COUNTRY", "LC_ADDRESS", "LC_PLACE", "OGG_CLUB", "OGG_EDUCATION",
    "OGG_RELIGION", "OG_DEPARTMENT", "OG_WORKPLACE", "PS_ID", "PS_NAME",
    "PS_NICKNAME", "QT_ACCOUNT_NUMBER", "QT_AGE", "QT_ALIEN_NUMBER",
    "QT_CARD_NUMBER", "QT_DRIVER_NUMBER", "QT_GRADE", "QT_IP", "QT_LENGTH",
    "QT_MOBILE", "QT_PASSPORT_NUMBER", "QT_PHONE", "QT_PLATE_NUMBER",
    "QT_RESIDENT_NUMBER", "QT_WEIGHT", "TMI_EMAIL", "TMI_SITE", "TM_BLOOD_TYPE",
)
_SAFE_SCHEMA_KEYS = {
    "data", "documents", "document", "records", "dialogue", "dialogues", "utterance",
    "utterances", "sentence", "sentences", "sentence_form", "sentence_id", "id",
    "doc_id", "document_id", "dialog_id", "dialog_type", "text", "form", "PNE",
    "pne", "NE", "entities", "entity", "mentions", "mention_id", "start", "end",
    "begin", "offset", "offsets", "label", "labels", "entity_type", "type", "tokens",
    "bio_tags", "tags", "source_group", "annotation_scope", "split", "metadata",
    "sent_idx", "PII_set", "sent_seq", "labelling_seq",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _path(record: Any, key: str, default: Any = _MISSING) -> Any:
    if not isinstance(key, str) or not key:
        raise ValueError("configured field paths must be nonempty strings")
    value = record
    for component in key.split("."):
        if not isinstance(value, dict) or component not in value:
            if default is not _MISSING:
                return default
            raise ValueError(f"required configured field is missing: {key}")
        value = value[component]
    return value


def _labels(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list of native type names")
    if any(not isinstance(item, str) or not item or item != item.strip() for item in value):
        raise ValueError(f"{name} contains an invalid type name")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} contains duplicate type names")
    return list(value)


def _config(config: Mapping[str, Any] | None, adapter: str) -> dict[str, Any]:
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("adapter-config must be a JSON object")
    if adapter == "kdpii-v1":
        if set(config) - {"aggregate_conversations"}:
            raise ValueError("kdpii-v1 only accepts aggregate_conversations; native fields are fixed")
        if "aggregate_conversations" in config and type(config["aggregate_conversations"]) is not bool:
            raise ValueError("aggregate_conversations must be a boolean")
        return dict(config)
    permitted = _COMMON_KEYS | (_SPAN_KEYS if adapter == "spans" else _BIO_KEYS)
    if set(config) - permitted:
        raise ValueError("adapter-config contains unsupported keys")
    result = dict(config)
    if result.get("offset_unit", "unicode_codepoint") != "unicode_codepoint":
        raise ValueError("only Unicode-code-point offsets are accepted")
    for name in ("single_record", "end_inclusive"):
        if name in result and type(result[name]) is not bool:
            raise ValueError(f"{name} must be a boolean")
    if "annotation_scope" in result:
        _labels(result["annotation_scope"], "annotation_scope")
    if "allowed_labels" in result:
        _labels(result["allowed_labels"], "allowed_labels")
    return result


def _records(path: str | Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = Path(path)
    raw = source.read_text(encoding="utf-8-sig")
    if source.suffix.lower() == ".jsonl":
        records = []
        for line_number, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"invalid source JSONL on line {line_number}") from exc
    else:
        try:
            records = json.loads(raw)
        except ValueError as exc:
            raise ValueError("source is not valid JSON; use .jsonl for JSON Lines") from exc
        if "records_key" in config:
            records = _path(records, config["records_key"])
        elif isinstance(records, dict) and config.get("single_record") is True:
            records = [records]
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ValueError("source must be a list of records; set records_key or single_record explicitly")
    return records


def _scope(record: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    record_scope = _path(record, config.get("annotation_scope_key", "annotation_scope"), None)
    declared = config.get("annotation_scope")
    if record_scope is None and declared is None:
        raise ValueError("annotation_scope is required; missing annotations cannot become negatives")
    scope = _labels(record_scope if record_scope is not None else declared, "annotation_scope")
    if declared is not None and set(scope) != set(declared):
        raise ValueError("record annotation_scope disagrees with adapter-config")
    if "allowed_labels" in config and not set(scope) <= set(config["allowed_labels"]):
        raise ValueError("annotation_scope includes an unknown native type")
    return scope


def _base(record: Mapping[str, Any], config: Mapping[str, Any], split: str | None) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("a source record must be a JSON object")
    text = _path(record, config.get("text_key", "text"))
    if not isinstance(text, str) or not text:
        raise ValueError("source text must be a nonempty string")
    doc_id = _path(record, config.get("id_key", "doc_id"))
    if not isinstance(doc_id, str) or not doc_id or doc_id != doc_id.strip():
        raise ValueError("source document ID must be a nonempty string")
    declared_split = _path(record, config.get("split_key", "split"), None)
    if declared_split is not None and split is not None and declared_split != split:
        raise ValueError("record split disagrees with its input split; implicit re-splitting is forbidden")
    if "group_key" in config:
        group = _path(record, config["group_key"])
    else:
        group = record.get("source_group", "text-sha256:" + _sha256_bytes(text.encode("utf-8")))
    return {
        "doc_id": doc_id, "text": text, "source_group": group,
        "annotation_scope": _scope(record, config),
        "split": split if split is not None else declared_split,
    }


def parse_span_record(
    record: Mapping[str, Any], *, config: Mapping[str, Any] | None = None,
    split: str | None = None, source: str = "KDPII",
) -> Document:
    """Convert a configured exact-span record; see module docstring for keys."""
    settings = _config(config, "spans")
    base = _base(record, settings, split)
    annotations = _path(record, settings.get("mentions_key", "mentions"))
    if not isinstance(annotations, list):
        raise ValueError("source mentions must be an explicit list")
    mentions = []
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            raise ValueError("a source mention must be a JSON object")
        start = _path(annotation, settings.get("start_key", "start"))
        end = _path(annotation, settings.get("end_key", "end"))
        if type(start) is not int or type(end) is not int:
            raise ValueError("source span offsets must be integers")
        if settings.get("end_inclusive", False):
            end += 1
        mention = Mention(
            mention_id=_path(annotation, settings.get("mention_id_key", "mention_id"), f"m{index + 1}"),
            start=start, end=end,
            text=_path(annotation, settings.get("mention_text_key", "text")),
            entity_type=_path(annotation, settings.get("label_key", "entity_type")),
            action=None,
        )
        mentions.append(mention)
    return Document(**base, mentions=mentions, source=source)


def parse_bio_record(
    record: Mapping[str, Any], *, config: Mapping[str, Any] | None = None,
    split: str | None = None, source: str = "KDPII",
) -> Document:
    """Convert strict BIO with exact reconstruction; invalid I tags are errors."""
    settings = _config(config, "bio")
    base = _base(record, settings, split)
    text = base["text"]
    tokens = _path(record, settings.get("tokens_key", "tokens"))
    tags = _path(record, settings.get("tags_key", "bio_tags"))
    if (not isinstance(tokens, list) or not isinstance(tags, list) or not tokens
            or len(tokens) != len(tags) or any(not isinstance(t, str) or not t for t in tokens)):
        raise ValueError("tokens and BIO tags must be nonempty equally sized lists")
    offsets = _path(record, settings.get("offsets_key", "offsets"), None)
    if offsets is None:
        if "".join(tokens) != text:
            raise ValueError("tokens do not exactly reconstruct text; supply exact offsets")
        offsets = []
        cursor = 0
        for token in tokens:
            offsets.append([cursor, cursor + len(token)])
            cursor += len(token)
    if not isinstance(offsets, list) or len(offsets) != len(tokens):
        raise ValueError("offsets must contain one [start, end] pair per token")
    previous = 0
    for token, pair in zip(tokens, offsets):
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(type(value) is not int for value in pair)):
            raise ValueError("each token offset must be a pair of integers")
        start, end = pair
        if not 0 <= previous <= start < end <= len(text):
            raise ValueError("token offsets are invalid, unordered, or overlapping")
        if text[start:end] != token:
            raise ValueError("token surface differs from its original text slice")
        if text[previous:start].strip():
            raise ValueError("token offsets omit non-whitespace original text")
        previous = end
    if text[previous:].strip():
        raise ValueError("token offsets omit non-whitespace original text")
    mentions: list[Mention] = []
    active: tuple[str, int, int] | None = None

    def flush() -> None:
        nonlocal active
        if active is not None:
            label, start, end = active
            mentions.append(Mention(f"m{len(mentions) + 1}", start, end, text[start:end], label))
            active = None

    for tag, (start, end) in zip(tags, offsets):
        if tag == "O":
            flush()
            continue
        if not isinstance(tag, str) or len(tag) < 3 or tag[:2] not in ("B-", "I-"):
            raise ValueError("BIO tags must be O, B-type, or I-type strings")
        label = tag[2:]
        if not label or label != label.strip():
            raise ValueError("BIO tag contains an invalid native type")
        if tag.startswith("B-"):
            flush()
            active = (label, start, end)
        else:
            if active is None or active[0] != label:
                raise ValueError("invalid BIO transition; I-type needs a preceding matching B/I-type")
            active = (label, active[1], end)
    flush()
    return Document(**base, mentions=mentions, source=source)


def _native_group(record: Mapping[str, Any]) -> str:
    value = _path(record, "sent_idx")
    if not isinstance(value, str) or not value or value != value.strip() or "_" not in value:
        raise ValueError("native sent_idx must contain a conversation ID and underscore suffix")
    group = value.split("_", 1)[0]
    if not group:
        raise ValueError("native sent_idx has an empty conversation ID")
    return group


def parse_kdpii_v1_record(record: Mapping[str, Any], *, split: str | None = None) -> Document:
    """Validate one verified native sentence using spans AND character BIO."""
    group = _native_group(record)
    annotations = _path(record, "PII_set")
    if not isinstance(annotations, list):
        raise ValueError("native PII_set must be an explicit list")
    mentions = []
    seen_native_ids: set[int] = set()
    for index, item in enumerate(annotations, 1):
        if not isinstance(item, dict):
            raise ValueError("native PII_set must contain objects")
        native_id = _path(item, "id")
        if type(native_id) is not int or native_id in seen_native_ids:
            raise ValueError("native mention IDs must be unique integers within a sentence")
        seen_native_ids.add(native_id)
        mentions.append({"mention_id": f"m{index}", "start": _path(item, "begin"),
            "end": _path(item, "end"), "text": _path(item, "form"), "entity_type": _path(item, "label")})
    canonical = {
        "doc_id": _path(record, "sent_idx"), "text": _path(record, "sentence"),
        "source_group": group, "annotation_scope": list(KDPII_V1_LABELS), "mentions": mentions,
    }
    span_document = parse_span_record(canonical, split=split, source="KDPII-v1")
    characters = _path(record, "sent_seq")
    if not isinstance(characters, list) or any(not isinstance(c, str) or len(c) != 1 for c in characters):
        raise ValueError("native sent_seq must contain exact Unicode characters")
    bio_document = parse_bio_record({**canonical, "tokens": characters,
        "bio_tags": _path(record, "labelling_seq")}, split=split, source="KDPII-v1")
    signature = lambda doc: sorted((m.start, m.end, m.text, m.entity_type) for m in doc.mentions)
    if signature(span_document) != signature(bio_document):
        raise ValueError("native PII_set and character BIO annotations disagree")
    return span_document


def aggregate_conversations(documents: list[Document]) -> list[Document]:
    """Join a conversation's sentences in input order, shifting exact offsets."""
    groups: dict[str, list[Document]] = {}
    for document in documents:
        groups.setdefault(document.source_group, []).append(document)
    result = []
    for group, sentences in groups.items():
        if len({s.doc_id for s in sentences}) != len(sentences):
            raise ValueError("duplicate native sentence ID within a conversation")
        if len({s.split for s in sentences}) != 1:
            raise ValueError("cannot aggregate a conversation spanning multiple splits")
        if any(s.annotation_scope != sentences[0].annotation_scope for s in sentences):
            raise ValueError("conversation sentences have inconsistent annotation_scope")
        text = "\n".join(sentence.text for sentence in sentences)
        mentions = []
        shift = 0
        for sentence_index, sentence in enumerate(sentences, 1):
            mentions.extend(replace(m, mention_id=f"s{sentence_index}_{m.mention_id}",
                start=m.start + shift, end=m.end + shift) for m in sentence.mentions)
            shift += len(sentence.text) + 1
        result.append(Document(group, text, mentions, group, list(sentences[0].annotation_scope),
            source="KDPII-v1", split=sentences[0].split))
    return result


def _quarantine_reason(error: Exception) -> str:
    message = str(error)
    if "overlapping annotated" in message:
        return "overlapping_mentions"
    if "BIO" in message:
        return "bio_annotation_error"
    if "slice" in message or "length" in message or "start < end" in message:
        return "invalid_span"
    if "annotation_scope" in message:
        return "unknown_native_label"
    return "invalid_source_schema"


def _native_documents(records: list[dict[str, Any]], split: str, settings: Mapping[str, Any],
                      invalid_policy: str) -> tuple[list[Document], dict[str, Any]]:
    documents = []
    affected_groups: set[str] = set()
    problems = []
    all_groups = [_native_group(record) for record in records]
    raw_ids = [_path(record, "sent_idx") for record in records]
    if len(raw_ids) != len(set(raw_ids)):
        raise ValueError("duplicate native sentence IDs; explicit source repair is required")
    for index, record in enumerate(records, 1):
        try:
            documents.append(parse_kdpii_v1_record(record, split=split))
        except (TypeError, ValueError) as exc:
            if invalid_policy == "error":
                raise ValueError(f"invalid {split} source record {index}: {exc}") from exc
            group = all_groups[index - 1]
            affected_groups.add(group)
            problems.append({"record_index": index, "group_sha256": _sha256_bytes(group.encode("utf-8")),
                "reason_code": _quarantine_reason(exc)})
    kept = [doc for doc in documents if doc.source_group not in affected_groups]
    if settings.get("aggregate_conversations", True):
        kept = aggregate_conversations(kept)
    summary = {
        "input_sentence_count": len(records), "input_conversation_count": len(set(all_groups)),
        "invalid_sentence_count": len(problems), "quarantined_conversation_count": len(affected_groups),
        "quarantined_sentence_count": sum(group in affected_groups for group in all_groups),
        "reason_counts": dict(Counter(item["reason_code"] for item in problems)), "records": problems,
    }
    return kept, summary


def _safe_shape(value: Any, depth: int = 0) -> Any:
    if depth >= 4:
        return type(value).__name__
    if isinstance(value, dict):
        known = {key: _safe_shape(item, depth + 1) for key, item in value.items() if key in _SAFE_SCHEMA_KEYS}
        unknown_count = sum(key not in _SAFE_SCHEMA_KEYS for key in value)
        return {"kind": "object", "known_fields": known, "other_key_count": unknown_count}
    if isinstance(value, list):
        return {"kind": "array", "length": len(value), "first_item_shape": _safe_shape(value[0], depth + 1) if value else None}
    return "null" if value is None else type(value).__name__


def inspect_file(path: str | Path, adapter_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return counts, hashes and value-free shapes; never return raw text/IDs."""
    settings = dict(adapter_config or {})
    file_path = Path(path)
    content = file_path.read_bytes()
    decoded = content.decode("utf-8-sig")
    if file_path.suffix.lower() == ".jsonl":
        items = [json.loads(line) for line in decoded.splitlines() if line.strip()]
        root = items
    else:
        root = json.loads(decoded)
    selected = _path(root, settings["records_key"]) if "records_key" in settings else root
    return {
        "sha256": _sha256_bytes(content), "size_bytes": len(content),
        "root_kind": type(root).__name__,
        "record_count": len(selected) if isinstance(selected, list) else (1 if settings.get("single_record") is True else None),
        "shape": _safe_shape(selected),
        "notice": "Values and unrecognized key names are intentionally omitted; no native KDPII adapter is inferred.",
    }


def check_split_integrity(splits: Mapping[str, list[Document]]) -> dict[str, int]:
    """Reject duplicate documents and cross-split text/group leakage."""
    seen_ids: set[str] = set()
    seen_text: dict[str, str] = {}
    seen_group: dict[str, str] = {}
    for split, documents in splits.items():
        if not documents:
            raise ValueError("each provided official split must contain at least one document")
        for document in documents:
            document.validate()
            if document.split != split:
                raise ValueError("document split does not match its split collection")
            if document.doc_id in seen_ids:
                raise ValueError("duplicate document ID across source records")
            seen_ids.add(document.doc_id)
            digest = _sha256_bytes(document.text.encode("utf-8"))
            if digest in seen_text:
                raise ValueError("duplicate exact text within or across official splits")
            seen_text[digest] = split
            if document.source_group in seen_group and seen_group[document.source_group] != split:
                raise ValueError("source_group leakage across official splits")
            seen_group[document.source_group] = split
    return {"unique_documents": len(seen_ids), "unique_texts": len(seen_text), "unique_groups": len(seen_group)}


def _mapped(splits: Mapping[str, list[Document]], label_map: Mapping[str, str] | None) -> dict[str, list[Document]]:
    if label_map is None:
        return {name: list(docs) for name, docs in splits.items()}
    if not isinstance(label_map, dict) or not label_map:
        raise ValueError("label-map must be a nonempty JSON object")
    inventory = {label for docs in splits.values() for doc in docs for label in doc.annotation_scope}
    if set(label_map) != inventory:
        raise ValueError("label-map must cover exactly the declared native type inventory; unknown/omitted labels are rejected")
    values = list(label_map.values())
    if any(not isinstance(item, str) or not item or item != item.strip() for item in values):
        raise ValueError("label-map targets must be nonempty type names")
    return {
        name: [replace(doc,
            mentions=[replace(m, entity_type=label_map[m.entity_type]) for m in doc.mentions],
            annotation_scope=list(dict.fromkeys(label_map[label] for label in doc.annotation_scope)),
        ) for doc in docs]
        for name, docs in splits.items()
    }


def prepare_dataset(
    train: str | Path, valid: str | Path, test: str | Path,
    output_dir: str | Path, *, adapter: str = "spans",
    adapter_config: Mapping[str, Any] | None = None,
    label_map: Mapping[str, str] | None = None,
    invalid_policy: str = "error",
) -> dict[str, Any]:
    """Validate all inputs, preserve splits, then write JSONL and a manifest.

    The output directory may exist, but existing train/valid/test JSONL files or
    manifest.json are never overwritten. Exact duplicate text is rejected even
    within one split; no implicit deduplication or redistribution takes place.
    """
    if adapter not in ("spans", "bio", "kdpii-v1"):
        raise ValueError("adapter must be spans, bio, or kdpii-v1")
    if invalid_policy not in ("error", "quarantine"):
        raise ValueError("invalid_policy must be error or quarantine")
    if invalid_policy == "quarantine" and adapter != "kdpii-v1":
        raise ValueError("quarantine is supported only by the verified kdpii-v1 conversation preset")
    settings = _config(adapter_config, adapter)
    parser = parse_span_record if adapter == "spans" else parse_bio_record
    paths = {"train": Path(train), "valid": Path(valid), "test": Path(test)}
    splits = {}
    inputs = {}
    quarantine = {}
    native_group_splits: dict[str, str] = {}
    for split, path in paths.items():
        records = _records(path, settings)
        if adapter == "kdpii-v1":
            for record in records:
                group = _native_group(record)
                if group in native_group_splits and native_group_splits[group] != split:
                    raise ValueError("source_group leakage in native input splits before quarantine")
                native_group_splits[group] = split
            documents, quarantine[split] = _native_documents(records, split, settings, invalid_policy)
        else:
            documents = []
            for index, record in enumerate(records, 1):
                try:
                    documents.append(parser(record, config=settings, split=split, source="KDPII"))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid {split} source record {index}: {exc}") from exc
        splits[split] = documents
        inputs[split] = {"file_name": path.name, "sha256": _sha256_bytes(path.read_bytes())}
    integrity = check_split_integrity(splits)
    native_labels = sorted({m.entity_type for docs in splits.values() for doc in docs for m in doc.mentions})
    native_scope = sorted({label for docs in splits.values() for doc in docs for label in doc.annotation_scope})
    converted = _mapped(splits, label_map)
    destination = Path(output_dir)
    output_paths = {split: destination / f"{split}.jsonl" for split in paths}
    if any(path.exists() for path in [*output_paths.values(), destination / "manifest.json"]):
        raise FileExistsError("prepared outputs already exist; choose a new output directory")
    manifest = {
        "schema_version": "1.0", "source": "KDPII", "adapter": adapter,
        "adapter_config": settings, "input_files": inputs,
        "document_counts": {split: len(docs) for split, docs in converted.items()},
        "mention_counts": {split: sum(len(doc.mentions) for doc in docs) for split, docs in converted.items()},
        "native_label_inventory": native_labels, "native_annotation_scope": native_scope,
        "output_label_inventory": sorted({m.entity_type for docs in converted.values() for doc in docs for m in doc.mentions}),
        "label_map": dict(label_map) if label_map is not None else None,
        "invalid_policy": invalid_policy, "quarantine": quarantine,
        "decision_labels": {"null": sum(len(doc.mentions) for docs in converted.values() for doc in docs)},
        "split_policy": "preserved input train/valid/test; no inferred KDPII-19 selection",
        "duplicate_policy": "reject exact duplicate text and duplicate IDs; reject source_group overlap across splits",
        "integrity": integrity,
        "group_strategy": ("sent_idx prefix before first underscore; aggregate sentences in input order with newline"
            if adapter == "kdpii-v1" and settings.get("aggregate_conversations", True)
            else "sent_idx prefix before first underscore" if adapter == "kdpii-v1"
            else "configured group_key" if "group_key" in settings
            else "source_group if supplied, otherwise exact-text SHA-256"),
        "limitations": ["Exact-text checks do not establish persona/template independence; supply trusted source-group IDs.",
                        "annotation_scope completeness is explicitly declared by the source/config, not inferred from unlabelled text."],
        "outputs": {},
    }
    destination.mkdir(parents=True, exist_ok=True)
    for split, path in output_paths.items():
        write_jsonl(path, converted[split])
        manifest["outputs"][split] = {"file_name": path.name, "sha256": _sha256_bytes(path.read_bytes())}
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def _json_argument(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    text = value if value.lstrip().startswith("{") else Path(value).read_text(encoding="utf-8-sig")
    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("JSON configuration contains duplicate keys")
            result[key] = item
        return result
    parsed = json.loads(text, object_pairs_hook=unique_keys)
    if not isinstance(parsed, dict):
        raise ValueError("JSON argument must contain an object")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_command = commands.add_parser("inspect", help="print a value-free local schema summary")
    inspect_command.add_argument("path")
    inspect_command.add_argument("--adapter-config", help="JSON file or inline object")
    prepare = commands.add_parser("prepare", help="adapt explicitly configured records, preserving official splits")
    for split in ("train", "valid", "test"):
        prepare.add_argument(f"--{split}", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--adapter", choices=("spans", "bio", "kdpii-v1"), default="spans")
    prepare.add_argument("--adapter-config", help="JSON file or inline object; declares complete native annotation_scope")
    prepare.add_argument("--label-map", help="optional explicit JSON label-map file or inline object")
    prepare.add_argument("--invalid-policy", choices=("error", "quarantine"), default="error")
    args = parser.parse_args(argv)
    try:
        config = _json_argument(args.adapter_config)
        if args.command == "inspect":
            result = inspect_file(args.path, config)
        else:
            result = prepare_dataset(args.train, args.valid, args.test, args.output_dir,
                adapter=args.adapter, adapter_config=config, label_map=_json_argument(args.label_map),
                invalid_policy=args.invalid_policy)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
