"""UTF-8 JSONL persistence for the validated project schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import Document


def read_jsonl(path: str | Path) -> list[Document]:
    documents: list[Document] = []
    ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                document = Document.from_dict(json.loads(line))
                if document.doc_id in ids:
                    raise ValueError("duplicate doc_id in JSONL file")
                ids.add(document.doc_id)
                documents.append(document)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid document on JSONL line {line_number}: {exc}") from exc
    return documents


def write_jsonl(path: str | Path, docs: Iterable[Document]) -> None:
    # Validate the complete collection before touching the destination.
    documents = list(docs)
    ids: set[str] = set()
    for document in documents:
        if not isinstance(document, Document):
            raise ValueError("write_jsonl expects Document instances")
        document.validate()
        if document.doc_id in ids:
            raise ValueError("duplicate doc_id in JSONL collection")
        ids.add(document.doc_id)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for document in documents:
                handle.write(json.dumps(document.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
