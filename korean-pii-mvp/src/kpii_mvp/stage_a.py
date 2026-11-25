"""Stage A: native-label BIO detector with strict original-offset alignment.

Core imports and helper validation work without ML dependencies.  ``train`` and
``predict`` lazily import torch/transformers.  Training never opens a test split,
uses development span F1 for checkpoint selection, and writes provenance and
alignment reports next to the saved checkpoint.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

from .tokenization import IGNORE_INDEX, align_mentions, decode_windows, tokenize_document

DEFAULT_MODEL = "skt/A.X-Encoder-base"


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prediction_checkpoint_artifacts(model_path: Path) -> dict[str, str]:
    """Bind A prediction to the saved weights, config and tokenizer assets.

    Training provenance has a separate hash. Sharded HF weight/index files and
    custom runtime Python files are included; reports/logs are not model state.
    Relative POSIX names make a copied checkpoint's binding path-independent.
    """
    assets = {"config.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "added_tokens.json", "merges.txt",
              "vocab.txt", "vocab.json", "chat_template.jinja"}
    result = {}
    for path in sorted(model_path.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        if (name in assets or name.endswith((".safetensors", ".safetensors.index.json"))
                or (name.startswith("pytorch_model") and name.endswith((".bin", ".index.json")))
                or path.suffix in {".model", ".py"}):
            result[path.relative_to(model_path).as_posix()] = _sha256(path)
    return result


def _load_documents(path: str | Path) -> list[Any]:
    from .data import read_jsonl
    from .schema import Document
    rows = list(read_jsonl(Path(path)))
    documents = [row if isinstance(row, Document) else Document.from_dict(row) for row in rows]
    for document in documents:
        document.validate()
    return documents


def build_label_inventory(train_documents: Sequence[Any]) -> dict[str, int]:
    """Build labels only from training mentions, retaining native entity names."""
    types = sorted({m.entity_type for d in train_documents for m in d.mentions})
    if not types:
        raise ValueError("Training data contains no entity types")
    if any(not isinstance(t, str) or not t for t in types):
        raise ValueError("Entity types must be nonempty strings")
    labels = ["O"] + [label for kind in types for label in (f"B-{kind}", f"I-{kind}")]
    return {label: i for i, label in enumerate(labels)}


def validate_training_documents(train_documents: Sequence[Any], dev_documents: Sequence[Any],
                                label2id: Mapping[str, int]) -> None:
    """Reject unannotated negatives, split contamination, and single-head overlap."""
    if not train_documents or not dev_documents:
        raise ValueError("Both train and dev datasets must be nonempty")
    types = {name[2:] for name in label2id if name.startswith("B-")}
    for role, documents in (("train", train_documents), ("dev", dev_documents)):
        accepted_splits = {"train"} if role == "train" else {"dev", "valid", "validation"}
        seen_ids: set[str] = set()
        for document in documents:
            if document.split not in accepted_splits:
                raise ValueError(f"{role} input contains {document.doc_id!r} with split={document.split!r}; expected one of {sorted(accepted_splits)}")
            if document.doc_id in seen_ids:
                raise ValueError(f"Duplicate document id in {role}: {document.doc_id}")
            seen_ids.add(document.doc_id)
            missing_scope = types - set(document.annotation_scope or [])
            if missing_scope:
                raise ValueError(f"{document.doc_id}: incomplete annotation_scope for {sorted(missing_scope)}; missing labels cannot become O")
            previous_end = -1
            for mention in sorted(document.mentions, key=lambda m: (m.start, m.end)):
                if mention.entity_type not in types:
                    raise ValueError(f"{document.doc_id}: entity type {mention.entity_type!r} was not present in train")
                if mention.start < previous_end:
                    raise ValueError(f"{document.doc_id}: overlapping gold mentions cannot use one BIO head")
                previous_end = mention.end
    train_ids = {d.doc_id for d in train_documents}
    if train_ids & {d.doc_id for d in dev_documents}:
        raise ValueError("Train/dev share document ids")
    train_groups = {d.source_group for d in train_documents if d.source_group}
    shared_groups = train_groups & {d.source_group for d in dev_documents if d.source_group}
    if shared_groups:
        raise ValueError(f"Train/dev source groups overlap: {sorted(shared_groups)[:5]}")
    train_hashes = {hashlib.sha256(d.text.encode("utf-8")).hexdigest() for d in train_documents}
    if train_hashes & {hashlib.sha256(d.text.encode("utf-8")).hexdigest() for d in dev_documents}:
        raise ValueError("Train/dev contain identical source text")


def prepare_features(documents: Sequence[Any], tokenizer: Any, label2id: Mapping[str, int],
                     max_length: int, stride: int, boundary_policy: str = "ignore",
                     require_all_type_supervision: bool = True
                     ) -> tuple[list[dict[str, list[int]]], dict[str, Any]]:
    features: list[dict[str, list[int]]] = []
    id2label = {value: name for name, value in label2id.items()}
    report: dict[str, Any] = {
        "documents": len(documents), "windows": 0, "kept_windows": 0,
        "all_ignored_windows": 0, "complete_mention_window_instances": 0,
        "clipped_mention_window_instances": 0, "unaligned_mention_window_instances": 0,
        "ignored_tokens": 0, "mentions_never_fully_supervised": [], "details": [],
        "positive_tokens_by_type": {name[2:]: 0 for name in label2id if name.startswith("B-")},
    }
    for document in documents:
        represented: set[tuple[int, int, str]] = set()
        windows = tokenize_document(tokenizer, document.text, max_length, stride)
        report["windows"] += len(windows)
        for window in windows:
            labels, alignment = align_mentions(window, document.mentions, label2id, boundary_policy)
            report["complete_mention_window_instances"] += alignment.complete_mentions
            report["clipped_mention_window_instances"] += alignment.clipped_mentions
            report["unaligned_mention_window_instances"] += alignment.unaligned_mentions
            report["ignored_tokens"] += alignment.ignored_tokens
            report["details"].extend({"doc_id": document.doc_id, "window": window.index, **detail}
                                      for detail in alignment.details)
            for mention in document.mentions:
                positions = [i for i in window.content_indices
                             if window.offsets[i][0] < mention.end and window.offsets[i][1] > mention.start]
                if positions and (window.offsets[positions[0]][0], window.offsets[positions[-1]][1]) == (mention.start, mention.end):
                    if all(labels[i] != IGNORE_INDEX for i in positions):
                        represented.add((mention.start, mention.end, mention.entity_type))
            if all(value == IGNORE_INDEX for value in labels):
                report["all_ignored_windows"] += 1
                continue
            features.append({**window.model_inputs(), "labels": labels})
            report["kept_windows"] += 1
            for value in labels:
                if value != IGNORE_INDEX and id2label[value] != "O":
                    report["positive_tokens_by_type"][id2label[value][2:]] += 1
        for mention in document.mentions:
            key = (mention.start, mention.end, mention.entity_type)
            if key not in represented:
                report["mentions_never_fully_supervised"].append({
                    "doc_id": document.doc_id, "start": mention.start, "end": mention.end,
                    "entity_type": mention.entity_type,
                })
    if not features:
        raise ValueError("No supervised token windows remain after conservative alignment")
    zero_types = sorted(kind for kind, count in report["positive_tokens_by_type"].items() if count == 0)
    if require_all_type_supervision and zero_types:
        raise ValueError(f"Entity types have zero positive token supervision after alignment: {zero_types}; use token-aligned annotations or a compatible tokenizer, not O-only training")
    if boundary_policy == "reject" and report["mentions_never_fully_supervised"]:
        raise ValueError("Some gold mentions do not fit any complete token window; use a longer window or explicit --boundary-policy ignore")
    return features, report


def _ml_dependencies() -> tuple[Any, Any]:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError("Stage A training/prediction requires optional torch and transformers dependencies") from exc
    return torch, transformers


def _device(torch: Any, requested: str) -> Any:
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but CUDA is unavailable")
    return torch.device(requested)


def _collator(tokenizer: Any, torch: Any, include_labels: bool = False) -> Any:
    def collate(rows: Sequence[Mapping[str, list[int]]]) -> dict[str, Any]:
        inputs = [{key: value for key, value in row.items() if key != "labels"} for row in rows]
        batch = dict(tokenizer.pad(inputs, padding=True, return_tensors="pt"))
        if include_labels:
            length = int(batch["input_ids"].shape[1])
            labels = []
            for row in rows:
                padding = [IGNORE_INDEX] * (length - len(row["labels"]))
                labels.append(padding + row["labels"] if tokenizer.padding_side == "left" else row["labels"] + padding)
            batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch
    return collate


def _check_max_length(model: Any, max_length: int) -> None:
    capacity = getattr(model.config, "max_position_embeddings", None)
    if isinstance(capacity, int) and capacity > 0 and max_length > capacity:
        raise ValueError(f"max_length={max_length} exceeds model max_position_embeddings={capacity}")


def model_forward_inputs(model: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    """Keep core tensors and filter unsupported tokenizer-specific auxiliaries.

    A.X ships a BertTokenizerFast but its ModernBERT encoder does not accept
    token_type_ids.  Checking the model boundary also protects inference and
    tokenizers that ignore the request not to emit segment IDs.
    """
    parameters = inspect.signature(model.forward).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(batch)
    core = {"input_ids", "attention_mask", "labels"}
    rejected_core = (set(batch) & core) - set(parameters)
    if rejected_core:
        raise ValueError(f"Model forward does not accept required inputs {sorted(rejected_core)}")
    return {key: value for key, value in batch.items() if key in parameters}


def _load_training_model(transformers: Any, model_path: str,
                         label2id: Mapping[str, int], reset_head: bool,
                         trust_remote_code: bool) -> tuple[Any, dict[str, Any]]:
    """Reset a changed classifier explicitly by strictly transferring only its encoder."""
    config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    old_labels = {str(k): int(v) for k, v in (getattr(config, "label2id", None) or {}).items()}
    architectures = list(getattr(config, "architectures", None) or [])
    token_checkpoint = any("TokenClassification" in name for name in architectures)
    local_manifest = Path(model_path) / "training_manifest.json"
    token_checkpoint = token_checkpoint or local_manifest.is_file()
    changed = token_checkpoint and old_labels != dict(label2id)
    if changed and not reset_head:
        raise ValueError("Saved token-classification label inventory differs. Use --reset-head for explicit native-to-MVP head replacement")
    id2label = {value: name for name, value in label2id.items()}
    if changed:
        old_model = transformers.AutoModelForTokenClassification.from_pretrained(
            model_path, trust_remote_code=trust_remote_code)
        new_config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        new_config.num_labels = len(label2id)
        new_config.label2id, new_config.id2label = dict(label2id), id2label
        model = transformers.AutoModelForTokenClassification.from_config(new_config,
                                            trust_remote_code=trust_remote_code)
        # Strict transfer detects architectural mismatches instead of suppressing
        # arbitrary parameter-size failures with ignore_mismatched_sizes.
        model.base_model.load_state_dict(old_model.base_model.state_dict(), strict=True)
        del old_model
    else:
        config.num_labels = len(label2id)
        config.label2id, config.id2label = dict(label2id), id2label
        model = transformers.AutoModelForTokenClassification.from_pretrained(
            model_path, config=config, trust_remote_code=trust_remote_code)
    return model, {
        "source_architectures": architectures, "source_label2id": old_labels,
        "explicit_head_reset": bool(changed and reset_head),
        "encoder_transfer_strict": bool(changed),
    }


def predict_documents(model: Any, tokenizer: Any, documents: Sequence[Any],
                       id2label: Mapping[int, str], max_length: int, stride: int,
                       batch_size: int, device: Any, torch: Any
                       ) -> tuple[list[Any], dict[str, int]]:
    from .schema import Mention
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    collate = _collator(tokenizer, torch)
    output = []
    totals: dict[str, int] = {"documents": len(documents), "windows": 0}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for document in documents:
                windows = tokenize_document(tokenizer, document.text, max_length, stride)
                totals["windows"] += len(windows)
                predictions: list[list[int]] = []
                confidences: list[list[float]] = []
                for start in range(0, len(windows), batch_size):
                    part = windows[start:start + batch_size]
                    batch = collate([window.model_inputs() for window in part])
                    batch = {key: value.to(device) for key, value in batch.items()}
                    logits = model(**model_forward_inputs(model, batch)).logits
                    scores, labels = torch.softmax(logits.float(), dim=-1).max(dim=-1)
                    for i, window in enumerate(part):
                        length = len(window.input_ids)
                        offset = int(labels.shape[1]) - length if tokenizer.padding_side == "left" else 0
                        predictions.append(labels[i, offset:offset + length].cpu().tolist())
                        confidences.append(scores[i, offset:offset + length].cpu().tolist())
                spans, report = decode_windows(windows, predictions, id2label, confidences)
                for key, value in report.items():
                    totals[key] = totals.get(key, 0) + value
                mentions = [Mention(mention_id=f"{document.doc_id}:pred:{i:04d}", start=span.start,
                    end=span.end, text=document.text[span.start:span.end], entity_type=span.entity_type,
                    action=None, reason_code=None) for i, span in enumerate(spans, 1)]
                predicted = replace(document, mentions=mentions)
                predicted.validate()
                output.append(predicted)
    finally:
        model.train(was_training)
    return output, totals


def span_metrics(gold_documents: Sequence[Any], predicted_documents: Sequence[Any]) -> dict[str, Any]:
    """Micro exact-boundary/type entity metrics, independent of MASK/KEEP."""
    gold_by_id = {d.doc_id: d for d in gold_documents}
    predicted_by_id = {d.doc_id: d for d in predicted_documents}
    if len(gold_by_id) != len(gold_documents) or len(predicted_by_id) != len(predicted_documents):
        raise ValueError("Detection metrics require unique document ids")
    if set(gold_by_id) != set(predicted_by_id):
        raise ValueError("Gold/predicted document ids do not match")
    if any(document.text != predicted_by_id[doc_id].text for doc_id, document in gold_by_id.items()):
        raise ValueError("Gold/predicted original document text does not match")
    gold = {(d.doc_id, m.start, m.end, m.entity_type) for d in gold_documents for m in d.mentions}
    predicted = {(d.doc_id, m.start, m.end, m.entity_type) for d in predicted_documents for m in d.mentions}
    correct = len(gold & predicted)
    precision = correct / len(predicted) if predicted else 0.0
    recall = correct / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    per_type = {}
    for kind in sorted({x[3] for x in gold | predicted}):
        g, p = {x for x in gold if x[3] == kind}, {x for x in predicted if x[3] == kind}
        tp = len(g & p)
        per_type[kind] = {"gold": len(g), "predicted": len(p), "correct": tp,
                          "precision": tp / len(p) if p else 0.0,
                          "recall": tp / len(g) if g else 0.0,
                          "f1": 2 * tp / (len(g) + len(p)) if g or p else 0.0}
    return {"precision": precision, "recall": recall, "f1": f1,
            "gold": len(gold), "predicted": len(predicted), "correct": correct, "per_type": per_type}


def _dataset_manifest(path: Path, documents: Sequence[Any]) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "documents": len(documents),
            "mentions": sum(len(d.mentions) for d in documents),
            "doc_ids": [d.doc_id for d in documents],
            "source_groups": sorted({d.source_group for d in documents}),
            "splits": sorted({str(d.split) for d in documents})}


def train(args: argparse.Namespace) -> dict[str, Any]:
    train_path, dev_path, output_dir = Path(args.train), Path(args.dev), Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output-dir must be empty to avoid overwriting a saved model")
    if args.epochs < 1 or args.max_steps < 0 or args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("epochs/batch sizes must be positive and max_steps must be nonnegative")
    if args.learning_rate <= 0 or not math.isfinite(args.learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    if not 0 <= args.weight_decay < 1:
        raise ValueError("weight_decay must be in [0, 1)")
    train_documents, dev_documents = _load_documents(train_path), _load_documents(dev_path)
    label2id = build_label_inventory(train_documents)
    validate_training_documents(train_documents, dev_documents, label2id)
    torch, transformers = _ml_dependencies()
    device = _device(torch, args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path,
        use_fast=True, trust_remote_code=args.trust_remote_code)
    model, head_transfer = _load_training_model(transformers, args.model_name_or_path,
        label2id, args.reset_head, args.trust_remote_code)
    _check_max_length(model, args.max_length)
    features, alignment = prepare_features(train_documents, tokenizer, label2id,
                                          args.max_length, args.stride, args.boundary_policy)
    # Dev alignment is diagnosed but the checkpoint metric still compares to the
    # original exact gold spans, never the surviving/easier subset.
    _, dev_alignment = prepare_features(dev_documents, tokenizer, label2id,
                                        args.max_length, args.stride, args.boundary_policy,
                                        require_all_type_supervision=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    _json_write(output_dir / "labels.json", {"label2id": label2id,
               "id2label": {value: name for name, value in label2id.items()}})
    _json_write(output_dir / "alignment_report.json", {"train": alignment, "dev": dev_alignment})
    if alignment["mentions_never_fully_supervised"]:
        print(f"WARNING: {len(alignment['mentions_never_fully_supervised'])} train mentions are never fully supervised; see alignment_report.json", file=sys.stderr)
    loader = torch.utils.data.DataLoader(features, batch_size=args.batch_size, shuffle=True,
        collate_fn=_collator(tokenizer, torch, include_labels=True),
        generator=torch.Generator().manual_seed(args.seed))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    configuration = {key: value for key, value in vars(args).items() if key not in {"func"}}
    manifest: dict[str, Any] = {
        "schema_version": "1.0", "stage": "A", "model_name_or_path": args.model_name_or_path,
        "label2id": label2id, "configuration": configuration,
        "datasets": {"train": _dataset_manifest(train_path, train_documents),
                     "dev": _dataset_manifest(dev_path, dev_documents)},
        "head_transfer": head_transfer, "checkpoint_selection": "dev exact-span micro F1",
        "test_used": False, "external_reporting": False,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "history": [], "alignment_report": "alignment_report.json",
    }
    best_f1 = -1.0
    step = 0
    started = time.monotonic()
    id2label = {value: name for name, value in label2id.items()}
    for epoch in range(args.epochs):
        model.train()
        loss_sum, batches = 0.0, 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**model_forward_inputs(model, batch)).loss
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Non-finite training loss; inspect labels and model configuration")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            batches += 1
            loss_sum += float(loss.detach().cpu())
            if args.max_steps and step >= args.max_steps:
                break
        predicted, decode_report = predict_documents(model, tokenizer, dev_documents, id2label,
            args.max_length, args.stride, args.eval_batch_size, device, torch)
        metrics = span_metrics(dev_documents, predicted)
        entry = {"epoch": epoch + 1, "step": step, "train_loss": loss_sum / max(batches, 1),
                 "dev": metrics, "dev_decoding": decode_report}
        manifest["history"].append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            manifest["best_epoch"], manifest["best_step"] = epoch + 1, step
            manifest["best_dev_metrics"] = metrics
            best_dir = output_dir / "best"
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            _json_write(best_dir / "labels.json", {"label2id": label2id,
                        "id2label": id2label})
        manifest["completed_steps"] = step
        manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
        _json_write(output_dir / "training_manifest.json", manifest)
        _json_write(output_dir / "best" / "training_manifest.json", manifest)
        if args.max_steps and step >= args.max_steps:
            break
    return manifest


def predict(args: argparse.Namespace) -> dict[str, Any]:
    output_path, input_path = Path(args.output), Path(args.input)
    manifest_path = Path(args.manifest) if args.manifest else output_path.with_suffix(output_path.suffix + ".manifest.json")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Prediction output must differ from its input")
    if manifest_path.resolve() in {output_path.resolve(), input_path.resolve()}:
        raise ValueError("Manifest must not overwrite input or prediction output")
    model_path = Path(args.model_dir)
    training_manifest_path = model_path / "training_manifest.json"
    if not training_manifest_path.is_file():
        raise ValueError("Stage A prediction requires a saved checkpoint with training_manifest.json")
    training_manifest_sha256 = _sha256(training_manifest_path)
    checkpoint_artifacts = _prediction_checkpoint_artifacts(model_path)
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    if training_manifest.get("stage") != "A":
        raise ValueError("Checkpoint manifest is not Stage A")
    saved_config = training_manifest.get("configuration", {})
    max_length = args.max_length if args.max_length is not None else int(saved_config.get("max_length", 512))
    stride = args.stride if args.stride is not None else int(saved_config.get("stride", 128))
    label2id = {str(name): int(value) for name, value in training_manifest["label2id"].items()}
    documents = _load_documents(args.input)
    torch, transformers = _ml_dependencies()
    device = _device(torch, args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path), use_fast=True,
                                        trust_remote_code=args.trust_remote_code)
    model = transformers.AutoModelForTokenClassification.from_pretrained(str(model_path),
                                        trust_remote_code=args.trust_remote_code)
    if dict(model.config.label2id) != label2id:
        raise ValueError("Checkpoint config and saved label manifest disagree")
    _check_max_length(model, max_length)
    model.to(device)
    started = time.monotonic()
    predicted, decoding_report = predict_documents(model, tokenizer, documents,
        {value: name for name, value in label2id.items()}, max_length, stride,
        args.batch_size, device, torch)
    if (_sha256(training_manifest_path) != training_manifest_sha256
            or _prediction_checkpoint_artifacts(model_path) != checkpoint_artifacts):
        raise ValueError("Stage A checkpoint changed during prediction; rerun with a fixed checkpoint")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for document in predicted:
            stream.write(json.dumps(document.to_dict(), ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": "1.0", "stage": "A-prediction", "model_dir": str(model_path.resolve()),
        "training_manifest_sha256": training_manifest_sha256,
        "checkpoint_artifacts": checkpoint_artifacts,
        "input": {"path": str(input_path.resolve()), "sha256": _sha256(input_path)},
        "output": {"path": str(output_path.resolve()), "sha256": _sha256(output_path)},
        "label2id": label2id, "max_length": max_length, "stride": stride,
        "device": args.device, "decoding": decoding_report,
        "elapsed_seconds": round(time.monotonic() - started, 3), "external_reporting": False,
    }
    _json_write(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="Train from explicit train/dev JSONL only")
    train_parser.add_argument("--train", required=True)
    train_parser.add_argument("--dev", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL)
    train_parser.add_argument("--reset-head", action="store_true")
    train_parser.add_argument("--max-length", type=int, default=512)
    train_parser.add_argument("--stride", type=int, default=128)
    train_parser.add_argument("--boundary-policy", choices=["ignore", "reject"], default="ignore")
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--max-steps", type=int, default=0,
                              help="Optional smoke cap; 0 means all epochs")
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--eval-batch-size", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=3e-5)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    train_parser.add_argument("--trust-remote-code", action="store_true")
    train_parser.set_defaults(func=train)
    predict_parser = commands.add_parser("predict", help="Predict all text windows with saved label inventory")
    predict_parser.add_argument("--model-dir", required=True)
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--output", required=True)
    predict_parser.add_argument("--manifest")
    predict_parser.add_argument("--max-length", type=int)
    predict_parser.add_argument("--stride", type=int)
    predict_parser.add_argument("--batch-size", type=int, default=8)
    predict_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    predict_parser.add_argument("--trust-remote-code", action="store_true")
    predict_parser.set_defaults(func=predict)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"stage_a: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
