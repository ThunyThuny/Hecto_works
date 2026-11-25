"""Direct B training and candidate probability scoring with explicit evidence.

No synthetic corpus is generated and no missing action becomes KEEP. Training
requires review attestations by default. A separately opted, hash-bound
synthetic experiment preserves PENDING records and declares its evidence.
Only dev classification loss selects a checkpoint. KEEP thresholds belong to
the separate calibration step.
Torch/Transformers are imported lazily so data tooling works offline.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Sequence

from .b_inputs import (DEFAULT_INPUT_VERSION, INPUT_VERSIONS, FIXED_POLICY_ID, LABEL2ID, MARKERS, POLICY_CONFIG,
                       allowed_to_keep, canonical_type, keep_eligible,
                       input_sha256, input_spec, prepare_candidate_input, policy_sha256)
from .stage_a import (_check_max_length, _device, _load_documents,
                      _ml_dependencies, model_forward_inputs)

DEFAULT_ENCODER = "/workspace/korean-pii-mvp/runs/a-native-gpu5-20261019_020722/best"
MANIFEST_VERSION = "B_DIRECT_V1"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reviewed_data_provenance(documents: Sequence[Any]) -> dict[str, Any]:
    """Record declared sources without certifying that their contents are real.

    Explicit synthetic*/real* tags are required for a known category. Legacy
    dataset names, including KDPII, stay unknown instead of being assumed real.
    Human annotation approval and source authenticity are separate claims.
    """
    counts = Counter(document.source for document in documents)
    category_counts: Counter[str] = Counter()
    for source, count in counts.items():
        lowered = source.casefold()
        category = "unknown"
        for known in ("synthetic", "real", "mixed"):
            if lowered == known or lowered.startswith((known + "_", known + ":", known + "/")):
                category = known
                break
        category_counts[category] += count
    present = set(category_counts)
    if not present or "unknown" in present:
        classification = "unknown"
    elif "mixed" in present or present == {"real", "synthetic"}:
        classification = "mixed"
    else:
        classification = next(iter(present))
    return {
        "classification": classification,
        "source_counts": dict(sorted(counts.items())),
        "declared_category_counts": dict(sorted(category_counts.items())),
        "basis": "Document.source declarations; human review does not certify source authenticity",
    }


def validate_train_dev(train_documents: Sequence[Any], dev_documents: Sequence[Any]) -> None:
    """Reject contamination and unlabeled KDPII before loading ML dependencies."""
    if not train_documents or not dev_documents:
        raise ValueError("Both reviewed train and dev documents are required")
    for role, documents in (("train", train_documents), ("dev", dev_documents)):
        ids = [document.doc_id for document in documents]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate {role} document IDs")
        accepted = {"train"} if role == "train" else {"dev", "valid", "validation"}
        for document in documents:
            document.validate()
            if document.split not in accepted:
                raise ValueError(f"{role}: unexpected split for {document.doc_id}")
            if document.policy_id != FIXED_POLICY_ID:
                raise ValueError(f"{document.doc_id}: explicit policy_id={FIXED_POLICY_ID} required")
            for mention in document.mentions:
                if mention.action not in LABEL2ID:
                    raise ValueError(f"{document.doc_id}/{mention.mention_id}: missing MASK/KEEP label; null is not KEEP")
                if mention.action == "KEEP" and not allowed_to_keep(document, mention):
                    raise ValueError(f"{document.doc_id}/{mention.mention_id}: KEEP conflicts with fixed policy or approved metadata")
    for field, label in ((lambda d: d.doc_id, "document IDs"),
                         (lambda d: d.source_group, "source groups"),
                         (lambda d: hashlib.sha256(d.text.encode("utf-8")).hexdigest(), "identical source text")):
        if {field(document) for document in train_documents} & {field(document) for document in dev_documents}:
            raise ValueError(f"Train/dev overlap in {label}")
    train_classes = {mention.action for document in train_documents for mention in document.mentions
                     if keep_eligible(mention.entity_type)}
    if train_classes != set(LABEL2ID):
        raise ValueError("Reviewed contextual B train candidates must contain both MASK and KEEP")
    dev_classes = {mention.action for document in dev_documents for mention in document.mentions
                   if keep_eligible(mention.entity_type)}
    if dev_classes != set(LABEL2ID):
        raise ValueError("Reviewed contextual B dev candidates must contain both MASK and KEEP")
    train_types = {canonical_type(mention.entity_type) for document in train_documents
                   for mention in document.mentions if keep_eligible(mention.entity_type)}
    dev_types = {canonical_type(mention.entity_type) for document in dev_documents
                 for mention in document.mentions if keep_eligible(mention.entity_type)}
    if not dev_types:
        raise ValueError("Dev has no contextual candidates")
    if dev_types - train_types:
        raise ValueError(f"Dev contextual types absent in train: {sorted(dev_types - train_types)}")
    from .b_data import validate_split_separation
    validate_split_separation({"train": train_documents, "dev": dev_documents})


def load_initial_model(transformers: Any, encoder_path: str,
                       trust_remote_code: bool = False) -> tuple[Any, Any, dict[str, Any]]:
    """Strictly copy the source backbone into a fresh 2-label classifier.

    An A token head is never loaded into B and A files are never modified.
    Embeddings are copied at the original vocabulary size before two marker
    embeddings are added. Arbitrary shape mismatches are not ignored.
    """
    config = transformers.AutoConfig.from_pretrained(encoder_path, trust_remote_code=trust_remote_code)
    source_architectures = list(getattr(config, "architectures", None) or [])
    source = transformers.AutoModel.from_pretrained(encoder_path, trust_remote_code=trust_remote_code)
    config.num_labels = 2
    config.problem_type = "single_label_classification"
    config.label2id, config.id2label = dict(LABEL2ID), {value: key for key, value in LABEL2ID.items()}
    model = transformers.AutoModelForSequenceClassification.from_config(config, trust_remote_code=trust_remote_code)
    model.base_model.load_state_dict(source.state_dict(), strict=True)
    del source
    tokenizer = transformers.AutoTokenizer.from_pretrained(encoder_path, use_fast=True,
                                                           trust_remote_code=trust_remote_code)
    markers_added = tokenizer.add_special_tokens({"additional_special_tokens": list(MARKERS)})
    model.resize_token_embeddings(len(tokenizer))
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    source_path = Path(encoder_path)
    provenance = {str(path): _sha256(path) for path in
                  (source_path / "config.json", source_path / "training_manifest.json",
                   source_path.parent / "training_manifest.json") if path.is_file()}
    return model, tokenizer, {"encoder_source": encoder_path, "source_provenance_sha256": provenance,
        "source_architectures": source_architectures, "encoder_transfer_strict": True,
        "classification_head": "fresh_two_label", "marker_tokens_added": markers_added}


def prepare_features(documents: Sequence[Any], tokenizer: Any,
                     max_length: int = 512,
                     input_version: str = DEFAULT_INPUT_VERSION) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build only contextual training examples, rejecting incomplete evidence."""
    rows: list[dict[str, Any]] = []
    report = {"documents": len(documents), "contextual_candidates": 0,
              "policy_mask_candidates": 0, "MASK": 0, "KEEP": 0}
    for document in documents:
        for mention in document.mentions:
            if mention.action not in LABEL2ID:
                raise ValueError("Every training mention needs an explicit MASK/KEEP label")
            if not keep_eligible(mention.entity_type):
                if mention.action != "MASK":
                    raise ValueError("Unsupported or mandatory-protected types cannot be KEEP")
                report["policy_mask_candidates"] += 1
                continue
            prepared = prepare_candidate_input(tokenizer, document, mention, max_length,
                                               input_version=input_version)
            if not prepared["available"]:
                raise ValueError(f"{document.doc_id}/{mention.mention_id}: {prepared['reason']}; "
                                 "provide an independently reviewed complete context unit or a supported larger max_length")
            rows.append({**prepared["model_inputs"], "labels": LABEL2ID[mention.action]})
            report["contextual_candidates"] += 1
            report[mention.action] += 1
    if not rows:
        raise ValueError("No contextual B training candidates")
    return rows, report


def _collator(tokenizer: Any, torch: Any, include_labels: bool = False) -> Any:
    def collate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        inputs = [{key: value for key, value in row.items() if key != "labels"} for row in rows]
        batch = dict(tokenizer.pad(inputs, padding=True, return_tensors="pt"))
        if include_labels:
            batch["labels"] = torch.tensor([row["labels"] for row in rows], dtype=torch.long)
        return batch
    return collate


def _dev_metrics(model: Any, loader: Any, device: Any, torch: Any) -> dict[str, Any]:
    model.eval()
    total_loss, count, true_positive, false_positive, false_negative, correct = 0.0, 0, 0, 0, 0, 0
    gold_mask, gold_keep = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**model_forward_inputs(model, batch))
            predicted = output.logits.argmax(dim=-1)
            labels = batch["labels"]
            size = int(labels.shape[0])
            total_loss += float(output.loss.item()) * size
            count += size
            correct += int((predicted == labels).sum().item())
            true_positive += int(((predicted == 1) & (labels == 1)).sum().item())
            false_positive += int(((predicted == 1) & (labels == 0)).sum().item())
            false_negative += int(((predicted == 0) & (labels == 1)).sum().item())
            gold_mask += int((labels == LABEL2ID["MASK"]).sum().item())
            gold_keep += int((labels == LABEL2ID["KEEP"]).sum().item())
    denominator = 2 * true_positive + false_positive + false_negative
    return {"loss": total_loss / count, "candidates": count, "accuracy": correct / count,
            "keep_f1_at_argmax": 2 * true_positive / denominator if denominator else 0.0,
            "false_keep_at_argmax": false_positive,
            "false_mask_at_argmax": false_negative,
            "gold_mask_candidates": gold_mask, "gold_keep_candidates": gold_keep,
            "false_keep_rate_at_argmax": false_positive / gold_mask if gold_mask else 0.0,
            "false_mask_rate_at_argmax": false_negative / gold_keep if gold_keep else 0.0,
            "note": "Argmax metrics are diagnostic; release threshold requires separate dev calibration"}


def _training_options(args: argparse.Namespace) -> dict[str, Any]:
    """Defaults preserve the original training contract for legacy callers."""
    options = {"input_version": getattr(args, "input_version", DEFAULT_INPUT_VERSION),
               "mask_class_weight": getattr(args, "mask_class_weight", 1.0),
               "label_smoothing": getattr(args, "label_smoothing", 0.0),
               "early_stopping_patience": getattr(args, "early_stopping_patience", 0)}
    input_spec(options["input_version"])
    weight, smoothing, patience = (options["mask_class_weight"], options["label_smoothing"],
                                   options["early_stopping_patience"])
    if isinstance(weight, bool) or not math.isfinite(weight) or weight < 1:
        raise ValueError("mask_class_weight must be finite and >= 1")
    if isinstance(smoothing, bool) or not math.isfinite(smoothing) or not 0 <= smoothing < 1:
        raise ValueError("label_smoothing must be finite and in [0,1)")
    if type(patience) is not int or patience < 0:
        raise ValueError("early_stopping_patience must be an integer >= 0 (0 disables it)")
    return options


def _training_loss(logits: Any, labels: Any, torch: Any,
                   mask_class_weight: float = 1.0, label_smoothing: float = 0.0) -> Any:
    """Cost-sensitive training only; selection still uses ordinary dev CE.

    A MASK example predicted KEEP incurs the higher MASK-class cost. Smoothing
    discourages saturated probabilities but does not certify safe confidence;
    all KEEP decisions still require independent dev calibration.
    """
    logits = logits.float()
    weights = logits.new_tensor([mask_class_weight, 1.0])
    return torch.nn.functional.cross_entropy(logits, labels, weight=weights,
                                             label_smoothing=label_smoothing)


def train(args: argparse.Namespace) -> dict[str, Any]:
    from .b_data import document_identities, load_reviews
    from .b_experiment import validate_b_evidence

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output-dir must be empty; existing checkpoints cannot be overwritten")
    if args.epochs < 1 or args.max_steps < 0 or min(args.batch_size, args.eval_batch_size) < 1:
        raise ValueError("epochs and batch sizes must be positive; max_steps must be >= 0")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or not 0 <= args.weight_decay < 1:
        raise ValueError("weight_decay must be in [0,1)")
    options = _training_options(args)
    train_documents = _load_documents(args.train)
    dev_documents = _load_documents(args.dev)
    experiment_dir = getattr(args, "synthetic_experiment_dir", None)
    train_review = validate_b_evidence(train_documents, load_reviews(args.train_reviews),
        expected_split="train", policy_id=FIXED_POLICY_ID, synthetic_experiment_dir=experiment_dir)
    dev_review = validate_b_evidence(dev_documents, load_reviews(args.dev_reviews),
        expected_split="dev", policy_id=FIXED_POLICY_ID, synthetic_experiment_dir=experiment_dir)
    if experiment_dir and train_review["packet_manifest_sha256"] != dev_review["packet_manifest_sha256"]:
        raise ValueError("Synthetic experiment packet changed between train/dev validation")
    validate_train_dev(train_documents, dev_documents)
    torch, transformers = _ml_dependencies()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _device(torch, args.device)
    model, tokenizer, initialization = load_initial_model(transformers, args.encoder_name_or_path,
                                                        args.trust_remote_code)
    _check_max_length(model, args.max_length)
    train_rows, train_report = prepare_features(train_documents, tokenizer, args.max_length, options["input_version"])
    dev_rows, dev_report = prepare_features(dev_documents, tokenizer, args.max_length, options["input_version"])
    model.to(device)
    collate = _collator(tokenizer, torch, include_labels=True)
    train_loader = torch.utils.data.DataLoader(train_rows, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = torch.utils.data.DataLoader(dev_rows, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_dir = output_dir / "best"
    manifest: dict[str, Any] = {
        "stage": MANIFEST_VERSION, "policy_id": FIXED_POLICY_ID, "policy_config": POLICY_CONFIG,
        "policy_sha256": policy_sha256(),
        "input_spec": input_spec(options["input_version"]),
        "input_sha256": input_sha256(options["input_version"]),
        "label2id": dict(LABEL2ID), "selection_metric": "dev_candidate_cross_entropy_loss",
        "selection_direction": "minimize", "context_policy": "whole_reviewed_document_required",
        "initialization": initialization,
        "config": {key: getattr(args, key) for key in ("epochs", "max_steps", "max_length", "batch_size", "eval_batch_size",
                    "learning_rate", "weight_decay", "seed", "device")},
        "loss_config": {"name": "weighted_cross_entropy", "class_weights": {
            "MASK": options["mask_class_weight"], "KEEP": 1.0},
            "label_smoothing": options["label_smoothing"],
            "selection_loss": "unweighted_unsmoothed_dev_cross_entropy"},
        "early_stopping": {"patience": options["early_stopping_patience"],
                           "metric": "dev_candidate_cross_entropy_loss", "min_delta": 0.0},
        "data": {"train": {"sha256": _sha256(args.train), "reviews_sha256": _sha256(args.train_reviews),
                              "identities": document_identities(train_documents),
                              "source_provenance": reviewed_data_provenance(train_documents),
                              **train_report, "review_validation": train_review},
                 "dev": {"sha256": _sha256(args.dev), "reviews_sha256": _sha256(args.dev_reviews),
                            "identities": document_identities(dev_documents),
                            "source_provenance": reviewed_data_provenance(dev_documents),
                            **dev_report, "review_validation": dev_review}},
        "history": [], "best_epoch": None, "best_dev_loss": None,
        "threshold": None, "calibration_required": True,
        "training_data_policy": ("unreviewed_synthetic_experiment; generated labels; human_reviewed=false"
                                 if experiment_dir else
                                 "human_approved_reviewed_data; declared source provenance recorded per split"),
        "training_data_provenance": reviewed_data_provenance([*train_documents, *dev_documents]),
    }
    manifest["config"].update(options)
    if experiment_dir:
        manifest["synthetic_experiment"] = train_review
        manifest["human_reviewed"] = False
    start_time, global_step, best_loss = time.monotonic(), 0, float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, seen = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**model_forward_inputs(model, {key: value for key, value in batch.items() if key != "labels"}))
            loss = _training_loss(output.logits, batch["labels"], torch,
                                  options["mask_class_weight"], options["label_smoothing"])
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Nonfinite B training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = int(batch["labels"].shape[0])
            epoch_loss += float(loss.detach().item()) * size
            seen += size
            global_step += 1
            if args.max_steps and global_step >= args.max_steps:
                break
        metrics = _dev_metrics(model, dev_loader, device, torch)
        if not math.isfinite(metrics["loss"]):
            raise RuntimeError("Nonfinite B dev loss")
        event = {"epoch": epoch, "global_step": global_step, "train_loss": epoch_loss / seen, "dev": metrics}
        manifest["history"].append(event)
        if metrics["loss"] < best_loss:
            epochs_without_improvement = 0
            best_loss = metrics["loss"]
            manifest["best_epoch"], manifest["best_dev_loss"] = epoch, best_loss
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            manifest["checkpoint_artifacts"] = {str(path.relative_to(best_dir)): _sha256(path)
                for path in sorted(best_dir.rglob("*")) if path.is_file() and path.name != "b_manifest.json"}
        else:
            epochs_without_improvement += 1
        stop_reason = None
        if args.max_steps and global_step >= args.max_steps:
            stop_reason = "max_steps_reached"
        elif options["early_stopping_patience"] and epochs_without_improvement >= options["early_stopping_patience"]:
            stop_reason = "early_stopping_dev_loss"
        elif epoch == args.epochs:
            stop_reason = "epochs_completed"
        manifest["epochs_without_improvement"] = epochs_without_improvement
        manifest["training_stop_reason"] = stop_reason
        manifest["elapsed_seconds"] = time.monotonic() - start_time
        manifest["global_step"] = global_step
        _write_json(output_dir / "b_manifest.json", manifest)
        _write_json(best_dir / "b_manifest.json", manifest)
        print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)
        if stop_reason:
            break
    return manifest


def score_documents(model: Any, tokenizer: Any, documents: Sequence[Any],
                    device: Any, torch: Any, max_length: int = 512,
                    batch_size: int = 8, policy_id: str = FIXED_POLICY_ID,
                    input_version: str = DEFAULT_INPUT_VERSION) -> list[dict[str, Any]]:
    """Return P(KEEP) without applying a release threshold or emitting raw PII."""
    from .b_data import document_fingerprint

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if policy_id != FIXED_POLICY_ID:
        raise ValueError("Unsupported B policy")
    _check_max_length(model, max_length)
    records, features, locations = [], [], []
    seen_ids = set()
    for document in documents:
        document.validate()
        if document.doc_id in seen_ids:
            raise ValueError("Duplicate document IDs")
        seen_ids.add(document.doc_id)
        if document.policy_id != policy_id:
            raise ValueError(f"{document.doc_id}: explicit policy_id={policy_id} required before B scoring")
        record = {"doc_id": document.doc_id, "document_sha256": document_fingerprint(document),
                  "policy_id": policy_id, "mentions": []}
        for mention in document.mentions:
            prepared = prepare_candidate_input(tokenizer, document, mention, max_length, policy_id, input_version)
            result = {"mention_id": mention.mention_id, "start": mention.start, "end": mention.end,
                      "entity_type": mention.entity_type, "p_keep": None, "available": prepared["available"],
                      "reason": prepared["reason"]}
            record["mentions"].append(result)
            if prepared["available"]:
                features.append(prepared["model_inputs"])
                locations.append(result)
        records.append(record)
    collate = _collator(tokenizer, torch)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(features), batch_size):
                batch = collate(features[start:start + batch_size])
                batch = {key: value.to(device) for key, value in batch.items()}
                logits = model(**model_forward_inputs(model, batch)).logits
                if logits.ndim != 2 or int(logits.shape[1]) != 2:
                    raise ValueError("B checkpoint must produce exactly two logits [MASK, KEEP]")
                if int(logits.shape[0]) != len(features[start:start + batch_size]):
                    raise ValueError("B checkpoint changed the number of scored candidates")
                probabilities = torch.softmax(logits.float(), dim=-1)[:, LABEL2ID["KEEP"]].cpu().tolist()
                for result, probability in zip(locations[start:start + batch_size], probabilities):
                    if math.isfinite(probability) and 0 <= probability <= 1:
                        result["p_keep"], result["reason"] = probability, "scored"
                    else:
                        result["available"], result["reason"] = False, "nonfinite_model_score"
    finally:
        model.train(was_training)
    return records


def load_checkpoint_manifest(model_dir: str | Path) -> dict[str, Any]:
    path = Path(model_dir) / "b_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("stage") != MANIFEST_VERSION or manifest.get("policy_id") != FIXED_POLICY_ID
            or manifest.get("label2id") != LABEL2ID or manifest.get("policy_config") != POLICY_CONFIG
            or manifest.get("policy_sha256") != policy_sha256()):
        raise ValueError("Checkpoint is not a compatible fixed-policy B model")
    if not isinstance(manifest.get("config", {}).get("max_length"), int):
        raise ValueError("B manifest requires its training max_length")
    input_version = manifest["config"].get("input_version", DEFAULT_INPUT_VERSION)
    expected_spec = input_spec(input_version)
    if ("input_version" in manifest["config"] or "input_spec" in manifest or "input_sha256" in manifest):
        if manifest.get("input_spec") != expected_spec or manifest.get("input_sha256") != input_sha256(input_version):
            raise ValueError("B checkpoint input specification changed; retrain and recalibrate changed inputs")
    artifacts = manifest.get("checkpoint_artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("B manifest requires saved model/tokenizer artifact hashes")
    for relative, expected in artifacts.items():
        candidate = Path(model_dir) / relative
        if not candidate.resolve().is_relative_to(Path(model_dir).resolve()):
            raise ValueError("Checkpoint artifact path escapes model-dir")
        if not candidate.is_file() or _sha256(candidate) != expected:
            raise ValueError(f"B checkpoint artifact changed: {relative}")
    return manifest


def predict_scores(documents: Sequence[Any], model_dir: str | Path, device: str = "cpu",
                   batch_size: int = 8, max_length: int | None = None,
                   trust_remote_code: bool = False) -> list[dict[str, Any]]:
    manifest = load_checkpoint_manifest(model_dir)
    trained_length = manifest["config"]["max_length"]
    if max_length is not None and max_length != trained_length:
        raise ValueError("Scoring max_length must equal the training manifest; recalibrate a changed input policy")
    torch, transformers = _ml_dependencies()
    target = _device(torch, device)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
    if getattr(model.config, "label2id", None) != LABEL2ID:
        raise ValueError("Model label2id disagrees with B manifest")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_dir), use_fast=True, trust_remote_code=trust_remote_code)
    model.to(target)
    return score_documents(model, tokenizer, documents, target, torch, max_length=trained_length, batch_size=batch_size,
                           input_version=manifest["config"].get("input_version", DEFAULT_INPUT_VERSION))


def score(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = Path(args.model_dir)
    output = Path(args.output)
    sidecar_path = Path(str(output) + ".manifest.json")
    input_path = Path(args.input)
    targets = [output.resolve(), sidecar_path.resolve()]
    if input_path.resolve() in targets or len(set(targets)) != 2:
        raise ValueError("Score output and sidecar cannot overwrite the input")
    if any(path.is_relative_to(model_dir.resolve()) for path in targets):
        raise ValueError("Score outputs cannot overwrite B checkpoint artifacts")
    if any(path.exists() for path in (output, sidecar_path)):
        raise ValueError("Score output or sidecar already exists; use new paths to preserve prior evidence")
    a_provenance = _a_score_provenance(input_path)
    documents = _load_documents(input_path)
    records = predict_scores(documents, model_dir, args.device, args.batch_size,
                            args.max_length, args.trust_remote_code)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records), encoding="utf-8")
    sidecar = {"stage": "B_SCORES_V1", "policy_id": FIXED_POLICY_ID,
               "b_manifest_sha256": _sha256(model_dir / "b_manifest.json"),
               "input_sha256": _sha256(args.input), "output_sha256": _sha256(output),
               **a_provenance,
               "documents": len(records), "mentions": sum(len(row["mentions"]) for row in records),
               "max_length": load_checkpoint_manifest(model_dir)["config"]["max_length"]}
    _write_json(sidecar_path, sidecar)
    return sidecar


def _a_score_provenance(input_path: Path) -> dict[str, Any]:
    """Bind integrated scores to A's checkpoint and exact input-window policy."""
    adjacent = Path(str(input_path) + ".manifest.json")
    result = {"a_training_manifest_sha256": None, "a_prediction_signature_sha256": None}
    if not adjacent.is_file():
        return result
    manifest = json.loads(adjacent.read_text(encoding="utf-8"))
    if manifest.get("stage") == "A-candidate-refinement":
        from .candidate_refinement import validate_refinement_manifest
        manifest = validate_refinement_manifest(input_path)
    elif manifest.get("stage") != "A-prediction":
        raise ValueError("Adjacent candidate manifest is not an A-prediction manifest")
    if manifest.get("output", {}).get("sha256") != _sha256(input_path):
        raise ValueError("Adjacent A prediction manifest does not match candidate file")
    parent_digest = manifest.get("training_manifest_sha256")
    if not isinstance(parent_digest, str) or len(parent_digest) != 64 or any(character not in "0123456789abcdef" for character in parent_digest):
        raise ValueError("A prediction manifest requires a valid training_manifest_sha256")
    # Stage A writes these at top level; accept no inferred defaults, because
    # changing max_length/stride also changes the candidate universe.
    for key in ("max_length", "stride"):
        if type(manifest.get(key)) is not int or manifest[key] < (1 if key == "max_length" else 0):
            raise ValueError(f"A prediction manifest requires explicit {key}")
    signature = {key: manifest[key] for key in ("training_manifest_sha256", "max_length", "stride")}
    if "checkpoint_artifacts" in manifest:
        signature["checkpoint_artifacts"] = manifest["checkpoint_artifacts"]
    if manifest.get("stage") == "A-candidate-refinement":
        signature["candidate_refinement"] = {key: manifest["refinement"][key]
                                              for key in ("version", "config_sha256")}
    result["a_training_manifest_sha256"] = parent_digest
    result["a_prediction_signature_sha256"] = hashlib.sha256(json.dumps(signature, sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train only from reviewed MASK/KEEP data; no test input")
    for name in ("train", "train-reviews", "dev", "dev-reviews", "output-dir"):
        training.add_argument("--" + name, required=True)
    training.add_argument("--encoder-name-or-path", default=DEFAULT_ENCODER)
    training.add_argument("--synthetic-experiment-dir",
        help="Explicit opt-in for an immutable generated bulk packet; PENDING is retained and recorded")
    training.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    training.add_argument("--epochs", type=int, default=3)
    training.add_argument("--max-steps", type=int, default=0)
    training.add_argument("--max-length", type=int, default=512)
    training.add_argument("--batch-size", type=int, default=4)
    training.add_argument("--eval-batch-size", type=int, default=8)
    training.add_argument("--learning-rate", type=float, default=3e-5)
    training.add_argument("--weight-decay", type=float, default=0.01)
    training.add_argument("--input-version", choices=INPUT_VERSIONS, default=DEFAULT_INPUT_VERSION,
        help="Input contract saved with the checkpoint; v2 repeats target-local context and retains the complete document")
    training.add_argument("--mask-class-weight", type=float, default=1.0,
        help="MASK-class training loss weight >= 1; does not relax the calibrated KEEP policy")
    training.add_argument("--label-smoothing", type=float, default=0.0)
    training.add_argument("--early-stopping-patience", type=int, default=0,
        help="Stop after this many non-improving dev-loss epochs; 0 disables early stopping")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--trust-remote-code", action="store_true")
    training.set_defaults(handler=train)
    scoring = commands.add_parser("score", help="Write label-free-input candidate P(KEEP) with provenance sidecar")
    for name in ("input", "model-dir", "output"):
        scoring.add_argument("--" + name, required=True)
    scoring.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    scoring.add_argument("--batch-size", type=int, default=8)
    scoring.add_argument("--max-length", type=int, default=None)
    scoring.add_argument("--trust-remote-code", action="store_true")
    scoring.set_defaults(handler=score)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
