"""Reviewed-dev calibration and frozen A+B test evaluation.

An explicit hash-bound synthetic experiment option retains PENDING evidence
and labels every calibration/test report as not human-reviewed.

Thresholds only remove spans from a fixed A candidate inventory. Consequently
the all-candidates-MASK baseline includes exactly the same A misses as A+B.
Scores and reports contain no original text. Canonical inputs remain private
research files; a masked rendering is not a guarantee that A found all PII.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .evaluation import evaluate_masking
from .masking import intervals_from_mentions, render_masked_text
from .schema import Document
from .stage_b import PrivacyDecision, apply_decisions

FIXED_POLICY_ID = "career_external_share_v1"
CALIBRATION_SCHEMA = "kpii-b-calibration-1"


def _json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _policy_config(policy_id: str) -> dict[str, Any]:
    from .b_inputs import POLICY_SPEC
    if policy_id != FIXED_POLICY_ID or POLICY_SPEC.get("policy_id") != policy_id:
        raise ValueError("Only the fixed career_external_share_v1 policy is supported")
    return json.loads(json.dumps(POLICY_SPEC, ensure_ascii=False, allow_nan=False))


def _index_docs(documents: Iterable[Document], label: str) -> dict[str, Document]:
    result: dict[str, Document] = {}
    for document in documents:
        if not isinstance(document, Document):
            raise ValueError(f"{label} must contain canonical Document instances")
        document.validate()
        if document.doc_id in result:
            raise ValueError(f"Duplicate {label} doc_id")
        result[document.doc_id] = document
    if not result:
        raise ValueError(f"{label} cannot be empty")
    return result


def load_scores(path: str | Path) -> list[dict[str, Any]]:
    """Load the score sidecar, validating JSON; pairing validation is separate."""
    records = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line, object_pairs_hook=_unique_json_keys,
                                   parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid score JSON on line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError("Each score record must be an object")
            records.append(value)
    return records


def _unique_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def validate_scores(candidate_docs: Iterable[Document], scores: Iterable[Mapping[str, Any]],
                    policy_id: str = FIXED_POLICY_ID) -> dict[str, dict[str, dict[str, Any]]]:
    """Require one finite score (or explicit unavailable score) per A candidate."""
    from .b_data import document_fingerprint
    _policy_config(policy_id)
    candidates = _index_docs(candidate_docs, "candidate")
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for record in scores:
        required = {"doc_id", "document_sha256", "policy_id", "mentions"}
        if not isinstance(record, Mapping) or set(record) != required:
            raise ValueError("Score document has missing or unsupported fields")
        doc_id = record["doc_id"]
        if not isinstance(doc_id, str) or doc_id not in candidates or doc_id in result:
            raise ValueError("Score IDs must match candidate documents exactly without duplicates")
        document = candidates[doc_id]
        if document.policy_id != policy_id or record["policy_id"] != policy_id:
            raise ValueError("Candidate and score policies must match the fixed policy")
        if record["document_sha256"] != document_fingerprint(document):
            raise ValueError("Candidate fingerprint differs from scored input")
        if not isinstance(record["mentions"], list):
            raise ValueError("Scored mentions must be an explicit list")
        mentions = {m.mention_id: m for m in document.mentions}
        indexed: dict[str, dict[str, Any]] = {}
        for score in record["mentions"]:
            fields = {"mention_id", "start", "end", "entity_type", "p_keep", "available", "reason"}
            if not isinstance(score, dict) or set(score) != fields:
                raise ValueError("Mention score has missing or unsupported fields")
            mention_id = score["mention_id"]
            if not isinstance(mention_id, str) or mention_id not in mentions or mention_id in indexed:
                raise ValueError("Score mention IDs must match candidates exactly without duplicates")
            mention = mentions[mention_id]
            if (type(score["start"]) is not int or type(score["end"]) is not int or
                (score["start"], score["end"], score["entity_type"]) !=
                    (mention.start, mention.end, mention.entity_type)):
                raise ValueError("Score span/type differs from its candidate")
            if type(score["available"]) is not bool:
                raise ValueError("Score available must be boolean")
            p_keep = score["p_keep"]
            if p_keep is None:
                if score["available"]:
                    raise ValueError("Available scores require a probability")
            elif (type(p_keep) not in (int, float) or not math.isfinite(p_keep) or
                  not 0 <= p_keep <= 1 or not score["available"]):
                raise ValueError("Probabilities must be finite 0..1 and available")
            if not isinstance(score["reason"], str) or not score["reason"]:
                raise ValueError("Score reason must be a nonempty string")
            indexed[mention_id] = dict(score)
        if set(indexed) != set(mentions):
            raise ValueError("Missing candidate scores")
        result[doc_id] = indexed
    if set(result) != set(candidates):
        raise ValueError("Missing candidate score documents")
    return result


def _pair_gold_candidates(gold_docs: Iterable[Document], candidate_docs: Iterable[Document],
                          policy_id: str, split: str) -> tuple[list[Document], list[Document]]:
    gold, candidates = _index_docs(gold_docs, "gold"), _index_docs(candidate_docs, "candidate")
    if set(gold) != set(candidates):
        raise ValueError("Gold and A candidate document IDs must match exactly")
    permitted = {"dev", "valid", "validation"} if split == "dev" else {"test"}
    for doc_id, document in gold.items():
        candidate = candidates[doc_id]
        if document.split not in permitted:
            raise ValueError(f"This operation requires {split} documents")
        if document.policy_id != policy_id or candidate.policy_id != policy_id:
            raise ValueError("Gold and candidates must use the fixed policy")
        if (document.text != candidate.text or document.source_group != candidate.source_group or
                document.trusted_context != candidate.trusted_context or candidate.split != document.split):
            raise ValueError("Candidate text, group, split, and trusted metadata must match gold")
        if any(m.action not in ("MASK", "KEEP") for m in document.mentions):
            raise ValueError("Every gold mention requires explicit MASK/KEEP; null is not KEEP")
    from .b_data import validate_split_separation
    validate_split_separation({split: gold.values()})
    return list(gold.values()), [candidates[key] for key in gold]


def _mask_records(candidates: list[Document], scores: dict[str, dict[str, dict[str, Any]]],
                  threshold: float | None, policy_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from .b_inputs import allowed_to_keep
    if threshold is not None and (type(threshold) not in (int, float) or
                                   not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("KEEP threshold must be null (disabled) or finite 0..1")
    masks, decisions_records = [], []
    for document in candidates:
        decisions = []
        for mention in document.mentions:
            score = scores[document.doc_id][mention.mention_id]
            eligible = allowed_to_keep(document, mention, policy_id=policy_id)
            keep = (threshold is not None and eligible and score["available"] and
                    score["p_keep"] is not None and score["p_keep"] >= threshold)
            reason = ("calibrated_keep" if keep else "policy_mask" if not eligible else
                      "keep_disabled" if threshold is None else
                      "score_unavailable" if not score["available"] else "below_keep_threshold")
            decisions.append(PrivacyDecision(mention.mention_id, "KEEP" if keep else "MASK", reason))
        decided = apply_decisions(document, decisions)
        intervals = intervals_from_mentions(document.text, decided.mentions)
        masks.append({"doc_id": document.doc_id, "text": document.text, "intervals": intervals})
        decisions_records.append({"doc_id": document.doc_id, "mentions": [
            {"mention_id": m.mention_id, "start": m.start, "end": m.end,
             "entity_type": m.entity_type, "action": m.action, "reason_code": m.reason_code}
            for m in decided.mentions]})
    return masks, decisions_records


def _identities(documents: Iterable[Document]) -> dict[str, list[str]]:
    from .b_data import document_identities
    return document_identities(documents)


def _reviewed(documents: list[Document], reviews: Any, policy_id: str,
              synthetic_experiment_dir: str | Path | None = None) -> dict[str, Any]:
    from .b_experiment import validate_b_evidence
    from .b_inputs import allowed_to_keep
    evidence = validate_b_evidence(documents, reviews, expected_split=None, policy_id=policy_id,
                                  synthetic_experiment_dir=synthetic_experiment_dir)
    for document in documents:
        if any(m.action == "KEEP" and not allowed_to_keep(document, m, policy_id=policy_id)
               for m in document.mentions):
            raise ValueError("Gold KEEP conflicts with fixed-policy type/trusted metadata gates")
    return evidence


def _experiment_digest(evidence: Any) -> str:
    if (not isinstance(evidence, Mapping) or evidence.get("human_reviewed") is not False or
            evidence.get("experiment_mode") != "unreviewed_synthetic_experiment"):
        raise ValueError("Malformed unreviewed synthetic experiment evidence")
    return _digest_string(evidence.get("packet_manifest_sha256"), "packet_manifest_sha256")


def _bind_experiment(evidence: Mapping[str, Any] | None, *,
                     calibration: Mapping[str, Any] | None = None,
                     model_training_evidence: Mapping[str, Any] | None = None) -> None:
    """Keep opt-in, calibration, test and B training on one explicit packet."""
    calibrated_evidence = calibration.get("synthetic_experiment") if calibration is not None else None
    if evidence is None:
        if calibrated_evidence is not None or model_training_evidence is not None:
            raise ValueError("Unreviewed synthetic calibration/model requires --synthetic-experiment-dir")
        return
    digest = _experiment_digest(evidence)
    if calibration is not None:
        if calibrated_evidence is None:
            raise ValueError("Cannot mix reviewed calibration with an unreviewed synthetic experiment")
        if _experiment_digest(calibrated_evidence) != digest:
            raise ValueError("Calibration and test must use the same synthetic packet_manifest_sha256")
    if model_training_evidence is not None and _experiment_digest(model_training_evidence) != digest:
        raise ValueError("B training and evaluation must use the same synthetic packet_manifest_sha256")


def _tag_experiment(result: dict[str, Any], evidence: Mapping[str, Any] | None) -> dict[str, Any]:
    if evidence is not None:
        result.update(synthetic_experiment=dict(evidence), human_reviewed=False,
                      experiment_mode="unreviewed_synthetic_experiment")
    return result


def _reject_reused_identities(current: Mapping[str, list[str]], reference: Any, label: str) -> None:
    if not isinstance(reference, dict) or set(reference) != set(current):
        raise ValueError(f"Missing or incompatible {label} identity bindings")
    for key, values in current.items():
        if not isinstance(reference[key], list) or not reference[key]:
            raise ValueError(f"Malformed {label} identity bindings")
        for digest in reference[key]:
            _digest_string(digest, f"{label} identity")
        if set(values) & set(reference[key]):
            raise ValueError(f"Data reuses {label} document/text/source group")


def calibrate_b(gold_documents: Iterable[Document], candidate_documents: Iterable[Document],
                scores: Iterable[Mapping[str, Any]], reviews: Any, *, model_manifest_sha256: str,
                policy_id: str = FIXED_POLICY_ID, max_added_leak_rate: float = 0.0,
                min_overmask_improvement: float = 0.0,
                a_training_manifest_sha256: str | None = None,
                a_prediction_signature_sha256: str | None = None,
                synthetic_experiment_dir: str | Path | None = None,
                model_training_evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Select on dev only; disable KEEP absent a strict improvement.

    ``max_added_leak_rate`` is a development constraint, not a production privacy
    guarantee. Sampling uncertainty and baseline A misses remain visible.
    PENDING synthetic gold requires an explicit immutable-packet opt-in.
    """
    config = _policy_config(policy_id)
    _digest_string(model_manifest_sha256, "model_manifest_sha256")
    if a_training_manifest_sha256 is not None:
        _digest_string(a_training_manifest_sha256, "a_training_manifest_sha256")
    if a_prediction_signature_sha256 is not None:
        _digest_string(a_prediction_signature_sha256, "a_prediction_signature_sha256")
    for name, value in (("max_added_leak_rate", max_added_leak_rate),
                        ("min_overmask_improvement", min_overmask_improvement)):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite 0..1")
    gold, candidates = _pair_gold_candidates(gold_documents, candidate_documents, policy_id, "dev")
    review = _reviewed(gold, reviews, policy_id, synthetic_experiment_dir)
    experiment = review if synthetic_experiment_dir is not None else None
    _bind_experiment(experiment, model_training_evidence=model_training_evidence)
    indexed = validate_scores(candidates, scores, policy_id)
    baseline_masks, _ = _mask_records(candidates, indexed, None, policy_id)
    baseline = evaluate_masking(gold, baseline_masks)
    if not baseline["overall"]["gold_mask_mentions"] or not baseline["overall"]["gold_keep_mentions"]:
        raise ValueError("Calibration needs both reviewed MASK and KEEP gold mentions")
    from .b_inputs import allowed_to_keep
    thresholds = sorted({indexed[doc.doc_id][m.mention_id]["p_keep"]
                         for doc in candidates for m in doc.mentions
                         if allowed_to_keep(doc, m, policy_id=policy_id)
                         and indexed[doc.doc_id][m.mention_id]["available"]
                         and indexed[doc.doc_id][m.mention_id]["p_keep"] is not None}, reverse=True)
    selected_threshold, selected = None, baseline
    sweep = []
    for threshold in thresholds:
        masks, _ = _mask_records(candidates, indexed, threshold, policy_id)
        metric = evaluate_masking(gold, masks)
        added_leak = metric["overall"]["leak_rate"] - baseline["overall"]["leak_rate"]
        improvement = baseline["overall"]["overmask_rate"] - metric["overall"]["overmask_rate"]
        feasible = added_leak <= max_added_leak_rate + 1e-12
        sweep.append({"threshold": threshold, "overall": metric["overall"],
                      "added_leak_rate": added_leak, "overmask_improvement": improvement,
                      "feasible": feasible})
        if not feasible or improvement <= 1e-12 or improvement + 1e-12 < min_overmask_improvement:
            continue
        key = (metric["overall"]["overmask_rate"], metric["overall"]["leak_rate"], -threshold)
        best_key = (selected["overall"]["overmask_rate"], selected["overall"]["leak_rate"],
                    -selected_threshold if selected_threshold is not None else -math.inf)
        if key < best_key:
            selected_threshold, selected = threshold, metric
    return _tag_experiment({
        "schema_version": CALIBRATION_SCHEMA,
        "policy_id": policy_id, "policy_config": config, "policy_sha256": _json_digest(config),
        "b_manifest_sha256": model_manifest_sha256,
        "a_training_manifest_sha256": a_training_manifest_sha256,
        "a_prediction_signature_sha256": a_prediction_signature_sha256,
        "calibration_split": "dev", "dev_identities": _identities(gold),
        "candidate_identities": _identities(candidates),
        "threshold": selected_threshold, "keep_enabled": selected_threshold is not None,
        "constraints": {"max_added_leak_rate": max_added_leak_rate,
                        "min_overmask_improvement": min_overmask_improvement},
        "selection": "min_overmask_then_min_leak_then_highest_threshold; strict_improvement_required",
        "baseline": baseline, "selected": selected, "sweep": sweep,
        "limitations": ["Finite dev observations do not guarantee zero future leakage.",
                        "B removes A candidates; A misses remain in every leakage denominator."],
    }, experiment)


def _validate_calibration(calibration: Mapping[str, Any], model_manifest_sha256: str,
                          policy_id: str, a_training_manifest_sha256: str | None = None,
                          a_prediction_signature_sha256: str | None = None) -> None:
    config = _policy_config(policy_id)
    _digest_string(model_manifest_sha256, "model_manifest_sha256")
    if (calibration.get("schema_version") != CALIBRATION_SCHEMA or
            calibration.get("policy_id") != policy_id or calibration.get("calibration_split") != "dev" or
            calibration.get("policy_sha256") != _json_digest(config) or
            calibration.get("policy_config") != config):
        raise ValueError("Calibration schema/policy differs from current implementation")
    if calibration.get("b_manifest_sha256") != model_manifest_sha256:
        raise ValueError("Calibration belongs to a different B model")
    expected_a = calibration.get("a_training_manifest_sha256")
    if expected_a is not None:
        _digest_string(expected_a, "calibration a_training_manifest_sha256")
        if a_training_manifest_sha256 != expected_a:
            raise ValueError("Calibration and inference must use the same fixed A model")
    expected_signature = calibration.get("a_prediction_signature_sha256")
    if expected_signature is not None:
        _digest_string(expected_signature, "calibration a_prediction_signature_sha256")
        if a_prediction_signature_sha256 != expected_signature:
            raise ValueError("Calibration and inference must use the same A prediction configuration")
    threshold = calibration.get("threshold")
    if type(calibration.get("keep_enabled")) is not bool or calibration["keep_enabled"] != (threshold is not None):
        raise ValueError("Calibration threshold and keep_enabled disagree")
    if threshold is not None and (type(threshold) not in (int, float) or
                                   not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("Invalid calibrated threshold")
    if "synthetic_experiment" in calibration:
        _experiment_digest(calibration["synthetic_experiment"])
        if calibration.get("human_reviewed") is not False:
            raise ValueError("Unreviewed synthetic calibration cannot claim human review")


def apply_calibrated_b(candidate_documents: Iterable[Document], scores: Iterable[Mapping[str, Any]],
                       calibration: Mapping[str, Any], *, model_manifest_sha256: str,
                       policy_id: str = FIXED_POLICY_ID,
                       a_training_manifest_sha256: str | None = None,
                       a_prediction_signature_sha256: str | None = None,
                       model_training_evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Apply a frozen reviewed-dev threshold without requiring gold labels."""
    _validate_calibration(calibration, model_manifest_sha256, policy_id, a_training_manifest_sha256,
                          a_prediction_signature_sha256)
    experiment = calibration.get("synthetic_experiment")
    _bind_experiment(experiment, calibration=calibration,
                     model_training_evidence=model_training_evidence)
    candidates = list(_index_docs(candidate_documents, "candidate").values())
    indexed = validate_scores(candidates, scores, policy_id)
    masks, decisions = _mask_records(candidates, indexed, calibration["threshold"], policy_id)
    return _tag_experiment({"decisions": decisions, "mask_records": masks, "masked_documents": [
        {"doc_id": record["doc_id"], "masked_text": render_masked_text(record["text"], record["intervals"])}
        for record in masks]}, experiment)


def evaluate_ab(gold_documents: Iterable[Document], candidate_documents: Iterable[Document],
                scores: Iterable[Mapping[str, Any]], reviews: Any, calibration: Mapping[str, Any], *,
                model_manifest_sha256: str, policy_id: str = FIXED_POLICY_ID,
                a_training_manifest_sha256: str | None = None,
                a_prediction_signature_sha256: str | None = None,
                synthetic_experiment_dir: str | Path | None = None,
                model_training_evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate on disjoint test gold with reviewed or explicit packet evidence."""
    _validate_calibration(calibration, model_manifest_sha256, policy_id, a_training_manifest_sha256,
                          a_prediction_signature_sha256)
    if synthetic_experiment_dir is None:
        _bind_experiment(None, calibration=calibration, model_training_evidence=model_training_evidence)
    threshold = calibration["threshold"]
    gold, candidates = _pair_gold_candidates(gold_documents, candidate_documents, policy_id, "test")
    review = _reviewed(gold, reviews, policy_id, synthetic_experiment_dir)
    experiment = review if synthetic_experiment_dir is not None else None
    _bind_experiment(experiment, calibration=calibration, model_training_evidence=model_training_evidence)
    identities = _identities(gold)
    dev = calibration.get("dev_identities")
    _reject_reused_identities(identities, dev, "calibration")
    indexed = validate_scores(candidates, scores, policy_id)
    baseline_masks, _ = _mask_records(candidates, indexed, None, policy_id)
    chosen_masks, decisions = _mask_records(candidates, indexed, threshold, policy_id)
    baseline, chosen = evaluate_masking(gold, baseline_masks), evaluate_masking(gold, chosen_masks)
    delta = {key: chosen["overall"][key] - baseline["overall"][key]
             for key in ("leak_rate", "overmask_rate", "document_leak_rate",
                         "leaked_mask_mentions", "overmasked_keep_mentions")}
    if delta["leaked_mask_mentions"] < 0:
        raise RuntimeError("Invariant violated: candidate removal cannot repair A misses")
    return _tag_experiment({
        "evaluation": "frozen_A_plus_calibrated_B_test", "policy_id": policy_id,
        "b_manifest_sha256": model_manifest_sha256, "threshold": threshold,
        "a_training_manifest_sha256": a_training_manifest_sha256,
        "a_prediction_signature_sha256": a_prediction_signature_sha256,
        "keep_enabled": threshold is not None, "test_identities": identities,
        "baseline_all_mask": baseline, "calibrated_b": chosen,
        "delta_b_minus_baseline": delta, "decisions": decisions,
        "masked_documents": [
            {"doc_id": record["doc_id"], "masked_text": render_masked_text(record["text"], record["intervals"])}
            for record in chosen_masks],
    }, experiment)


def _score_provenance(scores_path: str | Path, candidates_path: str | Path,
                      model_manifest_path: str | Path, *, require_a: bool = False,
                      verify_checkpoint: bool = True) -> dict[str, Any]:
    model_path = Path(model_manifest_path)
    if model_path.name != "b_manifest.json":
        raise ValueError("--model-manifest must point to checkpoint/b_manifest.json")
    if verify_checkpoint:
        from .b_model import load_checkpoint_manifest
        model_metadata = load_checkpoint_manifest(model_path.parent)
    else:
        model_metadata = json.loads(model_path.read_text(encoding="utf-8-sig"))
    digest = _file_digest(model_path)
    manifest = json.loads(Path(str(scores_path) + ".manifest.json").read_text(encoding="utf-8-sig"),
                          object_pairs_hook=_unique_json_keys)
    if (not isinstance(manifest, dict) or manifest.get("stage") != "B_SCORES_V1" or
            manifest.get("policy_id") != FIXED_POLICY_ID or manifest.get("b_manifest_sha256") != digest or
            manifest.get("input_sha256") != _file_digest(candidates_path) or
            manifest.get("output_sha256") != _file_digest(scores_path)):
        raise ValueError("Score file/model/input provenance mismatch; score again before evaluation")
    a_digest = manifest.get("a_training_manifest_sha256")
    if require_a or a_digest is not None:
        _digest_string(a_digest, "A+B commands require A prediction a_training_manifest_sha256")
    a_signature = manifest.get("a_prediction_signature_sha256")
    if require_a or a_signature is not None:
        _digest_string(a_signature, "A+B commands require A prediction a_prediction_signature_sha256")
    result = {"b_manifest_sha256": digest, "scores_sha256": manifest["output_sha256"],
            "candidates_sha256": manifest["input_sha256"], "a_training_manifest_sha256": a_digest,
            "a_prediction_signature_sha256": a_signature,
            "model_data_identities": {name: model_metadata.get("data", {}).get(name, {}).get("identities")
                                      for name in ("train", "dev")}}
    if "synthetic_experiment" in model_metadata:
        _experiment_digest(model_metadata["synthetic_experiment"])
        if model_metadata.get("human_reviewed") is not False:
            raise ValueError("Unreviewed synthetic model cannot claim human review")
        result["synthetic_experiment"] = model_metadata["synthetic_experiment"]
    return result


def _preflight(inputs: Iterable[str | Path], outputs: Iterable[str | Path]) -> None:
    sources = {Path(p).resolve() for p in inputs}
    targets = [Path(p).resolve() for p in outputs]
    if len(set(targets)) != len(targets) or sources & set(targets):
        raise ValueError("Output paths must be distinct and cannot overwrite inputs")
    for target in targets:
        if target.exists():
            raise ValueError("Output already exists; choose a fresh output path")


def _write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    for name in ("calibrate", "evaluate", "apply"):
        command = sub.add_parser(name)
        for option in ("candidates", "scores", "model-manifest"):
            command.add_argument("--" + option, required=True)
        if name != "apply":
            command.add_argument("--gold", required=True)
            command.add_argument("--reviews", required=True)
            command.add_argument("--synthetic-experiment-dir",
                                 help="Explicitly use original PENDING synthetic labels for research evaluation")
        command.add_argument("--policy-id", default=FIXED_POLICY_ID)
        if name == "calibrate":
            command.add_argument("--output", required=True)
            command.add_argument("--max-added-leak-rate", type=float, default=0.0)
            command.add_argument("--min-overmask-improvement", type=float, default=0.0)
        else:
            command.add_argument("--calibration", required=True)
            command.add_argument("--output-dir", required=True)
            if name == "apply":
                command.add_argument("--mask-records", help="Optional INTERNAL original-text evaluation sidecar")
    args = parser.parse_args(argv)
    from .b_data import load_reviews
    from .data import read_jsonl
    inputs = [args.candidates, args.scores, args.model_manifest, str(args.scores) + ".manifest.json"]
    if args.operation != "apply":
        inputs.extend((args.gold, args.reviews))
    if args.operation == "calibrate":
        outputs = [args.output]
    else:
        inputs.append(args.calibration)
        output_dir = Path(args.output_dir)
        if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
            raise ValueError("Evaluation output directory must be new or empty")
        names = ("ab-evaluation.json", "decisions.jsonl", "masked.jsonl") if args.operation == "evaluate" else ("decisions.jsonl", "masked.jsonl")
        outputs = [output_dir / name for name in names]
        if args.operation == "apply" and args.mask_records:
            outputs.append(Path(args.mask_records))
    _preflight(inputs, outputs)
    provenance = _score_provenance(args.scores, args.candidates, args.model_manifest, require_a=True)
    common = dict(candidate_documents=read_jsonl(args.candidates), scores=load_scores(args.scores),
                  model_manifest_sha256=provenance["b_manifest_sha256"], policy_id=args.policy_id,
                  a_training_manifest_sha256=provenance["a_training_manifest_sha256"],
                  a_prediction_signature_sha256=provenance["a_prediction_signature_sha256"])
    if "synthetic_experiment" in provenance:
        common["model_training_evidence"] = provenance["synthetic_experiment"]
    if args.operation != "apply":
        common.update(gold_documents=read_jsonl(args.gold), reviews=load_reviews(args.reviews),
                      synthetic_experiment_dir=args.synthetic_experiment_dir)
        current_identities = _identities(common["gold_documents"])
        _reject_reused_identities(current_identities, provenance["model_data_identities"]["train"], "B training")
        if args.operation == "evaluate":
            _reject_reused_identities(current_identities, provenance["model_data_identities"]["dev"], "B development")
    # The full bindings remain in the checked model manifest; avoid duplicating
    # thousands of identities in every report.
    provenance.pop("model_data_identities")
    if args.operation == "calibrate":
        result = calibrate_b(**common, max_added_leak_rate=args.max_added_leak_rate,
                             min_overmask_improvement=args.min_overmask_improvement)
        result["provenance"] = provenance
        _write_json(args.output, result)
        print(json.dumps({"threshold": result["threshold"], "keep_enabled": result["keep_enabled"],
                          "selected_overall": result["selected"]["overall"]}, indent=2))
    elif args.operation == "evaluate":
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8-sig"))
        result = evaluate_ab(**common, calibration=calibration)
        decisions, masked = result.pop("decisions"), result.pop("masked_documents")
        result["provenance"] = {**provenance, "calibration_sha256": _file_digest(args.calibration)}
        _write_json(outputs[0], result)
        for target, records in ((outputs[1], decisions), (outputs[2], masked)):
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps({"baseline": result["baseline_all_mask"]["overall"],
                          "calibrated_b": result["calibrated_b"]["overall"],
                          "delta": result["delta_b_minus_baseline"]}, indent=2))
    else:
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8-sig"))
        result = apply_calibrated_b(**common, calibration=calibration)
        pairs = [(outputs[0], result["decisions"]), (outputs[1], result["masked_documents"])]
        if args.mask_records:
            pairs.append((outputs[2], result["mask_records"]))
        for target, records in pairs:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps({"documents": len(result["masked_documents"]),
                          "keep_enabled": calibration["keep_enabled"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
