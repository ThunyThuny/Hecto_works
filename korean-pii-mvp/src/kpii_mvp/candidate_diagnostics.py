"""Compare raw/refined candidate coverage without changing strict metrics.

This is an offline diagnostic: gold labels never alter candidates or B inputs.
No model, tokenizer or GPU is loaded. Reports contain IDs/counts, not source text.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .b_inputs import allowed_to_keep, canonical_type, keep_eligible
from .data import read_jsonl
from .evaluation import evaluate_masking


def _index(documents):
    result = {}
    for document in documents:
        document.validate()
        if document.doc_id in result:
            raise ValueError("Duplicate diagnostic document ID")
        result[document.doc_id] = document
    if not result:
        raise ValueError("Diagnostic documents cannot be empty")
    return result


def diagnose_candidates(gold_documents, candidate_documents):
    gold, candidates = _index(gold_documents), _index(candidate_documents)
    if set(gold) != set(candidates):
        raise ValueError("Gold/candidate diagnostic IDs must match exactly")
    by_type = defaultdict(Counter)
    totals = Counter()
    masks = []
    for doc_id, document in gold.items():
        candidate = candidates[doc_id]
        if (document.text, document.split, document.trusted_context) != (
                candidate.text, candidate.split, candidate.trusted_context):
            raise ValueError("Gold/candidate text, split and trusted context must agree")
        covered = {i for m in candidate.mentions for i in range(m.start, m.end)}
        masks.append({"doc_id": doc_id, "text": document.text,
                      "intervals": [[m.start, m.end] for m in candidate.mentions]})
        totals["a_candidates"] += len(candidate.mentions)
        totals["keep_gate_eligible_candidates"] += sum(allowed_to_keep(candidate, m) for m in candidate.mentions)
        annotated = {i for m in document.mentions for i in range(m.start, m.end)}
        totals["masked_unannotated_characters"] += len(covered - annotated)
        for mention in document.mentions:
            if mention.action not in {"MASK", "KEEP"}:
                raise ValueError("Diagnostic gold requires explicit actions")
            stats = by_type[canonical_type(mention.entity_type) + ":" + mention.action]
            stats["gold_mentions"] += 1
            exact = [m for m in candidate.mentions if (m.start, m.end) == (mention.start, mention.end)
                     and canonical_type(m.entity_type) == canonical_type(mention.entity_type)]
            stats["exact_span_and_type_candidates"] += bool(exact)
            stats["exact_span_and_type_keep_eligible"] += any(allowed_to_keep(candidate, m) for m in exact)
            missing = [i for i in range(mention.start, mention.end) if i not in covered]
            category = ("fully_covered" if not missing else "no_overlap" if len(missing) == mention.end - mention.start
                        else "whitespace_only_uncovered" if all(document.text[i].isspace() for i in missing)
                        else "partial_nonspace_uncovered")
            stats[category] += 1
    strict = evaluate_masking(list(gold.values()), masks)
    return {"counts": dict(totals), "by_type_and_action": {k: dict(v) for k, v in sorted(by_type.items())},
            "strict_allmask": {k: strict[k] for k in ("definitions", "overall", "by_type")}}


def compare_candidates(gold_documents, raw_documents, refined_documents):
    gold, raw, refined = list(gold_documents), list(raw_documents), list(refined_documents)
    raw_index, refined_index = _index(raw), _index(refined)
    if set(raw_index) != set(refined_index):
        raise ValueError("Raw/refined document IDs must match")
    for doc_id, document in raw_index.items():
        other = refined_index[doc_id]
        if document.text != other.text:
            raise ValueError("Candidate refinement must preserve original text")
        old = {i for m in document.mentions for i in range(m.start, m.end)}
        new = {i for m in other.mentions for i in range(m.start, m.end)}
        if not old.issubset(new):
            raise ValueError("Candidate refinement must not remove raw A coverage")
    before, after = diagnose_candidates(gold, raw), diagnose_candidates(gold, refined)
    return {"schema_version": "kpii-candidate-diagnostics-1", "model_execution_performed": False,
            "strict_metric_unchanged": True,
            "note": "Whitespace-only misses remain strict leaks. A+refiner is a separate pipeline from raw A.",
            "raw_a": before, "refined_a": after,
            "strict_delta_refined_minus_raw": {k: after["strict_allmask"]["overall"][k] - before["strict_allmask"]["overall"][k]
                for k in ("leaked_mask_mentions", "overmasked_keep_mentions", "leak_rate", "overmask_rate")}}


def diagnose_classification(gold_documents, scores):
    from .b_evaluation import validate_scores
    gold = list(gold_documents)
    indexed = validate_scores(gold, scores)
    counts = Counter({"MASK_to_MASK": 0, "MASK_to_KEEP": 0, "KEEP_to_MASK": 0, "KEEP_to_KEEP": 0,
                      "policy_always_mask": 0, "unavailable_contextual": 0})
    for document in gold:
        for mention in document.mentions:
            if mention.action not in {"MASK", "KEEP"}:
                raise ValueError("Classification diagnostic requires explicit gold actions")
            if not keep_eligible(mention.entity_type):
                counts["policy_always_mask"] += 1
                continue
            score = indexed[document.doc_id][mention.mention_id]
            if not score["available"]:
                counts["unavailable_contextual"] += 1
                continue
            # Equal logits choose MASK (label ID 0), as torch.argmax does.
            predicted = "KEEP" if score["p_keep"] > 0.5 else "MASK"
            counts[mention.action + "_to_" + predicted] += 1
    tp, fp, fn = counts["KEEP_to_KEEP"], counts["MASK_to_KEEP"], counts["KEEP_to_MASK"]
    total = sum(counts[k] for k in ("MASK_to_MASK", "MASK_to_KEEP", "KEEP_to_MASK", "KEEP_to_KEEP"))
    return {"schema_version": "kpii-b-gold-candidate-classification-1", "counts": dict(counts),
            "scorable_candidates": total,
            "accuracy": (counts["MASK_to_MASK"] + tp) / total if total else None,
            "keep_f1_at_argmax": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "false_keep_rate": fp / (fp + counts["MASK_to_MASK"]) if fp + counts["MASK_to_MASK"] else None,
            "note": "Gold-candidate argmax diagnostic on scorable contextual mentions; not calibrated A+B performance.",
            "model_execution_performed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    compare = commands.add_parser("diagnose")
    for name in ("gold", "raw-candidates", "candidates", "output"):
        compare.add_argument("--" + name, required=True)
    classify = commands.add_parser("classify")
    for name in ("gold", "scores", "model-manifest", "output"):
        classify.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    keys = ("gold", "raw_candidates", "candidates") if args.operation == "diagnose" else ("gold", "scores", "model_manifest")
    paths = [Path(getattr(args, key)) for key in keys]
    target = Path(args.output)
    if target.exists() or target.resolve() in {p.resolve() for p in paths}:
        raise ValueError("Diagnostic output must be new and outside inputs")
    if args.operation == "diagnose":
        report = compare_candidates(*(read_jsonl(path) for path in paths))
    else:
        from .b_evaluation import _score_provenance, load_scores
        provenance = _score_provenance(args.scores, args.gold, args.model_manifest)
        report = diagnose_classification(read_jsonl(args.gold), load_scores(args.scores))
        report["provenance"] = provenance
        if "synthetic_experiment" in provenance:
            report.update(human_reviewed=False, experiment_mode="unreviewed_synthetic_experiment")
    report["input_sha256"] = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in zip(keys, paths)}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(target), "strict_delta_refined_minus_raw": report.get("strict_delta_refined_minus_raw"),
                      "model_execution_performed": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
