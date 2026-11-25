"""Public command line; prediction/evaluation files are internal research data."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

COMMANDS = {
    "inspect-kdpii": ("kdpii", "inspect"),
    "prepare-kdpii": ("kdpii", "prepare"),
    "train-a": ("stage_a", "train"),
    "predict-a": ("stage_a", "predict"),
    "export-b-review": ("b_data", "export-review"),
    "import-b-review": ("b_data", "import-review"),
    "validate-b-data": ("b_data", "validate"),
    "prepare-b-a-input": ("b_pipeline", "prepare-a-input"),
    "preflight-b": ("b_pipeline", "preflight"),
    "train-b": ("b_model", "train"),
    "score-b": ("b_model", "score"),
    "refine-a-candidates": ("candidate_refinement", "refine"),
    "diagnose-b-candidates": ("candidate_diagnostics", "diagnose"),
    "diagnose-b-classification": ("candidate_diagnostics", "classify"),
    "audit-b-inputs": ("b_input_audit", "audit"),
    "generate-synthetic-v4": ("synthetic_v4", ""),
    "calibrate-b": ("b_evaluation", "calibrate"),
    "evaluate-ab": ("b_evaluation", "evaluate"),
    "apply-b": ("b_evaluation", "apply"),
    "generate-synthetic": ("synthetic", "generate"),
    "generate-synthetic-bulk": ("synthetic_bulk", "generate"),
    "audit-synthetic": ("synthetic_audit", "audit"),
    "publish-synthetic-version": ("synthetic_versioning", "publish"),
    "lock-a": ("synthetic_flow", "lock-a"),
    "verify-a-lock": ("synthetic_flow", "verify-a-lock"),
    "prepare-synthetic-reviewed": ("synthetic_flow", "prepare-reviewed"),
}


def _records(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _json_output(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _check_paths(inputs, outputs):
    sources = {Path(path).resolve() for path in inputs if path}
    targets = [Path(path).resolve() for path in outputs if path]
    if len(set(targets)) != len(targets):
        raise ValueError("Output paths must be distinct; sanitized output cannot share an internal sidecar")
    if sources & set(targets):
        raise ValueError("Output paths must not overwrite any input")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in COMMANDS:
        import importlib
        module, operation = COMMANDS[argv[0]]
        result = importlib.import_module("." + module, __package__).main(([operation] if operation else []) + argv[1:])
        return result if isinstance(result, int) else 0
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        sub.add_parser(name, help="Use this command with --help for its options")
    validate = sub.add_parser("validate", help="Validate canonical JSONL without printing source text")
    validate.add_argument("--input", required=True)
    detection = sub.add_parser("evaluate-detection")
    detection.add_argument("--gold", required=True)
    detection.add_argument("--pred", required=True)
    detection.add_argument("--output", required=True)
    masking = sub.add_parser("evaluate-masking")
    masking.add_argument("--gold", required=True)
    masking.add_argument("--masks", required=True)
    masking.add_argument("--output", required=True)
    mask = sub.add_parser("mask", help="Explicit all-candidates-MASK baseline, not a trained B")
    mask.add_argument("--input", required=True)
    mask.add_argument("--output", required=True, help="Sanitized JSONL with doc_id and masked_text only")
    mask.add_argument("--mask-records", help="Optional internal evaluation sidecar; contains ORIGINAL text")
    args = parser.parse_args(argv)
    if args.command == "evaluate-detection":
        _check_paths([args.gold, args.pred], [args.output])
    elif args.command == "evaluate-masking":
        _check_paths([args.gold, args.masks], [args.output])
    elif args.command == "mask":
        _check_paths([args.input], [args.output, args.mask_records])
    from .data import read_jsonl
    if args.command == "validate":
        docs = read_jsonl(args.input)
        ids = [doc.doc_id for doc in docs]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate doc_id in canonical JSONL")
        counts = Counter(m.entity_type for d in docs for m in d.mentions)
        print(json.dumps({"documents": len(docs), "mentions": sum(counts.values()), "entity_types": dict(sorted(counts.items()))}, ensure_ascii=False, indent=2))
    elif args.command == "evaluate-detection":
        from .evaluation import evaluate_detection
        result = evaluate_detection(read_jsonl(args.gold), read_jsonl(args.pred))
        _json_output(args.output, result)
        print(json.dumps(result["overall"], ensure_ascii=False, indent=2))
    elif args.command == "evaluate-masking":
        from .evaluation import evaluate_masking
        result = evaluate_masking(read_jsonl(args.gold), _records(args.masks))
        _json_output(args.output, result)
        print(json.dumps(result["overall"], ensure_ascii=False, indent=2))
    elif args.command == "mask":
        from .masking import intervals_from_mentions, render_masked_text
        from .stage_b import AllMaskDecision, apply_decisions
        documents = read_jsonl(args.input)
        sanitized, internal = [], []
        for document in documents:
            decided = apply_decisions(document, AllMaskDecision().decide(document))
            intervals = intervals_from_mentions(document.text, decided.mentions)
            sanitized.append({"doc_id": document.doc_id, "masked_text": render_masked_text(document.text, intervals)})
            internal.append({"doc_id": document.doc_id, "text": document.text, "intervals": [{"start": start, "end": end} for start, end in intervals]})
        for path, values in [(args.output, sanitized), (args.mask_records, internal)]:
            if path:
                target = Path(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("w", encoding="utf-8") as stream:
                    for value in values:
                        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        print(json.dumps({"documents": len(sanitized), "decision_model": "untrained_all_mask_baseline"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
