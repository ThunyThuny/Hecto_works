"""Check B inputs using a saved tokenizer only; never load model weights."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

from .b_inputs import FOCUSED_INPUT_VERSION, MARKERS, prepare_candidate_input
from .data import read_jsonl


def audit_inputs(tokenizer, splits, *, input_version=FOCUSED_INPUT_VERSION, max_length=512):
    if type(max_length) is not int or max_length < 8:
        raise ValueError("max_length must be an integer >= 8")
    reports, failures = {}, []
    for split, documents in splits.items():
        counts = Counter()
        longest = 0
        for document in documents:
            for mention in document.mentions:
                prepared = prepare_candidate_input(tokenizer, document, mention, 100000, input_version=input_version)
                if prepared["reason"] == "policy_always_mask":
                    counts["policy_always_mask"] += 1
                    continue
                counts["contextual_candidates"] += 1
                length = len(prepared["model_inputs"]["input_ids"]) if prepared["available"] else 0
                longest = max(longest, length)
                if not prepared["available"] or length > max_length:
                    counts["unavailable_or_too_long"] += 1
                    failures.append({"split": split, "doc_id": document.doc_id, "mention_id": mention.mention_id,
                                     "tokens": length, "reason": prepared["reason"] if not prepared["available"] else "context_exceeds_max_length"})
        reports[split] = {**counts, "max_observed_tokens": longest}
    return {"schema_version": "kpii-b-input-audit-1", "passed": not failures, "input_version": input_version,
            "max_length": max_length, "splits": reports, "failures": failures,
            "tokenizer_only": True, "model_weights_loaded": False, "training_performed": False}


def load_tokenizer(tokenizer_dir):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True, local_files_only=True)
    tokenizer.add_special_tokens({"additional_special_tokens": list(MARKERS)})
    return tokenizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["audit"])
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--input-version", default=FOCUSED_INPUT_VERSION)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    target = Path(args.output) if args.output else None
    if target and (target.exists() or target.resolve().is_relative_to(Path(args.packet_dir).resolve())
                   or target.resolve().is_relative_to(Path(args.tokenizer_dir).resolve())):
        raise ValueError("Input audit report must be new and outside the packet/checkpoint")
    splits = {s: read_jsonl(Path(args.packet_dir) / "proposals" / (s + ".jsonl")) for s in ("train", "valid", "test")}
    report = audit_inputs(load_tokenizer(args.tokenizer_dir), splits, input_version=args.input_version, max_length=args.max_length)
    if target:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
