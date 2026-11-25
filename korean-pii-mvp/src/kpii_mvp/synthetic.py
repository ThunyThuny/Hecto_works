"""Publish V2 fictional proposals and PENDING review packets; never train."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .b_data import export_review
from .b_inputs import CONTEXTUAL_TYPES, FIXED_POLICY_ID, policy_sha256
from .data import write_jsonl
from .synthetic_v1 import _write_preview
from .synthetic_v2 import GENERATOR_VERSION, SPLITS, TEMPLATE_SPEC, TYPES, build_proposals


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   indent=2, allow_nan=False) + "\n", encoding="utf-8")


def generate(output_dir, *, seed=20261020, train_count=30, valid_count=10, test_count=10):
    """Validate before publishing; existing human review files are never replaced."""
    from .synthetic_audit import audit_corpus
    destination = Path(output_dir).absolute()
    if destination.is_symlink():
        raise ValueError("output directory must not be a symbolic link")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("output directory is nonempty; use a new version directory")
    docs, plan = build_proposals(seed=seed, train_count=train_count,
                                valid_count=valid_count, test_count=test_count)
    audit = audit_corpus(docs, plan)
    if audit["blockers"]:
        raise ValueError("Generated corpus failed semantic audit: " + json.dumps(audit["blockers"], ensure_ascii=False))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        write_jsonl(temporary / "proposed_documents.jsonl", docs)
        for split in SPLITS:
            write_jsonl(temporary / "proposals" / f"{split}.jsonl", [d for d in docs if d.split == split])
        _dump_json(temporary / "generation_plan.json", plan)
        _dump_json(temporary / "automated_audit.json", audit)
        export_review(docs, temporary / "review", policy_id=FIXED_POLICY_ID)
        _write_preview(temporary, docs)
        sources = {name: _sha(Path(__file__).with_name(name))
                   for name in ("synthetic.py", "synthetic_v2.py", "synthetic_audit.py", "synthetic_v1.py")}
        files = {p.relative_to(temporary).as_posix(): _sha(p)
                 for p in sorted(temporary.rglob("*")) if p.is_file()}
        manifest = {
            "generator_version": GENERATOR_VERSION, "seed": seed,
            "policy_id": FIXED_POLICY_ID, "policy_sha256": policy_sha256(),
            "generator_code_sha256": _sha(Path(__file__)), "generator_sources_sha256": sources,
            "templates_sha256": hashlib.sha256(json.dumps(TEMPLATE_SPEC, ensure_ascii=False,
                sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "documents": len(docs), "mentions": sum(len(d.mentions) for d in docs),
            "proposed_action_counts": dict(Counter(m.action for d in docs for m in d.mentions)),
            "split_counts": plan["split_counts"],
            "candidate_count_histogram": dict(Counter(len(d.mentions) for d in docs)),
            "contextual_type_counts": {split: dict(Counter(m.entity_type for d in docs if d.split == split
                for m in d.mentions if m.entity_type in CONTEXTUAL_TYPES)) for split in SPLITS},
            "review_status": "PENDING", "training_executed": False,
            "metadata_provenance": "SIMULATED fictional policy facts; no external verification",
            "audit_kind": "automated semantic checks, not human approval",
            "limitations": plan["limitations"], "files_sha256": files,
        }
        _dump_json(temporary / "generation_manifest.json", manifest)
        if destination.is_symlink():
            raise ValueError("destination became a symbolic link")
        if destination.exists():
            destination.rmdir()
        temporary.rename(destination)
    finally:
        if temporary.exists():
            assert temporary.resolve().parent == destination.parent.resolve()
            assert temporary.name.startswith(f".{destination.name}.staging-")
            shutil.rmtree(temporary)
    return {"output_dir": str(destination), "generator_version": GENERATOR_VERSION,
            "documents": len(docs), "mentions": manifest["mentions"],
            "review_status": "PENDING", "audit_blockers": 0, "training_executed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("generate", help="V2 proposals only; never approves or trains")
    command.add_argument("--output-dir", required=True)
    command.add_argument("--seed", type=int, default=20261020)
    command.add_argument("--train-count", type=int, default=30)
    command.add_argument("--valid-count", type=int, default=10)
    command.add_argument("--test-count", type=int, default=10)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.output_dir, seed=args.seed, train_count=args.train_count,
        valid_count=args.valid_count, test_count=args.test_count), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
