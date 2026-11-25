"""Export 100 stratified PENDING copies without altering the 3,000 originals."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from kpii_mvp.b_data import export_review
from kpii_mvp.b_inputs import FIXED_POLICY_ID
from kpii_mvp.data import read_jsonl, write_jsonl
from kpii_mvp.synthetic_v1 import _write_preview


def export_sample(input_dir, output_dir, seed=20261021, selection_manifest=None):
    source, target = Path(input_dir).resolve(), Path(output_dir).absolute()
    if target.exists() or target.is_symlink():
        raise ValueError("Use a fresh review sample directory")
    docs = read_jsonl(source / "proposed_documents.jsonl")
    plan = json.loads((source / "generation_plan.json").read_text(encoding="utf-8"))
    records = {row["doc_id"]: row for row in plan["documents"]}
    rng = random.Random(seed)
    groups = defaultdict(list)
    features, weights = {}, Counter()
    for doc in docs:
        record = records[doc.doc_id]
        groups[doc.split, record["family_id"], record["mode"]].append(doc)
        kinds = {m.mention_id: m.entity_type for m in doc.mentions}
        features[doc.doc_id] = {(doc.split, kinds[r["mention_id"]], r["quota_subrole"])
                               for r in record["relations"] if r["relation"] != "name"}
        weights.update(features[doc.doc_id])
    chosen, covered, used_clauses = [], set(), set()
    for split in ("train", "valid", "test"):
        family_ids = sorted({family for s, family, _ in groups if s == split})
        requests = []
        for i, family in enumerate(family_ids):
            quotas = ({"all_mask": 1, "mixed": 4, "all_keep": 1} if split == "train" else
                      {"all_mask": int(i % 2 == 0), "mixed": 1, "all_keep": int(i % 2 == 1)})
            requests.extend((family, mode) for mode, count in quotas.items() for _ in range(count))
        rng.shuffle(requests)
        for family, mode in requests:
            candidates = [d for d in groups[split, family, mode] if d.doc_id not in {x.doc_id for x in chosen}]
            rng.shuffle(candidates)
            def score(doc):
                unseen = features[doc.doc_id] - covered
                clauses = set(records[doc.doc_id]["clause_ids"]) - used_clauses
                return sum(100000 / weights[f] for f in unseen) + len(clauses) / 100
            doc = max(candidates, key=score)
            chosen.append(doc)
            covered.update(features[doc.doc_id])
            used_clauses.update(records[doc.doc_id]["clause_ids"])
    if selection_manifest:
        selected_ids = json.loads(Path(selection_manifest).read_text(encoding="utf-8"))["doc_ids"]
        by_id = {d.doc_id: d for d in docs}
        if len(selected_ids) != 100 or len(set(selected_ids)) != 100:
            raise ValueError("Selection manifest must contain exactly 100 distinct IDs")
        chosen = [by_id[doc_id] for doc_id in selected_ids]
        covered = set().union(*(features[d.doc_id] for d in chosen))
    chosen.sort(key=lambda d: (d.split, d.doc_id))
    expected = set(weights)
    assert len(chosen) == 100
    target.mkdir(parents=True)
    write_jsonl(target / "proposed_documents.jsonl", chosen)
    export_review(chosen, target / "review", policy_id=FIXED_POLICY_ID)
    _write_preview(target, chosen)
    manifest = {"kind": "AI and human review sample; does not approve training data", "documents": len(chosen),
                "parent_proposals_sha256": hashlib.sha256((source / "proposed_documents.jsonl").read_bytes()).hexdigest(),
                "split_counts": dict(Counter(d.split for d in chosen)),
                "contextual_mode_counts": dict(Counter(records[d.doc_id]["mode"] for d in chosen)),
                "type_subrole_split_cells_covered": len(covered), "total_cells": len(expected),
                "missing_cells": sorted(expected - covered), "doc_ids": [d.doc_id for d in chosen],
                "review_status": "PENDING", "training_executed": False}
    with (target / "sample_manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--selection-manifest", help="Keep reviewed document IDs while rechecking corrected text")
    args = parser.parse_args()
    result = export_sample(args.input_dir, args.output_dir, args.seed, args.selection_manifest)
    print(json.dumps({k:v for k,v in result.items() if k != "doc_ids"}, ensure_ascii=False, indent=2))
