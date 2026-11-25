"""V4 data integrity and hard-context regressions, without ML or training."""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from kpii_mvp.b_data import load_reviews
from kpii_mvp.b_inputs import allowed_to_keep
from kpii_mvp.data import read_jsonl
from kpii_mvp.synthetic_v4 import (DEFAULT_SEED, GENERATOR_VERSION,
    FOCUS_CLAUSES, LEXICONS, audit_v4, build_proposals, generate)


class SyntheticV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents, cls.plan = build_proposals()

    def test_full_packet_counts_and_new_identity(self):
        self.assertEqual(Counter(d.split for d in self.documents), {"train": 2000, "valid": 500, "test": 500})
        self.assertEqual(sum(len(d.mentions) for d in self.documents), 16140)
        self.assertEqual(sum(m.entity_type != "NAME" for d in self.documents for m in d.mentions), 15540)
        self.assertTrue(all(d.source == GENERATOR_VERSION and d.doc_id.startswith(f"syn-v4-{DEFAULT_SEED}-") for d in self.documents))
        self.assertFalse(self.plan["human_reviewed"])
        self.assertIn("after_v3_error_analysis", self.plan["holdout_status"])

    def test_exact_offsets_approvals_and_no_answer_disclosure(self):
        for document in self.documents:
            document.validate()
            for mention in document.mentions:
                self.assertEqual(document.text[mention.start:mention.end], mention.text)
                self.assertNotIn("가상", mention.text)
                if mention.action == "KEEP":
                    self.assertTrue(allowed_to_keep(document, mention))
            for token in ("MASK", "KEEP", "정답", "개인정보이므로"):
                self.assertNotIn(token, document.text)

    def test_every_value_inventory_and_family_is_split_disjoint(self):
        owners, families = defaultdict(set), defaultdict(set)
        for document in self.documents:
            families[document.source_group].add(document.split)
            for mention in document.mentions:
                owners[mention.entity_type, mention.text.casefold()].add(document.split)
        self.assertTrue(all(len(splits) == 1 for splits in owners.values()))
        self.assertTrue(all(len(splits) == 1 for splits in families.values()))

    def test_contacts_are_safe_and_private_public_separate(self):
        roles = defaultdict(set)
        for document, record in zip(self.documents, self.plan["documents"]):
            by_id = {r["mention_id"]: r for r in record["relations"]}
            for mention in document.mentions:
                if mention.entity_type in {"PHONE", "EMAIL", "ADDRESS"}:
                    roles[mention.entity_type, mention.text].add(by_id[mention.mention_id]["relation"] == "private")
                if mention.entity_type == "PHONE":
                    self.assertRegex(mention.text, r"^000-\d{4}-\d{4}$")
                if mention.entity_type == "EMAIL":
                    self.assertTrue(mention.text.endswith(".example"))
        self.assertTrue(all(len(values) == 1 for values in roles.values()))

    def test_approved_personal_examples_are_mask_even_with_public_same_value(self):
        paired = Counter()
        for document in self.documents:
            by_value = defaultdict(set)
            for mention in document.mentions:
                if mention.entity_type in {"POSITION", "DEPARTMENT"}:
                    by_value[mention.entity_type, mention.text].add(mention.action)
            for (kind, _), actions in by_value.items():
                if actions == {"MASK", "KEEP"}:
                    paired[document.split, kind] += 1
        for split in ("train", "valid", "test"):
            for kind in ("POSITION", "DEPARTMENT"):
                self.assertGreater(paired[split, kind], 0)
        self.assertIn("선임연구원", LEXICONS["train"]["POSITION"])
        self.assertTrue(any("채용" in c for c in FOCUS_CLAUSES["train"]["POSITION"]["private"]))

    def test_audit_detects_cross_split_value_reuse(self):
        # Move an intact document and its family to a different split while
        # leaving the rest intact: copied values must be exposed explicitly.
        train = next(d for d in self.documents if d.split == "train")
        altered = replace(train, doc_id="copied-to-test", split="test", source_group="copied-family")
        plan = deepcopy(self.plan)
        record = deepcopy(next(r for r in plan["documents"] if r["doc_id"] == train.doc_id))
        record.update(doc_id=altered.doc_id, split="test", family_id=altered.source_group)
        plan["documents"].append(record)
        audit = audit_v4([*self.documents, altered], plan)
        self.assertIn("entity_value_crosses_splits", {b["code"] for b in audit["blockers"]})

    def test_full_audit_no_blockers_or_warnings(self):
        report = audit_v4(self.documents, self.plan)
        self.assertTrue(report["passed"])
        self.assertEqual(report["blockers"], [])
        self.assertEqual(report["warnings"], [])
        self.assertEqual(report["cross_split_entity_value_overlap_count"], 0)

    def test_packet_contains_pending_50_sample_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            target = Path(directory) / "v4"
            result = generate(target)
            self.assertFalse(result["training_executed"])
            sample = read_jsonl(target / "review_sample_50" / "documents.jsonl")
            self.assertEqual(len(sample), 50)
            self.assertEqual(Counter(d.split for d in sample), {"train": 20, "valid": 15, "test": 15})
            reviews = load_reviews(target / "review_sample_50" / "reviews.jsonl")
            self.assertTrue(all(r["review_status"] == "PENDING" for r in reviews.values()))
            manifest = json.loads((target / "generation_manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["human_reviewed"])
            self.assertIn("review_sample_50/selection.json", manifest["files_sha256"])
            before = (target / "generation_manifest.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "fresh version"):
                generate(target)
            self.assertEqual(before, (target / "generation_manifest.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
