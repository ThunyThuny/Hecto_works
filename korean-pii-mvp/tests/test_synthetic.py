"""Frozen V1 regression fixtures; V2 has separate core/audit/publishing tests."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kpii_mvp.b_data import load_reviews, validate_reviewed_documents, validate_split_separation
from kpii_mvp.b_inputs import CONTEXTUAL_TYPES, allowed_to_keep, canonical_type
from kpii_mvp.data import read_jsonl
from kpii_mvp.synthetic_v1 import GENERATOR_VERSION, SPLITS, build_proposals, generate


class SyntheticTests(unittest.TestCase):
    def test_korean_subject_particles_and_type_endings(self):
        docs, _ = build_proposals()
        for doc in docs:
            for wrong in ("김가람가", "이가온가", "최다온가", "정새봄가", "윤하람가", "강아름가"):
                self.assertNotIn(wrong, doc.text)
            for mention in doc.mentions:
                if mention.entity_type == "SCHOOL":
                    self.assertTrue(mention.text.endswith("대학교"))
                elif mention.entity_type == "DEPARTMENT":
                    self.assertTrue(mention.text.endswith("팀"))
                elif mention.entity_type == "POSITION":
                    self.assertTrue(mention.text.endswith("책임자"))

    def test_default_counts_and_exact_unicode_offsets(self):
        docs, plan = build_proposals()
        self.assertEqual(len(docs), 50)
        self.assertEqual(sum(len(d.mentions) for d in docs), 200)
        self.assertEqual(plan["split_counts"], {"train": 30, "valid": 10, "test": 10})
        for doc in docs:
            self.assertEqual(doc.source, GENERATOR_VERSION)
            self.assertEqual(len(doc.mentions), 4)
            self.assertEqual([m.action for m in doc.mentions].count("KEEP"), 1)
            ordered = sorted(doc.mentions, key=lambda m: m.start)
            for mention in ordered:
                self.assertEqual(doc.text[mention.start:mention.end], mention.text)
                self.assertNotEqual(canonical_type(mention.entity_type), "OTHER")
            self.assertTrue(all(a.end <= b.start for a, b in zip(ordered, ordered[1:])))
            self.assertEqual([m.action for m in doc.mentions if m.entity_type == "NAME"], ["MASK"])

    def test_same_approved_value_has_opposite_actions_with_exact_occurrences(self):
        docs, _ = build_proposals()
        for doc in docs:
            keep = next(m for m in doc.mentions if m.action == "KEEP")
            private = next(m for m in doc.mentions if m.text == keep.text and m.action == "MASK")
            self.assertNotEqual((keep.start, keep.end), (private.start, private.end))
            self.assertTrue(allowed_to_keep(doc, keep))
            self.assertTrue(allowed_to_keep(doc, private))  # gate alone cannot infer context
            unapproved = next(m for m in doc.mentions if m.reason_code == "PUBLIC_LOOKING_UNAPPROVED_VALUE")
            self.assertFalse(allowed_to_keep(doc, unapproved))
            self.assertEqual(unapproved.action, "MASK")

    def test_each_split_has_all_seven_types_and_disjoint_preallocated_families(self):
        docs, plan = build_proposals()
        partitions = {split: [d for d in docs if d.split == split] for split in SPLITS}
        validate_split_separation(partitions)
        family_sets = []
        for split in SPLITS:
            observed = {m.entity_type for d in partitions[split] for m in d.mentions}
            self.assertTrue(CONTEXTUAL_TYPES <= observed)
            families = {d.source_group for d in partitions[split]}
            self.assertEqual(len(families), 5)
            self.assertEqual(len(plan["family_allocation"][split]), 5)
            family_sets.append(families)
        for index, families in enumerate(family_sets):
            for other in family_sets[index + 1:]:
                self.assertFalse(families & other)

    def test_seed_reproducibility_and_variation(self):
        first, first_plan = build_proposals(seed=123)
        second, second_plan = build_proposals(seed=123)
        third, third_plan = build_proposals(seed=456)
        self.assertEqual([d.to_dict() for d in first], [d.to_dict() for d in second])
        self.assertEqual(first_plan, second_plan)
        self.assertNotEqual(first_plan["family_allocation"], third_plan["family_allocation"])
        self.assertNotEqual([d.text for d in first], [d.text for d in third])

    def test_safe_fictional_contact_values(self):
        docs, _ = build_proposals()
        for doc in docs:
            for mention in doc.mentions:
                if mention.entity_type == "EMAIL":
                    self.assertTrue(mention.text.endswith(".example"))
                if mention.entity_type == "PHONE":
                    self.assertTrue(mention.text.startswith("000-0000-"))
                if mention.entity_type in {"WORKPLACE", "SCHOOL", "ADDRESS", "DEPARTMENT", "POSITION"}:
                    self.assertTrue(mention.text.startswith("가상"))

    def test_published_review_packet_pending_and_manifest_bound(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp) / "packet"
            summary = generate(root)
            self.assertFalse(summary["training_executed"])
            manifest = json.loads((root / "generation_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["documents"], 50)
            self.assertEqual(manifest["proposed_action_counts"], {"MASK": 150, "KEEP": 50})
            self.assertEqual(manifest["review_status"], "PENDING")
            self.assertIn("SIMULATED", manifest["metadata_provenance"])
            for name, digest in manifest["files_sha256"].items():
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest)
            docs = read_jsonl(root / "review" / "documents.jsonl")
            self.assertEqual([d.to_dict() for d in docs], [d.to_dict() for d in read_jsonl(root / "proposed_documents.jsonl")])
            reviews = load_reviews(root / "review" / "reviews.jsonl")
            self.assertTrue(all(r["review_status"] == "PENDING" and r["reviewer_id"] is None
                                and r["reviewed_at"] is None and not r["candidate_inventory_complete"]
                                and not r["action_labels_complete"] for r in reviews.values()))
            with self.assertRaisesRegex(ValueError, "PENDING"):
                validate_reviewed_documents(docs, reviews)
            self.assertEqual(sum(len(read_jsonl(root / "proposals" / f"{s}.jsonl")) for s in SPLITS), 50)
            preview = (root / "review.html").read_text(encoding="utf-8")
            self.assertIn('class="mask"', preview)
            self.assertIn('class="keep"', preview)
            self.assertNotIn("<script", preview)

    def test_all_packet_bytes_reproducible_without_output_path_or_time(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            first, second = Path(temp) / "first", Path(temp) / "second"
            generate(first, seed=99, train_count=7, valid_count=7, test_count=7)
            generate(second, seed=99, train_count=7, valid_count=7, test_count=7)
            for file in first.rglob("*"):
                if file.is_file():
                    self.assertEqual(file.read_bytes(), (second / file.relative_to(first)).read_bytes())

    def test_reject_existing_data_and_keep_sentinel(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp) / "existing"
            root.mkdir()
            sentinel = root / "human_review.csv"
            sentinel.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nonempty"):
                generate(root)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
            self.assertEqual(list(root.iterdir()), [sentinel])

    def test_accept_empty_directory_and_clean_failed_staging(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp) / "empty"
            root.mkdir()
            with patch("kpii_mvp.synthetic_v1.export_review", side_effect=RuntimeError("test failure")):
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    generate(root)
            self.assertEqual(list(root.iterdir()), [])
            self.assertFalse(list(Path(temp).glob(".*.staging-*")))
            generate(root, train_count=7, valid_count=7, test_count=7)
            self.assertTrue((root / "generation_manifest.json").is_file())

    def test_invalid_counts_rejected_before_files(self):
        for kwargs in ({"train_count": 6}, {"valid_count": -1}, {"test_count": True},
                       {"train_count": 4000}, {"seed": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_proposals(**kwargs)


if __name__ == "__main__":
    unittest.main()
