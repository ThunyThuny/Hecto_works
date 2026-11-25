"""Unreviewed synthetic opt-in stays bound to the original fictional packet."""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kpii_mvp.b_data import load_reviews, pending_review
from kpii_mvp.b_experiment import (EXPERIMENT_KIND, validate_b_evidence,
                                  validate_experiment_packet)
from kpii_mvp.data import read_jsonl
from kpii_mvp.synthetic_bulk import generate


class SyntheticExperimentEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        cls.packet = Path(cls.temporary.name) / "packet"
        generate(cls.packet)
        cls.docs = read_jsonl(cls.packet / "proposed_documents.jsonl")
        cls.reviews = load_reviews(cls.packet / "review" / "reviews.jsonl")
        cls.doc = cls.docs[0]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @contextmanager
    def changed_file(self, relative, transform):
        target = self.packet / relative
        old = target.read_bytes()
        try:
            target.write_bytes(transform(old))
            yield target
        finally:
            target.write_bytes(old)

    @contextmanager
    def changed_manifest(self, transform):
        def update(old):
            record = json.loads(old)
            transform(record)
            return json.dumps(record, ensure_ascii=False).encode()
        with self.changed_file("generation_manifest.json", update):
            yield

    def evidence(self, docs=None, reviews=None, **kwargs):
        docs = docs if docs is not None else [self.doc]
        reviews = reviews if reviews is not None else {d.doc_id: self.reviews[d.doc_id] for d in docs}
        return validate_b_evidence(docs, reviews, synthetic_experiment_dir=self.packet, **kwargs)

    def test_original_packet_is_explicitly_unreviewed_and_hash_bound(self):
        report = validate_experiment_packet(self.packet)
        self.assertEqual(report["data_kind"], EXPERIMENT_KIND)
        self.assertFalse(report["human_reviewed"])
        self.assertEqual(report["review_status"], "PENDING")
        self.assertEqual(report["documents"], 3000)
        self.assertEqual(report["contextual_candidates"], 15540)
        self.assertEqual(report["packet_manifest_sha256"], hashlib.sha256(
            (self.packet / "generation_manifest.json").read_bytes()).hexdigest())
        self.assertFalse(report["a_lock_verified"])

    def test_exact_train_subset_is_allowed_without_approving_records(self):
        docs = [d for d in self.docs if d.split == "train"][:3]
        report = self.evidence(docs, expected_split="train")
        self.assertEqual(report["documents"], 3)
        self.assertEqual(report["source_provenance"]["classification"], "synthetic")
        self.assertEqual(report["review_evidence"], EXPERIMENT_KIND)
        self.assertFalse(report["human_reviewed"])
        self.assertTrue(all(record == pending_review(doc) for doc, record in
            ((d, self.reviews[d.doc_id]) for d in self.docs)))

    def test_default_pending_gate_is_unchanged(self):
        with self.assertRaisesRegex(ValueError, "PENDING"):
            validate_b_evidence([self.doc], {self.doc.doc_id: self.reviews[self.doc.doc_id]})

    def test_default_approved_gate_is_delegated_exactly(self):
        report = {"review_evidence": "ordinary_approval"}
        with patch("kpii_mvp.b_experiment.validate_reviewed_documents", return_value=report) as validate:
            self.assertIs(validate_b_evidence([], {}, expected_split="train"), report)
        validate.assert_called_once_with([], {}, expected_split="train", policy_id="career_external_share_v1")

    def test_edited_gold_rebound_to_pending_is_still_rejected(self):
        first = self.doc.mentions[0]
        changed = replace(first, action="KEEP" if first.action == "MASK" else "MASK")
        doc = replace(self.doc, mentions=[changed, *self.doc.mentions[1:]])
        with self.assertRaisesRegex(ValueError, "original hash-bound packet"):
            self.evidence([doc], {doc.doc_id: pending_review(doc)})

    def test_relabelled_split_and_extra_document_are_rejected(self):
        doc = replace(self.doc, split="test", doc_id=self.doc.doc_id + "-copy")
        with self.assertRaisesRegex(ValueError, "original hash-bound packet"):
            self.evidence([doc], {doc.doc_id: pending_review(doc)})

    def test_real_mixed_unknown_sources_cannot_use_opt_in(self):
        for source in ("real", "real_career", "mixed", "unknown", "synthetic_other"):
            with self.subTest(source=source):
                doc = replace(self.doc, source=source)
                with self.assertRaisesRegex(ValueError, "cannot relax"):
                    self.evidence([doc], {doc.doc_id: pending_review(doc)})

    def test_wrong_policy_split_and_unlabelled_actions_fail(self):
        with self.assertRaisesRegex(ValueError, "fixed-policy"):
            self.evidence(policy_id="other")
        with self.assertRaisesRegex(ValueError, "expected split"):
            self.evidence(expected_split="test")
        doc = replace(self.doc, mentions=[replace(self.doc.mentions[0], action=None), *self.doc.mentions[1:]])
        with self.assertRaisesRegex(ValueError, "explicit MASK/KEEP"):
            self.evidence([doc], {doc.doc_id: pending_review(doc)})

    def test_fake_approval_and_changed_notes_are_not_silently_consumed(self):
        for key, value in (("review_status", "APPROVED"), ("reviewer_id", "fake"),
                           ("reviewed_at", "2026-10-19T10:00:00+09:00"),
                           ("candidate_inventory_complete", True), ("action_labels_complete", True),
                           ("notes", "human work begun")):
            with self.subTest(key=key):
                record = dict(self.reviews[self.doc.doc_id], **{key: value})
                with self.assertRaisesRegex(ValueError, "original PENDING"):
                    self.evidence(reviews={self.doc.doc_id: record})

    def test_missing_or_extra_review_inventory_fails(self):
        with self.assertRaisesRegex(ValueError, "match submitted documents"):
            self.evidence(reviews={})
        with self.assertRaisesRegex(ValueError, "match submitted documents"):
            self.evidence(reviews=self.reviews)

    def test_modified_packet_bytes_fail_even_if_json_is_valid(self):
        with self.changed_file("proposed_documents.jsonl", lambda old: old + b"\n"):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_experiment_packet(self.packet)

    def test_required_hashes_and_unsafe_paths_are_rejected(self):
        with self.changed_manifest(lambda m: m["files_sha256"].pop("review/reviews.jsonl")):
            with self.assertRaisesRegex(ValueError, "bind the full"):
                validate_experiment_packet(self.packet)
        with self.changed_manifest(lambda m: m["files_sha256"].update({"../escape": "0" * 64})):
            with self.assertRaisesRegex(ValueError, "manifest path"):
                validate_experiment_packet(self.packet)

    def test_manifest_generator_policy_or_pending_state_cannot_be_relaxed(self):
        for key, value in (("generator_version", "real"), ("policy_sha256", "0" * 64),
                           ("review_status", "APPROVED"), ("training_executed", True)):
            with self.subTest(key=key), self.changed_manifest(lambda m: m.update({key: value})):
                with self.assertRaises(ValueError):
                    validate_experiment_packet(self.packet)

    def test_recomputed_audit_is_required_even_when_original_report_passes(self):
        with patch("kpii_mvp.synthetic_bulk_audit.audit_bulk", return_value={
                "passed": False, "blockers": [{"code": "current_error"}], "warnings": []}):
            with self.assertRaisesRegex(ValueError, "recomputed bulk audit"):
                validate_experiment_packet(self.packet)

    def test_rehashed_review_changes_still_require_original_pending_declarations(self):
        path = "review/reviews.jsonl"
        def update(old):
            lines = old.decode().splitlines()
            record = json.loads(lines[0])
            record["reviewer_id"] = "someone"
            lines[0] = json.dumps(record, ensure_ascii=False)
            return ("\n".join(lines) + "\n").encode()
        with self.changed_file(path, update) as target:
            with self.changed_manifest(lambda m: m["files_sha256"].update({path: hashlib.sha256(target.read_bytes()).hexdigest()})):
                with self.assertRaisesRegex(ValueError, "original PENDING"):
                    validate_experiment_packet(self.packet)


if __name__ == "__main__":
    unittest.main()
