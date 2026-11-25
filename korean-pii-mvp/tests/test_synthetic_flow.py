"""Synthetic review gates and frozen-A bindings, without model execution.

All APPROVED records below are artificial unit fixtures, never annotations
that the generation CLI may emit and never a training/evaluation corpus.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from kpii_mvp.b_data import document_fingerprint, pending_review
from kpii_mvp.b_inputs import FIXED_POLICY_ID
from kpii_mvp.data import read_jsonl, write_jsonl
from kpii_mvp.schema import Document, Mention
from kpii_mvp.synthetic_flow import create_a_lock, prepare_reviewed, verify_a_lock


def approved_fixture(document):
    return {**pending_review(document), "review_status": "APPROVED",
            "reviewer_id": "artificial_unit_fixture_reviewer",
            "reviewed_at": "2026-10-19T09:00:00+09:00",
            "candidate_inventory_complete": True, "action_labels_complete": True}


def document_fixture(split):
    # A single example includes both classes with the same approval flags.
    text = f"{split} 검수용 가상 문서. 한빛학교를 졸업했습니다. 한빛학교 행사 안내입니다."
    value = "한빛학교"
    first, second = text.index(value), text.rindex(value)
    return Document(
        f"fixture-{split}", text,
        [Mention("m1", first, first + len(value), value, "SCHOOL", "MASK", "PERSON_EDUCATION"),
         Mention("m2", second, second + len(value), value, "SCHOOL", "KEEP", "PUBLIC_ORGANIZATION")],
        f"fixture-group-{split}", ["SCHOOL"], source="synthetic_v1", split=split,
        policy_id=FIXED_POLICY_ID,
        trusted_context={"approved_public_orgs": [value], "document_type": "general_notice"})


class SyntheticFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.tmp.name)
        self.model = self.root / "a-checkpoint"
        self.model.mkdir()
        (self.model / "training_manifest.json").write_text(json.dumps({
            "stage": "A", "label2id": {"O": 0, "B-SCHOOL": 1, "I-SCHOOL": 2}}), encoding="utf-8")
        (self.model / "config.json").write_text("{}", encoding="utf-8")
        (self.model / "model.safetensors").write_bytes(b"tiny-unit-fixture-weights-not-a-model")
        (self.model / "tokenizer.json").write_text("{}", encoding="utf-8")
        self.lock = create_a_lock(self.model)
        self.lock_path = self.root / "a_checkpoint.lock.json"
        self.lock_path.write_text(json.dumps(self.lock), encoding="utf-8")
        self.documents_path = self.root / "documents.jsonl"
        self.reviews_path = self.root / "reviews.jsonl"
        self.output = self.root / "reviewed"
        self.documents = [document_fixture(split) for split in ("train", "valid", "test")]
        self.save_inputs()

    def tearDown(self):
        self.tmp.cleanup()

    def save_inputs(self, *, pending=False):
        write_jsonl(self.documents_path, self.documents)
        records = [(pending_review(document) if pending else approved_fixture(document))
                   for document in self.documents]
        self.reviews_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n"
                                                for record in records), encoding="utf-8")

    def prepare(self):
        return prepare_reviewed(self.documents_path, self.reviews_path, self.output,
                                self.model, self.lock_path)

    def test_lock_roundtrip_and_explicit_training_manifest_binding(self):
        digest = hashlib.sha256((self.model / "training_manifest.json").read_bytes()).hexdigest()
        pinned = create_a_lock(self.model, expected_manifest_sha256=digest)
        self.assertEqual(verify_a_lock(pinned, self.model)["status"], "verified")
        self.assertEqual(verify_a_lock(self.lock_path, self.model)["status"], "verified")
        with self.assertRaises(ValueError):
            create_a_lock(self.model, expected_manifest_sha256="0" * 64)

    def test_changed_config_weights_or_new_tokenizer_asset_invalidates_lock(self):
        for filename in ("config.json", "model.safetensors", "added_tokens.json"):
            with self.subTest(filename=filename):
                path = self.model / filename
                original = path.read_bytes() if path.exists() else None
                path.write_bytes(b"changed-artifact")
                with self.assertRaises(ValueError):
                    verify_a_lock(self.lock, self.model)
                if original is None:
                    path.unlink()
                else:
                    path.write_bytes(original)

    def test_changed_training_manifest_invalidates_lock(self):
        (self.model / "training_manifest.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            verify_a_lock(self.lock, self.model)

    def test_lock_is_bound_to_exact_checkpoint_path(self):
        copied = self.root / "different-a-checkpoint"
        copied.mkdir()
        for source in self.model.iterdir():
            (copied / source.name).write_bytes(source.read_bytes())
        with self.assertRaises(ValueError):
            verify_a_lock(self.lock, copied)

    def test_pending_reviews_are_not_promoted_and_create_no_output(self):
        self.save_inputs(pending=True)
        before = self.reviews_path.read_bytes()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.assertEqual(self.reviews_path.read_bytes(), before)
        self.assertIn('"PENDING"', before.decode("utf-8"))

    def test_approved_unit_fixtures_split_without_changing_fingerprints(self):
        result = self.prepare()
        self.assertEqual(result["documents"], 3)
        self.assertEqual(result["splits"], {"train": 1, "valid": 1, "test": 1})
        self.assertFalse(result["training_performed"])
        for document in self.documents:
            docs_file = self.output / f"{document.split}.jsonl"
            reviews_file = self.output / f"{document.split}.reviews.jsonl"
            self.assertTrue(docs_file.is_file())
            self.assertTrue(reviews_file.is_file())
            actual = read_jsonl(docs_file)
            self.assertEqual(len(actual), 1)
            self.assertEqual(document_fingerprint(actual[0]), document_fingerprint(document))
            review = json.loads(reviews_file.read_text(encoding="utf-8").strip())
            self.assertEqual(review["document_sha256"], document_fingerprint(document))
            self.assertEqual(review["review_status"], "APPROVED")
        self.assertEqual(verify_a_lock(self.output / "a_checkpoint.lock.json", self.model)["status"], "verified")

    def test_keep_without_trusted_approval_is_rejected_before_writing(self):
        self.documents[0] = replace(self.documents[0], trusted_context={"approved_public_orgs": []})
        self.save_inputs()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_cross_split_group_reuse_is_rejected_before_writing(self):
        self.documents[1] = replace(self.documents[1], source_group=self.documents[0].source_group)
        self.save_inputs()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_normalized_cross_split_copy_is_rejected(self):
        # Whitespace-only changes must not create an apparently unseen split.
        train = self.documents[0]
        copied = replace(train, doc_id="fixture-valid", source_group="different-group", split="valid",
                         text=train.text + " \n")
        self.documents[1] = copied
        self.save_inputs()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_split_without_contextual_keep_class_is_rejected(self):
        self.documents[2] = replace(self.documents[2], mentions=[
            replace(mention, action="MASK") for mention in self.documents[2].mentions])
        self.save_inputs()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_real_source_documents_are_not_accepted_as_synthetic(self):
        self.documents[0] = replace(self.documents[0], source="actual_internal_resume")
        self.save_inputs()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_existing_output_directory_is_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("reviewer content", encoding="utf-8")
        with self.assertRaises((ValueError, FileExistsError)):
            self.prepare()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "reviewer content")
        self.assertEqual(list(self.output.iterdir()), [sentinel])


if __name__ == "__main__":
    unittest.main()
