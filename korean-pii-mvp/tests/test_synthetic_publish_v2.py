"""V2 publication gates and preservation of PENDING review artifacts."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kpii_mvp.b_data import load_reviews, validate_reviewed_documents
from kpii_mvp.data import read_jsonl
from kpii_mvp.synthetic import generate


class SyntheticV2PublishTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_pending_packet_and_all_hashes_bound(self):
        target = self.root / "packet"
        summary = generate(target)
        self.assertEqual(summary["documents"], 50)
        self.assertFalse(summary["training_executed"])
        manifest = json.loads((target / "generation_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["generator_version"], "synthetic_rule_based_v2")
        for name, digest in manifest["files_sha256"].items():
            self.assertEqual(hashlib.sha256((target / name).read_bytes()).hexdigest(), digest)
        audit = json.loads((target / "automated_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["blockers"], [])
        docs = read_jsonl(target / "review/documents.jsonl")
        reviews = load_reviews(target / "review/reviews.jsonl")
        self.assertTrue(all(row["review_status"] == "PENDING" and row["reviewer_id"] is None for row in reviews.values()))
        with self.assertRaisesRegex(ValueError, "PENDING"):
            validate_reviewed_documents(docs, reviews)
        self.assertGreater(len({len(d.mentions) for d in docs}), 2)
        self.assertTrue((target / "review.html").is_file())

    def test_seed_reproduces_every_published_byte(self):
        first, second = self.root / "a", self.root / "b"
        generate(first, seed=20261020)
        generate(second, seed=20261020)
        for file in first.rglob("*"):
            if file.is_file():
                self.assertEqual(file.read_bytes(), (second / file.relative_to(first)).read_bytes())

    def test_existing_review_is_never_overwritten(self):
        target = self.root / "human-review"
        target.mkdir()
        sentinel = target / "mentions.csv"
        sentinel.write_text("review in progress", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            generate(target)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "review in progress")

    def test_semantic_blocker_prevents_any_publication(self):
        target = self.root / "blocked"
        with patch("kpii_mvp.synthetic_audit.audit_corpus", return_value={"blockers": [{"code": "private_contact_retained"}]}):
            with self.assertRaisesRegex(ValueError, "semantic audit"):
                generate(target)
        self.assertFalse(target.exists())

    def test_export_failure_leaves_empty_input_directory_and_no_staging(self):
        target = self.root / "empty"
        target.mkdir()
        with patch("kpii_mvp.synthetic.export_review", side_effect=RuntimeError("export failed")):
            with self.assertRaisesRegex(RuntimeError, "export failed"):
                generate(target)
        self.assertEqual(list(target.iterdir()), [])
        self.assertEqual(list(self.root.glob(".*.staging-*")), [])


if __name__ == "__main__":
    unittest.main()
