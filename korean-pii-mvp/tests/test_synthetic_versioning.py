"""Publication preserves actual review edits and never trains or refreshes A."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kpii_mvp.b_data import export_review
from kpii_mvp.data import write_jsonl
from kpii_mvp.schema import Document, Mention
from kpii_mvp.synthetic_versioning import publish_version


class SyntheticVersioningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.temp.name).resolve()
        self.old = self.root / "current"
        self.new = self.root / "staged"
        self.archive = self.root / "archives" / "v1"
        self.old.mkdir()
        self.new.mkdir()
        self.source = self.root / "legacy.py"
        self.source.write_bytes(b"# legacy generator\n")
        self._dump(self.old / "generation_manifest.json", {
            "generator_code_sha256": hashlib.sha256(self.source.read_bytes()).hexdigest()})
        (self.old / "human-edits.csv").write_bytes(b"keep exactly these human edits\r\n")
        lock = {"stage": "A", "checkpoint_artifacts": {"model.safetensors": "fixed"}}
        self._dump(self.old / "a_checkpoint.lock.json", lock)
        self._dump(self.new / "a_checkpoint.lock.json", lock)
        doc = Document(doc_id="small-1", text="이름 김민수", mentions=[
            Mention("m1", 3, 6, "김민수", "NAME", "MASK", "PERSON_NAME")],
            source_group="small-1", annotation_scope=["NAME"], source="synthetic",
            split="train", policy_id="career_external_share_v1")
        write_jsonl(self.new / "proposed_documents.jsonl", [doc])
        export_review([doc], self.new / "review", policy_id=doc.policy_id)
        self._dump(self.new / "generation_plan.json", {"documents": [{
            "doc_id": doc.doc_id, "split": "train", "relations": [
                {"mention_id": "m1", "relation": "name", "proposed_action": "MASK"}]}]})
        self._dump(self.new / "automated_audit.json", {
            "passed": True, "documents": 1, "validated_documents": 1, "blockers": []})
        self._refresh_manifest()

    def tearDown(self):
        self.temp.cleanup()

    def _dump(self, path, data):
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _refresh_manifest(self):
        hashes = {path.relative_to(self.new).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in self.new.rglob("*") if path.is_file() and path.name != "generation_manifest.json"}
        self._dump(self.new / "generation_manifest.json", {
            "generator_version": "synthetic_rule_based_v2", "documents": 1,
            "review_status": "PENDING", "training_executed": False, "files_sha256": hashes})

    def test_archive_preserves_actual_files_and_optional_generator(self):
        before = {p.relative_to(self.old).as_posix(): p.read_bytes()
                  for p in self.old.rglob("*") if p.is_file()}
        result = publish_version(self.new, self.old, self.archive, self.source)
        self.assertFalse(result["training_executed"])
        self.assertFalse(self.new.exists())
        for name, original in before.items():
            self.assertEqual((self.archive / name).read_bytes(), original)
        self.assertEqual((self.archive / "generator_source_v1.py").read_bytes(), self.source.read_bytes())
        record = json.loads((self.archive / "archive_record.json").read_text(encoding="utf-8"))
        self.assertEqual(set(record["original_files_sha256"]), set(before))
        self.assertTrue((self.old / "review" / "reviews.jsonl").exists())

    def test_existing_archive_is_never_overwritten(self):
        self.archive.mkdir(parents=True)
        marker = self.archive / "do-not-overwrite"
        marker.write_bytes(b"existing")
        with self.assertRaisesRegex(ValueError, "already exists"):
            publish_version(self.new, self.old, self.archive)
        self.assertEqual(marker.read_bytes(), b"existing")
        self.assertTrue((self.old / "human-edits.csv").exists())

    def test_invalid_nested_outside_and_symlink_paths(self):
        with self.assertRaisesRegex(ValueError, "nested"):
            publish_version(self.new, self.old, self.old / "archive")
        with self.assertRaisesRegex(ValueError, "inside"):
            publish_version(self.new, self.old, self.root.parent / "outside-archive")
        symlink = self.root / "alias"
        try:
            symlink.symlink_to(self.new, target_is_directory=True)
        except (OSError, NotImplementedError):
            return  # Windows accounts may lack CreateSymbolicLink privileges.
        with self.assertRaisesRegex(ValueError, "symlinks"):
            publish_version(symlink, self.old, self.archive)

    def test_failed_second_rename_rolls_old_packet_back(self):
        real_rename = Path.rename
        def fail_staged(path, target):
            if path == self.new and Path(target) == self.old:
                raise OSError("simulated publication failure")
            return real_rename(path, target)
        with patch.object(Path, "rename", new=fail_staged):
            with self.assertRaisesRegex(OSError, "simulated"):
                publish_version(self.new, self.old, self.archive)
        self.assertTrue((self.old / "human-edits.csv").exists())
        self.assertTrue(self.new.is_dir())
        self.assertFalse(self.archive.exists())

    def test_approved_review_or_changed_a_lock_prevents_moves(self):
        review_path = self.new / "review" / "reviews.jsonl"
        record = json.loads(review_path.read_text(encoding="utf-8"))
        record.update(review_status="APPROVED", reviewer_id="reviewer", reviewed_at="2026-10-19T00:00:00+00:00",
                      candidate_inventory_complete=True, action_labels_complete=True)
        review_path.write_text(json.dumps(record), encoding="utf-8")
        self._refresh_manifest()
        with self.assertRaisesRegex(ValueError, "PENDING"):
            publish_version(self.new, self.old, self.archive)
        self.assertTrue((self.old / "human-edits.csv").exists())
        record.update(review_status="PENDING", reviewer_id=None, reviewed_at=None,
                      candidate_inventory_complete=False, action_labels_complete=False)
        review_path.write_text(json.dumps(record), encoding="utf-8")
        self._dump(self.new / "a_checkpoint.lock.json", {"stage": "A", "changed": True})
        self._refresh_manifest()
        with self.assertRaisesRegex(ValueError, "lock differs"):
            publish_version(self.new, self.old, self.archive)

    def test_blocked_audit_or_modified_generated_file_prevents_moves(self):
        audit = self.new / "automated_audit.json"
        self._dump(audit, {"passed": False, "documents": 1, "validated_documents": 1,
                           "blockers": [{"code": "private_contact_retained"}]})
        self._refresh_manifest()
        with self.assertRaisesRegex(ValueError, "passing audit"):
            publish_version(self.new, self.old, self.archive)
        self._dump(audit, {"passed": True, "documents": 1, "validated_documents": 1, "blockers": []})
        self._refresh_manifest()
        (self.new / "review" / "mentions.csv").write_bytes(b"changed after generation")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            publish_version(self.new, self.old, self.archive)

    def test_stale_passing_report_is_recomputed_from_current_plan(self):
        plan_path = self.new / "generation_plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan["documents"][0]["relations"][0]["relation"] = "unknown-role"
        self._dump(plan_path, plan)
        # An edited plan and regenerated manifest hashes can still accompany a
        # stale passing audit; semantic checking must inspect the current plan.
        self._refresh_manifest()
        with self.assertRaisesRegex(ValueError, "recomputed semantic audit"):
            publish_version(self.new, self.old, self.archive)
        self.assertTrue(self.old.is_dir())
        self.assertTrue(self.new.is_dir())
        self.assertFalse(self.archive.exists())

    def test_reserved_empty_directory_is_rejected_before_moving(self):
        (self.old / "archive_record.json").mkdir()
        with self.assertRaisesRegex(ValueError, "record names"):
            publish_version(self.new, self.old, self.archive)
        self.assertTrue((self.old / "archive_record.json").is_dir())
        self.assertTrue(self.new.is_dir())
        self.assertFalse(self.archive.exists())

    def test_changed_replacement_after_rename_causes_rollback(self):
        real_rename = Path.rename
        def modify_after_publish(path, target):
            result = real_rename(path, target)
            if path == self.new and Path(target) == self.old:
                (self.old / "injected-after-publish.txt").write_bytes(b"changed")
            return result
        with patch.object(Path, "rename", new=modify_after_publish):
            with self.assertRaisesRegex(RuntimeError, "replacement files changed"):
                publish_version(self.new, self.old, self.archive)
        self.assertTrue((self.old / "human-edits.csv").exists())
        self.assertFalse((self.old / "injected-after-publish.txt").exists())
        self.assertTrue((self.new / "injected-after-publish.txt").exists())
        self.assertFalse(self.archive.exists())

    def test_partial_archive_record_failure_removes_only_new_metadata(self):
        before = {p.relative_to(self.old).as_posix(): p.read_bytes()
                  for p in self.old.rglob("*") if p.is_file()}
        real_open = Path.open
        class PartialWriter:
            def __init__(self, file):
                self.file = file
            def __enter__(self):
                return self
            def write(self, value):
                self.file.write(value[:10])
                self.file.flush()
                raise OSError("simulated partial archive record write")
            def __exit__(self, *args):
                self.file.close()
        def fail_record_write(path, mode="r", *args, **kwargs):
            stream = real_open(path, mode, *args, **kwargs)
            return PartialWriter(stream) if path.name == "archive_record.json" and mode == "x" else stream
        with patch.object(Path, "open", new=fail_record_write):
            with self.assertRaisesRegex(OSError, "partial archive record"):
                publish_version(self.new, self.old, self.archive, self.source)
        after = {p.relative_to(self.old).as_posix(): p.read_bytes()
                 for p in self.old.rglob("*") if p.is_file()}
        self.assertEqual(after, before)
        self.assertTrue(self.new.is_dir())
        self.assertFalse(self.archive.exists())


if __name__ == "__main__":
    unittest.main()
