import json
from pathlib import Path
import tempfile
import unittest

from kpii_mvp.b_pipeline import a_inventory, prepare_a_documents
from kpii_mvp.schema import Document, Mention


class PipelineTests(unittest.TestCase):
    def test_scan_removes_all_gold_but_preserves_evidence(self):
        document = Document("d", "가람", [Mention("m", 0, 2, "가람", "NAME", "MASK", "private")],
                            "g", ["NAME"], policy_id="career_external_share_v1",
                            split="test", trusted_context={"document_type": "resume"})
        scanned = prepare_a_documents([document], ["PS_NAME", "QT_PHONE"])[0]
        self.assertEqual(scanned.mentions, [])
        self.assertEqual(scanned.annotation_scope, ["PS_NAME", "QT_PHONE"])
        self.assertEqual(scanned.text, document.text)
        self.assertEqual(scanned.trusted_context, document.trusted_context)
        self.assertEqual(document.mentions[0].action, "MASK")
        self.assertEqual(document.annotation_scope, ["NAME"])

    def test_inventory_requires_complete_bio(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "training_manifest.json"
            path.write_text(json.dumps({"stage": "A", "label2id": {"O": 0, "B-PS_NAME": 1, "I-PS_NAME": 2}}))
            self.assertEqual(a_inventory(directory), ["PS_NAME"])
            path.write_text(json.dumps({"stage": "A", "label2id": {"O": 0, "B-PS_NAME": 1}}))
            with self.assertRaises(ValueError):
                a_inventory(directory)
            for labels in ({"O": 0, "B-": 1, "I-": 2},
                           {"O": 0, "B-NAME": "1", "I-NAME": 2}):
                path.write_text(json.dumps({"stage": "A", "label2id": labels}))
                with self.assertRaises(ValueError):
                    a_inventory(directory)

    def test_missing_policy_rejected(self):
        with self.assertRaises(ValueError):
            prepare_a_documents([Document("d", "가", [], "g", ["NAME"])], ["PS_NAME"])


if __name__ == "__main__":
    unittest.main()
