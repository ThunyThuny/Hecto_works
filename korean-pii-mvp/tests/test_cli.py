import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from kpii_mvp.cli import main


class CliIntegrationTests(unittest.TestCase):
    def fixture(self, root):
        row = {
            "doc_id": "fixture", "text": "김가람은 한빛대학교 졸업생입니다.",
            "source_group": "fixture-group", "split": "test",
            "annotation_scope": ["NAME", "SCHOOL"],
            "mentions": [
                {"mention_id": "m1", "start": 0, "end": 3, "text": "김가람", "entity_type": "NAME", "action": "MASK"},
                {"mention_id": "m2", "start": 5, "end": 10, "text": "한빛대학교", "entity_type": "SCHOOL", "action": "KEEP"},
            ],
        }
        path = root / "gold.jsonl"
        path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    def test_sanitized_output_and_separate_mask_evaluation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            gold = self.fixture(root)
            output, sidecar, report = root / "masked.jsonl", root / "masks.jsonl", root / "report.json"
            self.assertEqual(main(["mask", "--input", str(gold), "--output", str(output), "--mask-records", str(sidecar)]), 0)
            sanitized = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(set(sanitized), {"doc_id", "masked_text"})
            self.assertNotIn("김가람", sanitized["masked_text"])
            self.assertEqual(main(["evaluate-masking", "--gold", str(gold), "--masks", str(sidecar), "--output", str(report)]), 0)
            self.assertTrue(report.is_file())

    def test_output_collisions_fail_before_any_file_is_overwritten(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            gold = self.fixture(root)
            original = gold.read_bytes()
            output = root / "output.jsonl"
            for arguments in [
                ["mask", "--input", str(gold), "--output", str(output), "--mask-records", str(output)],
                ["mask", "--input", str(gold), "--output", str(gold)],
                ["mask", "--input", str(gold), "--output", str(output), "--mask-records", str(gold)],
                ["evaluate-detection", "--gold", str(gold), "--pred", str(gold), "--output", str(gold)],
                ["evaluate-masking", "--gold", str(gold), "--masks", str(output), "--output", str(gold)],
            ]:
                with self.assertRaises(ValueError):
                    main(arguments)
                self.assertEqual(gold.read_bytes(), original)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
