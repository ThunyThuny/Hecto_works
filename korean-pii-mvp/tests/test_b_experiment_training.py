"""Training evidence must validate before any model dependency is loaded."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kpii_mvp.b_data import pending_review
from kpii_mvp.b_inputs import FIXED_POLICY_ID
from kpii_mvp.b_model import train, build_parser
from kpii_mvp.schema import Document, Mention


def sample(split):
    value = "가상누리대학교"
    text = f"{split} 자료. 지원자의 학력은 {value} 졸업입니다. 기관 안내: {value}의 상담 일정."
    first, last = text.index(value), text.rindex(value)
    return Document(f"{split}-one", text,
        [Mention("m1", first, first + len(value), value, "SCHOOL", "MASK"),
         Mention("m2", last, last + len(value), value, "SCHOOL", "KEEP")],
        source_group=f"{split}-group", annotation_scope=["SCHOOL"],
        source="synthetic_rule_based_v3_bulk", split=split, policy_id=FIXED_POLICY_ID,
        trusted_context={"approved_public_orgs": [value], "document_type": "general_notice"})


class ExperimentTrainingEntryTests(unittest.TestCase):
    def arguments(self, output, experimental=True):
        values = dict(output_dir=str(output), epochs=3, max_steps=0, batch_size=4, eval_batch_size=8,
                      learning_rate=3e-5, weight_decay=0.01, train="train.jsonl", dev="valid.jsonl",
                      train_reviews="train.reviews.jsonl", dev_reviews="valid.reviews.jsonl")
        if experimental:
            values["synthetic_experiment_dir"] = "original-packet"
        return SimpleNamespace(**values)

    def test_default_pending_is_rejected_before_ml_load_or_output_creation(self):
        docs = [sample("train"), sample("valid")]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            output = Path(temporary) / "model"
            reviews = [{d.doc_id: pending_review(d)} for d in docs]
            with patch("kpii_mvp.b_model._load_documents", side_effect=[[d] for d in docs]), \
                 patch("kpii_mvp.b_data.load_reviews", side_effect=reviews), \
                 patch("kpii_mvp.b_model._ml_dependencies") as ml:
                with self.assertRaisesRegex(ValueError, "PENDING"):
                    train(self.arguments(output, experimental=False))
                ml.assert_not_called()
                self.assertFalse(output.exists())

    def test_experiment_path_propagates_to_both_split_validations(self):
        docs = [sample("train"), sample("valid")]
        evidence = {"packet_manifest_sha256": "a" * 64, "human_reviewed": False,
                    "experiment_mode": "unreviewed_synthetic_experiment"}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            output = Path(temporary) / "model"
            with patch("kpii_mvp.b_model._load_documents", side_effect=[[d] for d in docs]), \
                 patch("kpii_mvp.b_data.load_reviews", return_value={}), \
                 patch("kpii_mvp.b_experiment.validate_b_evidence", return_value=evidence) as validate, \
                 patch("kpii_mvp.b_model._ml_dependencies", side_effect=RuntimeError("STOP_BEFORE_MODEL")):
                with self.assertRaisesRegex(RuntimeError, "STOP_BEFORE_MODEL"):
                    train(self.arguments(output))
                self.assertEqual(validate.call_count, 2)
                self.assertEqual([c.kwargs["expected_split"] for c in validate.call_args_list], ["train", "dev"])
                self.assertTrue(all(c.kwargs["synthetic_experiment_dir"] == "original-packet" for c in validate.call_args_list))
                self.assertFalse(output.exists())

    def test_changed_packet_between_splits_fails_before_model_loading(self):
        docs = [sample("train"), sample("valid")]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            output = Path(temporary) / "model"
            with patch("kpii_mvp.b_model._load_documents", side_effect=[[d] for d in docs]), \
                 patch("kpii_mvp.b_data.load_reviews", return_value={}), \
                 patch("kpii_mvp.b_experiment.validate_b_evidence", side_effect=[
                     {"packet_manifest_sha256": "a" * 64}, {"packet_manifest_sha256": "b" * 64}]), \
                 patch("kpii_mvp.b_model._ml_dependencies") as ml:
                with self.assertRaisesRegex(ValueError, "changed between"):
                    train(self.arguments(output))
                ml.assert_not_called()
                self.assertFalse(output.exists())

    def test_cli_opt_in_is_explicit_and_absent_by_default(self):
        base = ["train", "--train", "train", "--train-reviews", "tr", "--dev", "dev", "--dev-reviews", "dr", "--output-dir", "out"]
        self.assertIsNone(build_parser().parse_args(base).synthetic_experiment_dir)
        self.assertEqual(build_parser().parse_args(base + ["--synthetic-experiment-dir", "packet"]).synthetic_experiment_dir, "packet")


if __name__ == "__main__":
    unittest.main()
