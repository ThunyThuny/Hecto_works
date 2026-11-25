"""Explicit experiment opt-in and provenance binding; no model loads/training."""
import copy
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kpii_mvp.b_data import document_identities, pending_review
from kpii_mvp.b_evaluation import apply_calibrated_b, calibrate_b, evaluate_ab, main as evaluation_main
from kpii_mvp.b_pipeline import main as pipeline_main
from test_b_evaluation import MODEL_SHA, calibration, fixture, reviews, scores


PACKET_SHA = "c" * 64


def evidence(documents, review_records, **kwargs):
    """Boundary fake; packet verification itself has separate real tests."""
    if not kwargs.get("synthetic_experiment_dir"):
        from kpii_mvp.b_data import validate_reviewed_documents
        return validate_reviewed_documents(documents, review_records,
            expected_split=kwargs.get("expected_split"),
            policy_id=kwargs.get("policy_id", "career_external_share_v1"))
    return {"experiment_mode": "unreviewed_synthetic_experiment", "human_reviewed": False,
            "review_evidence": "unreviewed_synthetic_experiment", "packet_manifest_sha256": PACKET_SHA,
            "documents": len(documents), "split": kwargs.get("expected_split")}


class ExperimentEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.dev, self.dev_candidate = fixture()
        self.test, self.test_candidate = fixture("test1", "test", "test-group", "Eve OpenOrgX Hidden")
        self.pending_dev = {self.dev.doc_id: pending_review(self.dev)}
        self.pending_test = {self.test.doc_id: pending_review(self.test)}

    def experimental_calibration(self, **extra):
        with patch("kpii_mvp.b_experiment.validate_b_evidence", side_effect=evidence):
            return calibrate_b([self.dev], [self.dev_candidate], [scores(self.dev_candidate)],
                self.pending_dev, model_manifest_sha256=MODEL_SHA,
                synthetic_experiment_dir="immutable-packet", **extra)

    def test_calibration_reports_pending_evidence_without_faking_review(self):
        before = copy.deepcopy(self.pending_dev)
        selected = self.experimental_calibration()
        self.assertFalse(selected["human_reviewed"])
        self.assertEqual(selected["synthetic_experiment"]["packet_manifest_sha256"], PACKET_SHA)
        self.assertEqual(selected["threshold"], 0.9)
        self.assertEqual(before, self.pending_dev)

    def test_experimental_calibration_requires_opt_in_for_test_even_with_approved_test(self):
        selected = self.experimental_calibration()
        with self.assertRaisesRegex(ValueError, "synthetic-experiment-dir"):
            evaluate_ab([self.test], [self.test_candidate], [scores(self.test_candidate)],
                reviews(self.test), selected, model_manifest_sha256=MODEL_SHA)

    def test_test_report_and_apply_inherit_the_same_experiment_identity(self):
        selected = self.experimental_calibration()
        with patch("kpii_mvp.b_experiment.validate_b_evidence", side_effect=evidence):
            result = evaluate_ab([self.test], [self.test_candidate], [scores(self.test_candidate)],
                self.pending_test, selected, model_manifest_sha256=MODEL_SHA,
                synthetic_experiment_dir="immutable-packet",
                model_training_evidence=selected["synthetic_experiment"])
        self.assertFalse(result["human_reviewed"])
        self.assertEqual(result["synthetic_experiment"]["packet_manifest_sha256"], PACKET_SHA)
        applied = apply_calibrated_b([self.test_candidate], [scores(self.test_candidate)], selected,
            model_manifest_sha256=MODEL_SHA, model_training_evidence=selected["synthetic_experiment"])
        self.assertFalse(applied["human_reviewed"])
        self.assertEqual(set(applied["masked_documents"][0]), {"doc_id", "masked_text"})

    def test_packet_mismatch_between_calibration_test_or_model_is_rejected(self):
        selected = self.experimental_calibration()
        changed = copy.deepcopy(selected)
        changed["synthetic_experiment"]["packet_manifest_sha256"] = "d" * 64
        with patch("kpii_mvp.b_experiment.validate_b_evidence", side_effect=evidence):
            with self.assertRaisesRegex(ValueError, "same synthetic packet"):
                evaluate_ab([self.test], [self.test_candidate], [scores(self.test_candidate)],
                    self.pending_test, changed, model_manifest_sha256=MODEL_SHA,
                    synthetic_experiment_dir="immutable-packet")
        with self.assertRaisesRegex(ValueError, "same synthetic packet"):
            self.experimental_calibration(model_training_evidence=changed["synthetic_experiment"])
        with self.assertRaisesRegex(ValueError, "same synthetic packet"):
            apply_calibrated_b([self.test_candidate], [scores(self.test_candidate)], selected,
                model_manifest_sha256=MODEL_SHA, model_training_evidence=changed["synthetic_experiment"])

    def test_reviewed_calibration_cannot_be_mixed_into_unreviewed_test(self):
        selected = calibration()
        with patch("kpii_mvp.b_experiment.validate_b_evidence", side_effect=evidence):
            with self.assertRaisesRegex(ValueError, "mix reviewed calibration"):
                evaluate_ab([self.test], [self.test_candidate], [scores(self.test_candidate)],
                    self.pending_test, selected, model_manifest_sha256=MODEL_SHA,
                    synthetic_experiment_dir="immutable-packet")

    def test_experimental_model_cannot_silently_enter_reviewed_calibration(self):
        selected = self.experimental_calibration()
        with self.assertRaisesRegex(ValueError, "synthetic-experiment-dir"):
            calibrate_b([self.dev], [self.dev_candidate], [scores(self.dev_candidate)], reviews(self.dev),
                model_manifest_sha256=MODEL_SHA, model_training_evidence=selected["synthetic_experiment"])

    def test_calibration_cannot_relabel_experiment_as_human_reviewed(self):
        selected = self.experimental_calibration()
        selected["human_reviewed"] = True
        with self.assertRaisesRegex(ValueError, "cannot claim human review"):
            apply_calibrated_b([self.test_candidate], [scores(self.test_candidate)], selected,
                model_manifest_sha256=MODEL_SHA)

    def test_preflight_propagates_explicit_opt_in_to_all_three_splits(self):
        train, _ = fixture("train1", "train", "train-group", "Tom BoardOrg Secret")
        from kpii_mvp.data import write_jsonl
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            argv = ["preflight", "--synthetic-experiment-dir", str(root)]
            for name, doc in (("train", train), ("dev", self.dev), ("test", self.test)):
                doc = replace(doc, mentions=[replace(doc.mentions[0], entity_type="WORKPLACE")]
                              + doc.mentions[1:])
                documents_path, review_path = root / (name + ".jsonl"), root / (name + ".reviews.jsonl")
                write_jsonl(documents_path, [doc])
                review_path.write_text(json.dumps(pending_review(doc)) + "\n", encoding="utf-8")
                argv += ["--" + name, str(documents_path), "--" + name + "-reviews", str(review_path)]
            stream = io.StringIO()
            with patch("kpii_mvp.b_experiment.validate_b_evidence", side_effect=evidence) as check, redirect_stdout(stream):
                pipeline_main(argv)
            self.assertEqual(check.call_count, 3)
            self.assertEqual({call.kwargs["expected_split"] for call in check.call_args_list}, {"train", "dev", "test"})
            result = json.loads(stream.getvalue())
            self.assertEqual(result["status"], "ready_for_unreviewed_synthetic_experiment")
            self.assertFalse(result["training_performed"])
            self.assertFalse(result["human_reviewed"])

    def test_calibrate_and_evaluate_cli_forward_opt_in_and_model_evidence(self):
        train, _ = fixture("train1", "train", "train-group", "Tom BoardOrg Secret")
        packet_evidence = evidence([self.dev], self.pending_dev, synthetic_experiment_dir="packet")
        provenance = {"b_manifest_sha256": MODEL_SHA, "a_training_manifest_sha256": "a" * 64,
                      "a_prediction_signature_sha256": "b" * 64, "synthetic_experiment": packet_evidence,
                      "model_data_identities": {"train": document_identities([train]),
                                                "dev": document_identities([self.dev])}}
        inputs = {"dev.gold": [self.dev], "dev.candidates": [self.dev_candidate],
                  "test.gold": [self.test], "test.candidates": [self.test_candidate]}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            calibration_path = root / "calibration.json"
            def argv(role):
                return [role, "--gold", ("dev" if role == "calibrate" else "test") + ".gold",
                        "--reviews", "reviews", "--candidates",
                        ("dev" if role == "calibrate" else "test") + ".candidates",
                        "--scores", "scores", "--model-manifest", "model",
                        "--synthetic-experiment-dir", "packet"]
            with patch("kpii_mvp.b_experiment.validate_b_evidence", side_effect=evidence), \
                 patch("kpii_mvp.b_evaluation._score_provenance", side_effect=lambda *a, **k: copy.deepcopy(provenance)), \
                 patch("kpii_mvp.data.read_jsonl", side_effect=lambda path: inputs[path]), \
                 patch("kpii_mvp.b_data.load_reviews", side_effect=[self.pending_dev, self.pending_test]), \
                 patch("kpii_mvp.b_evaluation.load_scores", side_effect=[[scores(self.dev_candidate)], [scores(self.test_candidate)]]), \
                 redirect_stdout(io.StringIO()):
                evaluation_main(argv("calibrate") + ["--output", str(calibration_path)])
                evaluation_main(argv("evaluate") + ["--calibration", str(calibration_path),
                                                     "--output-dir", str(root / "evaluated")])
            result = json.loads((root / "evaluated" / "ab-evaluation.json").read_text())
            self.assertFalse(result["human_reviewed"])
            self.assertEqual(result["provenance"]["synthetic_experiment"]["packet_manifest_sha256"], PACKET_SHA)


if __name__ == "__main__":
    unittest.main()
