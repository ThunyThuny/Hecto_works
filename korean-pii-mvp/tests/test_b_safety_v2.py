"""Offline input, loss and compatibility contracts; no model is trained/loaded."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kpii_mvp.b_inputs import (DEFAULT_INPUT_VERSION, FOCUSED_INPUT_VERSION,
    FIXED_POLICY_ID, LABEL2ID, POLICY_CONFIG, input_sha256, input_spec,
    policy_sha256, prepare_candidate_input)
from kpii_mvp.b_model import (MANIFEST_VERSION, _a_score_provenance,
    _training_loss, _training_options, build_parser, load_checkpoint_manifest,
    predict_scores, prepare_features)
from test_b_model import CharacterTokenizer, document


def rendered(result):
    return "".join(chr(value - 1000) for value in result["model_inputs"]["input_ids"] if value >= 1000)


class TargetFocusTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def test_focus_targets_second_occurrence_and_retains_all_ownership_evidence(self):
        text = "한빛대학교는 공개 채용 기관입니다.\n신청인은 한빛대학교를 졸업했습니다.\n이력은 외부 공개에 동의하지 않았습니다."
        doc = document(text=text)
        begin = text.rindex("한빛대학교")
        mention = replace(doc.mentions[0], start=begin, end=begin + len("한빛대학교"))
        doc = replace(doc, mentions=[mention])
        result = prepare_candidate_input(self.tokenizer, doc, mention, 1000,
                                         input_version=FOCUSED_INPUT_VERSION)
        self.assertTrue(result["available"])
        focus = result["context"]["target_focus"]
        self.assertEqual(text[focus["start"]:focus["end"]], "신청인은 한빛대학교를 졸업했습니다.")
        self.assertIn("target_value=한빛대학교\ntarget_focus=신청인은 한빛대학교를 졸업했습니다.", rendered(result))
        self.assertIn("이력은 외부 공개에 동의하지 않았습니다.", rendered(result))
        ids = result["model_inputs"]["input_ids"]
        self.assertEqual(ids.count(700), 1)
        self.assertEqual(ids.count(701), 1)
        # Exact offsets still bracket the private occurrence in the whole source.
        prefix_before_target = self.tokenizer.encode(text[:begin])
        self.assertEqual(ids[ids.index(700) - len(prefix_before_target):ids.index(700)], prefix_before_target)

    def test_focus_does_not_read_gold_actions_reasons_or_generator_metadata(self):
        doc = document()
        baseline = prepare_candidate_input(self.tokenizer, doc, doc.mentions[0],
                                            input_version=FOCUSED_INPUT_VERSION)
        changed = replace(doc, mentions=[replace(doc.mentions[0], action="KEEP", reason_code="SECRET_REASON")],
            trusted_context={**doc.trusted_context, "scenario": "SECRET_KEEP", "gold_action": "KEEP"})
        result = prepare_candidate_input(self.tokenizer, changed, changed.mentions[0],
                                         input_version=FOCUSED_INPUT_VERSION)
        self.assertEqual(baseline, result)
        self.assertNotIn("SECRET", rendered(result))

    def test_v1_is_identical_to_original_default_and_v2_has_separate_contract(self):
        doc = document()
        default = prepare_candidate_input(self.tokenizer, doc, doc.mentions[0])
        explicit = prepare_candidate_input(self.tokenizer, doc, doc.mentions[0],
                                           input_version=DEFAULT_INPUT_VERSION)
        focused = prepare_candidate_input(self.tokenizer, doc, doc.mentions[0],
                                          input_version=FOCUSED_INPUT_VERSION)
        self.assertEqual(default, explicit)
        self.assertNotEqual(default["model_inputs"], focused["model_inputs"])
        self.assertNotEqual(input_sha256(DEFAULT_INPUT_VERSION), input_sha256(FOCUSED_INPUT_VERSION))

    def test_focus_overflow_rejects_without_truncating_source(self):
        doc = document(text="한빛대학교 " + "전체 소유 관계를 보존해야 합니다. " * 20)
        result = prepare_candidate_input(self.tokenizer, doc, doc.mentions[0], 250,
                                         input_version=FOCUSED_INPUT_VERSION)
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "context_exceeds_max_length")
        self.assertTrue(result["context"]["complete"])
        self.assertEqual(result["context"]["end"], len(doc.text))
        with self.assertRaisesRegex(ValueError, "independently reviewed"):
            prepare_features([doc], self.tokenizer, 250, FOCUSED_INPUT_VERSION)

    def test_unknown_input_contract_rejected(self):
        doc = document()
        with self.assertRaisesRegex(ValueError, "input_version"):
            prepare_candidate_input(self.tokenizer, doc, doc.mentions[0], input_version="v3")


class ConservativeLossTests(unittest.TestCase):
    def test_legacy_defaults_and_new_cli_options(self):
        self.assertEqual(_training_options(SimpleNamespace()), {
            "input_version": DEFAULT_INPUT_VERSION, "mask_class_weight": 1.0,
            "label_smoothing": 0.0, "early_stopping_patience": 0})
        args = build_parser().parse_args(["train", "--train", "train", "--train-reviews", "tr-reviews",
            "--dev", "valid", "--dev-reviews", "dv-reviews", "--output-dir", "run",
            "--input-version", FOCUSED_INPUT_VERSION, "--mask-class-weight", "3",
            "--label-smoothing", "0.02", "--early-stopping-patience", "1"])
        options = _training_options(args)
        self.assertEqual(options["mask_class_weight"], 3)
        self.assertEqual(options["early_stopping_patience"], 1)

    def test_invalid_training_controls_fail_before_ml_dependency_loading(self):
        for key, bad_values in {"mask_class_weight": [0.0, 0.5, float("inf"), True],
                                "label_smoothing": [-0.01, 1.0, float("nan"), True],
                                "early_stopping_patience": [-1, 1.0, True]}.items():
            for bad in bad_values:
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    _training_options(SimpleNamespace(**{key: bad}))

    def test_weighted_ce_applies_mask_cost_and_smoothing_to_fp32_logits(self):
        events = []
        class Logits:
            def float(self):
                events.append("float")
                return self
            def new_tensor(self, values):
                events.append(("weights", values))
                return ("device_and_dtype_matched", values)
        def ce(logits, labels, *, weight, label_smoothing):
            events.append(("cross_entropy", labels, weight, label_smoothing))
            return "loss"
        torch = SimpleNamespace(nn=SimpleNamespace(functional=SimpleNamespace(cross_entropy=ce)))
        self.assertEqual(_training_loss(Logits(), "labels", torch, 3.0, 0.02), "loss")
        self.assertEqual(events, ["float", ("weights", [3.0, 1.0]),
            ("cross_entropy", "labels", ("device_and_dtype_matched", [3.0, 1.0]), 0.02)])


class CheckpointInputContractTests(unittest.TestCase):
    def write_checkpoint(self, root, version=None):
        (root / "config.json").write_text("saved", encoding="utf-8")
        manifest = {"stage": MANIFEST_VERSION, "policy_id": FIXED_POLICY_ID,
            "policy_config": POLICY_CONFIG, "policy_sha256": policy_sha256(), "label2id": LABEL2ID,
            "config": {"max_length": 512},
            "checkpoint_artifacts": {"config.json": hashlib.sha256(b"saved").hexdigest()}}
        if version:
            manifest["config"]["input_version"] = version
            manifest["input_spec"], manifest["input_sha256"] = input_spec(version), input_sha256(version)
        (root / "b_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    def test_existing_checkpoint_without_new_fields_remains_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.write_checkpoint(root)
            self.assertEqual(load_checkpoint_manifest(root), manifest)

    def test_focused_checkpoint_and_tampered_input_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.write_checkpoint(root, FOCUSED_INPUT_VERSION)
            self.assertEqual(load_checkpoint_manifest(root), manifest)
            manifest["config"]["input_version"] = DEFAULT_INPUT_VERSION
            (root / "b_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input specification changed"):
                load_checkpoint_manifest(root)

    def test_scoring_uses_checkpoint_input_version_without_user_override(self):
        # Stub model/tokenizer handles prove dispatch without executing inference.
        manifest = {"config": {"max_length": 512, "input_version": FOCUSED_INPUT_VERSION}}
        model = SimpleNamespace(config=SimpleNamespace(label2id=LABEL2ID), to=lambda device: None)
        model_factory = SimpleNamespace(from_pretrained=lambda *args, **kwargs: model)
        tokenizer_factory = SimpleNamespace(from_pretrained=lambda *args, **kwargs: "tokenizer")
        transformers = SimpleNamespace(AutoModelForSequenceClassification=model_factory,
                                       AutoTokenizer=tokenizer_factory)
        with patch("kpii_mvp.b_model.load_checkpoint_manifest", return_value=manifest), \
             patch("kpii_mvp.b_model._ml_dependencies", return_value=("torch", transformers)), \
             patch("kpii_mvp.b_model._device", return_value="device"), \
             patch("kpii_mvp.b_model.score_documents", return_value=[]) as scoring:
            self.assertEqual(predict_scores([], "checkpoint"), [])
        self.assertEqual(scoring.call_args.kwargs["input_version"], FOCUSED_INPUT_VERSION)

    def test_refinement_policy_is_bound_to_a_candidate_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidates.jsonl"
            candidate.write_text("saved candidates", encoding="utf-8")
            manifest = {"stage": "A-candidate-refinement", "training_manifest_sha256": "a" * 64,
                "max_length": 512, "stride": 128,
                "output": {"sha256": hashlib.sha256(b"saved candidates").hexdigest()},
                "refinement": {"version": "address-position-v1", "config_sha256": "b" * 64}}
            Path(str(candidate) + ".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch("kpii_mvp.candidate_refinement.validate_refinement_manifest", return_value=manifest):
                first = _a_score_provenance(candidate)
                manifest["refinement"]["config_sha256"] = "c" * 64
                second = _a_score_provenance(candidate)
            self.assertEqual(first["a_training_manifest_sha256"], "a" * 64)
            self.assertNotEqual(first["a_prediction_signature_sha256"], second["a_prediction_signature_sha256"])


if __name__ == "__main__":
    unittest.main()
