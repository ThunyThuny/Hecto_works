"""Offline B safety/learning-contract checks, without a generated training corpus."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kpii_mvp.b_inputs import (CAND_START, CAND_END, FIXED_POLICY_ID, LABEL2ID,
    MARKERS, POLICY_CONFIG, allowed_to_keep, canonical_type, keep_eligible,
    policy_sha256, prepare_candidate_input, trusted_features)
from kpii_mvp.b_model import (MANIFEST_VERSION, build_parser, load_checkpoint_manifest,
                             load_initial_model, prepare_features, validate_train_dev,
                             _a_score_provenance, score)
from kpii_mvp.schema import Document, Mention


class CharacterTokenizer:
    """One code point per token; added marker IDs are outside that range."""
    unk_token_id = 0
    padding_side = "right"

    def __init__(self):
        self.markers = {CAND_START: 700, CAND_END: 701}

    def encode(self, text, add_special_tokens=False, truncation=False):
        assert add_special_tokens is False and truncation is False
        return [ord(character) + 1000 for character in text]

    def convert_tokens_to_ids(self, token):
        return self.markers.get(token, self.unk_token_id)

    def prepare_for_model(self, ids, **kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids": [10] + ids + [11], "attention_mask": [1] * (len(ids) + 2)}


def document(identifier="train-1", text="한빛대학교 일반 안내", action="MASK", kind="SCHOOL", split="train",
             approved=True, group=None):
    value = "한빛대학교"
    begin = text.index(value)
    return Document(doc_id=identifier, text=text,
        mentions=[Mention("m1", begin, begin + len(value), value, kind, action=action, reason_code="REVIEW_REASON")],
        source_group=group or identifier, annotation_scope=[kind], split=split, policy_id=FIXED_POLICY_ID,
        trusted_context={"approved_public_orgs": [value] if approved else [], "document_type": "general_notice"})


class CandidateInputTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.document = document()

    def prepare(self, doc=None, length=512):
        doc = doc or self.document
        return prepare_candidate_input(self.tokenizer, doc, doc.mentions[0], length)

    def test_exact_offsets_target_only_second_identical_value(self):
        doc = document(text="한빛대학교, 다시 한빛대학교 안내")
        start = doc.text.rindex("한빛대학교")
        mention = replace(doc.mentions[0], start=start, end=start + len("한빛대학교"))
        doc = replace(doc, mentions=[mention])
        result = self.prepare(doc)
        self.assertTrue(result["available"])
        ids = result["model_inputs"]["input_ids"]
        begin, end = ids.index(700), ids.index(701)
        self.assertEqual(ids[begin + 1:end], self.tokenizer.encode("한빛대학교"))
        self.assertEqual(ids.count(700), 1)
        self.assertEqual(ids.count(701), 1)
        # The first occurrence is still ordinary context before the marker.
        first = self.tokenizer.encode("한빛대학교, 다시 ")
        self.assertEqual(ids[begin - len(first):begin], first)

    def test_gold_actions_reasons_reviews_and_unknown_metadata_do_not_leak(self):
        baseline = self.prepare()
        changed = replace(self.document,
            mentions=[replace(self.document.mentions[0], action="KEEP", reason_code="GOLD_KEEP_SECRET")],
            trusted_context={**self.document.trusted_context, "action": "KEEP", "review_approved": True,
                             "gold_reason": "GOLD_KEEP_SECRET", "arbitrary": {"private": "value"}})
        self.assertEqual(baseline, self.prepare(changed))
        inputs = baseline["model_inputs"]["input_ids"]
        rendered = "".join(chr(value - 1000) for value in inputs if value >= 1000)
        self.assertNotIn("REVIEW_REASON", rendered)
        self.assertNotIn("GOLD_KEEP_SECRET", rendered)

    def test_approved_metadata_is_only_boolean_and_doc_type_allowlisted(self):
        facts = trusted_features(self.document, self.document.mentions[0])
        self.assertEqual(facts, {"approved_contact_match": False, "approved_org_match": True,
                                "approved_position_match": False, "document_type": "general_notice"})
        doc = replace(self.document, trusted_context={"document_type": "secret freeform GOLD_KEEP"})
        self.assertEqual(trusted_features(doc, doc.mentions[0])["document_type"], "unknown")

    def test_invalid_approval_list_is_rejected_instead_of_substring_membership(self):
        for value in ("한빛대학교", True, [None], [""]):
            with self.subTest(value=value):
                doc = replace(self.document, trusted_context={"approved_public_orgs": value})
                with self.assertRaisesRegex(ValueError, "list of nonempty strings"):
                    self.prepare(doc)

    def test_source_marker_injection_fails_closed(self):
        for marker in MARKERS:
            result = self.prepare(document(text=marker + " 한빛대학교"))
            self.assertFalse(result["available"])
            self.assertEqual(result["reason"], "reserved_marker_in_source")

    def test_missing_registered_marker_is_configuration_error(self):
        self.tokenizer.markers[CAND_END] = 0
        with self.assertRaisesRegex(ValueError, "registered candidate markers"):
            self.prepare()

    def test_long_context_fails_closed_even_when_candidate_is_short(self):
        doc = document(text="한빛대학교 " + "멀리 있는 소유 관계 근거 " * 100)
        result = self.prepare(doc)
        self.assertFalse(result["available"])
        self.assertIsNone(result["model_inputs"])
        self.assertEqual(result["reason"], "context_exceeds_max_length")
        self.assertTrue(result["context"]["complete"])
        self.assertEqual(result["context"]["end"], len(doc.text))

    def test_too_long_candidate_is_not_silently_truncated(self):
        doc = document()
        value = "가" * 600
        doc = replace(doc, text=value,
            mentions=[Mention("long", 0, len(value), value, "SCHOOL", action="MASK")])
        result = self.prepare(doc)
        self.assertFalse(result["available"])
        with self.assertRaisesRegex(ValueError, "independently reviewed"):
            prepare_features([doc], self.tokenizer)

    def test_other_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "policy differs"):
            self.prepare(replace(self.document, policy_id="other"))
        with self.assertRaisesRegex(ValueError, "implements only"):
            prepare_candidate_input(self.tokenizer, self.document, self.document.mentions[0], policy_id="other")

    def test_mandatory_mask_type_is_unavailable_to_classifier(self):
        for kind in ("NAME", "PS_NAME", "RESIDENT_NUMBER", "QT_RESIDENT_NUMBER", "QT_ACCOUNT"):
            with self.subTest(kind=kind):
                result = self.prepare(document(kind=kind))
                self.assertFalse(result["available"])
                self.assertEqual(result["reason"], "policy_always_mask")


class FixedPolicyTests(unittest.TestCase):
    def test_approval_is_required_and_exact(self):
        doc = document(action="KEEP")
        self.assertTrue(allowed_to_keep(doc, doc.mentions[0]))
        self.assertFalse(allowed_to_keep(replace(doc, policy_id=None), doc.mentions[0]))
        for context in ({}, {"approved_public_orgs": ["한빛대학교병원"]},
                        {"approved_public_contacts": ["한빛대학교"]}):
            self.assertFalse(allowed_to_keep(replace(doc, trusted_context=context), doc.mentions[0]))

    def test_contact_and_position_require_their_own_approved_lists(self):
        for kind, key in (("QT_PHONE", "approved_public_contacts"), ("TMI_EMAIL", "approved_public_contacts"),
                          ("LC_ADDRESS", "approved_public_contacts"), ("CV_POSITION", "approved_public_positions")):
            doc = document(kind=kind)
            self.assertFalse(allowed_to_keep(doc, doc.mentions[0]))
            doc = replace(doc, trusted_context={key: [doc.mentions[0].text]})
            self.assertTrue(allowed_to_keep(doc, doc.mentions[0]))

    def test_native_aliases_and_default_mask(self):
        for label in ("OGG_EDUCATION", "OG_WORKPLACE", "OG_DEPARTMENT"):
            doc = document(kind=label)
            self.assertTrue(keep_eligible(label))
            self.assertTrue(allowed_to_keep(doc, doc.mentions[0]))
        self.assertEqual(canonical_type("UNKNOWN"), "OTHER")
        self.assertFalse(keep_eligible("UNKNOWN"))
        doc = document(kind="PS_NAME")
        self.assertFalse(allowed_to_keep(doc, doc.mentions[0]))

    def test_policy_digest_is_deterministic(self):
        expected = hashlib.sha256(json.dumps(POLICY_CONFIG, sort_keys=True, ensure_ascii=False,
                                            separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(policy_sha256(), expected)


class TrainingContractTests(unittest.TestCase):
    def setUp(self):
        self.train = [document("tr1", action="MASK"), document("tr2", text="한빛대학교 기관 정보", action="KEEP")]
        self.dev = [document("dv1", text="한빛대학교 문의 안내", action="KEEP", split="dev"),
                    document("dv2", text="한빛대학교 졸업 이력", action="MASK", split="dev")]

    def test_valid_reviewed_dataset_shape_and_feature_classes(self):
        validate_train_dev(self.train, self.dev)
        rows, report = prepare_features(self.train, CharacterTokenizer())
        self.assertEqual({row["labels"] for row in rows}, {0, 1})
        self.assertEqual(report["contextual_candidates"], 2)

    def test_missing_action_never_becomes_keep(self):
        doc = self.train[0]
        invalid = replace(doc, mentions=[replace(doc.mentions[0], action=None)])
        with self.assertRaisesRegex(ValueError, "null is not KEEP"):
            validate_train_dev([invalid, self.train[1]], self.dev)
        with self.assertRaisesRegex(ValueError, "explicit MASK/KEEP"):
            prepare_features([invalid], CharacterTokenizer())

    def test_class_balance_required(self):
        with self.assertRaisesRegex(ValueError, "both MASK and KEEP"):
            validate_train_dev(self.train[:1], self.dev)

    def test_dev_requires_private_and_public_contextual_examples(self):
        with self.assertRaisesRegex(ValueError, "dev candidates must contain both MASK and KEEP"):
            validate_train_dev(self.train, self.dev[:1])
        # A mandatory NAME MASK does not exercise B's private/public decision.
        name = document("dev-name", text="한빛대학교 이름 표기", kind="PS_NAME", action="MASK", split="dev")
        with self.assertRaisesRegex(ValueError, "dev candidates must contain both MASK and KEEP"):
            validate_train_dev(self.train, [self.dev[0], name])

    def test_keep_that_runtime_policy_cannot_release_is_rejected(self):
        invalid = replace(self.train[1], trusted_context={})
        with self.assertRaisesRegex(ValueError, "KEEP conflicts"):
            validate_train_dev([self.train[0], invalid], self.dev)

    def test_test_split_cannot_be_used_for_selection(self):
        with self.assertRaisesRegex(ValueError, "unexpected split"):
            validate_train_dev(self.train, [replace(self.dev[0], split="test")])

    def test_source_group_and_text_leakage_rejected(self):
        for dev in (replace(self.dev[0], source_group="tr1"),
                    replace(self.dev[0], text=self.train[0].text)):
            with self.subTest(dev=dev):
                with self.assertRaisesRegex(ValueError, "Train/dev overlap"):
                    validate_train_dev(self.train, [dev])

    def test_unsupported_types_are_policy_rows_not_classifier_labels(self):
        rows, report = prepare_features([self.train[0], document("name", kind="PS_NAME")], CharacterTokenizer())
        self.assertEqual(len(rows), 1)
        self.assertEqual(report["policy_mask_candidates"], 1)

    def test_cli_requires_both_review_files_and_has_no_test_input(self):
        parser = build_parser()
        args = parser.parse_args(["train", "--train", "tr.jsonl", "--train-reviews", "tr-review.jsonl",
            "--dev", "dv.jsonl", "--dev-reviews", "dv-review.jsonl", "--output-dir", "run"])
        self.assertEqual(args.epochs, 3)
        self.assertFalse(hasattr(args, "test"))


class InitializerTests(unittest.TestCase):
    def test_fresh_two_label_head_strict_encoder_transfer_before_marker_resize(self):
        events = []
        class Base:
            def state_dict(self):
                return {"encoder.weight": "trained A weights"}
            def load_state_dict(self, weights, strict):
                events.append(("transfer", weights, strict))
        class Model:
            base_model = Base()
            def resize_token_embeddings(self, count):
                events.append(("resize", count))
            def tie_weights(self):
                events.append(("tie",))
        class AutoConfig:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return SimpleNamespace(architectures=["ModernBertForTokenClassification"], num_labels=67)
        class AutoModel:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                selftest.assertNotIn("ignore_mismatched_sizes", kwargs)
                return Base()
        class SequenceModel:
            @staticmethod
            def from_config(config, **kwargs):
                events.append(("newhead", config.num_labels, config.label2id))
                return Model()
        class Tokenizer:
            def add_special_tokens(self, value):
                events.append(("addmarkers", value))
                return 2
            def __len__(self):
                return 102
        class AutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return Tokenizer()
        selftest = self
        transformers = SimpleNamespace(AutoConfig=AutoConfig, AutoModel=AutoModel,
            AutoModelForSequenceClassification=SequenceModel, AutoTokenizer=AutoTokenizer)
        _, _, provenance = load_initial_model(transformers, "not-a-local-checkpoint")
        self.assertEqual(events[0], ("newhead", 2, LABEL2ID))
        self.assertEqual(events[1], ("transfer", {"encoder.weight": "trained A weights"}, True))
        self.assertEqual(events[3], ("resize", 102))
        self.assertTrue(provenance["encoder_transfer_strict"])

    def test_checkpoint_artifact_changes_are_rejected_before_loading_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "config.json"
            artifact.write_text("original", encoding="utf-8")
            manifest = {"stage": MANIFEST_VERSION, "policy_id": FIXED_POLICY_ID, "policy_config": POLICY_CONFIG,
                "policy_sha256": policy_sha256(), "label2id": LABEL2ID, "config": {"max_length": 512},
                "checkpoint_artifacts": {"config.json": hashlib.sha256(b"original").hexdigest()}}
            (root / "b_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(load_checkpoint_manifest(root), manifest)
            artifact.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                load_checkpoint_manifest(root)


class ScoreProvenanceTests(unittest.TestCase):
    def test_a_signature_is_independent_of_split_and_binds_window_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signatures = []
            for name in ("dev", "test"):
                candidate = root / (name + ".jsonl")
                candidate.write_text(name, encoding="utf-8")
                manifest = {"stage": "A-prediction", "training_manifest_sha256": "a" * 64,
                            "output": {"sha256": hashlib.sha256(name.encode()).hexdigest()},
                            "max_length": 512, "stride": 128}
                Path(str(candidate) + ".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                signatures.append(_a_score_provenance(candidate)["a_prediction_signature_sha256"])
            self.assertEqual(signatures[0], signatures[1])
            manifest["stride"] = 64
            Path(str(candidate) + ".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertNotEqual(_a_score_provenance(candidate)["a_prediction_signature_sha256"], signatures[0])
            candidate.write_text("mutated", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match candidate"):
                _a_score_provenance(candidate)

    def test_absent_a_manifest_is_explicit_b_only_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(_a_score_provenance(Path(directory) / "gold.jsonl"),
                {"a_training_manifest_sha256": None, "a_prediction_signature_sha256": None})

    def test_all_score_paths_are_preflighted_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            protected = root / "scores.jsonl.manifest.json"
            protected.write_text("reviewed input", encoding="utf-8")
            existing = root / "existing.jsonl"
            existing.write_text("prior evidence", encoding="utf-8")
            for output in (protected, root / "scores.jsonl", model / "config.json", existing):
                args = SimpleNamespace(input=str(protected), output=str(output), model_dir=str(model),
                    device="cpu", batch_size=8, max_length=None, trust_remote_code=False)
                with self.subTest(output=output):
                    with patch("kpii_mvp.b_model.predict_scores") as prediction:
                        with self.assertRaises(ValueError):
                            score(args)
                        prediction.assert_not_called()
            self.assertEqual(protected.read_text(encoding="utf-8"), "reviewed input")
            self.assertEqual(existing.read_text(encoding="utf-8"), "prior evidence")


if __name__ == "__main__":
    unittest.main()
