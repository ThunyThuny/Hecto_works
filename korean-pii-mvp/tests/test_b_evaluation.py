"""Small artificial unit fixtures, never a B training/evaluation corpus."""

import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kpii_mvp.b_data import document_fingerprint, document_identities, pending_review
from kpii_mvp.b_evaluation import (
    FIXED_POLICY_ID, _preflight, _score_provenance, apply_calibrated_b,
    calibrate_b, evaluate_ab, load_scores, main, validate_scores,
)
from kpii_mvp.data import write_jsonl
from kpii_mvp.schema import Document, Mention


MODEL_SHA = "a" * 64


def fixture(doc_id="dev1", split="valid", group="dev-group", text="Ada PublicCo Secret"):
    mentions = [Mention("m1", 0, 3, text[:3], "NAME", "MASK"),
                Mention("m2", 4, 12, text[4:12], "WORKPLACE", "KEEP"),
                Mention("m3", 13, 19, text[13:19], "NAME", "MASK")]
    gold = Document(doc_id, text, mentions, group, ["NAME", "WORKPLACE"],
                    split=split, policy_id=FIXED_POLICY_ID,
                    trusted_context={"approved_public_orgs": [text[4:12]]})
    candidates = replace(gold, mentions=[replace(m, action=None, reason_code=None) for m in mentions])
    return gold, candidates


def reviews(*documents):
    result = {}
    for doc in documents:
        review = pending_review(doc)
        review.update(review_status="APPROVED", reviewer_id="unit_test_reviewer",
                      reviewed_at="2026-10-19T09:00:00+09:00",
                      candidate_inventory_complete=True, action_labels_complete=True)
        result[doc.doc_id] = review
    return result


def scores(document, probabilities=None):
    probabilities = probabilities or {m.mention_id: 0.9 for m in document.mentions}
    return {"doc_id": document.doc_id, "document_sha256": document_fingerprint(document),
            "policy_id": FIXED_POLICY_ID, "mentions": [
                {"mention_id": m.mention_id, "start": m.start, "end": m.end,
                 "entity_type": m.entity_type, "p_keep": probabilities[m.mention_id],
                 "available": probabilities[m.mention_id] is not None,
                 "reason": "unit_fixture"}
                for m in document.mentions]}


def calibration(gold=None, candidates=None, score=None, **kwargs):
    if gold is None:
        gold, candidates = fixture()
    return calibrate_b([gold], [candidates], [score or scores(candidates)], reviews(gold),
                       model_manifest_sha256=MODEL_SHA, **kwargs)


class ScoreValidationTests(unittest.TestCase):
    def setUp(self):
        self.gold, self.candidate = fixture()
        self.record = scores(self.candidate)

    def test_valid_exact_candidate_scores(self):
        result = validate_scores([self.candidate], [self.record])
        self.assertEqual(result["dev1"]["m2"]["p_keep"], 0.9)

    def test_missing_extra_duplicate_and_changed_boundaries_fail(self):
        cases = []
        for edit in (lambda r: r["mentions"].pop(),
                     lambda r: r["mentions"].append(copy.deepcopy(r["mentions"][0])),
                     lambda r: r["mentions"][0].update(start=True),
                     lambda r: r["mentions"][0].update(end=2),
                     lambda r: r["mentions"][0].update(entity_type="SCHOOL"),
                     lambda r: r.update(text="raw text is disallowed"),
                     lambda r: r.update(document_sha256="b" * 64)):
            record = copy.deepcopy(self.record)
            edit(record)
            cases.append([record])
        cases.extend(([], [self.record, self.record]))
        for records in cases:
            with self.subTest(records=records), self.assertRaises(ValueError):
                validate_scores([self.candidate], records)

    def test_score_probability_validation(self):
        for probability, available in ((float("nan"), True), (float("inf"), True),
                                       (-0.1, True), (1.1, True), (True, True),
                                       (None, True), (0.8, False), (0.8, 1)):
            record = copy.deepcopy(self.record)
            record["mentions"][0].update(p_keep=probability, available=available)
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                validate_scores([self.candidate], [record])
        record = copy.deepcopy(self.record)
        record["mentions"][0].update(p_keep=None, available=False)
        validate_scores([self.candidate], [record])

    def test_actions_and_metadata_are_bound_by_fingerprint(self):
        edited = replace(self.candidate, mentions=[replace(self.candidate.mentions[0], action="KEEP")]
                         + self.candidate.mentions[1:])
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_scores([edited], [self.record])

    def test_json_duplicate_keys_and_nonfinite_values_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"
            for text in ('{"doc_id":"x","doc_id":"y"}', '{"p_keep":NaN}'):
                path.write_text(text + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_scores(path)


class CalibrationTests(unittest.TestCase):
    def test_selects_highest_safe_threshold_and_masks_names(self):
        gold, candidate = fixture()
        selected = calibration(gold, candidate, scores(candidate, {"m1": 1.0, "m2": 0.9, "m3": 1.0}))
        self.assertEqual(selected["threshold"], 0.9)
        self.assertEqual(selected["baseline"]["overall"]["overmask_rate"], 1.0)
        self.assertEqual(selected["selected"]["overall"]["overmask_rate"], 0.0)
        self.assertEqual(selected["selected"]["overall"]["leak_rate"], 0.0)
        applied = apply_calibrated_b([candidate], [scores(candidate)], selected, model_manifest_sha256=MODEL_SHA)
        self.assertEqual([m["action"] for m in applied["decisions"][0]["mentions"]], ["MASK", "KEEP", "MASK"])
        self.assertEqual(applied["masked_documents"][0]["masked_text"], "[MASK] PublicCo [MASK]")

    def test_no_safe_improvement_disables_keep(self):
        text = "PublicCo PublicCo"
        gold = Document("dev", text, [Mention("m1", 0, 8, "PublicCo", "WORKPLACE", "MASK"),
                                     Mention("m2", 9, 17, "PublicCo", "WORKPLACE", "KEEP")],
                        "dev", ["WORKPLACE"], split="dev", policy_id=FIXED_POLICY_ID,
                        trusted_context={"approved_public_orgs": ["PublicCo"]})
        candidate = replace(gold, mentions=[replace(m, action=None) for m in gold.mentions])
        result = calibration(gold, candidate, scores(candidate, {"m1": 0.99, "m2": 0.9}))
        self.assertFalse(result["keep_enabled"])
        self.assertIsNone(result["threshold"])

    def test_missing_and_partial_a_spans_remain_in_gold_leak_denominator(self):
        gold, candidate = fixture()
        candidate = replace(candidate, mentions=[replace(candidate.mentions[0], end=2, text="Ad"),
                                                  candidate.mentions[1]])
        result = calibration(gold, candidate)
        self.assertEqual(result["baseline"]["overall"]["gold_mask_mentions"], 2)
        self.assertEqual(result["baseline"]["overall"]["leaked_mask_mentions"], 2)
        self.assertEqual(result["selected"]["overall"]["leaked_mask_mentions"], 2)

    def test_unavailable_score_fails_closed(self):
        gold, candidate = fixture()
        result = calibration(gold, candidate, scores(candidate, {"m1": 1.0, "m2": None, "m3": 1.0}))
        self.assertFalse(result["keep_enabled"])

    def test_calibration_rejects_test_unreviewed_null_and_policy_conflicts(self):
        gold, candidate = fixture()
        bad_gold = [replace(gold, split="test"),
                    replace(gold, mentions=[replace(gold.mentions[0], action=None)] + gold.mentions[1:]),
                    replace(gold, trusted_context={})]
        for document in bad_gold:
            changed_candidate = replace(candidate, split=document.split, trusted_context=document.trusted_context)
            with self.subTest(document=document), self.assertRaises(ValueError):
                calibration(document, changed_candidate)
        pending = {gold.doc_id: pending_review(gold)}
        with self.assertRaises(ValueError):
            calibrate_b([gold], [candidate], [scores(candidate)], pending, model_manifest_sha256=MODEL_SHA)

    def test_repeated_normalized_dev_text_cannot_weight_the_threshold_twice(self):
        gold, candidate = fixture()
        duplicate, duplicate_candidate = fixture("dev-copy", "dev", "other-group", "Ada\tPublicCo Secret")
        with self.assertRaisesRegex(ValueError, "duplicate text"):
            calibrate_b([gold, duplicate], [candidate, duplicate_candidate],
                        [scores(candidate), scores(duplicate_candidate)], reviews(gold, duplicate),
                        model_manifest_sha256=MODEL_SHA)


class IntegratedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.calib = calibration()
        self.gold, self.candidate = fixture("test1", "test", "test-group", "Eve OpenOrgX Hidden")

    def test_frozen_test_reports_baseline_and_added_leaks(self):
        result = evaluate_ab([self.gold], [self.candidate], [scores(self.candidate)], reviews(self.gold),
                             self.calib, model_manifest_sha256=MODEL_SHA)
        self.assertEqual(result["calibrated_b"]["overall"]["leak_rate"], 0)
        self.assertEqual(result["delta_b_minus_baseline"]["overmask_rate"], -1.0)
        self.assertNotIn("Ada", json.dumps(self.calib, ensure_ascii=False))

    def test_different_model_and_invalid_threshold_rejected(self):
        for changed in ({"b_manifest_sha256": "b" * 64}, {"threshold": True},
                        {"threshold": None}, {"policy_sha256": "b" * 64}):
            artifact = {**self.calib, **changed}
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                evaluate_ab([self.gold], [self.candidate], [scores(self.candidate)], reviews(self.gold),
                            artifact, model_manifest_sha256=MODEL_SHA)

    def test_dev_text_doc_id_and_source_group_reuse_rejected(self):
        reused = [fixture("new", "test", "other", "Ada PublicCo Secret"),
                  fixture("dev1", "test", "other", "Eve OpenOrgX Hidden"),
                  fixture("new", "test", "dev-group", "Eve OpenOrgX Hidden")]
        for gold, candidate in reused:
            with self.subTest(gold=gold), self.assertRaisesRegex(ValueError, "reuses"):
                evaluate_ab([gold], [candidate], [scores(candidate)], reviews(gold), self.calib,
                            model_manifest_sha256=MODEL_SHA)

    def test_sanitized_output_has_only_two_fields(self):
        result = apply_calibrated_b([self.candidate], [scores(self.candidate)], self.calib,
                                   model_manifest_sha256=MODEL_SHA)
        self.assertEqual(set(result["masked_documents"][0]), {"doc_id", "masked_text"})
        for mention in result["decisions"][0]["mentions"]:
            self.assertNotIn("text", mention)

    def test_type_and_exact_metadata_gate_even_with_probability_one(self):
        tokens = [("123", "PHONE"), ("x@y", "EMAIL"), ("Seoul", "ADDRESS"),
                  ("Uni", "SCHOOL"), ("Boss", "POSITION"), ("acct", "ACCOUNT")]
        text = " ".join(token for token, _ in tokens)
        mentions, offset = [], 0
        for index, (token, kind) in enumerate(tokens):
            mentions.append(Mention(str(index), offset, offset + len(token), token, kind))
            offset += len(token) + 1
        document = Document("apply", text, mentions, "inference", [kind for _, kind in tokens],
                            policy_id=FIXED_POLICY_ID,
                            trusted_context={"approved_public_contacts": ["x@y", "Seoul-extra"],
                                             "approved_public_orgs": ["Uni"],
                                             "approved_public_positions": ["Boss"]})
        result = apply_calibrated_b([document], [scores(document, {m.mention_id: 1.0 for m in mentions})],
                                    self.calib, model_manifest_sha256=MODEL_SHA)
        self.assertEqual([m["action"] for m in result["decisions"][0]["mentions"]],
                         ["MASK", "KEEP", "MASK", "KEEP", "KEEP", "MASK"])

    def test_fixed_a_and_prediction_configuration_are_bound(self):
        artifact = {**self.calib, "a_training_manifest_sha256": "b" * 64,
                    "a_prediction_signature_sha256": "c" * 64}
        common = dict(candidate_documents=[self.candidate], scores=[scores(self.candidate)],
                      calibration=artifact, model_manifest_sha256=MODEL_SHA)
        for extra in ({}, {"a_training_manifest_sha256": "b" * 64},
                      {"a_training_manifest_sha256": "b" * 64, "a_prediction_signature_sha256": "d" * 64}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                apply_calibrated_b(**common, **extra)
        apply_calibrated_b(**common, a_training_manifest_sha256="b" * 64,
                           a_prediction_signature_sha256="c" * 64)

    def test_whitespace_variant_of_dev_text_is_rejected(self):
        gold, candidate = fixture("new-test", "test", "new-test-group", "Ada\tPublicCo Secret")
        with self.assertRaisesRegex(ValueError, "reuses"):
            evaluate_ab([gold], [candidate], [scores(candidate)], reviews(gold), self.calib,
                        model_manifest_sha256=MODEL_SHA)


class ProvenanceTests(unittest.TestCase):
    def test_cli_calibrate_evaluate_and_apply_use_sidecar_bindings(self):
        dev, dev_candidate = fixture()
        test, test_candidate = fixture("test1", "test", "test-group", "Eve OpenOrgX Hidden")
        train, _ = fixture("train1", "train", "train-group", "Tom BoardOrg Secret")
        metadata = {"data": {"train": {"identities": document_identities([train])},
                             "dev": {"identities": document_identities([dev])}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_manifest = root / "b_manifest.json"
            model_manifest.write_text("{}", encoding="utf-8")
            file_digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

            def write_inputs(gold, candidate):
                prefix = root / gold.doc_id
                gold_path = prefix.with_suffix(".gold.jsonl")
                candidate_path = prefix.with_suffix(".candidate.jsonl")
                reviews_path = prefix.with_suffix(".reviews.jsonl")
                scores_path = prefix.with_suffix(".scores.jsonl")
                write_jsonl(gold_path, [gold])
                write_jsonl(candidate_path, [candidate])
                reviews_path.write_text(json.dumps(reviews(gold)[gold.doc_id]) + "\n", encoding="utf-8")
                scores_path.write_text(json.dumps(scores(candidate)) + "\n", encoding="utf-8")
                Path(str(scores_path) + ".manifest.json").write_text(json.dumps({
                    "stage": "B_SCORES_V1", "policy_id": FIXED_POLICY_ID,
                    "b_manifest_sha256": file_digest(model_manifest),
                    "input_sha256": file_digest(candidate_path), "output_sha256": file_digest(scores_path),
                    "a_training_manifest_sha256": "b" * 64,
                    "a_prediction_signature_sha256": "c" * 64}), encoding="utf-8")
                return ["--gold", str(gold_path), "--reviews", str(reviews_path),
                        "--candidates", str(candidate_path), "--scores", str(scores_path),
                        "--model-manifest", str(model_manifest)]

            cal_path = root / "calibration.json"
            with patch("kpii_mvp.b_model.load_checkpoint_manifest", return_value=metadata), redirect_stdout(io.StringIO()):
                main(["calibrate"] + write_inputs(dev, dev_candidate) + ["--output", str(cal_path)])
                test_args = write_inputs(test, test_candidate)
                main(["evaluate"] + test_args + ["--calibration", str(cal_path), "--output-dir", str(root / "eval")])
                main(["apply"] + test_args[4:] + ["--calibration", str(cal_path), "--output-dir", str(root / "apply")])
                reused = replace(train, split="test")
                reused_candidate = replace(reused, mentions=[replace(m, action=None) for m in reused.mentions])
                with self.assertRaisesRegex(ValueError, "B training"):
                    main(["evaluate"] + write_inputs(reused, reused_candidate)
                         + ["--calibration", str(cal_path), "--output-dir", str(root / "reused")])
            self.assertEqual(json.loads(cal_path.read_text())["threshold"], 0.9)
            evaluation = json.loads((root / "eval" / "ab-evaluation.json").read_text())
            self.assertEqual(evaluation["delta_b_minus_baseline"]["leak_rate"], 0)
            self.assertEqual(set(json.loads((root / "apply" / "masked.jsonl").read_text())),
                             {"doc_id", "masked_text"})

    def test_edited_scores_or_model_or_input_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, candidates, score_file = root / "b_manifest.json", root / "candidate.jsonl", root / "score.jsonl"
            model.write_text("{}", encoding="utf-8")
            candidates.write_text("candidate", encoding="utf-8")
            score_file.write_text("score", encoding="utf-8")
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            sidecar = Path(str(score_file) + ".manifest.json")
            sidecar.write_text(json.dumps({"stage": "B_SCORES_V1", "policy_id": FIXED_POLICY_ID,
                                           "b_manifest_sha256": digest(model),
                                           "input_sha256": digest(candidates),
                                           "output_sha256": digest(score_file)}), encoding="utf-8")
            _score_provenance(score_file, candidates, model, verify_checkpoint=False)
            for file in (model, candidates, score_file):
                old = file.read_bytes()
                file.write_bytes(old + b"edit")
                with self.subTest(file=file), self.assertRaises(ValueError):
                    _score_provenance(score_file, candidates, model, verify_checkpoint=False)
                file.write_bytes(old)

    def test_outputs_cannot_overwrite_inputs_or_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing"
            path.write_text("source", encoding="utf-8")
            for inputs, outputs in (([path], [path]), ([], [path]), ([], [path, path])):
                with self.assertRaises(ValueError):
                    _preflight(inputs, outputs)


if __name__ == "__main__":
    unittest.main()
