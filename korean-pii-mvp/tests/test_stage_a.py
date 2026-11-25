"""Offline regression tests: gold boundaries, complete scope, overflow, head reset."""
from __future__ import annotations

import re
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kpii_mvp.stage_a import (build_label_inventory, validate_training_documents,
                             prepare_features, span_metrics, _load_training_model,
                             model_forward_inputs, predict,
                             _prediction_checkpoint_artifacts)
from kpii_mvp.tokenization import (IGNORE_INDEX, SpanPrediction, align_mentions,
                                   tokenize_document, decode_windows, resolve_spans)


class FakeTokenizer:
    """Faithful fast-tokenizer overflow shape, no HF downloads/dependencies."""
    is_fast = True
    padding_side = "right"

    def __init__(self, words: bool = False):
        self.words = words

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, **kwargs):
        pattern = r"\S+" if self.words else r"\S"
        offsets = [(m.start(), m.end()) for m in re.finditer(pattern, text)]
        capacity = kwargs["max_length"] - 2
        step = capacity - kwargs["stride"]
        rows = []
        start = 0
        while True:
            part = offsets[start:start + capacity]
            rows.append({"input_ids": [101] + list(range(start + 1, start + len(part) + 1)) + [102],
                         "attention_mask": [1] * (len(part) + 2),
                         "special_tokens_mask": [1] + [0] * len(part) + [1],
                         "offset_mapping": [(0, 0)] + part + [(0, 0)]})
            if start + capacity >= len(offsets):
                break
            start += step
        return {key: [row[key] for row in rows] for key in rows[0]}


def mention(start=0, end=2, entity_type="PER", text="가나"):
    return SimpleNamespace(start=start, end=end, entity_type=entity_type, text=text)


def document(doc_id, text, split, mentions=(), scope=("PER",), group=None):
    return SimpleNamespace(doc_id=doc_id, text=text, split=split, mentions=list(mentions),
                           annotation_scope=list(scope), source_group=group or doc_id)


LABELS = {"O": 0, "B-PER": 1, "I-PER": 2}


class AlignmentTests(unittest.TestCase):
    def test_exact_unicode_offsets_and_special_ignore(self):
        text = "앞 김민수 뒤"
        window = tokenize_document(FakeTokenizer(), text, max_length=20, stride=2)[0]
        labels, report = align_mentions(window, [mention(2, 5, text="김민수")], LABELS)
        self.assertEqual(labels, [-100, 0, 1, 2, 2, 0, -100])
        self.assertEqual(report.complete_mentions, 1)
        self.assertEqual(window.offsets[2:5], [(2, 3), (3, 4), (4, 5)])

    def test_entity_inside_one_token_is_ignored_never_expanded_or_negative(self):
        window = tokenize_document(FakeTokenizer(words=True), "김민수입니다 끝", 12, 1)[0]
        labels, report = align_mentions(window, [mention(0, 3, text="김민수")], LABELS)
        self.assertEqual(labels[1], IGNORE_INDEX)
        self.assertEqual(labels[2], LABELS["O"])
        self.assertEqual(report.unaligned_mentions, 1)
        self.assertEqual(report.complete_mentions, 0)
        with self.assertRaisesRegex(ValueError, "exactly token-aligned"):
            align_mentions(window, [mention(0, 3)], LABELS, "reject")

    def test_window_clipping_does_not_make_entity_tokens_negative(self):
        windows = tokenize_document(FakeTokenizer(), "abcdefghij", max_length=7, stride=3)
        entity = mention(3, 7, text="defg")
        first, first_report = align_mentions(windows[0], [entity], LABELS)
        self.assertEqual(first[-3:-1], [-100, -100])
        self.assertEqual(first_report.clipped_mentions, 1)
        second, second_report = align_mentions(windows[1], [entity], LABELS)
        self.assertEqual(second[2:6], [1, 2, 2, 2])
        self.assertEqual(second_report.complete_mentions, 1)

    def test_oversize_entity_is_reported_not_silently_labeled_o(self):
        doc = document("train-long", "abcdefghij", "train", [mention(0, 10, text="abcdefghij")])
        with self.assertRaisesRegex(ValueError, "No supervised token windows"):
            prepare_features([doc], FakeTokenizer(), LABELS, 6, 2)
        # Retain background elsewhere so a usable dataset can still emit report.
        doc = document("train-long", "x abcdefghij y", "train", [mention(2, 12, text="abcdefghij")])
        features, report = prepare_features([doc], FakeTokenizer(), LABELS, 6, 2,
                                            require_all_type_supervision=False)
        self.assertEqual(len(report["mentions_never_fully_supervised"]), 1)
        self.assertGreater(report["all_ignored_windows"], 0)
        self.assertTrue(features)
        with self.assertRaisesRegex(ValueError, "do not fit any complete token window"):
            prepare_features([doc], FakeTokenizer(), LABELS, 6, 2, "reject",
                             require_all_type_supervision=False)

    def test_o_only_training_supervision_fails_even_if_background_remains(self):
        doc = document("tr", "김민수입니다 끝", "train", [mention(0, 3, text="김민수")])
        with self.assertRaisesRegex(ValueError, "zero positive token supervision"):
            prepare_features([doc], FakeTokenizer(words=True), LABELS, 12, 1)
        # Full gold dev scoring remains possible even if the tokenizer cannot
        # represent those exact boundaries: do not filter gold to easier spans.
        _, report = prepare_features([doc], FakeTokenizer(words=True), LABELS, 12, 1,
                                      require_all_type_supervision=False)
        self.assertEqual(report["positive_tokens_by_type"], {"PER": 0})

    def test_every_training_entity_type_requires_positive_supervision(self):
        doc = document("tr", "김민수 이메일입니다 끝", "train",
                       [mention(0, 3, text="김민수"), mention(4, 7, "EMAIL", "이메일")],
                       scope=("PER", "EMAIL"))
        labels = {**LABELS, "B-EMAIL": 3, "I-EMAIL": 4}
        with self.assertRaisesRegex(ValueError, "EMAIL"):
            prepare_features([doc], FakeTokenizer(words=True), labels, 12, 1)

    def test_all_overflow_windows_reach_tail(self):
        text = "가" * 31 + "끝"
        windows = tokenize_document(FakeTokenizer(), text, max_length=8, stride=2)
        self.assertGreater(len(windows), 1)
        self.assertEqual(windows[-1].bounds[1], len(text))
        self.assertTrue(any(w.bounds[0] <= 31 < w.bounds[1] for w in windows))

    def test_no_unsafe_tokenizer_or_stride(self):
        slow = FakeTokenizer()
        slow.is_fast = False
        with self.assertRaisesRegex(ValueError, "fast tokenizer"):
            tokenize_document(slow, "가나다", 8, 2)
        with self.assertRaisesRegex(ValueError, "content capacity"):
            tokenize_document(FakeTokenizer(), "가나다", 8, 6)

    def test_silent_suffix_truncation_is_rejected(self):
        class BrokenTokenizer(FakeTokenizer):
            def __call__(self, text, **kwargs):
                rows = super().__call__(text, **kwargs)
                return {key: value[:1] for key, value in rows.items()}
        with self.assertRaisesRegex(ValueError, "omitted a non-whitespace"):
            tokenize_document(BrokenTokenizer(), "가나다라마바사아자차카", 6, 2)


class DecodingTests(unittest.TestCase):
    def test_bio_orphan_i_and_original_offsets(self):
        windows = tokenize_document(FakeTokenizer(), "앞 김민수 뒤", 20, 2)
        spans, _ = decode_windows(windows, [[0, 0, 2, 2, 2, 0, 0]], {v: k for k, v in LABELS.items()})
        self.assertEqual([(s.start, s.end, s.entity_type) for s in spans], [(2, 5, "PER")])

    def test_duplicate_window_predictions_collapse(self):
        spans, report = resolve_spans([
            SpanPrediction(10, 13, "PER", .8),
            SpanPrediction(10, 13, "PER", .9),
            SpanPrediction(10, 12, "ORG", .7),
        ])
        self.assertEqual([(s.start, s.end, s.entity_type) for s in spans], [(10, 13, "PER")])
        self.assertEqual(report["duplicates_removed"], 1)
        self.assertEqual(report["overlap_conflicts_removed"], 1)

    def test_long_entity_stitches_across_all_overflow_windows(self):
        windows = tokenize_document(FakeTokenizer(), "가나다라마바사아자차", max_length=6, stride=2)
        rows = [[0] + [1] + [2] * (len(w.input_ids) - 3) + [0] for w in windows]
        spans, report = decode_windows(windows, rows, {v: k for k, v in LABELS.items()})
        self.assertEqual([(s.start, s.end) for s in spans], [(0, 10)])
        self.assertGreater(report["stitched_fragments"], 0)
        self.assertEqual(report["remaining_edge_candidates"], 0)

    def test_adjacent_same_type_entities_are_not_merged(self):
        windows = tokenize_document(FakeTokenizer(), "가나다라", 10, 2)
        spans, _ = decode_windows(windows, [[0, 1, 2, 1, 2, 0]], {v: k for k, v in LABELS.items()})
        self.assertEqual([(s.start, s.end) for s in spans], [(0, 2), (2, 4)])

    def test_overlap_resolution_is_input_order_independent(self):
        candidates = [SpanPrediction(0, 4, "PER", .8), SpanPrediction(2, 6, "ORG", .8),
                      SpanPrediction(8, 9, "PER", .9)]
        self.assertEqual(resolve_spans(candidates), resolve_spans(list(reversed(candidates))))


class DatasetValidationTests(unittest.TestCase):
    def setUp(self):
        self.train = [document("tr", "가나 배경", "train", [mention()])]
        self.dev = [document("dv", "다라 문맥", "dev", [mention(text="다라")])]

    def test_native_labels_and_train_only_inventory(self):
        self.train[0].mentions[0].entity_type = "NATIVE_PERSON"
        labels = build_label_inventory(self.train)
        self.assertEqual(labels, {"O": 0, "B-NATIVE_PERSON": 1, "I-NATIVE_PERSON": 2})

    def test_complete_scope_required(self):
        self.dev[0].annotation_scope = []
        with self.assertRaisesRegex(ValueError, "incomplete annotation_scope"):
            validate_training_documents(self.train, self.dev, LABELS)

    def test_test_split_cannot_enter_training_or_selection(self):
        self.dev[0].split = "test"
        with self.assertRaisesRegex(ValueError, "split='test'"):
            validate_training_documents(self.train, self.dev, LABELS)

    def test_native_valid_split_is_explicit_dev_alias(self):
        self.dev[0].split = "valid"
        validate_training_documents(self.train, self.dev, LABELS)

    def test_unknown_dev_type_does_not_extend_inventory(self):
        self.dev[0].mentions[0].entity_type = "EMAIL"
        with self.assertRaisesRegex(ValueError, "not present in train"):
            validate_training_documents(self.train, self.dev, LABELS)

    def test_source_group_leakage_rejected(self):
        self.dev[0].source_group = self.train[0].source_group
        with self.assertRaisesRegex(ValueError, "source groups overlap"):
            validate_training_documents(self.train, self.dev, LABELS)

    def test_overlapping_gold_rejected(self):
        self.train[0].mentions.append(mention(1, 3))
        with self.assertRaisesRegex(ValueError, "overlapping gold"):
            validate_training_documents(self.train, self.dev, LABELS)

    def test_exact_span_metrics_do_not_reward_boundary_widening(self):
        predictions = [document("dv", "다라 문맥", "dev", [mention(0, 3)])]
        result = span_metrics(self.dev, predictions)
        self.assertEqual(result["correct"], 0)
        self.assertEqual(result["f1"], 0)
        self.assertEqual(span_metrics(self.dev, self.dev)["f1"], 1)

    def test_metrics_reject_same_id_with_different_source_text(self):
        predictions = [document("dv", "가나 문맥", "dev", [mention()])]
        with self.assertRaisesRegex(ValueError, "original document text"):
            span_metrics(self.dev, predictions)

    def test_metrics_reject_missing_and_duplicate_document_ids(self):
        with self.assertRaisesRegex(ValueError, "document ids do not match"):
            span_metrics(self.dev, [])
        with self.assertRaisesRegex(ValueError, "unique document ids"):
            span_metrics(self.dev + self.dev, self.dev)


class HeadTransferTests(unittest.TestCase):
    def fake_transformers(self):
        events = []
        class Config:
            architectures = ["ExampleForTokenClassification"]
            label2id = {"O": 0, "B-NATIVE": 1, "I-NATIVE": 2}
        class AutoConfig:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return Config()
        class Base:
            def state_dict(self):
                return {"encoder.weight": "weights"}
            def load_state_dict(self, state, strict):
                events.append(("transfer", state, strict))
        class AutoModel:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                events.append(("load", kwargs))
                return SimpleNamespace(base_model=Base())
            @staticmethod
            def from_config(config, **kwargs):
                events.append(("new_head", config.label2id))
                return SimpleNamespace(base_model=Base())
        return SimpleNamespace(AutoConfig=AutoConfig, AutoModelForTokenClassification=AutoModel), events

    def test_changed_labels_require_explicit_head_reset(self):
        transformers, events = self.fake_transformers()
        with self.assertRaisesRegex(ValueError, "--reset-head"):
            _load_training_model(transformers, "not-a-local-model", LABELS, False, False)
        self.assertEqual(events, [])

    def test_explicit_reset_transfers_encoder_strictly_and_reinitializes_head(self):
        transformers, events = self.fake_transformers()
        _, report = _load_training_model(transformers, "not-a-local-model", LABELS, True, False)
        self.assertTrue(report["explicit_head_reset"])
        self.assertTrue(report["encoder_transfer_strict"])
        self.assertEqual(events[-1], ("transfer", {"encoder.weight": "weights"}, True))
        self.assertTrue(all("ignore_mismatched_sizes" not in event[1]
                            for event in events if event[0] == "load"))


class ModelInputTests(unittest.TestCase):
    def test_bert_segment_ids_do_not_reach_modernbert_train_or_predict(self):
        class SegmentTokenizer(FakeTokenizer):
            def __call__(self, text, **kwargs):
                self.requested_segment_ids = kwargs.get("return_token_type_ids")
                encoded = super().__call__(text, **kwargs)
                # Reproduce a mismatched tokenizer even if it ignores the flag.
                encoded["token_type_ids"] = [[0] * len(row) for row in encoded["input_ids"]]
                return encoded
        class ModernBert:
            def forward(self, input_ids, attention_mask, labels=None):
                return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        tokenizer = SegmentTokenizer()
        window = tokenize_document(tokenizer, "김민수", 10, 2)[0]
        self.assertIs(tokenizer.requested_segment_ids, False)
        self.assertIn("token_type_ids", window.model_inputs())
        model = ModernBert()
        for training in (True, False):
            batch = window.model_inputs()
            if training:
                batch["labels"] = [-100, 1, 2, 2, -100]
            filtered = model_forward_inputs(model, batch)
            self.assertNotIn("token_type_ids", filtered)
            self.assertEqual(filtered["input_ids"], batch["input_ids"])
            self.assertEqual(filtered["attention_mask"], batch["attention_mask"])
            self.assertEqual("labels" in filtered, training)
            self.assertEqual(model.forward(**filtered)["labels"], batch.get("labels"))

    def test_required_core_inputs_are_never_silently_dropped(self):
        class IncompatibleModel:
            def forward(self, input_ids):
                return input_ids
        with self.assertRaisesRegex(ValueError, "required inputs"):
            model_forward_inputs(IncompatibleModel(), {"input_ids": [1], "attention_mask": [1], "labels": [0]})

    def test_forward_with_kwargs_retains_explicitly_supported_auxiliaries(self):
        class FlexibleModel:
            def forward(self, input_ids, **kwargs):
                return kwargs
        batch = {"input_ids": [1], "attention_mask": [1], "token_type_ids": [0], "labels": [0]}
        self.assertEqual(model_forward_inputs(FlexibleModel(), batch), batch)


class PredictionProvenanceTests(unittest.TestCase):
    def test_checkpoint_binding_includes_weights_tokenizer_and_runtime_sources(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            assets = ("config.json", "model-00001-of-00002.safetensors", "model.safetensors.index.json",
                      "tokenizer.json", "vocab.txt", "tokenization_custom.py")
            for name in assets + ("training_manifest.json", "README.md"):
                (root / name).write_text(name, encoding="utf-8")
            before = _prediction_checkpoint_artifacts(root)
            self.assertEqual(set(before), set(assets))
            for name in ("model-00001-of-00002.safetensors", "tokenizer.json"):
                (root / name).write_text("modified", encoding="utf-8")
                self.assertNotEqual(_prediction_checkpoint_artifacts(root)[name], before[name])
            self.assertEqual(before["config.json"], hashlib.sha256(b"config.json").hexdigest())

    def test_predict_records_artifacts_and_rejects_a_checkpoint_change_during_forward(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            model_dir = root / "best"
            model_dir.mkdir()
            for name in ("config.json", "model.safetensors", "tokenizer.json"):
                (model_dir / name).write_text("original " + name, encoding="utf-8")
            training = {"stage": "A", "label2id": LABELS, "configuration": {"max_length": 512, "stride": 128}}
            (model_dir / "training_manifest.json").write_text(json.dumps(training), encoding="utf-8")
            source = root / "input.jsonl"
            source.write_text("source document", encoding="utf-8")
            output = root / "predictions.jsonl"
            args = SimpleNamespace(input=str(source), output=str(output), model_dir=str(model_dir),
                manifest=None, max_length=None, stride=None, batch_size=2, device="cpu", trust_remote_code=False)
            model = SimpleNamespace(config=SimpleNamespace(label2id=LABELS), to=lambda _: None)
            transformers = SimpleNamespace(
                AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: object()),
                AutoModelForTokenClassification=SimpleNamespace(from_pretrained=lambda *a, **kw: model))
            prediction = SimpleNamespace(to_dict=lambda: {"doc_id": "test", "mentions": []})
            with patch("kpii_mvp.stage_a._load_documents", return_value=[]), \
                 patch("kpii_mvp.stage_a._ml_dependencies", return_value=(None, transformers)), \
                 patch("kpii_mvp.stage_a._device", return_value="cpu"), \
                 patch("kpii_mvp.stage_a.predict_documents", return_value=([prediction], {})):
                manifest = predict(args)
            self.assertEqual(manifest["checkpoint_artifacts"], _prediction_checkpoint_artifacts(model_dir))
            self.assertEqual(set(manifest["checkpoint_artifacts"]), {"config.json", "model.safetensors", "tokenizer.json"})
            saved = json.loads(Path(str(output) + ".manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["checkpoint_artifacts"], manifest["checkpoint_artifacts"])

            def change_weights(*_args, **_kwargs):
                (model_dir / "model.safetensors").write_text("changed during prediction", encoding="utf-8")
                return [prediction], {}
            args.output = str(root / "changed-output.jsonl")
            with patch("kpii_mvp.stage_a._load_documents", return_value=[]), \
                 patch("kpii_mvp.stage_a._ml_dependencies", return_value=(None, transformers)), \
                 patch("kpii_mvp.stage_a._device", return_value="cpu"), \
                 patch("kpii_mvp.stage_a.predict_documents", side_effect=change_weights):
                with self.assertRaisesRegex(ValueError, "checkpoint changed during prediction"):
                    predict(args)
            self.assertFalse(Path(args.output).exists())


if __name__ == "__main__":
    unittest.main()
