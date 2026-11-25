"""Small source fixtures verify data integrity, not the unverified research score."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from kpii_mvp.data import read_jsonl, write_jsonl
from kpii_mvp.kdpii import (
    KDPII_V1_LABELS, _json_argument, aggregate_conversations, check_split_integrity,
    inspect_file, parse_bio_record, parse_kdpii_v1_record, parse_span_record,
    prepare_dataset,
)
from kpii_mvp.schema import Document, Mention


def temporary_directory():
    # Keep fixtures inside the workspace on restricted Windows hosts.
    return tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)


def canonical(doc_id="a", text="😀김가람님", *, split=None, scope=None):
    return {
        "doc_id": doc_id, "text": text, "source_group": "group-" + doc_id,
        "annotation_scope": scope or ["PS_NAME"], "split": split,
        "mentions": [{"mention_id": "m1", "start": 1, "end": 4,
            "text": "김가람", "entity_type": "PS_NAME"}],
    }


def native(sent_idx="Data1.1_1_0", text="😀김가람님", *, negative=False):
    tags = ["O"] * len(text)
    annotations = []
    if not negative:
        start = text.index("김가람")
        end = start + 3
        annotations = [{"id": 0, "form": "김가람", "label": "PS_NAME", "begin": start, "end": end}]
        tags[start] = "B-PS_NAME"
        tags[start + 1:end] = ["I-PS_NAME"] * (end - start - 1)
    return {"sent_idx": sent_idx, "sentence": text, "PII_set": annotations,
        "sent_seq": list(text), "labelling_seq": tags}


class SchemaTests(unittest.TestCase):
    def test_unicode_half_open_roundtrip_and_frozen_fields(self):
        document = Document.from_dict(canonical())
        self.assertEqual(document.text[1:4], document.mentions[0].text)
        self.assertIsNone(document.mentions[0].action)
        self.assertEqual(Document.from_dict(document.to_dict()), document)
        with self.assertRaises(FrozenInstanceError):
            document.doc_id = "changed"

    def test_no_action_is_distinct_from_keep(self):
        mention = Mention("m", 0, 1, "가", "NATIVE_NEW_TYPE")
        self.assertIsNone(mention.to_dict()["action"])
        self.assertEqual(replace(mention, action="KEEP").to_dict()["action"], "KEEP")

    def test_missing_required_fields_are_not_silent_negatives(self):
        for field in ("mentions", "annotation_scope", "doc_id", "source_group"):
            record = canonical()
            del record[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                Document.from_dict(record)

    def test_off_by_one_utf16_and_wrong_substring_rejected(self):
        for start, end, surface in ((1, 5, "김가람"), (2, 5, "김가람"), (1, 4, "홍길동")):
            record = canonical()
            record["mentions"][0].update(start=start, end=end, text=surface)
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                Document.from_dict(record)

    def test_overlap_duplicate_ids_and_unknown_type_rejected(self):
        record = canonical()
        for mutation in ("overlap", "id", "unknown"):
            changed = json.loads(json.dumps(record))
            if mutation == "unknown":
                changed["mentions"][0]["entity_type"] = "UNDECLARED"
            else:
                second = dict(changed["mentions"][0])
                if mutation == "overlap":
                    second["mention_id"] = "m2"
                changed["mentions"].append(second)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                Document.from_dict(changed)

    def test_invalid_offsets_actions_and_nonfinite_context_rejected(self):
        with self.assertRaises(ValueError):
            Mention("m", True, 2, "가", "PS_NAME")
        with self.assertRaises(ValueError):
            Mention("m", 0, 1, "가", "PS_NAME", action="maybe")
        with self.assertRaises(ValueError):
            Document("a", "가", [], "g", ["PS_NAME"], trusted_context={"score": float("nan")})

    def test_explicit_negative_and_jsonl_roundtrip(self):
        negative = Document("n", "일반 문장", [], "negative-g", ["PS_NAME"])
        with temporary_directory() as temp:
            target = Path(temp) / "nested" / "data.jsonl"
            docs = [Document.from_dict(canonical()), negative]
            write_jsonl(target, docs)
            self.assertEqual(read_jsonl(target), docs)
            original = target.read_bytes()
            with self.assertRaises(ValueError):
                write_jsonl(target, [docs[0], docs[0]])
            self.assertEqual(target.read_bytes(), original)


class AdapterTests(unittest.TestCase):
    def test_spans_require_annotations_and_full_scope(self):
        for missing in ("mentions", "annotation_scope"):
            record = canonical()
            del record[missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                parse_span_record(record)
        with self.assertRaises(ValueError):
            parse_span_record(canonical(), config={"offset_unit": "utf16"})

    def test_generic_custom_keys_and_explicit_inclusive_end(self):
        source = {"uid": "x", "content": "김가람", "spans": [{"s": 0, "e": 2, "surface": "김가람", "kind": "CUSTOM"}]}
        config = {"id_key": "uid", "text_key": "content", "mentions_key": "spans",
            "start_key": "s", "end_key": "e", "mention_text_key": "surface", "label_key": "kind",
            "end_inclusive": True, "annotation_scope": ["CUSTOM"]}
        result = parse_span_record(source, config=config)
        self.assertEqual(result.mentions[0].end, 3)
        self.assertTrue(result.source_group.startswith("text-sha256:"))

    def test_bio_exact_character_reconstruction_with_emoji(self):
        record = canonical()
        record.update(tokens=list(record["text"]), bio_tags=["O", "B-PS_NAME", "I-PS_NAME", "I-PS_NAME", "O"])
        result = parse_bio_record(record)
        self.assertEqual((result.mentions[0].start, result.mentions[0].end), (1, 4))
        record["tokens"] = ["김가람"]
        record["bio_tags"] = ["B-PS_NAME"]
        with self.assertRaises(ValueError):
            parse_bio_record(record)

    def test_bio_offsets_allow_whitespace_but_cannot_omit_content(self):
        record = {"doc_id": "b", "text": "김 가람", "annotation_scope": ["PS_NAME"],
            "tokens": ["김", "가람"], "bio_tags": ["B-PS_NAME", "I-PS_NAME"], "offsets": [[0, 1], [2, 4]]}
        result = parse_bio_record(record)
        self.assertEqual(result.mentions[0].text, "김 가람")
        record["text"] = "김X가람"
        with self.assertRaises(ValueError):
            parse_bio_record(record)

    def test_invalid_bio_is_not_repaired(self):
        for tags in (["I-PS_NAME"], ["B-PS_NAME", "I-OTHER"], ["S-PS_NAME"]):
            text = "가" * len(tags)
            record = {"doc_id": "b", "text": text, "annotation_scope": ["PS_NAME", "OTHER"],
                "tokens": list(text), "bio_tags": tags}
            with self.subTest(tags=tags), self.assertRaises(ValueError):
                parse_bio_record(record)

    def test_native_verified_fields_and_full_33_scope(self):
        result = parse_kdpii_v1_record(native(), split="test")
        self.assertEqual(result.source_group, "Data1.1")
        self.assertEqual(len(result.annotation_scope), 33)
        self.assertEqual(set(result.annotation_scope), set(KDPII_V1_LABELS))
        self.assertEqual(result.mentions[0].text, "김가람")
        self.assertIsNone(result.mentions[0].action)

    def test_native_annotations_missing_or_disagree_are_errors(self):
        absent = native()
        del absent["PII_set"]
        disagreement = native()
        disagreement["PII_set"] = []
        unknown = native()
        unknown["PII_set"][0]["label"] = "UNKNOWN"
        for record in (absent, disagreement, unknown):
            with self.assertRaises(ValueError):
                parse_kdpii_v1_record(record)

    def test_aggregate_shifts_offsets_without_normalizing_emoji(self):
        first = parse_kdpii_v1_record(native(), split="train")
        second = parse_kdpii_v1_record(native("Data1.1_2_0", "다음 김가람"), split="train")
        result = aggregate_conversations([first, second])[0]
        self.assertEqual(result.text, "😀김가람님\n다음 김가람")
        self.assertEqual(result.mentions[1].start, len(first.text) + 1 + 3)
        result.validate()

    def test_duplicate_text_and_cross_split_group_rejected(self):
        first = Document.from_dict(canonical(split="train"))
        duplicate = replace(first, doc_id="b", source_group="other", split="test")
        with self.assertRaises(ValueError):
            check_split_integrity({"train": [first], "test": [duplicate]})
        changed = Document("b", "다른 문장", [], first.source_group, ["PS_NAME"], split="test")
        with self.assertRaises(ValueError):
            check_split_integrity({"train": [first], "test": [changed]})

    def test_prepare_preserves_splits_hashes_and_does_not_overwrite(self):
        with temporary_directory() as temp:
            folder = Path(temp)
            paths = {}
            for split, text in (("train", "훈련 김가람"), ("valid", "검증 김가람"), ("test", "평가 김가람")):
                path = folder / (split + ".json")
                path.write_text(json.dumps([native(f"{split}.1_1_0", text)]), encoding="utf-8")
                paths[split] = path
            manifest = prepare_dataset(**paths, output_dir=folder / "ready", adapter="kdpii-v1")
            self.assertEqual(manifest["document_counts"], {"train": 1, "valid": 1, "test": 1})
            self.assertEqual(len(manifest["native_annotation_scope"]), 33)
            self.assertEqual(len(manifest["input_files"]["train"]["sha256"]), 64)
            self.assertEqual(read_jsonl(folder / "ready" / "valid.jsonl")[0].split, "valid")
            with self.assertRaises(FileExistsError):
                prepare_dataset(**paths, output_dir=folder / "ready", adapter="kdpii-v1")

    def test_quarantine_excludes_whole_conversation_with_audit_counts(self):
        with temporary_directory() as temp:
            folder = Path(temp)
            bad = native("train.bad_2_0", "오류 김가람")
            bad["PII_set"][0]["end"] -= 1
            train = [native("train.bad_1_0", "같은대화 김가람"), bad, native("train.good_1_0", "정상 김가람")]
            paths = {}
            for split, records in (("train", train), ("valid", [native("valid.a_1_0", "검증 김가람")]),
                                   ("test", [native("test.a_1_0", "평가 김가람")])):
                path = folder / (split + ".json")
                path.write_text(json.dumps(records), encoding="utf-8")
                paths[split] = path
            with self.assertRaises(ValueError):
                prepare_dataset(**paths, output_dir=folder / "strict", adapter="kdpii-v1")
            self.assertFalse((folder / "strict").exists())
            manifest = prepare_dataset(**paths, output_dir=folder / "ready", adapter="kdpii-v1", invalid_policy="quarantine")
            self.assertEqual(manifest["quarantine"]["train"]["invalid_sentence_count"], 1)
            self.assertEqual(manifest["quarantine"]["train"]["quarantined_sentence_count"], 2)
            self.assertEqual(manifest["quarantine"]["train"]["quarantined_conversation_count"], 1)
            self.assertEqual(manifest["document_counts"]["train"], 1)
            audit = json.dumps(manifest["quarantine"], ensure_ascii=False)
            self.assertNotIn("김가람", audit)
            self.assertNotIn("train.bad", audit)

    def test_label_mapping_is_explicit_and_unknown_keys_fail(self):
        with temporary_directory() as temp:
            folder = Path(temp)
            paths = {}
            for split in ("train", "valid", "test"):
                path = folder / (split + ".json")
                path.write_text(json.dumps([native(f"{split}.a_1_0", f"{split} 김가람")]), encoding="utf-8")
                paths[split] = path
            incomplete = {"PS_NAME": "NAME"}
            with self.assertRaises(ValueError):
                prepare_dataset(**paths, output_dir=folder / "bad", adapter="kdpii-v1", label_map=incomplete)
            mapping = {label: label for label in KDPII_V1_LABELS}
            mapping.update(QT_PHONE="PHONE", QT_MOBILE="PHONE")
            manifest = prepare_dataset(**paths, output_dir=folder / "good", adapter="kdpii-v1", label_map=mapping)
            self.assertEqual(len(read_jsonl(folder / "good" / "train.jsonl")[0].annotation_scope), 32)
            self.assertEqual(len(manifest["native_annotation_scope"]), 33)
            mapping["TYPO"] = "NAME"
            with self.assertRaises(ValueError):
                prepare_dataset(**paths, output_dir=folder / "extra", adapter="kdpii-v1", label_map=mapping)
        with self.assertRaises(ValueError):
            _json_argument('{"PS_NAME": "NAME", "PS_NAME": "PERSON"}')

    def test_inspect_does_not_print_raw_values_or_unknown_dynamic_keys(self):
        with temporary_directory() as temp:
            path = Path(temp) / "sample.json"
            record = native()
            record["customer-secret-name"] = "never print this"
            path.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")
            summary = json.dumps(inspect_file(path), ensure_ascii=False)
            for sensitive in ("김가람", "Data1.1_1_0", "customer-secret-name", "never print this"):
                self.assertNotIn(sensitive, summary)


if __name__ == "__main__":
    unittest.main()
