"""Small artificial unit fixtures; never a B training or evaluation dataset."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

from kpii_mvp.b_data import (
    CSV_FIELDS, document_fingerprint, document_identities, export_review, import_review, load_reviews,
    main, pending_review, validate_reviewed_documents, validate_split_separation,
)
from kpii_mvp.data import read_jsonl, write_jsonl
from kpii_mvp.schema import Document, Mention


def fixture_doc(doc_id="test_fixture", *, action="MASK", split="train", text=None):
    text = text or "😀김가람, 예\n한빛학교 안내"
    name_start = text.index("김가람")
    school_start = text.index("한빛학교")
    return Document(
        doc_id, text,
        [Mention("m1", name_start, name_start + 3, "김가람", "PS_NAME", action),
         Mention("m2", school_start, school_start + 4, "한빛학교", "OGG_EDUCATION", "KEEP", "PUBLIC_ORGANIZATION")],
        "fixture-group-" + doc_id, ["PS_NAME", "OGG_EDUCATION"],
        source="artificial_unit_test_fixture", split=split, policy_id="career_external_share_v1",
        trusted_context={"approved_public_orgs": ["한빛학교"], "document_type": "general_notice"},
    )


def approved(doc):
    return {**pending_review(doc), "review_status": "APPROVED", "reviewer_id": "unit_test_fixture_reviewer",
            "reviewed_at": "2026-10-19T09:00:00+09:00", "candidate_inventory_complete": True,
            "action_labels_complete": True}


class ReviewDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_reviews(self, records):
        path = self.root / "reviews.jsonl"
        path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
        return path

    def write_csv(self, rows, fields=CSV_FIELDS):
        path = self.root / "edited.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def rows_for(self, doc):
        return [{"doc_id": doc.doc_id, **{key: ("" if value is None else str(value)) for key, value in mention.to_dict().items()}}
                for mention in doc.mentions]

    def test_full_fingerprint_is_stable_and_covers_labels_metadata_scope_and_split(self):
        doc = fixture_doc()
        self.assertEqual(document_fingerprint(doc), document_fingerprint(Document.from_dict(doc.to_dict())))
        changed = [
            replace(doc, mentions=[replace(doc.mentions[0], action="KEEP"), doc.mentions[1]]),
            replace(doc, trusted_context={"approved_public_orgs": []}),
            replace(doc, annotation_scope=doc.annotation_scope + ["QT_PHONE"]),
            replace(doc, split="dev"), replace(doc, source_group="changed"),
        ]
        for edit in changed:
            self.assertNotEqual(document_fingerprint(doc), document_fingerprint(edit))

    def test_export_never_approves_and_keeps_null_unlabelled(self):
        doc = fixture_doc(action=None)
        export_review([doc], self.root / "packet")
        docs = read_jsonl(self.root / "packet/documents.jsonl")
        reviews = load_reviews(self.root / "packet/reviews.jsonl")
        self.assertIsNone(docs[0].mentions[0].action)
        self.assertEqual(reviews[doc.doc_id]["review_status"], "PENDING")
        self.assertFalse(reviews[doc.doc_id]["candidate_inventory_complete"])
        self.assertIsNone(reviews[doc.doc_id]["reviewer_id"])
        with self.assertRaisesRegex(ValueError, "PENDING"):
            validate_reviewed_documents(docs, reviews)

    def test_export_explicit_policy_only_updates_export_copy(self):
        doc = replace(fixture_doc(), policy_id=None)
        export_review([doc], self.root / "packet", policy_id="career_external_share_v1")
        self.assertIsNone(doc.policy_id)
        self.assertEqual(read_jsonl(self.root / "packet/documents.jsonl")[0].policy_id, "career_external_share_v1")

    def test_null_action_is_not_keep_even_with_complete_review_declaration(self):
        doc = fixture_doc(action=None)
        with self.assertRaisesRegex(ValueError, "null never means KEEP"):
            validate_reviewed_documents([doc], {doc.doc_id: approved(doc)})

    def test_approved_actions_counted_and_declaration_limit_reported(self):
        doc = fixture_doc()
        report = validate_reviewed_documents([doc], {doc.doc_id: approved(doc)}, expected_split="train")
        self.assertEqual(report["actions"], {"KEEP": 1, "MASK": 1})
        self.assertEqual(report["review_evidence"], "explicit_reviewer_declarations_not_independently_verified")

    def test_stale_approval_rejected_after_action_or_trusted_metadata_change(self):
        doc = fixture_doc()
        for changed in (replace(doc, trusted_context={}),
                        replace(doc, mentions=[replace(doc.mentions[0], action="KEEP"), doc.mentions[1]])):
            with self.assertRaisesRegex(ValueError, "stale"):
                validate_reviewed_documents([changed], {doc.doc_id: approved(doc)})

    def test_approval_requires_complete_inventory_action_declaration_identity_and_timezone(self):
        doc = fixture_doc()
        bad = [
            {"candidate_inventory_complete": False}, {"action_labels_complete": False},
            {"reviewer_id": None}, {"reviewer_id": ""}, {"reviewer_id": " padded "},
            {"reviewed_at": None}, {"reviewed_at": "2026-10-19T09:00:00"},
            {"reviewed_at": "not-a-date"}, {"candidate_inventory_complete": 1},
        ]
        for edit in bad:
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                validate_reviewed_documents([doc], {doc.doc_id: {**approved(doc), **edit}})

    def test_policy_mismatch_missing_extra_and_duplicate_docs_fail(self):
        doc = fixture_doc()
        for docs, reviews in [
            ([replace(doc, policy_id=None)], {doc.doc_id: approved(doc)}),
            ([doc], {}), ([doc], {doc.doc_id: approved(doc), "extra": approved(doc)}),
            ([doc, doc], {doc.doc_id: approved(doc)}),
        ]:
            with self.assertRaises(ValueError):
                validate_reviewed_documents(docs, reviews)

    def test_dev_aliases_and_wrong_split(self):
        for split in ("dev", "valid", "validation"):
            doc = fixture_doc(split=split)
            self.assertEqual(validate_reviewed_documents([doc], {doc.doc_id: approved(doc)}, expected_split="dev")["split"], "dev")
            with self.assertRaisesRegex(ValueError, "split"):
                validate_reviewed_documents([doc], {doc.doc_id: approved(doc)}, expected_split="test")

    def test_load_reviews_duplicate_record_extra_fields_and_duplicate_json_keys_fail(self):
        doc = fixture_doc()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_reviews(self.write_reviews([pending_review(doc), pending_review(doc)]))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            load_reviews(self.write_reviews([{**pending_review(doc), "auto_approved": True}]))
        path = self.root / "dup-key.jsonl"
        raw = json.dumps(pending_review(doc))
        path.write_text('{"doc_id":"x",' + raw[1:] + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            load_reviews(path)

    def test_reviews_invalid_hash_or_nonfinite_field_rejected(self):
        doc = fixture_doc()
        for edit in ({"document_sha256": "bad"}, {"notes": float("nan")}, {"reviewer_id": float("inf")}):
            with self.assertRaises(ValueError):
                load_reviews(self.write_reviews([{**pending_review(doc), **edit}]))
        with self.assertRaises(ValueError):
            replace(doc, trusted_context={"bad": float("nan")})

    def test_csv_roundtrip_unicode_commas_and_embedded_newlines(self):
        text = "😀김가람, 예\n한빛학교 안내"
        doc = fixture_doc(text=text)
        # Include comma and newline inside a mention to verify CSV quoting.
        doc = replace(doc, mentions=[Mention("multiline", 1, text.index("한빛학교"), text[1:text.index("한빛학교")], "PS_NAME", "MASK"), doc.mentions[1]])
        export_review([doc], self.root / "packet")
        self.assertEqual(import_review([doc], self.root / "packet/mentions.csv"), [doc])

    def test_csv_blank_action_remains_null_and_missing_column_fails(self):
        doc = fixture_doc()
        rows = self.rows_for(doc)
        rows[0]["action"] = ""
        self.assertIsNone(import_review([doc], self.write_csv(rows))[0].mentions[0].action)
        fields = tuple(field for field in CSV_FIELDS if field != "action")
        with self.assertRaisesRegex(ValueError, "column"):
            import_review([doc], self.write_csv([{key: row[key] for key in fields} for row in rows], fields))

    def test_csv_unknown_or_duplicate_header_rejected(self):
        doc = fixture_doc()
        for fields in (CSV_FIELDS + ("secret",), CSV_FIELDS[:-1] + ("action",)):
            path = self.root / "bad.csv"
            path.write_text(",".join(fields) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "column"):
                import_review([doc], path)

    def test_csv_exact_slice_integer_offsets_unknown_docs_and_duplicate_ids(self):
        doc = fixture_doc()
        cases = [{"start": "1.0"}, {"start": "-1"}, {"start": " 1"},
                 {"text": "가짜값"}, {"doc_id": "unknown"}, {"action": "keep"}]
        for edit in cases:
            rows = self.rows_for(doc)
            rows[0].update(edit)
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                import_review([doc], self.write_csv(rows))
        rows = self.rows_for(doc)
        rows[1]["mention_id"] = rows[0]["mention_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            import_review([doc], self.write_csv(rows))

    def test_csv_adds_missed_candidate_and_drops_false_positive_only_explicitly(self):
        doc = fixture_doc()
        initial = replace(doc, mentions=[doc.mentions[0]])
        imported = import_review([initial], self.write_csv(self.rows_for(doc)))
        self.assertEqual(imported[0].mentions, doc.mentions)
        dropped = import_review([doc], self.write_csv(self.rows_for(doc)[1:]))
        self.assertEqual(dropped[0].mentions, [doc.mentions[1]])

    def test_zero_candidates_require_explicit_doc_row_and_never_infer_negative_from_csv_loss(self):
        doc = fixture_doc()
        with self.assertRaisesRegex(ValueError, "lost all rows"):
            import_review([doc], self.write_csv([]))
        marker = {field: "" for field in CSV_FIELDS}
        marker["doc_id"] = doc.doc_id
        self.assertEqual(import_review([doc], self.write_csv([marker]))[0].mentions, [])
        for rows in ([marker] + self.rows_for(doc), self.rows_for(doc) + [marker], [marker, marker]):
            with self.assertRaisesRegex(ValueError, "coexist"):
                import_review([doc], self.write_csv(rows))
        empty = replace(doc, mentions=[])
        export_review([empty], self.root / "packet")
        self.assertEqual(import_review([empty], self.root / "packet/mentions.csv"), [empty])

    def test_new_fingerprint_for_corrected_import_stays_pending(self):
        doc = fixture_doc(action=None)
        docs_path = self.root / "documents.jsonl"
        write_jsonl(docs_path, [doc])
        rows = self.rows_for(doc)
        rows[0]["action"] = "MASK"
        edited = self.write_csv(rows)
        output = self.root / "corrected.jsonl"
        reviews = self.root / "new-pending.jsonl"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["import-review", "--documents", str(docs_path), "--mentions", str(edited), "--output", str(output), "--prepare-reviews", str(reviews)]), 0)
        corrected = read_jsonl(output)[0]
        declaration = load_reviews(reviews)[doc.doc_id]
        self.assertEqual(declaration["review_status"], "PENDING")
        self.assertEqual(declaration["document_sha256"], document_fingerprint(corrected))
        self.assertNotEqual(declaration["document_sha256"], document_fingerprint(doc))

    def test_import_with_stale_reviews_does_not_write_output(self):
        doc = fixture_doc(action=None)
        docs = self.root / "documents.jsonl"
        write_jsonl(docs, [doc])
        rows = self.rows_for(doc)
        rows[0]["action"] = "MASK"
        output = self.root / "must-not-exist.jsonl"
        with self.assertRaisesRegex(ValueError, "stale"):
            main(["import-review", "--documents", str(docs), "--mentions", str(self.write_csv(rows)), "--output", str(output), "--reviews", str(self.write_reviews([approved(doc)]))])
        self.assertFalse(output.exists())

    def test_prevent_overwrite_inputs_existing_reviews_and_empty_data(self):
        doc = fixture_doc()
        export_review([doc], self.root / "packet")
        with self.assertRaisesRegex(ValueError, "already exists"):
            export_review([doc], self.root / "packet")
        docs = self.root / "docs.jsonl"
        write_jsonl(docs, [doc])
        with self.assertRaisesRegex(ValueError, "overwrite"):
            main(["import-review", "--documents", str(docs), "--mentions", str(self.write_csv(self.rows_for(doc))), "--output", str(docs)])
        with self.assertRaises(ValueError):
            validate_reviewed_documents([], {})
        with self.assertRaises(ValueError):
            export_review([], self.root / "empty")

    def test_split_group_leakage_duplicate_text_and_doc_ids_rejected(self):
        train = fixture_doc()
        dev = fixture_doc("dev_fixture", split="valid", text="다른 문서 김가람 한빛학교")
        self.assertEqual(validate_split_separation({"train": [train], "valid": [dev]}), {"train": 1, "dev": 1})
        variants = [replace(dev, source_group=train.source_group),
                    replace(train, doc_id="other", source_group="other", split="dev"),
                    replace(dev, doc_id=train.doc_id)]
        for changed in variants:
            with self.assertRaises(ValueError):
                validate_split_separation({"train": [train], "dev": [changed]})
        whitespace = fixture_doc("space", split="dev", text="😀김가람, 예   한빛학교 안내")
        with self.assertRaisesRegex(ValueError, "duplicate text"):
            validate_split_separation({"train": [train], "dev": [whitespace]})

    def test_split_validation_rejects_duplicate_text_within_split_and_alias_collision(self):
        doc = fixture_doc()
        with self.assertRaisesRegex(ValueError, "duplicate text"):
            validate_split_separation({"train": [doc, replace(doc, doc_id="second", source_group="g2")]})
        with self.assertRaises(ValueError):
            validate_split_separation({"dev": [replace(doc, split="dev")], "valid": [replace(doc, split="valid")]})

    def test_document_identities_share_canonical_schema_without_raw_source(self):
        doc = fixture_doc()
        identities = document_identities([doc])
        self.assertEqual(set(identities), {"document_sha256", "text_sha256", "normalized_text_sha256", "source_group_sha256", "doc_id_sha256"})
        self.assertEqual(identities["document_sha256"], [document_fingerprint(doc)])
        json_digest = lambda value: hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        self.assertEqual(identities["text_sha256"], [json_digest(doc.text)])
        self.assertEqual(identities["source_group_sha256"], [json_digest(doc.source_group)])
        self.assertEqual(identities["doc_id_sha256"], [json_digest(doc.doc_id)])
        self.assertNotIn(doc.text, json.dumps(identities, ensure_ascii=False))

    def test_normalized_identity_catches_fullwidth_and_whitespace_copy_without_changing_source(self):
        first = fixture_doc("first", text="Ａ 김가람\n한빛학교 안내")
        second = fixture_doc("second", split="dev", text="A  김가람 한빛학교 안내")
        left, right = document_identities([first]), document_identities([second])
        self.assertNotEqual(left["text_sha256"], right["text_sha256"])
        self.assertEqual(left["normalized_text_sha256"], right["normalized_text_sha256"])
        self.assertEqual(first.text[0], "Ａ")
        with self.assertRaisesRegex(ValueError, "duplicate text"):
            validate_split_separation({"train": [first], "dev": [second]})

    def test_identity_hash_lists_are_sorted_unique_and_duplicate_doc_ids_fail(self):
        first, second = fixture_doc("z"), fixture_doc("a", text="별도 김가람 한빛학교")
        self.assertEqual(document_identities([first, second]), document_identities([second, first]))
        for values in document_identities([first, second]).values():
            self.assertEqual(values, sorted(set(values)))
            self.assertTrue(all(len(value) == 64 for value in values))
        with self.assertRaises(ValueError):
            document_identities([first, first])


if __name__ == "__main__":
    unittest.main()
