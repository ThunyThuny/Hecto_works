"""Audit tests use tiny fictional records and never fit a model."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from kpii_mvp.b_data import export_review
from kpii_mvp.b_inputs import FIXED_POLICY_ID
from kpii_mvp.data import write_jsonl
from kpii_mvp.schema import Document, Mention
from kpii_mvp.synthetic_audit import audit_corpus, main


def fixture(doc_id="doc01", split="train", rows=None):
    rows = rows or [
        ("이전 근무처는 ", "한빛테크", "WORKPLACE", "MASK", "private", "입니다."),
        ("모집 기관인 ", "한빛테크", "WORKPLACE", "KEEP", "public_approved", "의 안내입니다."),
        ("개인 회신은 ", "person@one.example", "EMAIL", "MASK", "private", "로 받습니다."),
        ("이번 채용 문의는 ", "desk@two.example", "EMAIL", "KEEP", "public_approved", "로 보내세요."),
    ]
    text, mentions, relations = "", [], []
    metadata = {"document_type": "general_notice", "approved_public_contacts": [],
                "approved_public_orgs": [], "approved_public_positions": []}
    for index, (before, value, kind, action, role, after) in enumerate(rows):
        text += before
        start = len(text)
        text += value
        mention = Mention(f"m{index + 1}", start, len(text), value, kind, action)
        mentions.append(mention)
        text += after + "\n"
        clause_id = f"{doc_id}-clause-{index + 1}"
        relations.append({"mention_id": mention.mention_id, "relation": role,
                          "proposed_action": action, "clause_id": clause_id})
        if role == "public_approved" or (role == "private" and kind == "WORKPLACE"):
            key = ("approved_public_contacts" if kind in {"PHONE", "EMAIL", "ADDRESS"}
                   else "approved_public_positions" if kind == "POSITION" else "approved_public_orgs")
            if value not in metadata[key]:
                metadata[key].append(value)
    doc = Document(doc_id, text, mentions, f"family-{doc_id}",
                   sorted({item.entity_type for item in mentions}), split=split,
                   policy_id=FIXED_POLICY_ID, trusted_context=metadata)
    record = {"doc_id": doc_id, "split": split, "relations": relations,
              "clause_ids": [item["clause_id"] for item in relations]}
    return doc, record


def codes(report):
    return {item["code"] for item in report["blockers"]}


class SyntheticAuditTests(unittest.TestCase):
    def test_valid_proposals_are_read_only_and_stay_pending(self):
        doc, record = fixture()
        plan = {"documents": [record]}
        document_before, plan_before = doc.to_dict(), copy.deepcopy(plan)
        report = audit_corpus([doc], plan)
        self.assertTrue(report["passed"], report["blockers"])
        self.assertEqual(report["documents"], 1)
        self.assertEqual(report["mentions"], 4)
        self.assertEqual(report["actions"], {"MASK": 2, "KEEP": 2})
        self.assertEqual(report["review_status"], "PENDING")
        self.assertFalse(report["training_executed"])
        self.assertEqual(doc.to_dict(), document_before)
        self.assertEqual(plan, plan_before)

    def test_semantic_oracle_does_not_trust_proposed_action(self):
        doc, record = fixture(rows=[("학력: ", "한빛대학교", "SCHOOL", "KEEP", "private", " 졸업")])
        doc = Document.from_dict({**doc.to_dict(), "trusted_context": {
            "approved_public_orgs": ["한빛대학교"], "document_type": "resume"}})
        report = audit_corpus([doc], {"documents": [record]})
        self.assertIn("semantic_label_mismatch", codes(report))
        self.assertNotIn("keep_fails_policy_gate", codes(report))

    def test_private_contact_cannot_survive_as_keep_in_same_document(self):
        doc, record = fixture(rows=[
            ("내 회신 주소는 ", "person@one.example", "EMAIL", "MASK", "private", "입니다."),
            ("기관 연락은 ", "person@one.example", "EMAIL", "KEEP", "public_approved", "로 하세요."),
        ])
        report = audit_corpus([doc], {"documents": [record]})
        self.assertIn("private_contact_released_elsewhere", codes(report))

    def test_contact_collision_is_normalized_across_documents_and_splits(self):
        for kind, private, public in (("PHONE", "000-1234-5678", "000 1234 5678"),
                                      ("EMAIL", "Person@one.example", "person@ONE.example"),
                                      ("ADDRESS", "가상시  다온로 7", "가상시 다온로 7")):
            with self.subTest(kind=kind):
                first, a = fixture("one", "train", [("개인 정보: ", private, kind, "MASK", "private", ".")])
                second, b = fixture("two", "test", [("기관 방문 안내 ", public, kind, "KEEP", "public_approved", " 참고")])
                report = audit_corpus([first, second], {"documents": [a, b]})
                self.assertIn("private_contact_released_elsewhere", codes(report))
                collision = next(item for item in report["blockers"] if item["code"] == "private_contact_released_elsewhere")
                self.assertEqual(collision["doc_ids"], ["one", "two"])
                self.assertNotIn(private, json.dumps(collision, ensure_ascii=False))

    def test_company_opposite_labels_are_allowed(self):
        doc, record = fixture(rows=[
            ("이력서의 근무처는 ", "한빛테크", "WORKPLACE", "MASK", "private", "입니다."),
            ("모집 기관 ", "한빛테크", "WORKPLACE", "KEEP", "public_approved", " 안내입니다."),
        ])
        report = audit_corpus([doc], {"documents": [record]})
        self.assertTrue(report["passed"], report["blockers"])
        self.assertEqual(report["approval_by_action"]["MASK:approved=1"], 1)

    def test_keep_requires_exact_metadata_approval(self):
        doc, record = fixture()
        raw = doc.to_dict()
        raw["trusted_context"]["approved_public_contacts"] = []
        doc = Document.from_dict(raw)
        report = audit_corpus([doc], {"documents": [record]})
        self.assertIn("keep_fails_policy_gate", codes(report))
        self.assertIn("approved_role_missing_approval", codes(report))

    def test_unapproved_public_role_must_remain_mask(self):
        doc, record = fixture(rows=[("검증 전 연락처: ", "help@one.example", "EMAIL", "MASK", "public_unapproved", ".")])
        raw = doc.to_dict()
        raw["trusted_context"]["approved_public_contacts"] = ["help@one.example"]
        report = audit_corpus([Document.from_dict(raw)], {"documents": [record]})
        self.assertIn("unapproved_role_has_approval", codes(report))

    def test_name_always_masks_even_if_relation_claims_public(self):
        doc, record = fixture(rows=[("안내 담당 ", "김가람", "NAME", "KEEP", "public_approved", "입니다.")])
        report = audit_corpus([doc], {"documents": [record]})
        self.assertIn("semantic_label_mismatch", codes(report))
        self.assertIn("keep_fails_policy_gate", codes(report))

    def test_plan_must_cover_each_document_and_each_mention(self):
        doc, record = fixture()
        missing_record = audit_corpus([doc], {"documents": []})
        self.assertIn("plan_document_inventory_mismatch", codes(missing_record))
        record["relations"].pop()
        partial = audit_corpus([doc], {"documents": [record]})
        self.assertIn("plan_mention_inventory_mismatch", codes(partial))
        self.assertIn("unknown_semantic_role", codes(partial))

    def test_wrong_offsets_are_reported_without_mutating_other_documents(self):
        doc, record = fixture()
        object.__setattr__(doc.mentions[0], "start", 0)
        report = audit_corpus([doc], {"documents": [record]})
        self.assertIn("invalid_document_schema_or_spans", codes(report))
        self.assertEqual(report["validated_documents"], 0)

    def test_omitted_repeated_known_candidate_is_detected(self):
        doc, record = fixture(rows=[("담당자는 ", "김가람", "NAME", "MASK", "name", "입니다. 김가람에게 회신하세요.")])
        report = audit_corpus([doc], {"documents": [record]})
        self.assertIn("unannotated_known_candidate_occurrence", codes(report))

    def test_known_awkward_phrase_and_serial_position_are_blockers(self):
        first, a = fixture("one", rows=[("저는 ", "한빛대학교", "SCHOOL", "MASK", "private", "를 졸업한 학력을 적었습니다.")])
        second, b = fixture("two", rows=[("당시 직위: ", "가상한빛제20운영책임자", "POSITION", "MASK", "private", ".")])
        report = audit_corpus([first, second], {"documents": [a, b]})
        self.assertIn("known_awkward_phrase", codes(report))
        self.assertIn("serial_fictional_prefix_inside_position", codes(report))

    def test_exact_value_elided_core_reuse_cross_split_is_detected(self):
        first, a = fixture("one", "train", [("이전 근무처는 ", "한빛테크", "WORKPLACE", "MASK", "private", "입니다.")])
        second, b = fixture("two", "test", [("이전 근무처는 ", "새솔테크", "WORKPLACE", "MASK", "private", "입니다.")])
        report = audit_corpus([first, second], {"documents": [a, b]})
        self.assertIn("identical_core_clause_crosses_splits", codes(report))
        self.assertEqual(report["cross_split_identical_core_clause_count"], 1)

    def test_group_and_normalized_text_overlap_are_rejected(self):
        first, a = fixture("one", "train")
        second, b = fixture("two", "test")
        report = audit_corpus([first, second], {"documents": [a, b]})
        self.assertIn("split_identity_or_group_overlap", codes(report))

    def test_empty_corpus_cannot_pass(self):
        self.assertIn("empty_corpus", codes(audit_corpus([], {"documents": []})))

    def test_cli_preserves_inputs_and_requires_fresh_output(self):
        doc, record = fixture()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            source, plan_file, output = root / "docs.jsonl", root / "plan.json", root / "audit.json"
            write_jsonl(source, [doc])
            plan_file.write_text(json.dumps({"documents": [record]}, ensure_ascii=False), encoding="utf-8")
            export_review([doc], root / "review")
            reviews = root / "review" / "reviews.jsonl"
            before = {path: path.read_bytes() for path in (source, plan_file, reviews)}
            args = ["audit", "--input", str(source), "--plan", str(plan_file), "--output", str(output), "--reviews", str(reviews)]
            with redirect_stdout(StringIO()):
                self.assertEqual(main(args), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["review_records_checked"], 1)
            for path, contents in before.items():
                self.assertEqual(path.read_bytes(), contents)
            with self.assertRaisesRegex(ValueError, "fresh"):
                main(args)
            with self.assertRaisesRegex(ValueError, "fresh"):
                main(["audit", "--input", str(source), "--plan", str(plan_file), "--output", str(source)])


if __name__ == "__main__":
    unittest.main()
