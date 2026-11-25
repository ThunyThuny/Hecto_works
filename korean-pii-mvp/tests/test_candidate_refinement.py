"""Boundary refinement invariants; no model or golden test fitting is used."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from kpii_mvp.candidate_refinement import (
    refine_candidates_file, refine_document, validate_refinement_manifest,
)
from kpii_mvp.data import read_jsonl, write_jsonl
from kpii_mvp.schema import Document, Mention


def fixture(text, fragments):
    mentions = []
    for index, (value, kind) in enumerate(fragments):
        start = text.index(value)
        mentions.append(Mention(f"m{index}", start, start + len(value), value, kind))
    return Document("d1", text, mentions, "g1", list(dict.fromkeys(kind for _, kind in fragments)),
        policy_id="career_external_share_v1")


class CandidateRefinementTests(unittest.TestCase):
    def test_address_prefix_and_whitespace_fragments_merge(self):
        document = fixture("주소: 가상시 한빛구 도담길 240입니다.",
            [("가상시", "LC_ADDRESS"), ("한빛구 도담길 240", "LC_ADDRESS")])
        result, report = refine_document(document)
        self.assertEqual([m.text for m in result.mentions], ["가상시 한빛구 도담길 240"])
        self.assertEqual(report["merged_fragments"], 1)
        self.assertEqual(result.text, document.text)
        self.assertEqual(result.mentions[0].mention_id, "m0")

    def test_real_road_address_generic_prefix(self):
        for address, anchor in (("서울특별시 강남구 테헤란로 123", "강남구 테헤란로 123"),
                ("경기도 성남시 분당구 판교로 235", "분당구 판교로 235")):
            with self.subTest(address=address):
                result, _ = refine_document(fixture("방문 장소: " + address + ".", [(anchor, "ADDRESS")]))
                self.assertEqual(result.mentions[0].text, address)

    def test_position_prefix_suffix_and_fragment_merge(self):
        for value, fragments in (("선임연구원", ["선임"]), ("책임연구원", ["책임"]),
                ("운영책임자", ["운영", "임자"])):
            with self.subTest(value=value):
                result, _ = refine_document(fixture("기재 직위는 " + value + "입니다.",
                    [(piece, "CV_POSITION") for piece in fragments]))
                self.assertEqual([m.text for m in result.mentions], [value])

    def test_common_composite_positions_preserve_josa(self):
        for value, anchor in (("연구개발팀장", "팀장"), ("서비스매니저", "매니저"),
                ("교육센터장", "센터장"), ("기술책임자", "책임자"), ("기획연구원", "연구원")):
            with self.subTest(value=value):
                result, _ = refine_document(fixture("해당 직위는 " + value + "으로 기재했습니다.", [(anchor, "POSITION")]))
                self.assertEqual(result.mentions[0].text, value)
                self.assertNotIn("으로", result.mentions[0].text)

    def test_unknown_suffix_and_lexical_boundary_are_not_expanded(self):
        for text, anchor in (("대리점에서 일했다.", "대리"), ("운영 책임자는 별도다.", "운영"),
                ("직위는 선임연구원장이었다.", "선임")):
            with self.subTest(text=text):
                document = fixture(text, [(anchor, "POSITION")])
                result, _ = refine_document(document)
                self.assertEqual(result.mentions, document.mentions)

    def test_mandatory_mask_or_other_type_blocks_expansion(self):
        document = fixture("주소: 가상시 한빛구 도담길 240.",
            [("가상시", "PS_NAME"), ("한빛구 도담길 240", "LC_ADDRESS")])
        result, report = refine_document(document)
        self.assertEqual(result.mentions, document.mentions)
        self.assertEqual(report["rejected_proposals"]["cross_type_overlap"], 1)

    def test_never_shrinks_overextended_original(self):
        document = fixture("주소: 가상시 한빛구 도담길 240입니다.",
            [("한빛구 도담길 240입니다", "ADDRESS")])
        result, report = refine_document(document)
        self.assertEqual(result.mentions, document.mentions)
        self.assertTrue(report["rejected_proposals"]["would_shrink_original_span"])

    def test_no_cross_field_or_line_merge(self):
        for text in ("주소: 가상시\n한빛구 도담길 240. 전화: 010-1234-5678.",
                     "주소: 가상시; 별도 주소: 한빛구 도담길 240."):
            document = fixture(text, [("가상시", "ADDRESS"), ("한빛구 도담길 240", "ADDRESS")])
            result, _ = refine_document(document)
            self.assertEqual(len(result.mentions), 2)
            self.assertEqual(result.mentions[0].text, "가상시")

    def test_neighboring_addresses_remain_separate(self):
        text = "옛 주소: 서울특별시 강남구 테헤란로 123, 새 주소: 부산광역시 해운대구 센텀로 45."
        document = fixture(text, [("강남구 테헤란로 123", "ADDRESS"),
            ("해운대구 센텀로 45", "ADDRESS")])
        result, _ = refine_document(document)
        self.assertEqual([m.text for m in result.mentions],
            ["서울특별시 강남구 테헤란로 123", "부산광역시 해운대구 센텀로 45"])
        self.assertLess(result.mentions[0].end, result.mentions[1].start)

    def test_phone_like_sequence_is_not_absorbed_as_house_number(self):
        text = "방문: 서울특별시 강남구 테헤란로 010-1234-5678."
        document = fixture(text, [("강남구 테헤란로", "ADDRESS")])
        result, _ = refine_document(document)
        self.assertEqual(result.mentions, document.mentions)

    def test_address_bounded_floor_room(self):
        for suffix in (" 3층 301호", " 3층", " 301호"):
            with self.subTest(suffix=suffix):
                address = "서울특별시 강남구 테헤란로 123" + suffix
                result, _ = refine_document(fixture("주소: " + address + "입니다.",
                    [("강남구 테헤란로 123", "ADDRESS")]))
                self.assertEqual(result.mentions[0].text, address)

    def test_address_floor_room_never_crosses_next_phone_date_or_field(self):
        for tail in (" 전화: 010-1234-5678", " 010-1234-5678", " 날짜: 2026-10-19",
                     " 2026-10-19", " 담당자: 홍길동", "\n3층 301호"):
            with self.subTest(tail=tail):
                address = "서울특별시 강남구 테헤란로 123"
                result, _ = refine_document(fixture("주소: " + address + tail + ".",
                    [("강남구 테헤란로 123", "ADDRESS")]))
                self.assertEqual(result.mentions[0].text, address)

    def test_floor_foreign_candidate_blocks_address_extension(self):
        text = "주소: 서울특별시 강남구 테헤란로 123 3층 301호."
        document = fixture(text, [("강남구 테헤란로 123", "ADDRESS"), ("301호", "PS_ID")])
        result, report = refine_document(document)
        self.assertEqual(result.mentions, document.mentions)
        self.assertEqual(report["rejected_proposals"]["cross_type_overlap"], 1)

    def test_merged_identifiers_cannot_change_foreign_neighbors(self):
        text = "홍길동 선임연구원, 전화 010-1234-5678."
        document = fixture(text, [("홍길동", "PS_NAME"), ("선임", "CV_POSITION"),
            ("구원", "CV_POSITION"), ("010-1234-5678", "QT_PHONE")])
        result, report = refine_document(document)
        self.assertEqual([m.text for m in result.mentions], ["홍길동", "선임연구원", "010-1234-5678"])
        self.assertEqual(result.mentions[0], document.mentions[0])
        self.assertEqual(result.mentions[-1], document.mentions[-1])
        self.assertEqual(result.mentions[1].mention_id, "m1")
        self.assertEqual(report["events"][0]["merged_mention_ids"], ["m2"])

    def test_labels_reasons_and_approval_metadata_cannot_affect_refinement(self):
        document = fixture("직위는 선임연구원입니다.", [("선임", "POSITION")])
        labelled = replace(document, mentions=[replace(document.mentions[0], action="KEEP", reason_code="gold")],
            trusted_context={"approved_public_positions": ["선임연구원"], "review_status": "APPROVED"})
        baseline, baseline_report = refine_document(document)
        changed, changed_report = refine_document(labelled)
        self.assertEqual(baseline.mentions, changed.mentions)
        self.assertEqual(baseline_report, changed_report)
        self.assertIsNone(changed.mentions[0].action)
        self.assertIsNone(changed.mentions[0].reason_code)

    def test_idempotent_and_original_spans_remain_covered(self):
        document = fixture("😀 주소는 가상시 한빛구 도담길 240. 직위: 운영책임자.",
            [("한빛구 도담길 240", "ADDRESS"), ("운영", "POSITION"), ("임자", "POSITION")])
        result, _ = refine_document(document)
        again, report = refine_document(result)
        self.assertEqual(result.to_dict(), again.to_dict())
        self.assertEqual(report["changed_candidates"], 0)
        for old in document.mentions:
            self.assertTrue(any(new.start <= old.start and new.end >= old.end for new in result.mentions))

    def test_unsupported_types_are_untouched(self):
        document = fixture("직위는 선임연구원입니다.", [("선임", "UNSUPPORTED")])
        result, _ = refine_document(document)
        self.assertEqual(result.mentions, document.mentions)

    def test_manifest_chain_and_immutable_output(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            source, output = root / "raw.jsonl", root / "refined.jsonl"
            write_jsonl(source, [fixture("주소: 가상시 한빛구 도담길 240.", [("한빛구 도담길 240", "ADDRESS")])])
            parent = {"stage": "A-prediction", "training_manifest_sha256": "a" * 64,
                "max_length": 512, "stride": 128, "checkpoint_artifacts": {"model.bin": "b" * 64},
                "output": {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}
            Path(str(source) + ".manifest.json").write_text(json.dumps(parent), encoding="utf-8")
            result = refine_candidates_file(source, output)
            self.assertEqual(validate_refinement_manifest(output), result)
            self.assertEqual(read_jsonl(output)[0].mentions[0].text, "가상시 한빛구 도담길 240")
            with self.assertRaises(ValueError):
                refine_candidates_file(source, output)
            with self.assertRaises(ValueError):
                refine_candidates_file(source, source)
            source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "parent evidence changed"):
                validate_refinement_manifest(output)

    def test_refined_manifest_cannot_substitute_another_a_signature(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            source, output = root / "raw.jsonl", root / "refined.jsonl"
            write_jsonl(source, [fixture("직위: 책임연구원.", [("책임", "POSITION")])])
            parent = {"stage": "A-prediction", "training_manifest_sha256": "a" * 64,
                "max_length": 512, "stride": 128,
                "output": {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}
            Path(str(source) + ".manifest.json").write_text(json.dumps(parent), encoding="utf-8")
            manifest = refine_candidates_file(source, output)
            sidecar = Path(str(output) + ".manifest.json")
            for key, malicious in (("training_manifest_sha256", "b" * 64), ("stride", 64),
                                   ("checkpoint_artifacts", {"model.bin": "c" * 64})):
                with self.subTest(key=key):
                    altered = {**manifest, key: malicious}
                    sidecar.write_text(json.dumps(altered), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "original A provenance"):
                        validate_refinement_manifest(output)
            sidecar.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(validate_refinement_manifest(output), manifest)


if __name__ == "__main__":
    unittest.main()
