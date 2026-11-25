from dataclasses import replace
import unittest

from kpii_mvp.candidate_diagnostics import compare_candidates, diagnose_classification
from kpii_mvp.b_data import document_fingerprint
from kpii_mvp.schema import Document, Mention


class CandidateDiagnosticsTests(unittest.TestCase):
    def test_classification_counts_false_keep_and_excludes_unavailable(self):
        doc = Document("x", "대리 과장 주임", [Mention("a", 0, 2, "대리", "POSITION", "MASK"),
                        Mention("b", 3, 5, "과장", "POSITION", "KEEP"), Mention("c", 6, 8, "주임", "POSITION", "MASK")],
                       "family", ["POSITION"], policy_id="career_external_share_v1")
        scores = [{"doc_id": "x", "document_sha256": document_fingerprint(doc), "policy_id": doc.policy_id,
                   "mentions": [{"mention_id": m.mention_id, "start": m.start, "end": m.end,
                      "entity_type": m.entity_type, "p_keep": p, "available": p is not None,
                      "reason": "scored" if p is not None else "context_exceeds_max_length"}
                      for m, p in zip(doc.mentions, (1.0, 0.5, None))]}]
        report = diagnose_classification([doc], scores)
        self.assertEqual(report["counts"]["MASK_to_KEEP"], 1)
        self.assertEqual(report["counts"]["KEEP_to_MASK"], 1)
        self.assertEqual(report["counts"]["unavailable_contextual"], 1)
        self.assertEqual(report["scorable_candidates"], 2)
        self.assertEqual(report["accuracy"], 0)

    def test_whitespace_remains_strict_leak_and_refinement_is_separate(self):
        gold = Document("x", "가상시 나루구 도담길 1", [Mention("g", 0, 13, "가상시 나루구 도담길 1", "ADDRESS", "MASK")],
                        "family", ["ADDRESS"], split="test", policy_id="career_external_share_v1")
        raw = replace(gold, mentions=[Mention("a", 0, 3, "가상시", "ADDRESS"), Mention("b", 4, 13, "나루구 도담길 1", "ADDRESS")])
        refined = replace(gold, mentions=[replace(gold.mentions[0], action=None)])
        report = compare_candidates([gold], [raw], [refined])
        self.assertEqual(report["raw_a"]["by_type_and_action"]["ADDRESS:MASK"]["whitespace_only_uncovered"], 1)
        self.assertEqual(report["raw_a"]["strict_allmask"]["overall"]["leaked_mask_mentions"], 1)
        self.assertEqual(report["refined_a"]["strict_allmask"]["overall"]["leaked_mask_mentions"], 0)
        self.assertEqual(report["strict_delta_refined_minus_raw"]["leaked_mask_mentions"], -1)

    def test_refinement_cannot_drop_raw_mask_coverage(self):
        gold = Document("x", "선임연구원", [Mention("g", 0, 5, "선임연구원", "POSITION", "MASK")],
                        "family", ["POSITION"], policy_id="career_external_share_v1")
        raw = replace(gold, mentions=[replace(gold.mentions[0], action=None)])
        shortened = replace(gold, mentions=[Mention("a", 0, 2, "선임", "POSITION")])
        with self.assertRaisesRegex(ValueError, "must not remove"):
            compare_candidates([gold], [raw], [shortened])
