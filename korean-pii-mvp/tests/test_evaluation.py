import unittest

from kpii_mvp.evaluation import evaluate_detection, evaluate_masking
from kpii_mvp.masking import intervals_from_mentions, merge_intervals, render_masked_text
from kpii_mvp.stage_b import AllMaskDecision, PrivacyDecision, apply_decisions
from kpii_mvp.schema import Document, Mention


def mention(text, start, end, entity_type="NAME", action="MASK", mention_id="m1"):
    return {
        "mention_id": mention_id,
        "start": start,
        "end": end,
        "text": text[start:end],
        "entity_type": entity_type,
        "action": action,
    }


def document(doc_id, text, mentions=(), source_group="group1"):
    return {"doc_id": doc_id, "text": text, "mentions": list(mentions), "source_group": source_group}


def masks(doc_id, text, intervals):
    return {"doc_id": doc_id, "text": text, "intervals": intervals}


class DetectionEvaluationTests(unittest.TestCase):
    def test_exact_span_and_type_and_per_type_counts(self):
        text = "김가람 서울"
        gold = document("one", text, [mention(text, 0, 3), mention(text, 4, 6, "ADDRESS", mention_id="m2")])
        pred = document("one", text, [mention(text, 0, 3), mention(text, 4, 6, "SCHOOL", mention_id="m2")])
        result = evaluate_detection([gold], [pred])
        self.assertEqual(result["overall"]["true_positive"], 1)
        self.assertEqual(result["overall"]["precision"], 0.5)
        self.assertEqual(result["overall"]["recall"], 0.5)
        self.assertEqual(result["by_type"]["ADDRESS"]["false_negative"], 1)
        self.assertEqual(result["by_type"]["SCHOOL"]["false_positive"], 1)

    def test_spans_cannot_match_across_documents(self):
        text = "김가람"
        gold = [document("a", text, [mention(text, 0, 3)]), document("b", text)]
        pred = [document("b", text, [mention(text, 0, 3)]), document("a", text)]
        result = evaluate_detection(gold, pred)["overall"]
        self.assertEqual(result["true_positive"], 0)
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["false_negative"], 1)

    def test_duplicate_predicted_span_is_not_silently_dropped(self):
        text = "김가람"
        gold = document("a", text, [mention(text, 0, 3)])
        pred = document("a", text, [mention(text, 0, 3), mention(text, 0, 3, mention_id="duplicate")])
        result = evaluate_detection([gold], [pred])["overall"]
        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["false_positive"], 1)

    def test_source_group_metrics_are_separate(self):
        text = "김가람"
        gold = [document("a", text, [mention(text, 0, 3)], "a"), document("b", text, [mention(text, 0, 3)], "b")]
        pred = [document("a", text, [mention(text, 0, 3)]), document("b", text)]
        result = evaluate_detection(gold, pred)
        self.assertEqual(result["by_source_group"]["a"]["recall"], 1)
        self.assertEqual(result["by_source_group"]["b"]["recall"], 0)

    def test_missing_extra_duplicate_or_changed_document_fails(self):
        gold = document("a", "김가람")
        bad_sets = [[], [gold, gold], [document("b", "김가람")], [document("a", "다른 문장")]]
        for predicted in bad_sets:
            with self.subTest(predicted=predicted), self.assertRaises(ValueError):
                evaluate_detection([gold], predicted)
        with self.assertRaises(ValueError):
            evaluate_detection([gold, gold], [gold])

    def test_detection_can_use_undecided_gold(self):
        text = "김가람"
        gold = document("a", text, [mention(text, 0, 3, action=None)])
        self.assertEqual(evaluate_detection([gold], [gold])["overall"]["f1"], 1)

    def test_partial_prediction_does_not_count_as_full_span_coverage(self):
        text = "김가람"
        gold = document("a", text, [mention(text, 0, 3)])
        pred = document("a", text, [mention(text, 0, 2)])
        result = evaluate_detection([gold], [pred])["overall"]
        self.assertEqual(result["candidate_span_coverage_recall"], 0)
        self.assertEqual(result["candidate_counts"], {"covered_mentions": 0, "gold_mentions": 1})
        self.assertEqual(result["protected_span_coverage_recall"], 0)

    def test_overlapping_prediction_union_can_fully_cover_gold_span(self):
        text = "김가람"
        gold = document("a", text, [mention(text, 0, 3)])
        pred = document("a", text, [mention(text, 0, 2), mention(text, 1, 3, mention_id="m2")])
        result = evaluate_detection([gold], [pred])
        self.assertEqual(result["overall"]["f1"], 0)
        self.assertEqual(result["overall"]["candidate_span_coverage_recall"], 1)
        self.assertEqual(result["overall"]["protected_span_coverage_recall"], 1)
        self.assertEqual(result["by_type"]["NAME"]["candidate_span_coverage_recall"], 1)
        self.assertEqual(result["by_source_group"]["group1"]["candidate_span_coverage_recall"], 1)

    def test_wrong_predicted_type_can_cover_while_exact_detection_fails(self):
        text = "김가람"
        gold = document("a", text, [mention(text, 0, 3, "NAME")])
        pred = document("a", text, [mention(text, 0, 3, "SCHOOL")])
        result = evaluate_detection([gold], [pred])
        self.assertEqual(result["overall"]["f1"], 0)
        self.assertEqual(result["overall"]["candidate_span_coverage_recall"], 1)
        self.assertEqual(result["by_type"]["NAME"]["candidate_counts"]["covered_mentions"], 1)
        self.assertEqual(result["by_type"]["SCHOOL"]["candidate_counts"]["gold_mentions"], 0)
        self.assertIsNone(result["by_type"]["SCHOOL"]["protected_span_coverage_recall"])

    def test_no_explicit_gold_mask_has_null_protected_coverage(self):
        text = "김가람 서울"
        gold = document("a", text, [mention(text, 0, 3, action=None), mention(text, 4, 6, "ADDRESS", "KEEP", "m2")])
        result = evaluate_detection([gold], [gold])["overall"]
        self.assertEqual(result["candidate_span_coverage_recall"], 1)
        self.assertIsNone(result["protected_span_coverage_recall"])
        self.assertEqual(result["protected_counts"], {"covered_mask_mentions": 0, "gold_mask_mentions": 0})

    def test_protected_denominator_uses_only_explicit_gold_mask(self):
        text = "김가람 서울"
        gold = document("a", text, [mention(text, 0, 3), mention(text, 4, 6, "ADDRESS", None, "m2")])
        pred = document("a", text, [mention(text, 4, 6, "ADDRESS")])
        result = evaluate_detection([gold], [pred])["overall"]
        self.assertEqual(result["candidate_span_coverage_recall"], 0.5)
        self.assertEqual(result["protected_span_coverage_recall"], 0)
        self.assertEqual(result["protected_counts"], {"covered_mask_mentions": 0, "gold_mask_mentions": 1})


class IntegratedMaskingEvaluationTests(unittest.TestCase):
    def test_candidate_missing_from_a_is_counted_as_leak(self):
        text = "김가람 이나래"
        gold = document("a", text, [mention(text, 0, 3), mention(text, 4, 7, mention_id="m2")])
        result = evaluate_masking([gold], [masks("a", text, [(0, 3)])])["overall"]
        self.assertEqual(result["gold_mask_mentions"], 2)
        self.assertEqual(result["leaked_mask_mentions"], 1)
        self.assertEqual(result["leak_rate"], 0.5)
        self.assertEqual(result["document_leak_rate"], 1)

    def test_any_uncovered_character_counts_as_a_whole_mention_leak(self):
        text = "김가람"
        gold = document("a", text, [mention(text, 0, 3)])
        result = evaluate_masking([gold], [masks("a", text, [(0, 2)])])["overall"]
        self.assertEqual(result["leak_rate"], 1)

    def test_split_intervals_can_jointly_fully_cover_a_mention(self):
        text = "김가람"
        gold = document("a", text, [mention(text, 0, 3)])
        result = evaluate_masking([gold], [masks("a", text, [(0, 1), (1, 3)])])["overall"]
        self.assertEqual(result["leak_rate"], 0)

    def test_gap_between_intervals_is_a_leak(self):
        text = "김가람"
        gold = document("a", text, [mention(text, 0, 3)])
        result = evaluate_masking([gold], [masks("a", text, [(0, 1), (2, 3)])])["overall"]
        self.assertEqual(result["leak_rate"], 1)

    def test_keep_mention_overlap_counts_once_despite_overlapping_masks(self):
        text = "서울회사"
        gold = document("a", text, [mention(text, 0, 4, "WORKPLACE", "KEEP")])
        result = evaluate_masking([gold], [masks("a", text, [(0, 2), (1, 3)])])["overall"]
        self.assertEqual(result["overmasked_keep_mentions"], 1)
        self.assertEqual(result["overmask_rate"], 1)

    def test_fixed_candidate_subset_does_not_shrink_leak_denominator(self):
        text = "김가람 이나래 오도윤"
        gold = document("a", text, [mention(text, 0, 3), mention(text, 4, 7, mention_id="m2"), mention(text, 8, 11, mention_id="m3")])
        candidate_subset = document("a", text, [mention(text, 0, 3)])
        final_intervals = intervals_from_mentions(text, candidate_subset["mentions"])
        result = evaluate_masking([gold], [masks("a", text, final_intervals)])["overall"]
        self.assertEqual(result["gold_mask_mentions"], 3)
        self.assertEqual(result["leaked_mask_mentions"], 2)
        self.assertAlmostEqual(result["leak_rate"], 2 / 3)

    def test_document_denominator_excludes_documents_without_gold_mask(self):
        text = "김가람"
        gold = [document("a", text, [mention(text, 0, 3)]), document("b", text, [mention(text, 0, 3, action="KEEP")])]
        result = evaluate_masking(gold, [masks("a", text, []), masks("b", text, [])])["overall"]
        self.assertEqual(result["documents"], 2)
        self.assertEqual(result["documents_with_gold_mask"], 1)
        self.assertEqual(result["document_leak_rate"], 1)

    def test_null_gold_action_is_not_keep(self):
        text = "김가람"
        gold = document("a", text, [mention(text, 0, 3, action=None)])
        with self.assertRaisesRegex(ValueError, "null is not KEEP"):
            evaluate_masking([gold], [masks("a", text, [])])

    def test_mask_document_text_and_ids_must_match(self):
        gold = document("a", "김가람")
        for prediction in [[], [masks("a", "바뀐 원문", [])], [masks("a", "김가람", []), masks("a", "김가람", [])]]:
            with self.subTest(prediction=prediction), self.assertRaises(ValueError):
                evaluate_masking([gold], prediction)


class SpanMaskingTests(unittest.TestCase):
    def test_unicode_offsets_are_python_code_points(self):
        text = "안녕😀 김가람!"
        self.assertEqual(text[4:7], "김가람")
        self.assertEqual(render_masked_text(text, [(4, 7)]), "안녕😀 [MASK]!")
        gold = document("a", text, [mention(text, 4, 7)])
        result = evaluate_masking([gold], [masks("a", text, [(4, 7)])])
        self.assertEqual(result["overall"]["leak_rate"], 0)

    def test_only_selected_occurrence_is_replaced(self):
        self.assertEqual(render_masked_text("김가람 / 김가람", [(0, 3)]), "[MASK] / 김가람")

    def test_overlapping_touching_and_duplicate_spans_are_unioned(self):
        text = "abcdefghij"
        self.assertEqual(merge_intervals(text, [(4, 6), (1, 3), (3, 5), (1, 3), (8, 9)]), [(1, 6), (8, 9)])
        self.assertEqual(render_masked_text(text, [(4, 6), (1, 3), (3, 5), (8, 9)]), "a[MASK]gh[MASK]j")

    def test_invalid_spans_fail(self):
        for interval in [(-1, 1), (0, 0), (2, 1), (0, 4), (True, 2), (0, 1.5)]:
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                merge_intervals("abc", [interval])


class StageBBaselineTests(unittest.TestCase):
    def test_baseline_integrates_with_validated_document_dataclass(self):
        original = Document(
            doc_id="a", text="김가람", mentions=[Mention("m1", 0, 3, "김가람", "NAME")],
            source_group="g1", annotation_scope=["NAME"],
        )
        decided = apply_decisions(original, AllMaskDecision().decide(original))
        self.assertEqual(decided.mentions[0].action, "MASK")
        self.assertIsNone(original.mentions[0].action)
        self.assertEqual(intervals_from_mentions(decided.text, decided.mentions), [(0, 3)])

    def test_all_mask_is_explicitly_untrained_and_masks_public_text_claim(self):
        text = "공개 정보: 김가람"
        candidate = document("a", text, [mention(text, 7, 10, action=None)])
        baseline = AllMaskDecision()
        decided = apply_decisions(candidate, baseline.decide(candidate))
        self.assertEqual(decided["mentions"][0]["action"], "MASK")
        self.assertIsNone(candidate["mentions"][0]["action"])
        with self.assertRaises(NotImplementedError):
            baseline.train([candidate])

    def test_decisions_must_cover_each_candidate_exactly(self):
        text = "김가람"
        candidate = document("a", text, [mention(text, 0, 3, action=None)])
        with self.assertRaises(ValueError):
            apply_decisions(candidate, [])
        with self.assertRaises(ValueError):
            apply_decisions(candidate, [PrivacyDecision("other", "MASK", "test")])
        with self.assertRaises(ValueError):
            intervals_from_mentions(text, candidate["mentions"])


if __name__ == "__main__":
    unittest.main()
