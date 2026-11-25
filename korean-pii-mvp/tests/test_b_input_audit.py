from dataclasses import replace
import unittest
from kpii_mvp.b_input_audit import audit_inputs
from test_b_model import CharacterTokenizer, document


class BInputAuditTests(unittest.TestCase):
    def test_token_lengths_are_checked_without_training_or_dropping_context(self):
        tokenizer, doc = CharacterTokenizer(), document()
        good = audit_inputs(tokenizer, {"train": [doc]}, max_length=1000)
        self.assertTrue(good["passed"])
        longest = good["splits"]["train"]["max_observed_tokens"]
        self.assertGreater(longest, 8)
        bad = audit_inputs(tokenizer, {"train": [doc]}, max_length=longest - 1)
        self.assertFalse(bad["passed"])
        self.assertEqual(bad["failures"][0]["reason"], "context_exceeds_max_length")
        self.assertFalse(bad["model_weights_loaded"])
        self.assertFalse(bad["training_performed"])

    def test_reserved_source_markers_cannot_pass_preflight(self):
        doc = document()
        changed = replace(doc, text=doc.text + " [CAND_START]")
        report = audit_inputs(CharacterTokenizer(), {"valid": [changed]})
        self.assertFalse(report["passed"])
        self.assertEqual(report["failures"][0]["reason"], "reserved_marker_in_source")
