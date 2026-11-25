"""Source declarations stay distinct from human review; no training is run."""
import unittest

from kpii_mvp.b_model import reviewed_data_provenance
from kpii_mvp.schema import Document


class ProvenanceTests(unittest.TestCase):
    def doc(self, source):
        return Document('d1', '검수 문서', [], 'g1', ['SCHOOL'], source=source)

    def test_synthetic_counts_use_exact_document_sources(self):
        result = reviewed_data_provenance([self.doc('synthetic_rule_based_v1'),
            self.doc('synthetic_rule_based_v1'), self.doc('synthetic:llm_v1')])
        self.assertEqual(result['classification'], 'synthetic')
        self.assertEqual(result['source_counts'], {'synthetic:llm_v1': 1, 'synthetic_rule_based_v1': 2})
        self.assertEqual(result['declared_category_counts'], {'synthetic': 3})

    def test_real_requires_an_explicit_source_declaration(self):
        self.assertEqual(reviewed_data_provenance([self.doc('real:internal_resume')])['classification'], 'real')
        for source in ('unknown', 'KDPII-v1', 'ALIO', 'not_synthetic', 'realistic', 'synthetically'):
            with self.subTest(source=source):
                self.assertEqual(reviewed_data_provenance([self.doc(source)])['classification'], 'unknown')

    def test_synthetic_and_real_are_mixed(self):
        result = reviewed_data_provenance([self.doc('synthetic_rule_based_v1'), self.doc('real:internal')])
        self.assertEqual(result['classification'], 'mixed')
        self.assertEqual(result['declared_category_counts'], {'real': 1, 'synthetic': 1})

    def test_unknown_source_prevents_claiming_complete_known_origin(self):
        result = reviewed_data_provenance([self.doc('synthetic_rule_based_v1'), self.doc('legacy')])
        self.assertEqual(result['classification'], 'unknown')
        self.assertEqual(result['declared_category_counts'], {'synthetic': 1, 'unknown': 1})

    def test_empty_input_is_unknown(self):
        self.assertEqual(reviewed_data_provenance([])['classification'], 'unknown')

    def test_explicit_mixed_and_casefold(self):
        self.assertEqual(reviewed_data_provenance([self.doc('mixed:curated')])['classification'], 'mixed')
        self.assertEqual(reviewed_data_provenance([self.doc('SYNTHETIC_RULE_BASED_V1')])['classification'], 'synthetic')
        self.assertEqual(reviewed_data_provenance([self.doc('synthetic/rules_v1')])['classification'], 'synthetic')


if __name__ == '__main__':
    unittest.main()
