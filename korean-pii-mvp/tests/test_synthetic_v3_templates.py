"""Structural checks for the expanded, independently held-out clause banks."""
from __future__ import annotations

from string import Formatter
import re
import unittest

from kpii_mvp.synthetic_v2 import _particles
from kpii_mvp.synthetic_v3_templates import CLAUSE_BANKS, FAMILY_WRAPPERS, TEMPLATE_SPEC


class SyntheticV3TemplateTests(unittest.TestCase):
    def test_each_split_has_all_types_and_minimum_role_variants(self):
        expected_types = {"PHONE", "EMAIL", "ADDRESS", "WORKPLACE", "SCHOOL", "DEPARTMENT", "POSITION"}
        self.assertEqual(set(CLAUSE_BANKS), {"train", "valid", "test"})
        for split, banks in CLAUSE_BANKS.items():
            self.assertEqual(set(banks), expected_types)
            for kind, roles in banks.items():
                self.assertEqual(set(roles), {"private", "public"})
                for role, variants in roles.items():
                    self.assertIsInstance(variants, tuple)
                    self.assertEqual(len(variants), 16 if split == "train" else 8, (split, kind, role))
                    self.assertEqual(len(set(variants)), len(variants))

    def test_single_value_slot_and_known_particle_slots(self):
        allowed = {"v", "obj", "subject", "topic", "as", "past", "reported"}
        for split, banks in CLAUSE_BANKS.items():
            for kind, roles in banks.items():
                for role, variants in roles.items():
                    for template in variants:
                        fields = [field for _, field, _, _ in Formatter().parse(template) if field]
                        self.assertEqual(fields.count("v"), 1, (split, kind, role, template))
                        self.assertTrue(set(fields).issubset(allowed))
                        self.assertNotIn("\n", template)
                        for value in ("가상한빛연구소", "운영지원팀", "팀장", "대리", "000-0000-1008"):
                            rendered = template.format(v=value, **_particles(value))
                            self.assertEqual(rendered.count(value), 1)
                            self.assertNotIn("{", rendered)

    def test_full_clauses_are_disjoint_across_splits(self):
        clauses = {split: {line for roles in banks.values() for variants in roles.values() for line in variants}
                   for split, banks in CLAUSE_BANKS.items()}
        for first, second in (("train", "valid"), ("train", "test"), ("valid", "test")):
            self.assertFalse(clauses[first] & clauses[second])

    def test_no_extra_named_department_or_approval_assertion(self):
        # Specific named entities must come from {v}, not unannotated boilerplate.
        forbidden = ("입학처", "인사팀", "운영지원팀", "가상시", "가상한빛", "승인된", "미승인", "허용 목록", "MASK", "KEEP")
        for banks in CLAUSE_BANKS.values():
            for roles in banks.values():
                for variants in roles.values():
                    for line in variants:
                        self.assertFalse(any(word in line for word in forbidden), line)

    def test_neutral_headings_are_unique_and_public_templates_are_shared(self):
        headings = [title for titles in FAMILY_WRAPPERS.values() for title in titles]
        self.assertEqual(len(headings), 30)
        self.assertEqual(len(set(headings)), 30)
        for titles in FAMILY_WRAPPERS.values():
            self.assertEqual(len(titles), 10)
        self.assertIs(TEMPLATE_SPEC["clause_banks"], CLAUSE_BANKS)
        self.assertTrue(TEMPLATE_SPEC["public_templates_shared_across_approval_states"])

    def test_private_contact_use_does_not_assert_exclusive_ownership(self):
        # An approved public value may still be proposed for MASK when the
        # occurrence routes an individual's application correspondence.
        forbidden = ("본인 소유", "개인 계정", "제 개인 번호", "자택", "집 주소", "거주지", "사는 곳", "생활 근거지")
        for banks in CLAUSE_BANKS.values():
            for kind in ("PHONE", "EMAIL", "ADDRESS"):
                for line in banks[kind]["private"]:
                    self.assertFalse(any(word in line for word in forbidden), line)

    def test_variable_value_uses_agreeing_particles(self):
        # 이며/입니다 are valid copula forms after either a vowel or a final
        # consonant. The value-dependent postpositions must use placeholders.
        for banks in CLAUSE_BANKS.values():
            for roles in banks.values():
                for variants in roles.values():
                    for line in variants:
                        self.assertIsNone(re.search(r"\{v\}(?:으로|로|은|는|을|를|가)", line), line)
                        self.assertIsNone(re.search(r"\{v\}이(?!며)", line), line)
        clause = next(line for line in CLAUSE_BANKS["train"]["DEPARTMENT"]["private"] if "이동한 뒤" in line)
        self.assertIn("인사팀으로 이동한 뒤", clause.format(v="인사팀", **_particles("인사팀")))
        self.assertIn("관리실로 이동한 뒤", clause.format(v="관리실", **_particles("관리실")))
        self.assertIn("관리부로 이동한 뒤", clause.format(v="관리부", **_particles("관리부")))


if __name__ == "__main__":
    unittest.main()
