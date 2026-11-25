"""Semantic integrity checks for fictional review proposals; no model is trained."""
from __future__ import annotations

import unittest
from collections import Counter, defaultdict

from kpii_mvp.b_data import validate_split_separation
from kpii_mvp.b_inputs import allowed_to_keep
from kpii_mvp.synthetic_v2 import (
    CLAUSE_BANKS, CONTACT_TYPES, GENERATOR_VERSION, POSITIONS, SPLITS, TYPES, build_proposals,
)


class SyntheticV2CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs, cls.plan = build_proposals()

    def test_exact_offsets_complete_relations_and_policy_compatible_labels(self):
        self.assertEqual(len(self.docs), 50)
        for doc, record in zip(self.docs, self.plan["documents"]):
            doc.validate()
            self.assertEqual(doc.source, GENERATOR_VERSION)
            self.assertEqual(doc.doc_id, record["doc_id"])
            self.assertEqual(len(doc.mentions), len(record["relations"]))
            self.assertEqual({m.entity_type for m in doc.mentions if m.entity_type != "NAME"}, set(record["focus_types"]))
            for mention, relation in zip(doc.mentions, record["relations"]):
                self.assertEqual(mention.mention_id, relation["mention_id"])
                self.assertEqual(doc.text[mention.start:mention.end], mention.text)
                expected = "KEEP" if relation["relation"] == "public_approved" else "MASK"
                self.assertEqual(mention.action, expected)
                if expected == "KEEP":
                    self.assertTrue(allowed_to_keep(doc, mention))
                if relation["relation"] == "public_unapproved":
                    self.assertFalse(allowed_to_keep(doc, mention))

    def test_global_private_and_public_contact_values_are_disjoint(self):
        values = defaultdict(set)
        for doc, record in zip(self.docs, self.plan["documents"]):
            for mention, relation in zip(doc.mentions, record["relations"]):
                if mention.entity_type in CONTACT_TYPES:
                    role = "private" if relation["relation"] == "private" else "public"
                    # Normalize exactly as consumers commonly normalize contacts.
                    normalized = "".join(mention.text.casefold().split())
                    values[(mention.entity_type, role)].add(normalized)
        for kind in CONTACT_TYPES:
            self.assertTrue(values[(kind, "private")])
            self.assertTrue(values[(kind, "public")])
            self.assertFalse(values[(kind, "private")] & values[(kind, "public")])

    def test_contact_format_does_not_encode_role(self):
        features = defaultdict(lambda: defaultdict(set))
        for doc, record in zip(self.docs, self.plan["documents"]):
            for mention, relation in zip(doc.mentions, record["relations"]):
                role = "private" if relation["relation"] == "private" else "public"
                if mention.entity_type == "PHONE":
                    self.assertTrue(mention.text.startswith("000-0000-"))
                    features["phone_parity"][role].add(int(mention.text[-1]) % 2)
                if mention.entity_type == "EMAIL":
                    self.assertTrue(mention.text.endswith(".example"))
                    local, domain = mention.text.split("@")
                    features["email_prefix"][role].add(local.rstrip("0123456789"))
                    features["email_domain"][role].add(domain)
                if mention.entity_type == "ADDRESS":
                    self.assertTrue(mention.text.startswith("가상시 "))
                    features["address_district"][role].add(mention.text.split()[1])
        for feature in features.values():
            self.assertEqual(feature["private"], feature["public"])

    def test_approved_institution_values_can_still_require_masking(self):
        observed = defaultdict(set)
        for doc, record in zip(self.docs, self.plan["documents"]):
            for mention, relation in zip(doc.mentions, record["relations"]):
                if mention.entity_type in set(TYPES) - CONTACT_TYPES:
                    if allowed_to_keep(doc, mention):
                        observed[(mention.entity_type, mention.text)].add(mention.action)
        for kind in set(TYPES) - CONTACT_TYPES:
            self.assertTrue(any(k == kind and labels == {"MASK", "KEEP"}
                                for (k, _), labels in observed.items()))

    def test_document_compositions_and_name_presence_vary(self):
        inventories = Counter(len(doc.mentions) for doc in self.docs)
        self.assertEqual(set(inventories), set(range(3, 10)))
        self.assertEqual({sum(m.entity_type == "NAME" for m in d.mentions) for d in self.docs}, {0, 1})
        combinations = set()
        for doc in self.docs:
            kinds = {m.entity_type for m in doc.mentions if m.entity_type != "NAME"}
            self.assertTrue(2 <= len(kinds) <= 4)
            combinations.add(tuple(sorted(kinds)))
        self.assertGreater(len(combinations), 15)
        for split in SPLITS:
            selected = [doc for doc in self.docs if doc.split == split]
            labels = [{m.action for m in doc.mentions if m.entity_type != "NAME"} for doc in selected]
            self.assertIn({"MASK"}, labels)
            self.assertIn({"KEEP"}, labels)
            self.assertIn({"MASK", "KEEP"}, labels)

    def test_split_type_action_coverage_and_no_family_overlap(self):
        partitions = {split: [doc for doc in self.docs if doc.split == split] for split in SPLITS}
        validate_split_separation(partitions)
        self.assertEqual([len(partitions[s]) for s in SPLITS], [30, 10, 10])
        for split in SPLITS:
            observed = {(m.entity_type, m.action) for d in partitions[split] for m in d.mentions}
            self.assertTrue({(kind, action) for kind in TYPES for action in ("MASK", "KEEP")} <= observed)
        sets = [set(self.plan["family_allocation"][s]) for s in SPLITS]
        self.assertTrue(all(len(items) == 5 for items in sets))
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_span_bearing_clause_banks_are_disjoint_across_splits(self):
        banks = []
        for split in SPLITS:
            bank = {clause for kind in CLAUSE_BANKS[split].values()
                    for clauses in kind.values() for clause in clauses}
            self.assertEqual(len(bank), 56)
            banks.append(bank)
        self.assertFalse(banks[0] & banks[1] or banks[0] & banks[2] or banks[1] & banks[2])

    def test_natural_titles_and_known_awkward_phrases_removed(self):
        for doc in self.docs:
            for bad in ("졸업한 학력을 적었습니다", "운영책임자였습니다고", "이었습니다고", "김가람가", "최다온가", "입학처"):
                self.assertNotIn(bad, doc.text)
            for mention in doc.mentions:
                if mention.entity_type == "POSITION":
                    self.assertIn(mention.text, POSITIONS)
                    self.assertNotIn("가상", mention.text)
                if mention.entity_type in {"WORKPLACE", "SCHOOL"}:
                    self.assertTrue(mention.text.startswith("가상"))

    def test_document_type_matches_sources(self):
        for doc, record in zip(self.docs, self.plan["documents"]):
            roles = {relation["relation"] for relation in record["relations"]} - {"name"}
            if "private" in roles and (roles - {"private"}):
                self.assertEqual(doc.trusted_context["document_type"], "general_notice")
            elif roles == {"private"}:
                self.assertEqual(doc.trusted_context["document_type"], "resume")
            else:
                self.assertIn(doc.trusted_context["document_type"], {"general_notice", "job_posting"})

    def test_school_clauses_do_not_introduce_an_unannotated_department(self):
        for split in SPLITS:
            for role in ("private", "public"):
                for clause in CLAUSE_BANKS[split]["SCHOOL"][role]:
                    self.assertNotIn("입학처", clause)

    def test_reproducibility_and_input_limits(self):
        first, first_plan = build_proposals(seed=123, train_count=7, valid_count=7, test_count=7)
        again, again_plan = build_proposals(seed=123, train_count=7, valid_count=7, test_count=7)
        other, _ = build_proposals(seed=456, train_count=7, valid_count=7, test_count=7)
        self.assertEqual(first_plan, again_plan)
        self.assertEqual([d.to_dict() for d in first], [d.to_dict() for d in again])
        self.assertNotEqual([d.text for d in first], [d.text for d in other])
        for kwargs in ({"seed": True}, {"train_count": 6}, {"valid_count": -1},
                       {"test_count": True}, {"train_count": 4000}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_proposals(**kwargs)


if __name__ == "__main__":
    unittest.main()
