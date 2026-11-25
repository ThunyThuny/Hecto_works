"""Small fictional fixtures exercise allocation failures without model calls."""
from __future__ import annotations

import copy
import unittest

from kpii_mvp.b_inputs import FIXED_POLICY_ID
from kpii_mvp.schema import Document, Mention
from kpii_mvp.synthetic_bulk_audit import GENERATOR_VERSION, TYPES, audit_bulk


def fixture(doc_id="one", split="train", mode="mixed", family=None,
            *, approved_private=True, with_name=False):
    roles = ({"mixed": ("private", "public_approved", "public_unapproved"),
              "all_mask": ("private", "public_unapproved"),
              "all_keep": ("public_approved", "public_approved")})[mode]
    text, mentions, relations, approved = "", [], [], []
    for index, role in enumerate(roles):
        value = f"{doc_id}-{index}@fiction.example"
        text += f"{doc_id}의 {index}번 문장 {role}: "
        start = len(text)
        text += value
        action = "KEEP" if role == "public_approved" else "MASK"
        mention = Mention(f"m{index}", start, len(text), value, "EMAIL", action)
        mentions.append(mention)
        text += "을 기록했습니다.\n"
        clause = f"{split}-{doc_id}-clause-{index}"
        relations.append({"mention_id": mention.mention_id, "relation": role,
                          "proposed_action": action, "clause_id": clause})
        if role == "public_approved" or role == "private" and approved_private:
            approved.append(value)
    if with_name:
        text += "작성자 "
        start = len(text)
        text += "김가람"
        mentions.append(Mention("name", start, len(text), "김가람", "NAME", "MASK"))
        text += "입니다.\n"
        relations.append({"mention_id": "name", "relation": "name",
                          "proposed_action": "MASK", "clause_id": f"{doc_id}-name"})
    document = Document(doc_id, text, mentions, f"person-{doc_id}",
                        ["EMAIL", "NAME"] if with_name else ["EMAIL"],
                        source=GENERATOR_VERSION, split=split, policy_id=FIXED_POLICY_ID,
                        trusted_context={"document_type": "general_notice",
                                         "approved_public_contacts": approved})
    record = {"doc_id": doc_id, "split": split, "family_id": family or f"{split}-family",
              "persona_id": document.source_group, "mode": mode,
              "relations": relations, "clause_ids": [item["clause_id"] for item in relations]}
    return document, record


def plan_for(pairs, family_count=1):
    plan = {"generator_version": GENERATOR_VERSION,
            "family_count_per_split": family_count,
            "documents": [record for _, record in pairs],
            "quota_targets": {}, "mode_targets": {}}
    for document, record in pairs:
        split = document.split
        quotas = plan["quota_targets"].setdefault(split, {
            kind: {"MASK": 0, "KEEP": 0} for kind in TYPES})
        modes = plan["mode_targets"].setdefault(split, {
            "mixed": 0, "all_mask": 0, "all_keep": 0})
        modes[record["mode"]] += 1
        for mention in document.mentions:
            if mention.entity_type in TYPES:
                quotas[mention.entity_type][mention.action] += 1
    return plan


def codes(report, key="blockers"):
    return {item["code"] for item in report[key]}


def subrole_fixture():
    """One of every refined role for every B type, with actual approvals."""
    text, mentions, relations = "", [], []
    context = {"document_type": "general_notice", "approved_public_contacts": [],
               "approved_public_orgs": [], "approved_public_positions": []}
    roles = (("private", True, "private_approved", "MASK"),
             ("private", False, "private_unapproved", "MASK"),
             ("public_unapproved", False, "public_unapproved", "MASK"),
             ("public_approved", True, "public_approved", "KEEP"))
    for kind in TYPES:
        for index, (role, approved, subrole, action) in enumerate(roles):
            value = f"{kind}항목{index + 1}값"
            if kind == "PHONE":
                value = f"000-1000-{index + 1000}"
            elif kind == "EMAIL":
                value = f"unit{index}@fiction.example"
            text += f"유형 {kind}, 관계 {role}: "
            start = len(text)
            text += value
            mention_id = f"{kind}-{index}"
            mentions.append(Mention(mention_id, start, len(text), value, kind, action))
            text += "입니다.\n"
            relations.append({"mention_id": mention_id, "relation": role,
                              "proposed_action": action, "quota_subrole": subrole,
                              "simulated_approved_value": approved,
                              "clause_id": f"train-{mention_id}"})
            if approved:
                key = ("approved_public_contacts" if kind in {"PHONE", "EMAIL", "ADDRESS"}
                       else "approved_public_positions" if kind == "POSITION" else "approved_public_orgs")
                context[key].append(value)
    document = Document("subroles", text, mentions, "person-subroles", list(TYPES),
                        source=GENERATOR_VERSION, split="train", policy_id=FIXED_POLICY_ID,
                        trusted_context=context)
    record = {"doc_id": document.doc_id, "split": "train", "family_id": "train-family",
              "mode": "mixed", "relations": relations,
              "clause_ids": [row["clause_id"] for row in relations]}
    plan = plan_for([(document, record)])
    plan["subrole_targets_per_type"] = {"train": {
        "private_approved": 1, "private_unapproved": 1,
        "public_unapproved": 1, "public_approved": 1}}
    return document, plan


class SyntheticBulkAuditTests(unittest.TestCase):
    def test_exact_small_corpus_passes_without_mutation_or_approval(self):
        pairs = [fixture(split, split, with_name=True)
                 for split in ("train", "valid", "test")]
        documents = [doc for doc, _ in pairs]
        plan = plan_for(pairs)
        before = [doc.to_dict() for doc in documents], copy.deepcopy(plan)
        report = audit_bulk(iter(documents), plan)
        self.assertTrue(report["passed"], report["blockers"])
        self.assertEqual(report["b_candidates"], 9)
        self.assertEqual(report["name_candidates_excluded_from_b"], {
            "train": 1, "valid": 1, "test": 1})
        metrics = report["metadata_approval_only_accuracy"]["EMAIL"]
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["approved_private_mask_fraction"], 1 / 3)
        self.assertEqual(report["review_status"], "PENDING")
        self.assertTrue(report["human_review_required"])
        self.assertFalse(report["training_executed"])
        self.assertEqual(before, ([doc.to_dict() for doc in documents], plan))

    def test_type_action_quota_is_exact(self):
        pair = fixture()
        plan = plan_for([pair])
        plan["quota_targets"]["train"]["EMAIL"]["MASK"] += 1
        report = audit_bulk([pair[0]], plan)
        self.assertIn("type_action_quota_mismatch", codes(report))

    def test_missing_type_or_boolean_quota_cannot_pass(self):
        pair = fixture()
        for mutation, expected in (
                (lambda plan: plan["quota_targets"]["train"].pop("PHONE"), "invalid_type_quota_inventory"),
                (lambda plan: plan["quota_targets"]["train"]["EMAIL"].update(MASK=True), "invalid_type_action_quota")):
            with self.subTest(expected=expected):
                plan = plan_for([pair])
                mutation(plan)
                self.assertIn(expected, codes(audit_bulk([pair[0]], plan)))

    def test_declared_document_mode_and_mode_quota_must_match(self):
        pair = fixture()
        plan = plan_for([pair])
        plan["documents"][0]["mode"] = "all_keep"
        plan["mode_targets"]["train"]["mixed"] += 1
        report = audit_bulk([pair[0]], plan)
        self.assertIn("document_mode_mismatch", codes(report))
        self.assertIn("document_mode_quota_mismatch", codes(report))

    def test_family_mode_contingency_balanced_then_confounded(self):
        pairs = []
        for family in ("family-a", "family-b"):
            for index, mode in enumerate(("mixed", "mixed", "mixed", "all_mask", "all_keep")):
                pairs.append(fixture(f"{family}-{index}", mode=mode, family=family))
        plan = plan_for(pairs, family_count=2)
        report = audit_bulk([doc for doc, _ in pairs], plan)
        self.assertTrue(report["passed"], report["blockers"])
        self.assertEqual(report["family_mode_max_absolute_proportion_gap"], 0)
        # Preserve corpus quotas but move an all-KEEP and mixed record between families.
        plan["documents"][4]["family_id"] = "family-b"
        plan["documents"][5]["family_id"] = "family-a"
        report = audit_bulk([doc for doc, _ in pairs], plan)
        self.assertIn("family_mode_contingency_mismatch", codes(report))
        self.assertGreater(report["family_mode_max_absolute_proportion_gap"], 0)

    def test_family_ids_must_not_cross_splits(self):
        pairs = [fixture("one", "train", family="shared"),
                 fixture("two", "test", family="shared")]
        report = audit_bulk([doc for doc, _ in pairs], plan_for(pairs))
        self.assertIn("family_id_crosses_splits", codes(report))

    def test_approval_shortcut_and_few_counterexamples_are_warnings(self):
        pair = fixture(approved_private=False)
        report = audit_bulk([pair[0]], plan_for([pair]))
        self.assertTrue(report["passed"], report["blockers"])
        self.assertIn("high_metadata_approval_only_accuracy", codes(report, "warnings"))
        self.assertIn("insufficient_approved_private_mask_counterexamples", codes(report, "warnings"))
        self.assertEqual(report["metadata_approval_only_accuracy"]["EMAIL"]["accuracy"], 1.0)

    def test_inherited_semantic_and_span_checks_still_apply(self):
        document, record = fixture()
        plan = plan_for([(document, record)])
        object.__setattr__(document.mentions[0], "action", "KEEP")
        report = audit_bulk([document], plan)
        self.assertIn("semantic_label_mismatch", codes(report))
        self.assertIn("type_action_quota_mismatch", codes(report))
        object.__setattr__(document.mentions[0], "start", 0)
        report = audit_bulk([document], plan)
        self.assertIn("invalid_document_schema_or_spans", codes(report))

    def test_empty_or_wrong_plan_cannot_pass(self):
        report = audit_bulk([], {})
        self.assertIn("empty_corpus", codes(report))
        self.assertIn("wrong_bulk_generator_version", codes(report))
        self.assertIn("missing_or_invalid_quota_targets", codes(report))
        self.assertIn("missing_or_invalid_mode_targets", codes(report))

    def test_split_inventory_and_family_count_are_checked(self):
        pair = fixture()
        plan = plan_for([pair])
        plan["quota_targets"]["test"] = copy.deepcopy(plan["quota_targets"]["train"])
        plan["family_count_per_split"] = 10
        report = audit_bulk([pair[0]], plan)
        self.assertIn("quota_split_inventory_mismatch", codes(report))
        self.assertIn("family_count_mismatch", codes(report))

    def test_refined_subrole_quotas_are_derived_from_actual_approvals(self):
        document, plan = subrole_fixture()
        report = audit_bulk([document], plan)
        self.assertTrue(report["passed"], report["blockers"])
        self.assertEqual(report["b_type_subrole_counts"]["train"]["EMAIL"], {
            "private_approved": 1, "private_unapproved": 1,
            "public_unapproved": 1, "public_approved": 1})
        plan["subrole_targets_per_type"]["train"]["private_approved"] = 2
        report = audit_bulk([document], plan)
        mismatches = [row for row in report["blockers"]
                      if row["code"] == "type_subrole_quota_mismatch"]
        self.assertEqual(len(mismatches), 7)
        self.assertEqual({row["entity_type"] for row in mismatches}, set(TYPES))

    def test_declared_subrole_and_simulated_approval_are_not_trusted(self):
        document, plan = subrole_fixture()
        relation = plan["documents"][0]["relations"][0]
        relation["quota_subrole"] = "private_unapproved"
        relation["simulated_approved_value"] = False
        report = audit_bulk([document], plan)
        self.assertIn("declared_quota_subrole_mismatch", codes(report))
        self.assertIn("declared_simulated_approval_mismatch", codes(report))
        # Equality with integer one must not pass a claimed boolean field.
        relation["simulated_approved_value"] = 1
        report = audit_bulk([document], plan)
        self.assertIn("declared_simulated_approval_mismatch", codes(report))

    def test_refined_subrole_target_inventory_and_counts_are_validated(self):
        document, plan = subrole_fixture()
        plan["subrole_targets_per_type"]["test"] = {}
        plan["subrole_targets_per_type"]["train"]["private_approved"] = -1
        report = audit_bulk([document], plan)
        self.assertIn("subrole_split_inventory_mismatch", codes(report))
        self.assertIn("invalid_subrole_targets", codes(report))

    def test_repeated_clause_id_and_identical_candidate_line_are_blocked(self):
        document, record = fixture()
        plan = plan_for([(document, record)])
        plan["documents"][0]["clause_ids"][1] = record["clause_ids"][0]
        report = audit_bulk([document], plan)
        self.assertIn("repeated_clause_id_within_document", codes(report))
        # A different value does not trigger a full-line duplicate check.
        self.assertNotIn("repeated_span_bearing_line_within_document", codes(report))

        first_line = document.text.splitlines(keepends=True)[0]
        original = document.mentions[0]
        new_mention = Mention("repeated", len(document.text) + original.start,
                              len(document.text) + original.end, original.text,
                              original.entity_type, original.action)
        repeated_document = Document.from_dict({
            **document.to_dict(), "text": document.text + first_line,
            "mentions": [mention.to_dict() for mention in document.mentions + [new_mention]]})
        repeated_record = copy.deepcopy(record)
        repeated_record["relations"].append({
            **record["relations"][0], "mention_id": "repeated", "clause_id": "fresh-clause"})
        repeated_record["clause_ids"].append("fresh-clause")
        report = audit_bulk([repeated_document], plan_for([(repeated_document, repeated_record)]))
        self.assertIn("repeated_span_bearing_line_within_document", codes(report))


if __name__ == "__main__":
    unittest.main()
