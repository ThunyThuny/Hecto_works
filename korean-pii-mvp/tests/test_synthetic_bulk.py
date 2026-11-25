"""Full-size quota and deterministic-rendering regressions without ML packages."""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from kpii_mvp.b_inputs import allowed_to_keep
from kpii_mvp.synthetic_bulk import COUNTS, HISTOGRAMS, SUBROLES, TYPES, build_proposals, generate
from kpii_mvp.synthetic_bulk_audit import audit_bulk


class SyntheticBulkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs, cls.plan = build_proposals()

    def test_exact_full_scale_inventories_and_quotas(self):
        self.assertEqual(len(self.docs), 3000)
        self.assertEqual(sum(len(d.mentions) for d in self.docs), 16140)
        self.assertEqual(sum(m.entity_type == "NAME" for d in self.docs for m in d.mentions), 600)
        actual = Counter()
        for doc, record in zip(self.docs, self.plan["documents"]):
            for m, rel in zip(doc.mentions, record["relations"]):
                if m.entity_type != "NAME":
                    actual[doc.split, m.entity_type, rel["quota_subrole"]] += 1
        for split, values in SUBROLES.items():
            for kind in TYPES:
                for role, expected in values.items():
                    self.assertEqual(actual[split, kind, role], expected)

    def test_every_document_is_complete_and_has_varied_candidate_types(self):
        for doc, record in zip(self.docs, self.plan["documents"]):
            doc.validate()
            contextual = [m for m in doc.mentions if m.entity_type != "NAME"]
            self.assertTrue(3 <= len(contextual) <= 8)
            self.assertTrue(2 <= len({m.entity_type for m in contextual}) <= 4)
            self.assertTrue(3 <= len(doc.mentions) <= 9)
            for m in doc.mentions:
                self.assertEqual(doc.text[m.start:m.end], m.text)
            self.assertEqual(doc.trusted_context["document_type"], "general_notice")
            self.assertEqual(len(record["clause_ids"]), len(set(record["clause_ids"])))

    def test_titles_and_lengths_do_not_determine_contextual_mode(self):
        family_mode = defaultdict(Counter)
        length_mode = defaultdict(Counter)
        for doc, record in zip(self.docs, self.plan["documents"]):
            size = sum(m.entity_type != "NAME" for m in doc.mentions)
            family_mode[doc.split, doc.source_group][record["mode"]] += 1
            length_mode[doc.split, size][record["mode"]] += 1
        for (split, _), counts in family_mode.items():
            scale = COUNTS[split] // 50
            self.assertEqual(dict(counts), {"mixed": 3 * scale, "all_mask": scale, "all_keep": scale})
        for (split, size), counts in length_mode.items():
            scale = HISTOGRAMS[split][size] // 5
            self.assertEqual(dict(counts), {"mixed": 3 * scale, "all_mask": scale, "all_keep": scale})

    def test_personal_approval_counterexamples_and_global_contact_separation(self):
        values = defaultdict(set)
        approved_masks = Counter()
        for doc, record in zip(self.docs, self.plan["documents"]):
            for m, rel in zip(doc.mentions, record["relations"]):
                if m.entity_type in {"PHONE", "EMAIL", "ADDRESS"}:
                    values[m.entity_type, rel["relation"] == "private"].add(m.text.casefold())
                if rel["quota_subrole"] == "private_approved":
                    self.assertEqual(m.action, "MASK")
                    self.assertTrue(allowed_to_keep(doc, m))
                    approved_masks[doc.split, m.entity_type] += 1
        for kind in ("PHONE", "EMAIL", "ADDRESS"):
            self.assertFalse(values[kind, True] & values[kind, False])
        for split, counts in SUBROLES.items():
            for kind in TYPES:
                self.assertEqual(approved_masks[split, kind], counts["private_approved"])

    def test_full_audit_passes_without_metadata_shortcut(self):
        audit = audit_bulk(self.docs, self.plan)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["blockers"], [])
        self.assertEqual(audit["warnings"], [])
        self.assertEqual(audit["b_candidates"], 15540)
        self.assertEqual(audit["family_mode_max_absolute_proportion_gap"], 0)
        for metric in audit["metadata_approval_only_accuracy"].values():
            self.assertEqual(metric["accuracy"], 0.8)

    def test_seed_is_stable_across_python_hash_seeds(self):
        code = ("from kpii_mvp.synthetic_bulk import build_proposals; import hashlib,json; "
                "d,p=build_proposals(); print(hashlib.sha256(json.dumps([x.to_dict() for x in d],"
                "ensure_ascii=False,sort_keys=True).encode()).hexdigest())")
        outputs = []
        for hash_seed in ("17", "81"):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed)
            outputs.append(subprocess.check_output([sys.executable, "-c", code], env=env, text=True).strip())
        self.assertEqual(outputs[0], outputs[1])

    def test_nonempty_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            target = Path(temporary)
            sentinel = target / "human.csv"
            sentinel.write_text("human edits", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fresh version"):
                generate(target)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "human edits")

    def test_invalid_seed_fails(self):
        with self.assertRaises(ValueError):
            build_proposals(seed=True)


if __name__ == "__main__":
    unittest.main()
