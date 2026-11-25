"""Quota-controlled fictional B corpus. Generates PENDING packets, never trains."""
from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile

from .b_data import export_review, validate_split_separation
from .b_inputs import FIXED_POLICY_ID, policy_sha256
from .data import write_jsonl
from .schema import Document, Mention
from .synthetic_v1 import _write_preview
from .synthetic_v2 import TYPES, SPLITS, CONTACT_TYPES, NAMES, _particles, _institution, _approval_key
from .synthetic_v3_templates import CLAUSE_BANKS, FAMILY_WRAPPERS, TEMPLATE_SPEC

GENERATOR_VERSION = "synthetic_rule_based_v3_bulk"
COUNTS = {"train": 2000, "valid": 500, "test": 500}
HISTOGRAMS = {"train": {3: 200, 4: 400, 5: 600, 6: 400, 7: 300, 8: 100},
              "valid": {3: 60, 4: 130, 5: 135, 6: 100, 7: 55, 8: 20},
              "test": {3: 60, 4: 130, 5: 135, 6: 100, 7: 55, 8: 20}}
SUBROLES = {"train": {"private_approved": 300, "private_unapproved": 375,
                       "public_unapproved": 150, "public_approved": 675},
            "valid": {"private_approved": 72, "private_unapproved": 72,
                       "public_unapproved": 36, "public_approved": 180},
            "test": {"private_approved": 72, "private_unapproved": 72,
                      "public_unapproved": 36, "public_approved": 180}}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


class _Contacts:
    """Separate shuffled inventories; large address space, neutral role formats."""
    def __init__(self, rng):
        self.rng = rng
        codes = list(range(1, 100000))
        rng.shuffle(codes)
        self.codes = {"private": codes[::2], "public": codes[1::2]}
        self.cursor = Counter()
        self.used = {kind: set() for kind in CONTACT_TYPES}

    def value(self, kind, private):
        bucket = "private" if private else "public"
        while self.cursor[kind, bucket] < len(self.codes[bucket]):
            code = self.codes[bucket][self.cursor[kind, bucket]]
            self.cursor[kind, bucket] += 1
            if kind == "PHONE":
                value = f"000-{code // 10000:04d}-{code % 10000:04d}"
            elif kind == "EMAIL":
                value = f"{self.rng.choice(('mail','box','inbox','contact'))}{code}@{self.rng.choice(('naru','hanbit','saebom','daon'))}.example"
            else:
                district = self.rng.choice(("다온구", "누리구", "소담구", "해솔구", "온유구", "나루구", "새봄구", "한빛구"))
                road = self.rng.choice(("느티로", "솔빛로", "새봄길", "나루로", "은행길", "들꽃로", "도담길", "온유로", "아람길", "예담로", "푸른길", "한빛로"))
                value = f"가상시 {district} {road} {code % 997 + 1}"
            if value not in self.used[kind]:
                self.used[kind].add(value)
                return value
        raise ValueError("Fictional contact inventory exhausted")


def _flow_assign(documents, action, supplies, rng):
    """Integral bipartite flow assigns exact type quotas to full-document slots."""
    # Mutable edges are [to, reverse_index, residual_capacity].
    graph = [[] for _ in range(2 + len(TYPES) + len(documents))]
    source, sink = len(graph) - 2, len(graph) - 1

    def edge(left, right, capacity):
        forward = [right, len(graph[right]), capacity]
        backward = [left, len(graph[left]), 0]
        graph[left].append(forward)
        graph[right].append(backward)
        return forward

    for i, kind in enumerate(TYPES):
        edge(source, i, supplies[kind])
    references = []
    indices = list(range(len(documents)))
    rng.shuffle(indices)
    for j in indices:
        doc = documents[j]
        demand = doc["remaining"][action]
        if not demand:
            continue
        node = len(TYPES) + j
        edge(node, sink, demand)
        kinds = list(doc["allowed_types"])
        rng.shuffle(kinds)
        for kind in kinds:
            forward = edge(TYPES.index(kind), node, demand)
            references.append((j, kind, forward, demand))
    expected = sum(supplies.values())
    assert expected == sum(d["remaining"][action] for d in documents)
    total = 0
    while True:
        level = [-1] * len(graph)
        level[source] = 0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for other, _, capacity in graph[node]:
                if capacity and level[other] < 0:
                    level[other] = level[node] + 1
                    queue.append(other)
        if level[sink] < 0:
            break
        cursors = [0] * len(graph)

        def push(node, amount):
            if node == sink:
                return amount
            while cursors[node] < len(graph[node]):
                item = graph[node][cursors[node]]
                other, reverse, capacity = item
                if capacity and level[other] == level[node] + 1:
                    sent = push(other, min(amount, capacity))
                    if sent:
                        item[2] -= sent
                        graph[other][reverse][2] += sent
                        return sent
                cursors[node] += 1
            return 0

        while True:
            sent = push(source, expected - total)
            if not sent:
                break
            total += sent
    if total != expected:
        raise ValueError("Cannot fill type quotas within document inventories")
    for j, kind, item, capacity in references:
        documents[j]["rows"].extend({"kind": kind, "action": action} for _ in range(capacity - item[2]))


def _document_slots(split, rng):
    """Balance mode within each title family and candidate-count stratum."""
    mode_sizes = {"mixed": [], "all_mask": [], "all_keep": []}
    for size, count in HISTOGRAMS[split].items():
        mode_sizes["mixed"].extend([size] * (count * 3 // 5))
        mode_sizes["all_mask"].extend([size] * (count // 5))
        mode_sizes["all_keep"].extend([size] * (count // 5))
    slots = []
    for mode, sizes in mode_sizes.items():
        rng.shuffle(sizes)
        family_slots = [i % 10 for i in range(len(sizes))]
        rng.shuffle(family_slots)
        for size, family in zip(sizes, family_slots):
            slots.append({"mode": mode, "size": size, "family": family, "rows": []})
    mixed = [d for d in slots if d["mode"] == "mixed"]
    mask_target = sum(SUBROLES[split][r] for r in SUBROLES[split] if r != "public_approved") * len(TYPES)
    remaining_mask = mask_target - sum(d["size"] for d in slots if d["mode"] == "all_mask")
    # Each mixed document has at least one MASK and one KEEP candidate.
    lower = [max(1, d["size"] - 4) for d in mixed]
    upper = [min(4, d["size"] - 1) for d in mixed]
    tickets = [j for j in range(len(mixed)) for _ in range(upper[j] - lower[j])]
    rng.shuffle(tickets)
    extras = remaining_mask - sum(lower)
    if not 0 <= extras <= len(tickets):
        raise ValueError("Incompatible document/label quotas")
    increments = Counter(tickets[:extras])
    for j, doc in enumerate(mixed):
        doc["remaining"] = {"MASK": lower[j] + increments[j], "KEEP": doc["size"] - lower[j] - increments[j]}
    for doc in slots:
        if doc["mode"] != "mixed":
            doc["remaining"] = {"MASK": doc["size"] if doc["mode"] == "all_mask" else 0,
                                "KEEP": doc["size"] if doc["mode"] == "all_keep" else 0}
    # NAME frequency is independent of contextual mode and family; NAME itself
    # is policy-fixed MASK and excluded from the contextual mode definition.
    for mode in mode_sizes:
        for family in range(10):
            group = [d for d in slots if d["mode"] == mode and d["family"] == family]
            rng.shuffle(group)
            for j, doc in enumerate(group):
                doc["name"] = j < len(group) // 5
    rng.shuffle(slots)
    return slots


def build_proposals(*, seed=20261021):
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    rng = random.Random(seed)
    contacts = _Contacts(rng)
    documents, records = [], []
    quota_targets, mode_targets = {}, {}
    family_allocation = {split: [f"v3-{split}-family-{i:02d}" for i in range(1, 11)] for split in SPLITS}
    for split in SPLITS:
        slots = _document_slots(split, rng)
        per_type = SUBROLES[split]
        mask_per_type = sum(n for role, n in per_type.items() if role != "public_approved")
        quota_targets[split] = {kind: {"MASK": mask_per_type, "KEEP": per_type["public_approved"]} for kind in TYPES}
        mode_targets[split] = dict(Counter(d["mode"] for d in slots))
        available = {a: {kind: quota_targets[split][kind][a] for kind in TYPES} for a in ("MASK", "KEEP")}
        # Two distinct anchors guarantee 2..4 contextual types in every document.
        for doc in slots:
            actions = (["MASK", "KEEP"] if doc["mode"] == "mixed" else
                       ["MASK", "MASK"] if doc["mode"] == "all_mask" else ["KEEP", "KEEP"])
            selected = []
            for action in actions:
                choices = [k for k in TYPES if k not in selected and available[action][k] > 0]
                kind = max(choices, key=lambda k: available[action][k] + rng.random() * 40)
                available[action][kind] -= 1
                doc["remaining"][action] -= 1
                doc["rows"].append({"kind": kind, "action": action})
                selected.append(kind)
            others = [kind for kind in TYPES if kind not in selected]
            rng.shuffle(others)
            doc["allowed_types"] = selected + others[:min(doc["size"], rng.choice((3, 4))) - 2]
        for action in ("MASK", "KEEP"):
            _flow_assign(slots, action, available[action], rng)
        roles = {}
        for kind in TYPES:
            roles[kind] = [role for role, count in per_type.items() if role != "public_approved" for _ in range(count)]
            rng.shuffle(roles[kind])
        for doc_index, slot in enumerate(slots, 1):
            rows = slot["rows"]
            rng.shuffle(rows)
            for row in rows:
                subrole = "public_approved" if row["action"] == "KEEP" else roles[row["kind"]].pop()
                row["subrole"] = subrole
                row["role"] = "private" if subrole.startswith("private_") else subrole
                row["approved"] = subrole in {"private_approved", "public_approved"}
            approved_values = {}
            for kind in sorted({r["kind"] for r in rows if r["kind"] not in CONTACT_TYPES}):
                approved_values[kind] = _institution(kind, rng)
            context = {"document_type": "general_notice", "approved_public_contacts": [],
                       "approved_public_orgs": [], "approved_public_positions": []}
            for row in rows:
                kind = row["kind"]
                if kind in CONTACT_TYPES:
                    value = contacts.value(kind, row["role"] == "private")
                elif row["approved"]:
                    value = approved_values[kind]
                else:
                    value = _institution(kind, rng)
                    while value == approved_values[kind]:
                        value = _institution(kind, rng)
                row["value"] = value
                if row["approved"]:
                    context[_approval_key(kind)].append(value)
            for key in ("approved_public_contacts", "approved_public_orgs", "approved_public_positions"):
                context[key] = sorted(set(context[key]))
            doc_id = f"syn-v3-{seed}-{split}-{doc_index:04d}"
            family_id = family_allocation[split][slot["family"]]
            parts = [FAMILY_WRAPPERS[split][slot["family"]] + "\n",
                     "각 항목은 서로 다른 제출 자료와 기관 안내에서 발췌했습니다. 항목 번호가 다르면 출처도 다릅니다.\n"]
            mentions, relations, clauses = [], [], []
            cursor = sum(len(p) for p in parts)

            def add(value):
                nonlocal cursor
                parts.append(value)
                cursor += len(value)

            def mention(value, kind, role, subrole, approved, clause_id):
                start = cursor
                add(value)
                mid = f"m{len(mentions) + 1:02d}"
                action = "KEEP" if role == "public_approved" else "MASK"
                reason = ("POLICY_ALWAYS_MASK_NAME" if role == "name" else
                          "PERSONAL_USE_CONTEXT" if role == "private" else
                          "PUBLIC_ROLE_APPROVED_VALUE" if approved else "PUBLIC_LOOKING_UNAPPROVED_VALUE")
                mentions.append(Mention(mid, start, cursor, value, kind, action, reason))
                relations.append({"mention_id": mid, "relation": role, "quota_subrole": subrole,
                                  "proposed_action": action, "clause_id": clause_id,
                                  "simulated_approved_value": approved,
                                  "source_item_id": f"{doc_id}:item-{len(mentions):02d}"})
                clauses.append({"clause_id": clause_id, "relation": role, "mention_id": mid})

            if slot["name"]:
                add("자료 항목 1: 제출 자료의 작성자: ")
                mention(rng.choice(NAMES), "NAME", "name", "name", False, f"{split}.name.identity")
                add(".\n")
            used_clause_ids = set()
            for index, row in enumerate(rows):
                kind, role, value = row["kind"], row["role"], row["value"]
                bank_role = "private" if role == "private" else "public"
                bank = CLAUSE_BANKS[split][kind][bank_role]
                variant = rng.randrange(len(bank))
                # Reuse of a type in separate sources should still use a
                # different sentence. Fallback consumes no extra randomness,
                # keeping all following values and scenario assignments stable.
                attempts = 0
                while f"{split}.{kind.lower()}.{bank_role}.{variant + 1}" in used_clause_ids:
                    variant = (variant + 1) % len(bank)
                    attempts += 1
                    if attempts >= len(bank):
                        raise ValueError("Not enough distinct clauses inside one document")
                clause_id = f"{split}.{kind.lower()}.{bank_role}.{variant + 1}"
                used_clause_ids.add(clause_id)
                template = bank[variant]
                for key, particle in _particles(value).items():
                    template = template.replace("{" + key + "}", particle)
                before, after = template.split("{v}")
                add(f"자료 항목 {len(mentions) + 1}: " + before)
                mention(value, kind, role, row["subrole"], row["approved"], clause_id)
                add(after + ("\n" if index + 1 < len(rows) else ""))
            doc = Document(doc_id, "".join(parts), mentions, source_group=family_id,
                           annotation_scope=["NAME", *TYPES], source=GENERATOR_VERSION,
                           split=split, policy_id=FIXED_POLICY_ID, trusted_context=context)
            doc.validate()
            documents.append(doc)
            records.append({"doc_id": doc_id, "split": split, "family_id": family_id,
                            "persona_id": f"fictional-v3-{seed}-{split}-{doc_index:04d}",
                            "source_item_ids": [r["source_item_id"] for r in relations],
                            "mode": slot["mode"], "focus_types": sorted({r["kind"] for r in rows}),
                            "relations": relations, "clause_ids": [c["clause_id"] for c in clauses], "clauses": clauses})
        if any(roles.values()):
            raise ValueError("Unconsumed label-role quota")
    validate_split_separation({s: [d for d in documents if d.split == s] for s in SPLITS})
    plan = {"generator_version": GENERATOR_VERSION, "seed": seed, "split_counts": dict(COUNTS),
            "family_count_per_split": 10, "family_allocation": family_allocation,
            "quota_targets": quota_targets, "mode_targets": mode_targets,
            "subrole_targets_per_type": {s: dict(values) for s, values in SUBROLES.items()}, "documents": records,
            "limitations": ["Fictional rule-based compilation records; not authentic resumes or public notices.",
                            "Approval registries are simulated; approved private uses remain MASK under the fixed context policy.",
                            "Every numbered item has a separate source to avoid contradictory personal histories.",
                            "Name candidates are fixed MASK and excluded from B quotas and contextual document modes.",
                            "Split clause banks and families are distinct but vocabulary and relation semantics remain shared.",
                            "Generated labels remain PENDING until actual human review; no real-world performance is established."]}
    return documents, plan


def generate(output_dir, *, seed=20261021):
    from .synthetic_bulk_audit import audit_bulk
    destination = Path(output_dir).absolute()
    if destination.is_symlink() or destination.exists():
        raise ValueError("Output must be a fresh version directory; existing data is never overwritten")
    docs, plan = build_proposals(seed=seed)
    audit = audit_bulk(docs, plan)
    if audit["blockers"] or audit["warnings"]:
        raise ValueError("Bulk audit did not pass: " + json.dumps({k: audit[k] for k in ("blockers", "warnings")}, ensure_ascii=False))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        write_jsonl(temporary / "proposed_documents.jsonl", docs)
        for split in SPLITS:
            write_jsonl(temporary / "proposals" / f"{split}.jsonl", [d for d in docs if d.split == split])
        _json(temporary / "generation_plan.json", plan)
        _json(temporary / "automated_audit.json", audit)
        export_review(docs, temporary / "review", policy_id=FIXED_POLICY_ID)
        _write_preview(temporary, docs)
        # Normalize packet text to LF on every host before binding byte hashes.
        for path in temporary.rglob("*"):
            if path.is_file():
                path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
        sources = {name: _sha(Path(__file__).with_name(name)) for name in
                   ("synthetic_bulk.py", "synthetic_bulk_audit.py", "synthetic_v3_templates.py",
                    "synthetic_audit.py", "synthetic_v1.py", "synthetic_v2.py")}
        manifest = {"generator_version": GENERATOR_VERSION, "seed": seed,
                    "policy_id": FIXED_POLICY_ID, "policy_sha256": policy_sha256(),
                    "generator_sources_sha256": sources,
                    "templates_sha256": hashlib.sha256(json.dumps(TEMPLATE_SPEC, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                    "documents": len(docs), "split_counts": COUNTS,
                    "mentions": sum(len(d.mentions) for d in docs),
                    "contextual_candidates": sum(m.entity_type != "NAME" for d in docs for m in d.mentions),
                    "proposed_action_counts": dict(Counter(m.action for d in docs for m in d.mentions)),
                    "quota_targets": plan["quota_targets"], "mode_targets": plan["mode_targets"],
                    "review_status": "PENDING", "training_executed": False,
                    "metadata_provenance": "SIMULATED fictional approval registry, not externally verified",
                    "limitations": plan["limitations"],
                    "files_sha256": {p.relative_to(temporary).as_posix(): _sha(p) for p in sorted(temporary.rglob("*")) if p.is_file()}}
        _json(temporary / "generation_manifest.json", manifest)
        if destination.exists():
            raise ValueError("Destination appeared during generation")
        temporary.rename(destination)
    finally:
        if temporary.exists():
            assert temporary.resolve().parent == destination.parent.resolve()
            assert temporary.name.startswith(f".{destination.name}.staging-")
            shutil.rmtree(temporary)
    return {"output_dir": str(destination), "generator_version": GENERATOR_VERSION,
            "documents": len(docs), "mentions": manifest["mentions"],
            "contextual_candidates": manifest["contextual_candidates"],
            "review_status": "PENDING", "training_executed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("generate")
    command.add_argument("--output-dir", required=True)
    command.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.output_dir, seed=args.seed), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
