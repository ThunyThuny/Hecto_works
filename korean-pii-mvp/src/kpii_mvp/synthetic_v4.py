"""Generate versioned, PENDING V4 B proposals; no model work is performed.

V3's quota allocation is reused, but all values, identities and documents are
rendered again. New POSITION/DEPARTMENT clauses teach occurrence ownership,
including approved values in personal histories. This is a new synthetic
holdout created after V3 error analysis, not independent real-world evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random
import re
import shutil
import tempfile

from .b_data import export_review, validate_split_separation
from .b_inputs import FIXED_POLICY_ID, allowed_to_keep, policy_sha256
from .data import write_jsonl
from .schema import Document, Mention
from .synthetic_bulk import COUNTS, build_proposals as _allocate, _json, _sha
from .synthetic_bulk_audit import audit_bulk
from .synthetic_v1 import _write_preview
from .synthetic_v2 import CONTACT_TYPES, SPLITS, TYPES, _approval_key, _particles
from .synthetic_v3_templates import CLAUSE_BANKS, FAMILY_WRAPPERS

GENERATOR_VERSION = "synthetic_rule_based_v4_bulk"
DEFAULT_SEED = 20261029

# Values are fictional compositions, not collected people, institutions or
# addresses. The three entire value inventories are disjoint. Common title
# lexemes deliberately appear in both personal and public roles within a split.
LEXICONS = {
    "train": {
        "POSITION": ("대리", "과장", "매니저", "선임연구원", "운영책임자", "팀장", "연구개발팀장", "서비스매니저"),
        "DEPARTMENT": ("채용지원팀", "운영지원팀", "사업지원팀", "기술지원팀", "연구기획팀", "인사관리팀", "교육지원팀", "고객지원팀"),
        "stems": ("여울빛", "솔담", "이든봄", "온새결", "다솜빛", "해든솔"),
        "cities": ("솔담시", "여울시", "해든시"),
        "names": ("오솔비", "류해담", "신여울", "백도윤"),
    },
    "valid": {
        "POSITION": ("주임", "차장", "책임연구원", "센터장", "프로젝트매니저", "실무책임자", "기술연구원", "기획실장"),
        "DEPARTMENT": ("인재운영팀", "서비스운영팀", "경영관리팀", "품질관리팀", "연구개발실", "사업기획팀", "시설운영팀", "교육운영실"),
        "stems": ("윤슬결", "늘해랑", "가온담", "새온빛", "도래솔", "모아든"),
        "cities": ("윤슬시", "가온담시", "새온시"),
        "names": ("문서하", "장하늬", "서이든", "홍새온"),
    },
    "test": {
        "POSITION": ("부장", "수석연구원", "책임매니저", "사업실장", "교육센터장", "기술책임자", "기획연구원", "서비스팀장"),
        "DEPARTMENT": ("전형지원팀", "인재개발실", "경영지원실", "서비스기획팀", "사업개발팀", "운영관리실", "연구운영팀", "고객상담실"),
        "stems": ("보듬솔", "소율빛", "다온결", "푸른담", "아람결", "해밀담"),
        "cities": ("보듬시", "소율시", "해밀담시"),
        "names": ("안가을", "전시온", "노해온", "배나온"),
    },
}

# Each split has genuinely different span-bearing sentences, rather than only
# a different outer heading. Templates contain one value placeholder; its
# occurrence alone is annotated. Approval status is never written in the text.
FOCUS_CLAUSES = {
    "train": {
        "POSITION": {
            "private": (
                "공고의 자격 요건과 별개로, 지원자가 경력란에 적은 이전 직위는 {v}입니다.",
                "입사 지원자는 전 직장에서 {v}{as} 일하며 맡았던 업무를 설명했습니다.",
                "모집 안내를 읽은 지원자가 자신의 재직 직급을 덧붙였습니다. 당시 직급: {v}.",
                "경력증명서의 직위 칸에는 {v}{subject} 적혀 있습니다.",
                "지원자는 채용 문의를 하면서 본인의 마지막 직위가 {v}{reported}고 밝혔습니다.",
                "업무 인수인계 경험을 설명한 이력서에서 본인의 직위는 {v}{past}.",
                "지원자가 제출한 재직 내역 | 맡았던 직위 | {v}.",
                "이전 회사의 채용 업무를 담당했던 지원자는 그때 {v}{past}.",
            ),
            "public": (
                "개별 지원자의 경력과 별도로, 이번 공고에서 모집하는 직위는 {v}입니다.",
                "기관은 {v} 채용을 위해 업무 범위와 지원 요건을 안내했습니다.",
                "공개된 채용 계획의 모집 직급 칸에는 {v}{subject} 적혀 있습니다.",
                "지원자들은 이번에 공지된 {v} 모집 전형에 지원할 수 있습니다.",
                "신규 채용 인력의 예정 직위는 {v}이며, 담당 업무는 공고에 안내되어 있습니다.",
                "채용 담당자는 이번 모집 직위를 {v}{as} 설명했습니다.",
                "인력 모집 계획 | 채용 예정 직위 | {v}.",
                "경력 지원자를 위한 공고이며, 채용할 직위는 {v}입니다.",
            ),
        },
        "DEPARTMENT": {
            "private": (
                "지원자가 채용 업무를 맡았던 당시의 소속은 {v}{past}.",
                "공고 문의와 별개로, 지원자의 경력기술서에는 {v} 근무 이력이 적혀 있습니다.",
                "지원자는 이전 회사 {v}에서 접수 서류를 관리했다고 설명했습니다.",
                "제출된 이력서의 소속 부서 항목은 {v}입니다.",
                "지원자가 본인의 마지막 근무 부서를 {v}{as} 적었습니다.",
                "개별 경력 확인 자료 | 근무 부서 | {v}.",
                "채용 관련 경험을 소개하는 지원자는 {v} 소속으로 근무했습니다.",
                "지원서에 첨부된 재직 확인서에는 당시 부서가 {v}{reported}고 나옵니다.",
            ),
            "public": (
                "지원자의 이전 소속과 별도로, 이번 전형의 접수 담당 부서는 {v}입니다.",
                "공고에 따르면 {v}에서 제출 서류에 관한 문의를 받습니다.",
                "공개 모집의 지원 절차는 기관의 {v}에서 안내합니다.",
                "채용 안내의 담당 부서 칸에는 {v}{subject} 적혀 있습니다.",
                "기관은 이번 모집의 문의 창구를 {v}{as} 지정했습니다.",
                "공개 전형 안내 | 문의 담당 부서 | {v}.",
                "새로 지원하는 사람은 모집 일정에 관한 질문을 {v}에 전달할 수 있습니다.",
                "채용 담당자는 {v}에서 서류 접수 일정을 안내한다고 공지했습니다.",
            ),
        },
    },
    "valid": {
        "POSITION": {
            "private": (
                "구직자가 전 직장 경력을 정리한 표에서 재직 직급은 {v}{as} 확인됩니다.",
                "채용 관련 업무를 수행한 구직자는 본인의 직위를 {v}{as} 기입했습니다.",
                "공개 모집을 문의한 구직자의 이력서에는 {v} 경력이 포함되어 있습니다.",
                "제출한 경력 확인 자료에서 구직자의 직위는 {v}{past}.",
                "구직자가 직접 소개한 자신의 이전 직함은 {v}입니다.",
                "본인 재직 이력의 직위 항목: {v}.",
                "구직자는 전형 안내 경험을 소개하면서 당시 {v}{reported}고 답했습니다.",
                "이직을 준비하는 구직자가 맡고 있는 직급은 {v}입니다.",
            ),
            "public": (
                "공개된 인력 충원 계획은 {v} 직급의 인원을 모집하는 내용입니다.",
                "기관의 이번 채용 예정 직급이 {v}{as} 발표되었습니다.",
                "구직자에게 배포된 모집 자료의 직위 항목은 {v}입니다.",
                "모집 기관은 {v} 전형의 접수 기간을 알렸습니다.",
                "이번에 뽑을 인력의 직함은 {v}{as} 정해져 있습니다.",
                "전형별 모집 내역의 예정 직급: {v}.",
                "기관은 {v} 공개 모집을 진행하며, 지원 자격을 함께 제시했습니다.",
                "채용 설명회의 직위 안내에 {v}{subject} 포함되어 있습니다.",
            ),
        },
        "DEPARTMENT": {
            "private": (
                "구직자는 전 직장의 {v}에서 전형 업무를 수행한 경험을 소개했습니다.",
                "본인 경력 내역에서 확인되는 소속 부서명은 {v}입니다.",
                "모집 절차를 문의한 구직자의 이전 부서는 {v}{past}.",
                "재직 확인 자료는 구직자가 {v}에서 일했다고 설명합니다.",
                "구직자가 옮기기 전 소속된 조직은 {v}입니다.",
                "본인 재직 이력의 부서 항목: {v}.",
                "입사 전형을 운영한 경력을 적은 구직자는 {v} 소속이었습니다.",
                "구직자는 당시의 업무 분담을 설명하며 {v} 근무 사실을 적었습니다.",
            ),
            "public": (
                "이번 전형의 지원 방법은 기관 내 {v}에서 설명합니다.",
                "배포된 모집 자료는 접수 담당 조직을 {v}{as} 명시했습니다.",
                "공개 모집에 관한 일정 변경은 {v}에서 안내할 예정입니다.",
                "기관의 채용 창구를 맡은 부서명은 {v}입니다.",
                "모집 절차를 묻는 구직자는 {v}에 문의하면 됩니다.",
                "전형별 지원 안내의 담당 조직: {v}.",
                "공개된 자료에는 서류 제출 문의를 {v}에서 받는다고 적혀 있습니다.",
                "기관은 이번 채용 업무를 {v}에 배정했다고 알렸습니다.",
            ),
        },
    },
    "test": {
        "POSITION": {
            "private": (
                "응시자의 과거 이력을 검토하니 전 직장에서 맡은 직함은 {v}{past}.",
                "응시자는 모집 분야와 연관된 자신의 {v} 재직 경험을 제출했습니다.",
                "기관 공지를 확인한 응시자는 전 직장 직급이 {v}{reported}고 적었습니다.",
                "첨부된 재직 기록의 직급란에는 {v}{subject} 표시되어 있습니다.",
                "응시자가 경력 상담에서 밝힌 당시 직위는 {v}입니다.",
                "응시 이력 요약 / 본인이 맡았던 직함 / {v}.",
                "이전 사업장에서 전형 운영을 맡은 응시자의 직급은 {v}{past}.",
                "응시자는 {v}{as} 근무하면서 수행했던 업무를 자기소개에 담았습니다.",
            ),
            "public": (
                "이번 인원 모집 안내는 {v} 채용에 필요한 내용을 담고 있습니다.",
                "모집 기관이 공지한 신규 인력의 직위는 {v}입니다.",
                "응시 안내에 제시된 채용 직급은 {v}{as} 정해졌습니다.",
                "공개 전형에서 선발할 인력의 직함이 {v}{as} 안내되었습니다.",
                "기관이 배포한 직위별 모집 표에는 {v}{subject} 실려 있습니다.",
                "응시 안내 요약 / 이번에 모집할 직함 / {v}.",
                "기관은 {v} 모집에 응시할 사람을 위해 설명회를 공지했습니다.",
                "이번 채용 공지의 직급별 안내 중 모집 직위는 {v}입니다.",
            ),
        },
        "DEPARTMENT": {
            "private": (
                "응시자가 재직 기록에 밝힌 과거 소속은 {v}입니다.",
                "경력 서술에서 응시자는 {v}의 전형 운영 업무를 맡았다고 했습니다.",
                "기관 공지를 읽은 응시자의 직전 근무 부서는 {v}{past}.",
                "응시자가 낸 경력 자료는 {v} 재직 기간을 포함합니다.",
                "응시자는 자신의 업무 경험과 함께 소속 부서 {v}{obj} 적었습니다.",
                "응시 이력 요약 / 본인의 이전 소속 / {v}.",
                "응시자가 모집 절차를 안내하던 때의 소속 부서는 {v}입니다.",
                "과거 직장 이력을 작성한 응시자는 {v} 근무 사실을 밝혔습니다.",
            ),
            "public": (
                "배포된 응시 안내에서 접수를 맡은 부서는 {v}입니다.",
                "기관은 이번 전형의 질문을 {v}에서 확인한다고 공지했습니다.",
                "모집 공지에 제시된 문의 담당 조직명은 {v}입니다.",
                "응시 절차의 변경 사항은 기관 내 {v}에서 알립니다.",
                "공개 전형의 서류 관련 문의는 {v}에 전달하도록 안내됐습니다.",
                "응시 안내 요약 / 이번 전형 담당 조직 / {v}.",
                "기관의 {v}에서는 새로 지원하는 사람에게 접수 방법을 설명합니다.",
                "이번 모집 자료를 발송하는 담당 부서로 {v}{subject} 안내되어 있습니다.",
            ),
        },
    },
}


class _Inventory:
    def __init__(self, rng):
        self.rng = rng
        self.used = defaultdict(set)
        self.codes = {}
        self.cursor = Counter()
        for split, bounds in zip(SPLITS, ((1, 40000), (40000, 70000), (70000, 100000))):
            values = list(range(*bounds))
            rng.shuffle(values)
            for private, codes in ((True, values[::2]), (False, values[1::2])):
                self.codes[split, private] = codes

    def contact(self, kind, split, private):
        key = kind, split, private
        while True:
            index = self.cursor[key]
            self.cursor[key] += 1
            code = self.codes[split, private][index]
            if kind == "PHONE":
                value = f"000-{code // 10000:04d}-{code % 10000:04d}"
            elif kind == "EMAIL":
                value = f"desk{code}@{self.rng.choice(('mailbox','letter','inbox'))}.example"
            else:
                city = self.rng.choice(LEXICONS[split]["cities"])
                district = self.rng.choice(("솔빛구", "느티구", "들꽃구", "나루구", "은행구", "도담구"))
                road = self.rng.choice(("온새로", "솔나루길", "여울빛로", "해든길", "도래로", "새결길"))
                value = f"{city} {district} {road} {code % 997 + 1}"
                if self.rng.random() < 0.35:
                    floor = code % 20 + 1
                    value += f" {floor}층 {floor * 100 + code % 20 + 1}호"
            if value not in self.used[kind]:
                self.used[kind].add(value)
                return value

    def entity(self, kind, split):
        lexicon = LEXICONS[split]
        if kind in {"POSITION", "DEPARTMENT"}:
            return self.rng.choice(lexicon[kind])
        stem = self.rng.choice(lexicon["stems"])
        if kind == "WORKPLACE":
            return stem + self.rng.choice(("테크", "시스템즈", "연구소", "솔루션", "네트웍스"))
        if kind == "SCHOOL":
            return stem + self.rng.choice(("대학교", "과학대학", "고등학교", "정보대학"))
        raise ValueError("Unsupported entity inventory")


def build_proposals(*, seed=DEFAULT_SEED):
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    # Relation/quota allocation only; no V3 document text or value is retained.
    allocated, plan = _allocate(seed=seed)
    rng = random.Random(seed + 404)
    values = _Inventory(rng)
    plan = deepcopy(plan)
    plan["generator_version"] = GENERATOR_VERSION
    plan["holdout_status"] = "fresh_unreviewed_synthetic_holdout_created_after_v3_error_analysis"
    plan["human_reviewed"] = False
    plan["value_split_policy"] = "all_entity_values_disjoint_across_train_valid_test"
    plan["template_policy"] = "split_specific_clause_banks; new_POSITION_DEPARTMENT_clauses; other_banks_inherit_v3"
    plan["family_allocation"] = {s: [f"v4-{s}-family-{i:02d}" for i in range(1, 11)] for s in SPLITS}
    documents = []
    for original, record in zip(allocated, plan["documents"]):
        split = original.split
        doc_id = original.doc_id.replace("syn-v3-", "syn-v4-", 1)
        family = original.source_group.replace("v3-", "v4-", 1)
        rows = []
        for mention, relation in zip(original.mentions, record["relations"]):
            rows.append({"kind": mention.entity_type, "relation": relation["relation"],
                         "subrole": relation["quota_subrole"], "action": mention.action,
                         "approved": relation["simulated_approved_value"]})
        approved = {kind: values.entity(kind, split) for kind in sorted(
            {r["kind"] for r in rows if r["kind"] not in CONTACT_TYPES | {"NAME"}})}
        context = {"document_type": "general_notice", "approved_public_contacts": [],
                   "approved_public_orgs": [], "approved_public_positions": []}
        for row in rows:
            kind = row["kind"]
            if kind == "NAME":
                row["value"] = rng.choice(LEXICONS[split]["names"])
            elif kind in CONTACT_TYPES:
                row["value"] = values.contact(kind, split, row["relation"] == "private")
            elif row["approved"]:
                row["value"] = approved[kind]
            else:
                value = values.entity(kind, split)
                while value == approved[kind]:
                    value = values.entity(kind, split)
                row["value"] = value
            if row["approved"]:
                context[_approval_key(kind)].append(row["value"])
        for key in ("approved_public_contacts", "approved_public_orgs", "approved_public_positions"):
            context[key] = sorted(set(context[key]))
        # Shuffle where personal vs public occurrences occur; ordering carries
        # no label. Sources are independent to avoid contradictory biographies.
        rng.shuffle(rows)
        family_index = int(family.rsplit("-", 1)[1]) - 1
        parts = [FAMILY_WRAPPERS[split][family_index] + "\n",
                 "각 항목은 별개의 제출 자료 또는 기관 안내에서 발췌한 내용입니다.\n"]
        mentions, relations, clauses = [], [], []
        cursor = sum(map(len, parts))
        used = set()
        for index, row in enumerate(rows, 1):
            kind, value = row["kind"], row["value"]
            role = "private" if row["relation"] == "private" else "public"
            if kind == "NAME":
                template, clause_id = "제출 자료의 작성자는 {v}입니다.", f"v4.{split}.name.identity"
            else:
                bank = FOCUS_CLAUSES[split][kind][role] if kind in FOCUS_CLAUSES[split] else CLAUSE_BANKS[split][kind][role]
                variant = rng.randrange(len(bank))
                for _ in range(len(bank)):
                    clause_id = f"v4.{split}.{kind.lower()}.{role}.{variant + 1}"
                    if clause_id not in used:
                        break
                    variant = (variant + 1) % len(bank)
                else:
                    raise ValueError("Insufficient distinct templates inside document")
                used.add(clause_id)
                template = bank[variant]
            for key, particle in _particles(value).items():
                template = template.replace("{" + key + "}", particle)
            before, after = template.split("{v}")
            before = f"항목 {index}: " + before
            parts.append(before)
            cursor += len(before)
            mid, start = f"m{index:02d}", cursor
            parts.append(value)
            cursor += len(value)
            reason = "POLICY_ALWAYS_MASK_NAME" if kind == "NAME" else "PERSONAL_USE_CONTEXT" if row["relation"] == "private" else "PUBLIC_ROLE_APPROVED_VALUE" if row["approved"] else "PUBLIC_LOOKING_UNAPPROVED_VALUE"
            mentions.append(Mention(mid, start, cursor, value, kind, row["action"], reason))
            parts.append(after + "\n")
            cursor += len(after) + 1
            relations.append({"mention_id": mid, "relation": row["relation"], "quota_subrole": row["subrole"],
                              "proposed_action": row["action"], "simulated_approved_value": row["approved"],
                              "clause_id": clause_id, "source_item_id": f"{doc_id}:item-{index:02d}"})
            clauses.append({"clause_id": clause_id, "relation": row["relation"], "mention_id": mid})
        document = Document(doc_id, "".join(parts), mentions, family, ["NAME", *TYPES],
                            source=GENERATOR_VERSION, split=split, policy_id=FIXED_POLICY_ID,
                            trusted_context=context)
        documents.append(document)
        record.update({"doc_id": doc_id, "family_id": family,
                       "persona_id": f"fictional-v4-{seed}-{split}-{len(documents):04d}",
                       "relations": relations, "clauses": clauses,
                       "clause_ids": [c["clause_id"] for c in clauses],
                       "source_item_ids": [r["source_item_id"] for r in relations]})
    plan["limitations"] = [
        "Unreviewed fictional rule labels; review records remain PENDING.",
        "Created after V3 error analysis; the new test is not independent confirmatory real-world evidence.",
        "Position and department clauses are new; other split-specific clause banks inherit V3.",
        "Cities, organizations and people are fictional compositions and may coincidentally resemble real names.",
        "The public approval registry is simulated; membership alone never authorizes a personal occurrence.",
        "Full value and clause separation prevents exact reuse, not semantic paraphrase leakage.",
    ]
    validate_split_separation({s: [d for d in documents if d.split == s] for s in SPLITS})
    return documents, plan


def audit_v4(documents, plan):
    """Recheck legacy quotas and new value/role contracts without ML imports."""
    docs = list(documents)
    compatible = deepcopy(plan)
    compatible["generator_version"] = "synthetic_rule_based_v3_bulk"
    report = audit_bulk(docs, compatible)
    report["audit_version"] = "synthetic_bulk_audit_v4"
    blockers = report["blockers"]
    if plan.get("generator_version") != GENERATOR_VERSION:
        blockers.append({"code": "wrong_v4_generator_version", "doc_ids": []})
    owners = defaultdict(set)
    pair_counts = Counter()
    for doc in docs:
        local = defaultdict(set)
        if doc.source != GENERATOR_VERSION:
            blockers.append({"code": "wrong_v4_document_source", "doc_ids": [doc.doc_id]})
        for mention in doc.mentions:
            key = mention.entity_type, mention.text.casefold()
            owners[key].add(doc.split)
            local[key].add(mention.action)
            if "가상" in mention.text:
                blockers.append({"code": "literal_fictional_entity_prefix", "doc_ids": [doc.doc_id]})
            if mention.entity_type == "PHONE" and not re.fullmatch(r"000-\d{4}-\d{4}", mention.text):
                blockers.append({"code": "unsafe_phone_inventory", "doc_ids": [doc.doc_id]})
            if mention.entity_type == "EMAIL" and not mention.text.endswith(".example"):
                blockers.append({"code": "unsafe_email_inventory", "doc_ids": [doc.doc_id]})
            if mention.action == "KEEP" and not allowed_to_keep(doc, mention):
                blockers.append({"code": "v4_keep_approval_mismatch", "doc_ids": [doc.doc_id]})
        for (kind, _), actions in local.items():
            if kind not in CONTACT_TYPES and actions == {"MASK", "KEEP"}:
                pair_counts[doc.split, kind] += 1
        if any(token in doc.text for token in ("MASK", "KEEP", "정답", "개인정보이므로", "개인정보므로")):
            blockers.append({"code": "label_disclosure_in_document_text", "doc_ids": [doc.doc_id]})
    shared = [(kind, value) for (kind, value), splits in owners.items() if len(splits) > 1]
    if shared:
        blockers.append({"code": "entity_value_crosses_splits", "doc_ids": [], "count": len(shared)})
    report.update({"passed": not blockers, "human_reviewed": False,
                   "holdout_status": plan.get("holdout_status"),
                   "cross_split_entity_value_overlap_count": len(shared),
                   "same_value_mask_keep_pairs": {s: {k: pair_counts[s, k] for k in ("POSITION", "DEPARTMENT", "WORKPLACE", "SCHOOL")} for s in SPLITS},
                   "naturalness_certification": "not_established_by_this_automated_audit"})
    return report


def generate(output_dir, *, seed=DEFAULT_SEED, a_model_dir=None, expected_a_manifest_sha256=None):
    destination = Path(output_dir).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("Output must be a fresh version directory; existing data is never overwritten")
    docs, plan = build_proposals(seed=seed)
    audit = audit_v4(docs, plan)
    if audit["blockers"] or audit["warnings"]:
        raise ValueError("V4 audit failed: " + json.dumps({k: audit[k] for k in ("blockers", "warnings")}, ensure_ascii=False))
    lock = None
    if a_model_dir is not None:
        from .synthetic_flow import create_a_lock
        lock = create_a_lock(a_model_dir, expected_a_manifest_sha256)
        if destination.resolve().is_relative_to(Path(a_model_dir).resolve()):
            raise ValueError("Output must be outside fixed A checkpoint")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        write_jsonl(staging / "proposed_documents.jsonl", docs)
        for split in SPLITS:
            write_jsonl(staging / "proposals" / f"{split}.jsonl", [d for d in docs if d.split == split])
        _json(staging / "generation_plan.json", plan)
        _json(staging / "automated_audit.json", audit)
        export_review(docs, staging / "review", policy_id=FIXED_POLICY_ID)
        # A split-balanced sample is for human reading only, never approval.
        selected = []
        for split, size in (("train", 20), ("valid", 15), ("test", 15)):
            partition = [d for d in docs if d.split == split]
            sample_rng = random.Random(seed + 700 + SPLITS.index(split))
            selected.extend(sample_rng.sample(partition, size))
        export_review(selected, staging / "review_sample_50", policy_id=FIXED_POLICY_ID)
        _write_preview(staging, selected)
        for name in ("review.html", "review.md"):
            preview = staging / name
            content = preview.read_text(encoding="utf-8")
            content = content.replace("review/mentions.csv", "review_sample_50/mentions.csv")
            content = content.replace("규칙 기반 파일럿", "V4 규칙 기반 검수 표본")
            preview.write_text(content, encoding="utf-8")
        _json(staging / "review_sample_50" / "selection.json", {
            "documents": 50, "split_counts": {"train": 20, "valid": 15, "test": 15},
            "doc_ids": [d.doc_id for d in selected], "review_status": "PENDING", "human_reviewed": False})
        if lock is not None:
            _json(staging / "a_checkpoint.lock.json", lock)
        for path in staging.rglob("*"):
            if path.is_file():
                path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
        manifest = {"generator_version": GENERATOR_VERSION, "seed": seed,
                    "policy_id": FIXED_POLICY_ID, "policy_sha256": policy_sha256(),
                    "generator_sources_sha256": {name: _sha(Path(__file__).with_name(name)) for name in
                        ("synthetic_v4.py", "synthetic_bulk.py", "synthetic_bulk_audit.py", "synthetic_v3_templates.py", "synthetic_audit.py")},
                    "documents": len(docs), "split_counts": dict(COUNTS),
                    "mentions": sum(len(d.mentions) for d in docs),
                    "contextual_candidates": sum(m.entity_type != "NAME" for d in docs for m in d.mentions),
                    "proposed_action_counts": dict(Counter(m.action for d in docs for m in d.mentions)),
                    "quota_targets": plan["quota_targets"], "mode_targets": plan["mode_targets"],
                    "human_reviewed": False, "review_status": "PENDING", "training_executed": False,
                    "holdout_status": plan["holdout_status"], "limitations": plan["limitations"],
                    "metadata_provenance": "SIMULATED fictional approval registry, not externally verified",
                    "files_sha256": {p.relative_to(staging).as_posix(): _sha(p) for p in sorted(staging.rglob("*")) if p.is_file()}}
        _json(staging / "generation_manifest.json", manifest)
        if destination.exists():
            raise ValueError("Destination appeared during generation")
        staging.rename(destination)
    finally:
        if staging.exists():
            assert staging.resolve().parent == destination.parent.resolve()
            assert staging.name.startswith(f".{destination.name}.staging-")
            shutil.rmtree(staging)
    return {"output_dir": str(destination), "generator_version": GENERATOR_VERSION,
            "documents": len(docs), "mentions": manifest["mentions"],
            "contextual_candidates": manifest["contextual_candidates"], "review_sample_documents": 50,
            "review_status": "PENDING", "human_reviewed": False, "training_executed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--a-model-dir")
    parser.add_argument("--expected-a-manifest-sha256")
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.output_dir, seed=args.seed, a_model_dir=args.a_model_dir,
                              expected_a_manifest_sha256=args.expected_a_manifest_sha256), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
