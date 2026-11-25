"""Deterministic fictional B proposals and PENDING human-review packets.

No model, network service, training loop, or automatic review approval is used.
Template families are allocated to splits before relation plans are rendered.
This controlled pilot is not evidence of real-document performance.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .b_data import export_review, validate_split_separation
from .b_inputs import CONTEXTUAL_TYPES, FIXED_POLICY_ID, allowed_to_keep, policy_sha256
from .data import write_jsonl
from .schema import Document, Mention

GENERATOR_VERSION = "synthetic_rule_based_v1"
TYPES = ("PHONE", "EMAIL", "ADDRESS", "WORKPLACE", "SCHOOL", "DEPARTMENT", "POSITION")
SPLITS = ("train", "valid", "test")
NAMES = ("김가람", "이가온", "박나래", "최다온", "정새봄", "윤하람", "강아름", "임누리")

# Shared relation semantics are intentional; these 15 discourse templates are
# separate families, not a claim of 15 independent natural-language domains.
TEMPLATES = (
    "지원자 {name}의 전달 요청입니다.\n{private}\n함께 전달할 안내: {public}\n{unapproved}",
    "대외 공유 전 확인 메모\n{public}\n작성자 {name}가 적은 이력: {private}\n{unapproved}",
    "{name}의 지원서와 안내문을 함께 검토한다.\n지원서 부분: {private}\n안내문 부분: {unapproved}\n별도 안내: {public}",
    "수신 자료 정리\n{unapproved}\n{public}\n자료 작성자는 {name}이다. {private}",
    "게시 요청자는 {name}이다.\n{private}\n다음 두 안내를 덧붙였다.\n{unapproved}\n{public}",
    "인수인계 기록\n{public}\n{private}\n회신 작성: {name}\n{unapproved}",
    "외부 전달 초안의 참고 사항이다.\n{unapproved}\n{private}\n이를 보낸 {name}의 요청으로 다음 안내도 싣는다. {public}",
    "{name}가 두 자료를 묶어서 보냈다.\n첫 자료에는 다음 내용이 있다. {public}\n개인 설명에는 다음 내용이 있다. {private}\n추가 자료: {unapproved}",
    "문서 편집 검토\n{private}\n공고 쪽에는 다음 내용이 있다. {public}\n덧붙인 안내: {unapproved}\n편집 요청자: {name}",
    "전달 내용 확인\n{unapproved}\n보낸 사람 {name}의 설명은 다음과 같다. {private}\n함께 받은 게시문에는 다음 내용이 있다. {public}",
    "문서 공유 요청\n요청자 {name}\n본문: {private}\n첨부 안내 가: {public}\n첨부 안내 나: {unapproved}",
    "회신을 위해 내용을 정리했다.\n{public}\n{unapproved}\n{private}\n이 내용을 작성한 사람은 {name}이다.",
    "{name}가 제출한 자료를 옮긴다.\n공지에 실린 내용: {unapproved}\n본인의 설명: {private}\n다른 공지의 내용: {public}",
    "자료 대조 메모\n{private}\n다른 문서에서 옮긴 문장: {public}\n추가로 옮긴 문장: {unapproved}\n대조 요청: {name}",
    "게시할 문안을 모았다.\n{unapproved}\n게시문 다음에는 {name}의 설명이 이어진다. {private}\n마지막 안내는 다음과 같다. {public}",
)

# Only one span-bearing value occurs in each clause; offsets are accumulated
# while rendering, never found by str.find on repeated surface forms.
CLAUSES = {
    "PHONE": ("지원서에 적은 제 개인 연락처는 ", "입니다.", "채용 문의 전화는 ", "입니다."),
    "EMAIL": ("제 지원서의 개인 이메일은 ", "입니다.", "기관 채용 문의 이메일은 ", "입니다."),
    "ADDRESS": ("제 이력서에 적은 거주지는 ", "입니다.", "기관 방문 접수 장소는 ", "입니다."),
    "WORKPLACE": ("저는 ", "에서 근무한 경력을 적었습니다.", "이번 채용공고의 모집 기관은 ", "입니다."),
    "SCHOOL": ("저는 ", "를 졸업한 학력을 적었습니다.", "입학처 운영시간을 안내하는 기관은 ", "입니다."),
    "DEPARTMENT": ("제 이전 직장의 소속 부서는 ", "이었습니다.", "공고에서 지원서를 접수하는 부서는 ", "입니다."),
    "POSITION": ("제 경력기술서의 당시 직위는 ", "였습니다.", "이번 공고에서 공개 모집하는 직위는 ", "입니다."),
}
REASONS = {
    "PHONE": "PERSON_CONTACT", "EMAIL": "PERSON_CONTACT", "ADDRESS": "PERSON_ADDRESS",
    "WORKPLACE": "PERSON_EMPLOYMENT", "SCHOOL": "PERSON_EDUCATION",
    "DEPARTMENT": "PERSON_EMPLOYMENT", "POSITION": "PERSON_EMPLOYMENT",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def _value(kind: str, code: int, approved: bool) -> str:
    stem = "한빛" if approved else "새솔"
    if kind == "PHONE":
        return f"000-0000-{code:04d}"
    if kind == "EMAIL":
        return f"recruit{code}@{('hanbit' if approved else 'saesol')}.example"
    if kind == "ADDRESS":
        return f"가상시 {stem}구 다온로 {code}"
    if kind == "WORKPLACE":
        return f"가상{stem}제{code}테크"
    if kind == "SCHOOL":
        return f"가상{stem}제{code}대학교"
    if kind == "DEPARTMENT":
        return f"가상{stem}제{code}지원팀"
    return f"가상{stem}제{code}운영책임자"


def build_proposals(*, seed: int = 20261019, train_count: int = 30,
                    valid_count: int = 10, test_count: int = 10
                    ) -> tuple[list[Document], dict[str, Any]]:
    """Build independent canonical proposal objects and an explicit relation plan."""
    counts = dict(zip(SPLITS, (train_count, valid_count, test_count)))
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if any(type(count) is not int or count < len(TYPES) for count in counts.values()):
        raise ValueError("each split requires at least seven documents for contextual-type coverage")
    if sum(counts.values()) > 4000:
        raise ValueError("pilot generator is limited to 4000 documents; revise the scenario inventory before scaling")
    rng = random.Random(seed)
    family_order = list(range(len(TEMPLATES)))
    rng.shuffle(family_order)
    allocation = {split: family_order[index * 5:(index + 1) * 5]
                  for index, split in enumerate(SPLITS)}
    docs: list[Document] = []
    plans = []
    for split in SPLITS:
        type_order = list(TYPES)
        rng.shuffle(type_order)
        for index in range(counts[split]):
            serial = len(docs) + 1
            family = allocation[split][index % 5]
            kind = type_order[index % len(type_order)]
            primary = _value(kind, serial * 2, True)
            unapproved = _value(kind, serial * 2 + 1, False)
            name = NAMES[rng.randrange(len(NAMES))]
            doc_id = f"syn-{seed}-{split}-{index + 1:04d}"
            text_parts: list[str] = []
            mentions: list[Mention] = []
            cursor = 0

            def add_text(value: str) -> None:
                nonlocal cursor
                text_parts.append(value)
                cursor += len(value)

            def add_span(value: str, entity_type: str, action: str, reason: str) -> None:
                start = cursor
                add_text(value)
                mentions.append(Mention(f"m{len(mentions) + 1:02d}", start, cursor,
                                        value, entity_type, action, reason))

            template = TEMPLATES[family]
            # Korean subject particles depend on the final syllable's consonant.
            if (ord(name[-1]) - ord("가")) % 28:
                template = template.replace("{name}가", "{name}이")
            previous = 0
            relations = []
            for field in re.finditer(r"\{(name|private|public|unapproved)\}", template):
                add_text(template[previous:field.start()])
                role = field.group(1)
                if role == "name":
                    add_span(name, "NAME", "MASK", "POLICY_ALWAYS_MASK_NAME")
                else:
                    left, right = CLAUSES[kind][:2] if role == "private" else CLAUSES[kind][2:]
                    add_text(left)
                    value = primary if role != "unapproved" else unapproved
                    action = "KEEP" if role == "public" else "MASK"
                    reason = (REASONS[kind] if role == "private" else
                              "PUBLIC_ROLE_APPROVED_VALUE" if role == "public" else
                              "PUBLIC_LOOKING_UNAPPROVED_VALUE")
                    add_span(value, kind, action, reason)
                    relations.append({"mention_id": mentions[-1].mention_id,
                                      "relation": role, "proposed_action": action,
                                      "simulated_approved_value": role != "unapproved"})
                    add_text(right)
                previous = field.end()
            add_text(template[previous:])
            key = ("approved_public_contacts" if kind in {"PHONE", "EMAIL", "ADDRESS"}
                   else "approved_public_positions" if kind == "POSITION" else "approved_public_orgs")
            metadata = {"document_type": ("resume", "job_posting", "general_notice")[family % 3],
                        "approved_public_contacts": [], "approved_public_orgs": [],
                        "approved_public_positions": []}
            metadata[key] = [primary]
            doc = Document(doc_id, "".join(text_parts), mentions,
                           source_group=f"synthetic-family-{family + 1:02d}",
                           annotation_scope=["NAME", *TYPES], source=GENERATOR_VERSION,
                           split=split, policy_id=FIXED_POLICY_ID, trusted_context=metadata)
            if any(m.action == "KEEP" and not allowed_to_keep(doc, m) for m in mentions):
                raise ValueError("generator produced a policy-incompatible KEEP proposal")
            docs.append(doc)
            plans.append({"doc_id": doc_id, "split": split, "family_id": family + 1,
                          "persona_id": f"fictional-persona-{seed}-{split}-{index + 1:04d}",
                          "focus_type": kind, "fictional_values": {
                              "name": name, "private_and_public": primary,
                              "unapproved_public": unapproved},
                          "relations": relations})
    validate_split_separation({split: [d for d in docs if d.split == split] for split in SPLITS})
    plan = {"generator_version": GENERATOR_VERSION, "seed": seed,
            "split_counts": counts, "family_allocation": {
                split: [i + 1 for i in allocation[split]] for split in SPLITS},
            "documents": plans,
            "limitations": [
                "Controlled template proposals; no external LLM and no human approval.",
                "Approval-list facts are simulated for fictional values, not externally verified facts.",
                "Discourse-template families do not cross splits; base relation clauses are shared.",
                "Synthetic test results alone cannot establish real-document performance.",
                "Each document focuses on one contextual type; expand mixed-type scenarios after review."]}
    return docs, plan


def _marked_html(doc: Document) -> str:
    fragments, previous = [], 0
    for mention in sorted(doc.mentions, key=lambda m: m.start):
        fragments.append(html.escape(doc.text[previous:mention.start]))
        title = f"{mention.mention_id} | {mention.entity_type} | {mention.reason_code} | [{mention.start},{mention.end})"
        fragments.append(f'<mark class="{mention.action.lower()}" title="{html.escape(title, quote=True)}">'
                         f'{html.escape(mention.text)}<small> {mention.action}</small></mark>')
        previous = mention.end
    fragments.append(html.escape(doc.text[previous:]))
    return "".join(fragments)


def _write_preview(root: Path, docs: list[Document]) -> None:
    sections = []
    for doc in docs:
        rows = "".join(f"<tr><td>{html.escape(m.mention_id)}</td><td>{m.start}:{m.end}</td>"
                       f"<td>{html.escape(m.text)}</td><td>{m.entity_type}</td>"
                       f"<td>{m.action}</td><td>{html.escape(m.reason_code or '')}</td></tr>"
                       for m in doc.mentions)
        sections.append(f"<section><h2>{html.escape(doc.doc_id)} — PENDING</h2>"
                        f"<p>{doc.split} / {doc.source_group}</p><pre>{_marked_html(doc)}</pre>"
                        f"<details><summary>시뮬레이션 승인 목록과 위치</summary><pre>"
                        f"{html.escape(json.dumps(doc.trusted_context, ensure_ascii=False, indent=2))}</pre>"
                        f"<table><tr><th>ID</th><th>[start:end)</th><th>값</th><th>유형</th>"
                        f"<th>제안</th><th>사유</th></tr>{rows}</table></details></section>")
    page = """<!doctype html><html lang="ko"><meta charset="utf-8"><title>B 합성 문서 검수</title>
<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:30px auto;padding:0 18px;line-height:1.7}
section{border:1px solid #ccd3dd;border-radius:8px;padding:18px;margin:18px 0}pre{white-space:pre-wrap;font:inherit}
mark{padding:2px 3px;border-radius:3px}.mask{background:#ffdddd}.keep{background:#d4f5dd}small{font-size:70%;font-weight:bold}
td,th{border:1px solid #ccd3dd;padding:4px 8px;text-align:left}table{border-collapse:collapse}</style>
<h1>B 합성 문서 검수 — 모든 문서 PENDING</h1>
<p>가상 인물·기관·주소, 000 전화번호와 .example 이메일을 사용하는 규칙 기반 파일럿입니다.
색상은 생성기가 제안한 정답이며 검수 완료를 뜻하지 않습니다. 승인 목록 역시 가상 상황을 위한 시뮬레이션입니다.
실제 승인 근거 또는 실제 업무 문서가 아닙니다. 학습을 실행하지 않았습니다.</p>
<p>원문·위치·유형·제안 라벨을 확인하고 review/mentions.csv를 수정하세요.
추가 후보와 누락 후보도 확인하세요. 수정 후 새 지문을 준비하고 사람이 검수 완료 기록을 남겨야 학습할 수 있습니다.</p>
""" + "\n".join(sections) + "</html>\n"
    (root / "review.html").write_text(page, encoding="utf-8")
    lines = ["# B 합성 문서 검수 — PENDING", "", "모든 라벨은 검수 전 제안입니다. HTML은 review.html, 수정 양식은 review/mentions.csv입니다.",
             "승인 목록은 가상 시나리오의 시뮬레이션이며 실제 외부 검증이나 검수 승인이 아닙니다.", ""]
    for doc in docs:
        lines += [f"## {doc.doc_id} ({doc.split}, {doc.source_group})", "", "```text", doc.text, "```", "",
                  "| ID | [start,end) | 유형 | 값 | 제안 |", "|---|---|---|---|---|"]
        lines += [f"| {m.mention_id} | [{m.start},{m.end}) | {m.entity_type} | {m.text} | {m.action} |"
                  for m in doc.mentions]
        lines.append("")
    (root / "review.md").write_text("\n".join(lines), encoding="utf-8")


def generate(output_dir: str | Path, *, seed: int = 20261019, train_count: int = 30,
             valid_count: int = 10, test_count: int = 10) -> dict[str, Any]:
    """Publish a complete packet, rejecting an existing nonempty destination."""
    destination = Path(output_dir).absolute()
    if destination.is_symlink():
        raise ValueError("output directory must not be a symbolic link")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("output directory exists and is nonempty; choose a new directory")
    docs, plan = build_proposals(seed=seed, train_count=train_count,
                                valid_count=valid_count, test_count=test_count)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        write_jsonl(temporary / "proposed_documents.jsonl", docs)
        for split in SPLITS:
            write_jsonl(temporary / "proposals" / f"{split}.jsonl", (d for d in docs if d.split == split))
        _dump_json(temporary / "generation_plan.json", plan)
        export_review(docs, temporary / "review", policy_id=FIXED_POLICY_ID)
        _write_preview(temporary, docs)
        labels = Counter(m.action for doc in docs for m in doc.mentions)
        files = {path.relative_to(temporary).as_posix(): _sha(path)
                 for path in sorted(temporary.rglob("*")) if path.is_file()}
        manifest = {"generator_version": GENERATOR_VERSION, "seed": seed,
                    "policy_id": FIXED_POLICY_ID, "policy_sha256": policy_sha256(),
                    "generator_code_sha256": _sha(Path(__file__)),
                    "templates_sha256": hashlib.sha256(json.dumps(
                        {"templates": TEMPLATES, "clauses": CLAUSES}, ensure_ascii=False,
                        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
                    "documents": len(docs), "mentions": sum(len(d.mentions) for d in docs),
                    "proposed_action_counts": dict(labels), "split_counts": plan["split_counts"],
                    "contextual_type_counts": {split: dict(Counter(m.entity_type for d in docs if d.split == split
                        for m in d.mentions if m.entity_type in CONTEXTUAL_TYPES)) for split in SPLITS},
                    "review_status": "PENDING", "training_executed": False,
                    "metadata_provenance": "SIMULATED fictional policy facts; no external verification",
                    "limitations": plan["limitations"], "files_sha256": files}
        _dump_json(temporary / "generation_manifest.json", manifest)
        if destination.is_symlink():
            raise ValueError("destination changed to symbolic link before publication")
        if destination.exists():
            # rmdir only succeeds on an empty directory; never deletes user data.
            destination.rmdir()
        temporary.rename(destination)
    finally:
        if temporary.exists():
            assert temporary.parent == destination.parent and temporary.name.startswith(
                f".{destination.name}.staging-")
            shutil.rmtree(temporary)
    return {"output_dir": str(destination), "documents": len(docs),
            "mentions": sum(len(d.mentions) for d in docs), "review_status": "PENDING",
            "training_executed": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("generate", help="Generate fictional proposals; never approve or train")
    command.add_argument("--output-dir", required=True)
    command.add_argument("--seed", type=int, default=20261019)
    command.add_argument("--train-count", type=int, default=30)
    command.add_argument("--valid-count", type=int, default=10)
    command.add_argument("--test-count", type=int, default=10)
    args = parser.parse_args(argv)
    result = generate(args.output_dir, seed=args.seed, train_count=args.train_count,
                      valid_count=args.valid_count, test_count=args.test_count)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
