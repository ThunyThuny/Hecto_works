"""Relation-first fictional proposals with independent split clause banks.

This pure-stdlib module builds review proposals, never approved training data.
It performs no IO, model loading, inference, optimization or network requests.
Private contact values and public contact values have disjoint inventories;
institution values deliberately do not encode a label or approval status.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Any

from .b_data import validate_split_separation
from .b_inputs import FIXED_POLICY_ID, allowed_to_keep
from .schema import Document, Mention

GENERATOR_VERSION = "synthetic_rule_based_v2"
TYPES = ("PHONE", "EMAIL", "ADDRESS", "WORKPLACE", "SCHOOL", "DEPARTMENT", "POSITION")
SPLITS = ("train", "valid", "test")
NAMES = ("김가람", "이가온", "박나래", "최다온", "정새봄", "윤하람", "강아름", "임누리")
CONTACT_TYPES = frozenset(TYPES[:3])
REASONS = {"PHONE": "PERSON_CONTACT", "EMAIL": "PERSON_CONTACT", "ADDRESS": "PERSON_ADDRESS",
           "WORKPLACE": "PERSON_EMPLOYMENT", "SCHOOL": "PERSON_EDUCATION",
           "DEPARTMENT": "PERSON_EMPLOYMENT", "POSITION": "PERSON_EMPLOYMENT"}

# Every entry is a span-bearing clause.  The banks are split before rendering:
# changing an outer heading cannot turn a training clause into a test clause.
# Private and public variants intentionally share neutral entity value pools.
CLAUSE_BANKS = {
    "train": {
        "PHONE": {
            "private": ("지원서에 적힌 본인 연락처는 {v}입니다.",
                        "지원자는 면접 일정이 바뀌면 아래 번호로 연락해 달라고 요청했습니다. 번호: {v}.",
                        "지원자가 통화 가능한 번호를 남겼습니다. 연락처: {v}.",
                        "첨부된 이력서의 연락처 칸에는 {v}이라고 적혀 있습니다."),
            "public": ("이번 채용의 문의 전화는 {v}입니다.",
                       "채용 절차에 관한 문의는 안내문에 실린 번호로 받습니다. 전화: {v}.",
                       "기관 방문 예약은 접수 창구에서 받습니다. 예약 전화: {v}.",
                       "공고 하단에 채용 상담 번호가 안내되어 있습니다. 문의처: {v}."),
        },
        "EMAIL": {
            "private": ("지원자는 본인이 사용하는 이메일을 남겼습니다. 주소: {v}.",
                        "지원서에 기재한 지원자의 전자우편 주소는 {v}입니다.",
                        "지원자는 전형 결과를 다음 주소로 받고 싶다고 적었습니다. 이메일: {v}.",
                        "제출자의 회신용 이메일은 {v}입니다."),
            "public": ("공고에 안내된 채용 문의 이메일은 {v}입니다.",
                       "지원 절차에 관한 질문은 기관의 접수 메일함으로 받습니다. 주소: {v}.",
                       "제출 서류의 파일 형식은 채용 안내에 나온 메일로 문의할 수 있습니다. 이메일: {v}.",
                       "기관의 채용 상담 창구가 사용하는 전자우편 주소는 {v}입니다."),
        },
        "ADDRESS": {
            "private": ("지원자는 현재 {v}에 거주한다고 밝혔습니다.",
                        "이력서의 거주지 항목에는 {v}{subject} 기재되어 있습니다.",
                        "지원자가 우편물을 받을 집 주소를 남겼습니다. 주소: {v}.",
                        "지원서는 제출자의 거주지를 {v}{as} 표시하고 있습니다."),
            "public": ("방문 접수 장소는 {v}입니다.",
                       "채용 서류를 직접 제출하려면 {v}에 있는 접수 창구를 이용하면 됩니다.",
                       "기관 안내문은 방문객에게 다음 위치를 안내합니다. 주소: {v}.",
                       "공고에 실린 면접장 주소는 {v}입니다."),
        },
        "WORKPLACE": {
            "private": ("지원자는 {v}에서 근무한 경력을 적었습니다.",
                        "이력서에 따르면 지원자는 {v}에서 일했습니다.",
                        "지원자의 이전 직장은 {v}입니다.",
                        "지원자는 경력 사항에 {v} 재직 이력을 포함했습니다."),
            "public": ("이번 공고에서 인력을 모집하는 기관은 {v}입니다.",
                       "{v}{subject} 채용 일정을 안내했습니다.",
                       "공고의 채용 기관 항목에는 {v}{subject} 명시되어 있습니다.",
                       "{v}에서 채용 설명회를 엽니다."),
        },
        "SCHOOL": {
            "private": ("지원자는 {v}{obj} 졸업했습니다.",
                        "지원서의 최종 학력은 {v} 졸업입니다.",
                        "지원자는 {v}에서 공부한 내용을 자기소개서에 적었습니다.",
                        "제출된 학력 사항에 따르면 지원자는 {v} 출신입니다."),
            "public": ("{v}의 입학 상담 시간은 평일 오전 아홉 시부터 오후 여섯 시까지입니다.",
                       "{v}에서 채용 설명회를 개최합니다.",
                       "입학 상담 장소를 안내한 학교는 {v}입니다.",
                       "{v}{subject} 입학 상담 일정을 공지했습니다."),
        },
        "DEPARTMENT": {
            "private": ("지원자는 이전 직장의 {v}에서 일했습니다.",
                        "경력기술서에 적힌 당시 소속 부서는 {v}입니다.",
                        "지원자는 {v} 소속으로 맡았던 업무를 설명했습니다.",
                        "지원자의 마지막 근무 부서는 {v}{past}."),
            "public": ("공고에서 서류 접수를 담당하는 부서는 {v}입니다.",
                       "기관 안내에 따르면 {v}에서 방문객 문의를 받습니다.",
                       "채용 서류의 누락 여부는 {v}에서 확인합니다.",
                       "채용 안내문에 표시된 담당 부서는 {v}입니다."),
        },
        "POSITION": {
            "private": ("지원자의 당시 직위는 {v}{past}.",
                        "경력기술서에서 지원자는 {v}{as} 일했다고 설명했습니다.",
                        "지원자가 이전 직장에서 맡았던 직위는 {v}입니다.",
                        "이력서에는 {v} 재직 경력이 적혀 있습니다."),
            "public": ("이번 공고에서 모집하는 직위는 {v}입니다.",
                       "채용 안내의 모집 직위 항목은 {v}{as} 되어 있습니다.",
                       "기관은 {v} 채용에 필요한 자격 요건을 공지했습니다.",
                       "공고는 신규 채용 직위를 {v}{as} 안내하고 있습니다."),
        },
    },
    "valid": {
        "PHONE": {
            "private": ("제출자는 전형 결과를 전화로 알려 달라고 했습니다. 본인 번호: {v}.",
                        "신청인이 연락받을 전화번호를 기입했습니다. 번호: {v}.",
                        "아래 번호는 지원자가 자신의 연락 수단으로 전달한 것입니다. 전화번호: {v}.",
                        "지원자는 통화가 필요할 때 사용할 번호로 다음 번호를 보냈습니다. {v}."),
            "public": ("모집 안내에 실린 대표 문의 번호는 {v}입니다.",
                       "전형 관련 전화 상담 창구를 운영합니다. 상담 번호: {v}.",
                       "방문 전에 접수 창구로 전화해 일정을 확인할 수 있습니다. 창구 번호: {v}.",
                       "모집 일정에 대한 전화 문의처가 함께 실렸습니다. 전화번호: {v}."),
        },
        "EMAIL": {
            "private": ("제출자는 합격 여부를 받아 볼 본인 메일 주소를 적었습니다. {v}.",
                        "지원자가 답장을 받을 주소로 알려 준 이메일은 {v}입니다.",
                        "신청인이 직접 작성한 연락 수단 항목에 {v}이라는 메일 주소가 있습니다.",
                        "이 주소는 지원자가 전형 결과 수신에 사용한다고 밝혔습니다. 이메일: {v}."),
            "public": ("모집 요강에 있는 문의용 메일 주소는 {v}입니다.",
                       "기관은 전형 질문을 받을 메일함을 개설했습니다. 이메일: {v}.",
                       "방문 접수 시간에 관한 질문은 다음 창구에서 받습니다. 메일: {v}.",
                       "모집 안내의 전자우편 문의처 항목에는 {v}이라고 쓰여 있습니다."),
        },
        "ADDRESS": {
            "private": ("제출자는 현재 사는 곳을 {v}{as} 기재했습니다.",
                        "신청인이 적은 우편물 수령지는 본인 집입니다. 주소: {v}.",
                        "지원자가 생활하고 있다고 밝힌 곳은 {v}입니다.",
                        "이 주소는 지원자가 자기 거주지로 제출했습니다. {v}."),
            "public": ("모집 요강은 현장 접수처를 {v}{as} 안내합니다.",
                       "기관 설명회 참석 장소는 {v}입니다.",
                       "접수 안내에 방문할 위치가 적혀 있습니다. {v}.",
                       "기관에 직접 서류를 내는 경우 아래 장소로 방문하면 됩니다. 위치: {v}."),
        },
        "WORKPLACE": {
            "private": ("제출자는 직장 경력란에 {v}{obj} 적었습니다.",
                        "신청인이 과거에 재직했다고 밝힌 회사는 {v}입니다.",
                        "지원자의 업무 경험 설명에는 {v} 근무 이력이 나옵니다.",
                        "제출자는 {v}에서 담당했던 업무를 소개했습니다."),
            "public": ("모집을 진행하는 회사로 {v}{subject} 표시되어 있습니다.",
                       "{v}{subject} 이번 모집의 접수 기한을 발표했습니다.",
                       "채용 설명 자료는 모집 주체를 {v}{as} 소개합니다.",
                       "{v} 채용 설명회는 온라인으로 진행됩니다."),
        },
        "SCHOOL": {
            "private": ("제출자가 졸업했다고 밝힌 학교는 {v}입니다.",
                        "지원자는 학력란에 {v}{obj} 기재했습니다.",
                        "신청인은 {v} 재학 시절의 활동을 설명했습니다.",
                        "지원자는 {v}에서 이수한 과정을 지원서에 소개했습니다."),
            "public": ("{v}의 입학 상담은 사전 예약으로 진행됩니다.",
                       "설명회 주최 학교로 {v}{subject} 안내되어 있습니다.",
                       "{v}{subject} 입학 상담 예약 방법을 게시했습니다.",
                       "{v}에서 입학 설명 자료를 배포합니다."),
        },
        "DEPARTMENT": {
            "private": ("제출자의 경력란에는 {v} 배치 이력이 있습니다.",
                        "신청인은 {v}에서 수행한 일을 설명했습니다.",
                        "지원자가 당시 속해 있던 조직은 {v}입니다.",
                        "지원자는 전 직장의 {v} 근무 경험을 소개했습니다."),
            "public": ("모집 요강은 서류 제출 창구를 {v}{as} 지정했습니다.",
                       "방문 접수 업무는 기관의 {v}에서 맡습니다.",
                       "전형 안내를 게시한 담당 부서는 {v}입니다.",
                       "모집과 관련한 서류 확인은 {v} 소관입니다."),
        },
        "POSITION": {
            "private": ("제출자는 이전 직장에서 {v}{as} 재직했다고 적었습니다.",
                        "신청인이 경력란에 쓴 직함은 {v}입니다.",
                        "지원자는 본인이 맡았던 {v} 직무를 소개했습니다.",
                        "지원자가 퇴사하기 직전에 맡은 직위는 {v}{past}."),
            "public": ("모집 요강에 기재된 채용 직함은 {v}입니다.",
                       "이번 선발은 {v} 직위를 대상으로 진행됩니다.",
                       "기관은 채용 직급을 {v}{as} 명시했습니다.",
                       "모집 분야 안내에는 {v} 채용이라고 적혀 있습니다."),
        },
    },
    "test": {
        "PHONE": {
            "private": ("연락 방법을 묻자 지원자는 자신에게 연결되는 번호를 알려 주었습니다. {v}.",
                        "지원자가 남긴 회신 요청에는 본인의 전화번호가 포함되어 있습니다. {v}.",
                        "면접 안내를 받을 지원자의 전화번호는 {v}입니다.",
                        "지원자가 사용할 회신 번호를 별도로 전달했습니다. 번호는 {v}입니다."),
            "public": ("공고문은 기관의 채용 상담 전화를 {v}{as} 소개합니다.",
                       "지원 방법을 확인할 수 있는 기관 전화 창구입니다. 번호: {v}.",
                       "방문자에게 제공되는 접수 안내 전화는 {v}입니다.",
                       "접수 절차를 설명하는 전화 상담 번호가 게시되었습니다. {v}."),
        },
        "EMAIL": {
            "private": ("면접 결과를 기다리는 지원자는 본인 수신 주소를 전달했습니다. {v}.",
                        "지원자의 답신 요청에 적힌 메일 주소는 {v}입니다.",
                        "연락 수단을 확인하자 지원자는 본인 계정의 주소를 알려 주었습니다. {v}.",
                        "지원자에게 안내를 보낼 때 사용할 이메일입니다. {v}."),
            "public": ("공고문은 기관의 서류 문의 메일을 {v}{as} 소개합니다.",
                       "모집 담당 창구의 전자우편 주소가 게시되었습니다. {v}.",
                       "지원 방법에 관한 질문을 받는 기관 메일 주소는 {v}입니다.",
                       "기관은 지원자들의 질의를 다음 메일함에서 접수합니다. {v}."),
        },
        "ADDRESS": {
            "private": ("지원자의 현재 생활 근거지로 제출된 주소는 {v}입니다.",
                        "지원자는 집으로 우편 안내를 받겠다고 했습니다. 수령 주소: {v}.",
                        "거주 장소를 확인하자 지원자는 다음 주소를 답했습니다. {v}.",
                        "지원자의 거주지 확인 항목에 다음 주소가 들어 있습니다. {v}."),
            "public": ("공고문에 따르면 면접 응시자는 {v}에 있는 시험장으로 오면 됩니다.",
                       "접수 창구를 찾아오는 방문객에게 다음 위치를 안내합니다. {v}.",
                       "설명회가 열리는 기관 행사장 주소는 {v}입니다.",
                       "현장 지원을 위한 기관 접수 장소가 게시되었습니다. {v}."),
        },
        "WORKPLACE": {
            "private": ("재직 경험을 묻는 항목에서 지원자는 {v}{obj} 언급했습니다.",
                        "지원자가 전 직장으로 소개한 곳은 {v}입니다.",
                        "지원자의 경력 설명에는 {v}에서의 업무 경험이 담겨 있습니다.",
                        "지원자는 자신의 {v} 재직 경험을 근거로 지원 동기를 설명했습니다."),
            "public": ("공고문에서 채용 주체로 소개된 기관은 {v}입니다.",
                       "{v}{subject} 모집 대상과 전형 방법을 게시했습니다.",
                       "이번 채용의 시행 기관으로 {v}{subject} 안내됩니다.",
                       "{v}{subject} 게시한 공고는 접수 일정을 안내합니다."),
        },
        "SCHOOL": {
            "private": ("학업 경험을 설명하면서 지원자는 {v}{obj} 언급했습니다.",
                        "지원자가 자신의 출신 학교로 밝힌 곳은 {v}입니다.",
                        "지원자의 교육 이력에는 {v} 졸업 사실이 적혀 있습니다.",
                        "지원자는 {v}에서 배운 내용을 업무에 적용한 경험을 소개했습니다."),
            "public": ("공고문에 따르면 {v}의 입학 상담은 예약제로 운영됩니다.",
                       "입학 설명회 개최 기관은 {v}입니다.",
                       "{v}{subject} 게시한 안내는 입학 상담 절차를 설명합니다.",
                       "{v}의 신입생 설명회는 교내 강당에서 열립니다."),
        },
        "DEPARTMENT": {
            "private": ("조직 내 담당 업무를 설명하며 지원자는 {v} 근무 사실을 밝혔습니다.",
                        "지원자의 과거 소속 조직으로 확인된 부서는 {v}입니다.",
                        "지원자가 제출한 업무 이력에는 {v} 재직 사실이 적혀 있습니다.",
                        "지원자는 {v} 소속 시기에 맡았던 일을 설명했습니다."),
            "public": ("공고문에서 전형 문의 담당으로 소개된 부서는 {v}입니다.",
                       "서류 제출 방법은 기관의 {v}에서 안내합니다.",
                       "이번 접수의 창구 부서는 {v}{as} 지정되어 있습니다.",
                       "기관의 {v}{subject} 채용 질의에 답변합니다."),
        },
        "POSITION": {
            "private": ("직장 내 역할을 묻자 지원자는 당시 직위가 {v}{reported}고 답했습니다.",
                        "지원자의 마지막 직함으로 기재된 것은 {v}입니다.",
                        "지원자가 제출한 경력에는 {v}{as} 수행한 업무가 적혀 있습니다.",
                        "지원자는 종전 직장에서 맡은 {v} 역할을 설명했습니다."),
            "public": ("공고문에서 선발 대상으로 소개된 직위는 {v}입니다.",
                       "이번 공개 채용의 직급은 {v}입니다.",
                       "공고의 모집 대상 직위 항목에는 {v}{subject} 적혀 있습니다.",
                       "기관이 선발할 직위는 {v}{as} 안내되어 있습니다."),
        },
    },
}

FAMILY_WRAPPERS = {
    "train": ("채용 자료 확인 메모", "지원서와 첨부 안내 정리", "전형 자료 전달 메모", "지원 자료 편집 기록", "접수 자료 검토"),
    "valid": ("접수 내용 대조 기록", "모집 자료 확인서", "전달 문안 검토 기록", "신청 자료 정리본", "안내 자료와 제출 내용"),
    "test": ("수신 자료 검토서", "전형 문안 점검 메모", "제출 내용과 안내 사항", "채용 문서 취합 기록", "자료 전달 확인서"),
}
TEMPLATE_SPEC = {"clause_banks": CLAUSE_BANKS, "family_wrappers": FAMILY_WRAPPERS,
                 "split_clause_variants": 4, "mode_schedule": "seven-coverage-then-mixed-and-single-class"}
CANONICAL_TEMPLATE_SPEC = TEMPLATE_SPEC

ORG_STEMS = ("누리", "다온", "새봄", "한빛", "소담", "온유", "푸른", "나루", "아람", "해솔", "도담", "예담")
ORG_ENDINGS = ("테크", "솔루션", "시스템", "연구소")
DEPARTMENTS = ("운영지원팀", "사업지원팀", "인사팀", "채용지원팀", "연구지원팀", "기술지원팀", "기획팀", "교육지원팀")
POSITIONS = ("팀장", "선임연구원", "운영책임자", "책임연구원", "주임", "대리", "과장", "매니저")


def _particles(value: str) -> dict[str, str]:
    # Contact endings use their Korean digit names only for particle agreement.
    last = value[-1]
    if last.isdigit():
        last = {"0": "영", "1": "일", "2": "이", "3": "삼", "4": "사",
                "5": "오", "6": "육", "7": "칠", "8": "팔", "9": "구"}[last]
    jong = (ord(last) - 0xAC00) % 28 if "가" <= last <= "힣" else 0
    return {"obj": "을" if jong else "를", "subject": "이" if jong else "가",
            "topic": "은" if jong else "는", "as": "으로" if jong and jong != 8 else "로",
            "past": "이었습니다" if jong else "였습니다", "reported": "이었다" if jong else "였다"}


class _ContactInventory:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.used: dict[str, set[str]] = {kind: set() for kind in CONTACT_TYPES}
        codes = list(range(1, 10000))
        rng.shuffle(codes)
        # Neither role gets a contiguous range, fixed parity, prefix or domain.
        self.codes = {"private": codes[::2], "public": codes[1::2]}
        self.cursor = Counter()

    def value(self, kind: str, role: str) -> str:
        bucket = "private" if role == "private" else "public"
        while True:
            cursor_key = (kind, bucket)
            index = self.cursor[cursor_key]
            if index >= len(self.codes[bucket]):
                raise ValueError("fictional contact inventory exhausted; use smaller reviewed batches")
            code = self.codes[bucket][index]
            self.cursor[cursor_key] += 1
            if kind == "PHONE":
                value = f"000-0000-{code:04d}"
            elif kind == "EMAIL":
                prefix = self.rng.choice(("mail", "box", "inbox", "contact"))
                domain = self.rng.choice(("naru", "hanbit", "saebom", "daon"))
                value = f"{prefix}{code}@{domain}.example"
            else:
                # Fictional city, common street components and normal house numbers.
                district = self.rng.choice(("다온구", "누리구", "소담구", "해솔구"))
                road = self.rng.choice(("느티로", "솔빛로", "새봄길", "나루로", "은행길", "들꽃로"))
                value = f"가상시 {district} {road} {code % 97 + 1}"
            if value not in self.used[kind]:
                self.used[kind].add(value)
                return value


def _institution(kind: str, rng: random.Random) -> str:
    if kind == "WORKPLACE":
        return "가상" + rng.choice(ORG_STEMS) + rng.choice(ORG_ENDINGS)
    if kind == "SCHOOL":
        return "가상" + rng.choice(ORG_STEMS) + "대학교"
    if kind == "DEPARTMENT":
        return rng.choice(DEPARTMENTS)
    return rng.choice(POSITIONS)


def _approval_key(kind: str) -> str:
    return ("approved_public_contacts" if kind in CONTACT_TYPES else
            "approved_public_positions" if kind == "POSITION" else "approved_public_orgs")


def build_proposals(*, seed: int = 20261020, train_count: int = 30,
                    valid_count: int = 10, test_count: int = 10
                    ) -> tuple[list[Document], dict[str, Any]]:
    """Generate candidate-complete fictional proposals, not human approvals."""
    counts = dict(zip(SPLITS, (train_count, valid_count, test_count)))
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if any(type(n) is not int or n < 7 for n in counts.values()):
        raise ValueError("each split requires at least seven documents for contextual-type coverage")
    if sum(counts.values()) > 4000:
        raise ValueError("pilot generator is limited to 4000 documents; revise scenarios before scaling")
    rng = random.Random(seed)
    family_ids = list(range(1, 16))
    rng.shuffle(family_ids)
    allocation = {split: family_ids[i * 5:i * 5 + 5] for i, split in enumerate(SPLITS)}
    contacts = _ContactInventory(rng)
    docs: list[Document] = []
    plans: list[dict[str, Any]] = []
    for split in SPLITS:
        focus_order = list(TYPES)
        rng.shuffle(focus_order)
        for index in range(counts[split]):
            doc_id = f"syn-{seed}-{split}-{index + 1:04d}"
            family_index = index % 5
            family_id = allocation[split][family_index]
            # First seven documents guarantee both actions for every type.
            mode = "mixed" if index < 7 else ("all_mask" if index % 5 == 2 else
                    "all_keep" if index % 5 == 3 else "mixed")
            ntypes = (2, 3, 4, 3, 2, 4, 2)[index % 7]
            if mode == "all_keep":
                ntypes = 3 + index % 2
            primary = focus_order[index % 7]
            others = [kind for kind in TYPES if kind != primary]
            rng.shuffle(others)
            kinds = [primary, *others[:ntypes - 1]]
            include_name = mode != "all_keep" and index % 4 == 0
            # Per-kind shared institution value supports genuine counterfactuals.
            shared = {kind: _institution(kind, rng) for kind in kinds if kind not in CONTACT_TYPES}
            rows: list[dict[str, Any]] = []
            for kind in kinds:
                roles = (["private", "public_approved"] if mode == "mixed" else
                         ["public_approved"] if mode == "all_keep" else ["private"])
                # Some all-MASK documents contain an unverified public notice.
                if mode == "all_mask" and (index + TYPES.index(kind)) % 2 == 0:
                    roles.append("public_unapproved")
                for role in roles:
                    value = contacts.value(kind, role) if kind in CONTACT_TYPES else shared[kind]
                    if role == "public_unapproved" and kind not in CONTACT_TYPES:
                        while value == shared[kind]:
                            value = _institution(kind, rng)
                    rows.append({"kind": kind, "role": role, "value": value})
            # An occasional unapproved institution candidate varies density and
            # gives the same surface forms both approved and unapproved uses.
            if mode == "mixed" and len(rows) + int(include_name) < 9 and index % 4 == 1:
                kind = rng.choice([k for k in kinds if k not in CONTACT_TYPES] or
                                  [k for k in TYPES if k not in CONTACT_TYPES])
                used_values = {r["value"] for r in rows}
                value = _institution(kind, rng)
                for _ in range(50):
                    if value not in used_values:
                        break
                    value = _institution(kind, rng)
                if value not in used_values:
                    rows.append({"kind": kind, "role": "public_unapproved", "value": value})
                    if kind not in kinds:
                        kinds.append(kind)
            # Ensure the smallest single-class documents still have 3 candidates.
            while len(rows) + int(include_name) < 3:
                kind = next(k for k in TYPES if k not in {r["kind"] for r in rows})
                role = "private" if mode == "all_mask" else "public_approved"
                rows.append({"kind": kind, "role": role,
                             "value": contacts.value(kind, role) if kind in CONTACT_TYPES else _institution(kind, rng)})
                kinds.append(kind)
            rng.shuffle(rows)
            has_private = any(row["role"] == "private" for row in rows)
            has_public = any(row["role"] != "private" for row in rows)
            context: dict[str, Any] = {"document_type": ("general_notice" if has_private and has_public else
                                     "resume" if mode == "all_mask" else
                                     "general_notice" if "SCHOOL" in kinds else "job_posting"),
                                     "approved_public_contacts": [], "approved_public_orgs": [],
                                     "approved_public_positions": []}
            # Candidate-level approval facts are explicit and independent of
            # personal/public context. Private institutions remain approved.
            for row in rows:
                if row["role"] == "public_approved" or (row["role"] == "private" and row["kind"] not in CONTACT_TYPES):
                    context[_approval_key(row["kind"])].append(row["value"])
            for key in ("approved_public_contacts", "approved_public_orgs", "approved_public_positions"):
                context[key] = sorted(set(context[key]))
            # A value approved for one occurrence cannot be unapproved in the
            # same document. Distinct values are selected above; fail loudly.
            for row in rows:
                if row["role"] == "public_unapproved" and row["value"] in context[_approval_key(row["kind"])]:
                    raise ValueError("unapproved candidate collides with a document approval fact")
            parts: list[str] = []
            mentions: list[Mention] = []
            relations: list[dict[str, Any]] = []
            clauses: list[dict[str, Any]] = []
            cursor = 0

            def text(value: str) -> None:
                nonlocal cursor
                parts.append(value)
                cursor += len(value)

            def span(value: str, kind: str, role: str, clause_id: str) -> None:
                start = cursor
                text(value)
                action = "KEEP" if role == "public_approved" else "MASK"
                reason = ("POLICY_ALWAYS_MASK_NAME" if role == "name" else REASONS[kind] if role == "private" else
                          "PUBLIC_ROLE_APPROVED_VALUE" if role == "public_approved" else "PUBLIC_LOOKING_UNAPPROVED_VALUE")
                mention = Mention(f"m{len(mentions) + 1:02d}", start, cursor, value, kind, action, reason)
                mentions.append(mention)
                rel = {"mention_id": mention.mention_id, "relation": role, "proposed_action": action,
                       "clause_id": clause_id,
                       "simulated_approved_value": value in context.get(_approval_key(kind), []) if kind != "NAME" else False}
                relations.append(rel)
                clauses.append({"clause_id": clause_id, "relation": role, "mention_id": mention.mention_id})

            title = FAMILY_WRAPPERS[split][family_index]
            text(title + "\n")
            intro = ({"train": "지원자가 제출한 설명과 함께 전달된 기관 안내를 정리했습니다.",
                      "valid": "아래에는 신청인의 제출 내용과 별도로 받은 안내가 함께 실려 있습니다.",
                      "test": "접수한 개인 설명과 함께 첨부된 기관 자료의 내용을 옮겼습니다."}[split]
                     if has_private and has_public else
                     {"train": "지원자가 작성한 이력서의 일부입니다.",
                      "valid": "신청인이 제출한 경력 자료에서 필요한 내용을 정리했습니다.",
                      "test": "전형에 지원한 사람이 제출한 이력 사항입니다."}[split]
                     if mode == "all_mask" else
                     {"train": "기관에서 게시한 여러 안내문을 모았습니다.",
                      "valid": "아래는 기관이 배포한 안내 사항을 모은 것입니다.",
                      "test": "기관 안내문에 게시된 내용을 한데 모았습니다."}[split])
            text(intro + "\n")
            if include_name:
                text({"train": "자료 제출자: ", "valid": "신청서 작성자: ", "test": "이력 자료의 작성자: "}[split])
                span(rng.choice(NAMES), "NAME", "name", f"{split}.name.identity")
                text(".\n")
            public_notice = 0
            for row_index, row in enumerate(rows):
                kind, role, value = row["kind"], row["role"], row["value"]
                bank_role = "private" if role == "private" else "public"
                variant = rng.randrange(4)
                clause_id = f"{split}.{kind.lower()}.{bank_role}.{variant + 1}"
                template = CLAUSE_BANKS[split][kind][bank_role][variant]
                # Resolve suffixes before splitting the single literal slot.
                for key, particle in _particles(value).items():
                    template = template.replace("{" + key + "}", particle)
                before, after = template.split("{v}")
                if role != "private":
                    public_notice += 1
                    text({"train": "첨부 안내", "valid": "참고 자료", "test": "별첨 문안"}[split] +
                         f" {public_notice}: ")
                text(before)
                span(value, kind, role, clause_id)
                text(after + ("\n" if row_index + 1 < len(rows) else ""))
            doc = Document(doc_id, "".join(parts), mentions,
                           source_group=f"synthetic-v2-family-{family_id:02d}",
                           annotation_scope=["NAME", *TYPES], source=GENERATOR_VERSION,
                           split=split, policy_id=FIXED_POLICY_ID, trusted_context=context)
            doc.validate()
            if not 3 <= len(mentions) <= 9:
                raise ValueError("candidate inventory must contain between three and nine mentions")
            if any(m.action == "KEEP" and not allowed_to_keep(doc, m) for m in mentions):
                raise ValueError("generator proposed a KEEP outside the fixed policy")
            docs.append(doc)
            plans.append({"doc_id": doc_id, "split": split, "family_id": family_id,
                          "persona_id": f"fictional-persona-{seed}-{split}-{index + 1:04d}",
                          "mode": mode, "focus_types": kinds, "relations": relations,
                          "clause_ids": [row["clause_id"] for row in clauses], "clauses": clauses})
    validate_split_separation({split: [d for d in docs if d.split == split] for split in SPLITS})
    for split in SPLITS:
        observed = {(m.entity_type, m.action) for d in docs if d.split == split for m in d.mentions}
        if not {(kind, action) for kind in TYPES for action in ("MASK", "KEEP")} <= observed:
            raise ValueError("split lacks a contextual type/action pair")
    private_contacts = {(m.entity_type, m.text) for d, p in zip(docs, plans) for m, r in zip(d.mentions, p["relations"])
                        if m.entity_type in CONTACT_TYPES and r["relation"] == "private"}
    kept_contacts = {(m.entity_type, m.text) for d in docs for m in d.mentions
                     if m.entity_type in CONTACT_TYPES and m.action == "KEEP"}
    if private_contacts & kept_contacts:
        raise ValueError("private contact value also appears as KEEP")
    return docs, {"generator_version": GENERATOR_VERSION, "seed": seed,
                  "split_counts": counts, "family_allocation": allocation, "documents": plans,
                  "limitations": [
                      "Deterministic relational templates; no external LLM or human approval.",
                      "Approval facts are simulated for fictional entities, not externally verified.",
                      "Private and public contact inventories are disjoint; institution values are shared across roles.",
                      "Scenario families and span-bearing clause banks are separated before rendering each split.",
                      "Common relation semantics, vocabulary and value pools remain shared across splits.",
                      "A small synthetic pilot cannot establish real-document generalization or privacy safety."]}
