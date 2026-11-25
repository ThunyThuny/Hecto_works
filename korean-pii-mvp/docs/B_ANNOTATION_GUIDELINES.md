# B 실제 데이터 주석·검수 지침

현재 제공하는 것은 B 학습 코드와 실제 데이터 검수 도구이다. 이 문서는
`career_external_share_v1` 정책의 첫 버전을 설명한다. 실제 학습문서와 사람이
검수한 MASK/KEEP 정답은 사용자가 준비한다. KDPII의 기존 개체 유형이나 A의
예측에서 MASK/KEEP 정답을 자동 생성하지 않는다. 합성 파일럿으로 성능을
주장하거나 B가 학습됐다고 간주하지 않는다.

## 1. 주석 단위와 정책

주석자는 B가 실제 입력으로 받는 **문서 전체와 신뢰 메타데이터**를 읽는다.
개체 유형, 표현의 위치, 보호 여부는 별도 필드로 유지한다.

| 대상 | MASK | KEEP가 가능한 조건 |
|---|---|---|
| 이름·주민등록번호 | 항상 MASK | 첫 정책에서 허용하지 않음 |
| 전화·이메일·주소 | 개인 연락처·거주지, 근거 부족 | 정확한 값이 `approved_public_contacts`에 있고 문맥도 공용 정보임 |
| 회사·학교·부서 | 개인의 소속·학력·근무 이력 | 정확한 값이 `approved_public_orgs`에 있고 공적 안내 문맥임 |
| 직위 | 개인의 경력 정보 | 정확한 값이 `approved_public_positions`에 있고 공적 안내 문맥임 |
| 그 외 native33 유형 | 첫 정책에서 MASK | 지원 유형 확장 전까지 허용하지 않음 |

메타데이터에 값이 있다고 자동 KEEP은 아니다. 같은 승인된 학교명도 개인의
학력에 연결되면 MASK이고, 기관 일반 안내라면 KEEP이 될 수 있다. 모델의
KEEP 점수와 검증셋에서 고른 임계값도 통과해야 실제로 보존한다.

신뢰 목록은 업무 시스템의 공식 연락처·기관 등록정보 등 **문서 밖의 검증된
정책 출처**에서 제공한다. 본문에 “공개 정보”라고 적혀 있다는 사실이나 생성기의
숨겨진 계획은 근거가 아니다. 주석자가 원하는 KEEP 정답을 만들기 위해 목록에
값을 임의 추가하지 않는다. 승인 목록의 출처·적용 시점은 업무 데이터 관리 기록에
남긴다. 모델에 전달할 필드는 별도로 제한되므로 검수 메모에 정답을 적어도 입력
피처로 사용되지 않는다.

native 라벨은 그대로 사용할 수 있다. 지원되는 별칭은 코드의
`b_inputs.TYPE_ALIASES`를 따른다. 예를 들어 `PS_NAME`은 이름,
`OGG_EDUCATION`은 학교, `QT_MOBILE`은 전화로 처리된다. 낯선 유형을 임의로
허용 유형으로 바꾸지 않는다.

판단 근거가 부족하면 `MASK`로 표시하고 `INSUFFICIENT_CONTEXT` 같은 사유를
남긴다. 서로 다른 검수자가 합의하지 못한 사례는 승인하지 않고 조정한다.

## 2. 실제 문서 준비와 분할

원문이 바뀌면 문자 위치도 바뀐다. 원문 확정 후 Unicode 문자 기준의
`[start, end)` 구간을 주석한다. Python의 `text[start:end]`가 `mention.text`와
정확히 같아야 한다. 한글, 공백, 줄바꿈, 이모지도 모두 원문에 포함한다.
현재 canonical 스키마는 겹치는 구간을 허용하지 않는다.

문서에는 다음 필드가 필요하다.

- `doc_id`: 문서별 고유 ID.
- `source_group`: 같은 원본, 같은 작성자/페르소나, 같은 서식의 변형을 묶는 그룹.
- `split`: `train`, `dev`, `test`. 기존 A의 `valid`/`validation`은 dev 별칭으로 허용.
- `annotation_scope`: 해당 문서에서 빠짐없이 주석할 유형 목록.
- `policy_id`: `career_external_share_v1`.
- `trusted_context`: 위 정책의 신뢰 목록과 필요한 문서 종류.
- `text`, `mentions`: 원문과 후보 목록. 초기 `action`은 `null`일 수 있다.

학습·검증·테스트를 **검수 전에 그룹 단위로 분리**한다. 같은 문서의 요약·수정본,
같은 사람의 연관 경력 문서 등을 다른 분할에 넣지 않는다. 도구는 ID 중복,
공백·Unicode 정규화 후 같은 원문, 분할 간 `source_group` 재사용을 거부한다.
의미가 비슷한 변형까지 자동으로 식별하지는 않으므로 그룹을 직접 지정해야 한다.

고정된 A의 예측을 주석 출발점으로 쓸 수 있지만, **A가 놓친 MASK 대상과 KEEP
대상도 원문에서 직접 추가**해야 한다. 예측 후보만 평가 정답으로 쓰면 A의 누락이
평가에서 사라진다. 잘못 탐지한 후보는 삭제하고 경계·유형 오류는 수정한다.

## 3. 검수 패킷 내보내기

컨테이너의 가상환경을 활성화한 뒤 실제 문서 파일을 지정한다.

```bash
python -m kpii_mvp.b_data export-review \
  --input data/private/b/train-proposals.jsonl \
  --output-dir data/private/b/review-train-v1 \
  --policy-id career_external_share_v1
```

`--policy-id`는 내보낸 사본의 적용 정책만 지정한다. 기존 라벨을 승인하거나
본문을 공용 정보로 인정하는 동작이 아니다. 원본 입력은 변경하지 않는다.
검증·테스트도 서로 다른 실제 문서로 패킷을 만든다.

| 파일 | 용도 |
|---|---|
| `documents.jsonl` | 원문, 범위, 분할, 신뢰 메타데이터와 초기 후보 |
| `mentions.csv` | 사람이 수정하는 후보·구간·유형·MASK/KEEP·사유 |
| `reviews.jsonl` | 문서별 `PENDING` 선언과 원본 패킷 지문 |

파일에 원문 개인정보가 들어 있다. 실제 데이터 접근권한이 적용되는 위치에서
관리한다. 출력 파일이 이미 있으면 덮어쓰지 않으므로 새 버전 경로를 사용한다.

## 4. CSV 주석

열은 `doc_id, mention_id, start, end, text, entity_type, action, reason_code`이다.
빈 `action`은 미주석이다. **빈 값·null·누락 후보는 KEEP이 아니다.**

각 문서에 대해 다음을 확인한다.

1. 원문 전체를 읽고 주석 범위 안의 모든 후보를 확인한다.
2. A의 오탐을 삭제하고 누락 후보를 행으로 추가한다.
3. `start`, `end`, `text`, `entity_type`을 원문과 대조한다.
4. 후보마다 `MASK` 또는 `KEEP`을 명시한다.
5. `PERSON_CONTACT`, `PERSON_EDUCATION`, `PERSON_EMPLOYMENT`,
   `PUBLIC_CONTACT`, `PUBLIC_ORGANIZATION`, `POLICY_ALWAYS_MASK`,
   `INSUFFICIENT_CONTEXT` 등 사유를 기록한다. 사유는 분석용이며 B 입력이 아니다.

`mention_id`는 문서 안에서 고유해야 한다. 새 후보에는 새 ID를 부여한다.
유형이 `annotation_scope`에 없으면 먼저 `documents.jsonl`의 범위 선언을
수정하고 그 유형도 전수 주석한다. 유형 범위를 일부만 주석해 완전 주석으로
선언하지 않는다. 원문을 수정해야 한다면 모든 구간을 다시 확인한다.

**모든 문서는 CSV에 적어도 한 행이 있어야 한다.** 후보가 실제로 하나도 없으면
`doc_id`만 채우고 나머지 7개 열은 비운 행을 하나 둔다. 내보내기 도구도 음성
문서를 이 형태로 표시한다. 이 행과 실제 후보 행을 같은 문서에 함께 둘 수 없다.
문서의 행을 모두 실수로 지우면 가져오기가 실패한다. 일부 후보 행 누락은 기계가
완전하게 판별할 수 없으므로 원문 전수 검수가 필요하다.

CSV는 UTF-8 BOM과 표준 인용부호를 사용한다. 쉼표·줄바꿈을 포함한 표현은
CSV 파서로 편집한다. 스프레드시트로 열 경우 전화번호 앞자리·날짜·수식 문자열이
자동 변환되지 않도록 열을 텍스트로 가져온다. 자동 변환된 표현은 원문 검사에서
오류로 처리된다. 정수 위치에 `1.0`, 음수, 공백 등을 입력하지 않는다.

## 5. 수정본 가져오기와 승인

먼저 수정 CSV로 canonical 문서를 만들고 **수정 후 내용의 새 지문**을 생성한다.

```bash
python -m kpii_mvp.b_data import-review \
  --documents data/private/b/review-train-v1/documents.jsonl \
  --mentions data/private/b/review-train-v1/mentions.csv \
  --output data/private/b/train-reviewed.jsonl \
  --prepare-reviews data/private/b/train-review-declarations.jsonl
```

이 단계는 라벨 정답을 승인하지 않는다. 새 선언은 항상 `PENDING`이고 완료
플래그도 `false`이다. 첫 내보내기의 지문은 action·경계를 수정하면 유효하지 않다.
이전 지문을 재사용하거나 지문만 수동 변경해 기존 승인을 살리지 않는다.

검수자는 **생성된 최종 canonical 문서**와 원문을 다시 대조한 뒤 선언 파일에 다음을
직접 기록한다.

- `review_status`: 완료한 문서만 `APPROVED`.
- `reviewer_id`: 실제 검수자 식별자.
- `reviewed_at`: 실제 검수 시각, 예: `2026-10-19T15:00:00+09:00`.
- `candidate_inventory_complete`: 전체 후보를 확인했을 때만 `true`.
- `action_labels_complete`: 모든 후보의 정책 라벨을 확인했을 때만 `true`.
- `notes`: 불일치 조정 내역이나 필요한 검수 메모.

위 시각은 형식 설명용이다. 도구가 검수자의 신원이나 실제 검수 행위를 인증하는
것은 아니다. 승인 선언은 제공된 검수 기록이며, 보고서에도 이 한계를 표시한다.
원문·구간·라벨·메타데이터·분할 중 하나라도 승인 후 바뀌면 지문이 달라져
학습/평가가 거부된다. 변경본은 다시 PENDING 지문을 만들고 검수한다.

```bash
python -m kpii_mvp.b_data validate \
  --input data/private/b/train-reviewed.jsonl \
  --reviews data/private/b/train-review-declarations.jsonl \
  --split train \
  --policy-id career_external_share_v1
```

검증은 파일 구조, 구간, 승인 선언, 완전 주석 선언, 정책 ID, 분할과 지문 일치를
확인한다. 의미상 정답인지는 사람이 판단해야 한다. B 학습·추론은 추가로
KEEP 허용 유형·신뢰 목록 조건·입력 길이 등을 검사한다. 입력이 너무 길면
소유 관계 문장이 조용히 잘리지 않도록 거부한다. 이때 더 큰 지원 입력 길이를
선택하거나 독립적으로 판단 가능한 문맥 단위로 나누고 다시 검수한다.

## 6. 실제 데이터 준비가 끝난 뒤

train과 dev는 각각 MASK/KEEP의 유효한 학습·검증 사례가 필요하다. 문서 수만
맞추지 말고 유형별 두 라벨의 수와 소유 관계의 다양성을 확인한다. 일부 유형은
정책상 항상 MASK이므로 KEEP 균형을 억지로 만들지 않는다.

train으로 B를 학습하고, dev의 **정답 후보**와 **고정 A의 예측 후보**를 구분해
검증한다. 최종 KEEP 임계값은 dev의 통합 누출·과잉 마스킹으로 고른다. test는
최종 평가에만 사용하며 test의 라벨·문서로 모델이나 임계값을 고르지 않는다.
데이터가 아직 없다면 이 준비 상태에서 멈추고 실제 주석을 기다린다.
