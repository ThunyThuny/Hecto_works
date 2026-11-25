# B 실제 데이터 학습 준비와 실행

B 코드와 검수 도구를 제공하며, 실제 B 업무 데이터와 학습된 B 체크포인트·성능 수치는
아직 없습니다. KDPII의 `action=null`을 MASK/KEEP로 자동 변환하지 않습니다.
이 문서는 실제 업무 데이터 절차입니다. 추가한 합성 50문서 파일럿과 검수 절차는
`SYNTHETIC_WORKFLOW.md`, 준비 검증 결과는 `SYNTHETIC_VALIDATION.md`를 따릅니다.

## 근거와 추가 설계

기술소개서의 근거는 A/B 역할 분리, gold span을 이용한 B 학습, 직접형·관계형
B의 아이디어입니다. 이 구현은 **직접형 B의 첫 실험**으로, 후보를 표시한 전체
문맥을 2분류합니다. 공개되지 않은 원 연구 코드를 복원한 구현은 아닙니다.

A 인코더 복사, 특수 후보 마커, 고정 정책과 승인 목록, 검수 선언·해시,
dev loss 체크포인트, dev 누출 제약으로 KEEP 임계값 선택, 실제 A 후보 test
평가는 모두 추가 설계입니다. 관계 추론 모듈·공동 미세조정은 아직 구현하지 않았습니다.

## 실제 데이터와 검수 양식

업무 문서를 원본·작성자·서식 변형의 `source_group` 단위로 train/dev/test에
나눕니다. 원문 전체에서 MASK와 KEEP 후보를 함께 주석합니다. A 예측은 초안으로
쓸 수 있지만 A가 놓친 후보도 사람이 추가해야 합니다. 주석 범위와 실제 텍스트를
보존하며 코드포인트 `[start,end)`를 사용합니다. 개인정보 접근권한이 적용되는
저장소에서 관리합니다. 상세 정책은 `B_ANNOTATION_GUIDELINES.md`에 있습니다.

최종 학습 디렉터리는 다음 파일을 갖춰야 합니다. 파일명 `valid`는 dev입니다.

```text
data/private/b-v1/
  train.jsonl              # canonical Document, 모든 후보 action=MASK/KEEP
  train.reviews.jsonl      # 각 최종 문서 지문과 실제 검수자의 APPROVED 선언
  valid.jsonl
  valid.reviews.jsonl
  test.jsonl
  test.reviews.jsonl
```

문서 수를 미리 채워 넣지 않았습니다. 각 분할의 MASK/KEEP 수, 유형과 문맥의
다양성을 실제 주석 후 확인합니다. 항상 MASK 유형에는 KEEP 예제를 만들지 않습니다.
공통 문서, NFKC/공백 정규화 후 같은 텍스트, 분할 간 그룹 재사용을 거부합니다.
문맥상 비슷한 수정본은 그룹을 직접 지정해 누수를 막아야 합니다.

컨테이너에서 train 검수 패킷을 생성하는 예입니다. 입력은 실제 canonical
문서와 후보이며, 없는 후보를 라벨이나 정답으로 채워주지는 않습니다.

```bash
cd /workspace/korean-pii-mvp
source scripts/activate_gpu5.sh
kpii export-b-review \
  --input data/private/train-proposals.jsonl \
  --output-dir data/private/review-train-v1 \
  --policy-id career_external_share_v1
```

`documents.jsonl`과 `mentions.csv`를 대조하며 잘못된 후보를 수정하고 누락 후보를
추가합니다. `action`을 채운 뒤 최종 문서와 새 PENDING 지문을 만듭니다.

```bash
kpii import-b-review \
  --documents data/private/review-train-v1/documents.jsonl \
  --mentions data/private/review-train-v1/mentions.csv \
  --output data/private/b-v1/train.jsonl \
  --prepare-reviews data/private/b-v1/train.reviews.jsonl
```

실제 검수자가 최종 canonical 파일을 확인한 뒤 `reviews.jsonl`의
`review_status=APPROVED`, `reviewer_id`, 시간대가 있는 `reviewed_at`,
`candidate_inventory_complete=true`, `action_labels_complete=true`를 기록합니다.
도구는 검수자의 신원이나 검수 행위 자체를 인증하지 않습니다. 승인 후 원문·라벨·
메타데이터를 바꾸면 지문이 달라지므로 다시 검수해야 합니다. valid/test도 별도로
이 절차를 진행합니다. 빈 CSV 헤더와 메타데이터 틀은 `annotation_templates/b`에 있습니다.

## 정책과 B 입력

첫 정책은 `career_external_share_v1`입니다. 이름·주민등록번호 및 미지원 유형은
항상 MASK입니다. 연락처·주소는 `approved_public_contacts`, 회사·학교·부서는
`approved_public_orgs`, 직위는 `approved_public_positions`에 정확한 원문 값이
등록되어 있어야 KEEP이 가능합니다. 목록은 검증된 외부 업무 시스템에서 제공합니다.
동일한 공용 기관명도 개인의 학력·경력에 연결되면 MASK입니다.

입력은 정책·유형·제한된 신뢰 피처와 후보 마커가 있는 원문 전체입니다. gold action,
reason_code, 검수 선언이나 임의 메타데이터는 모델에 전달하지 않습니다. 512토큰에
문서 전체가 들어가지 않으면 학습은 오류를 내고 추론은 MASK로 처리합니다.
문맥을 자르는 대신 지원하는 더 큰 길이를 선택하거나 독립적인 문맥 단위로 나누어
다시 주석합니다. `train_eval_b_gpu5.sh`의 기본값은 512입니다. 길이를 바꾼 별도
실험은 train-b의 `--max-length`를 지정하고 다시 점수·임계값을 만듭니다.

## 데이터 → 학습 → 평가 순서

1. 위 세 분할과 검수 선언을 완성하고 `preflight-b`로 전체 구조·정책·분할을 검증합니다.
2. B는 **train의 gold 후보**로 학습합니다. 완료된 A 체크포인트의 인코더를 복사하고
   새로운 MASK/KEEP 헤드와 후보 마커 임베딩을 만듭니다. A 파일은 수정하지 않습니다.
3. dev의 gold 후보 분류 loss가 가장 낮은 B를 `best/`에 저장합니다. 이 단계의 argmax
   점수는 진단용입니다. 최종 KEEP 임계값은 아직 정하지 않습니다.
4. dev 문서의 gold 후보·라벨을 제거한 사본을 만듭니다. 고정 A로 후보를 탐지하고
   선택한 B로 각 후보의 KEEP 확률을 계산합니다. native A 유형 전체를 사본의 예측
   범위로 설정하므로 업무 gold 범위가 좁아도 추론할 수 있습니다. 사본을 gold로 쓰지 않습니다.
5. 완전 검수한 dev gold와 비교해 임계값을 정합니다. 같은 A의 AllMask보다 **추가
   누출률이 0 이하**인 후보 중 과잉 마스킹을 줄이는 값을 선택합니다. 개선이 없으면
   KEEP을 비활성화해 AllMask와 같게 만듭니다. dev 관측상 0은 미래 누출 보장이 아닙니다.
6. 별도 test에서 동일 A·B·입력 설정·정책·임계값을 고정하여 한 번 평가합니다.
   모델·정책·후보·점수 해시가 다르면 평가를 거부합니다. B train/dev 및 임계값 dev와
   test의 중복도 검사합니다.

실제 데이터가 준비된 뒤 다음 명령을 **직접 실행하면 학습이 시작됩니다**.

```bash
bash scripts/train_eval_b_gpu5.sh data/private/b-v1
```

기본 A는 `/workspace/korean-pii-mvp/runs/a-native-gpu5-20261019_020722/best`입니다.
물리 GPU 5만 노출하고 `/workspace/kpii-env`를 사용합니다. 출력 경로와 A를
지정하려면 `bash scripts/train_eval_b_gpu5.sh DATA_DIR NEW_RUN_DIR A_MODEL_DIR`로 실행합니다.
독립 명령 옵션은 `kpii train-b --help`, `kpii calibrate-b --help` 등에서 확인합니다.

최종 산출물은 `model/best/b_manifest.json`, `calibration.json`,
`test-evaluation/ab-evaluation.json`, `test-evaluation/masked.jsonl`입니다.
표현 단위 누출률·과잉 마스킹률·문서 누출률과 AllMask 대비 차이를 확인합니다.
분모는 완전 검수 gold 전체이며 A가 놓친 MASK 표현도 누출입니다. B는 A 후보를
보존할지 정할 뿐 A의 미탐을 복구하지 못합니다. test 결과를 보고 재설계했다면
그 test는 개발에 사용한 셋으로 기록하고 다음 최종 평가는 새로운 독립 데이터로 합니다.

운영 입력은 `prepare-b-a-input → predict-a → score-b → apply-b` 순서로 처리합니다.
`apply-b --output-dir`의 `masked.jsonl`은 `doc_id`와 치환 텍스트만 포함합니다.
연구용 입력·후보 파일은 원문 개인정보를 포함합니다. A 미탐이 남을 수 있으므로
masked 결과가 모든 개인정보를 제거했다는 보장은 없습니다.
