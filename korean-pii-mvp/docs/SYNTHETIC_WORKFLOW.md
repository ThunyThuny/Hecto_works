# 합성 50문서 검수와 고정 A·B 실험 준비

프로젝트는 `/workspace/korean-pii-mvp`, 현재 V2 검수 패킷은
`/workspace/nfs/datasets/dylee/PII/synthetic_data`입니다. 이번 작업은 생성·검증까지이며
B 학습은 실행하지 않습니다.

## V2 50문서의 성격

외부 LLM을 호출하지 않는 규칙·템플릿 기반 파일럿입니다. 관계 계획을 만든 뒤
문장을 렌더링하며 정확한 문자 위치를 기록합니다. 같은 승인 기관·직책도 개인의
학력·경력 문맥에서는 MASK, 공적 안내에서는 KEEP으로 제안합니다. 개인 연락처와
공용 연락처는 값의 풀을 분리하며, 개인 연락처가 KEEP 후보에 재등장하지 않도록
정규화한 전화·이메일·주소를 전체 문서에 걸쳐 검사합니다. 공용으로 표현해도
승인 목록에 없는 값은 MASK이고, 이름은 첫 정책에서 항상 MASK입니다.

총 50문서(train 30 / valid 10 / test 10)이며 문서당 후보는 3~9개입니다.
총 285후보(MASK 156 / KEEP 129) 중 이름 10개를 제외한 문맥 판단 후보는 275개입니다.
7개 문맥 유형이 각 분할에 모두 포함됩니다. 모든 라벨은 생성 초안입니다.

기관·인물·주소는 가상 상황이고 전화는 `000`, 이메일은 `.example`을 사용합니다.
승인 목록도 **가상 시나리오의 시뮬레이션**이며 실제 기관 확인이나 사람의 검수
승인이 아닙니다. 출처는 `synthetic_rule_based_v2`, seed는 `20261020`입니다.

15개 문서 서식 계열을 생성 전에 5개씩 분할하고 같은 계열의 변형을 다른 분할로
보내지 않습니다. 페르소나 ID도 문서별로 구분하고, 후보를 담은 핵심 문장 은행도
분할마다 별도로 둡니다. 값만 제거했을 때 동일해지는 핵심 문장의 분할 간 중복을
검사합니다. 같은 기본 어휘와 정책 의미는 공유하므로 실제 문체·문맥 일반화의
증거로 해석하지 않습니다. 가상 값은
고정 A의 학습 분포와 달라 탐지 성능도 실제 업무를 대표하지 않습니다.

## 파일

```text
synthetic_data/
  review.html                    # 색상으로 표시한 50문서와 제안 라벨
  review.md                      # 원문·구간·유형·제안 라벨 표
  proposed_documents.jsonl       # 전체 생성 초안
  proposals/{train,valid,test}.jsonl
  generation_plan.json           # 관계·페르소나·분할 계획; 모델 입력에 사용하지 않음
  generation_manifest.json       # seed·코드/정책/파일 해시·유형/라벨 수
  automated_audit.json           # 라벨·구간·연락처 충돌·분할 문장 중복 검사
  quality_review.json            # 전체 문서에 대한 AI 검토 결과; 사람 승인 아님
  a_checkpoint.lock.json         # 고정 A의 경로·manifest·가중치·토크나이저 해시
  review/
    documents.jsonl              # 검수 원문·후보·메타데이터
    mentions.csv                 # 수정할 후보·구간·MASK/KEEP·사유
    reviews.jsonl                # 모두 PENDING, 완료 플래그 false
```

검수 중 수정한 파일은 최초 생성 manifest의 해시와 달라질 수 있습니다. 최초
생성본의 provenance와 최종 검수본의 지문은 별도로 관리합니다.

V1은 `/workspace/nfs/datasets/dylee/PII/synthetic_data_archive/v1_20261019_before_v2`에
원본 파일 그대로 별도 보관합니다. 보관 폴더의
`archive_record.json`에는 이전 파일 전체의 SHA256이 있고 `generator_source_v1.py`는
이전 생성 소스입니다. 프로젝트의 `synthetic_v1.py`와 V1 회귀 테스트도 유지합니다.
자동 검사와 AI 자연스러움 검토가 통과해도 `review/reviews.jsonl`은 모두 PENDING입니다.
최종 재검토 결과와 대표 사례는 [SYNTHETIC_V2_VALIDATION.md](SYNTHETIC_V2_VALIDATION.md)에 있습니다.

## 검수와 최종 데이터 만들기

`review.html` 또는 `review.md`로 원문을 읽고 `review/mentions.csv`를 수정합니다.
경계·유형·라벨·사유뿐 아니라 누락 후보도 확인합니다. 수정 CSV를 canonical
문서로 가져와 수정 후 내용의 새 PENDING 선언을 생성합니다.

```bash
cd /workspace/korean-pii-mvp
source /workspace/kpii-env/bin/activate
DATA=/workspace/nfs/datasets/dylee/PII/synthetic_data

kpii import-b-review \
  --documents "$DATA/review/documents.jsonl" \
  --mentions "$DATA/review/mentions.csv" \
  --output "$DATA/reviewed-documents.jsonl" \
  --prepare-reviews "$DATA/reviewed.reviews.jsonl"
```

검수자는 최종 문서를 확인한 뒤 `reviewed.reviews.jsonl`에 실제 `reviewer_id`,
시간대 포함 `reviewed_at`, `review_status=APPROVED`,
`candidate_inventory_complete=true`, `action_labels_complete=true`를 기록합니다.
지문은 바꾸지 않습니다. 승인 후 내용을 바꾸면 다시 검수합니다. 자동 승인 명령은
없으며, 도구는 검수자 신원이나 실제 검수 행위 자체를 인증하지 않습니다.

```bash
kpii prepare-synthetic-reviewed \
  --documents "$DATA/reviewed-documents.jsonl" \
  --reviews "$DATA/reviewed.reviews.jsonl" \
  --a-lock "$DATA/a_checkpoint.lock.json" \
  --output-dir "$DATA/ready-v2"
```

검수 지문·정책·라벨·분할 중복·고정 A를 검증한 뒤
`ready-v2/{train,valid,test}.jsonl` 및 대응 `.reviews.jsonl`, A lock을 만듭니다.
PENDING이 하나라도 있으면 중단합니다. 원래 분할과 지문을 유지하며 관계 계획·
정답 사유를 B 입력으로 넘기지 않습니다. 출력은 새 경로를 사용합니다.

## 학습 없이 확인하기

검수와 위 내보내기를 마친 뒤 다음 명령은 검사만 수행합니다.

```bash
bash scripts/train_eval_b_synthetic_gpu5.sh --check-only "$DATA/ready-v2"
```

`--check-only`는 모델·GPU를 로드하지 않고 학습·추론·학습 출력 디렉터리 생성도
수행하지 않습니다. 미검수 패킷을 직접 학습 데이터로 지정하면 거부됩니다.

## 나중에 학습을 시작할 명령

**다음 명령은 직접 실행하면 실제 학습을 시작합니다. 이번 준비에서는 실행하지 않습니다.**

```bash
bash scripts/train_eval_b_synthetic_gpu5.sh "$DATA/ready-v2"
```

고정 A는 `/workspace/korean-pii-mvp/runs/a-native-gpu5-20261019_020722/best`입니다.
시작 전·종료 후 경로·manifest·가중치·설정·토크나이저 해시를 검사합니다.
A 재학습은 없으며 B 초기화용 인코더 복사와 후보 추론에만 사용합니다.
물리 GPU 5의 UUID만 노출하고 `/workspace/kpii-env`를 사용합니다.

흐름은 B gold 후보 train → dev loss 최적 B → 고정 A dev 후보 → B KEEP 점수 →
dev 임계값 고정 → 고정 A+B test 평가입니다. 같은 A 후보의 AllMask와 누출·과잉
마스킹을 비교하고 A가 놓친 MASK도 누출 분모에 포함합니다. 결과는
`runs/b-synthetic-gpu5-*/model/best`, `calibration.json`,
`test-evaluation/ab-evaluation.json`에 저장되며 합성 출처를 기록합니다.

50문서는 검수·연결 확인용입니다. 확대 시 새 버전 경로를 사용하며 문서 서식뿐
아니라 관계·문체 다양성도 늘리고 다시 검수해야 합니다.

```bash
kpii generate-synthetic --output-dir "$DATA/../synthetic_data_next" \
  --seed 20261020 --train-count 1000 --valid-count 250 --test-count 250
```

합성 test 점수와 실제 업무 성능은 구분합니다. 실제 문서를 확보하면 별도 실제
dev에서 임계값을 다시 선택하고, 학습·조정에 쓰지 않은 실제 test로 검증합니다.
