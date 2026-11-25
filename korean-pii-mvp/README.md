# 한국어 개인정보 마스킹 연구 MVP

기술소개서의 `후보 탐지 A → 문맥 판단 B → 원문 구간 치환` 시나리오를 단계별로 구현하는 프로젝트입니다. **KDPII 전처리·A 학습과 평가에 이어 B 학습·추론 코드, 검수 양식, KEEP 임계값 선택과 A+B 통합 평가를 구현했습니다.** A의 전체 학습은 완료했습니다. B용 합성 50문서 파일럿 생성·검수 파이프라인도 제공하며, 승인된 MASK/KEEP 데이터로 B를 학습하도록 구성했습니다. B 학습은 실행하지 않았습니다. 기본 순서는 [B_WORKFLOW.md](docs/B_WORKFLOW.md), 합성 실험은 [SYNTHETIC_WORKFLOW.md](docs/SYNTHETIC_WORKFLOW.md), 정책은 [B_ANNOTATION_GUIDELINES.md](docs/B_ANNOTATION_GUIDELINES.md)에 있습니다.

## 구현 상태

본격 B 실험용 [합성3,000문서 V3](docs/SYNTHETIC_BULK3000.md)를 별도 경로에 제공합니다.
train2,000/valid500/test500, B후보15,540개이며 전부 PENDING입니다. 기존50은 유지하고
제목·라벨 연동, 연락처 풀, 승인 반례, 분할별 문장 은행을 보완했습니다. 학습은 실행하지 않았습니다.

현재 생성 라벨을 사용하는 첫 B 실험 명령은
[B_SYNTHETIC_EXPERIMENT.md](docs/B_SYNTHETIC_EXPERIMENT.md)에 있습니다.
명시적 실험 옵션은 원본 PENDING을 유지하고 `human_reviewed=false`로 기록합니다.

합성데이터 V2 파일럿도 제공합니다. 개인·공용 연락처를 분리한 50문서·285후보이며,
문서당 후보 3~9개와 분할별 핵심 문장 은행을 사용합니다. 기존 V1은 별도 보관했습니다.
[SYNTHETIC_V2_VALIDATION.md](docs/SYNTHETIC_V2_VALIDATION.md)에 전체 재검토 결과,
[SYNTHETIC_WORKFLOW.md](docs/SYNTHETIC_WORKFLOW.md)에
50문서 생성·검수·승인 후 분할·고정 A와 GPU 5용 B 실행 순서를 정리했습니다.
생성 초안은 전부 PENDING이며 B 학습은 별도 실행 단계입니다.

| 구성 | 상태 |
|---|---|
| 원문 Unicode 문자 `[start, end)` 스키마 | 구현, 원문 substring·구간·유형·중복 검증 |
| KDPII v1 | 실제 배포 JSON 전용 어댑터, 대화 집계, 원본 분할 보존 |
| 데이터 오류 | 기본 중단, 명시적 quarantine 옵션과 감사 manifest |
| A | A.X 등 HF 인코더 + BIO token classification, 긴 문서 중첩 창 |
| 정렬 오류 | 경계가 토큰 중간이면 관련 감독을 `-100` 처리하고 보고; gold를 넓히거나 O로 바꾸지 않음 |
| 학습 제어 | train 라벨만 사용, dev 체크포인트 선택, test 혼입 차단 |
| 단계별 전이 | native 라벨 학습 후 `--reset-head`로 업무 유형 헤드 재구성 |
| 마스킹·평가 | AllMask 기준모델, 표현 단위 누출·과잉 마스킹, 문서 단위 누출 |
| B 데이터 | 실제 원문 검수 CSV, PENDING/APPROVED 선언, 정답·원문 지문, 분할 중복 차단 |
| B 코드 | A 인코더 복사 + 새 2분류 헤드, gold 후보 학습, dev loss 최적 체크포인트 |
| B 통합 | 고정 A 후보 점수, dev KEEP 임계값 선택, 별도 test 평가, AllMask 비교 |
| 후속 구현 | 실제 데이터로 B 학습, 정규식 후보 병합, Gazetteer/CRF/경계 MLP, 공동 학습 |

PDF에서 확인되는 것은 A/B 역할 분리, 직접형·관계형 B, gold span을 활용한 B 학습, 공동 미세조정, 관계 계획 기반 합성데이터, 표현 단위 평가입니다. 본 프로젝트의 디렉터리·스키마·오류 처리·순차 학습·CLI는 추가 설계입니다. 정확한 KDPII-19 목록이나 원 연구의 학습 조건을 복원했다고 주장하지 않습니다.

## 프로젝트 구조

```text
configs/mvp-policy.example.json     # 이후 업무 데이터·정책의 설계 참고
schemas/document.schema.json       # 외부 도구용 JSON Schema
scripts/download_kdpii.py           # 원본 3개 파일 다운로드·체크섬
src/kpii_mvp/
  schema.py                        # Mention, Document 및 엄격한 검증
  data.py                          # canonical JSONL I/O
  kdpii.py                         # 실제 v1 + 명시적 spans/BIO 어댑터
  tokenization.py                  # offset 정렬, overflow, BIO 복원
  stage_a.py                       # train/predict, 체크포인트·manifest
  stage_b.py                       # B 확장 Protocol, AllMaskDecision
  b_data.py                        # 실제 주석·검수 교환 및 승인 검증
  b_inputs.py                      # 고정 정책·유형 별칭·후보 문맥 입력
  b_model.py                       # gold 후보 B 학습 / 후보 KEEP 확률
  b_pipeline.py                    # 정답 없는 A 입력·실제 데이터 사전 검증
  b_evaluation.py                  # dev 임계값·고정 A+B test·최종 치환
  synthetic.py                     # V2 검증·PENDING 검수 패킷 생성
  synthetic_v2.py                  # 연락처 분리·다양한 관계/문장 생성
  synthetic_v1.py                  # 이전 생성 소스 보존
  synthetic_audit.py               # 라벨·구간·연락처 충돌·분할 문장 감사
  synthetic_versioning.py          # 기존 데이터 해시 보존·버전 교체/복구
  synthetic_bulk.py                # 3,000문서 유형·역할·모드 정확 할당 생성
  synthetic_v3_templates.py        # 분할별448개 핵심 문장·중립 제목
  synthetic_bulk_audit.py          # 정확 할당·세부 역할·단순 승인 규칙 편향 검사
  synthetic_flow.py                # A 체크포인트 lock·승인 후 데이터 내보내기
  masking.py                       # 원문 구간 합집합·치환
  evaluation.py                    # exact detection / 최종 masking 평가
  cli.py                           # kpii 명령
data/examples/                     # 코드 동작 확인용 가상 문서만 포함
tests/                             # stdlib unittest
```

## 설치

Python 3.10 이상을 사용합니다. 데이터 처리·평가·단위 테스트는 별도 Python 패키지가 필요 없습니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train]'
python -m unittest discover -s tests
```

7776 컨테이너의 프로젝트 경로는 `/workspace/korean-pii-mvp`, 전용 독립 가상환경은 `/workspace/kpii-env`입니다. PyTorch와 Transformers를 이 환경에 직접 설치하며, 다른 환경의 site-packages를 참조하지 않습니다. 실행 설정은 **물리 GPU 5번만 노출**합니다. PyTorch에서는 그 장치가 논리 `cuda:0`으로 보입니다.

```bash
cd /workspace/korean-pii-mvp
source scripts/activate_gpu5.sh
python -m pip check
```

## 1. 데이터 계약

`Document`에는 `doc_id`, `text`, `mentions`, `source_group`, `annotation_scope`가 필수입니다. `annotation_scope`는 **해당 문서에서 완전히 주석한 개체 유형**입니다. `mentions=[]`도 명시적으로 제공해야 하며 필드 누락을 음성 예제로 취급하지 않습니다.

```json
{
  "schema_version": "1.0",
  "doc_id": "example",
  "source_group": "template-family-01",
  "source": "domain-reviewed",
  "split": "train",
  "policy_id": "career_external_share_v1",
  "trusted_context": {},
  "annotation_scope": ["NAME", "SCHOOL"],
  "text": "김가람은 한빛대학교 출신입니다.",
  "mentions": [
    {"mention_id": "m1", "start": 0, "end": 3, "text": "김가람", "entity_type": "NAME", "action": "MASK"},
    {"mention_id": "m2", "start": 5, "end": 10, "text": "한빛대학교", "entity_type": "SCHOOL", "action": "MASK"}
  ]
}
```

원문 기준 Python Unicode code point 인덱스를 사용합니다. UTF-8 byte나 JavaScript UTF-16 인덱스와 혼용하지 않습니다. `text[start:end] == mention.text`가 반드시 성립해야 합니다. 한 BIO 헤드가 표현할 수 없는 겹치는 정답은 거부합니다. `action=null`은 보호 판단 미주석이며 KEEP이 아닙니다. 실제 교차 필드 검증의 기준은 `schema.py`입니다.

```bash
kpii validate --input data/examples/train.jsonl
```

## 2. KDPII 다운로드·전처리

원본: <https://zenodo.org/records/10968609>. 개정본과 혼합하지 않습니다.

```bash
python scripts/download_kdpii.py --output-dir data/raw/kdpii-v1
kpii inspect-kdpii data/raw/kdpii-v1/train.json
kpii prepare-kdpii \
  --train data/raw/kdpii-v1/train.json \
  --valid data/raw/kdpii-v1/valid.json \
  --test data/raw/kdpii-v1/test.json \
  --adapter kdpii-v1 \
  --invalid-policy quarantine \
  --output-dir data/processed/kdpii-v1
```

기존 전처리 결과를 덮어쓰지 않습니다. 설정을 바꿔 재실행할 때는 새 출력 디렉터리를 지정합니다. 오류를 먼저 검사하려면 `--invalid-policy error`를 사용합니다.

실제 원본의 `sent_idx`, `sentence`, `PII_set[{id,form,label,begin,end}]`, `sent_seq`, `labelling_seq`를 읽습니다. `begin/end`의 원문 구간과 문자 BIO를 함께 검증합니다. `sent_idx`의 첫 `_` 이전 값을 대화 그룹으로 사용하고, 같은 대화의 문장을 파일 순서대로 줄바꿈으로 연결하여 구간을 이동합니다. 입력 분할은 그대로 유지하며 오류 대화 전체를 격리합니다.

원본 33개 유형을 유지하고 모든 개인정보 판단 라벨은 `action=null`로 둡니다. 예시 정책의 9개 유형으로 자동 축소하지 않습니다. 명시적인 전체 라벨 매핑은 `--label-map`으로 제공할 수 있으며 매핑되지 않은 유형을 몰래 O로 바꾸지 않습니다.

Manifest에는 파일 SHA-256, 라벨 목록, 데이터 수, 그룹·중복 검사, 격리 이유·건수가 남습니다. 원문이나 개인정보를 검사 로그에 출력하지 않습니다.

7776에서 원본 체크섬 검증 및 전처리를 실행한 결과:

| 분할 | 원본 문장 | 원본 대화 | 사용 대화 | 사용 mention | 격리 대화 |
|---|---:|---:|---:|---:|---:|
| train | 40,109 | 3,664 | 3,663 | 17,890 | 1 |
| valid | 5,011 | 459 | 459 | 2,292 | 0 |
| test | 4,891 | 458 | 456 | 2,189 | 2 |

이 수치는 본 어댑터·격리 정책의 결과이며 PDF의 KDPII-19 실험 분할과 다릅니다. 같은 원문 변형·페르소나·템플릿의 독립성은 그룹 ID만으로 증명되지 않으므로 자체 데이터에는 생성 계획의 그룹을 제공해야 합니다.

## 3. A의 native 라벨 학습

```bash
kpii train-a \
  --train data/processed/kdpii-v1/train.jsonl \
  --dev data/processed/kdpii-v1/valid.jsonl \
  --model-name-or-path skt/A.X-Encoder-base \
  --output-dir runs/a-native-v1 \
  --device cuda --max-length 512 --stride 128 \
  --batch-size 4 --eval-batch-size 8 --epochs 3
```

이 컨테이너에서는 `scripts/activate_gpu5.sh`로 물리 GPU 5의 UUID를 확인해 해당 장치만 노출합니다. CLI 기본 device는 CPU입니다. 위 명령은 전체 학습 예시이며 검증 작업에서 전체 학습을 자동 실행하지 않습니다.

A.X의 공식 카드가 요구하는 `transformers>=4.51`을 사용합니다. 빠른 토크나이저의 원문 offset을 사용하며 문서 끝까지 모든 중첩 창을 처리합니다. 특수 토큰·패딩·잘린 개체·정렬 불가능한 개체의 감독은 `-100` 처리합니다. `alignment_report.json`에 제외된 감독과 유형별 양성 토큰 수를 남기며 어느 학습 유형이든 양성 감독이 0이면 중단합니다. `--boundary-policy reject`는 경계 문제를 즉시 오류로 처리하는 더 엄격한 옵션입니다.

체크포인트는 dev의 **원래 gold 전체에 대한 exact span+type F1**으로 선택합니다. 정렬된 부분만 골라 dev gold를 줄이지 않습니다. `training_manifest.json`, `alignment_report.json`, `best/labels.json`과 모델·토크나이저가 저장됩니다. 테스트셋은 학습·모델 선택에 사용하지 않습니다.

## 4. 추론·탐지 평가

```bash
kpii predict-a \
  --model-dir runs/a-native-v1/best \
  --input data/processed/kdpii-v1/test.jsonl \
  --output runs/a-native-v1/test-predictions.jsonl --device cuda
kpii evaluate-detection \
  --gold data/processed/kdpii-v1/test.jsonl \
  --pred runs/a-native-v1/test-predictions.jsonl \
  --output runs/a-native-v1/test-detection.json
```

예측은 원문 좌표의 canonical JSONL이며 후보 `action`은 null입니다. 원문·문서 ID가 다른 gold/pred 조합은 거부하고, 전체·유형별·그룹별 exact P/R/F1 및 후보 구간 커버리지를 제공합니다. 명시적인 MASK gold가 있을 때는 보호 구간 커버리지도 계산하고, 해당 정답이 없으면 null로 표시합니다. 학습·예측 파일과 manifest는 내부 연구 산출물로 원문을 포함합니다.

## 5. 업무 유형으로 순차 미세조정

보호·보존 후보를 모두 주석한 업무 데이터의 train/dev를 만든 뒤 다음처럼 헤드를 명시적으로 교체합니다.

```bash
kpii train-a \
  --train data/domain-v1/train.jsonl --dev data/domain-v1/valid.jsonl \
  --model-name-or-path runs/a-native-v1/best --reset-head \
  --output-dir runs/a-domain-v1 --device cuda
```

인코더만 엄격히 전이하고 새 BIO 헤드를 학습합니다. 저장된 분류 모델의 라벨 체계가 다르면 `--reset-head` 없이 실행할 수 없습니다. 다른 데이터가 주석하지 않은 유형을 O로 학습시키는 단순 병합을 피합니다. `annotation_scope`가 전체 학습 유형을 덮지 않는 문서는 거부합니다.

## 6. B와 최종 평가

`stage_b.ContextDecisionModel`의 `decide(document)`는 A 후보마다 하나의 `PrivacyDecision`을 반환하는 계약입니다. B 학습은 gold 후보로 시작하고, 문맥·유형·신뢰 메타데이터로 판단해야 합니다. 생성기의 숨겨진 관계 정답이나 reason_code를 입력으로 넘기지 않습니다.

현재 제공하는 기준모델은 모든 A 후보를 MASK합니다.

```bash
kpii mask --input data/examples/test.jsonl \
  --output runs/example/externally-shareable.jsonl \
  --mask-records runs/example/internal-mask-records.jsonl
kpii evaluate-masking --gold data/examples/test.jsonl \
  --masks runs/example/internal-mask-records.jsonl \
  --output runs/example/masking-metrics.json
```

`--output`에는 doc_id와 masked_text만 저장합니다. 선택적인 `--mask-records`는 원문·구간을 포함하는 내부 평가 sidecar이며 별도 경로여야 합니다. KDPII 원본에는 MASK/KEEP 정답이 없으므로 해당 데이터로 통합 privacy 평가를 시도하면 오류로 중단합니다.

통합 누출은 gold MASK 표현 중 한 글자라도 남은 비율, 과잉 마스킹은 gold KEEP 표현 중 한 글자라도 가린 비율입니다. A의 미탐도 분모와 누출에 포함합니다. 새 `calibrate-b`/`evaluate-ab`는 같은 A에 대해 전체 MASK와 학습 B를 비교하며 KEEP 임계값을 dev에서 선택해 test에 고정합니다. 고정 A의 후보를 선택하기만 하는 B는 A의 누락을 복구할 수 없습니다.

실제 데이터 검수 후 다음 스크립트가 B 학습부터 통합 평가까지 실행합니다. 현재는 실행하지 않았습니다.

```bash
bash scripts/train_eval_b_gpu5.sh data/private/b-v1
```

입력 파일 구성과 검수 명령은 [B_WORKFLOW.md](docs/B_WORKFLOW.md)를 따릅니다. `annotation_templates/b`에는 빈 CSV 헤더와 신뢰 메타데이터 양식만 있습니다. 실제 패킷은 `kpii export-b-review`로 생성합니다.

## V4 보완 실험 준비

고정 A 뒤의 주소·직위 구간 보정, 문장 초점 B 입력, 위험한 KEEP 반례를 추가한
V4 실행 방법은 [B_V4_EXPERIMENT.md](docs/B_V4_EXPERIMENT.md)를 확인합니다.
`scripts/train_eval_b_v4_gpu5.sh --check-only`는 학습 없이 데이터와 입력 길이를 점검합니다.
기존 V3 실험과 스크립트는 그대로 보존하며 새 실험은 명시적으로 실행해야 합니다.

## 검증 산출물과 다음 작업

GPU 5 전용 전체 학습·best 체크포인트 선택·테스트 평가를 한 번에 실행하려면 다음 스크립트를 사용합니다. 이 스크립트를 명시적으로 실행할 때 전체 학습이 시작됩니다.

```bash
bash /workspace/korean-pii-mvp/scripts/train_eval_a_gpu5.sh
```

VRAM 측정은 독립적으로 실행할 수 있습니다. FP32·AdamW·학습 batch 4·평가 batch 8·512토큰의 고정 길이 배치를 사용하며, 정확도 평가나 전체 데이터 학습은 수행하지 않습니다.

```bash
source /workspace/korean-pii-mvp/scripts/activate_gpu5.sh
cd /workspace/korean-pii-mvp
python scripts/profile_gpu5_vram.py --output logs/gpu5-vram-profile.json
```

`docs/VALIDATION.md`에 초기 서버 검증 범위를 기록합니다. B 추가분은 `docs/B_VALIDATION.md`를 확인합니다. 소수 step의 과거 smoke 체크포인트는 코드 실행 확인용이며 성능 모델로 사용하지 않습니다. 다음 순서는 실제 업무 데이터 주석·검수 → B 학습 → 고정 A 후보 dev 임계값 → 독립 test 평가입니다. 업무 A 재학습이나 정규식 결합으로 후보가 바뀌면 B 통합 임계값도 다시 검증합니다.
