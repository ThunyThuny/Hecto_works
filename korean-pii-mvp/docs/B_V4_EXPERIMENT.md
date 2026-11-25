# 고정 A + 후보 구간 보정 + B V2 실험

이 변경은 기존 A 체크포인트를 재학습하거나 수정하지 않는다. 기존 V3 데이터·실험
결과·실행 스크립트도 보존한다. 새 데이터는 `synthetic_data_bulk3000_v4`에 저장한다.
미검수 합성 라벨이며 `PENDING`, `human_reviewed=false`를 유지한다.

## 변경 내용

1. `refine-a-candidates`: 원문과 A 구간만 사용해 같은 행의 도로명 주소·번지·제한된
   층/호와 사전 내 복합 직위를 보정한다. 다른 유형을 침범하거나 원래 탐지 범위를
   줄이지 않는다. 원본 A 후보는 별도 파일로 보관하고 보정 설정을 해시로 연결한다.
   A가 전혀 찾지 못한 표현이나 사전 밖 표현을 새로 탐지하지는 않는다.
2. `target_focus_v2`: 후보가 속한 문장과 값을 앞에 다시 보여준다. 전체 원문과
   정확한 후보 오프셋 마커도 유지한다. 라벨·사유·생성기 관계 정답은 읽지 않는다.
   입력 한도를 넘기면 문맥을 자르지 않고 학습 전 검사에서 거부한다.
3. B 학습: MASK 클래스 비용 3, label smoothing 0.02, 학습률 1e-5,
   최대 3 epoch, early stopping patience 1. 검증의 일반 CE 최소 체크포인트를 선택한다.
   이는 개선을 검증하기 위한 초기 설정이며 성능 향상을 보장하지 않는다.
4. [V4 데이터](SYNTHETIC_V4.md): 개인 경력과 공개 안내에 같은 직위·부서가 등장하는
   반례, 현실적인 가공 주소, 분할별 새 값·문장·ID를 사용한다. 개인/공용 연락처는
   분리한다. V3 오류 분석 후 생성한 합성 holdout이며 실제 문서 성능 근거는 아니다.

## 준비 상태만 확인

```bash
cd /workspace/korean-pii-mvp
source /workspace/kpii-env/bin/activate
bash scripts/train_eval_b_v4_gpu5.sh --check-only \
  /workspace/nfs/datasets/dylee/PII/synthetic_data_bulk3000_v4
```

파일 해시·분할·정책·A 잠금·V2 토큰 길이를 검사한다. 저장된 토크나이저만 읽으며
모델 가중치 적재, GPU 선택, 추론, 학습, 실험 출력 디렉터리 생성은 수행하지 않는다.

## 사용자가 나중에 실행할 재학습·평가

```bash
cd /workspace/korean-pii-mvp
source /workspace/kpii-env/bin/activate
bash scripts/train_eval_b_v4_gpu5.sh \
  /workspace/nfs/datasets/dylee/PII/synthetic_data_bulk3000_v4
```

새 `runs/b-synthetic-v4-gpu5-날짜_시간`을 만든다. 기존 출력 경로가 있으면 중단한다.
스크립트는 GPU 5의 UUID만 노출하고 A 파일 잠금을 전후 검증한다. 실행 순서는
B 학습 → valid A 추론·구간 보정·B 점수 → valid 임계값 선택 → 고정 test 평가다.
임계값 조건은 기존과 동일하게 AllMask 대비 추가 누출 0이다. 조건을 통과하지 못하면
KEEP을 비활성화한다. B 가중치나 보정 규칙이 달라지면 임계값을 다시 선택한다.

## 결과를 읽는 순서

* `model/b_manifest.json`: epoch별 학습/검증 loss, 최적 epoch, 입력·loss 설정.
* `valid-candidate-diagnostics.json`, `test-candidate-diagnostics.json`: 원래 A와
  A+보정의 strict AllMask를 비교한다. 공백만 남음·비공백 부분 누락·전체 미탐,
  KEEP 후보의 구간/유형 완전 일치와 승인 일치 수를 구분한다. 주석 밖 가려진 문자
  수도 기록한다. 공백 오류도 기존 strict 누출에서 제외하지 않는다.
* `valid-b-classification.json`, `test-b-classification.json`: 정답 후보를 넣은 B의
  argmax 정확도·KEEP F1·잘못된 KEEP 비율. 최종 A+B 지표와 별개다.
* `calibration.json`: valid에서 선택한 임계값과 KEEP 활성화 여부.
* `test-evaluation/ab-evaluation.json`: **보정 후보를 모두 MASK**하는 기준선과
  **동일한 보정 후보에 B를 적용**한 결과를 비교한다. 원래 A와의 차이는 앞의
  candidate diagnostics로 따로 확인한다. 후보 우주가 다른 기준선을 섞지 않는다.

검수용 50문서의 `review.html`과 `review_sample_50/mentions.csv`를 먼저 읽는다.
자동 점검 통과는 자연스러움이나 사람 검수 완료를 의미하지 않는다. 기존 test는
오류 분석·회귀 비교용으로 보존하며 새 test를 학습/임계값 선택에 사용하지 않는다.
