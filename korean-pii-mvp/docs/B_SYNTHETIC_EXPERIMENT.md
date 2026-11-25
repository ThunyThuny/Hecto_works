# 미검수 합성라벨로 B 학습·평가하기

현재 3,000문서는 생성기 라벨과 AI 표본 검토를 갖춘 PENDING 데이터입니다.
이 실험 경로는 원본 검수 기록을 보존하고 `unreviewed_synthetic_experiment`,
`human_reviewed=false`, 원본 패킷 해시를 학습·임계값·평가 산출물에 기록합니다.
일반 검수 데이터 경로의 APPROVED 요구는 유지합니다.

7776 컨테이너 안에서 다음을 실행합니다. 첫 명령은 CPU 검사만 수행하며 모델·GPU를
로드하거나 실행 디렉터리를 만들지 않습니다.

```bash
cd /workspace/korean-pii-mvp
source /workspace/kpii-env/bin/activate

bash scripts/train_eval_b_synthetic_experiment_gpu5.sh --check-only \
  /workspace/nfs/datasets/dylee/PII/synthetic_data_bulk3000_v3
```

다음 명령은 실제 B 학습을 시작하고, 고정 A를 사용한 임계값 선택과 테스트 평가까지
순서대로 진행합니다. 이번 코드 구성·확인 작업에서는 실행하지 않았습니다.

```bash
cd /workspace/korean-pii-mvp
source /workspace/kpii-env/bin/activate

bash scripts/train_eval_b_synthetic_experiment_gpu5.sh \
  /workspace/nfs/datasets/dylee/PII/synthetic_data_bulk3000_v3
```

출력 위치를 직접 지정하려면 마지막 인자로 새 경로를 추가합니다. 기존 실행 경로,
원본 데이터 안쪽, A 체크포인트 안쪽은 거부합니다.

```bash
bash scripts/train_eval_b_synthetic_experiment_gpu5.sh \
  /workspace/nfs/datasets/dylee/PII/synthetic_data_bulk3000_v3 \
  /workspace/korean-pii-mvp/runs/b-synthetic-experiment-gpu5-v1
```

- GPU: 물리 5번의 UUID만 노출. torch에는 논리 `cuda:0`으로 보입니다.
- A: `runs/a-native-gpu5-20261019_020722/best`의 파일을 유지합니다. B는 A 인코더를
  새 모델로 복사해 학습하고, A 자체는 후보 추론에만 사용합니다. A 자산과 잠금은
  시작·종료 시 해시를 검증합니다.
- B: 3 epochs, train batch4, eval batch8, max length512, learning rate3e-5,
  weight decay0.01, seed42. train2,000문서의 문맥 후보10,500개를 사용합니다.
- 선택: valid500문서의 후보 분류 loss로 `model/best`를 선택합니다.
- 통합: 고정 A의 valid 후보와 B 점수로 KEEP 임계값을 결정하고, 그 뒤 test500문서를
  평가합니다. test는 모델·임계값 선택에 사용하지 않습니다.

기본 실행 폴더는 `runs/b-synthetic-experiment-gpu5-YYYYMMDD_HHMMSS`이며 시각은
Asia/Seoul입니다. 주요 산출물은 다음과 같습니다.

```text
experiment_manifest.json             # 미검수 합성 실험과 원본 해시
preflight.json                       # 전체 패킷·분할 검사
data/                                # 실행별 분할 사본, PENDING 기록 유지
train.log                            # B 학습 로그
model/best/                          # dev loss 최적 B
model/best/b_manifest.json            # 학습 근거·후보 수·검증 이력
calibration.json                     # valid에서 정한 임계값
test-evaluation/ab-evaluation.json    # 고정 A+B와 AllMask 비교
```

실험 옵션은 해시가 맞고 자동 검사를 통과하는 원본 V3 패킷과 일치하는 문서에만
적용합니다. 실제·혼합·출처 불명 데이터나 수정된 정답을 이 옵션으로 통과시키지 않습니다.
수정된 검수본은 기존 [검수 경로](SYNTHETIC_WORKFLOW.md)를 사용합니다.

결과는 합성 정답에 대한 실험 성능입니다. 실제 업무 문서 성능은 별도로 평가합니다.
