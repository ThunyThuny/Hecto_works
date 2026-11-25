# 컨테이너 전용 환경 및 GPU 5 VRAM 측정

2026-10-19 H200_7776 컨테이너에서 구성·검증했습니다.

## 실행 환경

- 프로젝트: `/workspace/korean-pii-mvp`
- 독립 가상환경: `/workspace/kpii-env`
- Python 3.10.12 / PyTorch 2.5.1+cu124 / Transformers 4.57.6
- tokenizers 0.22.2 / safetensors 0.8.0 / numpy 1.26.4
- `include-system-site-packages=false`; torch·Transformers·numpy가 모두 이 환경 내부에 설치돼 있습니다.
- 다른 Python 환경의 site-packages 참조 없이 의존성 검사와 78개 단위 테스트를 통과했습니다.
- 재설치용 패키지 목록은 프로젝트의 `requirements-gpu5.lock.txt`에 저장했습니다.

다음 한 줄이 가상환경과 캐시 경로를 설정하고, 물리 GPU 5의 UUID를 확인해 그 장치만 노출합니다.

```bash
source /workspace/korean-pii-mvp/scripts/activate_gpu5.sh
```

PyTorch 내부에서는 보이는 장치가 하나이므로 물리 GPU 5가 논리 `cuda:0`입니다. 학습 CLI에는 `--device cuda`를 지정합니다.

## 전체 A 학습·테스트 실행

```bash
bash /workspace/korean-pii-mvp/scripts/train_eval_a_gpu5.sh
```

이 명령을 실행하면 전체 KDPII 학습 데이터로 3 epoch 학습하고, 검증 exact span/type micro-F1 최고 체크포인트로 테스트 추론·평가까지 순서대로 수행합니다. 데이터 준비와 메모리 측정을 수행하면서 전체 학습을 자동으로 시작하지는 않았습니다.

## VRAM 실측

실제 A.X 인코더와 native 33개 유형의 BIO 헤드를 로딩하고 **FP32·AdamW·학습 배치 4·평가 배치 8·512토큰**의 고정 길이 합성 토큰 배치로 측정했습니다. 첫 optimizer step, 추가 3 step, optimizer와 마지막 gradient를 유지한 평가 2회가 포함됩니다. 정확도 측정용 데이터가 아닙니다.

| 항목 | 관측 최대 |
|---|---:|
| PyTorch tensor allocation | 3.68 GiB |
| PyTorch allocator reservation | 4.63 GiB |
| nvidia-smi GPU 5 전체 점유량 | 5,485 MiB = 5.36 GiB |
| 동일 설정의 권장 가용 VRAM | 8 GiB 이상 |

GPU 5는 H200이며 표시된 총 VRAM은 143,771 MiB, 약 140.4 GiB입니다. 측정 전·후 GPU 5 사용량은 0 MiB로 확인했습니다. 전체 장치 샘플에는 단일 compute PID만 관측됐습니다. 컨테이너 PID와 호스트 PID가 달라 프로세스별 메모리 일치는 사용할 수 없었으며, tensor/allocator 통계와 GPU 전체 점유량을 구분해 기록했습니다.

권장치는 관측값에 최소 2 GiB 또는 30% 여유를 추가한 값입니다. 전체 데이터 학습을 끝까지 실행해 측정한 수치는 아니며, 토큰 길이·배치 크기·정밀도·attention 구현을 바꾸면 다시 측정해야 합니다.

원본 측정 결과: `logs/gpu5-vram-profile.json`. 환경 검사 결과: `logs/gpu5-environment.json`.

```bash
source /workspace/korean-pii-mvp/scripts/activate_gpu5.sh
cd /workspace/korean-pii-mvp
python scripts/profile_gpu5_vram.py --output logs/gpu5-vram-profile.json
```
