# 알고리즘 및 구현 인수인계

Built with Stability AI

작성일: 2026-10-08. 목적: **mask + SD-Turbo 방식으로 실제 객체 교체가 가능한지 재현하고, 실행 가능한 통합 코드를 인계한다.**

실행 순서는 [README_KO.md](README_KO.md)를 따릅니다. 배포본에는 베이스 모델과 학습된 어댑터가 포함되어 있습니다.

## 1. 시스템 구성

| 항목 | 구현 |
|---|---|
| 생성 모델 | SD-Turbo, 고정 revision `b261bac6fd2cf515557d5d0707481eafa0485ec2` |
| 배포 어댑터 | A best, `pix2pix_turbo_mask_v5`, 추가 학습 1200 step |
| UNet 입력 | 이미지 latent 4채널 + 편집 마스크 1채널 |
| 학습 모듈 | UNet LoRA rank 8, VAE LoRA rank 4, 5채널 conv_in, VAE skip conv |
| 추론 단계 | timestep 999에서 UNet 1회, DDPM scheduler step 1회 |
| 객체 ROI | 편집 마스크 bbox 중심의 정사각형, context 1.5, 512×512 |
| 소스 조건 | 편집 영역 내부 RGB를 정규화 공간에서 0.5배로 줄임 |
| 복원 | VAE skip 연결, 편집 영역의 skip 차단, 원래 좌표에 생성 ROI 복원 |
| 최종 합성 | 객체 core 완전 생성 + 편집 마스크 안쪽 거리 기반 경계 feather |

참고 구현은 [GaParmar/img2img-turbo](https://github.com/GaParmar/img2img-turbo)의 commit `86f54146590ffb4543c8cf85b5a36657da670924`입니다. VAE forward 함수의 출처와 MIT 라이선스는 `code/vendor/`에 보존했습니다. 본 구현은 마스크 채널과 마스크 제어 skip을 추가한 커스텀 학습 모델이며, 공식 SD-Turbo를 그대로 호출하는 일반 inpainting pipeline이 아닙니다.

## 2. 알고리즘 흐름

```text
입력 RGB I + 객체 마스크 O + 목표 문장 P
  ↓ O를 12px 팽창 → 편집 마스크 E
  ↓ E의 bbox × 1.5 주변 영역을 정사각형 crop → 512×512
  ↓ X = normalize(I); Xc = X·(1−E) + 0.5·X·E
  ↓ VAE encoder → latent z + skip feature
  ↓ m = nearest_resize(E, 64×64)
  ↓ UNet([z,m], timestep=999, text_encoder(P)) → epsilon
  ↓ DDPM scheduler.step(epsilon, 999, z) → z′
  ↓ 각 skip feature × (1−해당 크기의 E)
  ↓ VAE decoder(z′, masked skips) → 생성 crop
  ↓ crop을 입력 좌표에 복원 → raw R
  ↓ alpha 합성 → 최종 RGB F
```

확산 denoiser는 한 번 호출하지만 텍스트 인코딩, VAE 인코딩/디코딩, crop과 합성 연산도 수행합니다. 전체 프로그램이 단일 연산이라는 의미는 아닙니다.

추론은 VAE posterior의 `mode()`를 사용합니다. DDPM generator는 seed 0으로 고정됩니다. CFG의 두 분기나 negative prompt는 사용하지 않습니다. 같은 장치·버전에서 재현을 확인했지만 다른 CUDA 라이브러리나 GPU의 픽셀 일치까지 보장하지는 않습니다.

## 3. 마스크와 dilation의 역할

두 마스크를 구분합니다.

| 마스크 | 역할 |
|---|---|
| O: 객체 마스크 | 교체할 원래 객체 영역. 최종 결과에서 alpha=1 |
| E: 편집 마스크 | 생성 모델에 전달하는 영역. 원래 객체의 주변 공간을 포함 |

`dilation(O, 12)`는 25×25 **정사각형 MaxFilter**입니다. 원형 팽창이나 학습형 segmentation이 아닙니다. 기존 객체 경계의 잔상을 줄이고 새로운 털·윤곽에 약간의 공간을 제공합니다. 12픽셀은 긴 변 512인 실험에서 사용한 경험값이며 최적 반경을 찾은 실험은 없습니다.

편집 영역이 지나치게 작으면 새 객체가 잘리거나 기존 윤곽이 남습니다. 지나치게 크면 주변 물체까지 변경할 수 있습니다. 그림자나 새 객체가 차지해야 할 추가 공간이 필요하면 E를 직접 설계해야 합니다. **보존 대상은 E 밖이며 O 밖 전체가 아닙니다.** O와 E 사이의 고리 영역은 편집할 수 있습니다.

모델 안에서는 5번째 latent 채널이 E를 전달합니다. 동시에 VAE skip feature에서 E 내부의 원래 객체 특징을 차단합니다. 이미지를 마지막에 붙이는 것만으로 마스크를 처리한 구현이 아닙니다.

## 4. mask 후처리: 부드러운 alpha 합성

`code/mask_postprocess.py`가 실제 구현입니다. E를 1픽셀 0 패딩한 뒤 내부 Euclidean distance transform을 계산합니다.

```text
d = distance_transform_inside(E)
t = clip((d − 0.5) / feather_px, 0, 1)
alpha = t² · (3 − 2t)
alpha[O] = 1
alpha[¬E] = 0
F = round(I · (1−alpha) + R · alpha)
F[¬E] = I[¬E]
```

기본 feather는 6픽셀입니다. 0이면 hard composite입니다. 객체 core는 원본과 섞지 않으므로 원래 고양이 등의 잔상을 줄입니다. 바깥 고리에서는 완만한 alpha 전이가 생깁니다. E를 객체 경계와 똑같이 지정하면 core alpha=1 때문에 객체 외곽에 feather 공간이 거의 없습니다.

이것은 **거리 기반 alpha feathering**이며 털·머리카락의 투명도를 추정하는 학습형 image matting은 아닙니다. 경계 연결을 개선하지만 잘못된 머리 수, 다리, 차체 구조, 시점, 색상이나 그림자를 교정하지 않습니다.

## 5. 코드와 모델 의존성

| 파일/폴더 | 담당 기능 |
|---|---|
| `demo.py` | 인자 검사, 샘플/사용자 입력, 실제 UNet trace, 합성, 결과 갤러리 |
| `run_demo.sh` | 폴더 위치와 Python 실행 파일을 지정하는 실행 래퍼 |
| `code/img2img_runtime.py` | 5채널 UNet, LoRA, VAE skip, 단일 diffusion forward |
| `code/opt_runtime.py` | A 어댑터 로딩, crop/restore, CPU offload, 추론 |
| `code/mask_postprocess.py` | GPU 없이 실행할 수 있는 픽셀 보존 합성 |
| `models/sd-turbo/` | 완전한 로컬 모델 snapshot, revision과 라이선스 포함 |
| `weights/adapter_A.pt` | 배포 학습 가중치 |
| `samples/` | 입력·객체 마스크·기준 결과 6개 및 prompt/provenance |
| `outputs/verified_offload/` | 통합 코드로 다시 생성한 결과와 실행 검증 |
| `verification/` | 실제 터미널 실행 로그 및 후처리 확인 기록 |
| `SHA256.json`, `verify_bundle.py` | 파일 무결성 인수 검사 |

본 패키지는 기존 실험 폴더나 특정 머신의 절대 경로를 import하지 않습니다. 과거 실험의 모델/가중치/예제를 패키지 내부로 복사했습니다. 출력 폴더가 이미 있으면 덮어쓰지 않고 오류를 냅니다. 일반 사용자에게는 `demo.py` 또는 `run_demo.sh`가 진입점입니다.

모델 파라미터는 FP32이며 forward는 CUDA BF16 autocast입니다. 기본 offload는 파라미터를 FP32 상태로 CPU에 보관하고 실행 시 모듈별로 GPU로 옮깁니다. dtype을 낮춘 다른 가중치로 바꾼 방식이 아닙니다. 로딩/추론 peak memory와 GPU 이름은 실행 기록에 저장합니다.

어댑터 SHA256:

```text
69d4ddadb02b3241aca43e79444971cd51bfe689881f9a38939c89d013357d79
```

로더는 어댑터 형식, base revision, adapter key 집합을 확인하고 진입점은 배포 가중치 SHA256까지 검사합니다. 다른 실험 가중치를 임의로 넣으면 거부합니다. 파일 해시는 변경을 탐지하는 용도이며 공개 배포용 서명은 아닙니다.

## 6. 학습 이력과 성능 해석

이번 A는 이전 v5 가중치에서 **추가로** 미세조정했습니다. 원본 SD-Turbo에서 156쌍만 학습한 결과가 아닙니다.

| 항목 | 기록 |
|---|---|
| 추가 학습 데이터 | 기존 79쌍 + 신규 77쌍 = 독립 source 156쌍, source/target RGB 총 312장 |
| 검증 / 신규 테스트 | 12쌍 / 독립 source 40장 |
| 신규 source | Open Images V7 train |
| target | 선별한 BrushNet + Realistic Vision V6 합성 target; 실촬영 쌍 아님 |
| 기존 v5 초기화 SHA256 | `473afa7509f3bc32ffe9e4d1c83e6cae79d65907c91c9cbd82ad500e307d8b03` |
| 추가 학습 | 1200 step, lr 2e-5, seed 20261003, 클래스 균형 샘플링 |
| 학습 가능 파라미터 | 9,496,776 |
| 선택 기준 | 300 step마다 검증 ROI LPIPS 평가; best는 1200 step |
| 손실 | mask 정규화 L1 + LPIPS + CLIP 목표/소스 분리 + ROI GAN |

최소 학습 이미지 수를 결정하는 learning curve는 아직 없습니다. 156쌍을 신규 모델 학습의 최소 요건으로 제시하면 안 됩니다.

40장 신규 테스트는 네 방향 각 10장입니다. A의 엄격한 육안 합격: 고양이→개 5/10, 자동차→소형 픽업 1/10, 개→고양이 0/10, 트럭→세단 0/10. 총 6/40입니다. 동일 테스트의 BrushNet은 19/40이었습니다. 육안 평가는 방법을 볼 수 있는 실험자 평가이며 독립적인 blind benchmark가 아닙니다.

따라서 **단일 단계 국소 객체 교체의 실행 가능성과 성공 사례는 증명했지만, 전반적인 품질 우위나 모든 클래스의 지원은 증명하지 않았습니다.** 패키지의 6개 예제는 그 성공 사례입니다. 6개만 실행하여 얻은 6/6을 일반 성공률로 보고하지 마세요.

## 7. 어떤 입력에서 자연스러운가

검증 사례에서는 단일 객체, 분명한 마스크, 비슷한 자세와 크기의 교체가 유리했습니다. 고양이→개는 기존 포즈를 유지하고 주변 공간이 있는 경우 성공했습니다. 자동차→소형 픽업은 차체 비율이 크게 바뀌지 않는 사례에서 성공했습니다.

큰 트럭→작은 세단처럼 공간과 구조가 크게 달라지거나, 가려짐·복잡한 윤곽·다중 객체가 있는 경우에는 잔상, 잘린 바퀴, 중복된 머리, 앞뒤 방향 변화 등이 남았습니다. 이는 관찰한 경향이며 정량화된 성공 조건이나 보장된 규칙은 아닙니다.

## 8. 인수 완료 조건과 증거

문서의 명령을 **`/tmp` 작업 디렉터리, 빈 PYTHONPATH, HF 오프라인 모드**에서 실행했습니다. 기존 실험 폴더를 작업 경로에 넣지 않았습니다. `verification/offload.log`와 `outputs/verified_offload/verification.json`이 실행 증거입니다.

각 예제의 기준:

- 실제 UNet 호출 1회, 입력 `[1,5,64,64]`, timestep 999.
- 5번째 입력 채널이 nearest-resize 편집 마스크와 정확히 일치.
- 최종 PNG의 E 외부 변경 픽셀 수 0.
- O 내부가 생성 raw와 정확히 일치.
- 통합 offload 출력이 이전 성공 기준 PNG와 픽셀 단위 일치하는지 별도 기록.

무결성은 `python3 verify_bundle.py`로 확인합니다. 생성 품질은 `browse.html`의 입력/결과를 직접 함께 확인해야 합니다. 위 기능 검사는 자연스러움 자체의 자동 점수는 아닙니다.

## 9. 장애 대응과 다음 개발

| 증상 | 조치 |
|---|---|
| CUDA를 찾지 못함 | CUDA PyTorch, 드라이버, GPU 접근 권한 확인. Mac에서는 결과만 열람 |
| CUDA OOM | 기본 cpu-offload 사용, 여유 GPU 확보. 다른 작업을 종료하지 않음 |
| 모델/가중치 누락 | 배포 폴더 전체 복사, `verify_bundle.py` 실행 |
| checkpoint/revision 불일치 | 배포한 A 어댑터와 고정 base를 한 쌍으로 사용 |
| 파일 존재 오류 | 새 출력 폴더 지정 |
| 마스크 크기/포함 오류 | 입력과 좌표 일치, 흰색 객체, O가 E 안에 있는지 확인 |
| 객체 잔상 또는 경계 절단 | O를 정확히 수정하고 E에 추가 공간 확보. 무조건 큰 팽창으로 해결하지 않음 |
| 형태/시점 오류 | 후처리로 해결되지 않음. 해당 결과를 실패로 기록하고 학습 데이터/구조 감독 개선 |

개선 우선순위: 클래스별 균형과 자세 다양성 확대, 어려운 source/target 쌍 보강, 최종 합성 이미지와 경계 감독, 구조·시점 보존 감독, 독립 평가 및 여러 seed 검증. 추가 학습과 비교 실험을 수행하기 전에는 개선된 성능이라고 보고하지 않습니다.

## 10. 배포 구조 및 출처

- 패키지 루트: `mask_sd_turbo_demo_20261008/`; GPU 머신의 원하는 위치에 배치합니다.
- 실행 진입점: `run_demo.sh` 또는 `demo.py`.
- 모델과 가중치: `models/sd-turbo/`, `weights/adapter_A.pt`.
- 실행 기록과 결과: `verification/`, `outputs/verified_offload/`.
- 모든 모델·샘플 경로는 패키지 루트를 기준으로 해석하며, 특정 계정이나 서버 경로에 의존하지 않습니다.
- img2img-turbo의 MIT 파일: `code/vendor/LICENSE.img2img-turbo`.
- SD-Turbo의 배포 snapshot에 동봉된 라이선스: `models/sd-turbo/LICENSE.md`; 모델 카드: `models/sd-turbo/README.md`.
- 예제 원본 URL: `samples/manifest.json`. Open Images의 source provenance를 유지했습니다.

이 문서는 동봉된 고정 모델 snapshot의 라이선스 파일을 참조합니다. 별도의 법률 해석이나 새로운 상용 사용 허가는 제공하지 않습니다.
