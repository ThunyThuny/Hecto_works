# Mask + SD-Turbo + LoRA 객체 교체 데모

Built with Stability AI

**Agent 시작점:** 먼저 [AGENTS.md](AGENTS.md)를 읽고, 이 실행 가이드와 [HANDOFF_KO.md](HANDOFF_KO.md)를 순서대로 확인하세요.

**입력 이미지 + 객체 마스크 + 목표 객체 문장 → 한 번의 UNet 추론 → 경계가 부드러운 국소 교체 결과.**

이 패키지는 학습된 A 어댑터, SD-Turbo 베이스 모델, 성공 예제 6개, 추론 코드와 실제 실행 결과를 포함합니다. 별도 학습이나 모델 다운로드 없이 오프라인 추론이 가능합니다. CUDA 및 BF16 연산을 지원하는 NVIDIA GPU가 필요합니다. 알고리즘과 학습 이력은 [HANDOFF_KO.md](HANDOFF_KO.md)에 있습니다.

## 1. GPU 머신에서 실행

완전한 패키지를 GPU 머신의 원하는 위치에 복사한 뒤, 의존성이 설치된 Python 환경에서 실행합니다. 환경이 없다면 5절의 설치 절차를 먼저 진행하세요.

```bash
cd mask_sd_turbo_demo_20261008
export DEMO_PYTHON=python3
bash run_demo.sh --check
bash run_demo.sh --all-samples --output outputs/my_first_run
```

`outputs/my_first_run/browse.html`을 브라우저로 열면 입력, 두 종류의 마스크, 합성 알파와 결과를 볼 수 있습니다. 출력 폴더는 새 이름이어야 합니다. 다시 실행할 때 `outputs/my_second_run`처럼 이름을 바꾸세요.

한 개만 실행하려면:

```bash
bash run_demo.sh --sample 048_c9a6b76a3203fae8 --output outputs/one_case
```

기본 모드는 `cpu-offload`입니다. FP32 파라미터와 BF16 autocast를 유지하면서 모듈별 가중치를 CPU와 GPU 사이에서 전송합니다. 실제 연산은 GPU에서 수행됩니다. 전송 때문에 속도가 느려지지만, 검증한 환경에서는 원래 CUDA 실행 결과와 6개 모두 픽셀이 일치했습니다. 실제 메모리와 시간은 [검증 결과](outputs/verified_offload/verification.json)에 있습니다.

메모리가 충분한 경우 `--device-mode cuda`로 모델 전체를 GPU에 올릴 수 있습니다. 통합 패키지의 인수 검증은 기본 offload 모드로 수행했습니다. 기존 CUDA 실험의 속도와 현재 공유 GPU의 offload 속도는 직접 비교하지 마세요.

## 2. 자신의 이미지로 실행

```bash
bash run_demo.sh \
  --input /absolute/path/image.png \
  --object-mask /absolute/path/object_mask.png \
  --prompt "a realistic photo of a dog, in the same pose and location, consistent lighting and perspective" \
  --dilate-px 12 --feather-px 6 \
  --output outputs/custom_case
```

마스크는 **흰색=교체할 객체, 검은색=배경**이며 입력 이미지와 가로·세로 크기가 같아야 합니다. 값 128 이상을 흰색으로 이진화합니다. 자동 분할 기능은 포함하지 않았습니다. EXIF 회전이나 크기 변경은 이미지와 마스크에 동일하게 적용한 뒤 입력하세요.

기본적으로 객체 마스크를 12픽셀 팽창해 편집 마스크를 만듭니다. 직접 편집 마스크를 준비한 경우 `--edit-mask /absolute/path/edit_mask.png`를 추가하세요. 이 경우 팽창을 적용하지 않습니다. 편집 마스크는 객체 마스크 전체를 포함해야 합니다. 12/6픽셀은 입력 이미지 좌표 기준이며, HD 이미지에서는 별도로 조정해야 합니다.

입력 해상도는 그대로 유지하고 객체 주변만 512×512로 변환하여 추론합니다. 데모 예제는 긴 변이 512픽셀입니다. 임의 해상도에서 실행 가능한 것과 동일한 품질을 보장하는 것은 다릅니다.

## 3. 출력 구조

```text
outputs/<run>/
  browse.html                 결과 확인 화면
  verification.json           전체 환경·실행·검증 기록
  <sample_id>/
    input.png                 실제 입력 RGB
    object_mask.png           원본 객체 마스크
    edit_mask.png             팽창된 편집 마스크
    raw.png                   합성 전 생성 결과
    alpha.png                 시각화용 8비트 알파
    output.png                최종 교체 결과
    metadata.json             prompt, UNet trace, 시간, 픽셀 검사
```

합성은 내부 float 알파로 계산합니다. `alpha.png`는 반올림된 시각화이므로 그 파일로 재합성할 경우 일부 픽셀이 달라질 수 있습니다. 원본 마스크와 아래 후처리 코드를 사용하면 동일한 합성이 가능합니다.

## 4. 생성된 결과에 후처리만 적용

이 단계는 GPU 없이 실행할 수 있습니다.

```bash
"$DEMO_PYTHON" code/mask_postprocess.py \
  --input samples/048_c9a6b76a3203fae8/input.png \
  --generated outputs/verified_offload/048_c9a6b76a3203fae8/raw.png \
  --object-mask samples/048_c9a6b76a3203fae8/object_mask.png \
  --edit-mask outputs/verified_offload/048_c9a6b76a3203fae8/edit_mask.png \
  --feather-px 6 --output outputs/postprocess_only.png
```

최종 출력은 PNG를 사용하세요. JPEG 압축은 편집 영역 밖의 픽셀도 변경하므로 픽셀 보존 검증에 적합하지 않습니다.

## 5. GPU 머신의 실행 환경 준비

`mask_sd_turbo_demo_20261008` **전체 폴더**를 복사하세요. `models/`, `weights/`, `samples/`가 필요합니다. Python 파일만 복사하면 모델이 없습니다. 원래 실험 폴더를 참조하는 import, 모델 심볼릭 링크, 네트워크 추론 호출은 없습니다.

검증한 소프트웨어 환경: Linux, Python 3.10.12, torch 2.11.0+cu128. CUDA 및 BF16을 지원하는 NVIDIA GPU와 호환 드라이버가 필요합니다. Python 라이브러리 버전은 `requirements.txt`에 고정했습니다. 기존 호환 환경을 먼저 사용하세요. 다른 서버에서 환경이 없는 경우에만 다음과 같이 설치할 수 있습니다. 이 설치 절차 자체를 새 가상환경에서 다시 검증하지는 않았으며, 실행 검증은 기존 환경에서 수행했습니다.

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
export DEMO_PYTHON="$PWD/.venv/bin/python"
bash run_demo.sh --check
bash run_demo.sh --all-samples --output outputs/new_server
```

모델 추론은 오프라인입니다. 위 의존성 설치에는 인터넷이 필요합니다. 생성 추론은 CUDA GPU에서 실행하며, GPU 없는 머신에서도 문서와 생성 결과를 열람할 수 있습니다.

인계에는 모델과 가중치가 포함된 전체 배포 패키지를 사용하세요. 코드와 문서만 있는 개발 사본으로는 추론할 수 없습니다.

## 6. 인수 검사

```bash
python3 verify_bundle.py
```

배포 파일 SHA256을 확인합니다. 새로 만든 출력 폴더는 배포 manifest에 없으므로 검사 대상이 아닙니다. 추론 성공 여부는 각 실행의 `verification.json`을 확인하세요. `all_verified=true`, `outside_changed_pixels=0`, 한 개의 `unet_trace`와 `mask_channel_exact=true`가 기준입니다.

6개 예제는 선별한 성공 사례입니다. 전체 40장 테스트에서 A의 엄격한 육안 합격은 6장이었으며, 임의 객체와 모든 입력에서 성공한다고 주장하지 않습니다.
