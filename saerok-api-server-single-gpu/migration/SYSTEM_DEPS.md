# 시스템 의존성 조사 (Step 2)

## OS / GPU / 드라이버
| 항목 | 값 |
|------|-----|
| OS | Ubuntu 22.04.5 LTS (Jammy) |
| GPU | NVIDIA L40S × 3 (각 46068 MiB ≈ 46GB) |
| 드라이버 | 570.195.03 |
| 드라이버 CUDA | 12.8 (cu128 torch 지원 OK) |

> 참고: CLAUDE.md 는 "L40S 48GB" 라고 적었으나 실측 가용 VRAM 은 약 46GB.
> 조사 시점 GPU2 에 32GB 사용 중(Gemma 로드됨), GPU0 6.8GB.

## ⚠️ CUDA 버전 두 개 공존 (이관 핵심 함정)
| 용도 | CUDA 버전 | 출처 |
|------|-----------|------|
| **런타임** (torch / whisper / pyannote / wav2vec2) | **12.8** | `nvidia-*-cu12==12.8.*` pip wheel (번들) |
| **빌드** (llama-cpp-python CUDA 컴파일) | **12.4** | 시스템 toolkit `/usr/local/cuda-12.4`, `nvcc 12.4.131` |

- `CUDA_HOME` 미설정. 시스템 toolkit 은 `/usr/local/cuda` → `cuda-12.4` 심링크.
- `llama-cpp-python==0.3.20` 는 시스템 nvcc(12.4)로 CUDA 빌드됨.
- **Dockerfile 시사점**: 사용자가 지정한 "CUDA 12.8 **runtime** + cuDNN9" 베이스는 nvcc 가 없어 llama-cpp CUDA 빌드 불가.
  → (A) `-devel` 이미지에서 빌드 후 runtime 으로 멀티스테이지 복사, 또는
  → (B) llama-cpp 사전빌드 wheel 사용, 또는
  → (C) toolkit 12.4 와 runtime 12.8 혼용(현재 서버 방식 그대로 재현).
  5단계에서 이 선택지를 Dockerfile 주석으로 남김.

## ffmpeg / 오디오 시스템 라이브러리 (apt)
| 패키지 | 버전 | 용도 |
|--------|------|------|
| ffmpeg | 7:4.4.2-0ubuntu0.22.04.1 (`/usr/bin/ffmpeg`, `ffprobe`) | stage_convert: 16kHz mono WAV + speechnorm |
| libsndfile1 | 1.0.31-2ubuntu0.2 | soundfile / 오디오 I/O |
| libsoxr0 | 0.1.3-4build2 | 리샘플링 |
| libgomp1 | 12.3.0-1ubuntu1~22.04.3 | OpenMP (ctranslate2 / llama-cpp CPU 스레드) |

→ Dockerfile `apt-get install`: `ffmpeg libsndfile1 libsoxr0 libgomp1`

## 환경변수
코드가 실제 참조(`os.getenv`/`os.environ`)하는 키:
`HF_TOKEN`, `PORT`, `MAX_QUEUE`, `PIPELINE_TIMEOUT`, `CALLBACK_ALLOWED_HOSTS`.
→ 마스킹 템플릿: `migration/env_template.txt`

### GPU 배치는 env 가 아니라 코드 하드코딩
`pipeline_gemma4.py`:
- `WhisperModel(..., device_index=0)`  → GPU0
- `...to("cuda:2")` (wav2vec2 alignment) → GPU2
- `to(torch.device("cuda:1"))` (pyannote) → GPU1
- `tensor_split=[0,0,1]` (Gemma 전 레이어 GPU2)
- `forced_align(..., device="cuda:2")`

→ `CUDA_VISIBLE_DEVICES` 로 제어 불가. **본부 서버도 GPU 3장(인덱스 0/1/2) 필요**. GPU 수가 다르면 코드 수정 필요(4단계 하드코딩 목록에 포함).

### HF 캐시
- `HF_HOME` 미설정 → 기본 `~/.cache/huggingface` (현재 **4.1G**) 사용.
- 여기에 whisper large-v3 / pyannote / wav2vec2 가중치 캐시됨 (3단계 인벤토리).
- 컨테이너에서는 볼륨 마운트 대상.
