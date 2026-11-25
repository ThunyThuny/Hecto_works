#!/bin/bash
# ==========================================
#  Saerok API 환경 셋업 (Gemma 4 26B-A4B 버전)
#  전제: Ubuntu 22.04+, NVIDIA GPU x3 (L40S)
#  LLM: google/gemma-4-26B-A4B-it (Unsloth GGUF UD-Q8_K_XL, ~27.9GB)
# ==========================================

set -eo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}OK${NC} $1"; }
warn() { echo -e "  ${YELLOW}WARNING${NC} $1"; }
fail() { echo -e "  ${RED}FAIL${NC} $1"; exit 1; }

echo "=========================================="
echo "  Saerok API — Setup (Gemma 4 26B-A4B)"
echo "=========================================="


# ──────────────────────────────────────────
# 1. NVIDIA 드라이버
# ──────────────────────────────────────────
echo ""
echo "[1/8] NVIDIA 드라이버"

if command -v nvidia-smi &> /dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    ok "드라이버 $DRIVER_VER | GPU ${GPU_COUNT}장 | $GPU_NAME"

    if [ "$GPU_COUNT" -lt 3 ]; then
        warn "GPU ${GPU_COUNT}장 감지. 최적 성능은 3장 필요 (GPU0:STT, GPU1:Diar, GPU2:LLM)"
    fi
else
    warn "NVIDIA 드라이버 없음 → 설치 중..."
    apt-get update
    apt-get install -y ubuntu-drivers-common
    ubuntu-drivers install
    ok "드라이버 설치 완료 (재부팅 필요할 수 있음)"

    if ! command -v nvidia-smi &> /dev/null; then
        fail "드라이버 설치 후 nvidia-smi 실행 불가. 재부팅 후 다시 실행하세요."
    fi
fi


# ──────────────────────────────────────────
# 2. CUDA Toolkit
# ──────────────────────────────────────────
echo ""
echo "[2/8] CUDA Toolkit"

if command -v nvcc &> /dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    ok "CUDA $CUDA_VER"

    # Gemma 4: CUDA 13.2 runtime은 출력 품질 저하 알려진 이슈
    if echo "$CUDA_VER" | grep -q "13.2"; then
        warn "⚠ CUDA 13.2는 Gemma 4 GGUF 출력 품질 저하 이슈가 보고됨. CUDA 12.x 사용 권장"
    fi
else
    if command -v nvidia-smi &> /dev/null; then
        CUDA_DRIVER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        if [ -n "$CUDA_DRIVER" ]; then
            ok "CUDA $CUDA_DRIVER (드라이버 번들)"
        else
            warn "nvcc 없음 → CUDA Toolkit 설치 중..."
            wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
            dpkg -i cuda-keyring_1.1-1_all.deb
            apt-get update
            apt-get install -y cuda-toolkit-12-6
            rm -f cuda-keyring_1.1-1_all.deb
            export PATH=/usr/local/cuda-12.6/bin:$PATH
            export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
            ok "CUDA 12.6 설치 완료"
        fi
    else
        fail "nvidia-smi도 없음. NVIDIA 드라이버부터 설치하세요."
    fi
fi


# ──────────────────────────────────────────
# 3. ffmpeg
# ──────────────────────────────────────────
echo ""
echo "[3/8] ffmpeg"

if command -v ffmpeg &> /dev/null; then
    ok "$(ffmpeg -version 2>&1 | head -1)"
else
    warn "ffmpeg 없음 → 설치 중..."
    apt-get update
    apt-get install -y ffmpeg
    ok "ffmpeg 설치 완료"
fi


# ──────────────────────────────────────────
# 4. Python
# ──────────────────────────────────────────
echo ""
echo "[4/8] Python"

if command -v python3 &> /dev/null; then
    PY_VER=$(python3 --version 2>&1)
    PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
    if [ "$PY_MINOR" -ge 10 ]; then
        ok "$PY_VER"
    else
        fail "Python 3.10+ 필요 (현재: $PY_VER)"
    fi
else
    fail "Python3 없음. python3.11 설치하세요: apt install python3.11 python3.11-venv"
fi


# ──────────────────────────────────────────
# 5. PyTorch + Python 패키지
# ──────────────────────────────────────────
echo ""
echo "[5/8] Python 패키지"

# PyTorch
TORCH_OK=$(python3 -c "import torch; print('1' if torch.cuda.is_available() else '0')" 2>/dev/null || echo "0")

if [ "$TORCH_OK" = "1" ]; then
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
    ok "PyTorch $TORCH_VER (CUDA 사용 가능)"
else
    warn "PyTorch 없거나 CUDA 미지원 → 설치 중..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    ok "PyTorch 설치 완료"
fi

# 핵심 패키지
# numpy<2.3 핀: 의존 패키지들과의 호환성 보수적 고정
echo "  핵심 패키지 설치 중..."
pip install -q \
    "numpy<2.3" \
    faster-whisper \
    pyannote.audio \
    transformers \
    soundfile \
    nvidia-ml-py \
    fastapi \
    uvicorn \
    httpx \
    python-multipart \
    2>&1 | tail -1
ok "Python 패키지 설치 완료"

# llama-cpp-python CUDA 빌드
# Gemma 4 요구: 0.3.20 이상 (텐서 shape mismatch 등 Gemma 4 fix 포함)
LLAMA_MIN_VER="0.3.20"

LLAMA_CUR_VER=$(python3 -c "import llama_cpp; print(llama_cpp.__version__)" 2>/dev/null || echo "none")

version_ge() {
    # $1 >= $2 인지 비교
    [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]
}

if [ "$LLAMA_CUR_VER" = "none" ]; then
    warn "llama-cpp-python 없음 → CUDA 빌드 설치 중 (Gemma 4 지원 위해 최신 버전)..."
    CMAKE_ARGS="-DGGML_CUDA=on" pip install "llama-cpp-python>=$LLAMA_MIN_VER" --force-reinstall --no-cache-dir 2>&1 | tail -3
    ok "llama-cpp-python CUDA 빌드 완료"
elif version_ge "$LLAMA_CUR_VER" "$LLAMA_MIN_VER"; then
    ok "llama-cpp-python $LLAMA_CUR_VER (Gemma 4 호환)"
else
    warn "llama-cpp-python $LLAMA_CUR_VER → Gemma 4 요구 $LLAMA_MIN_VER+ 이상 필요. 업그레이드 중..."
    CMAKE_ARGS="-DGGML_CUDA=on" pip install "llama-cpp-python>=$LLAMA_MIN_VER" --force-reinstall --no-cache-dir 2>&1 | tail -3
    ok "llama-cpp-python 업그레이드 완료"
fi

# 불필요한 의존성 제거
# - torchcodec: pyannote-audio가 설치하지만 waveform dict 입력 시 불필요. CUDA 버전 불일치로 libnvrtc 에러 유발
# - pynvml: deprecated 래퍼. nvidia-ml-py가 동일 API 제공 (import pynvml 호환)
pip uninstall -y torchcodec pynvml 2>/dev/null || true
ok "불필요 패키지 정리 완료 (torchcodec, pynvml)"

# Silero VAD 사전 다운로드
echo "  Silero VAD 캐시 다운로드 중..."
python3 -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)" 2>&1 | tail -1
ok "Silero VAD 캐시 완료"


# ──────────────────────────────────────────
# 6. HuggingFace 토큰
# ──────────────────────────────────────────
echo ""
echo "[6/8] HuggingFace 토큰"

if [ -n "$HF_TOKEN" ]; then
    ok "HF_TOKEN 설정됨"
else
    warn "HF_TOKEN 환경변수 없음"
    echo "       설정 필요: export HF_TOKEN=hf_your_token_here"
    echo "       토큰 생성: https://huggingface.co/settings/tokens"
    echo "       pyannote 모델 동의: https://huggingface.co/pyannote/speaker-diarization-community-1"
fi


# ──────────────────────────────────────────
# 7. LLM 모델 다운로드 (Gemma 4 26B-A4B)
# ──────────────────────────────────────────
echo ""
echo "[7/8] LLM 모델 (Gemma 4 26B-A4B-it, Unsloth UD-Q8_K_XL)"

LLM_DIR="/workspace/meeting_AI/models"
LLM_REPO="unsloth/gemma-4-26B-A4B-it-GGUF"
LLM_FILE="gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf"
LLM_PATH="$LLM_DIR/$LLM_FILE"

# 이전 EXAONE 모델 정리 (디스크 공간 확보)
EXAONE_FILE="$LLM_DIR/EXAONE-3.5-7.8B-Instruct-Q4_K_M.gguf"
if [ -f "$EXAONE_FILE" ]; then
    EXAONE_SIZE=$(du -h "$EXAONE_FILE" | awk '{print $1}')
    warn "이전 EXAONE 모델 발견 ($EXAONE_SIZE) → 삭제하여 디스크 공간 확보"
    rm -f "$EXAONE_FILE"
    ok "EXAONE 모델 삭제 완료"
fi

if [ -f "$LLM_PATH" ]; then
    SIZE=$(du -h "$LLM_PATH" | awk '{print $1}')
    ok "모델 존재: $LLM_PATH ($SIZE)"
else
    # 디스크 여유 확인 (최소 35GB 권장 — UD-Q8_K_XL ~28GB + 임시 공간)
    AVAIL_GB=$(df -BG /workspace | awk 'NR==2 {gsub("G",""); print $4}')
    if [ "$AVAIL_GB" -lt 35 ]; then
        warn "디스크 여유 ${AVAIL_GB}GB — Gemma 4 26B-A4B UD-Q8_K_XL (~28GB) 다운로드에 빠듯합니다"
    else
        ok "디스크 여유 ${AVAIL_GB}GB — 다운로드 가능"
    fi

    warn "모델 없음 → 다운로드 중 (~28GB, 시간 소요)..."
    mkdir -p "$LLM_DIR"

    if ! command -v hf &> /dev/null; then
        pip install -q -U "huggingface_hub[cli]"
    fi

    hf download \
        "$LLM_REPO" \
        "$LLM_FILE" \
        --local-dir "$LLM_DIR"

    if [ -f "$LLM_PATH" ]; then
        SIZE=$(du -h "$LLM_PATH" | awk '{print $1}')
        ok "다운로드 완료: $LLM_PATH ($SIZE)"
    else
        fail "다운로드 실패"
    fi
fi


# ──────────────────────────────────────────
# 8. 환경 검증
# ──────────────────────────────────────────
echo ""
echo "[8/8] 환경 검증"

python3 << 'PYCHECK'
import sys

checks = []

checks.append(f"Python:          {sys.version.split()[0]}")

try:
    import torch
    checks.append(f"PyTorch:         {torch.__version__}")
    checks.append(f"CUDA:            {torch.version.cuda}")
    checks.append(f"GPU 수:          {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        checks.append(f"  GPU {i}:          {name} ({mem:.0f}GB)")
except Exception as e:
    checks.append(f"PyTorch:         FAIL ({e})")

try:
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    checks.append(f"faster-whisper:  OK")
except Exception as e:
    checks.append(f"faster-whisper:  FAIL ({e})")

try:
    from pyannote.audio import Pipeline
    checks.append(f"pyannote:        OK")
except Exception as e:
    checks.append(f"pyannote:        FAIL ({e})")

try:
    import llama_cpp
    from llama_cpp import Llama
    ver = llama_cpp.__version__
    # Gemma 4 최소 요구 버전 확인
    parts = [int(x) for x in ver.split(".")[:3]]
    min_parts = [0, 3, 20]
    is_ok = parts >= min_parts
    if is_ok:
        checks.append(f"llama-cpp:       OK ({ver}, Gemma 4 호환)")
    else:
        checks.append(f"llama-cpp:       FAIL ({ver}, Gemma 4 요구 0.3.20+)")
except Exception as e:
    checks.append(f"llama-cpp:       FAIL ({e})")

try:
    import soundfile
    checks.append(f"soundfile:       OK")
except Exception as e:
    checks.append(f"soundfile:       FAIL ({e})")

try:
    import fastapi, uvicorn, httpx
    checks.append(f"fastapi:         OK ({fastapi.__version__})")
    checks.append(f"uvicorn:         OK ({uvicorn.__version__})")
    checks.append(f"httpx:           OK ({httpx.__version__})")
except Exception as e:
    checks.append(f"API 패키지:      FAIL ({e})")

try:
    import pynvml
    pynvml.nvmlInit()
    checks.append(f"pynvml:          OK")
except:
    checks.append(f"pynvml:          없음 (선택사항)")

import os
llm_path = "/workspace/meeting_AI/models/gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf"
if os.path.exists(llm_path):
    size_gb = os.path.getsize(llm_path) / 1024**3
    checks.append(f"LLM 모델:        OK ({size_gb:.1f}GB) — Gemma 4 26B-A4B UD-Q8_K_XL")
else:
    checks.append(f"LLM 모델:        없음!")

for c in checks:
    print(f"  {c}")

fails = [c for c in checks if "FAIL" in c or "없음!" in c]
if fails:
    print(f"\n  {len(fails)}개 문제 발견")
else:
    print(f"\n  모든 환경 준비 완료")
PYCHECK

echo ""
echo "=========================================="
echo "  실행 방법"
echo "=========================================="
echo ""
echo "  # 환경변수 설정"
echo "  export HF_TOKEN=hf_your_token_here"
echo ""
echo "  # API 서버 시작 (port 8080)"
echo "  cd /workspace/saerok_api && python main_gemma4.py"
echo ""
echo "  # Health 체크"
echo "  curl http://localhost:8080/health"
echo ""
echo "  # 전사 요청"
echo '  curl -X POST http://localhost:8080/api/transcribe \'
echo '    -F "file=@audio.m4a" \'
echo '    -F "callback_url=http://your-backend/callback"'
echo ""
echo "=========================================="
