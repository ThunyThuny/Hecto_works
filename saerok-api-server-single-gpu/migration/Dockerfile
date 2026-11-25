# Saerok API (Gemma 4) — 본부 서버(L40S x6) Docker 이관용
# 베이스: CUDA 12.8 runtime + cuDNN 9 (nvcc 없음).
#   llama-cpp-python 은 사전빌드 wheel 로 설치하므로 컴파일러/nvcc 불필요 (llama_cpp_install.md 선택지 B).
# 빌드 컨텍스트는 리포 루트(상위 디렉터리). compose: context: .. / dockerfile: migration/Dockerfile
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# [여기 추가됨!] apt-get update 중 임시 파일 생성 오류를 방지하기 위해 /tmp 권한 개방
RUN chmod 1777 /tmp

# ──────────────────────────────────────────
# 시스템 패키지
#   - 오디오/런타임 (요구 목록): ffmpeg libsndfile1 libsoxr0 libgomp1
#   - 파이썬 3.11 부트스트랩: 휠이 cp311 빌드라 3.11 필요. Ubuntu 22.04 기본은 3.10 → deadsnakes PPA.
#   - curl: /health 헬스체크 + get-pip
# ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
        ffmpeg libsndfile1 libsoxr0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# python/python3 → 3.11, pip 부트스트랩
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && python3.11 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# 백업 wheel + 슬림 의존성 목록만 먼저 복사 (레이어 캐시 효율). 모델은 이미지에 굽지 않음 — 볼륨 마운트.
COPY migration/wheels/ /tmp/wheels/
COPY migration/requirements_slim.txt /tmp/requirements_slim.txt

# ──────────────────────────────────────────
# (a) torch / torchaudio / torchvision (cu128)
#     백업 wheel(torch) 우선(--find-links), 백업에 없는 것(torchaudio/torchvision)은 cu128 인덱스에서.
#     로컬 torch-2.10.0+cu128 wheel 이 ==2.10.0 을 충족 → 그대로 사용.
# ──────────────────────────────────────────
RUN python3.11 -m pip install --no-cache-dir \
        --find-links /tmp/wheels \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.10.0 torchaudio==2.10.0 torchvision==0.25.0

# ──────────────────────────────────────────
# (b) llama-cpp-python==0.3.20 — 백업 wheel 오프라인 설치 (--no-index 로 소스빌드 폴백 차단).
#     버전 고정 필수. wheel: cu124 빌드(sm_89 포함, generic libcudart.so.12) → 12.8 런타임 전방호환.
# ──────────────────────────────────────────
RUN python3.11 -m pip install --no-cache-dir \
        --find-links /tmp/wheels \
        /tmp/wheels/llama_cpp_python-0.3.20-py3-none-linux_x86_64.whl

# ──────────────────────────────────────────
# (c) 나머지 — requirements_slim.txt (앱 실사용 의존성만; 결정 3).
#     slim 은 torch/torchaudio/torchvision/llama_cpp_python 을 이미 제외(위 (a)(b)에서 설치) →
#     중복/소스빌드 폴백 방지. ctranslate2 4.7.1 등은 PyPI 기본, nvidia-*-cu12 12.8 은 torch 설치가 자동 해석.
#     'packaging' 은 slim 에 평문으로 기재(원본 conda-local file:// 경로 정규화 — 결정 1 유지).
# ──────────────────────────────────────────
RUN python3.11 -m pip install --no-cache-dir -r /tmp/requirements_slim.txt \
    && rm -rf /tmp/wheels

# ──────────────────────────────────────────
# 앱 코드 (모델/프롬프트 catalog 포함, 모델 가중치는 제외)
# ──────────────────────────────────────────
COPY main_gemma4.py pipeline_gemma4.py pipeline_utils.py ./
COPY prompts/ ./prompts/
COPY previews/ ./previews/

# 문서용. 실제 바인딩 포트는 PORT env (compose 에서 8080/8081).
EXPOSE 8080

# uvicorn 으로 기동, PORT env 사용 (미설정 시 8080). host 0.0.0.0 = 컨테이너 표준.
# shell 형식 — ${PORT}, ${ROOT_PATH} 확장을 위해.
# ROOT_PATH 가 설정돼 있으면(예: /api0) --root-path 부여 → nginx 프리픽스 프록시 하위에서 Swagger/openapi 정상.
CMD uvicorn main_gemma4:app --host 0.0.0.0 --port ${PORT:-8080} ${ROOT_PATH:+--root-path ${ROOT_PATH}}