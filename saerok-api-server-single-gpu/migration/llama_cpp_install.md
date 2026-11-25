# llama-cpp-python 설치 방식 검증 (Dockerfile 확정 전)

**목표**: 소스 빌드(nvcc 필요) 없이 **사전빌드 wheel** 로 설치 가능한지 검증.
**결론: 가능 (선택지 B 채택 권장).** nvcc/멀티스테이지 빌드 불필요.

---

## 1. 현재 설치본 (0.3.20) 분석

| 항목 | 값 |
|------|-----|
| 버전 | 0.3.20 (Gemma 4 요구 최소치, 현재 서버 정상 동작 — `/health` `models_loaded:true`, `llm_model:gemma-4-26B-A4B-it-UD-Q8_K_XL`) |
| 빌드 방식 | **로컬 소스 컴파일** (`scikit-build-core`, `CMAKE_ARGS=-DGGML_CUDA=on`, setup_gemma4.sh:172) |
| Wheel 태그 | `py3-none-linux_x86_64` (ctypes 바인딩 → Python 버전 무관) |
| libggml-cuda.so | 136M, **sm_89 단일 아키텍처** 빌드 |
| CUDA 링크 | `libcudart.so.12` / `libcublas.so.12` / `libcublasLt.so.12` (generic 12.x soname) → 현재 `/usr/local/cuda`=12.4 로 해석 |

핵심: 링크가 특정 마이너(12.4)가 아니라 **generic `.so.12`** → CUDA 12.x 런타임 아무 버전이나 충족.

---

## 2. abetlen 공식 CUDA 사전빌드 wheel 인덱스 조사

`https://abetlen.github.io/llama-cpp-python/whl/<variant>/` 실측 조회 결과:

| variant | 0.3.20 제공? | 비고 |
|---------|:---:|------|
| cu121 | ✅ | |
| cu122 | ✅ | |
| cu123 | ✅ | |
| **cu124** | ✅ | **채택 대상** (12.x 최신 중 0.3.20 있음) |
| cu125 | ❌ | 0.3.22 부터만 존재 |
| **cu126** | ❌ | 인덱스/wheel 자체 없음 |
| **cu128** | ❌ | 인덱스/wheel 자체 없음 |

→ **cu126/cu128 네이티브 사전빌드 wheel 은 존재하지 않음.** 12.x 사전빌드의 사실상 상한이 cu124.

### cu124 의 0.3.20 wheel 실측 검증 (migration/wheels/ 에 백업 완료)
파일: `llama_cpp_python-0.3.20-py3-none-linux_x86_64.whl` (1.49 GB)

| 검증 항목 | 결과 | 판정 |
|-----------|------|:---:|
| Python 태그 | `py3-none` (cp311 자동 호환) | ✅ |
| 내장 SM arch | `sm_70 75 80 86 **89** 90` (fat binary) | ✅ L40S(sm_89) 포함 |
| CUDA soname | `libcudart.so.12` / `libcublas.so.12` / `libcublasLt.so.12` | ✅ generic 12.x |
| libggml-cuda.so | 820M (전 아키텍처라 로컬 136M보다 큼) | (무방 — 모델 26G 대비 미미) |

---

## 3. 결론 — 선택지 B 채택

### ✅ 왜 가능한가
- cu124 사전빌드 wheel(0.3.20)이 **L40S(sm_89) 포함**, **Python 3.11 호환**, **현재와 동일 버전**.
- wheel 은 CUDA 12.4 로 빌드됐지만 `libcudart.so.12` generic 링크 → **CUDA 12.x 런타임이면 동작** (12.x 마이너 전방호환). **12.8-runtime 베이스가 `.so.12` 런타임 라이브러리를 제공하므로 nvcc 없이 실행 가능.**
- 사용자가 지시한 "CUDA 12.8 **runtime** 베이스" 그대로 단일 스테이지로 충분 → 멀티스테이지/devel 불필요.

### 설치 커맨드 (Dockerfile 용, 2가지)

**(권장) 오프라인 — 백업 wheel 직접 설치** (재현성 100%, 네트워크 무관):
```dockerfile
COPY migration/wheels/llama_cpp_python-0.3.20-py3-none-linux_x86_64.whl /tmp/
RUN pip install /tmp/llama_cpp_python-0.3.20-py3-none-linux_x86_64.whl
```

**(대안) 온라인 — 인덱스에서 설치**:
```dockerfile
RUN pip install llama-cpp-python==0.3.20 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```
> 버전 `==0.3.20` 고정 필수. 미고정 시 PyPI sdist 로 빠져 소스 빌드(nvcc 필요)로 폴백될 수 있음.
> `--extra-index-url` 사용 시 의존성(diskcache/jinja2/numpy/typing-extensions)은 PyPI 에서, 본체는 abetlen wheel 에서 받음.

### 잔여 리스크 / 검증 권장
- wheel 은 CUDA **12.4** 빌드, 런타임은 **12.8** → 마이너 전방호환에 의존. 표준적으로 안전하나, 최초 빌드 후 `GET /health` 로 `models_loaded:true` + 실제 전사 1건으로 GPU 추론 동작 확인 권장.
- 12.8 로 **정확히** 빌드된 wheel 이 필요하면 그때만 선택지 A(멀티스테이지 devel 빌드)로 전환.

### 폴백 (B 불가 시)
- **A. 멀티스테이지**: `cuda:12.8-devel` 에서 `CMAKE_ARGS=-DGGML_CUDA=on pip install`(sm_89 만 빌드해 슬림) → `cuda:12.8-runtime` 로 `llama_cpp/lib/*.so` + 패키지 복사.
- **C. toolkit 혼용**: 최종 이미지에 cuda-toolkit-12-4 포함(이미지 비대) — 비권장.
