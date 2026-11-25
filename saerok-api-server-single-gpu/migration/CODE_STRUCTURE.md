# 코드/설정 구조 + 하드코딩 경로 (Step 4)

## 코드 구조
| 역할 | 파일 | 비고 |
|------|------|------|
| **앱 진입점** | `main_gemma4.py` | `app = FastAPI(... lifespan=lifespan)` (L171), `uvicorn.run(app, host="0.0.0.0", port=PORT)` (L633). 엔드포인트·콜백·VAD·로깅 |
| **파이프라인 / 모델 로딩** | `pipeline_gemma4.py` | `load_models`, stage 함수들, 튜닝 상수, LLM 상수 |
| **유틸** | `pipeline_utils.py` | CTC 정렬, 화자 병합, 한국어 정규화, 기본 프롬프트, `MEETING_TYPES` |
| **프롬프트 카탈로그** | `prompts/{seminar,meeting,team_report,counsel}.md` | 요청마다 재로드 (재시작 불필요) |
| **프리뷰** | `previews/{...}_preview.md` | |
| **환경 셋업** | `setup_gemma4.sh` | 드라이버/CUDA/ffmpeg/pip/llama-cpp 빌드/모델 다운로드 |
| **의존성 스냅샷** | `requirements.txt` (참고용), 정본은 setup 스크립트 |

진입점 외 별도 설정파일(.env/.toml/.cfg) 없음 — 설정은 전부 환경변수 + 코드 상수.

## ⚠️ 하드코딩 경로 — 본부 서버 이관 시 환경변수화 대상

| # | 위치 | 현재 값 | 환경변수화 제안 |
|---|------|---------|-----------------|
| 1 | `main_gemma4.py:52` | `LOG_DIR = Path("/workspace/saerok_api/logs")` | `LOG_DIR` |
| 2 | `main_gemma4.py:75` | `TEMP_DIR = Path("/tmp/saerok_api")` | `TEMP_DIR` |
| 3 | `pipeline_gemma4.py:30` | `os.makedirs("/workspace/meeting_AI/models")` | `MODELS_DIR` |
| 4 | `pipeline_gemma4.py:50` | `LLM_PATH = "/workspace/meeting_AI/models/gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf"` | `LLM_PATH` |
| 5 | `setup_gemma4.sh:216,219,343` | `LLM_DIR=/workspace/meeting_AI/models`, `llm_path=...gguf` | (셋업 스크립트는 컨테이너에선 미사용 가능) |

> 위 4개 파이썬 상수가 핵심. 전부 `os.getenv("X", "<현재 기본값>")` 패턴으로 바꾸면
> 본부 서버 경로 변경이 코드 수정 없이 가능. **현재는 env 미지원이므로 Dockerfile 에서
> 동일 경로를 그대로 만들어주거나(=경로 호환), 코드를 env 화하거나 택1.**

## 하드코딩이지만 환경변수화 불필요 (경로 아님)
- **모델 repo ID** (HF 캐시로 자동 해석되므로 그대로 둬도 됨):
  - `WhisperModel("large-v3", device_index=0)` → Systran 캐시
  - `kresnik/wav2vec2-large-xlsr-korean` (`pipeline_gemma4.py:84-85`)
  - `pyannote/speaker-diarization-community-1` (`:91`, `token=HF_TOKEN`)
  - `snakers4/silero-vad` (`main_gemma4.py:129,134`, torch.hub)
- **호스트**: `host="0.0.0.0"` (컨테이너 표준, 변경 불필요). 포트는 이미 `PORT` env.

## ⚠️ GPU 디바이스 하드코딩 (재확인 — env 제어 불가)
`pipeline_gemma4.py`: `device_index=0`(GPU0), `to("cuda:2")`(GPU2 wav2vec2), `cuda:1`(GPU1 pyannote),
`tensor_split=[0,0,1]`(Gemma→GPU2), `forced_align(device="cuda:2")`.
→ **본부 서버도 GPU 인덱스 0/1/2 3장 필수.** 구성이 다르면 코드 직접 수정 필요.
