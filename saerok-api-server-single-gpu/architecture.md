# Saerok API 아키텍처

---

## 전체 흐름 (외부 → 내부 → 외부)

```
[클라이언트]
    │
    │  POST /api/transcribe (오디오 파일 + callback_url 전송)
    ▼
[FastAPI 서버 (포트 8080)]
    │
    │  1. 요청 검증 (파일 형식, 크기, callback_url SSRF 체크)
    │  2. 대기열 확인 (4개 초과 시 429 반환)
    │  3. 즉시 응답: {"status": "accepted", "file_id": "xxx"}
    │  4. 백그라운드 태스크로 파이프라인 시작
    │
    ▼
[파이프라인 (백그라운드)]
    │
    │  Stage 1. 오디오 변환 (CPU)
    │    ffmpeg → 16kHz 모노 WAV + 음량 정규화
    │    Silero VAD로 음성 유무 체크 (뒤 30초 → 무음이면 앞 90초, 둘 다 무음이면 즉시 종료)
    │    → callback: 10%
    │
    │  Stage 2. 음성 인식 + 화자 분리 (GPU0 ∥ GPU1 동시)
    │    ┌─────────────────────────┬──────────────────────────┐
    │    │ GPU0: STT               │ GPU1: Diarization        │
    │    │ Whisper large-v3 x2     │ pyannote x2              │
    │    │ Queue(2)로 인스턴스 배정 │ Queue(2)로 인스턴스 배정  │
    │    │ BatchedInference(bs=24) │ 전체 waveform 통째 처리   │
    │    │ → 텍스트 + 단어 타임스탬프│ → 화자별 시간 구간        │
    │    └─────────────────────────┴──────────────────────────┘
    │    화자 분리 후처리: A-b-A 단발 플립 스무딩 + 초단문 흡수 + 동일화자 병합
    │    → callback: 40%
    │
    │  Stage 3. 병합 (CPU)
    │    Whisper 자체 단어 타임스탬프로 화자 매칭 (별도 정렬 스테이지 없음)
    │    + 단어 단위 A-b-A 스무딩 + 인접 동일화자 병합
    │    + 한국어 정규화 + 구두점 + 환각 제거
    │    → callback: 80%
    │
    │  Stage 4. 회의록 생성 (GPU2)
    │    Gemma 4 26B-A4B LLM (llm_lock, GPU2 전용)
    │    n_ctx=128K, flash_attn, swa_full=False
    │    → 구조화된 Markdown 회의록 생성
    │
    │  → callback: 100% (최종 결과 전체 전송)
    ▼
[클라이언트 callback_url]
    진행률 콜백 4회 수신 (10% → 40% → 80% → 100%)
    최종 콜백에 세그먼트 + 화자 + 회의록 포함
```

---

## GPU 배치도

```
┌──────────────────────────────────────────────────────┐
│ GPU0 (L40S 48GB)                    사용: 23GB/48GB  │
│                                                      │
│  Whisper #0 ──┐                                      │
│               ├── asyncio.Queue(2) → 빈 인스턴스 배정 │
│  Whisper #1 ──┘                                      │
│                                                      │
│  역할: STT 전담                                       │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ GPU1 (L40S 48GB)                    사용: 5.5GB/48GB │
│                                                      │
│  pyannote #0 ──┐                                     │
│                ├── asyncio.Queue(2) → 독립 인스턴스    │
│  pyannote #1 ──┘                                     │
│                                                      │
│  역할: 화자 분리 전담                                  │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ GPU2 (L40S 48GB)                   사용: ~38GB/48GB  │
│                                                      │
│  Gemma 4  ── asyncio.Lock (llm_lock)                 │
│             (26B-A4B MoE, n_ctx=128K, ~28GB+KV~8GB)  │
│                                                      │
│  GPU2 는 LLM 전용 (별도 정렬 스테이지 없음)           │
│                                                      │
│  역할: 회의록 생성 전담                                │
└──────────────────────────────────────────────────────┘
```

---

## 동시성 모델

```
요청 최대 4건 동시 (MAX_QUEUE=4, 초과 시 429)

[요청 A] Convert → STT(GPU0) ∥ Diar(GPU1) → Merge → LLM(GPU2)
[요청 B]          Convert → STT(GPU0) ∥ Diar(GPU1) → Merge → LLM(GPU2)
[요청 C]                   Convert → STT(GPU0) ∥ Diar(GPU1) → Merge → ...
[요청 D]                            Convert → STT(GPU0) ∥ Diar(GPU1) → ...

                    ↑ 4건이 각 스테이지에 1건씩 배치되어 모든 GPU 동시 가동
```

---

## 서버 라이프사이클

```
[시작]
  FastAPI lifespan 진입
  → /tmp/saerok_api/ 잔여 임시 파일 정리
  → httpx.AsyncClient 생성 (콜백 연결 풀링)
  → 모델 로딩 백그라운드 시작 (asyncio.create_task)
  → 서버 즉시 기동 (모델 로드 전 요청은 503 반환)
  → 모델 로드 완료 후 models_loaded = True

[운영]
  GET /health      → 모델 로드 상태 확인
  GET /api/status  → 활성 작업 수, GPU 배치 정보
  POST /api/transcribe → 요청 접수 + 백그라운드 파이프라인 실행

[종료]
  httpx 클라이언트 닫기
  LLM 모델 정리 (llm.close())
```

---

## 다이어그램 구성 가이드

아키텍처 다이어그램 그릴 때 3개 영역으로 나누면 깔끔함:

1. **외부 (클라이언트 ↔ API 서버)** — 요청/응답/콜백 흐름
2. **내부 (파이프라인 4단계)** — 스테이지별 처리 흐름과 병렬 구간
3. **GPU 배치도** — 3개 GPU의 역할 분리와 동시성 모델
