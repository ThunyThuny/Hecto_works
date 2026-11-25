# Saerok API 사용 방법

## 서버 실행

### 1. 환경 설정 (최초 1회)
```bash
cd /workspace/saerok_api
bash setup_gemma4.sh
export HF_TOKEN=hf_...
```

### 2. 백그라운드 실행 (tmux)
```bash
# tmux 설치 (최초 1회)
apt-get install -y tmux

# 세션 생성 + 서버 실행
tmux new -s saerok
export HF_TOKEN="hf_..."   # 실제 토큰은 별도 보관(시크릿). git에 커밋 금지
cd /workspace/saerok_api && python main_gemma4.py

# 빠져나오기 (서버 계속 돌아감)
Ctrl+B → D
```

### 3. tmux 주요 명령어
```bash
tmux ls                    # 세션 목록 확인
tmux attach -t saerok      # 세션 다시 접속 (로그 확인)
tmux kill-session -t saerok # 세션 종료 (서버 중지)
```

### 4. 서버 상태 확인
```bash
curl http://localhost:8080/health
curl http://localhost:8080/api/status
```

---

## API 사용

### 음성 파일 전사 요청
```bash
curl -X POST http://localhost:8080/api/transcribe \
  -F "file=@회의녹음.m4a" \
  -F "callback_url=https://your-server.com/callback" \
  -F "num_speakers=3"
```

| 파라미터 | 필수 | 설명 |
|----------|------|------|
| file | O | 오디오 파일 (.wav, .mp3, .m4a, .ogg, .webm / 최대 500MB) |
| callback_url | O | 결과 수신할 URL (진행률 + 최종 결과 전송) |
| num_speakers | X | 화자 수 힌트 (미지정 시 자동 감지) |

### 응답
```json
// 즉시 응답 (접수)
{"status": "accepted", "file_id": "abc123", "queue": "1/4"}

// 대기열 초과 시
HTTP 429: {"error": "처리 대기열이 가득 찼습니다 (4개)"}

// 모델 로딩 중
HTTP 503: {"error": "모델이 아직 로딩 중입니다."}
```

### 콜백 (callback_url로 POST 전송)
```json
// 진행률 콜백 (4회)
{"stage": "오디오 준비 중...", "progress": 10}
{"stage": "음성 분석 중...", "progress": 40}
{"stage": "회의록 생성 중...", "progress": 80}

// 완료 콜백
{
  "progress": 100,
  "result": {
    "segments": [...],
    "speakers": ["SPEAKER_00", "SPEAKER_01"],
    "minutes": "# 회의록\n...",
    "duration": 2268.5,
    "duration_formatted": "37:49",
    "whisper_model": "large-v3",
    "diarization_model": "pyannote/speaker-diarization-community-1",
    "llm_model": "gemma-4-26B-A4B-it-UD-Q8_K_XL",
    "prompt_version": "v1"
  }
}

// 에러 콜백
{"error": "음성이 감지되지 않았습니다.", "error_code": "NO_SPEECH", "progress": -1}
```

---

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| HF_TOKEN | (필수) | HuggingFace 토큰 (pyannote 모델 다운로드용) |
| PORT | 8080 | 서버 포트 |
| MAX_QUEUE | 4 | 최대 동시 처리 수 (초과 시 429) |
| PIPELINE_TIMEOUT | 900 | 파이프라인 타임아웃 (초, 기본 15분) |
| CALLBACK_ALLOWED_HOSTS | (비어있음) | 콜백 허용 호스트 (콤마 구분, 비어있으면 SSRF 차단 모드) |

---

## 로그

- 콘솔 + 파일 동시 출력
- 파일 경로: `/workspace/saerok_api/logs/saerok.log`
- 일별 로테이션, 7일 보관, KST 시간대
- DEBUG 로그 보기: 코드에서 `logging.basicConfig(level=logging.DEBUG, ...)` 변경
