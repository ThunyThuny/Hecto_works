"""
Saerok 파이프라인 (Gemma 4 26B-A4B 버전) — L40S 48GB x3 (3 GPU 전용)
모델 로드 + 스테이지 함수 (main_gemma4.py에서 import하여 사용)

LLM: google/gemma-4-26B-A4B-it (Unsloth Dynamic UD-Q8_K_XL, ~28GB)
  - MoE: 총 26B / 활성 4B (토큰당 8 expert)
  - Hybrid attention (local sliding + global) → 긴 컨텍스트에서 KV cache 효율적
  - Native context: 256K
  - System role 정식 지원

GPU 배치:
  GPU0: Whisper x2 (STT 전담)
  GPU1: pyannote x2 (Diarization)
  GPU2: LLM (Gemma 4) 전담

Gemma 4 권장 sampling (공식):
  temperature=1.0, top_p=0.95, top_k=64, repetition_penalty=1.0(비활성)
"""

import logging
import os
import subprocess
import time
import warnings

###dylee : 원본도 물리 GPU 3,4,5만 보이도록 제한하고, 코드 내부에서는 논리 GPU 0,1,2로 재매핑
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"

import torch
import soundfile as sf

warnings.filterwarnings("ignore")
MODELS_DIR = os.getenv("MODELS_DIR", "/workspace/data/nfs/models/llm/foundation")
os.makedirs(MODELS_DIR, exist_ok=True)

logger = logging.getLogger("saerok.pipeline")

from pipeline_utils import (
    postprocess_diarization, build_transcript_text,
    parse_minutes_response, MINUTES_SYSTEM_PROMPT,
    load_prompt_template, MEETING_TYPES,
)

# ──────────────────────────────────────────
# 튜닝 파라미터 (테스트 시 여기만 수정)
# ──────────────────────────────────────────
BATCH_SIZE = 24             # Whisper batch size
NUM_WHISPER = 2             # Whisper 인스턴스 수 (GPU0)
DIAR_CONCURRENCY = 2        # GPU1 Diar 동시성 (2 = 동시 2개 허용)

# Diarization 후처리 튜닝 (과분리 제어) — env 로 조정, 기본값은 기존 동작 유지
DIAR_COLLAR = float(os.getenv("DIAR_COLLAR", "0.5"))                    # 같은 화자 인접 병합 gap 상한(s)
DIAR_MIN_DURATION = float(os.getenv("DIAR_MIN_DURATION", "0.3"))        # 이보다 짧은 구간 흡수(s)
DIAR_SMOOTH_MAX = float(os.getenv("DIAR_SMOOTH_MAX_DURATION", "1.0"))   # A-b-A 스무딩: 이보다 짧고 양옆 같은 화자면 흡수(s, 0=off). 검증 완료 기본 ON

# Gemma 4 LLM 설정
HF_TOKEN = os.getenv("HF_TOKEN")
###dylee : Gemma와 pyannote를 다운로드하지 않고 현재 서버의 로컬 모델 경로에서 로드
LLM_PATH = os.getenv(
    "LLM_PATH",
    "/workspace/data/nfs/models/llm/foundation/gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf",
)
PYANNOTE_MODEL_PATH = os.getenv(
    "PYANNOTE_MODEL_PATH",
    "/root/.cache/huggingface/hub/models--pyannote--speaker-diarization-community-1/"
    "snapshots/3533c8cf8e369892e6b79ff1bf80f7b0286a54ee/config.yaml",
)
LLM_N_CTX = 131072          # Gemma 4 native 256K. 128K로 확장 (3시간 회의 여유, KV cache ~8GB)
LLM_N_BATCH = 512           # MoE는 큰 prefill batch가 항상 유리하지 않음 (EXAONE의 2048에서 축소)
LLM_N_THREADS = 4           # MoE expert routing에 약간의 CPU 사용 (EXAONE의 1에서 증가)

# Gemma 4 공식 권장 sampling
LLM_TEMPERATURE = 1.0
LLM_TOP_P = 0.95
LLM_TOP_K = 64
LLM_MAX_TOKENS = 8192

# 메타데이터 (콜백 result에 포함)
LLM_MODEL_NAME = "gemma-4-26B-A4B-it-UD-Q8_K_XL"


# ──────────────────────────────────────────
# 모델 로더
# ──────────────────────────────────────────
def load_models():
    ###dylee : 로컬 모델 파일이 없으면 원격 다운로드를 시도하지 않고 즉시 원인을 표시
    if not os.path.isfile(PYANNOTE_MODEL_PATH):
        raise FileNotFoundError(f"Local pyannote config not found: {PYANNOTE_MODEL_PATH}")
    if not os.path.isfile(LLM_PATH):
        raise FileNotFoundError(f"Local Gemma model not found: {LLM_PATH}")
    logger.info(f"[Load] Models (Whisper x{NUM_WHISPER} on logical GPU0=physical3, Diar on logical GPU1=physical4, LLM on logical GPU2=physical5)")
    logger.info(f"[Load] Local pyannote: {PYANNOTE_MODEL_PATH}")
    logger.info(f"[Load] LLM: {LLM_MODEL_NAME} (n_ctx={LLM_N_CTX}, n_batch={LLM_N_BATCH}, n_threads={LLM_N_THREADS})")
    t0 = time.time()

    from faster_whisper import WhisperModel, BatchedInferencePipeline

    whisper_pool = []
    for i in range(NUM_WHISPER):
        m = WhisperModel("large-v3", device="cuda", device_index=0, compute_type="float16")
        whisper_pool.append(BatchedInferencePipeline(model=m))
        logger.info(f"  Whisper {i} (logical GPU0=physical3): OK")

    from pyannote.audio import Pipeline as DiarPipeline
    diar_pool = []
    for i in range(DIAR_CONCURRENCY):
        ###dylee : Hugging Face 저장소 ID 대신 로컬 캐시의 config.yaml을 직접 로드
        d = DiarPipeline.from_pretrained(PYANNOTE_MODEL_PATH, token=HF_TOKEN or None)
        d = d.to(torch.device("cuda:1"))
        diar_pool.append(d)
        logger.info(f"  pyannote {i} (logical GPU1=physical4): OK")

    from llama_cpp import Llama
    llm = Llama(
        model_path=LLM_PATH,
        n_ctx=LLM_N_CTX,
        n_gpu_layers=-1,
        n_threads=LLM_N_THREADS,
        n_threads_batch=LLM_N_THREADS,
        tensor_split=[0, 0, 1],   # 전 레이어 GPU2에
        verbose=False,
        flash_attn=True,          # 26B-A4B(MoE)는 flash_attn 안전 (31B Dense의 hang 버그와 무관)
        n_batch=LLM_N_BATCH,
        swa_full=False,           # SWA(sliding window) layer는 window 크기만 KV cache 할당.
                                  # 기본 True면 n_ctx 전체로 잡혀 128K에서 OOM (llama.cpp PR #13194).
                                  # 회의록 single-shot 생성 워크로드에서 품질 영향 없음.
    )
    logger.info(f"  LLM (logical GPU2=physical5): OK (Gemma 4 26B-A4B, n_ctx={LLM_N_CTX}, flash_attn, swa_full=False, n_batch={LLM_N_BATCH})")
    logger.info(f"  Total: {time.time()-t0:.1f}s")

    return whisper_pool, {
        "diar_pool": diar_pool,
        "llm": llm,
    }


# ──────────────────────────────────────────
# 스테이지 함수
# ──────────────────────────────────────────
def stage_convert(input_path, output_path):
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", "speechnorm=e=10:r=0.005:l=1",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 변환 실패: {result.stderr[:500]}")
    return sf.info(output_path).duration


def stage_stt(batched_model, audio_path):
    segments, info = batched_model.transcribe(
        audio_path, language="ko", batch_size=BATCH_SIZE,
        word_timestamps=True, condition_on_previous_text=False,
        hallucination_silence_threshold=2.0, no_repeat_ngram_size=3,
        repetition_penalty=1.1, vad_filter=True,
        vad_parameters=dict(onset=0.3, offset=0.35,
                           min_silence_duration_ms=1000,
                           speech_pad_ms=400, min_speech_duration_ms=250),
    )
    result = []
    for seg in segments:
        words = [{"word": w.word, "start": w.start, "end": w.end} for w in seg.words] if seg.words else []
        result.append({"start": seg.start, "end": seg.end, "text": seg.text, "words": words})
    return result


def stage_diarization(pipeline, waveform, num_speakers=None, dump_path=None):
    kwargs = {"waveform": waveform.unsqueeze(0), "sample_rate": 16000}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    output = pipeline(kwargs)
    annotation = getattr(output, "speaker_diarization", output)
    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    if dump_path:  # 원본(후처리 전) 세그먼트 덤프 — 오프라인 파라미터 스윕용
        try:
            import json
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(segments, f, ensure_ascii=False)
            logger.info(f"  [Diar] 원본 세그먼트 {len(segments)}개 덤프 → {dump_path}")
        except Exception as e:
            logger.warning(f"  [Diar] 덤프 실패: {e}")
    return postprocess_diarization(
        segments, collar=DIAR_COLLAR, min_duration=DIAR_MIN_DURATION,
        smooth_max_duration=DIAR_SMOOTH_MAX,
    )


def stage_llm_with_prompt(llm, segments, prompt_template,
                          system_prompt=None, include_speakers=True, sampling=None):
    """프롬프트·샘플링을 외부 주입받는 stage_llm 변종.

    /api/test-prompt 와 /api/transcribe(meeting_type 지정 시) 양쪽에서 사용.

    Args:
        segments: merge_results 결과의 segments 배열 (또는 호환되는 dict 리스트)
        prompt_template: {transcript_text} 자리표시자 포함된 템플릿 문자열
        system_prompt: 미지정 시 MINUTES_SYSTEM_PROMPT
        include_speakers: False 시 화자 라벨 없이 transcript 빌드 (STT-only 시뮬레이션)
        sampling: {"temperature","top_p","top_k","max_tokens"} 부분 또는 전체. 미지정 시 운영 default.

    Returns:
        (minutes_markdown, usage_dict)
    """
    transcript = build_transcript_text({"segments": segments}, include_speakers=include_speakers)

    if "{transcript_text}" in prompt_template:
        prompt = prompt_template.replace("{transcript_text}", transcript)
    else:
        prompt = prompt_template + "\n\n## 회의 녹취록\n\n" + transcript

    sys_prompt = system_prompt or MINUTES_SYSTEM_PROMPT
    s = sampling or {}
    temperature = s.get("temperature", LLM_TEMPERATURE)
    top_p = s.get("top_p", LLM_TOP_P)
    top_k = s.get("top_k", LLM_TOP_K)
    max_tokens = s.get("max_tokens", LLM_MAX_TOKENS)

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]
    input_tokens = llm.tokenize((sys_prompt + prompt).encode("utf-8"), add_bos=False)
    n_input = len(input_tokens)
    n_ctx = llm.n_ctx()
    logger.info(f"[LLM-custom] 녹취록 길이: {len(transcript)}자 / 입력 토큰: {n_input} / n_ctx: {n_ctx} / max_tokens: {max_tokens} / include_speakers: {include_speakers}")
    if n_input + max_tokens > n_ctx:
        logger.warning(f"[LLM-custom] ⚠ 입력 토큰({n_input}) + max_tokens({max_tokens}) = {n_input + max_tokens} > n_ctx({n_ctx}) — 녹취록이 잘릴 수 있음!")

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    choices = response.get("choices", [])
    if not choices:
        return parse_minutes_response(None), {}
    finish_reason = choices[0].get("finish_reason", "unknown")
    content = choices[0].get("message", {}).get("content")
    usage = response.get("usage", {})
    logger.info(f"[LLM-custom] 완료 — finish_reason: {finish_reason} / prompt_tokens: {usage.get('prompt_tokens', '?')} / completion_tokens: {usage.get('completion_tokens', '?')}")
    if finish_reason == "length":
        logger.warning(f"[LLM-custom] ⚠ 출력이 max_tokens({max_tokens})에서 잘림")
    return parse_minutes_response(content), usage
