"""Saerok Gemma 4 pipeline for one CUDA GPU.

The caller loads Whisper, diarization, and the LLM one at a time.  Keeping only
one large model resident is important on a single GPU, especially with a long-
context Gemma model.
"""

import gc
import json
import logging
import os
import subprocess
import time
import warnings

# Select one physical GPU before importing torch.  An existing
# CUDA_VISIBLE_DEVICES value always wins; otherwise SAEROK_CUDA_DEVICE selects
# the physical device and defaults to GPU 0.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.getenv("SAEROK_CUDA_DEVICE", "0"))

import torch
import soundfile as sf

warnings.filterwarnings("ignore")
MODELS_DIR = os.getenv("MODELS_DIR", "/workspace/data/nfs/models/llm/foundation")
os.makedirs(MODELS_DIR, exist_ok=True)

logger = logging.getLogger("saerok.pipeline")

from pipeline_utils_v2 import (
    AGENDA_SYSTEM_PROMPT,
    MINUTES_SYSTEM_PROMPT,
    build_agenda_transcript_text,
    build_agenda_transcripts_from_ranges,
    build_transcript_text,
    load_prompt_file,
    parse_agenda_response,
    parse_minutes_response,
    postprocess_diarization,
)

BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "16"))
DIAR_COLLAR = float(os.getenv("DIAR_COLLAR", "0.5"))
DIAR_MIN_DURATION = float(os.getenv("DIAR_MIN_DURATION", "0.3"))
DIAR_SMOOTH_MAX = float(os.getenv("DIAR_SMOOTH_MAX_DURATION", "1.0"))
HF_TOKEN = os.getenv("HF_TOKEN")
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3")
DIARIZATION_MODEL_NAME = os.getenv(
    "DIARIZATION_MODEL",
    "pyannote/speaker-diarization-community-1",
)
LLM_PATH = os.getenv("LLM_PATH", os.path.join(MODELS_DIR, "gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf"))
LLM_N_CTX = int(os.getenv("LLM_N_CTX", "131072"))
LLM_N_BATCH = int(os.getenv("LLM_N_BATCH", "512"))
LLM_N_THREADS = int(os.getenv("LLM_N_THREADS", "4"))
LLM_TEMPERATURE = 1.0
LLM_TOP_P = 0.95
LLM_TOP_K = 64
LLM_MAX_TOKENS = 8192
LLM_MODEL_NAME = "gemma-4-26B-A4B-it-UD-Q8_K_XL"


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU를 사용할 수 없습니다. nvidia-smi와 컨테이너 GPU 노출 설정을 확인하세요."
        )


def load_whisper_model():
    """Load one Whisper instance on the only visible CUDA device."""
    _require_cuda()
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    started_at = time.time()
    model = WhisperModel(
        WHISPER_MODEL_NAME,
        device="cuda",
        device_index=0,
        compute_type="float16",
    )
    logger.info("[Load] Whisper %s on cuda:0: %.1fs", WHISPER_MODEL_NAME, time.time() - started_at)
    return BatchedInferencePipeline(model=model)


def load_diarization_model():
    """Load one pyannote pipeline on the only visible CUDA device."""
    _require_cuda()
    from pyannote.audio import Pipeline as DiarPipeline

    started_at = time.time()
    kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
    model = DiarPipeline.from_pretrained(DIARIZATION_MODEL_NAME, **kwargs)
    if model is None:
        raise RuntimeError(
            "pyannote 모델을 불러오지 못했습니다. HF_TOKEN과 gated-model 사용 동의를 확인하세요."
        )
    model = model.to(torch.device("cuda:0"))
    logger.info(
        "[Load] Diarization %s on cuda:0: %.1fs",
        DIARIZATION_MODEL_NAME,
        time.time() - started_at,
    )
    return model


def load_llm_model():
    """Load Gemma on the only visible CUDA device."""
    _require_cuda()
    if not os.path.isfile(LLM_PATH):
        raise FileNotFoundError(f"Local Gemma model not found: {LLM_PATH}")

    from llama_cpp import Llama

    started_at = time.time()
    model = Llama(
        model_path=LLM_PATH,
        n_ctx=LLM_N_CTX,
        n_gpu_layers=-1,
        n_threads=LLM_N_THREADS,
        n_threads_batch=LLM_N_THREADS,
        main_gpu=0,
        verbose=False,
        flash_attn=True,
        n_batch=LLM_N_BATCH,
        swa_full=False,
    )
    logger.info(
        "[Load] LLM %s on cuda:0 (n_ctx=%d, n_batch=%d): %.1fs",
        LLM_MODEL_NAME,
        LLM_N_CTX,
        LLM_N_BATCH,
        time.time() - started_at,
    )
    return model


def release_gpu_memory() -> None:
    """Release Python references collected by the caller and clear CUDA caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
        vad_parameters=dict(onset=0.3, offset=0.35, min_silence_duration_ms=1000, speech_pad_ms=400, min_speech_duration_ms=250),
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
    if dump_path:
        try:
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(segments, f, ensure_ascii=False)
            logger.info(f"  [Diar] 원본 세그먼트 {len(segments)}개 덤프 → {dump_path}")
        except Exception as e:
            logger.warning(f"  [Diar] 덤프 실패: {e}")
    return postprocess_diarization(segments, collar=DIAR_COLLAR, min_duration=DIAR_MIN_DURATION, smooth_max_duration=DIAR_SMOOTH_MAX)


###dylee : 요약 전에 안건 분리용 LLM 단계를 추가

def stage_agenda_extract(llm, segments, prompt_template: str | None = None, system_prompt: str | None = None, include_speakers: bool = True):
    transcript = build_agenda_transcript_text({"segments": segments}, include_speakers=include_speakers)
    agenda_prompt_template = prompt_template or load_prompt_file("agenda_extract")
    if "{transcript_text}" in agenda_prompt_template:
        prompt = agenda_prompt_template.replace("{transcript_text}", transcript)
    else:
        prompt = agenda_prompt_template + "\n\n## 회의 녹취록\n\n" + transcript
    sys_prompt = system_prompt or AGENDA_SYSTEM_PROMPT
    response = llm.create_chat_completion(
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.2,
        top_p=0.9,
        top_k=40,
    )
    choices = response.get("choices", [])
    content = choices[0].get("message", {}).get("content") if choices else None
    parsed = parse_agenda_response(content)
    parsed = build_agenda_transcripts_from_ranges(segments, parsed, include_speakers=include_speakers)
    logger.info(f"[AGENDA] 안건 {parsed.get('agenda_count', 0)}개 추출 완료")
    return parsed


def compact_agenda_for_prompt(agenda_list):
    """Remove transcript copies that are already supplied as transcript_text."""
    if not isinstance(agenda_list, dict):
        return agenda_list
    compact = dict(agenda_list)
    compact["agendas"] = [
        {key: value for key, value in agenda.items() if key != "agenda_transcript"}
        for agenda in agenda_list.get("agendas", [])
        if isinstance(agenda, dict)
    ]
    compact["agenda_count"] = len(compact["agendas"])
    return compact


def build_final_prompt(prompt_template, transcript, agenda_list):
    if "{transcript_text}" in prompt_template:
        prompt = prompt_template.replace("{transcript_text}", transcript)
    else:
        prompt = prompt_template + "\n\n## 회의 녹취록\n\n" + transcript
    if agenda_list is not None:
        agenda_json = json.dumps(agenda_list, ensure_ascii=False, indent=2)
        if "{agenda_json}" in prompt:
            prompt = prompt.replace("{agenda_json}", agenda_json)
        else:
            prompt = "## 안건 분리 결과\n\n```json\n" + agenda_json + "\n```\n\n" + prompt
    return prompt


def _split_transcript_by_tokens(llm, transcript, max_chunk_tokens=45000):
    chunks = []
    current_lines = []
    current_tokens = 0
    for line in transcript.splitlines():
        line_tokens = len(llm.tokenize((line + "\n").encode("utf-8"), add_bos=False))
        if current_lines and current_tokens + line_tokens > max_chunk_tokens:
            chunks.append("\n".join(current_lines))
            current_lines = []
            current_tokens = 0
        current_lines.append(line)
        current_tokens += line_tokens
    if current_lines:
        chunks.append("\n".join(current_lines))
    return chunks


def _summarize_transcript_chunks(llm, transcript):
    """Map a transcript into compact, chronological chunk summaries."""
    chunks = _split_transcript_by_tokens(llm, transcript)
    summaries = []
    system_prompt = (
        "You compress one chronological section of a meeting transcript. "
        "Preserve agenda changes, decisions, evidence, disagreements, action items, "
        "owners, deadlines, numbers, and unresolved issues. Do not invent facts."
    )
    for index, chunk in enumerate(chunks, start=1):
        logger.warning(
            "[LLM-custom] summarizing overflow chunk %d/%d (%d chars)",
            index,
            len(chunks),
            len(chunk),
        )
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"## Transcript chunk {index}/{len(chunks)}\n\n{chunk}\n\n"
                        "Write a dense Korean factual summary in chronological order."
                    ),
                },
            ],
            max_tokens=2048,
            temperature=0.2,
            top_p=0.9,
            top_k=40,
        )
        choices = response.get("choices", [])
        content = choices[0].get("message", {}).get("content") if choices else None
        summaries.append(
            f"### 녹취 분할 요약 {index}/{len(chunks)}\n"
            + (content.strip() if content else "(분할 요약 생성 실패)")
        )
    return "\n\n".join(summaries)


###dylee : 최종 회의록 프롬프트에 agenda_list를 추가로 주입할 수 있게 확장

def stage_llm_with_prompt(llm, segments, prompt_template, system_prompt=None, include_speakers=True, sampling=None, agenda_list=None):
    transcript = build_transcript_text({"segments": segments}, include_speakers=include_speakers)
    compact_agenda = compact_agenda_for_prompt(agenda_list)
    prompt = build_final_prompt(prompt_template, transcript, compact_agenda)

    sys_prompt = system_prompt or MINUTES_SYSTEM_PROMPT
    s = sampling or {}
    temperature = s.get("temperature", LLM_TEMPERATURE)
    top_p = s.get("top_p", LLM_TOP_P)
    top_k = s.get("top_k", LLM_TOP_K)
    max_tokens = s.get("max_tokens", LLM_MAX_TOKENS)
    input_tokens = llm.tokenize((sys_prompt + prompt).encode("utf-8"), add_bos=False)
    n_input = len(input_tokens)
    n_ctx = llm.n_ctx()
    logger.info(f"[LLM-custom] 녹취록 길이: {len(transcript)}자 / 입력 토큰: {n_input} / n_ctx: {n_ctx} / max_tokens: {max_tokens} / include_speakers: {include_speakers}")
    if n_input + max_tokens > n_ctx:
        logger.warning(
            "[LLM-custom] input overflow after agenda compaction: %d + %d > %d; "
            "using chunk summaries",
            n_input,
            max_tokens,
            n_ctx,
        )
        transcript = _summarize_transcript_chunks(llm, transcript)
        prompt = build_final_prompt(prompt_template, transcript, compact_agenda)
        input_tokens = llm.tokenize((sys_prompt + prompt).encode("utf-8"), add_bos=False)
        n_input = len(input_tokens)
        logger.info("[LLM-custom] compacted final input tokens: %d", n_input)
    if n_input + max_tokens > n_ctx:
        raise ValueError(
            f"Final prompt still exceeds context after compaction: "
            f"{n_input} + {max_tokens} > {n_ctx}"
        )

    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
    response = llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k)
    choices = response.get("choices", [])
    if not choices:
        return parse_minutes_response(None), {}
    finish_reason = choices[0].get("finish_reason", "unknown")
    content = choices[0].get("message", {}).get("content")
    usage = response.get("usage", {})
    logger.info(f"[LLM-custom] 완료 — finish_reason: {finish_reason} / prompt_tokens: {usage.get('prompt_tokens', '?')} / completion_tokens: {usage.get('completion_tokens', '?')}")
    return parse_minutes_response(content), usage
