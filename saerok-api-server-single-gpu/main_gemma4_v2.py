import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import soundfile as sf
import torch

from pipeline_gemma4_v2 import (
    DIARIZATION_MODEL_NAME,
    LLM_MODEL_NAME,
    WHISPER_MODEL_NAME,
    build_final_prompt,
    compact_agenda_for_prompt,
    load_diarization_model,
    load_llm_model,
    load_whisper_model,
    release_gpu_memory,
    stage_agenda_extract,
    stage_convert,
    stage_diarization,
    stage_llm_with_prompt,
    stage_stt,
)
from pipeline_utils_v2 import (
    MEETING_TYPES,
    build_agenda_transcript_text,
    build_transcript_text,
    load_prompt_file,
    load_prompt_template,
    merge_results,
)

###dylee : 단일 실행용 기본 입력/출력 경로 설정 유지
DEFAULT_INPUT_AUDIO_PATH = Path("/workspace/data/nfs/datasets/dylee/Saerok/test_files/1hour.webm")
BASE_RESULT_DIR = Path("/workspace/Result")
DEFAULT_MEETING_TYPE = os.getenv("MEETING_TYPE", "meeting")
DEFAULT_NUM_SPEAKERS = int(os.getenv("NUM_SPEAKERS", "0")) or None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("saerok.cli")


###dylee : 원본 서버 대신 테스트 실행 인자를 받는 CLI 진입 유지

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Saerok single-file pipeline runner")
    parser.add_argument("input_audio", nargs="?", default=str(DEFAULT_INPUT_AUDIO_PATH), help="input audio path")
    parser.add_argument("--meeting-type", default=DEFAULT_MEETING_TYPE, help="prompt meeting type")
    parser.add_argument("--num-speakers", type=int, default=DEFAULT_NUM_SPEAKERS, help="fixed speaker count")
    parser.add_argument("--output-root", default=str(BASE_RESULT_DIR), help="base output directory")
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


###dylee : 입력 파일명 기준 하위 폴더에 결과 저장

def build_output_dir(input_audio_path: Path, output_root: Path) -> Path:
    folder_name = input_audio_path.stem
    output_dir = output_root / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


###dylee : 원본 FastAPI와 동일한 Silero VAD 모델 로딩을 CLI 변수 방식으로 적용
def load_silero_vad():
    try:
        vad_model, utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad",
            source="github", trust_repo=True,
        )
    except Exception:
        vad_model, utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad",
            trust_repo=True,
        )
    logger.info("Silero VAD load done")
    return vad_model, utils[0]


###dylee : 원본 FastAPI와 동일하게 뒤 30초부터 확인하고 필요할 때 앞 90초를 검사
def check_has_speech(vad_model, vad_get_speech, waveform: torch.Tensor, sample_rate: int = 16000) -> bool:
    if vad_model is None:
        return True

    back_samples = sample_rate * 30
    front_samples = sample_rate * 90
    audio_sec = len(waveform) / sample_rate

    if len(waveform) <= back_samples + front_samples:
        t0 = time.time()
        timestamps = vad_get_speech(waveform, vad_model, sampling_rate=sample_rate)
        elapsed = time.time() - t0
        has_speech = len(timestamps) > 0
        logger.info("[VAD] full audio check (%.1fs) - %.0fms / speech: %s", audio_sec, elapsed * 1000, has_speech)
        return has_speech

    t0 = time.time()
    back = waveform[-back_samples:]
    back_ts = vad_get_speech(back, vad_model, sampling_rate=sample_rate)
    back_elapsed = time.time() - t0
    if back_ts:
        logger.info("[VAD] last 30s check - %.0fms / speech: True", back_elapsed * 1000)
        return True

    t1 = time.time()
    front = waveform[:front_samples]
    front_ts = vad_get_speech(front, vad_model, sampling_rate=sample_rate)
    front_elapsed = time.time() - t1
    has_speech = len(front_ts) > 0
    logger.info(
        "[VAD] last 30s %.0fms + first 90s %.0fms / speech: %s",
        back_elapsed * 1000, front_elapsed * 1000, has_speech,
    )
    return has_speech


###dylee : 요약 전 회의 길이 메타데이터 계산

def get_meeting_length(duration_seconds: float) -> dict:
    return {"seconds": round(duration_seconds, 3), "formatted": format_duration(duration_seconds)}


###dylee : 안건 분리/최종 요약 전에 전체 입력 토큰 수 계산

def get_input_token_count(llm, prompt_text: str, system_prompt: str = "") -> int:
    input_tokens = llm.tokenize((system_prompt + prompt_text).encode("utf-8"), add_bos=False)
    return len(input_tokens)


def main() -> int:
    args = parse_args()
    input_audio_path = Path(args.input_audio)
    meeting_type = args.meeting_type
    num_speakers = args.num_speakers or None
    output_root = Path(args.output_root)
    if not input_audio_path.exists():
        raise FileNotFoundError(f"input audio file not found: {input_audio_path}")
    if meeting_type not in MEETING_TYPES:
        raise ValueError(f"Unknown meeting_type: {meeting_type!r}. Available: {MEETING_TYPES}")
    if num_speakers is not None and num_speakers < 1:
        raise ValueError("--num-speakers must be 1 or greater")

    output_dir = build_output_dir(input_audio_path, output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = output_dir / f"{input_audio_path.stem}_{timestamp}_converted.wav"
    json_path = output_dir / f"{input_audio_path.stem}_{timestamp}_result.json"
    md_path = output_dir / f"{input_audio_path.stem}_{timestamp}_minutes.md"
    ###dylee : 안건 분리 결과를 별도 JSON 파일로 저장
    agenda_json_path = output_dir / f"{input_audio_path.stem}_{timestamp}_agenda_list.json"

    logger.info("input: %s", input_audio_path)
    logger.info("output dir: %s", output_dir)
    logger.info("meeting_type: %s", meeting_type)
    whisper = None
    diar = None
    llm = None

    try:
        vad_model, vad_get_speech = load_silero_vad()
        t_total = time.time()

        logger.info("Stage 1/5: audio convert")
        duration = stage_convert(str(input_audio_path), str(wav_path))
        meeting_length = get_meeting_length(duration)
        logger.info("audio duration: %.1fs (%s)", meeting_length["seconds"], meeting_length["formatted"])

        data, sr = sf.read(str(wav_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        waveform = torch.from_numpy(data)
        del data

        ###dylee : 원본 FastAPI와 동일하게 음성이 없으면 STT/pyannote 실행 전에 종료
        if not check_has_speech(vad_model, vad_get_speech, waveform):
            logger.error("No speech detected")
            del waveform
            return 1

        # A single GPU cannot safely keep Whisper, pyannote, and the 28GB LLM
        # resident together.  Load, execute, and release each model in order.
        logger.info("Stage 2/5: STT (single GPU, sequential)")
        stage_started_at = time.time()
        whisper = load_whisper_model()
        try:
            transcription = stage_stt(whisper, str(wav_path))
        finally:
            del whisper
            whisper = None
            release_gpu_memory()
        logger.info("STT done: %.1fs / %d segments", time.time() - stage_started_at, len(transcription))

        ###dylee : 원본 FastAPI와 동일하게 STT 결과가 비어 있으면 Merge 전에 종료
        if not transcription:
            logger.error("STT failed: no transcription segments")
            del waveform
            return 1

        logger.info("Stage 3/5: Diarization (single GPU, sequential)")
        stage_started_at = time.time()
        diar = load_diarization_model()
        try:
            diarization = stage_diarization(
                diar,
                waveform,
                num_speakers=num_speakers,
            )
        finally:
            del diar
            diar = None
            release_gpu_memory()
        logger.info(
            "Diarization done: %.1fs / %d segments",
            time.time() - stage_started_at,
            len(diarization),
        )

        del waveform
        del vad_model
        release_gpu_memory()

        logger.info("Stage 4/5: Merge")
        result = merge_results(diarization, transcription)
        result["filename"] = input_audio_path.name
        result["created_at"] = datetime.now().isoformat()
        result["duration"] = meeting_length["seconds"]
        result["duration_formatted"] = meeting_length["formatted"]
        result["segment_count"] = len(result["segments"])
        result["whisper_model"] = WHISPER_MODEL_NAME
        result["diarization_model"] = DIARIZATION_MODEL_NAME
        result["llm_model"] = LLM_MODEL_NAME
        result["meeting_type"] = meeting_type
        result["input_path"] = str(input_audio_path)
        result["output_dir"] = str(output_dir)

        # meeting keeps the v2 agenda-aware prompt.  Other supported meeting
        # types use their matching prompt instead of silently ignoring the CLI
        # argument.
        agenda_prompt_template = load_prompt_file("agenda_extract")
        final_prompt_template = (
            load_prompt_file("meeting_v2")
            if meeting_type == "meeting"
            else load_prompt_template(meeting_type)
        )

        logger.info("Stage 5/5: LLM agenda + summary (single GPU)")
        llm = load_llm_model()

        agenda_list = stage_agenda_extract(llm, result["segments"], prompt_template=agenda_prompt_template)
        result["agenda_list"] = agenda_list
        agenda_json_path.write_text(json.dumps(agenda_list, ensure_ascii=False, indent=2), encoding="utf-8")

        # Count exactly the prompts supplied to the LLM, including transcript
        # and compact agenda JSON.
        agenda_transcript = build_agenda_transcript_text(
            {"segments": result["segments"]},
            include_speakers=True,
        )
        agenda_prompt_text = agenda_prompt_template.replace(
            "{transcript_text}",
            agenda_transcript,
        ) if "{transcript_text}" in agenda_prompt_template else agenda_prompt_template
        transcript_text = build_transcript_text(
            {"segments": result["segments"]},
            include_speakers=True,
        )
        final_prompt_text = build_final_prompt(
            final_prompt_template,
            transcript_text,
            compact_agenda_for_prompt(agenda_list),
        )
        result["agenda_input_token_count"] = get_input_token_count(llm, agenda_prompt_text)
        result["final_input_token_count"] = get_input_token_count(llm, final_prompt_text)

        minutes, usage = stage_llm_with_prompt(
            llm,
            result["segments"],
            final_prompt_template,
            None,
            True,
            None,
            agenda_list=agenda_list,
        )
        result["minutes"] = minutes
        result["llm_usage"] = usage

        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(minutes, encoding="utf-8")

        logger.info("all done: %.1fs", time.time() - t_total)
        logger.info("JSON result: %s", json_path)
        logger.info("minutes result: %s", md_path)
        return 0
    finally:
        if whisper is not None:
            del whisper
        if diar is not None:
            del diar
        if llm is not None and hasattr(llm, "close"):
            try:
                llm.close()
            except Exception:
                logger.exception("LLM close failed")
        if llm is not None:
            del llm
        release_gpu_memory()


if __name__ == "__main__":
    raise SystemExit(main())
