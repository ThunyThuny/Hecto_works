import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

###dylee : 실행 단계별 시간과 리소스 사용량을 기록하기 위한 계측 함수 추가
import resource
import subprocess


def _safe_round(value, digits=3):
    try:
        return round(float(value), digits)
    except Exception:
        return value


def get_process_memory_mb() -> float | None:
    try:
        return _safe_round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
    except Exception:
        return None


def get_gpu_resource_snapshot() -> list:
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return []

    snapshots = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) != 5:
            continue
        index, name, memory_total, memory_used, utilization_gpu = parts
        snapshots.append({
            'gpu_index': index,
            'name': name,
            'memory_total_mb': _safe_round(memory_total, 1),
            'memory_used_mb': _safe_round(memory_used, 1),
            'utilization_gpu_percent': _safe_round(utilization_gpu, 1),
        })
    return snapshots


def get_resource_snapshot() -> dict:
    return {
        'timestamp': datetime.now().isoformat(),
        'process_memory_mb': get_process_memory_mb(),
        'visible_cuda_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
        'gpu': get_gpu_resource_snapshot(),
    }


def build_stage_record(stage_name: str, start_time: float, extra: dict | None = None) -> dict:
    record = {
        'stage': stage_name,
        'elapsed_seconds': _safe_round(time.time() - start_time),
        'resource_snapshot': get_resource_snapshot(),
    }
    if extra:
        record.update(extra)
    return record


import soundfile as sf
import torch

from pipeline_gemma4 import (
    load_models,
    stage_convert,
    stage_stt,
    stage_diarization,
    stage_llm_with_prompt,
    LLM_MODEL_NAME,
)
from pipeline_utils import merge_results, load_prompt_template

DEFAULT_INPUT_AUDIO_PATH = Path("/workspace/data/nfs/datasets/dylee/Saerok/test_files/1hour.webm")
###dylee : 원본 계열 결과는 /workspace/Result/ori 아래에 저장
BASE_RESULT_DIR = Path("/workspace/Result/ori")
DEFAULT_MEETING_TYPE = os.getenv("MEETING_TYPE", "meeting")
DEFAULT_NUM_SPEAKERS = int(os.getenv("NUM_SPEAKERS", "0")) or None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("saerok.cli")


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


def build_output_dir(input_audio_path: Path, output_root: Path, timestamp: str) -> Path:
    folder_name = f"{input_audio_path.stem}_{timestamp}"
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


def main() -> int:
    profile_started_at = time.time()
    profile_started_iso = datetime.now().isoformat()
    profile_records = []
    args = parse_args()
    input_audio_path = Path(args.input_audio)
    meeting_type = args.meeting_type
    num_speakers = args.num_speakers or None
    output_root = Path(args.output_root)

    if not input_audio_path.exists():
        raise FileNotFoundError(f"input audio file not found: {input_audio_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = build_output_dir(input_audio_path, output_root, timestamp)
    wav_path = output_dir / f"{input_audio_path.stem}_{timestamp}_converted.wav"
    json_path = output_dir / f"{input_audio_path.stem}_{timestamp}_result.json"
    md_path = output_dir / f"{input_audio_path.stem}_{timestamp}_minutes.md"
    profile_path = output_dir / f"{input_audio_path.stem}_{timestamp}_profile.json"

    logger.info("input: %s", input_audio_path)
    logger.info("output dir: %s", output_dir)
    logger.info("meeting_type: %s", meeting_type)
    logger.info("?? ?? ??")
    stage_started_at = time.time()
    whisper_pool, models = load_models()
    profile_records.append(build_stage_record("model_loading", stage_started_at))
    whisper = whisper_pool[0]
    diar = models["diar_pool"][0]
    llm = models["llm"]

    try:
        stage_started_at = time.time()
        vad_model, vad_get_speech = load_silero_vad()
        profile_records.append(build_stage_record("vad_model_loading", stage_started_at))
        t_total = time.time()

        logger.info("Stage 1/4: ??? ??")
        stage_started_at = time.time()
        duration = stage_convert(str(input_audio_path), str(wav_path))
        profile_records.append(build_stage_record(
            "audio_conversion",
            stage_started_at,
            {"audio_duration_seconds": _safe_round(duration)},
        ))
        logger.info("??? ??: %.1fs (%s)", duration, format_duration(duration))

        stage_started_at = time.time()
        data, sr = sf.read(str(wav_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        waveform = torch.from_numpy(data)
        del data
        profile_records.append(build_stage_record(
            "audio_loading",
            stage_started_at,
            {"sample_rate": sr, "waveform_samples": len(waveform)},
        ))

        ###dylee : 원본 FastAPI와 동일하게 음성이 없으면 STT/pyannote 실행 전에 종료
        stage_started_at = time.time()
        has_speech = check_has_speech(vad_model, vad_get_speech, waveform)
        profile_records.append(build_stage_record(
            "vad_speech_check",
            stage_started_at,
            {"has_speech": has_speech},
        ))
        if not has_speech:
            logger.error("No speech detected")
            del waveform
            return 1

        ###dylee : 원본 FastAPI와 동일하게 STT와 pyannote를 서로 다른 GPU에서 병렬 실행
        logger.info("Stage 2/4 + Stage 3/4: STT + Diarization (parallel)")

        def run_stt_branch():
            started_at = time.time()
            branch_result = stage_stt(whisper, str(wav_path))
            record = build_stage_record(
                "stt",
                started_at,
                {"segment_count": len(branch_result)},
            )
            return branch_result, record

        def run_diarization_branch():
            started_at = time.time()
            branch_result = stage_diarization(diar, waveform, num_speakers=num_speakers)
            record = build_stage_record(
                "diarization",
                started_at,
                {"segment_count": len(branch_result)},
            )
            return branch_result, record

        analysis_started_at = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            stt_future = executor.submit(run_stt_branch)
            diarization_future = executor.submit(run_diarization_branch)
            transcription, stt_record = stt_future.result()
            diarization, diarization_record = diarization_future.result()

        profile_records.extend([stt_record, diarization_record])
        parallel_record = build_stage_record(
            "parallel_speech_analysis",
            analysis_started_at,
        )
        profile_records.append(parallel_record)
        stt_elapsed = stt_record["elapsed_seconds"]
        diarization_elapsed = diarization_record["elapsed_seconds"]
        logger.info("STT done: %.1fs / %d segments", stt_elapsed, len(transcription))
        logger.info("Diarization done: %.1fs / %d segments", diarization_elapsed, len(diarization))
        logger.info("Parallel speech analysis done: %.1fs", parallel_record["elapsed_seconds"])

        ###dylee : 원본 FastAPI와 동일하게 STT 결과가 비어 있으면 Merge 전에 종료
        if not transcription:
            logger.error("STT failed: no transcription segments")
            del waveform
            return 1

        del waveform

        ###dylee : 두 병렬 작업이 모두 완료된 이후 기존 위치에서 Merge 실행
        logger.info("Stage 4/4: Merge + LLM")
        stage_started_at = time.time()
        result = merge_results(diarization, transcription)
        profile_records.append(build_stage_record(
            "merge",
            stage_started_at,
            {"segment_count": len(result["segments"])},
        ))
        result["filename"] = input_audio_path.name
        result["created_at"] = datetime.now().isoformat()
        result["duration"] = round(duration, 3)
        result["duration_formatted"] = format_duration(duration)
        result["segment_count"] = len(result["segments"])
        result["whisper_model"] = "large-v3"
        result["diarization_model"] = "pyannote/speaker-diarization-community-1"
        result["llm_model"] = LLM_MODEL_NAME
        result["meeting_type"] = meeting_type
        result["input_path"] = str(input_audio_path)
        result["output_dir"] = str(output_dir)

        prompt_template = load_prompt_template(meeting_type)
        stage_started_at = time.time()
        minutes, usage = stage_llm_with_prompt(
            llm,
            result["segments"],
            prompt_template,
            None,
            True,
            None,
        )
        profile_records.append(build_stage_record(
            "llm_summary",
            stage_started_at,
            {"usage": usage},
        ))
        result["minutes"] = minutes
        result["llm_usage"] = usage
        result["profiling"] = {
            "started_at": profile_started_iso,
            "completed_at": datetime.now().isoformat(),
            "pipeline_elapsed_seconds": _safe_round(time.time() - profile_started_at),
            "input_path": str(input_audio_path),
            "output_dir": str(output_dir),
            "stages": profile_records,
            "final_resource_snapshot": get_resource_snapshot(),
        }

        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(minutes, encoding="utf-8")
        profile_path.write_text(
            json.dumps(result["profiling"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info("?? ??: %.1fs", time.time() - t_total)
        logger.info("JSON ??: %s", json_path)
        logger.info("??? ??: %s", md_path)
        logger.info("Profile result: %s", profile_path)
        return 0
    finally:
        if llm and hasattr(llm, "close"):
            try:
                llm.close()
            except Exception:
                pass
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
