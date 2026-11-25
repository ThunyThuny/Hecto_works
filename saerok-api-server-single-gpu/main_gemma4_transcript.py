import argparse
import json
import logging
import os
import resource
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import soundfile as sf
import torch

from pipeline_gemma4 import (
    HF_TOKEN,
    PYANNOTE_MODEL_PATH,
    stage_convert,
    stage_diarization,
    stage_stt,
)
from pipeline_utils import build_transcript_text, merge_results


DEFAULT_INPUT_AUDIO_PATH = Path(
    "/workspace/data/nfs/datasets/dylee/Saerok/test_files/1hour.webm"
)
DEFAULT_OUTPUT_ROOT = Path("/workspace/Result/stt_transcript")
DEFAULT_NUM_SPEAKERS = int(os.getenv("NUM_SPEAKERS", "0")) or None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("saerok.transcript")


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
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return []

    snapshots = []
    for line in completed.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        index, name, memory_total, memory_used, utilization_gpu = parts
        snapshots.append(
            {
                "gpu_index": index,
                "name": name,
                "memory_total_mb": _safe_round(memory_total, 1),
                "memory_used_mb": _safe_round(memory_used, 1),
                "utilization_gpu_percent": _safe_round(utilization_gpu, 1),
            }
        )
    return snapshots


def get_resource_snapshot() -> dict:
    return {
        "timestamp": datetime.now().isoformat(),
        "process_memory_mb": get_process_memory_mb(),
        "visible_cuda_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "gpu": get_gpu_resource_snapshot(),
    }


def build_stage_record(
    stage_name: str,
    start_time: float,
    extra: dict | None = None,
) -> dict:
    record = {
        "stage": stage_name,
        "elapsed_seconds": _safe_round(time.time() - start_time),
        "resource_snapshot": get_resource_snapshot(),
    }
    if extra:
        record.update(extra)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract STT, diarization, and merged speaker transcript",
    )
    parser.add_argument(
        "input_audio",
        nargs="?",
        default=str(DEFAULT_INPUT_AUDIO_PATH),
        help="input media path",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=DEFAULT_NUM_SPEAKERS,
        help="fixed speaker count",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="base output directory",
    )
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def build_output_dir(
    input_audio_path: Path,
    output_root: Path,
    timestamp: str,
) -> Path:
    output_dir = output_root / f"{input_audio_path.stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_speech_models():
    """Load only the models needed for STT and speaker diarization."""
    if not os.path.isfile(PYANNOTE_MODEL_PATH):
        raise FileNotFoundError(
            f"Local pyannote config not found: {PYANNOTE_MODEL_PATH}"
        )

    from faster_whisper import BatchedInferencePipeline, WhisperModel
    from pyannote.audio import Pipeline as DiarPipeline

    logger.info("Loading Whisper large-v3 on logical GPU 0")
    whisper_model = WhisperModel(
        "large-v3",
        device="cuda",
        device_index=0,
        compute_type="float16",
    )
    whisper = BatchedInferencePipeline(model=whisper_model)

    logger.info("Loading pyannote diarization on logical GPU 1")
    diarization = DiarPipeline.from_pretrained(
        PYANNOTE_MODEL_PATH,
        token=HF_TOKEN or None,
    )
    diarization = diarization.to(torch.device("cuda:1"))
    return whisper, diarization


def load_silero_vad():
    try:
        vad_model, utils = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            source="github",
            trust_repo=True,
        )
    except Exception:
        vad_model, utils = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            trust_repo=True,
        )
    logger.info("Silero VAD load done")
    return vad_model, utils[0]


def check_has_speech(
    vad_model,
    vad_get_speech,
    waveform: torch.Tensor,
    sample_rate: int = 16000,
) -> bool:
    if vad_model is None:
        return True

    back_samples = sample_rate * 30
    front_samples = sample_rate * 90
    audio_seconds = len(waveform) / sample_rate

    if len(waveform) <= back_samples + front_samples:
        started_at = time.time()
        timestamps = vad_get_speech(
            waveform,
            vad_model,
            sampling_rate=sample_rate,
        )
        logger.info(
            "[VAD] full audio check (%.1fs) - %.0fms / speech: %s",
            audio_seconds,
            (time.time() - started_at) * 1000,
            bool(timestamps),
        )
        return bool(timestamps)

    started_at = time.time()
    back_timestamps = vad_get_speech(
        waveform[-back_samples:],
        vad_model,
        sampling_rate=sample_rate,
    )
    back_elapsed = time.time() - started_at
    if back_timestamps:
        logger.info(
            "[VAD] last 30s check - %.0fms / speech: True",
            back_elapsed * 1000,
        )
        return True

    started_at = time.time()
    front_timestamps = vad_get_speech(
        waveform[:front_samples],
        vad_model,
        sampling_rate=sample_rate,
    )
    front_elapsed = time.time() - started_at
    has_speech = bool(front_timestamps)
    logger.info(
        "[VAD] last 30s %.0fms + first 90s %.0fms / speech: %s",
        back_elapsed * 1000,
        front_elapsed * 1000,
        has_speech,
    )
    return has_speech


def main() -> int:
    pipeline_started_at = time.time()
    pipeline_started_iso = datetime.now().isoformat()
    profile_records = []
    args = parse_args()
    input_audio_path = Path(args.input_audio)
    output_root = Path(args.output_root)
    num_speakers = args.num_speakers or None

    if not input_audio_path.exists():
        raise FileNotFoundError(f"input media file not found: {input_audio_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = build_output_dir(input_audio_path, output_root, timestamp)
    output_prefix = f"{input_audio_path.stem}_{timestamp}"
    wav_path = output_dir / f"{output_prefix}_converted.wav"
    stt_path = output_dir / f"{output_prefix}_stt.json"
    diarization_path = output_dir / f"{output_prefix}_diarization.json"
    transcript_json_path = output_dir / f"{output_prefix}_transcript.json"
    transcript_text_path = output_dir / f"{output_prefix}_transcript.txt"
    profile_path = output_dir / f"{output_prefix}_profile.json"

    logger.info("Input: %s", input_audio_path)
    logger.info("Output directory: %s", output_dir)

    stage_started_at = time.time()
    whisper, diarization_model = load_speech_models()
    profile_records.append(build_stage_record("model_loading", stage_started_at))

    try:
        stage_started_at = time.time()
        vad_model, vad_get_speech = load_silero_vad()
        profile_records.append(
            build_stage_record("vad_model_loading", stage_started_at)
        )

        stage_started_at = time.time()
        duration = stage_convert(str(input_audio_path), str(wav_path))
        profile_records.append(
            build_stage_record(
                "audio_conversion",
                stage_started_at,
                {"audio_duration_seconds": _safe_round(duration)},
            )
        )

        stage_started_at = time.time()
        data, sample_rate = sf.read(str(wav_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        waveform = torch.from_numpy(data)
        del data
        profile_records.append(
            build_stage_record(
                "audio_loading",
                stage_started_at,
                {
                    "sample_rate": sample_rate,
                    "waveform_samples": len(waveform),
                },
            )
        )

        stage_started_at = time.time()
        has_speech = check_has_speech(
            vad_model,
            vad_get_speech,
            waveform,
            sample_rate=sample_rate,
        )
        profile_records.append(
            build_stage_record(
                "vad_speech_check",
                stage_started_at,
                {"has_speech": has_speech},
            )
        )
        if not has_speech:
            logger.error("No speech detected")
            return 1

        def run_stt_branch():
            started_at = time.time()
            branch_result = stage_stt(whisper, str(wav_path))
            return branch_result, build_stage_record(
                "stt",
                started_at,
                {"segment_count": len(branch_result)},
            )

        def run_diarization_branch():
            started_at = time.time()
            branch_result = stage_diarization(
                diarization_model,
                waveform,
                num_speakers=num_speakers,
            )
            return branch_result, build_stage_record(
                "diarization",
                started_at,
                {"segment_count": len(branch_result)},
            )

        analysis_started_at = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            stt_future = executor.submit(run_stt_branch)
            diarization_future = executor.submit(run_diarization_branch)
            transcription, stt_record = stt_future.result()
            diarization, diarization_record = diarization_future.result()
        profile_records.extend([stt_record, diarization_record])
        profile_records.append(
            build_stage_record("parallel_speech_analysis", analysis_started_at)
        )

        if not transcription:
            logger.error("STT failed: no transcription segments")
            return 1

        stt_path.write_text(
            json.dumps(transcription, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        diarization_path.write_text(
            json.dumps(diarization, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        stage_started_at = time.time()
        transcript_result = merge_results(diarization, transcription)
        profile_records.append(
            build_stage_record(
                "merge",
                stage_started_at,
                {"segment_count": len(transcript_result["segments"])},
            )
        )

        transcript_result.update(
            {
                "filename": input_audio_path.name,
                "created_at": datetime.now().isoformat(),
                "duration": round(duration, 3),
                "duration_formatted": format_duration(duration),
                "segment_count": len(transcript_result["segments"]),
                "whisper_model": "large-v3",
                "diarization_model": "pyannote/speaker-diarization-community-1",
                "input_path": str(input_audio_path),
                "output_dir": str(output_dir),
                "stt_result_path": str(stt_path),
                "diarization_result_path": str(diarization_path),
            }
        )
        transcript_text = build_transcript_text(
            transcript_result,
            include_speakers=True,
        )
        transcript_json_path.write_text(
            json.dumps(transcript_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        transcript_text_path.write_text(transcript_text, encoding="utf-8")

        profile = {
            "started_at": pipeline_started_iso,
            "completed_at": datetime.now().isoformat(),
            "pipeline_elapsed_seconds": _safe_round(
                time.time() - pipeline_started_at
            ),
            "input_path": str(input_audio_path),
            "output_dir": str(output_dir),
            "stages": profile_records,
            "final_resource_snapshot": get_resource_snapshot(),
        }
        profile_path.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info("STT result: %s", stt_path)
        logger.info("Diarization result: %s", diarization_path)
        logger.info("Transcript JSON: %s", transcript_json_path)
        logger.info("Transcript text: %s", transcript_text_path)
        logger.info("Profile result: %s", profile_path)
        return 0
    finally:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
