"""Batch speaker/timestamp transcripts on physical GPU 5 only.

One invocation creates one result folder and one TXT per supported recording.
Converted audio is temporary. No Gemma, summaries, or persistent JSON output.
TXT contains only the speaker/timestamp transcript, without file metadata headers.
"""

import argparse
import logging
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unicodedata

GPU_INDEX = 5
DEFAULT_INPUT_DIR = Path("/workspace/EXP_Data/기계학습")
DEFAULT_OUTPUT_ROOT = Path("/workspace/Result/transcript_gpu5")
DEFAULT_WHISPER_MODEL = Path("/workspace/data/nfs/models/audio/task/faster-whisper-large-v3")
DEFAULT_DIARIZATION_MODEL = Path(
    "/workspace/data/nfs/models/audio/task/pyannote-speaker-diarization-community-1/config.yaml"
)
SUPPORTED_EXTENSIONS = {
    ".aac", ".flac", ".m4a", ".mkv", ".mov", ".mp3", ".mp4", ".ogg",
    ".opus", ".wav", ".webm", ".wma",
}
KST = timezone(timedelta(hours=9))
logger = logging.getLogger("saerok.transcript.gpu5.batch")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe recordings into one TXT per file using physical GPU 5 only."
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--num-speakers", type=int, help="optional fixed speaker count")
    args = parser.parse_args()
    if args.num_speakers is not None and args.num_speakers < 1:
        parser.error("--num-speakers must be 1 or greater")
    return args


def select_gpu5():
    """Resolve the physical index and replace any inherited CUDA device mask."""
    completed = subprocess.run(
        [
            "nvidia-smi", f"--id={GPU_INDEX}",
            "--query-gpu=uuid", "--format=csv,noheader",
        ],
        capture_output=True, text=True, check=True, timeout=15,
    )
    gpu_uuid = completed.stdout.strip()
    if not re.fullmatch(
        r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
        gpu_uuid,
    ):
        raise RuntimeError(f"Cannot identify physical GPU {GPU_INDEX}: {gpu_uuid!r}")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    os.environ["SAEROK_CUDA_DEVICE"] = gpu_uuid
    return gpu_uuid


def normalize_uuid(value):
    if isinstance(value, bytes):
        value = value.hex() if len(value) == 16 else value.decode("ascii")
    return str(value).lower().removeprefix("gpu-").replace("-", "")


def validate_gpu5(torch, expected_uuid):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one available CUDA GPU: physical GPU 5")
    properties = torch.cuda.get_device_properties(0)
    actual_uuid = getattr(properties, "uuid", None)
    if actual_uuid is None or normalize_uuid(actual_uuid) != normalize_uuid(expected_uuid):
        raise RuntimeError(
            f"GPU identity mismatch: expected {expected_uuid}, got {actual_uuid}"
        )
    return {
        "physical_index": GPU_INDEX,
        "uuid": expected_uuid,
        "logical_device": "cuda:0",
        "name": properties.name,
    }


def configure_local_models():
    whisper = Path(os.getenv("WHISPER_MODEL", str(DEFAULT_WHISPER_MODEL))).expanduser().resolve()
    diarization = Path(
        os.getenv("DIARIZATION_MODEL", str(DEFAULT_DIARIZATION_MODEL))
    ).expanduser().resolve()
    if diarization.is_dir():
        diarization = diarization / "config.yaml"
    for name in ("model.bin", "config.json", "tokenizer.json"):
        if not (whisper / name).is_file():
            raise FileNotFoundError(f"Local Whisper file not found: {whisper / name}")
    if not diarization.is_file():
        raise FileNotFoundError(f"Local pyannote config not found: {diarization}")
    # The v2 module captures these values when it is imported.
    os.environ["WHISPER_MODEL"] = str(whisper)
    os.environ["DIARIZATION_MODEL"] = str(diarization)
    return whisper, diarization




def diarize(model, waveform, num_speakers):
    from pipeline_gemma4_v2 import DIAR_COLLAR, DIAR_MIN_DURATION, DIAR_SMOOTH_MAX
    from pipeline_utils_v2 import postprocess_diarization

    audio = {"waveform": waveform.unsqueeze(0), "sample_rate": 16000}
    kwargs = {} if num_speakers is None else {"num_speakers": num_speakers}
    output = model(audio, **kwargs)
    # The transcript assigns one speaker per word; use disjoint turns when the
    # community-1 model provides them, instead of its overlapping annotation.
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(output, "speaker_diarization", output)
    segments = [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    return postprocess_diarization(
        segments, collar=DIAR_COLLAR, min_duration=DIAR_MIN_DURATION,
        smooth_max_duration=DIAR_SMOOTH_MAX,
    )

def natural_sort_key(path):
    name = unicodedata.normalize("NFC", path.name).casefold()
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))


def result_filename(index, input_audio):
    name = unicodedata.normalize("NFC", input_audio.name).replace("\n", "_").replace("\r", "_")
    while len(name.encode("utf-8")) > 210:
        name = name[:-1]
    # The ordinal prevents collisions, including equal stems with different extensions.
    return f"{index:03d}_{name}_transcript.txt"


def transcribe_one(input_audio, scratch_dir, runtime, num_speakers):
    whisper = None
    diarization_model = None
    waveform = None
    vad_model = None
    vad_utils = None
    try:
        wav_path = scratch_dir / "converted.wav"
        runtime.stage_convert(str(input_audio), str(wav_path))
        data, sample_rate = runtime.sf.read(str(wav_path), dtype="float32")
        if sample_rate != 16000:
            raise RuntimeError(f"Expected sample rate 16000, got {sample_rate}")
        if data.ndim > 1:
            data = data.mean(axis=1)
        waveform = runtime.torch.from_numpy(data)
        del data

        # Check all audio, including speech that appears only in the middle.
        vad_model, vad_utils = runtime.torch.hub.load(
            "snakers4/silero-vad", "silero_vad", source="github", trust_repo=True,
        )
        has_speech = bool(vad_utils[0](waveform, vad_model, sampling_rate=sample_rate))
        vad_model = None
        vad_utils = None
        if not has_speech:
            raise RuntimeError("No speech detected in the recording")

        logger.info("STT on GPU 5: %s", input_audio.name)
        whisper = runtime.load_whisper_model()
        try:
            transcription = runtime.stage_stt(whisper, str(wav_path))
        finally:
            whisper = None
            runtime.release_gpu_memory()
        if not transcription:
            raise RuntimeError("STT returned no transcription segments")

        logger.info("Speaker diarization on GPU 5: %s", input_audio.name)
        diarization_model = runtime.load_diarization_model()
        try:
            diarization = diarize(diarization_model, waveform, num_speakers)
        finally:
            diarization_model = None
            runtime.release_gpu_memory()
        if not diarization:
            raise RuntimeError("Diarization returned no speaker segments")

        result = runtime.merge_results(diarization, transcription)
        if not result["segments"]:
            raise RuntimeError("Merge returned an empty transcript")
        transcript = runtime.build_transcript_text(result, include_speakers=True)
        if not transcript.strip():
            raise RuntimeError("Transcript text is empty")
        return transcript + '\n'
    finally:
        whisper = None
        diarization_model = None
        waveform = None
        vad_model = None
        vad_utils = None
        runtime.release_gpu_memory()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    input_files = sorted(
        (path for path in input_dir.iterdir()
         if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS),
        key=lambda path: (natural_sort_key(path), path.name),
    )
    if not input_files:
        raise RuntimeError(f"No supported audio/video files found in: {input_dir}")
    configure_local_models()
    expected_uuid = select_gpu5()

    # GPU 5 is pinned before any CUDA-related imports or model initialization.
    import soundfile as sf
    import torch

    gpu = validate_gpu5(torch, expected_uuid)
    from pipeline_gemma4_v2 import (
        load_diarization_model, load_whisper_model, release_gpu_memory,
        stage_convert, stage_stt,
    )
    from pipeline_utils_v2 import build_transcript_text, merge_results

    runtime = SimpleNamespace(
        sf=sf, torch=torch, load_diarization_model=load_diarization_model,
        load_whisper_model=load_whisper_model, release_gpu_memory=release_gpu_memory,
        stage_convert=stage_convert, stage_stt=stage_stt,
        build_transcript_text=build_transcript_text, merge_results=merge_results,
    )
    logger.info("Physical GPU 5 isolated as cuda:0: %s (%s)", gpu["name"], gpu["uuid"])
    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(args.output_root).expanduser().resolve() / f"EXP_Mutimodal_machinelearning_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    logger.info("Input recordings: %d; results: %s", len(input_files), output_dir)
    started = time.perf_counter()
    failures = 0
    for index, input_audio in enumerate(input_files, start=1):
        result_path = output_dir / result_filename(index, input_audio)
        logger.info("[%d/%d] Start: %s", index, len(input_files), input_audio.name)
        try:
            with tempfile.TemporaryDirectory(prefix="saerok_gpu5_") as scratch:
                text = transcribe_one(input_audio, Path(scratch), runtime, args.num_speakers)
            result_path.write_text(text, encoding="utf-8")
            logger.info("[%d/%d] Saved: %s", index, len(input_files), result_path)
        except Exception as exc:
            failures += 1
            logger.exception("[%d/%d] Failed: %s", index, len(input_files), input_audio.name)
            # Failed inputs keep an empty TXT; diagnostic details stay in console logs.
            result_path.write_text("", encoding="utf-8")
        # Run after the except block so exception tracebacks cannot retain models.
        release_gpu_memory()
    logger.info(
        "Batch complete: %d succeeded, %d failed, %.1fs; results: %s",
        len(input_files) - failures, failures, time.perf_counter() - started, output_dir,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("GPU 5 batch transcription failed")
        raise SystemExit(1)
