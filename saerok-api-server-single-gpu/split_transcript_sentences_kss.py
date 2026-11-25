import argparse
import json
import logging
from pathlib import Path

import kss

from split_transcript_sentences import (
    discover_inputs,
    find_word_character_ranges,
    format_timestamp,
    normalize_sentence_text,
    parse_text_transcript,
    resolve_json_input,
    sentence_timestamps,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("saerok.kss_sentence_splitter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a speaker transcript into Korean sentences with KSS. "
            "A transcript JSON is recommended because it preserves word timestamps."
        ),
    )
    parser.add_argument(
        "input_path",
        help="*_transcript.json, *_transcript.txt, or a directory",
    )
    parser.add_argument(
        "--output-dir",
        help="output directory; by default, files are written beside each input",
    )
    parser.add_argument(
        "--backend",
        default="fast",
        help=(
            "KSS backend (default: fast). Use 'auto' for morphology-based "
            "splitting; auto selects the much slower pecab on this server."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="KSS worker count (default: 1)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing KSS output files",
    )
    return parser.parse_args()


def split_with_offsets(
    text: str,
    backend: str,
    num_workers: int,
) -> list[tuple[str, int, int]]:
    pieces = kss.split_sentences(
        text,
        backend=backend,
        num_workers=num_workers,
        strip=False,
    )
    output = []
    cursor = 0
    for piece in pieces:
        if not piece:
            continue
        start = text.find(piece, cursor)
        if start < 0:
            stripped = piece.strip()
            start = text.find(stripped, cursor)
            if start < 0:
                raise ValueError(
                    f"KSS output could not be aligned to source text: {piece!r}"
                )
            piece = stripped
        end = start + len(piece)
        output.append((piece, start, end))
        cursor = end
    return output


def split_json_transcript(
    input_path: Path,
    backend: str,
    num_workers: int,
) -> list[dict]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError(f"'segments' list not found: {input_path}")

    output = []
    sentence_id = 1
    for segment_index, segment in enumerate(segments):
        segment_text = str(segment.get("text", ""))
        if not segment_text.strip():
            continue

        segment_start = float(segment.get("start", 0.0))
        segment_end = float(segment.get("end", segment_start))
        speaker = str(segment.get("speaker", "UNKNOWN"))
        word_ranges = find_word_character_ranges(
            segment_text,
            segment.get("words") or [],
        )

        sentence_parts = split_with_offsets(
            segment_text,
            backend,
            num_workers,
        )
        if not sentence_parts:
            sentence_parts = [(segment_text, 0, len(segment_text))]

        for sentence_text, char_start, char_end in sentence_parts:
            text = normalize_sentence_text(sentence_text)
            if not text:
                continue
            start, end = sentence_timestamps(
                char_start,
                char_end,
                word_ranges,
                segment_start,
                segment_end,
            )
            output.append(
                {
                    "sentence_id": sentence_id,
                    "source_segment_index": segment_index,
                    "speaker": speaker,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "start_time": format_timestamp(start),
                    "end_time": format_timestamp(end),
                    "text": text,
                }
            )
            sentence_id += 1
    return output


def split_text_transcript(
    input_path: Path,
    backend: str,
    num_workers: int,
) -> list[dict]:
    output = []
    sentence_id = 1
    for segment_index, segment in enumerate(parse_text_transcript(input_path)):
        for sentence_text, _, _ in split_with_offsets(
            segment["text"],
            backend,
            num_workers,
        ):
            text = normalize_sentence_text(sentence_text)
            if not text:
                continue
            output.append(
                {
                    "sentence_id": sentence_id,
                    "source_segment_index": segment_index,
                    "speaker": segment["speaker"],
                    "start_time": segment["timestamp"],
                    "end_time": None,
                    "text": text,
                }
            )
            sentence_id += 1
    return output


def output_paths(
    input_path: Path,
    output_dir: Path | None,
) -> tuple[Path, Path]:
    name = input_path.name
    if name.endswith("_transcript.json"):
        stem = name[: -len("_transcript.json")]
    elif name.endswith("_transcript.txt"):
        stem = name[: -len("_transcript.txt")]
    else:
        stem = input_path.stem
    destination = output_dir or input_path.parent
    destination.mkdir(parents=True, exist_ok=True)
    return (
        destination / f"{stem}_sentences_kss.json",
        destination / f"{stem}_sentences_kss.txt",
    )


def write_results(
    input_path: Path,
    sentences: list[dict],
    output_dir: Path | None,
    backend: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    json_path, text_path = output_paths(input_path, output_dir)
    if not overwrite and (json_path.exists() or text_path.exists()):
        raise FileExistsError(
            f"Output already exists. Use --overwrite: {json_path}, {text_path}"
        )

    json_path.write_text(
        json.dumps(
            {
                "source_path": str(input_path),
                "splitter": "kss",
                "kss_version": kss.__version__,
                "backend": backend,
                "sentence_count": len(sentences),
                "sentences": sentences,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = []
    for item in sentences:
        if item.get("end_time"):
            timestamp = f"{item['start_time']}-{item['end_time']}"
        else:
            timestamp = item["start_time"]
        lines.append(f"{item['speaker']} [{timestamp}]: {item['text']}")
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, text_path


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir) if args.output_dir else None
    inputs = discover_inputs(input_path)
    if not inputs:
        logger.error("No transcript files found: %s", input_path)
        return 1

    failures = []
    for index, source_path in enumerate(inputs, start=1):
        logger.info("[%d/%d] Splitting with KSS: %s", index, len(inputs), source_path)
        try:
            json_input = resolve_json_input(source_path)
            if json_input:
                sentences = split_json_transcript(
                    json_input,
                    args.backend,
                    args.num_workers,
                )
                source_for_output = json_input
            else:
                sentences = split_text_transcript(
                    source_path,
                    args.backend,
                    args.num_workers,
                )
                source_for_output = source_path
            json_path, text_path = write_results(
                source_for_output,
                sentences,
                output_dir,
                args.backend,
                args.overwrite,
            )
            logger.info(
                "Saved %d KSS sentences: %s, %s",
                len(sentences),
                json_path,
                text_path,
            )
        except Exception as error:
            failures.append((str(source_path), str(error)))
            logger.exception("Failed: %s", source_path)

    if failures:
        logger.error("Completed with %d failure(s): %s", len(failures), failures)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
