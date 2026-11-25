import argparse
import json
import logging
import re
from pathlib import Path

from kiwipiepy import Kiwi


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("saerok.sentence_splitter")

TRANSCRIPT_LINE_PATTERN = re.compile(
    r"^(?P<speaker>\S+)\s+\[(?P<timestamp>[^\]]+)\]:\s*(?P<text>.*)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a speaker transcript into Korean sentences with kiwipiepy. "
            "A transcript JSON is recommended because it preserves word timestamps."
        ),
    )
    parser.add_argument(
        "--input_path",
        help="*_transcript.json, *_transcript.txt, or a directory",
    )
    parser.add_argument(
        "--output-dir",
        help="output directory; by default, files are written beside each input",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing sentence output files",
    )
    return parser.parse_args()


def format_timestamp(seconds: float) -> str:
    total_milliseconds = max(0, round(float(seconds) * 1000))
    total_seconds, milliseconds = divmod(total_milliseconds, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        value = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        value = f"{minutes:02d}:{seconds:02d}"
    if milliseconds:
        value += f".{milliseconds:03d}"
    return value


def normalize_sentence_text(text: str) -> str:
    return " ".join(text.split()).strip()


def find_word_character_ranges(segment_text: str, words: list[dict]) -> list[dict]:
    """Map Whisper words to character ranges in the merged segment text."""
    ranges = []
    cursor = 0
    for word in words:
        token = str(word.get("word", "")).strip()
        if not token:
            continue

        start_index = segment_text.find(token, cursor)
        if start_index < 0:
            # Fallback for spacing differences between merged text and word tokens.
            start_index = segment_text.find(token)
        if start_index < 0:
            continue

        end_index = start_index + len(token)
        ranges.append(
            {
                "char_start": start_index,
                "char_end": end_index,
                "start": word.get("start"),
                "end": word.get("end"),
            }
        )
        cursor = end_index
    return ranges


def sentence_timestamps(
    sentence_start: int,
    sentence_end: int,
    word_ranges: list[dict],
    segment_start: float,
    segment_end: float,
) -> tuple[float, float]:
    overlapping = [
        item
        for item in word_ranges
        if item["char_end"] > sentence_start and item["char_start"] < sentence_end
    ]
    timed = [
        item
        for item in overlapping
        if item.get("start") is not None and item.get("end") is not None
    ]
    if timed:
        return float(timed[0]["start"]), float(timed[-1]["end"])
    return float(segment_start), float(segment_end)


def split_json_transcript(input_path: Path, kiwi: Kiwi) -> list[dict]:
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

        sentences = kiwi.split_into_sents(segment_text)
        if not sentences:
            sentences = [type("Sentence", (), {
                "text": segment_text,
                "start": 0,
                "end": len(segment_text),
            })()]

        for sentence in sentences:
            text = normalize_sentence_text(sentence.text)
            if not text:
                continue
            start, end = sentence_timestamps(
                sentence.start,
                sentence.end,
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


def parse_text_transcript(input_path: Path) -> list[dict]:
    segments = []
    current = None
    for raw_line in input_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = TRANSCRIPT_LINE_PATTERN.match(line)
        if match:
            if current:
                segments.append(current)
            current = {
                "speaker": match.group("speaker"),
                "timestamp": match.group("timestamp"),
                "text": match.group("text"),
            }
        elif current:
            current["text"] += " " + line
        else:
            raise ValueError(
                f"Transcript line does not include speaker/timestamp: {raw_line!r}"
            )
    if current:
        segments.append(current)
    return segments


def split_text_transcript(input_path: Path, kiwi: Kiwi) -> list[dict]:
    output = []
    sentence_id = 1
    for segment_index, segment in enumerate(parse_text_transcript(input_path)):
        for sentence in kiwi.split_into_sents(segment["text"]):
            text = normalize_sentence_text(sentence.text)
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


def resolve_json_input(input_path: Path) -> Path | None:
    if input_path.suffix.lower() == ".json":
        return input_path
    if input_path.name.endswith("_transcript.txt"):
        candidate = input_path.with_suffix(".json")
        if candidate.is_file():
            return candidate
    return None


def output_paths(input_path: Path, output_dir: Path | None) -> tuple[Path, Path]:
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
        destination / f"{stem}_sentences.json",
        destination / f"{stem}_sentences.txt",
    )


def write_results(
    input_path: Path,
    sentences: list[dict],
    output_dir: Path | None,
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


def discover_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    json_inputs = sorted(input_path.rglob("*_transcript.json"))
    if json_inputs:
        return json_inputs
    return sorted(input_path.rglob("*_transcript.txt"))


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir) if args.output_dir else None
    inputs = discover_inputs(input_path)
    if not inputs:
        logger.error("No transcript files found: %s", input_path)
        return 1

    kiwi = Kiwi()
    failures = []
    for index, source_path in enumerate(inputs, start=1):
        logger.info("[%d/%d] Splitting: %s", index, len(inputs), source_path)
        try:
            json_input = resolve_json_input(source_path)
            if json_input:
                sentences = split_json_transcript(json_input, kiwi)
                source_for_output = json_input
            else:
                sentences = split_text_transcript(source_path, kiwi)
                source_for_output = source_path
            json_path, text_path = write_results(
                source_for_output,
                sentences,
                output_dir,
                args.overwrite,
            )
            logger.info(
                "Saved %d sentences: %s, %s",
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
