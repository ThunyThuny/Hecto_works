import argparse
import json
import re
import statistics
from pathlib import Path


TRANSCRIPT_LINE_PATTERN = re.compile(r"^\S+ \[[^]]+\]:\s*")
END_PUNCTUATION_PATTERN = re.compile(r"""[.!?…。！？]["'”’)]?$""")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Kiwi and KSS sentence splitting results",
    )
    parser.add_argument("transcript_path", help="original *_transcript.txt")
    parser.add_argument("--kiwi-json", help="Kiwi *_sentences.json")
    parser.add_argument("--kss-json", help="KSS *_sentences_kss.json")
    parser.add_argument("--output", help="comparison JSON output path")
    return parser.parse_args()


def default_paths(transcript_path: Path) -> tuple[Path, Path, Path]:
    if not transcript_path.name.endswith("_transcript.txt"):
        raise ValueError("Input filename must end with '_transcript.txt'")
    stem = transcript_path.name[: -len("_transcript.txt")]
    return (
        transcript_path.parent / f"{stem}_sentences.json",
        transcript_path.parent / f"{stem}_sentences_kss.json",
        transcript_path.parent / f"{stem}_split_comparison.json",
    )


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def read_original_text(path: Path) -> str:
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            texts.append(TRANSCRIPT_LINE_PATTERN.sub("", line))
    return compact("".join(texts))


def read_sentences(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sentences = payload.get("sentences")
    if not isinstance(sentences, list):
        raise ValueError(f"'sentences' list not found: {path}")
    return sentences


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0
    index = max(0, min(len(values) - 1, int(len(values) * ratio) - 1))
    return sorted(values)[index]


def boundaries(sentences: list[dict]) -> set[int]:
    output = set()
    cursor = 0
    for sentence in sentences[:-1]:
        cursor += len(compact(str(sentence.get("text", ""))))
        output.add(cursor)
    return output


def metrics(sentences: list[dict], original_text: str) -> dict:
    texts = [str(item.get("text", "")) for item in sentences]
    lengths = [len(text) for text in texts]
    durations = [
        float(item["end"]) - float(item["start"])
        for item in sentences
        if item.get("start") is not None and item.get("end") is not None
    ]
    reconstructed = compact("".join(texts))
    return {
        "sentence_count": len(sentences),
        "content_preserved": reconstructed == original_text,
        "non_whitespace_characters": len(reconstructed),
        "characters": {
            "min": min(lengths) if lengths else 0,
            "median": statistics.median(lengths) if lengths else 0,
            "mean": round(statistics.mean(lengths), 3) if lengths else 0,
            "p95": percentile(lengths, 0.95),
            "max": max(lengths) if lengths else 0,
        },
        "short_sentences_le_5": sum(length <= 5 for length in lengths),
        "long_sentences_gt_100": sum(length > 100 for length in lengths),
        "without_terminal_punctuation": sum(
            not END_PUNCTUATION_PATTERN.search(text) for text in texts
        ),
        "duration_seconds": {
            "median": round(statistics.median(durations), 3) if durations else 0,
            "mean": round(statistics.mean(durations), 3) if durations else 0,
            "p95": round(percentile(durations, 0.95), 3),
            "max": round(max(durations), 3) if durations else 0,
            "invalid_count": sum(duration < 0 for duration in durations),
            "zero_count": sum(duration == 0 for duration in durations),
        },
    }


def main() -> int:
    args = parse_args()
    transcript_path = Path(args.transcript_path)
    default_kiwi, default_kss, default_output = default_paths(transcript_path)
    kiwi_path = Path(args.kiwi_json) if args.kiwi_json else default_kiwi
    kss_path = Path(args.kss_json) if args.kss_json else default_kss
    output_path = Path(args.output) if args.output else default_output

    original_text = read_original_text(transcript_path)
    kiwi_sentences = read_sentences(kiwi_path)
    kss_sentences = read_sentences(kss_path)
    kiwi_boundaries = boundaries(kiwi_sentences)
    kss_boundaries = boundaries(kss_sentences)
    common = kiwi_boundaries & kss_boundaries
    union = kiwi_boundaries | kss_boundaries

    comparison = {
        "transcript_path": str(transcript_path),
        "original_non_whitespace_characters": len(original_text),
        "kiwi_path": str(kiwi_path),
        "kss_path": str(kss_path),
        "kiwi": metrics(kiwi_sentences, original_text),
        "kss": metrics(kss_sentences, original_text),
        "boundaries": {
            "kiwi_count": len(kiwi_boundaries),
            "kss_count": len(kss_boundaries),
            "common_count": len(common),
            "kiwi_only_count": len(kiwi_boundaries - kss_boundaries),
            "kss_only_count": len(kss_boundaries - kiwi_boundaries),
            "jaccard_similarity": round(len(common) / len(union), 6) if union else 1,
        },
    }
    output_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Original characters: {len(original_text):,}")
    for name in ("kiwi", "kss"):
        item = comparison[name]
        print(
            f"{name.upper()}: {item['sentence_count']:,} sentences, "
            f"content_preserved={item['content_preserved']}, "
            f"median={item['characters']['median']} chars, "
            f"p95={item['characters']['p95']} chars, "
            f"short<=5={item['short_sentences_le_5']}, "
            f"long>100={item['long_sentences_gt_100']}"
        )
    boundary = comparison["boundaries"]
    print(
        f"Boundary agreement: common={boundary['common_count']}, "
        f"kiwi_only={boundary['kiwi_only_count']}, "
        f"kss_only={boundary['kss_only_count']}, "
        f"jaccard={boundary['jaccard_similarity']:.4f}"
    )
    print(f"Saved comparison: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
