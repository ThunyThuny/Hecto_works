import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path


SOURCE_SCRIPT = Path(__file__).with_name("main_gemma4_v2_profile.py")
DEFAULT_INPUT_DIR = Path("/workspace/Data/Test_sample_data")
DEFAULT_OUTPUT_ROOT = Path("/workspace/Result/0728/v2_profile")
SUPPORTED_EXTENSIONS = {
    ".aac", ".flac", ".m4a", ".mkv", ".mov", ".mp3",
    ".mp4", ".ogg", ".opus", ".wav", ".webm", ".wma",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("saerok.v2.profile.batch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run main_gemma4_v2_profile.py sequentially for every media file",
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--meeting-type", default="meeting")
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--delay", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input directory not found: {input_dir}")
    if args.delay < 0:
        raise ValueError("--delay must be zero or greater")

    input_files = sorted(
        (
            path for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )
    if not input_files:
        logger.error("No supported media files found in %s", input_dir)
        return 1

    logger.info("Batch start: %d files", len(input_files))
    failures = []
    for index, input_path in enumerate(input_files, start=1):
        logger.info("[%d/%d] Start: %s", index, len(input_files), input_path)
        command = [
            sys.executable,
            str(SOURCE_SCRIPT),
            str(input_path),
            "--meeting-type",
            args.meeting_type,
            "--output-root",
            args.output_root,
        ]
        if args.num_speakers:
            command.extend(["--num-speakers", str(args.num_speakers)])
        if args.dry_run:
            logger.info("[%d/%d] Dry run command: %s", index, len(input_files), command)
            continue

        started_at = time.time()
        completed = subprocess.run(command, check=False)
        elapsed = time.time() - started_at
        if completed.returncode == 0:
            logger.info("[%d/%d] Done in %.1fs: %s", index, len(input_files), elapsed, input_path.name)
        else:
            failures.append((input_path.name, completed.returncode))
            logger.error(
                "[%d/%d] Failed with exit code %d after %.1fs: %s",
                index,
                len(input_files),
                completed.returncode,
                elapsed,
                input_path.name,
            )

        if index < len(input_files):
            logger.info("Waiting %.1fs before the next file", args.delay)
            time.sleep(args.delay)

    if failures:
        logger.error("Batch completed with %d failure(s): %s", len(failures), failures)
        return 1
    logger.info("Batch completed successfully: %d files", len(input_files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
