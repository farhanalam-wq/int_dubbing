#!/usr/bin/env python3
"""
Stage 01 — Audio Extraction
============================
Reads: data/raw/{clip_id}.{mp4,mkv,avi,...}
Writes:
  data/stage_01_extracted/{clip_id}__extracted__48kHz_stereo.wav  ← 48kHz stereo master (remix/lipsync)
  data/stage_01_extracted/{clip_id}__extracted__16kHz_mono.wav    ← 16kHz mono (ASR input)

Usage:
  python scripts/01_extract.py --clip-id clip001
  python scripts/01_extract.py --clip-id clip001 --input-file data/raw/clip001.mp4
"""

import argparse
import glob
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from loguru import logger

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH = Path("configs/pipeline.yaml")

SUPPORTED_VIDEO_EXTS = [".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".ts"]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_input_file(raw_dir: Path, clip_id: str, explicit_path: str | None) -> Path:
    """Locate input video file. Explicit path wins; otherwise scan data/raw/ by clip_id."""
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            logger.error(f"Explicit input file not found: {p}")
            sys.exit(1)
        return p

    # Search for any supported extension
    for ext in SUPPORTED_VIDEO_EXTS:
        candidate = raw_dir / f"{clip_id}{ext}"
        if candidate.exists():
            logger.info(f"Found input: {candidate}")
            return candidate

    # Broader glob in case of unexpected naming
    matches = list(raw_dir.glob(f"{clip_id}*"))
    matches = [m for m in matches if m.suffix.lower() in SUPPORTED_VIDEO_EXTS]
    if len(matches) == 1:
        logger.info(f"Found input (glob): {matches[0]}")
        return matches[0]
    elif len(matches) > 1:
        logger.error(
            f"Ambiguous input: multiple files match '{clip_id}*' in {raw_dir}. "
            f"Pass --input-file explicitly.\n  {[str(m) for m in matches]}"
        )
        sys.exit(1)

    logger.error(
        f"No input file found for clip_id='{clip_id}' in {raw_dir}.\n"
        f"  Expected e.g. {raw_dir / clip_id}.mp4\n"
        f"  Tip: copy assets/test_hindi_video.mp4 → data/raw/clip001.mp4"
    )
    sys.exit(1)


def check_ffmpeg() -> None:
    """Ensure ffmpeg is on PATH and callable."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, check=True
        )
        version_line = result.stdout.splitlines()[0] if result.stdout else "unknown"
        logger.info(f"ffmpeg OK: {version_line}")
    except FileNotFoundError:
        logger.error("ffmpeg not found on PATH. Install ffmpeg and ensure it is accessible.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg version check failed: {e}")
        sys.exit(1)


def run_ffmpeg(args: list[str], label: str) -> None:
    """Run an ffmpeg command, streaming stderr to the logger."""
    cmd = ["ffmpeg", "-y"] + args  # -y: overwrite output without prompting
    logger.debug(f"Running [{label}]: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"ffmpeg failed [{label}]:\n{result.stderr}")
        sys.exit(1)
    logger.debug(f"ffmpeg stderr [{label}]:\n{result.stderr[-2000:]}")  # last 2k chars


def get_duration_seconds(path: Path) -> float:
    """Use ffprobe to get media duration in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"ffprobe failed on {path}: {result.stderr}")
        return 0.0
    data = json.loads(result.stdout)
    return float(data.get("format", {}).get("duration", 0.0))


def flag_needs_review(needs_review_path: Path, clip_id: str, stage: str, flag: str, detail: str) -> None:
    """Append a flag object to logs/needs_review.json."""
    try:
        with open(needs_review_path) as f:
            flags = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        flags = []

    flags.append({
        "clip_id": clip_id,
        "segment_id": None,
        "stage": stage,
        "flag": flag,
        "detail": detail,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    with open(needs_review_path, "w") as f:
        json.dump(flags, f, indent=2, ensure_ascii=False)

    logger.warning(f"⚑  needs_review flagged → {needs_review_path}: {flag} — {detail}")


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_48khz_stereo(input_file: Path, output_file: Path) -> None:
    """Extract 48kHz stereo WAV master. Downmix to stereo if source has >2 channels."""
    logger.info(f"Extracting 48kHz stereo master → {output_file}")
    run_ffmpeg(
        [
            "-i", str(input_file),
            "-vn",                    # drop video stream
            "-acodec", "pcm_s16le",   # uncompressed 16-bit PCM
            "-ar", "48000",           # 48kHz
            "-ac", "2",               # stereo (downmix if surround)
            str(output_file)
        ],
        label="48kHz_stereo"
    )


def extract_16khz_mono(input_file: Path, output_file: Path) -> None:
    """Extract 16kHz mono WAV for ASR engines."""
    logger.info(f"Extracting 16kHz mono (ASR) → {output_file}")
    run_ffmpeg(
        [
            "-i", str(input_file),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",               # mono
            str(output_file)
        ],
        label="16kHz_mono"
    )


def verify_output(path: Path, clip_id: str, stage: str, needs_review_path: Path) -> float:
    """Basic sanity check on output: file exists, non-zero size, readable duration."""
    if not path.exists() or path.stat().st_size == 0:
        flag_needs_review(
            needs_review_path, clip_id, stage,
            flag="output_missing_or_empty",
            detail=f"Expected output not found or empty: {path}"
        )
        return 0.0

    duration = get_duration_seconds(path)
    if duration < 0.5:
        flag_needs_review(
            needs_review_path, clip_id, stage,
            flag="output_suspiciously_short",
            detail=f"Duration {duration:.2f}s < 0.5s for {path}"
        )
    else:
        logger.success(f"✓ {path.name}  ({duration:.2f}s, {path.stat().st_size / 1e6:.1f} MB)")
    return duration


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 01 — Extract audio stems from input video."
    )
    parser.add_argument(
        "--clip-id", required=False,
        help="Clip identifier used in file naming (e.g. clip001). "
             "Defaults to pipeline.yaml project.clip_id_default."
    )
    parser.add_argument(
        "--input-file", required=False, default=None,
        help="Explicit path to input video. If omitted, searches data/raw/{clip_id}.*"
    )
    parser.add_argument(
        "--config", default=str(CONFIG_PATH),
        help=f"Path to pipeline.yaml (default: {CONFIG_PATH})"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging."
    )
    args = parser.parse_args()

    # ── Setup logging ──────────────────────────────────────────────────────────
    logger.remove()
    level = "DEBUG" if args.verbose else "INFO"
    logger.add(sys.stderr, level=level, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    log_file = Path("logs") / "01_extract.log"
    log_file.parent.mkdir(exist_ok=True)
    logger.add(str(log_file), level="DEBUG", rotation="10 MB",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}")

    logger.info("=== Stage 01: Audio Extraction ===")

    # ── Load config ────────────────────────────────────────────────────────────
    cfg = load_config()
    clip_id = args.clip_id or cfg["project"]["clip_id_default"]
    logger.info(f"clip_id: {clip_id}")

    raw_dir      = Path(cfg["paths"]["raw"])
    out_dir      = Path(cfg["paths"]["stage_01"])
    needs_review = Path(cfg["paths"]["needs_review"])

    master_sr   = cfg["extraction"]["master_sample_rate"]   # 48000
    master_ch   = cfg["extraction"]["master_channels"]       # 2
    asr_sr      = cfg["extraction"]["asr_sample_rate"]       # 16000
    asr_ch      = cfg["extraction"]["asr_channels"]          # 1

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Pre-flight ─────────────────────────────────────────────────────────────
    check_ffmpeg()

    input_file = find_input_file(raw_dir, clip_id, args.input_file)
    logger.info(f"Input: {input_file}  ({input_file.stat().st_size / 1e6:.1f} MB)")

    # ── Output paths (file naming convention from §1) ──────────────────────────
    # {clip_id}__extracted__{variant}.wav
    out_48k = out_dir / f"{clip_id}__extracted__{master_sr // 1000}kHz_stereo.wav"
    out_16k = out_dir / f"{clip_id}__extracted__{asr_sr // 1000}kHz_mono.wav"

    # ── Extract ────────────────────────────────────────────────────────────────
    extract_48khz_stereo(input_file, out_48k)
    extract_16khz_mono(input_file, out_16k)

    # ── Verify ─────────────────────────────────────────────────────────────────
    dur_48k = verify_output(out_48k, clip_id, "01_extract", needs_review)
    dur_16k = verify_output(out_16k, clip_id, "01_extract", needs_review)

    # Durations should match (within rounding); flag if they diverge > 1s
    if abs(dur_48k - dur_16k) > 1.0:
        flag_needs_review(
            needs_review, clip_id, "01_extract",
            flag="duration_mismatch_between_extracts",
            detail=f"48kHz={dur_48k:.2f}s vs 16kHz={dur_16k:.2f}s — may indicate partial extraction"
        )

    logger.info("=== Stage 01 complete ===")
    logger.info(f"  48kHz master : {out_48k}")
    logger.info(f"  16kHz ASR    : {out_16k}")
    logger.info(f"  Log          : {log_file}")


if __name__ == "__main__":
    main()
