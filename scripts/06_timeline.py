#!/usr/bin/env python3
"""
Stage 06 — Canonical Timeline Merge
=====================================
Reads:  data/stage_03_transcript/{clip_id}__stt__{engine}.json     ← STT text
        data/stage_04_aligned/{clip_id}__aligned__words.json        ← word timestamps
        data/stage_04_aligned/{clip_id}__diarized__speakers.json    ← speaker labels
Writes: data/stage_05_timeline/{clip_id}__timeline.json

Merges all three into the canonical timeline JSON (schema defined in §5 of TASK.md).
This is a custom Python merge layer — pure CPU logic, runs in <1 second.

Key merge rules:
- Speaker labels from diarization are assigned based on word timestamp overlap.
- Natural speech pauses (>650ms) or speaker transitions define segment boundaries.
- Words are grouped into dialogue segments with exact start_ms and end_ms.
- Slices reference audio from data/stage_02_separated/{clip_id}__separated__vocal.wav for TTS cloning.

Usage:
  python scripts/06_timeline.py --clip-id clip001
  python scripts/06_timeline.py --clip-id clip001 --stt-engine indicconformer
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Force UTF-8 on Windows console
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ["PYTHONIOENCODING"] = "utf-8"

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("06_timeline")
    logging.basicConfig(level=logging.INFO)

try:
    import yaml
except ImportError:
    yaml = None

# ── Paths & Config ────────────────────────────────────────────────────────────

CONFIG_PATH = Path("configs/pipeline.yaml")


def load_config() -> dict:
    if yaml is None:
        raise RuntimeError("yaml module is required on host to read config")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_speaker_for_word(word_mid_ms: int, speaker_turns: list[dict], default_speaker: str = "SPEAKER_00") -> str:
    """Finds which speaker turn encompasses the word midpoint, or the closest turn."""
    best_speaker = None
    min_dist = float("inf")

    for turn in speaker_turns:
        t_start = turn["start_ms"]
        t_end = turn["end_ms"]
        if t_start <= word_mid_ms <= t_end:
            return turn["speaker"]
        
        # Track distance if outside all turns
        dist = min(abs(word_mid_ms - t_start), abs(word_mid_ms - t_end))
        if dist < min_dist:
            min_dist = dist
            best_speaker = turn["speaker"]

    return best_speaker or default_speaker


def main():
    parser = argparse.ArgumentParser(description="Stage 06 — Canonical Timeline Merge")
    parser.add_argument("--clip-id", type=str, default=None, help="Clip identifier (e.g. clip001)")
    parser.add_argument(
        "--stt-engine",
        type=str,
        default="indicconformer",
        help="Winning STT engine used for alignment (default: indicconformer)",
    )
    args = parser.parse_args()

    cfg = load_config()
    clip_id = args.clip_id or cfg["project"]["clip_id_default"]
    stt_engine = args.stt_engine
    src_lang = cfg["project"].get("source_language", "hi")
    tgt_lang = cfg["project"].get("target_language", "bn")

    # Setup Logging
    log_dir = Path(cfg["paths"]["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "06_timeline.log",
        rotation="10 MB",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
    logger.info(f"=== Stage 06: Timeline Merge started for clip_id='{clip_id}' ===")

    # Resolve inputs
    stage03_dir = Path(cfg["paths"]["stage_03"])
    stage04_dir = Path(cfg["paths"]["stage_04"])
    stage05_dir = Path(cfg["paths"]["stage_05"])
    stage05_dir.mkdir(parents=True, exist_ok=True)

    stt_path = stage03_dir / f"{clip_id}__stt__{stt_engine}.json"
    words_path = stage04_dir / f"{clip_id}__aligned__words.json"
    speakers_path = stage04_dir / f"{clip_id}__diarized__speakers.json"

    for req_file in [words_path, speakers_path]:
        if not req_file.exists():
            logger.error(f"Required input file missing: {req_file}")
            sys.exit(1)

    with open(words_path, "r", encoding="utf-8") as f:
        words_data = json.load(f)

    with open(speakers_path, "r", encoding="utf-8") as f:
        speakers_data = json.load(f)

    all_words = words_data.get("words", [])
    speaker_turns = speakers_data.get("turns", [])
    speakers_list = speakers_data.get("speakers", ["SPEAKER_00"])
    default_speaker = speakers_list[0] if speakers_list else "SPEAKER_00"

    logger.info(f"Loaded {len(all_words)} words and {len(speaker_turns)} speaker turns.")

    if not all_words:
        logger.error("No words found in alignment output.")
        sys.exit(1)

    # Assign each word to its speaker turn
    tagged_words = []
    first_speech_onset_ms = speaker_turns[0]["start_ms"] if speaker_turns else 0

    for idx, w in enumerate(all_words):
        w_start = w["start_ms"]
        w_end = w["end_ms"]
        # If the first word has a long lead-in from 0 to speech onset, clamp start to speech onset
        if idx == 0 and w_start < first_speech_onset_ms:
            w_start = max(0, first_speech_onset_ms - 200)

        mid_ms = (w_start + w_end) // 2
        spk = find_speaker_for_word(mid_ms, speaker_turns, default_speaker)

        tagged_words.append({
            "word": w["word"],
            "start_ms": w_start,
            "end_ms": w_end,
            "speaker": spk,
            "score": w.get("score", 1.0),
        })

    # Group words chronologically into natural dialogue segments:
    # A segment break occurs when:
    # 1. The speaker changes.
    # 2. The pause between consecutive words exceeds 550 ms.
    # 3. Punctuation indicates a sentence boundary ('।', '?').
    PAUSE_THRESHOLD_MS = 550

    segments = []
    current_words = []

    for idx, tw in enumerate(tagged_words):
        if not current_words:
            current_words.append(tw)
            continue

        prev_w = current_words[-1]
        speaker_changed = tw["speaker"] != prev_w["speaker"]
        pause_exceeded = (tw["start_ms"] - prev_w["end_ms"]) > PAUSE_THRESHOLD_MS
        prev_has_terminal = prev_w["word"].endswith("।") or prev_w["word"].endswith("?")

        if speaker_changed or pause_exceeded or prev_has_terminal:
            seg_start = current_words[0]["start_ms"]
            seg_end = current_words[-1]["end_ms"]
            seg_speaker = current_words[0]["speaker"]
            seg_text = " ".join(w["word"] for w in current_words)
            seg_idx = len(segments) + 1

            segments.append({
                "segment_id": f"{clip_id}_seg{seg_idx:03d}",
                "speaker_id": seg_speaker,
                "start_ms": seg_start,
                "end_ms": seg_end,
                "source_text": seg_text,
                "source_text_engine": stt_engine,
                "words": [
                    {
                        "text": w["word"],
                        "start_ms": w["start_ms"],
                        "end_ms": w["end_ms"],
                    }
                    for w in current_words
                ],
                "translated_text_raw": "",
                "translated_text_final": "",
                "tts_reference_audio": f"data/stage_02_separated/{clip_id}__separated__vocal.wav#{seg_start}-{seg_end}",
                "tts_output_candidates": {},
                "duration_fit_ms": 0,
                "flags": [],
            })
            current_words = [tw]
        else:
            current_words.append(tw)

    if current_words:
        seg_start = current_words[0]["start_ms"]
        seg_end = current_words[-1]["end_ms"]
        seg_speaker = current_words[0]["speaker"]
        seg_text = " ".join(w["word"] for w in current_words)
        seg_idx = len(segments) + 1

        segments.append({
            "segment_id": f"{clip_id}_seg{seg_idx:03d}",
            "speaker_id": seg_speaker,
            "start_ms": seg_start,
            "end_ms": seg_end,
            "source_text": seg_text,
            "source_text_engine": stt_engine,
            "words": [
                {
                    "text": w["word"],
                    "start_ms": w["start_ms"],
                    "end_ms": w["end_ms"],
                }
                for w in current_words
            ],
            "translated_text_raw": "",
            "translated_text_final": "",
            "tts_reference_audio": f"data/stage_02_separated/{clip_id}__separated__vocal.wav#{seg_start}-{seg_end}",
            "tts_output_candidates": {},
            "duration_fit_ms": 0,
            "flags": [],
        })

    # Assemble canonical schema (§5)
    canonical_timeline = {
        "clip_id": clip_id,
        "source_language": src_lang,
        "target_language": tgt_lang,
        "total_segments": len(segments),
        "total_words": len(all_words),
        "segments": segments,
    }

    # Save canonical timeline
    out_path = stage05_dir / f"{clip_id}__timeline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(canonical_timeline, f, ensure_ascii=False, indent=2)

    logger.info(f"Canonical timeline saved to: {out_path}")
    logger.info(f"Total merged segments: {len(segments)}")

    # Preview segments
    for s in segments[:5]:
        logger.info(f"  [{s['segment_id']}] ({s['start_ms']}ms -> {s['end_ms']}ms, {s['end_ms'] - s['start_ms']}ms) {s['speaker_id']}: {s['source_text']}")
    if len(segments) > 5:
        logger.info(f"  ... and {len(segments) - 5} more segments.")

    logger.info(f"=== Stage 06: Timeline Merge completed successfully for '{clip_id}' ===")


if __name__ == "__main__":
    main()
