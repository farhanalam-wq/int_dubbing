#!/usr/bin/env python3
"""
Stage 04 — Word-Level Timestamp Alignment (WhisperX)
======================================================
Reads:  data/stage_01_extracted/{clip_id}__extracted__16kHz_mono.wav
        data/stage_03_transcript/{clip_id}__stt__{winning_engine}.json   ← STT winner from §6.1 (IndicConformer)
Writes: data/stage_04_aligned/{clip_id}__aligned__words.json

Uses WhisperX forced alignment (wav2vec2-based Hindi phoneme aligner) on Modal L4 GPU
to attach precise millisecond word-level timestamps to the winning STT transcript.

Usage:
  python scripts/04_align.py --clip-id clip001
  python scripts/04_align.py --clip-id clip001 --stt-engine indicconformer
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
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

import modal

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("04_align")
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


# ── Modal App & Container Environment ─────────────────────────────────────────

app = modal.App("dubbing-stage-04-align")

align_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch>=2.2.0",
        "torchaudio>=2.2.0",
        "whisperx>=3.1.1",
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "soundfile>=0.12.1",
        "librosa>=0.10.1",
        "numpy>=1.26.0",
        "loguru>=0.7.0",
        "pyyaml>=6.0",
    )
)


@app.function(
    gpu="L4",
    image=align_image,
    timeout=900,
    secrets=[modal.Secret.from_name("hf-secret")],
)
def align_words_modal(
    audio_bytes: bytes,
    clip_id: str,
    segments_to_align: list[dict],
    language_code: str = "hi",
) -> dict:
    """Runs WhisperX forced alignment on Modal L4 GPU."""
    import os
    import tempfile
    import numpy as np
    import soundfile as sf
    import torch
    import whisperx

    work_dir = tempfile.mkdtemp(prefix="align_")
    audio_path = os.path.join(work_dir, f"{clip_id}__align_16k.wav")
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading audio for WhisperX alignment on {device}...")
    audio_data, sr = sf.read(audio_path)
    if sr != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        sr = 16000
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
    audio_data = audio_data.astype(np.float32)

    total_audio_sec = len(audio_data) / 16000.0
    print(f"Total audio duration: {total_audio_sec:.2f}s, Segments to align: {len(segments_to_align)}")

    print(f"Loading WhisperX alignment model for language '{language_code}'...")
    model_a, metadata = whisperx.load_align_model(
        language_code=language_code,
        device=device,
    )

    print("Running forced alignment...")
    start_time = time.time()
    align_result = whisperx.align(
        segments_to_align,
        model_a,
        metadata,
        audio_data,
        device,
        return_char_alignments=False,
    )
    align_duration = time.time() - start_time
    print(f"Forced alignment finished in {align_duration:.2f}s")

    # Format structured output with millisecond timestamps
    formatted_segments = []
    flat_words = []

    for seg_idx, seg in enumerate(align_result.get("segments", [])):
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        seg_text = seg.get("text", "").strip()

        seg_words = []
        for w in seg.get("words", []):
            word_text = w.get("word", "").strip()
            if not word_text:
                continue
            # If start/end missing in WhisperX output, fallback to segment bounds
            w_start = float(w.get("start", seg_start))
            w_end = float(w.get("end", seg_end if w_start >= seg_end else w_start + 0.1))
            w_score = float(w.get("score", 1.0)) if w.get("score") is not None else 1.0

            word_obj = {
                "word": word_text,
                "start": round(w_start, 3),
                "end": round(w_end, 3),
                "start_ms": int(round(w_start * 1000)),
                "end_ms": int(round(w_end * 1000)),
                "score": round(w_score, 3),
            }
            seg_words.append(word_obj)
            flat_words.append(word_obj)

        formatted_segments.append({
            "segment_id": f"{clip_id}_seg{seg_idx+1:03d}",
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "start_ms": int(round(seg_start * 1000)),
            "end_ms": int(round(seg_end * 1000)),
            "text": seg_text,
            "words": seg_words,
        })

    return {
        "clip_id": clip_id,
        "language": language_code,
        "alignment_duration_sec": round(align_duration, 2),
        "total_words": len(flat_words),
        "segments": formatted_segments,
        "words": flat_words,
    }


# ── Sentence Chunking Helper ──────────────────────────────────────────────────

def split_into_sentence_segments(text: str, total_duration_sec: float) -> list[dict]:
    """Splits full transcript into natural shastric Hindi phrases with proportional time estimates."""
    # Split on danda '।', question mark, exclamation, or key clause markers
    # Also handle natural pause conjunctions
    delimiters = r"(?<=[।!?])\s+|(?<=\s)(?=किंतु|पर हम|या तो|या फिर|एक सत्य|हम तो|तो क्या)"
    raw_clauses = re.split(delimiters, text.strip())
    clauses = [c.strip() for c in raw_clauses if c.strip()]

    if not clauses:
        clauses = [text.strip()]

    total_chars = sum(len(c) for c in clauses)
    if total_chars == 0:
        return [{"start": 0.0, "end": total_duration_sec, "text": text}]

    # Assign proportional time intervals based on length
    segments = []
    current_time = 0.0
    for c in clauses:
        fraction = len(c) / total_chars
        duration = fraction * total_duration_sec
        start = current_time
        end = min(current_time + duration, total_duration_sec)
        segments.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "text": c,
        })
        current_time = end

    return segments


# ── Local Driver & Alignment Runner ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 04: Word-Level Timestamp Alignment (WhisperX)")
    parser.add_argument("--clip-id", type=str, default=None, help="Clip identifier (e.g. clip001)")
    parser.add_argument(
        "--stt-engine",
        type=str,
        default="indicconformer",
        help="Winning STT engine from Stage 03 to align (default: indicconformer)",
    )
    args = parser.parse_args()

    cfg = load_config()
    clip_id = args.clip_id or cfg["project"]["clip_id_default"]
    stt_engine = args.stt_engine

    # Setup Logging
    log_dir = Path(cfg["paths"]["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "04_align.log",
        rotation="10 MB",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
    logger.info(f"=== Stage 04: Word Alignment started for clip_id='{clip_id}' (STT engine: '{stt_engine}') ===")

    # Resolve input audio
    stage01_dir = Path(cfg["paths"]["stage_01"])
    input_wav = stage01_dir / f"{clip_id}__extracted__16kHz_mono.wav"
    if not input_wav.exists():
        logger.error(f"Required 16kHz mono audio not found: {input_wav}")
        sys.exit(1)

    # Resolve input transcript
    stage03_dir = Path(cfg["paths"]["stage_03"])
    transcript_path = stage03_dir / f"{clip_id}__stt__{stt_engine}.json"
    if not transcript_path.exists():
        logger.error(f"Required STT transcript not found: {transcript_path}")
        logger.error(f"Run Stage 03 first: python scripts/03_stt.py --clip-id {clip_id}")
        sys.exit(1)

    with open(transcript_path, "r", encoding="utf-8") as f:
        stt_data = json.load(f)

    full_text = stt_data.get("full_text", "").strip()
    if not full_text:
        logger.error(f"STT transcript in {transcript_path} is empty.")
        sys.exit(1)

    stage04_dir = Path(cfg["paths"]["stage_04"])
    stage04_dir.mkdir(parents=True, exist_ok=True)

    # Calculate audio duration
    with open(input_wav, "rb") as f:
        audio_bytes = f.read()

    # Split transcript text into initial sentence segments for WhisperX
    total_audio_sec = len(audio_bytes) / (16000 * 2)  # 16-bit mono
    sentence_segments = split_into_sentence_segments(full_text, total_audio_sec)
    logger.info(f"Prepared {len(sentence_segments)} sentence segments for forced alignment.")

    logger.info(f"Submitting alignment job to Modal (GPU: L4, WhisperX Hindi)...")
    start_time = time.time()

    with modal.enable_output():
        with app.run():
            align_result = align_words_modal.remote(
                audio_bytes=audio_bytes,
                clip_id=clip_id,
                segments_to_align=sentence_segments,
                language_code="hi",
            )

    elapsed = time.time() - start_time
    logger.info(f"Modal execution completed in {elapsed:.1f}s")
    logger.info(f"Total aligned words: {align_result['total_words']} across {len(align_result['segments'])} segments")

    # Add metadata
    align_result["stt_engine"] = stt_engine
    align_result["clip_id"] = clip_id

    # Save output
    output_path = stage04_dir / f"{clip_id}__aligned__words.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(align_result, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved word-level alignment output to: {output_path}")

    # Preview
    if align_result["words"]:
        first_few = align_result["words"][:5]
        last_few = align_result["words"][-5:]
        logger.info(f"First 5 words: {[w['word'] + ' (' + str(w['start_ms']) + 'ms)' for w in first_few]}")
        logger.info(f"Last 5 words:  {[w['word'] + ' (' + str(w['start_ms']) + 'ms)' for w in last_few]}")

    logger.info(f"=== Stage 04: Word Alignment completed successfully for '{clip_id}' ===")


if __name__ == "__main__":
    main()
