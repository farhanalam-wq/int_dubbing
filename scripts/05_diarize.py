#!/usr/bin/env python3
"""
Stage 05 — Speaker Diarization (Pyannote)
===========================================
Reads:  data/stage_01_extracted/{clip_id}__extracted__48kHz_stereo.wav
        (Pyannote performs best on native master audio)
Writes: data/stage_04_aligned/{clip_id}__diarized__speakers.json
        (written into stage_04 alongside alignment — both feed stage 06 timeline merge)

Uses Pyannote (community-1 / 3.1) on Modal L4 GPU with HF_TOKEN secret.

Usage:
  python scripts/05_diarize.py --clip-id clip001
"""

import argparse
import json
import os
import subprocess
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
    logger = logging.getLogger("05_diarize")
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

app = modal.App("dubbing-stage-05-diarize")

diarize_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch>=2.2.0",
        "torchaudio>=2.2.0",
        "pyannote.audio>=3.3.0",
        "soundfile>=0.12.1",
        "numpy>=1.26.0",
        "loguru>=0.7.0",
        "pyyaml>=6.0",
    )
)


@app.function(
    gpu="L4",
    image=diarize_image,
    timeout=900,
    secrets=[modal.Secret.from_name("hf-secret")],
)
def diarize_audio_modal(
    audio_bytes: bytes,
    clip_id: str,
    hf_repo: str = "pyannote/speaker-diarization-community-1",
    min_speakers: int = 1,
    max_speakers: int = 10,
) -> dict:
    """Runs Pyannote speaker diarization on Modal L4 GPU."""
    import os
    import tempfile
    import numpy as np
    import soundfile as sf
    import torch
    from pyannote.audio import Pipeline

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is missing in Modal environment secrets.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Pyannote pipeline on {device}...")

    pipeline = None
    used_repo = hf_repo
    try:
        print(f"Trying primary diarization model: {hf_repo}...")
        pipeline = Pipeline.from_pretrained(hf_repo, token=hf_token)
    except Exception as e:
        fallback_repo = "pyannote/speaker-diarization-3.1"
        print(f"Primary model ({hf_repo}) failed: {e}. Falling back to {fallback_repo}...")
        used_repo = fallback_repo
        pipeline = Pipeline.from_pretrained(fallback_repo, token=hf_token)

    pipeline.to(device)

    work_dir = tempfile.mkdtemp(prefix="diarize_")
    audio_path = os.path.join(work_dir, f"{clip_id}__diarize.wav")
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    # Read audio via soundfile to bypass torchcodec loader issues
    audio_data, sr = sf.read(audio_path)
    waveform = torch.from_numpy(audio_data).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.t()  # (channels, time)

    total_duration_sec = waveform.shape[1] / float(sr)
    print(f"Running diarization on {total_duration_sec:.2f}s audio (SR={sr}, min={min_speakers}, max={max_speakers})...")

    start_time = time.time()
    diarization_out = pipeline(
        {"waveform": waveform, "sample_rate": sr},
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    duration = time.time() - start_time
    print(f"Diarization finished in {duration:.2f}s")

    annotation = getattr(diarization_out, "speaker_diarization", diarization_out)

    turns = []
    speakers_set = set()

    for item in annotation.itertracks(yield_label=True):
        if len(item) == 3:
            turn, _, speaker = item
        else:
            turn, speaker = item

        speaker_str = str(speaker)
        speakers_set.add(speaker_str)
        turns.append({
            "speaker": speaker_str,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "start_ms": int(round(turn.start * 1000)),
            "end_ms": int(round(turn.end * 1000)),
        })

    # If no turns found (e.g. quiet audio), provide single fallback turn
    if not turns:
        print("Warning: No speaker turns detected by Pyannote. Creating fallback turn.")
        speakers_set.add("SPEAKER_00")
        turns.append({
            "speaker": "SPEAKER_00",
            "start": 0.0,
            "end": round(total_duration_sec, 3),
            "start_ms": 0,
            "end_ms": int(round(total_duration_sec * 1000)),
        })

    return {
        "clip_id": clip_id,
        "model_repo": used_repo,
        "inference_time_sec": round(duration, 2),
        "num_speakers": len(speakers_set),
        "speakers": sorted(list(speakers_set)),
        "turns": turns,
    }


# ── Local Driver & Saving ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 05 — Speaker Diarization (Pyannote)")
    parser.add_argument("--clip-id", type=str, default=None, help="Clip identifier (e.g. clip001)")
    args = parser.parse_args()

    cfg = load_config()
    clip_id = args.clip_id or cfg["project"]["clip_id_default"]

    # Setup Logging
    log_dir = Path(cfg["paths"]["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "05_diarize.log",
        rotation="10 MB",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
    logger.info(f"=== Stage 05: Speaker Diarization started for clip_id='{clip_id}' ===")

    # Resolve input audio: Stage 01 48kHz stereo master audio
    stage01_dir = Path(cfg["paths"]["stage_01"])
    input_wav = stage01_dir / f"{clip_id}__extracted__48kHz_stereo.wav"
    if not input_wav.exists():
        logger.error(f"Required 48kHz master audio not found: {input_wav}")
        logger.error("Run Stage 01 first: python scripts/01_extract.py --clip-id {clip_id}")
        sys.exit(1)

    stage04_dir = Path(cfg["paths"]["stage_04"])
    stage04_dir.mkdir(parents=True, exist_ok=True)

    diar_cfg = cfg.get("diarization", {})
    hf_repo = diar_cfg.get("hf_repo", "pyannote/speaker-diarization-community-1")
    min_speakers = int(diar_cfg.get("min_speakers", 1))
    max_speakers = int(diar_cfg.get("max_speakers", 10))

    logger.info(f"Reading input audio: {input_wav} ({input_wav.stat().st_size / (1024*1024):.2f} MB)")
    with open(input_wav, "rb") as f:
        audio_bytes = f.read()

    logger.info(f"Submitting diarization job to Modal (GPU: L4, Model: {hf_repo})...")
    start_time = time.time()

    with modal.enable_output():
        with app.run():
            result = diarize_audio_modal.remote(
                audio_bytes=audio_bytes,
                clip_id=clip_id,
                hf_repo=hf_repo,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

    elapsed = time.time() - start_time
    logger.info(f"Modal execution completed in {elapsed:.1f}s")
    logger.info(f"Model used: {result['model_repo']}, Discovered speakers: {result['num_speakers']} ({result['speakers']})")
    logger.info(f"Total speaker turns detected: {len(result['turns'])}")

    # Save output to stage_04_aligned (matching TASK.md §1 & §5)
    output_path = stage04_dir / f"{clip_id}__diarized__speakers.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved diarized speaker segments to: {output_path}")

    # Preview first few speaker turns
    for t in result["turns"][:5]:
        logger.info(f"  [{t['speaker']}] {t['start']:.2f}s -> {t['end']:.2f}s ({t['end_ms'] - t['start_ms']}ms)")

    logger.info(f"=== Stage 05: Speaker Diarization completed successfully for '{clip_id}' ===")


if __name__ == "__main__":
    main()
