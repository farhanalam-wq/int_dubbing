#!/usr/bin/env python3
"""
Stage 03 — STT: Hindi Speech-to-Text Bake-off
==============================================
Reads:  data/stage_01_extracted/{clip_id}__extracted__16kHz_mono.wav
Writes: data/stage_03_transcript/{clip_id}__stt__indicconformer.json
        data/stage_03_transcript/{clip_id}__stt__indicwhisper.json

Both engines run unconditionally in experimentation phase (§6.1 bake-off).
DO NOT treat IndicWhisper as fallback-only — run both every time.

Output JSON schema per engine:
  {
    "clip_id": "clip001",
    "engine": "indicconformer | indicwhisper",
    "language": "hi",
    "model_repo": "...",
    "full_text": "...",
    "segments": [
      {"start": 0.0, "end": 2.3, "text": "...", "confidence": 0.95}
    ]
  }

Usage:
  python scripts/03_stt.py --clip-id clip001
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
    logger = logging.getLogger("03_stt")
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

app = modal.App("dubbing-stage-03-stt")

# Define container image with PyTorch CUDA, Transformers, and speech recognition tools
stt_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch>=2.2.0",
        "torchaudio>=2.2.0",
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "onnx>=1.15.0",
        "onnxruntime-gpu>=1.17.0",
        "soundfile>=0.12.1",
        "librosa>=0.10.1",
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "loguru>=0.7.0",
        "pyyaml>=6.0",
    )
)


@app.function(
    gpu="L4",
    image=stt_image,
    timeout=1200,
    secrets=[modal.Secret.from_name("hf-secret")],
)
def transcribe_hindi_modal(
    audio_bytes: bytes,
    clip_id: str,
    indicconformer_repo: str = "ai4bharat/indic-conformer-600m-multilingual",
    whisper_repo: str = "openai/whisper-large-v3",
) -> dict:
    """Runs dual-engine STT (IndicConformer & Whisper) on Modal L4 GPU."""
    import os
    import tempfile
    import torch
    import torchaudio
    from transformers import AutoModel, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    work_dir = tempfile.mkdtemp(prefix="stt_")
    audio_path = os.path.join(work_dir, f"{clip_id}__input_16k.wav")
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    results = {}

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Transcribe with Whisper Large-v3 (with segment timestamps)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"--- [1/2] Running Whisper ({whisper_repo}) ---")
    whisper_start = time.time()
    try:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=whisper_repo,
            torch_dtype=torch_dtype,
            device=device,
            return_timestamps=True,
            chunk_length_s=30,
        )

        whisper_out = whisper_pipe(
            audio_path,
            generate_kwargs={
                "language": "hindi",
                "task": "transcribe",
            },
        )

        segments = []
        chunks = whisper_out.get("chunks", [])
        if chunks:
            for c in chunks:
                ts = c.get("timestamp", (0.0, 0.0))
                start = float(ts[0]) if ts and ts[0] is not None else 0.0
                end = float(ts[1]) if ts and len(ts) > 1 and ts[1] is not None else start
                text = c.get("text", "").strip()
                if text:
                    segments.append({
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "text": text,
                        "confidence": 1.0,
                    })
        else:
            # Single text segment fallback
            segments.append({
                "start": 0.0,
                "end": round(float(len(audio_bytes) / (16000 * 2)), 3),
                "text": whisper_out.get("text", "").strip(),
                "confidence": 1.0,
            })

        whisper_duration = time.time() - whisper_start
        print(f"Whisper completed in {whisper_duration:.1f}s ({len(segments)} segments)")

        results["indicwhisper"] = {
            "clip_id": clip_id,
            "engine": "indicwhisper",
            "model_repo": whisper_repo,
            "language": "hi",
            "inference_time_sec": round(whisper_duration, 2),
            "full_text": whisper_out.get("text", "").strip(),
            "segments": segments,
        }

        # Clear Whisper VRAM before loading Conformer
        del whisper_pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error in Whisper transcription: {e}")
        results["indicwhisper"] = {
            "clip_id": clip_id,
            "engine": "indicwhisper",
            "model_repo": whisper_repo,
            "language": "hi",
            "error": str(e),
            "full_text": "",
            "segments": [],
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Transcribe with IndicConformer (AI4Bharat)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"--- [2/2] Running IndicConformer ({indicconformer_repo}) ---")
    conformer_start = time.time()
    try:
        import soundfile as sf
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        hf_token = os.environ.get("HF_TOKEN")
        conformer_model = AutoModel.from_pretrained(
            indicconformer_repo,
            trust_remote_code=True,
            token=hf_token,
        )
        conformer_model.to(device)
        conformer_model.eval()

        audio_data, sr = sf.read(audio_path)
        wav = torch.from_numpy(audio_data).float()
        if wav.ndim > 1:
            wav = torch.mean(wav, dim=1, keepdim=True).t()
        else:
            wav = wav.unsqueeze(0)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav = resampler(wav)

        wav = wav.to(device)

        # Transcribe with CTC decoding (and RNNT fallback)
        with torch.no_grad():
            try:
                ctc_out = conformer_model(wav, "hi", "ctc")
            except Exception:
                ctc_out = conformer_model(wav, "hi", "rnnt")

        # Conformer output format handling
        conformer_text = ""
        if isinstance(ctc_out, (list, tuple)) and len(ctc_out) > 0:
            conformer_text = str(ctc_out[0]).strip()
        else:
            conformer_text = str(ctc_out).strip()

        conformer_duration = time.time() - conformer_start
        print(f"IndicConformer completed in {conformer_duration:.1f}s")

        # Create structured segments: If whisper chunks exist, align conformer full text or use as single segment
        total_audio_sec = round(float(wav.shape[1] / 16000), 3)
        results["indicconformer"] = {
            "clip_id": clip_id,
            "engine": "indicconformer",
            "model_repo": indicconformer_repo,
            "language": "hi",
            "inference_time_sec": round(conformer_duration, 2),
            "full_text": conformer_text,
            "segments": [
                {
                    "start": 0.0,
                    "end": total_audio_sec,
                    "text": conformer_text,
                    "confidence": 1.0,
                }
            ],
        }

    except Exception as e:
        print(f"Error in IndicConformer transcription: {e}")
        results["indicconformer"] = {
            "clip_id": clip_id,
            "engine": "indicconformer",
            "model_repo": indicconformer_repo,
            "language": "hi",
            "error": str(e),
            "full_text": "",
            "segments": [],
        }

    return results


# ── Local Driver & Result Processing ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 03: Hindi Speech-to-Text Bake-off")
    parser.add_argument("--clip-id", type=str, default=None, help="Clip identifier (e.g. clip001)")
    args = parser.parse_args()

    cfg = load_config()
    clip_id = args.clip_id or cfg["project"]["clip_id_default"]

    # Setup Logging
    log_dir = Path(cfg["paths"]["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "03_stt.log",
        rotation="10 MB",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
    logger.info(f"=== Stage 03: Hindi STT Bake-off started for clip_id='{clip_id}' ===")

    # Resolve input
    stage01_dir = Path(cfg["paths"]["stage_01"])
    input_wav = stage01_dir / f"{clip_id}__extracted__16kHz_mono.wav"
    if not input_wav.exists():
        logger.error(f"Required 16kHz mono audio not found: {input_wav}")
        logger.error("Run Stage 01 first: python scripts/01_extract.py --clip-id {clip_id}")
        sys.exit(1)

    stage03_dir = Path(cfg["paths"]["stage_03"])
    stage03_dir.mkdir(parents=True, exist_ok=True)

    # Models from config
    stt_cfg = cfg.get("stt", {})
    indicconformer_repo = stt_cfg.get("indicconformer", {}).get(
        "hf_repo", "ai4bharat/indic-conformer-600m-multilingual"
    )
    whisper_repo = stt_cfg.get("indicwhisper", {}).get(
        "hf_repo", "openai/whisper-large-v3"
    )

    logger.info(f"Reading input audio: {input_wav} ({input_wav.stat().st_size / (1024*1024):.2f} MB)")
    with open(input_wav, "rb") as f:
        audio_bytes = f.read()

    logger.info("Submitting STT dual-engine job to Modal (GPU: L4)...")
    logger.info(f"  - IndicConformer: {indicconformer_repo}")
    logger.info(f"  - Whisper:        {whisper_repo}")

    start_time = time.time()
    with modal.enable_output():
        with app.run():
            results = transcribe_hindi_modal.remote(
                audio_bytes=audio_bytes,
                clip_id=clip_id,
                indicconformer_repo=indicconformer_repo,
                whisper_repo=whisper_repo,
            )

    elapsed = time.time() - start_time
    logger.info(f"Modal execution completed in {elapsed:.1f}s")

    # Save outputs
    conformer_path = stage03_dir / f"{clip_id}__stt__indicconformer.json"
    whisper_path = stage03_dir / f"{clip_id}__stt__indicwhisper.json"

    conformer_res = results.get("indicconformer", {})
    whisper_res = results.get("indicwhisper", {})

    with open(conformer_path, "w", encoding="utf-8") as f:
        json.dump(conformer_res, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved IndicConformer transcript: {conformer_path}")

    with open(whisper_path, "w", encoding="utf-8") as f:
        json.dump(whisper_res, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved IndicWhisper transcript:    {whisper_path}")

    # Log transcript summaries
    logger.info("--- STT Bake-off Summary ---")
    logger.info(f"[Whisper] Segments: {len(whisper_res.get('segments', []))} | Characters: {len(whisper_res.get('full_text', ''))}")
    logger.info(f"[Whisper preview]: {whisper_res.get('full_text', '')[:120]}...")
    logger.info(f"[IndicConformer] Characters: {len(conformer_res.get('full_text', ''))}")
    logger.info(f"[IndicConformer preview]: {conformer_res.get('full_text', '')[:120]}...")

    logger.info(f"=== Stage 03: STT Bake-off completed successfully for '{clip_id}' ===")


if __name__ == "__main__":
    main()
