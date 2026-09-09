#!/usr/bin/env python3
"""
Stage 02 — Source Separation (Vocal / BGM)
===========================================
Reads:  data/stage_01_extracted/{clip_id}__extracted__48kHz_stereo.wav
Writes: data/stage_02_separated/{clip_id}__separated__vocal.wav
        data/stage_02_separated/{clip_id}__separated__bgm.wav

If fallback (Demucs) is triggered or selected:
        data/stage_02_separated/{clip_id}__separated__vocal__demucs.wav
        data/stage_02_separated/{clip_id}__separated__bgm__demucs.wav

Primary:  UVR + BS-RoFormer (audio-separator package) on Modal L4 GPU
Fallback: Demucs v4 htdemucs_ft (auto-triggered if QC check fails — §4.2)

QC rule (§4.2):
  If vocal RMS < threshold across >20% of 1-second chunks, OR if primary fails,
  automatically fallback to Demucs v4 and flag in logs/needs_review.json.

Usage:
  python scripts/02_separate.py --clip-id clip001
  python scripts/02_separate.py --clip-id clip001 --engine demucs
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

# Force UTF-8 on Windows console to support Modal's Unicode spinners and checkmarks (\u2713)
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
    logger = logging.getLogger("02_separate")
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

app = modal.App("dubbing-stage-02-separate")

# Define container image on Debian Slim with FFmpeg, PyTorch CUDA, and separation tools
separation_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch>=2.2.0",
        "torchaudio>=2.2.0",
        "audio-separator[gpu]>=0.30.0",
        "demucs>=4.0.1",
        "soundfile>=0.12.1",
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "loguru>=0.7.0",
        "pyyaml>=6.0",
    )
)


@app.function(
    gpu="L4",
    image=separation_image,
    timeout=900,
)
def separate_audio_remote(
    audio_bytes: bytes,
    filename: str,
    engine: str = "auto",
    uvr_model: str = "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    demucs_model: str = "htdemucs_ft",
    silence_threshold_db: float = -50.0,
    silence_max_fraction: float = 0.20,
) -> dict:
    """Runs source separation inside Modal L4 GPU container."""
    import math
    import os
    import shutil
    import subprocess
    import tempfile
    import numpy as np
    import soundfile as sf
    from audio_separator.separator import Separator

    def compute_silence_fraction(audio_path: str, chunk_sec: float = 1.0, threshold_db: float = -50.0) -> float:
        """Calculate the fraction of 1-second chunks below RMS threshold."""
        try:
            data, sr = sf.read(audio_path)
            if data.ndim > 1:
                data = np.mean(data, axis=1)  # convert to mono for RMS
            chunk_len = int(chunk_sec * sr)
            if chunk_len <= 0 or len(data) == 0:
                return 0.0
            
            n_chunks = len(data) // chunk_len
            if n_chunks == 0:
                rms = np.sqrt(np.mean(data**2) + 1e-12)
                db = 20 * math.log10(rms) if rms > 0 else -100.0
                return 1.0 if db < threshold_db else 0.0

            silent_chunks = 0
            for i in range(n_chunks):
                chunk = data[i * chunk_len : (i + 1) * chunk_len]
                rms = np.sqrt(np.mean(chunk**2) + 1e-12)
                db = 20 * math.log10(rms) if rms > 0 else -100.0
                if db < threshold_db:
                    silent_chunks += 1

            return silent_chunks / n_chunks
        except Exception as e:
            print(f"Error computing RMS: {e}")
            return 0.0

    work_dir = tempfile.mkdtemp(prefix="separation_")
    try:
        input_path = os.path.join(work_dir, filename)
        with open(input_path, "wb") as f:
            f.write(audio_bytes)

        vocal_path = None
        bgm_path = None
        used_engine = engine
        fallback_triggered = False
        fallback_reason = None
        silence_fraction = 0.0

        # Helper to run Demucs
        def run_demucs() -> tuple[str, str]:
            print(f"Running Demucs ({demucs_model})...")
            demucs_out = os.path.join(work_dir, "demucs_out")
            os.makedirs(demucs_out, exist_ok=True)
            cmd = [
                "python3", "-m", "demucs.separate",
                "-n", demucs_model,
                "-d", "cuda",
                "--two-stems", "vocals",
                "-o", demucs_out,
                input_path,
            ]
            subprocess.run(cmd, check=True)
            # Find output files
            stem_dir = os.path.join(demucs_out, demucs_model, os.path.splitext(filename)[0])
            d_vocals = os.path.join(stem_dir, "vocals.wav")
            d_bgm = os.path.join(stem_dir, "no_vocals.wav")
            if not os.path.exists(d_vocals) or not os.path.exists(d_bgm):
                raise RuntimeError(f"Demucs output missing in {stem_dir}")
            return d_vocals, d_bgm

        # Helper to run BS-RoFormer via audio-separator
        def run_bs_roformer() -> tuple[str, str]:
            print(f"Running audio-separator with model {uvr_model}...")
            uvr_out = os.path.join(work_dir, "uvr_out")
            os.makedirs(uvr_out, exist_ok=True)
            separator = Separator(output_dir=uvr_out, output_format="WAV")
            separator.load_model(model_filename=uvr_model)
            out_files = separator.separate(input_path)
            
            u_vocals = None
            u_bgm = None
            for out_f in out_files:
                full_path = os.path.join(uvr_out, out_f)
                lower_f = out_f.lower()
                if "vocal" in lower_f:
                    u_vocals = full_path
                elif "instrumental" in lower_f or "no_vocal" in lower_f or "bgm" in lower_f:
                    u_bgm = full_path
                    
            if not u_vocals or not u_bgm:
                # If naming convention differed, pick the two wav files
                wav_files = [os.path.join(uvr_out, f) for f in os.listdir(uvr_out) if f.endswith(".wav")]
                for wf in wav_files:
                    if "vocal" in wf.lower():
                        u_vocals = wf
                    else:
                        u_bgm = wf
            if not u_vocals or not u_bgm:
                raise RuntimeError(f"Could not identify vocal and instrumental stems from: {out_files}")
            return u_vocals, u_bgm

        # Execution logic
        if engine == "demucs":
            vocal_path, bgm_path = run_demucs()
            used_engine = "demucs"
        else:
            # Primary BS-RoFormer
            try:
                vocal_path, bgm_path = run_bs_roformer()
                used_engine = "uvr_bs_roformer"
                # QC Check: Silence check on vocal stem
                silence_fraction = compute_silence_fraction(
                    vocal_path, threshold_db=silence_threshold_db
                )
                print(f"BS-RoFormer vocal silence fraction: {silence_fraction:.2%}")
                if silence_fraction > silence_max_fraction:
                    fallback_triggered = True
                    fallback_reason = (
                        f"Vocal stem silence fraction {silence_fraction:.2%} exceeded "
                        f"threshold {silence_max_fraction:.2%}"
                    )
                    print(f"QC Alert: {fallback_reason}. Triggering Demucs fallback...")
                    vocal_path, bgm_path = run_demucs()
                    used_engine = "demucs"
            except Exception as e:
                fallback_triggered = True
                fallback_reason = f"Primary BS-RoFormer failed with error: {str(e)}"
                print(f"{fallback_reason}. Triggering Demucs fallback...")
                vocal_path, bgm_path = run_demucs()
                used_engine = "demucs"

        with open(vocal_path, "rb") as f:
            vocal_data = f.read()
        with open(bgm_path, "rb") as f:
            bgm_data = f.read()

        return {
            "status": "success",
            "used_engine": used_engine,
            "fallback_triggered": fallback_triggered,
            "fallback_reason": fallback_reason,
            "silence_fraction": silence_fraction,
            "vocal_bytes": vocal_data,
            "bgm_bytes": bgm_data,
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ── Local Driver & QC Logging ─────────────────────────────────────────────────

def run_ffprobe_duration(filepath: Path) -> float:
    """Probe audio duration in seconds via ffprobe."""
    import shutil
    ffprobe_bin = shutil.which("ffprobe") or "ffprobe"
    cmd = [
        ffprobe_bin, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(filepath),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(res.stdout.strip())


def append_needs_review(needs_review_path: Path, entry: dict) -> None:
    """Thread-safe / error-safe append to logs/needs_review.json."""
    flags = []
    if needs_review_path.exists():
        try:
            with open(needs_review_path, "r", encoding="utf-8") as f:
                flags = json.load(f)
        except Exception:
            flags = []
    flags.append(entry)
    with open(needs_review_path, "w", encoding="utf-8") as f:
        json.dump(flags, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Stage 02: Source Separation (Vocal / BGM)")
    parser.add_argument("--clip-id", type=str, default=None, help="Clip identifier (e.g. clip001)")
    parser.add_argument(
        "--engine",
        type=str,
        choices=["auto", "bs_roformer", "demucs"],
        default="auto",
        help="Separation engine to use (default: auto - BS-RoFormer with Demucs fallback)",
    )
    args = parser.parse_args()

    cfg = load_config()
    clip_id = args.clip_id or cfg["project"]["clip_id_default"]

    # Setup Logging
    log_dir = Path(cfg["paths"]["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "02_separate.log",
        rotation="10 MB",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
    logger.info(f"=== Stage 02: Source Separation started for clip_id='{clip_id}' ===")

    # Resolve input
    stage01_dir = Path(cfg["paths"]["stage_01"])
    input_wav = stage01_dir / f"{clip_id}__extracted__48kHz_stereo.wav"
    if not input_wav.exists():
        logger.error(f"Required master input audio not found: {input_wav}")
        logger.error("Run Stage 01 first: python scripts/01_extract.py --clip-id {clip_id}")
        sys.exit(1)

    stage02_dir = Path(cfg["paths"]["stage_02"])
    stage02_dir.mkdir(parents=True, exist_ok=True)

    needs_review_path = Path(cfg["paths"]["needs_review"])

    # Separation config
    sep_cfg = cfg.get("separation", {})
    uvr_model = sep_cfg.get("uvr_model_name", "model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    demucs_model = sep_cfg.get("demucs_model", "htdemucs_ft")
    silence_db = float(sep_cfg.get("vocal_rms_silence_threshold_db", -50.0))
    silence_fraction = float(sep_cfg.get("vocal_silence_max_fraction", 0.20))

    logger.info(f"Reading input audio: {input_wav} ({input_wav.stat().st_size / (1024*1024):.2f} MB)")
    with open(input_wav, "rb") as f:
        audio_bytes = f.read()

    logger.info(f"Submitting separation job to Modal (GPU: L4, Engine mode: {args.engine})...")
    start_time = time.time()

    with modal.enable_output():
        with app.run():
            result = separate_audio_remote.remote(
                audio_bytes=audio_bytes,
                filename=input_wav.name,
                engine=args.engine,
                uvr_model=uvr_model,
                demucs_model=demucs_model,
                silence_threshold_db=silence_db,
                silence_max_fraction=silence_fraction,
            )

    elapsed = time.time() - start_time
    logger.info(f"Modal execution completed in {elapsed:.1f}s")
    logger.info(f"Used engine: {result['used_engine']}, Fallback triggered: {result['fallback_triggered']}")

    # Output file paths
    vocal_primary_out = stage02_dir / f"{clip_id}__separated__vocal.wav"
    bgm_primary_out = stage02_dir / f"{clip_id}__separated__bgm.wav"

    # Save outputs
    with open(vocal_primary_out, "wb") as f:
        f.write(result["vocal_bytes"])
    with open(bgm_primary_out, "wb") as f:
        f.write(result["bgm_bytes"])

    logger.info(f"Saved primary vocal stem: {vocal_primary_out}")
    logger.info(f"Saved primary BGM stem:   {bgm_primary_out}")

    # If fallback was triggered or demucs explicitly run, also save variant files
    if result["used_engine"] == "demucs":
        demucs_vocal_out = stage02_dir / f"{clip_id}__separated__vocal__demucs.wav"
        demucs_bgm_out = stage02_dir / f"{clip_id}__separated__bgm__demucs.wav"
        with open(demucs_vocal_out, "wb") as f:
            f.write(result["vocal_bytes"])
        with open(demucs_bgm_out, "wb") as f:
            f.write(result["bgm_bytes"])
        logger.info(f"Saved Demucs variant stems: {demucs_vocal_out.name}, {demucs_bgm_out.name}")

    # Record QC flags if fallback was triggered
    if result["fallback_triggered"]:
        flag_entry = {
            "clip_id": clip_id,
            "stage": "stage_02_separated",
            "flag": "separation_fallback_used",
            "reason": result["fallback_reason"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                "silence_fraction": result["silence_fraction"],
                "threshold_db": silence_db,
            },
        }
        append_needs_review(needs_review_path, flag_entry)
        logger.warning(f"QC Flag recorded in {needs_review_path}: {result['fallback_reason']}")

    # Validation: Verify duration against input
    try:
        in_dur = run_ffprobe_duration(input_wav)
        vocal_dur = run_ffprobe_duration(vocal_primary_out)
        bgm_dur = run_ffprobe_duration(bgm_primary_out)
        logger.info(f"Durations — Input: {in_dur:.2f}s | Vocal: {vocal_dur:.2f}s | BGM: {bgm_dur:.2f}s")
        if abs(in_dur - vocal_dur) > 0.5:
            logger.warning(f"Duration drift between input and vocal: {abs(in_dur - vocal_dur):.2f}s")
    except Exception as e:
        logger.warning(f"Could not probe output durations: {e}")

    logger.info(f"=== Stage 02: Source Separation completed successfully for '{clip_id}' ===")


if __name__ == "__main__":
    main()
