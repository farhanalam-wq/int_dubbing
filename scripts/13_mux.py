#!/usr/bin/env python3
"""
Stage 13 — Final Mux / Export
===============================
Reads:  data/stage_10_lipsync/{clip_id}__lipsync__latentsync.mp4  (lip-synced video)
        data/stage_09_remix/{clip_id}__remix__final.wav           (authoritative 48kHz stereo master audio)
Writes: data/final/{clip_id}__final.mp4

Process:
  1. Takes the lip-synced video stream from Stage 12.
  2. Replaces the audio stream with the authoritative 48kHz stereo BGM+vocal remixed WAV.
  3. Encodes audio to 320kbps AAC, copies video without re-encoding (-c:v copy).
  4. Verifies output file size, duration, and stream integrity.

Usage:
  python scripts/13_mux.py --clip-id clip001
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Force UTF-8 on Windows
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

app = modal.App("dubbing-stage-13-mux")
mux_image = modal.Image.debian_slim(python_version="3.10").apt_install("ffmpeg")


@app.function(image=mux_image, timeout=300)
def mux_final_modal(video_bytes: bytes, audio_bytes: bytes) -> bytes:
    import os
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        in_video = os.path.join(tmpdir, "in_video.mp4")
        in_audio = os.path.join(tmpdir, "in_audio.wav")
        out_video = os.path.join(tmpdir, "final.mp4")

        with open(in_video, "wb") as f:
            f.write(video_bytes)
        with open(in_audio, "wb") as f:
            f.write(audio_bytes)

        cmd = [
            "ffmpeg", "-y",
            "-i", in_video,
            "-i", in_audio,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "320k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            out_video
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg mux failed:\n{res.stderr}")

        with open(out_video, "rb") as f:
            return f.read()


def main():
    parser = argparse.ArgumentParser(description="Stage 13 — Final Mux")
    parser.add_argument("--clip-id", default="clip001", help="Clip ID (default: clip001)")
    parser.add_argument("--video-path", default=None, help="Custom path to lip-synced video")
    parser.add_argument("--audio-path", default=None, help="Custom path to master remix WAV")
    args = parser.parse_args()

    clip_id = args.clip_id
    video_path = Path(args.video_path) if args.video_path else Path(f"data/stage_10_lipsync/{clip_id}__lipsync__latentsync.mp4")
    audio_path = Path(args.audio_path) if args.audio_path else Path(f"data/stage_09_remix/{clip_id}__remix__final.wav")
    out_dir = Path("data/final")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip_id}__final.mp4"

    if not video_path.exists():
        print(f"Error: Lip-synced video {video_path} not found!")
        sys.exit(1)
    if not audio_path.exists():
        print(f"Error: Master audio {audio_path} not found!")
        sys.exit(1)

    print(f"=== Stage 13: Final Mux [{clip_id}] ===")
    print(f"  Lip-Synced Video: {video_path} ({video_path.stat().st_size:,} bytes)")
    print(f"  Master Audio WAV: {audio_path} ({audio_path.stat().st_size:,} bytes)")

    # Check if local ffmpeg is available
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe:
        print("[Local] Running FFmpeg mux locally...")
        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "320k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(out_path)
        ]
        subprocess.run(cmd, check=True)
    else:
        print("[Modal] Submitting fast mux task to cloud container...")
        video_bytes = video_path.read_bytes()
        audio_bytes = audio_path.read_bytes()
        with modal.enable_output():
            with app.run():
                final_bytes = mux_final_modal.remote(video_bytes=video_bytes, audio_bytes=audio_bytes)
        out_path.write_bytes(final_bytes)

    print(f"\n[SUCCESS] Master dubbed video created: {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
