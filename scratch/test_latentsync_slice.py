import os
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

app = modal.App("dubbing-latentsync")

# Build LatentSync 1.6 container image with weights pre-baked
latentsync_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "build-essential")
    .pip_install(
        "torch>=2.2.0",
        "torchvision",
        "diffusers>=0.30.0",
        "transformers>=4.40.0",
        "decord>=0.6.0",
        "accelerate>=0.26.0",
        "einops>=0.7.0",
        "omegaconf>=2.3.0",
        "opencv-python-headless",
        "mediapipe",
        "face-alignment",
        "imageio",
        "imageio-ffmpeg",
        "DeepCache",
        "kornia",
        "soundfile",
        "librosa",
        "huggingface_hub",
        "insightface",
        "onnxruntime-gpu",
        "ffmpeg-python",
        "python_speech_features",
        "scenedetect",
        "lpips",
    )
    .run_commands(
        "git clone https://github.com/bytedance/LatentSync.git /root/LatentSync",
        "python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='ByteDance/LatentSync-1.6', local_dir='/root/LatentSync/checkpoints')\"",
    )
)


@app.function(
    gpu="A10G",
    image=latentsync_image,
    timeout=1200,
)
def run_latentsync_modal(
    video_bytes: bytes,
    audio_bytes: bytes,
    start_sec: float = 0.0,
    duration_sec: float = None,
    inference_steps: int = 20,
    guidance_scale: float = 1.5,
) -> dict:
    import os
    import subprocess
    import tempfile
    import sys

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_video = os.path.join(tmpdir, "raw_video.mp4")
        raw_audio = os.path.join(tmpdir, "raw_audio.wav")
        proc_video = os.path.join(tmpdir, "proc_video.mp4")
        proc_audio = os.path.join(tmpdir, "proc_audio.wav")
        out_video = os.path.join(tmpdir, "synced_output.mp4")

        with open(raw_video, "wb") as f:
            f.write(video_bytes)
        with open(raw_audio, "wb") as f:
            f.write(audio_bytes)

        # Slice video and audio if start_sec or duration_sec specified
        if duration_sec is not None or start_sec > 0:
            print(f"[Modal] Slicing media from {start_sec:.2f}s for {duration_sec}s...")
            v_cmd = ["ffmpeg", "-y", "-ss", str(start_sec)]
            a_cmd = ["ffmpeg", "-y", "-ss", str(start_sec)]
            if duration_sec:
                v_cmd += ["-t", str(duration_sec)]
                a_cmd += ["-t", str(duration_sec)]
            v_cmd += ["-i", raw_video, "-c:v", "libx264", "-c:a", "aac", proc_video]
            a_cmd += ["-i", raw_audio, proc_audio]

            subprocess.run(v_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            subprocess.run(a_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        else:
            proc_video = raw_video
            proc_audio = raw_audio

        print(f"[Modal] Input video: {os.path.getsize(proc_video):,} bytes")
        print(f"[Modal] Input audio: {os.path.getsize(proc_audio):,} bytes")

        # Execute LatentSync from /root/LatentSync
        cmd = [
            sys.executable, "-m", "scripts.inference",
            "--unet_config_path", "configs/unet/stage2_512.yaml",
            "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
            "--inference_steps", str(inference_steps),
            "--guidance_scale", str(guidance_scale),
            "--enable_deepcache",
            "--video_path", proc_video,
            "--audio_path", proc_audio,
            "--video_out_path", out_video,
            "--temp_dir", os.path.join(tmpdir, "latentsync_tmp"),
        ]

        print(f"[Modal] Running command: {' '.join(cmd)}")
        res = subprocess.run(cmd, cwd="/root/LatentSync", capture_output=True, text=True)
        print("[Modal] Subprocess returncode:", res.returncode)
        if res.returncode != 0:
            print("[Modal] STDOUT:\n", res.stdout[-2000:] if len(res.stdout) > 2000 else res.stdout)
            print("[Modal] STDERR:\n", res.stderr[-2000:] if len(res.stderr) > 2000 else res.stderr)
            raise RuntimeError(f"LatentSync inference failed with code {res.returncode}")

        if not os.path.exists(out_video):
            raise FileNotFoundError(f"Expected output video not found at {out_video}")

        out_size = os.path.getsize(out_video)
        print(f"[Modal] Output video successfully created: {out_size:,} bytes")
        with open(out_video, "rb") as f:
            synced_bytes = f.read()

        return {
            "synced_video_bytes": synced_bytes,
            "size_bytes": out_size,
        }


def main():
    raw_video = Path("data/raw/clip001.mp4")
    clean_vocal = Path("data/stage_07_tts/clip001__dubbed_vocal_full.wav")
    test_dir = Path("data/stage_10_lipsync")
    test_dir.mkdir(parents=True, exist_ok=True)

    print("Reading full input files...")
    video_bytes = raw_video.read_bytes()
    audio_bytes = clean_vocal.read_bytes()

    # Step 1: Smoke test slice of 5.0 seconds (12.90s to 17.90s)
    print("\n--- Submitting 5-Second LatentSync Smoke Test to Modal (A10G GPU) ---")
    print("  Slice: 12.90s -> 17.90s ('সংঘর্ষ! হৃদয়ে আজ ইচ্ছা জাগে...')")

    with modal.enable_output():
        with app.run():
            result = run_latentsync_modal.remote(
                video_bytes=video_bytes,
                audio_bytes=audio_bytes,
                start_sec=12.90,
                duration_sec=5.00,
                inference_steps=20,
                guidance_scale=1.5,
            )

    out_test_path = test_dir / "test_slice_5s__lipsync__latentsync.mp4"
    out_test_path.write_bytes(result["synced_video_bytes"])
    print(f"\n[SUCCESS] Smoke test slice generated: {out_test_path} ({len(result['synced_video_bytes']):,} bytes)")


if __name__ == "__main__":
    main()
