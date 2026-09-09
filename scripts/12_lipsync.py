#!/usr/bin/env python3
"""
Stage 12 — Video Lip Sync (LatentSync 1.6)
===========================================
Reads:  data/raw/{clip_id}.mp4                                (original video)
        data/stage_07_tts/{clip_id}__dubbed_vocal_full.wav     (clean vocal stem for phoneme tracking)
Writes: data/stage_10_lipsync/{clip_id}__lipsync__latentsync.mp4

Architecture:
  - ByteDance LatentSync 1.6 on Modal A10G/A100 GPU
  - UNet: stage2_512.yaml (512x512 resolution)
  - Audio Encoder: Whisper-tiny on isolated vocal stem (prevents BGM noise interference)
  - Acceleration: DeepCache (interval=3, branch=0)
  - Inference steps: 20 | Guidance scale: 1.5

Usage:
  python scripts/12_lipsync.py --clip-id clip001
  python scripts/12_lipsync.py --clip-id clip001 --test-seconds 10
"""

import argparse
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

app = modal.App("dubbing-stage-12-lipsync")

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
    .env({
        "LD_LIBRARY_PATH": "/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/site-packages/nvidia/curand/lib:/usr/local/lib/python3.10/site-packages/nvidia/cufft/lib"
    })
)


@app.function(
    gpu="A100-80GB",
    image=latentsync_image,
    timeout=3600,  # 60 minutes for full 93.69s video on A100-80GB
)
def run_latentsync_full_modal(
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

        print(f"[Modal] Input video size: {os.path.getsize(proc_video):,} bytes")
        print(f"[Modal] Input audio size: {os.path.getsize(proc_audio):,} bytes")

        # Apply robust face-handling patches to LatentSync so it never crashes on intro or transitions
        ip_path = "/root/LatentSync/latentsync/utils/image_processor.py"
        with open(ip_path, "r", encoding="utf-8") as f:
            ip_content = f.read()

        target_code = '        if bbox is None:\n            raise RuntimeError("Face not detected")'
        replacement_code = """        if bbox is None:
            if hasattr(self, "last_landmarks") and self.last_landmarks is not None:
                landmarks3 = self.last_landmarks
                face, affine_matrix = self.restorer.align_warp_face(image.copy(), landmarks3=landmarks3, smooth=True)
                box = [0, 0, face.shape[1], face.shape[0]]
                face = cv2.resize(face, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)
                face = rearrange(torch.from_numpy(face), "h w c -> c h w")
                return face, box, affine_matrix
            else:
                h, w, _ = image.shape
                min_dim = min(h, w)
                crop = image[(h - min_dim)//2 : (h + min_dim)//2, (w - min_dim)//2 : (w + min_dim)//2]
                face = cv2.resize(crop, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)
                face = rearrange(torch.from_numpy(face), "h w c -> c h w")
                return face, None, None"""

        if target_code in ip_content:
            ip_content = ip_content.replace(target_code, replacement_code)
            old_store = "landmarks3 = np.round([pt_left_eye, pt_right_eye, pt_nose])"
            new_store = "landmarks3 = np.round([pt_left_eye, pt_right_eye, pt_nose])\n        self.last_landmarks = landmarks3"
            ip_content = ip_content.replace(old_store, new_store)
            with open(ip_path, "w", encoding="utf-8") as f:
                f.write(ip_content)
            print("[Modal] Patched image_processor.py for robust face handling.")

        lp_path = "/root/LatentSync/latentsync/pipelines/lipsync_pipeline.py"
        with open(lp_path, "r", encoding="utf-8") as f:
            lp_content = f.read()

        target_restore = "        for index, face in enumerate(tqdm.tqdm(faces)):\n            x1, y1, x2, y2 = boxes[index]"
        replacement_restore = """        for index, face in enumerate(tqdm.tqdm(faces)):
            if boxes[index] is None or affine_matrices[index] is None:
                out_frames.append(video_frames[index])
                continue
            x1, y1, x2, y2 = boxes[index]"""

        if target_restore in lp_content:
            lp_content = lp_content.replace(target_restore, replacement_restore)
            with open(lp_path, "w", encoding="utf-8") as f:
                f.write(lp_content)
            print("[Modal] Patched lipsync_pipeline.py for graceful frame restoration.")

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

        print(f"[Modal] Running LatentSync 1.6 inference: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            cwd="/root/LatentSync",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in iter(process.stdout.readline, ""):
            line_str = line.strip()
            if line_str:
                print(f"[LatentSync] {line_str}", flush=True)
        process.stdout.close()
        returncode = process.wait()

        print("[Modal] Subprocess returncode:", returncode)
        if returncode != 0:
            raise RuntimeError(f"LatentSync inference failed with code {returncode}")

        if not os.path.exists(out_video):
            raise FileNotFoundError(f"Expected output video not found at {out_video}")

        out_size = os.path.getsize(out_video)
        print(f"[Modal] Lip-synced video created: {out_size:,} bytes")
        with open(out_video, "rb") as f:
            synced_bytes = f.read()

        return {
            "synced_video_bytes": synced_bytes,
            "size_bytes": out_size,
        }


def main():
    parser = argparse.ArgumentParser(description="Stage 12 — Video Lip Sync")
    parser.add_argument("--clip-id", default="clip001", help="Clip ID (default: clip001)")
    parser.add_argument("--video-path", default=None, help="Custom path to raw input video")
    parser.add_argument("--audio-path", default=None, help="Custom path to vocal WAV")
    parser.add_argument("--test-seconds", type=float, default=None, help="Run on test slice of N seconds")
    parser.add_argument("--start-sec", type=float, default=0.0, help="Start second for test slice")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps (default: 20)")
    parser.add_argument("--guidance", type=float, default=1.5, help="Guidance scale (default: 1.5)")
    args = parser.parse_args()

    clip_id = args.clip_id
    video_path = Path(args.video_path) if args.video_path else Path(f"data/raw/{clip_id}.mp4")
    # Condition LatentSync on clean vocal stem for optimal Whisper phoneme tracking
    audio_path = Path(args.audio_path) if args.audio_path else Path(f"data/stage_07_tts/{clip_id}__dubbed_vocal_full.wav")
    out_dir = Path("data/stage_10_lipsync")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"Error: Video file {video_path} not found!")
        sys.exit(1)
    if not audio_path.exists():
        print(f"Error: Audio file {audio_path} not found!")
        sys.exit(1)

    print(f"=== Stage 12: Video Lip Sync [{clip_id}] ===")
    print(f"  Input Video: {video_path} ({video_path.stat().st_size:,} bytes)")
    print(f"  Input Audio (Clean Stem): {audio_path} ({audio_path.stat().st_size:,} bytes)")
    if args.test_seconds:
        print(f"  Mode: TEST SLICE ({args.start_sec:.1f}s -> {args.start_sec + args.test_seconds:.1f}s)")
    else:
        print("  Mode: FULL CLIP MONOLOGUE (93.69s)")

    print("\nReading input files...")
    video_bytes = video_path.read_bytes()
    audio_bytes = audio_path.read_bytes()

    print("\n[Modal] Submitting LatentSync 1.6 inference job to Modal A10G GPU...")
    with modal.enable_output():
        with app.run():
            result = run_latentsync_full_modal.remote(
                video_bytes=video_bytes,
                audio_bytes=audio_bytes,
                start_sec=args.start_sec,
                duration_sec=args.test_seconds,
                inference_steps=args.steps,
                guidance_scale=args.guidance,
            )

    suffix = f"_slice_{int(args.test_seconds)}s" if args.test_seconds else ""
    out_path = out_dir / f"{clip_id}__lipsync__latentsync{suffix}.mp4"
    out_path.write_bytes(result["synced_video_bytes"])
    print(f"\n[SUCCESS] Wrote lip-synced video: {out_path} ({len(result['synced_video_bytes']):,} bytes)")


if __name__ == "__main__":
    main()
