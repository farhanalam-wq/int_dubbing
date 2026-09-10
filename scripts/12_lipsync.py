#!/usr/bin/env python3
"""
Stage 12 — Video Lip Sync (LatentSync 1.6, Paper-Aligned)
=========================================================
Reads:  data/raw/{clip_id}.mp4                                (original video)
        data/stage_07_tts/{clip_id}__dubbed_vocal_full.wav     (clean vocal stem for phoneme tracking)
Writes: data/stage_10_lipsync/{clip_id}__lipsync__latentsync.mp4

Paper-Aligned Architecture (Li et al., arXiv:2412.09262v2):
  - Hardware: NVIDIA A100-80GB GPU on Modal
  - UNet: stage2_512.yaml (512x512 resolution, SD 1.5 backbone)
  - Strict 25.0 FPS video & 16.0 kHz mono audio pre-normalization (Whisper 50Hz token stride)
  - Guidance scale: 2.0 (suppresses shortcut learning prior, enforces crisp articulation)
  - Inference steps: 20 (exact DDIM trajectory; DeepCache disabled by default for production quality)
  - Temporal landmark exponential smoothing (alpha=0.8) to eliminate jaw/mouth boundary jitter
  - Scene-based parallel batching on Modal (.map) across visual shot cuts

Usage:
  python scripts/12_lipsync.py --clip-id clip001
  python scripts/12_lipsync.py --clip-id clip001 --test-seconds 5 --start-sec 12.9
  python scripts/12_lipsync.py --clip-id clip001 --no-parallel
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
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

# Point to primary api.modal.com endpoint (avoids api.modal2.com SSL certificate chain mismatch on Windows)
if "MODAL_SERVER_URL" not in os.environ:
    os.environ["MODAL_SERVER_URL"] = "https://api.modal.com"

import modal
import yaml


CONFIG_PATH = Path("configs/pipeline.yaml")


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


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


def apply_latentsync_patches(smooth_alpha: float = 0.8):
    """
    Applies surgical runtime patches to ByteDance LatentSync:
      1. Fallback handling for non-face scenes (e.g. title pans, scenery).
      2. Exponential Moving Average (EMA) landmark smoothing across consecutive frames
         to eliminate high-frequency affine coordinate jitter (preventing jaw trembling).
      3. Graceful pass-through during full video frame re-composition.
    """
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
        new_store = f"""landmarks3 = np.round([pt_left_eye, pt_right_eye, pt_nose])
        if hasattr(self, "last_landmarks") and self.last_landmarks is not None:
            landmarks3 = np.round({smooth_alpha} * landmarks3 + {1.0 - smooth_alpha} * self.last_landmarks)
        self.last_landmarks = landmarks3"""
        ip_content = ip_content.replace(old_store, new_store)
        with open(ip_path, "w", encoding="utf-8") as f:
            f.write(ip_content)
        print(f"[Modal] Patched image_processor.py: face fallback + EMA smoothing (alpha={smooth_alpha}).")

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
        print("[Modal] Patched lipsync_pipeline.py: graceful frame restoration.")


def execute_latentsync_job(
    tmpdir: str,
    raw_video: str,
    raw_audio: str,
    start_sec: float,
    duration_sec: float | None,
    inference_steps: int,
    guidance_scale: float,
    enable_deepcache: bool,
    smooth_alpha: float = 0.8,
) -> tuple[bytes, int]:
    """
    Internal execution helper on Modal worker:
      1. Normalizes media to strict 25.0 FPS video and 16.0 kHz mono audio.
      2. Injects paper-aligned patches (EMA smoothing + face fallback).
      3. Executes LatentSync with exact DDIM steps and guidance scale.
    """
    proc_video = os.path.join(tmpdir, "proc_video.mp4")
    proc_audio = os.path.join(tmpdir, "proc_audio.wav")
    out_video = os.path.join(tmpdir, "synced_output.mp4")

    # Paper Alignment: 25.0 FPS video and 16.0 kHz mono audio normalization
    v_cmd = ["ffmpeg", "-y"]
    a_cmd = ["ffmpeg", "-y"]
    if start_sec > 0:
        v_cmd += ["-ss", f"{start_sec:.3f}"]
        a_cmd += ["-ss", f"{start_sec:.3f}"]
    if duration_sec is not None:
        v_cmd += ["-t", f"{duration_sec:.3f}"]
        a_cmd += ["-t", f"{duration_sec:.3f}"]
    v_cmd += ["-i", raw_video, "-filter:v", "fps=25", "-c:v", "libx264", "-pix_fmt", "yuv420p", proc_video]
    a_cmd += ["-i", raw_audio, "-ar", "16000", "-ac", "1", proc_audio]

    subprocess.run(v_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    subprocess.run(a_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    print(
        f"[Modal] Pre-normalized media: video={os.path.getsize(proc_video):,} bytes (25.0 FPS), "
        f"audio={os.path.getsize(proc_audio):,} bytes (16.0 kHz mono)"
    )

    apply_latentsync_patches(smooth_alpha=smooth_alpha)

    cmd = [
        sys.executable, "-m", "scripts.inference",
        "--unet_config_path", "configs/unet/stage2_512.yaml",
        "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
        "--inference_steps", str(inference_steps),
        "--guidance_scale", str(guidance_scale),
        "--video_path", proc_video,
        "--audio_path", proc_audio,
        "--video_out_path", out_video,
        "--temp_dir", os.path.join(tmpdir, "latentsync_tmp"),
    ]
    if enable_deepcache:
        cmd.append("--enable_deepcache")

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

    if returncode != 0:
        raise RuntimeError(f"LatentSync inference failed with code {returncode}")

    if not os.path.exists(out_video):
        raise FileNotFoundError(f"Expected output video not found at {out_video}")

    out_size = os.path.getsize(out_video)
    print(f"[Modal] Lip-synced video created: {out_size:,} bytes")
    with open(out_video, "rb") as f:
        synced_bytes = f.read()

    return synced_bytes, out_size


@app.function(
    gpu="A100-80GB",
    image=latentsync_image,
    timeout=3600,
)
def run_latentsync_full_modal(
    video_bytes: bytes,
    audio_bytes: bytes,
    start_sec: float = 0.0,
    duration_sec: float = None,
    inference_steps: int = 20,
    guidance_scale: float = 2.0,
    enable_deepcache: bool = False,
    smooth_alpha: float = 0.8,
) -> dict:
    """Monolithic single-worker execution on Modal A100-80GB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_video = os.path.join(tmpdir, "raw_video.mp4")
        raw_audio = os.path.join(tmpdir, "raw_audio.wav")
        with open(raw_video, "wb") as f:
            f.write(video_bytes)
        with open(raw_audio, "wb") as f:
            f.write(audio_bytes)

        synced_bytes, out_size = execute_latentsync_job(
            tmpdir=tmpdir,
            raw_video=raw_video,
            raw_audio=raw_audio,
            start_sec=start_sec,
            duration_sec=duration_sec,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            enable_deepcache=enable_deepcache,
            smooth_alpha=smooth_alpha,
        )

        return {
            "synced_video_bytes": synced_bytes,
            "size_bytes": out_size,
        }


@app.function(
    gpu="A100-80GB",
    image=latentsync_image,
    timeout=1800,
)
def run_latentsync_scene_modal(chunk_input: dict) -> dict:
    """Distributed parallel scene-worker execution on Modal A100-80GB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_video = os.path.join(tmpdir, "raw_video.mp4")
        raw_audio = os.path.join(tmpdir, "raw_audio.wav")
        with open(raw_video, "wb") as f:
            f.write(chunk_input["video_bytes"])
        with open(raw_audio, "wb") as f:
            f.write(chunk_input["audio_bytes"])

        synced_bytes, out_size = execute_latentsync_job(
            tmpdir=tmpdir,
            raw_video=raw_video,
            raw_audio=raw_audio,
            start_sec=chunk_input["start_sec"],
            duration_sec=chunk_input["duration_sec"],
            inference_steps=chunk_input.get("inference_steps", 20),
            guidance_scale=chunk_input.get("guidance_scale", 2.0),
            enable_deepcache=chunk_input.get("enable_deepcache", False),
            smooth_alpha=chunk_input.get("smooth_alpha", 0.8),
        )

        return {
            "scene_idx": chunk_input["scene_idx"],
            "start_sec": chunk_input["start_sec"],
            "duration_sec": chunk_input["duration_sec"],
            "synced_video_bytes": synced_bytes,
            "size_bytes": out_size,
        }


def detect_scene_cuts(
    video_path: Path,
    threshold: float = 0.35,
    min_scene_duration: float = 2.0,
) -> list[tuple[float, float]]:
    """
    Detects visual shot transitions via FFmpeg scene filter to enable
    artifact-free parallel batching across camera angles.
    """
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    res = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    total_duration = float(res.stdout.strip())

    # Detect scene change timestamps
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-filter:v", f"select=gt(scene\\,{threshold}),showinfo",
        "-f", "null", "-"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)

    cut_times = []
    pattern = re.compile(r"pts_time:\s*([\d\.]+)")
    for line in res.stderr.splitlines():
        if "pts_time:" in line:
            m = pattern.search(line)
            if m:
                t = float(m.group(1))
                if 0.5 < t < total_duration - 0.5:
                    cut_times.append(round(t, 3))

    # Enforce minimum scene duration threshold
    filtered_cuts = [0.0]
    for ct in cut_times:
        if ct - filtered_cuts[-1] >= min_scene_duration:
            filtered_cuts.append(ct)

    if total_duration - filtered_cuts[-1] < min_scene_duration and len(filtered_cuts) > 1:
        filtered_cuts[-1] = total_duration
    else:
        filtered_cuts.append(total_duration)

    scenes = []
    for i in range(len(filtered_cuts) - 1):
        scenes.append((filtered_cuts[i], filtered_cuts[i + 1]))

    return scenes


def main():
    cfg = load_config()
    ls_cfg = cfg.get("lipsync", {}).get("latentsync", {})

    default_steps = ls_cfg.get("inference_steps", 20)
    default_guidance = ls_cfg.get("guidance_scale", 2.0)
    default_deepcache = ls_cfg.get("enable_deepcache", False)
    default_smooth_alpha = ls_cfg.get("landmark_smooth_alpha", 0.8)
    default_parallel = ls_cfg.get("parallel_scenes", True)

    parser = argparse.ArgumentParser(description="Stage 12 — Video Lip Sync (LatentSync 1.6, Paper-Aligned)")
    parser.add_argument("--clip-id", default="clip001", help="Clip ID (default: clip001)")
    parser.add_argument("--video-path", default=None, help="Custom path to raw input video")
    parser.add_argument("--audio-path", default=None, help="Custom path to vocal WAV")
    parser.add_argument("--test-seconds", type=float, default=None, help="Run on test slice of N seconds")
    parser.add_argument("--start-sec", type=float, default=0.0, help="Start second for test slice")
    parser.add_argument("--steps", type=int, default=default_steps, help=f"Inference steps (default: {default_steps})")
    parser.add_argument("--guidance", type=float, default=default_guidance, help=f"Guidance scale (default: {default_guidance})")
    parser.add_argument("--enable-deepcache", action="store_true", default=default_deepcache, help="Enable DeepCache acceleration (default: False)")
    parser.add_argument("--smooth-alpha", type=float, default=default_smooth_alpha, help=f"Landmark EMA smoothing alpha (default: {default_smooth_alpha})")
    parser.add_argument("--parallel", action="store_true", default=default_parallel, help="Enable scene-based parallel batching on Modal (default: True)")
    parser.add_argument("--no-parallel", dest="parallel", action="store_false", help="Disable parallel scene batching; run monolithic")
    args = parser.parse_args()

    clip_id = args.clip_id
    video_path = Path(args.video_path) if args.video_path else Path(f"data/raw/{clip_id}.mp4")
    audio_path = Path(args.audio_path) if args.audio_path else Path(f"data/stage_07_tts/{clip_id}__dubbed_vocal_full.wav")
    out_dir = Path("data/stage_10_lipsync")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"Error: Video file {video_path} not found!")
        sys.exit(1)
    if not audio_path.exists():
        print(f"Error: Audio file {audio_path} not found!")
        sys.exit(1)

    print(f"=== Stage 12: Video Lip Sync [{clip_id}] (Paper-Aligned) ===")
    print(f"  Input Video: {video_path} ({video_path.stat().st_size:,} bytes)")
    print(f"  Input Audio (Clean Stem): {audio_path} ({audio_path.stat().st_size:,} bytes)")
    print(f"  Hardware Tier: NVIDIA A100-80GB (Locked)")
    print(f"  Normalization: Strict 25.0 FPS video | 16.0 kHz mono audio")
    print(f"  Guidance Scale: {args.guidance} (Target: suppress shortcut learning prior)")
    print(f"  Inference Steps: {args.steps} DDIM")
    print(f"  DeepCache: {'ENABLED' if args.enable_deepcache else 'DISABLED (Exact DDIM Trajectory)'}")
    print(f"  Landmark Smoothing: EMA alpha={args.smooth_alpha}")

    video_bytes = video_path.read_bytes()
    audio_bytes = audio_path.read_bytes()

    # Case A: Test slice requested
    if args.test_seconds is not None:
        print(f"\n[Mode: TEST SLICE] {args.start_sec:.2f}s -> {args.start_sec + args.test_seconds:.2f}s")
        print("\n[Modal] Submitting slice to Modal A100-80GB GPU...")
        with modal.enable_output():
            with app.run():
                result = run_latentsync_full_modal.remote(
                    video_bytes=video_bytes,
                    audio_bytes=audio_bytes,
                    start_sec=args.start_sec,
                    duration_sec=args.test_seconds,
                    inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    enable_deepcache=args.enable_deepcache,
                    smooth_alpha=args.smooth_alpha,
                )

        suffix = f"_slice_{int(args.test_seconds)}s"
        out_path = out_dir / f"{clip_id}__lipsync__latentsync{suffix}.mp4"
        out_path.write_bytes(result["synced_video_bytes"])
        print(f"\n[SUCCESS] Wrote test slice: {out_path} ({len(result['synced_video_bytes']):,} bytes)")
        return

    # Case B: Scene-Based Parallel Batching (Full Video)
    if args.parallel:
        print("\n[Mode: SCENE-BASED PARALLEL BATCHING]")
        print("Detecting visual shot transitions via FFmpeg...")
        scenes = detect_scene_cuts(video_path, threshold=0.35, min_scene_duration=2.0)
        print(f"Detected {len(scenes)} visual shots:")
        for idx, (s_start, s_end) in enumerate(scenes):
            print(f"  Shot {idx:02d}: {s_start:6.2f}s -> {s_end:6.2f}s (duration: {s_end - s_start:5.2f}s)")

        chunk_inputs = []
        for idx, (s_start, s_end) in enumerate(scenes):
            chunk_inputs.append({
                "scene_idx": idx,
                "start_sec": s_start,
                "duration_sec": round(s_end - s_start, 3),
                "video_bytes": video_bytes,
                "audio_bytes": audio_bytes,
                "inference_steps": args.steps,
                "guidance_scale": args.guidance,
                "enable_deepcache": args.enable_deepcache,
                "smooth_alpha": args.smooth_alpha,
            })

        print(f"\n[Modal] Dispatching {len(chunk_inputs)} scenes concurrently across A100-80GB workers...")
        with modal.enable_output():
            with app.run():
                results = list(run_latentsync_scene_modal.map(chunk_inputs))

        print(f"\n[Modal] All {len(results)} scenes successfully generated. Assembling output...")
        results.sort(key=lambda r: r["scene_idx"])

        parts_dir = out_dir / f"{clip_id}__scene_parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        filelist_path = parts_dir / "concat_list.txt"

        with open(filelist_path, "w", encoding="utf-8") as f:
            for r in results:
                part_file = parts_dir / f"scene_{r['scene_idx']:03d}.mp4"
                part_file.write_bytes(r["synced_video_bytes"])
                f.write(f"file '{part_file.resolve().as_posix()}'\n")

        out_path = out_dir / f"{clip_id}__lipsync__latentsync.mp4"
        concat_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(filelist_path),
            "-c", "copy",
            str(out_path)
        ]
        subprocess.run(concat_cmd, check=True)
        print(f"\n[SUCCESS] Wrote concatenated lip-synced video: {out_path} ({out_path.stat().st_size:,} bytes)")
        return

    # Case C: Monolithic Sequential Execution (Full Video)
    print("\n[Mode: MONOLITHIC FULL MONOLOGUE (A100-80GB)]")
    with modal.enable_output():
        with app.run():
            result = run_latentsync_full_modal.remote(
                video_bytes=video_bytes,
                audio_bytes=audio_bytes,
                start_sec=0.0,
                duration_sec=None,
                inference_steps=args.steps,
                guidance_scale=args.guidance,
                enable_deepcache=args.enable_deepcache,
                smooth_alpha=args.smooth_alpha,
            )

    out_path = out_dir / f"{clip_id}__lipsync__latentsync.mp4"
    out_path.write_bytes(result["synced_video_bytes"])
    print(f"\n[SUCCESS] Wrote lip-synced video: {out_path} ({len(result['synced_video_bytes']):,} bytes)")


if __name__ == "__main__":
    main()
