import os
import sys
from pathlib import Path
import modal

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

app = modal.App("dubbing-bgm-remix")

remix_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install("soundfile", "numpy")
)


@app.function(
    image=remix_image,
    timeout=300,
)
def remix_audio_modal(
    vocal_wav_bytes: bytes,
    bgm_wav_bytes: bytes,
    video_mp4_bytes: bytes = None,
    vocal_gain_db: float = 0.0,
    bgm_gain_db: float = 0.0,
    sidechain_ducking: bool = True,
) -> dict:
    import os
    import subprocess
    import tempfile
    import soundfile as sf
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        vocal_path = os.path.join(tmpdir, "vocal.wav")
        bgm_path = os.path.join(tmpdir, "bgm.wav")
        out_wav_path = os.path.join(tmpdir, "remix_final.wav")
        preview_mp4_path = os.path.join(tmpdir, "preview.mp4")

        with open(vocal_path, "wb") as f:
            f.write(vocal_wav_bytes)
        with open(bgm_path, "wb") as f:
            f.write(bgm_wav_bytes)

        # Inspect durations and sample rates
        v_info = sf.info(vocal_path)
        b_info = sf.info(bgm_path)
        print(f"[Modal] Vocal: {v_info.duration:.3f}s, {v_info.samplerate}Hz, {v_info.channels}ch")
        print(f"[Modal] BGM:   {b_info.duration:.3f}s, {b_info.samplerate}Hz, {b_info.channels}ch")

        # Build FFmpeg filtergraph:
        # [0:a] = Vocal (mono/stereo 48k), [1:a] = BGM (stereo 44.1k)
        # Resample both to 48000Hz stereo
        # Apply gentle highpass (80Hz) to vocal to clean mic proximity rumble
        # Apply sidechain ducking to BGM triggered by vocal
        # Then amix, then peak limiter to -1.0 dBFS
        if sidechain_ducking:
            # Vocal is split into [voc_sc] (for sidechain detection) and [voc_mix] (for final mixing)
            filter_complex = (
                "[0:a]aformat=sample_rates=48000:channel_layouts=stereo,highpass=f=80,asplit=2[voc_sc][voc_mix]; "
                "[1:a]aformat=sample_rates=48000:channel_layouts=stereo[bgm]; "
                "[bgm][voc_sc]sidechaincompress=threshold=0.035:ratio=2.2:attack=40:release=350[ducked_bgm]; "
                "[ducked_bgm][voc_mix]amix=inputs=2:normalize=0:duration=first:weights=1.0 1.0[mixed]; "
                "[mixed]alimiter=limit=0.89:attack=5:release=50[out]"
            )
        else:
            filter_complex = (
                "[0:a]aformat=sample_rates=48000:channel_layouts=stereo,highpass=f=80[voc]; "
                "[1:a]aformat=sample_rates=48000:channel_layouts=stereo[bgm]; "
                "[bgm][voc]amix=inputs=2:duration=first:weights=0.85 1.0[mixed]; "
                "[mixed]alimiter=limit=0.89:attack=5:release=50[out]"
            )

        cmd = [
            "ffmpeg", "-y",
            "-i", vocal_path,
            "-i", bgm_path,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-c:a", "pcm_s16le",
            "-ar", "48000",
            out_wav_path
        ]

        print("[Modal] Running FFmpeg Remix Command...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print("[Modal] FFmpeg error:\n", res.stderr)
            raise RuntimeError(f"FFmpeg failed with returncode {res.returncode}")

        out_info = sf.info(out_wav_path)
        out_data, _ = sf.read(out_wav_path)
        max_peak = float(np.max(np.abs(out_data)))
        rms_val = float(np.sqrt(np.mean(out_data**2)))
        print(f"[Modal] Remixed Master: {out_info.duration:.3f}s, {out_info.samplerate}Hz, Peak: {max_peak:.4f}, RMS: {rms_val:.4f}")

        with open(out_wav_path, "rb") as f:
            remix_wav_bytes = f.read()

        preview_mp4_bytes = None
        if video_mp4_bytes:
            raw_mp4_path = os.path.join(tmpdir, "raw_video.mp4")
            with open(raw_mp4_path, "wb") as f:
                f.write(video_mp4_bytes)

            mux_cmd = [
                "ffmpeg", "-y",
                "-i", raw_mp4_path,
                "-i", out_wav_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "320k",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                preview_mp4_path
            ]
            print("[Modal] Muxing preview video...")
            res_mux = subprocess.run(mux_cmd, capture_output=True, text=True)
            if res_mux.returncode == 0 and os.path.exists(preview_mp4_path):
                print(f"[Modal] Preview MP4 generated: {os.path.getsize(preview_mp4_path):,} bytes")
                with open(preview_mp4_path, "rb") as f:
                    preview_mp4_bytes = f.read()
            else:
                print("[Modal] Preview mux failed:", res_mux.stderr)

        return {
            "remix_wav_bytes": remix_wav_bytes,
            "preview_mp4_bytes": preview_mp4_bytes,
            "duration": out_info.duration,
            "samplerate": out_info.samplerate,
            "peak": max_peak,
            "rms": rms_val,
        }


def main():
    vocal_path = Path("data/stage_07_tts/clip001__dubbed_vocal_full.wav")
    bgm_path = Path("data/stage_02_separated/clip001__separated__bgm.wav")
    video_path = Path("data/raw/clip001.mp4")
    out_dir = Path("data/stage_09_remix")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not vocal_path.exists():
        print(f"Error: {vocal_path} not found!")
        return
    if not bgm_path.exists():
        print(f"Error: {bgm_path} not found!")
        return

    print("Reading input files...")
    vocal_bytes = vocal_path.read_bytes()
    bgm_bytes = bgm_path.read_bytes()
    video_bytes = video_path.read_bytes() if video_path.exists() else None

    print("Submitting BGM remix job to Modal...")
    with modal.enable_output():
        with app.run():
            result = remix_audio_modal.remote(
                vocal_wav_bytes=vocal_bytes,
                bgm_wav_bytes=bgm_bytes,
                video_mp4_bytes=video_bytes,
                sidechain_ducking=True,
            )

    out_wav = out_dir / "clip001__remix__final.wav"
    out_wav.write_bytes(result["remix_wav_bytes"])
    print(f"\n[SUCCESS] Wrote remixed master audio: {out_wav} ({len(result['remix_wav_bytes']):,} bytes)")
    print(f"  Duration: {result['duration']:.3f}s, Peak: {result['peak']:.4f}, RMS: {result['rms']:.4f}")

    if result.get("preview_mp4_bytes"):
        out_mp4 = out_dir / "clip001__remix__preview.mp4"
        out_mp4.write_bytes(result["preview_mp4_bytes"])
        print(f"[SUCCESS] Wrote remixed preview video: {out_mp4} ({len(result['preview_mp4_bytes']):,} bytes)")


if __name__ == "__main__":
    main()
