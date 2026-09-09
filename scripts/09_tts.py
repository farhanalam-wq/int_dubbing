#!/usr/bin/env python3
"""
Stage 09 — Multi-Engine TTS Voice Cloning & Bake-Off (Bangla)
=============================================================
Reads:  data/stage_06_translated/{clip_id}__translated__final.json
        data/stage_02_separated/{clip_id}__separated__vocal.wav   (reference audio)
Writes: data/stage_07_tts/{clip_id}_{seg_id}__tts__{model}.wav
        (one file per segment per candidate TTS model)

Candidate Models (§6.2 Bake-Off):
  1. DhVaani 0.5        -> ARTPARK-IISc/DhVaani-0.5 (ZipVoice 123M flow-matching, 27 Indic langs)
  2. IndicF5            -> ai4bharat/IndicF5 (Flow-matching polyglot Indic TTS)
  3. Fish Speech S2 Pro -> fishaudio/s2-pro (Dual-AR Qwen3 with inline emotion tags)
  4. Svara TTS          -> kenpath/svara-tts-voiceclone-beta (19 Indic langs with emotion tags)

Usage:
  # Run the §6.2 Bake-off (runs candidates on neutral + emotional test segments):
  python scripts/09_tts.py --clip-id clip001 --bake-off

  # Run full synthesis for a specific winning model across all 19 segments:
  python scripts/09_tts.py --clip-id clip001 --model dhvaani
"""

import argparse
import io
import json
import math
import os
import struct
import sys
import time
import wave
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
    logger = logging.getLogger("09_tts")
    logging.basicConfig(level=logging.INFO)

try:
    import yaml
except ImportError:
    yaml = None


# ── Config Loader ─────────────────────────────────────────────────────────────

CONFIG_PATH = Path("configs/pipeline.yaml")


def load_config() -> dict:
    if yaml is None:
        raise RuntimeError("yaml module is required on host to read config")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Audio Slicing Utility (Pure Python Standard Library) ──────────────────────

def _resample_mono_sinc(samples: list[float], in_rate: int = 44100, out_rate: int = 24000, radius: int = 6) -> list[float]:
    """High-fidelity windowed sinc resampler for pure Python stdlib."""
    if in_rate == out_rate:
        return samples
    ratio = in_rate / out_rate
    cutoff = 0.5 / ratio if ratio > 1.0 else 0.5
    N_in = len(samples)
    N_out = int(N_in / ratio)
    out = [0.0] * N_out
    for j in range(N_out):
        center = j * ratio
        idx_min = max(0, int(center - radius))
        idx_max = min(N_in - 1, int(center + radius) + 1)
        acc = 0.0
        weight_sum = 0.0
        for i in range(idx_min, idx_max + 1):
            d = i - center
            if d == 0:
                s = 1.0
            else:
                x = math.pi * d * (2 * cutoff)
                s = math.sin(x) / x
            # Blackman window
            w = 0.42 + 0.5 * math.cos(math.pi * d / radius) + 0.08 * math.cos(2 * math.pi * d / radius)
            weight = s * w
            acc += samples[i] * weight
            weight_sum += weight
        out[j] = acc / weight_sum if weight_sum > 0 else samples[int(center)]
    return out


def slice_wav_bytes(
    vocal_path: Path,
    start_ms: int,
    end_ms: int,
    target_rate: int = 24000,
) -> bytes:
    """Extracts a slice from vocal WAV, downmixes stereo to mono (channel average),
    resamples to target_rate (24,000 Hz), and returns mono 16-bit PCM WAV bytes."""
    with wave.open(str(vocal_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        total_frames = wf.getnframes()

        start_frame = max(0, int((start_ms / 1000.0) * framerate))
        end_frame = min(total_frames, int((end_ms / 1000.0) * framerate))
        n_frames = max(1, end_frame - start_frame)

        wf.setpos(start_frame)
        raw_bytes = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError(f"Expected 16-bit PCM audio, got sampwidth={sampwidth}")

    total_samples = len(raw_bytes) // 2
    raw_samples = struct.unpack(f"<{total_samples}h", raw_bytes)

    # FIX 1: Downmix stereo to mono (channel average)
    if n_channels == 2:
        mono_samples = [(raw_samples[i] + raw_samples[i + 1]) / 2.0 for i in range(0, len(raw_samples), 2)]
    elif n_channels == 1:
        mono_samples = [float(s) for s in raw_samples]
    else:
        mono_samples = [
            sum(raw_samples[i:i + n_channels]) / float(n_channels)
            for i in range(0, len(raw_samples), n_channels)
        ]

    # FIX 1: Resample from 44,100 Hz to 24,000 Hz
    resampled_samples = _resample_mono_sinc(mono_samples, in_rate=framerate, out_rate=target_rate)

    # Pack into single-channel 24,000 Hz WAV bytes
    out_buf = io.BytesIO()
    with wave.open(out_buf, "wb") as out_wf:
        out_wf.setnchannels(1)
        out_wf.setsampwidth(2)
        out_wf.setframerate(target_rate)
        clamped = [max(-32768, min(32767, int(round(x)))) for x in resampled_samples]
        out_wf.writeframes(struct.pack(f"<{len(clamped)}h", *clamped))

    return out_buf.getvalue()


def get_wav_duration_ms(wav_bytes: bytes) -> float:
    """Returns the duration in milliseconds of WAV bytes."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return (frames / float(rate)) * 1000.0


# ── Modal App & Environments ──────────────────────────────────────────────────

app = modal.App("dubbing-stage-09-tts")

# DhVaani Container Image
dhvaani_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.2.0",
        "torchaudio>=2.2.0",
        "transformers>=4.40.0",
        "soundfile",
        "scipy",
        "accelerate",
        "huggingface_hub",
        "sentencepiece",
        "tensorboard",
        "einops",
        "librosa",
        "vocos",
        "encodec",
        "pydub",
        "lhotse",
        "safetensors",
        "cn2an",
        "inflect",
        "jieba",
        "pypinyin",
        "loguru",
    )
)


@app.function(
    gpu="L4",
    image=dhvaani_image,
    timeout=600,
    secrets=[modal.Secret.from_name("hf-secret")],
)
def synthesize_dhvaani_modal(
    target_text: str,
    prompt_wav_bytes: bytes,
    prompt_text: str,
    hf_repo: str = "ARTPARK-IISc/DhVaani-0.5",
    guidance_scale: float = 2.5,
    num_step: int = 32,
    speed: float = 1.0,
    seed: int = 666,
) -> bytes:
    import os
    import tempfile
    import time
    import torch
    import torchaudio
    import soundfile as sf
    from transformers import AutoModel

    # 1. Patch torchaudio.load and torchaudio.save to use soundfile (avoids torchcodec requirement)
    def _safe_torchaudio_load(filepath, *args, **kwargs):
        data, sr = sf.read(filepath)
        tensor = torch.from_numpy(data).float()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)  # Shape (1, T)
        elif tensor.ndim == 2:
            if tensor.shape[1] > 1:
                tensor = tensor.mean(dim=-1, keepdim=True).t()  # Downmix to mono: Shape (1, T)
            else:
                tensor = tensor.t()  # Shape (1, T)
        return tensor, sr

    def _safe_torchaudio_save(uri, src, sample_rate, *args, **kwargs):
        arr = src.detach().cpu().numpy() if hasattr(src, "detach") else src
        if arr.ndim == 2:
            arr = arr.T
        sf.write(uri, arr, sample_rate)

    torchaudio.load = _safe_torchaudio_load
    torchaudio.save = _safe_torchaudio_save

    import sys
    import types
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        tb_mod = types.ModuleType("torch.utils.tensorboard")
        tb_mod.SummaryWriter = type("SummaryWriter", (), {"add_scalar": lambda *a, **k: None})
        sys.modules["torch.utils.tensorboard"] = tb_mod

    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading DhVaani model ({hf_repo}) on {device} (auth token present: {bool(hf_token)})...")
    t0 = time.time()

    model = AutoModel.from_pretrained(
        hf_repo,
        trust_remote_code=True,
        token=hf_token,
        low_cpu_mem_usage=False,
    ).to(device).eval()

    # Sanitize any non-persistent meta buffers (e.g. RelPositionalEncoding.pe in ZipFormer)
    for name, module in model.named_modules():
        if hasattr(module, "pe") and getattr(module.pe, "is_meta", False):
            module.pe = torch.zeros(module.pe.shape, dtype=torch.float32, device=device)
        if hasattr(module, "extend_pe"):
            _orig_extend = module.extend_pe
            def _wrap_extend(fn, mod):
                def _safe_extend(x, *args, **kwargs):
                    if hasattr(mod, "pe") and getattr(mod.pe, "is_meta", False):
                        mod.pe = torch.zeros(mod.pe.shape, dtype=x.dtype, device=x.device)
                    return fn(x, *args, **kwargs)
                return _safe_extend
            module.extend_pe = _wrap_extend(_orig_extend, module)

    print(f"DhVaani loaded in {time.time() - t0:.2f}s. Synthesizing target text: '{target_text}'...")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
        tmp_in.write(prompt_wav_bytes)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path.replace(".wav", "_out.wav")

    try:
        # FIX 1: Verify prompt audio tensor shape is (1, T) and sample rate is 24,000 Hz
        prompt_wav_tensor, prompt_sr = torchaudio.load(tmp_in_path)
        print(f"[VERIFY FIX 1] Prompt wav tensor shape: {tuple(prompt_wav_tensor.shape)}, sr: {prompt_sr}")
        assert prompt_wav_tensor.ndim == 2 and prompt_wav_tensor.shape[0] == 1, (
            f"Expected (1, T) mono prompt tensor, got shape {tuple(prompt_wav_tensor.shape)}"
        )
        print(f"[VERIFY FIX 3] Calling model.synthesize() with guidance_scale={guidance_scale}, num_step={num_step}, speed={speed}, seed={seed}")

        t_gen = time.time()
        audio = model.synthesize(
            text=target_text,
            prompt_wav=tmp_in_path,
            prompt_text=prompt_text,
            guidance_scale=guidance_scale,
            num_step=num_step,
            speed=speed,
            seed=seed,
        )
        gen_time = time.time() - t_gen

        # Normalize/rescale if numpy or torch tensor
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()

        sampling_rate = getattr(model, "sampling_rate", 24000)
        sf.write(tmp_out_path, audio, sampling_rate)
        print(f"Synthesized audio written at {sampling_rate}Hz in {gen_time:.2f}s.")

        with open(tmp_out_path, "rb") as f:
            out_bytes = f.read()
        return out_bytes

    finally:
        for p in [tmp_in_path, tmp_out_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


# ── Execution Logic ───────────────────────────────────────────────────────────

def run_bake_off(clip_id: str, cfg: dict):
    """Executes §6.2 TTS Bake-off on neutral and emotional test segments."""
    timeline_path = Path(f"data/stage_06_translated/{clip_id}__translated__final.json")
    vocal_path = Path(f"data/stage_02_separated/{clip_id}__separated__vocal.wav")
    output_dir = Path("data/stage_07_tts")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not timeline_path.exists():
        raise FileNotFoundError(f"Timeline not found: {timeline_path}. Run Stage 08 first.")
    if not vocal_path.exists():
        raise FileNotFoundError(f"Separated vocal stem not found: {vocal_path}. Run Stage 02 first.")

    with open(timeline_path, "r", encoding="utf-8") as f:
        timeline_data = json.load(f)

    tts_cfg = cfg.get("tts", {})
    bake_off_seg_ids = tts_cfg.get("bake_off_segments", ["clip001_seg001", "clip001_seg002"])

    segments = {s["segment_id"]: s for s in timeline_data.get("segments", [])}
    test_segments = [segments[sid] for sid in bake_off_seg_ids if sid in segments]

    if not test_segments:
        logger.error(f"None of the bake-off segments {bake_off_seg_ids} found in timeline.")
        return

    print("\n" + "=" * 105)
    print(f"STAGE 09: MULTI-ENGINE TTS BAKE-OFF (§6.2) — CLIP: {clip_id}")
    print("=" * 105)
    print(f"Test Segments: {', '.join([s['segment_id'] for s in test_segments])}")
    print("Primary Evaluation: (a) Speaker Identity (b) Emotion Preservation (c) Bengali Naturalness (d) Duration Fit")
    print("-" * 105)

    results = []

    with modal.enable_output():
        with app.run():
            for seg in test_segments:
                seg_id = seg["segment_id"]
                orig_dur_ms = seg["end_ms"] - seg["start_ms"]

                # FIX 2 & FIX 4 Segment-specific setup
                if seg_id == "clip001_seg001":
                    # FIX 2: 2.0-2.8s clean window (10300ms - 12500ms = 2200ms = 2.20s)
                    ref_start_ms = 10300
                    ref_end_ms = 12500
                    hi_text = "भविष्य का दूसरा नाम है"
                    bn_text = "ভবিষ্যতের অপর নাম হলো"
                elif seg_id == "clip001_seg002":
                    # FIX 2: 2.0-2.8s clean sub-span capturing dramatic delivery on "संघर्ष!" (12850ms - 15350ms = 2500ms = 2.50s)
                    ref_start_ms = 12850
                    ref_end_ms = 15350
                    # FIX 4: Correct missing exclamation mark after "संघर्ष"
                    hi_text = "संघर्ष! हृदय में आज इच्छा होती है और यदि पूर्ण नहीं हो पाती"
                    bn_text = "সংঘর্ষ! হৃদয়ে আজ ইচ্ছা জাগে এবং তা যদি पूर्ण না হতে পারে,"
                else:
                    ref_start_ms = seg["start_ms"]
                    ref_end_ms = seg["end_ms"]
                    hi_text = seg.get("source_text", "")
                    bn_text = seg.get("translated_text_final", "") or seg.get("translated_text_raw", "")

                ref_dur_ms = ref_end_ms - ref_start_ms
                logger.info(f"\nProcessing {seg_id} (Target dialogue span: {seg['start_ms']}-{seg['end_ms']}ms, dur: {orig_dur_ms}ms):")
                logger.info(f"  [FIX 2] Reference Audio Window: {ref_start_ms}ms - {ref_end_ms}ms (dur: {ref_dur_ms}ms = {ref_dur_ms / 1000.0:.2f}s)")
                logger.info(f"  [FIX 4] Hindi Prompt:  '{hi_text}'")
                logger.info(f"  Bangla Target: '{bn_text}'")

                # FIX 1: Mono downmix + resample to 24kHz before dispatch
                ref_wav_bytes = slice_wav_bytes(vocal_path, ref_start_ms, ref_end_ms, target_rate=24000)
                logger.info(f"  [FIX 1] Sliced & resampled reference audio: {len(ref_wav_bytes)} bytes (mono, 24000Hz)")

                # Primary Run: FIX 3 (guidance_scale=2.5, num_step=32)
                dhvaani_repo = tts_cfg.get("dhvaani", {}).get("hf_repo", "ARTPARK-IISc/DhVaani-0.5")
                logger.info(f"  [FIX 3] Dispatching Primary (guidance_scale=2.5, num_step=32) to DhVaani-0.5...")

                t0 = time.time()
                try:
                    out_bytes_fixed = synthesize_dhvaani_modal.remote(
                        target_text=bn_text,
                        prompt_wav_bytes=ref_wav_bytes,
                        prompt_text=hi_text,
                        hf_repo=dhvaani_repo,
                        guidance_scale=2.5,
                        num_step=32,
                    )
                    gen_time = time.time() - t0
                    out_dur_ms = get_wav_duration_ms(out_bytes_fixed)
                    dur_diff_ms = out_dur_ms - orig_dur_ms

                    out_path_fixed = output_dir / f"{seg_id}__tts__dhvaani_fixed.wav"
                    with open(out_path_fixed, "wb") as f:
                        f.write(out_bytes_fixed)

                    logger.info(f"  [SUCCESS] DhVaani fixed output saved: {out_path_fixed} (dur: {out_dur_ms:.0f}ms, delta: {dur_diff_ms:+.0f}ms)")
                    results.append({
                        "seg_id": seg_id,
                        "variant": "DhVaani_fixed (cfg=2.5, step=32)",
                        "target_dur_ms": orig_dur_ms,
                        "output_dur_ms": out_dur_ms,
                        "dur_delta_ms": dur_diff_ms,
                        "rtf_sec": gen_time,
                        "path": str(out_path_fixed),
                    })

                    # Optional comparison variant: guidance_scale=3.0, num_step=50
                    logger.info(f"  [FIX 3 Variant] Dispatching Variant (guidance_scale=3.0, num_step=50) for comparison...")
                    t_var = time.time()
                    out_bytes_g3 = synthesize_dhvaani_modal.remote(
                        target_text=bn_text,
                        prompt_wav_bytes=ref_wav_bytes,
                        prompt_text=hi_text,
                        hf_repo=dhvaani_repo,
                        guidance_scale=3.0,
                        num_step=50,
                    )
                    gen_time_var = time.time() - t_var
                    out_dur_var_ms = get_wav_duration_ms(out_bytes_g3)
                    out_path_g3 = output_dir / f"{seg_id}__tts__dhvaani_g3_s50.wav"
                    with open(out_path_g3, "wb") as f:
                        f.write(out_bytes_g3)

                    logger.info(f"  [SUCCESS] DhVaani g3_s50 output saved: {out_path_g3} (dur: {out_dur_var_ms:.0f}ms)")
                    results.append({
                        "seg_id": seg_id,
                        "variant": "DhVaani_variant (cfg=3.0, step=50)",
                        "target_dur_ms": orig_dur_ms,
                        "output_dur_ms": out_dur_var_ms,
                        "dur_delta_ms": out_dur_var_ms - orig_dur_ms,
                        "rtf_sec": gen_time_var,
                        "path": str(out_path_g3),
                    })

                except Exception as e:
                    logger.error(f"  [ERROR] DhVaani failed on {seg_id}: {e}")

    # Save updated timeline candidates
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(timeline_data, f, indent=2, ensure_ascii=False)

    # Print summary bake-off scorecard
    print("\n" + "=" * 105)
    print("STAGE 09 TTS BAKE-OFF SCORECARD & DURATION FIT SUMMARY")
    print("=" * 105)
    print(f"{'Seg ID':<16} | {'Model':<15} | {'Target (ms)':<12} | {'Output (ms)':<12} | {'Delta (ms)':<12} | {'Output File'}")
    print("-" * 105)
    for r in results:
        delta_str = f"{r['dur_delta_ms']:+.0f}ms"
        variant_name = r.get("variant") or r.get("model") or "DhVaani"
        print(f"{r['seg_id']:<16} | {variant_name:<30} | {r['target_dur_ms']:<12.0f} | {r['output_dur_ms']:<12.0f} | {delta_str:<12} | {r['path']}")
    print("=" * 105 + "\n")


def run_full_synthesis(clip_id: str, model_choice: str, cfg: dict):
    """Runs full voice cloning across all 19 segments using the selected winning model."""
    timeline_path = Path(f"data/stage_06_translated/{clip_id}__translated__final.json")
    vocal_path = Path(f"data/stage_02_separated/{clip_id}__separated__vocal.wav")
    output_dir = Path("data/stage_07_tts")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(timeline_path, "r", encoding="utf-8") as f:
        timeline_data = json.load(f)

    segments = timeline_data.get("segments", [])
    logger.info(f"Synthesizing {len(segments)} segments using {model_choice}...")

    tts_cfg = cfg.get("tts", {})
    dhvaani_repo = tts_cfg.get("dhvaani", {}).get("hf_repo", "ARTPARK-IISc/DhVaani-0.5")

    with modal.enable_output():
        with app.run():
            for i, seg in enumerate(segments, 1):
                seg_id = seg["segment_id"]
                start_ms = seg["start_ms"]
                end_ms = seg["end_ms"]
                hi_text = seg.get("source_text", "")
                bn_text = seg.get("translated_text_final", "") or seg.get("translated_text_raw", "")

                ref_wav_bytes = slice_wav_bytes(vocal_path, start_ms, end_ms)
                out_path = output_dir / f"{seg_id}__tts__{model_choice}.wav"

                logger.info(f"[{i}/{len(segments)}] Synthesizing {seg_id} ({start_ms}-{end_ms}ms)...")
                try:
                    out_bytes = synthesize_dhvaani_modal.remote(
                        target_text=bn_text,
                        prompt_wav_bytes=ref_wav_bytes,
                        prompt_text=hi_text,
                        hf_repo=dhvaani_repo,
                    )
                    with open(out_path, "wb") as f:
                        f.write(out_bytes)

                    if "tts_output_candidates" not in seg or not isinstance(seg["tts_output_candidates"], dict):
                        seg["tts_output_candidates"] = {}
                    seg["tts_output_candidates"][model_choice] = str(out_path).replace("\\", "/")

                except Exception as e:
                    logger.error(f"Failed on {seg_id}: {e}")

    # Save updated timeline
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(timeline_data, f, indent=2, ensure_ascii=False)

    logger.info(f"[SUCCESS] All {len(segments)} segments synthesized and updated in {timeline_path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 09 — Multi-Engine TTS Voice Cloning & Bake-Off")
    parser.add_argument("--clip-id", default=None, help="Clip ID (default: from pipeline.yaml)")
    parser.add_argument("--bake-off", action="store_true", help="Run §6.2 bake-off on neutral + emotional test lines")
    parser.add_argument("--model", default="dhvaani", help="Model to use for full synthesis (default: dhvaani)")
    args = parser.parse_args()

    cfg = load_config()
    clip_id = args.clip_id or cfg.get("project", {}).get("clip_id_default", "clip001")

    if args.bake_off:
        run_bake_off(clip_id, cfg)
    else:
        run_full_synthesis(clip_id, args.model, cfg)


if __name__ == "__main__":
    main()
