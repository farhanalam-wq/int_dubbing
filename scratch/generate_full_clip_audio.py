import io
import json
import math
import os
import struct
import sys
import time
import wave
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

app = modal.App("dubbing-full-clip-generation")

# Unified Modal Image with Indic Parler-TTS + OpenVoice v2 + FFmpeg
full_dub_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.2.0",
        "torchaudio>=2.2.0",
        "transformers",
        "soundfile",
        "scipy",
        "librosa",
        "silero-vad",
        "wavmark",
        "huggingface_hub",
        "pydub",
        "faster-whisper>=1.0.0",
        "whisper_timestamped",
        "inflect",
        "eng_to_ipa",
        "jieba",
        "cn2an",
        "pypinyin",
        "unidic_lite",
        "unidecode",
        "jamo",
        "git+https://github.com/huggingface/parler-tts.git",
    )
    .run_commands("pip install --no-deps git+https://github.com/myshell-ai/OpenVoice.git")
)


@app.function(
    gpu="L4",
    image=full_dub_image,
    timeout=900,
    secrets=[modal.Secret.from_name("hf-secret")],
)
def generate_full_clip_modal(
    segments: list[dict],
    reference_wav_bytes: bytes,
    master_duration_sec: float = 93.6925,
    tau: float = 0.35,
) -> dict:
    """
    Synthesizes and voice-converts all 19 segments using Indic Parler-TTS + OpenVoice v2,
    fits duration with ffmpeg atempo to respect pacing, and composites onto a 48kHz master canvas.
    """
    import io
    import os
    import subprocess
    import tempfile
    import time
    import numpy as np
    import soundfile as sf
    import torch
    from huggingface_hub import hf_hub_download
    from openvoice.api import ToneColorConverter
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[Modal] Initializing on {device}...")

    # 1. Load Indic Parler-TTS
    parler_repo = "ai4bharat/indic-parler-tts"
    print(f"[Modal] Loading Parler-TTS ({parler_repo})...")
    parler_model = ParlerTTSForConditionalGeneration.from_pretrained(parler_repo).to(device)
    prompt_tokenizer = AutoTokenizer.from_pretrained(parler_repo)

    desc_name = getattr(parler_model.config, "text_encoder", None)
    if desc_name and hasattr(desc_name, "_name_or_path"):
        desc_name = desc_name._name_or_path
    else:
        desc_name = "google/flan-t5-large"
    description_tokenizer = AutoTokenizer.from_pretrained(desc_name)

    # 2. Load OpenVoice v2 ToneColorConverter
    print("[Modal] Loading OpenVoice v2 ToneColorConverter...")
    hf_token = os.environ.get("HF_TOKEN")
    config_path = hf_hub_download(
        repo_id="myshell-ai/OpenVoiceV2",
        filename="config.json",
        subfolder="converter",
        token=hf_token,
    )
    checkpoint_path = hf_hub_download(
        repo_id="myshell-ai/OpenVoiceV2",
        filename="checkpoint.pth",
        subfolder="converter",
        token=hf_token,
    )
    converter = ToneColorConverter(config_path, device=device)
    converter.load_ckpt(checkpoint_path)

    # 3. Extract Krishna target speaker embedding
    with tempfile.TemporaryDirectory() as tmpdir:
        ref_path = os.path.join(tmpdir, "target_krishna.wav")
        with open(ref_path, "wb") as f:
            f.write(reference_wav_bytes)

        print("[Modal] Extracting target speaker embedding from Krishna reference audio...")
        target_se = converter.extract_se([ref_path])

        # Master 48kHz canvas (duration matching clip001 master)
        master_sr = 48000
        total_samples = int(round(master_duration_sec * master_sr))
        master_canvas = np.zeros(total_samples, dtype=np.float32)

        individual_segments_wav = {}
        segment_scorecard = []

        print(f"\n[Modal] Processing all {len(segments)} segments...")

        for idx, seg in enumerate(segments, 1):
            seg_id = seg["segment_id"]
            start_ms = seg["start_ms"]
            end_ms = seg["end_ms"]
            target_dur_sec = (end_ms - start_ms) / 1000.0
            text = seg.get("translated_text_final") or seg.get("translated_text_raw", "")
            desc = seg.get("style_description", "")

            print(f"\n--- [{idx}/{len(segments)}] {seg_id} ({start_ms}ms -> {end_ms}ms, slot={target_dur_sec:.2f}s) ---")
            print(f"  Text: {text}")

            t0 = time.time()

            # (A) Generate with Indic Parler-TTS
            input_ids = description_tokenizer(desc, return_tensors="pt").input_ids.to(device)
            prompt_input_ids = prompt_tokenizer(text, return_tensors="pt").input_ids.to(device)

            with torch.no_grad():
                gen_audio = parler_model.generate(
                    input_ids=input_ids,
                    prompt_input_ids=prompt_input_ids,
                    min_new_tokens=20,
                )

            parler_arr = gen_audio.cpu().numpy().squeeze()
            parler_sr = parler_model.config.sampling_rate

            parler_tmp = os.path.join(tmpdir, f"{seg_id}_parler.wav")
            sf.write(parler_tmp, parler_arr, parler_sr)

            # (B) Convert Tone Color with OpenVoice
            source_se = converter.extract_se([parler_tmp])
            converted_tmp = os.path.join(tmpdir, f"{seg_id}_converted.wav")
            converter.convert(
                audio_src_path=parler_tmp,
                src_se=source_se,
                tgt_se=target_se,
                output_path=converted_tmp,
                tau=tau,
            )

            # Read converted audio to check duration
            conv_data, conv_sr = sf.read(converted_tmp)
            conv_dur_sec = len(conv_data) / float(conv_sr)

            # (C) Duration fit via FFmpeg atempo if needed
            # We want the dialogue to comfortably fit in [start_ms, end_ms] without bleeding into the next segment
            delta_sec = conv_dur_sec - target_dur_sec
            fitted_tmp = os.path.join(tmpdir, f"{seg_id}_fitted.wav")

            # Check next segment start to determine maximum allowed duration before overlap
            next_start_ms = segments[idx]["start_ms"] if idx < len(segments) else (master_duration_sec * 1000.0)
            max_allowed_dur_sec = (next_start_ms - start_ms) / 1000.0 - 0.05  # 50ms safety cushion

            applied_tempo = 1.0
            if conv_dur_sec > max_allowed_dur_sec or abs(delta_sec) > 0.15:
                # Compute required tempo
                raw_tempo = conv_dur_sec / target_dur_sec
                # Clamp tempo between 0.85x and 1.20x to maintain pitch naturalness
                clamped_tempo = max(0.85, min(1.20, raw_tempo))
                applied_tempo = clamped_tempo

                # Run ffmpeg atempo filter
                cmd = [
                    "ffmpeg", "-y", "-i", converted_tmp,
                    "-filter:a", f"atempo={clamped_tempo:.4f}",
                    "-ar", str(master_sr),
                    fitted_tmp,
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            else:
                # Direct resample to 48kHz
                cmd = [
                    "ffmpeg", "-y", "-i", converted_tmp,
                    "-ar", str(master_sr),
                    fitted_tmp,
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            final_data, final_sr = sf.read(fitted_tmp)
            final_dur_sec = len(final_data) / float(final_sr)
            elapsed = time.time() - t0

            print(f"  Result: {conv_dur_sec:.2f}s -> fitted {final_dur_sec:.2f}s (slot: {target_dur_sec:.2f}s, tempo: {applied_tempo:.2f}x) in {elapsed:.2f}s")

            # (D) Save individual segment audio bytes
            with open(fitted_tmp, "rb") as f:
                individual_segments_wav[seg_id] = f.read()

            # (E) Composite onto Master Canvas
            start_sample = int(round((start_ms / 1000.0) * master_sr))
            end_sample = start_sample + len(final_data)

            # Prevent writing beyond master canvas
            if end_sample > total_samples:
                final_data = final_data[: total_samples - start_sample]
                end_sample = total_samples

            master_canvas[start_sample:end_sample] += final_data

            segment_scorecard.append({
                "seg_id": seg_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "target_dur_s": round(target_dur_sec, 2),
                "synth_dur_s": round(conv_dur_sec, 2),
                "final_dur_s": round(final_dur_sec, 2),
                "tempo": round(applied_tempo, 2),
                "gen_time_s": round(elapsed, 2),
            })

        # (F) Normalize master canvas to prevent clipping (peak = -1.0 dBFS = 0.89)
        max_peak = np.max(np.abs(master_canvas))
        print(f"\n[Modal] Master Canvas Assembled! Peak Amplitude: {max_peak:.3f}")
        if max_peak > 0.89:
            master_canvas = master_canvas * (0.89 / max_peak)
            print(f"[Modal] Normalized master canvas to -1.0 dBFS peak.")

        master_out_path = os.path.join(tmpdir, "clip001__dubbed_vocal_full.wav")
        sf.write(master_out_path, master_canvas, master_sr, format="WAV", subtype="PCM_16")

        with open(master_out_path, "rb") as f:
            master_wav_bytes = f.read()

        return {
            "master_wav_bytes": master_wav_bytes,
            "master_duration_sec": len(master_canvas) / float(master_sr),
            "master_sample_rate": master_sr,
            "individual_segments": individual_segments_wav,
            "scorecard": segment_scorecard,
        }


def get_segment_style(seg_id: str, text: str) -> str:
    """Returns a tailored Indic Parler style prompt capturing the dramatic arc of Krishna's monologue."""
    if seg_id == "clip001_seg001":
        return (
            "A deep, resonant male voice delivers a calm, philosophical statement in a reflective, "
            "measured cadence with clear studio acoustics."
        )
    elif seg_id == "clip001_seg002":
        return (
            "A deep, resonant male voice delivers an intensely dramatic theatrical monologue. "
            "He opens with an explosive, powerful proclamation on the first word, moving into an impassioned, "
            "melodic cadence with clear studio acoustics."
        )
    elif seg_id in ("clip001_seg003", "clip001_seg004", "clip001_seg005", "clip001_seg006"):
        return (
            "A deep, resonant male voice delivers a dramatic theatrical monologue with deliberate breath pauses "
            "and an introspective, expressive cadence, clear studio acoustics."
        )
    elif seg_id in ("clip001_seg007", "clip001_seg008", "clip001_seg009"):
        return (
            "A deep, resonant male voice speaks with crisp philosophical clarity and deliberate emphasis, "
            "clear close-mic studio recording."
        )
    elif seg_id in ("clip001_seg010", "clip001_seg011", "clip001_seg012", "clip001_seg013", "clip001_seg014", "clip001_seg015"):
        return (
            "A deep, resonant male voice delivers an impassioned dramatic speech with rising theatrical urgency "
            "and solemn gravitas, clear studio acoustics."
        )
    else:  # seg016 to seg019 (Climactic finale)
        return (
            "A deep, resonant male voice delivers a soaring, authoritative proclamation, building in emotional power "
            "and ending with booming theatrical conviction, clear studio acoustics."
        )


def main():
    timeline_path = Path("data/stage_06_translated/clip001__translated__final.json")
    ref_path = Path("data/stage_07_tts/clip001_option_a_original_hindi.wav")
    output_dir = Path("data/stage_07_tts")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not timeline_path.exists():
        raise FileNotFoundError(f"Timeline not found: {timeline_path}")
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_path}")

    with open(timeline_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(ref_path, "rb") as f:
        ref_bytes = f.read()

    segments = data["segments"]

    # Enrich segments with styled prompts
    for s in segments:
        text = s.get("translated_text_final") or s.get("translated_text_raw", "")
        s["style_description"] = get_segment_style(s["segment_id"], text)

    print("=" * 90)
    print(f"FULL-VIDEO-WIDTH DUBBING GENERATION ({len(segments)} SEGMENTS, 93.69s CANVAS)")
    print("=" * 90)

    with modal.enable_output():
        with app.run():
            result = generate_full_clip_modal.remote(
                segments=segments,
                reference_wav_bytes=ref_bytes,
                master_duration_sec=93.6925,
                tau=0.35,
            )

    # 1. Save master assembled vocal track
    master_wav_path = output_dir / "clip001__dubbed_vocal_full.wav"
    with open(master_wav_path, "wb") as f:
        f.write(result["master_wav_bytes"])
    print(f"\n[SAVED MASTER TRACK] {master_wav_path} ({len(result['master_wav_bytes'])} bytes, dur={result['master_duration_sec']:.2f}s)")

    # 2. Save individual segments
    for seg_id, wav_bytes in result["individual_segments"].items():
        seg_file = output_dir / f"{seg_id}__tts__cloned.wav"
        with open(seg_file, "wb") as f:
            f.write(wav_bytes)

    # 3. Print Scorecard Summary
    print("\n" + "=" * 95)
    print("FULL-CLIP SEGMENT TIMING & PACING SCORECARD")
    print("=" * 95)
    print(f"{'Segment':<16} | {'Timeline Slot':<18} | {'Slot Dur':<9} | {'Synth Dur':<10} | {'Fitted Dur':<11} | {'Tempo':<7} | {'Time'}")
    print("-" * 95)
    for sc in result["scorecard"]:
        slot_str = f"{sc['start_ms']}ms -> {sc['end_ms']}ms"
        print(
            f"{sc['seg_id']:<16} | {slot_str:<18} | {sc['target_dur_s']:>7.2f}s | {sc['synth_dur_s']:>8.2f}s | "
            f"{sc['final_dur_s']:>9.2f}s | {sc['tempo']:>5.2f}x | {sc['gen_time_s']:>5.2f}s"
        )
    print("=" * 95)
    print(f"\n[SUCCESS] Entire 93.69s dubbed vocal canvas assembled and saved to: {master_wav_path}")


if __name__ == "__main__":
    main()
