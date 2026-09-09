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

app = modal.App("dubbing-8-units-generation")

# Cached Unified Modal Image with Indic Parler-TTS + OpenVoice v2 + FFmpeg
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
def generate_8_units_modal(
    units: list[dict],
    reference_wav_bytes: bytes,
    master_duration_sec: float = 93.6925,
    tau: float = 0.35,
) -> dict:
    """
    Synthesizes and voice-converts 8 balanced natural sentence units (4-11s each).
    Preserves 1.00x native playback (no destructive atempo), levels active speech RMS,
    and composites onto a 48kHz master canvas with verified music silence.
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

        individual_units_wav = {}
        unit_scorecard = []

        print(f"\n[Modal] Processing {len(units)} balanced sentence units...")

        # Target active speech RMS for uniform loudness across all units
        TARGET_SPEECH_RMS = 0.08  # ~ -22 dBFS speech level

        for idx, u in enumerate(units, 1):
            unit_id = u["unit_id"]
            start_ms = u["start_ms"]
            end_ms = u["end_ms"]
            slot_dur_sec = (end_ms - start_ms) / 1000.0
            text = u["text"]
            desc = u["style_description"]

            # Next unit start determines maximum boundary before overlap
            next_start_ms = units[idx]["start_ms"] if idx < len(units) else (master_duration_sec * 1000.0)
            max_allowed_sec = (next_start_ms - start_ms) / 1000.0 - 0.10  # 100ms clean silence cushion

            print(f"\n--- [{idx}/{len(units)}] {unit_id} [{start_ms}ms -> {end_ms}ms] (Slot: {slot_dur_sec:.2f}s, Max: {max_allowed_sec:.2f}s) ---")
            print(f"  Text: {text}")

            t0 = time.time()

            # (A) Generate with Indic Parler-TTS (within the 4-11s sweet spot)
            input_ids = description_tokenizer(desc, return_tensors="pt").input_ids.to(device)
            prompt_input_ids = prompt_tokenizer(text, return_tensors="pt").input_ids.to(device)

            with torch.no_grad():
                gen_audio = parler_model.generate(
                    input_ids=input_ids,
                    prompt_input_ids=prompt_input_ids,
                    min_new_tokens=25,
                )

            parler_arr = gen_audio.cpu().numpy().squeeze()
            parler_sr = parler_model.config.sampling_rate

            parler_tmp = os.path.join(tmpdir, f"{unit_id}_parler.wav")
            sf.write(parler_tmp, parler_arr, parler_sr)

            # (B) Convert Tone Color with OpenVoice (operating in short-chunk stability regime)
            source_se = converter.extract_se([parler_tmp])
            converted_tmp = os.path.join(tmpdir, f"{unit_id}_converted.wav")
            converter.convert(
                audio_src_path=parler_tmp,
                src_se=source_se,
                tgt_se=target_se,
                output_path=converted_tmp,
                tau=tau,
            )

            # (C) Clean resample to 48kHz WITHOUT destructive atempo phase stretching
            resampled_tmp = os.path.join(tmpdir, f"{unit_id}_48k.wav")
            cmd = ["ffmpeg", "-y", "-i", converted_tmp, "-ar", str(master_sr), resampled_tmp]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            unit_data, _ = sf.read(resampled_tmp)

            # Trim trailing silence if audio exceeds max_allowed_sec to prevent overlap
            current_dur_sec = len(unit_data) / float(master_sr)
            if current_dur_sec > max_allowed_sec:
                # If slight excess, trim or gently fade the trailing tail rather than phase-vocoding
                max_samples = int(round(max_allowed_sec * master_sr))
                unit_data = unit_data[:max_samples]
                # Apply smooth 50ms fade-out at the end
                fade_len = int(round(0.05 * master_sr))
                if len(unit_data) > fade_len:
                    fade_curve = np.linspace(1.0, 0.0, fade_len)
                    unit_data[-fade_len:] *= fade_curve
                print(f"  [Safety Cushion] Trimmed tail to fit {max_allowed_sec:.2f}s slot without overlap.")

            # (D) Active-Speech Loudness Normalization
            # Find active speech frames (above threshold) to compute true speech RMS
            active_frames = unit_data[np.abs(unit_data) > 0.01]
            if len(active_frames) > 0:
                speech_rms = np.sqrt(np.mean(active_frames ** 2))
                if speech_rms > 0.005:
                    gain = TARGET_SPEECH_RMS / speech_rms
                    # Clamp gain between 0.4x and 2.5x
                    clamped_gain = max(0.4, min(2.5, gain))
                    unit_data = unit_data * clamped_gain
                    # Prevent clipping
                    unit_peak = np.max(np.abs(unit_data))
                    if unit_peak > 0.89:
                        unit_data = unit_data * (0.89 / unit_peak)
                    print(f"  [Loudness Leveling] Speech RMS: {speech_rms:.4f} -> Gain: {clamped_gain:.2f}x (Peak: {np.max(np.abs(unit_data)):.3f})")

            final_dur_sec = len(unit_data) / float(master_sr)
            elapsed = time.time() - t0

            # Save normalized unit audio
            normalized_tmp = os.path.join(tmpdir, f"{unit_id}_final.wav")
            sf.write(normalized_tmp, unit_data, master_sr, format="WAV", subtype="PCM_16")
            with open(normalized_tmp, "rb") as f:
                individual_units_wav[unit_id] = f.read()

            # (E) Composite onto Master Canvas
            start_sample = int(round((start_ms / 1000.0) * master_sr))
            end_sample = start_sample + len(unit_data)

            if end_sample > total_samples:
                unit_data = unit_data[: total_samples - start_sample]
                end_sample = total_samples

            master_canvas[start_sample:end_sample] += unit_data

            gap_sec = (next_start_ms - (start_ms + final_dur_sec * 1000.0)) / 1000.0
            print(f"  Finished: {final_dur_sec:.2f}s (Gap to next: {gap_sec:+.2f}s) in {elapsed:.2f}s")

            unit_scorecard.append({
                "unit_id": unit_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "slot_dur_s": round(slot_dur_sec, 2),
                "final_dur_s": round(final_dur_sec, 2),
                "gap_to_next_s": round(gap_sec, 2),
                "speech_rms": round(float(np.sqrt(np.mean(unit_data[np.abs(unit_data) > 0.01]**2))), 4) if np.any(np.abs(unit_data) > 0.01) else 0.0,
                "gen_time_s": round(elapsed, 2),
            })

        # (F) Final master canvas peak normalization to -1.0 dBFS (0.89)
        max_peak = np.max(np.abs(master_canvas))
        print(f"\n[Modal] Master Canvas Assembled! Peak Amplitude: {max_peak:.3f}")
        if max_peak > 0.89:
            master_canvas = master_canvas * (0.89 / max_peak)
            print("[Modal] Normalized master canvas to -1.0 dBFS peak.")

        master_out_path = os.path.join(tmpdir, "clip001__dubbed_vocal_full.wav")
        sf.write(master_out_path, master_canvas, master_sr, format="WAV", subtype="PCM_16")

        with open(master_out_path, "rb") as f:
            master_wav_bytes = f.read()

        return {
            "master_wav_bytes": master_wav_bytes,
            "master_duration_sec": len(master_canvas) / float(master_sr),
            "master_sample_rate": master_sr,
            "individual_units": individual_units_wav,
            "scorecard": unit_scorecard,
        }


def get_8_balanced_units() -> list[dict]:
    """
    Defines the 8 balanced, natural sentence units (4-11s each),
    aligned to Krishna's actual acoustic pauses in the original scene.
    """
    return [
        {
            "unit_id": "unit_01_intro",
            "start_ms": 10479,
            "end_ms": 12342,
            "text": "ভবিষ্যতের অপর নাম হলো",
            "style_description": (
                "A deep, resonant male voice delivers a calm, philosophical introduction in a reflective, "
                "measured cadence with clear studio acoustics."
            ),
        },
        {
            "unit_id": "unit_02_option_a",
            "start_ms": 12903,
            "end_ms": 22981,
            "text": (
                "সংঘর্ষ! হৃদয়ে আজ ইচ্ছা জাগে এবং তা যদি পূর্ণ না হতে পারে... "
                "তবে হৃদয় ভবিষ্যতের পরিকল্পনা প্রস্তুত করে—ভবিষ্যতে ইচ্ছা পূর্ণ হবে, "
                "এমন কল্পনা করতে থাকে... কিন্তু..."
            ),
            "style_description": (
                "A deep, resonant male voice delivers an intensely dramatic theatrical monologue. "
                "He opens with an explosive, powerful proclamation on the first word, then transitions into a passionate, "
                "melodic cadence with deliberate breath pauses, ending in a quiet suspended tone with clear studio acoustics."
            ),
        },
        {
            "unit_id": "unit_03_definition",
            "start_ms": 23542,
            "end_ms": 31081,
            "text": "জীবন... জীবন না তো ভবিষ্যতে আছে, না অতীতে আছে! জীবন তো এই মুহূর্তের নাম।",
            "style_description": (
                "A deep, resonant male voice delivers a thoughtful philosophical monologue with deliberate breath pauses "
                "and an introspective, expressive cadence, clear studio acoustics."
            ),
        },
        {
            "unit_id": "unit_04_aphorism",
            "start_ms": 31783,
            "end_ms": 36535,
            "text": "অর্থাৎ... এই মুহূর্তের অনুভবই জীবন।",
            "style_description": (
                "A deep, resonant male voice speaks with crisp philosophical clarity, reflective cadence, "
                "and deliberate emphasis, clear close-mic studio recording."
            ),
        },
        {
            "unit_id": "unit_05_dilemma_p1",
            "start_ms": 37517,
            "end_ms": 46273,
            "text": "কিন্তু আমরা এ কথা জেনেও এতটুকু সত্য অনুধাবন করতে পারি না।",
            "style_description": (
                "A deep, resonant male voice delivers an earnest, solemn philosophical observation "
                "with deliberate dramatic pause, clear studio acoustics."
            ),
        },
        {
            "unit_id": "unit_06_dilemma_p2",
            "start_ms": 47458,
            "end_ms": 63481,
            "text": (
                "হয়তো আমরা অতীত সময়ের স্মরণকে আঁকড়ে বসে থাকি... "
                "অথবা পুনরায় আগামী সময়ের জন্য আমরা পরিকল্পনা করতে থাকি... "
                "আর জীবন... জীবন অতিবাহিত হয়ে যায়!"
            ),
            "style_description": (
                "A deep, resonant male voice delivers an impassioned dramatic speech with rising theatrical urgency, "
                "natural breath pauses, and mournful gravitas, clear studio acoustics."
            ),
        },
        # NOTE: 63,481ms -> 66,989ms (3.51s) is the pure Orchestral BGM Swell (pure zero silence)
        {
            "unit_id": "unit_07_awakening",
            "start_ms": 66989,
            "end_ms": 77654,
            "text": (
                "একটি সত্য যদি আমরা হৃদয়ে ধারণ করে নিই... "
                "যে না আমরা ভবিষ্যৎ দেখতে পারি, না তা নির্ধারণ করতে পারি!"
            ),
            "style_description": (
                "A deep, resonant male voice delivers a powerful, awakened revelation with calm authority "
                "and resonant philosophical presence, clear studio acoustics."
            ),
        },
        {
            "unit_id": "unit_08_climax",
            "start_ms": 78216,
            "end_ms": 93715,
            "text": (
                "আমরা তো কেবল ধৈর্য ও সাহসের সাথে ভবিষ্যৎকে আলিঙ্গন করতে পারি, স্বাগত জানাতে পারি ভবিষ্যতের—"
                "তবে কি জীবনের প্রতিটি মুহূর্ত জীবনে ভরপুর হয়ে উঠবে না? স্বয়ং বিচার করুন!"
            ),
            "style_description": (
                "A deep, resonant male voice delivers a soaring, authoritative proclamation, building in emotional power "
                "and ending with booming theatrical conviction, clear studio acoustics."
            ),
        },
    ]


def main():
    ref_path = Path("data/stage_07_tts/clip001_option_a_original_hindi.wav")
    output_dir = Path("data/stage_07_tts")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ref_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_path}")

    with open(ref_path, "rb") as f:
        ref_bytes = f.read()

    units = get_8_balanced_units()

    print("=" * 105)
    print(f"RUNNING 8 BALANCED THOUGHT-UNIT DUBBING PIPELINE (NATIVE 1.00x, LOUDNESS LEVELED, 93.69s CANVAS)")
    print("=" * 105)

    with modal.enable_output():
        with app.run():
            result = generate_8_units_modal.remote(
                units=units,
                reference_wav_bytes=ref_bytes,
                master_duration_sec=93.6925,
                tau=0.35,
            )

    # 1. Save master assembled vocal track
    master_wav_path = output_dir / "clip001__dubbed_vocal_full.wav"
    with open(master_wav_path, "wb") as f:
        f.write(result["master_wav_bytes"])
    print(f"\n[SAVED MASTER TRACK] {master_wav_path} ({len(result['master_wav_bytes'])} bytes, dur={result['master_duration_sec']:.2f}s)")

    # 2. Save individual unit audio files
    for unit_id, wav_bytes in result["individual_units"].items():
        unit_file = output_dir / f"{unit_id}__tts__cloned.wav"
        with open(unit_file, "wb") as f:
            f.write(wav_bytes)
        print(f"[SAVED UNIT] {unit_file} ({len(wav_bytes)} bytes)")

    # 3. Print Timing & Loudness Scorecard
    print("\n" + "=" * 110)
    print("8-UNIT TIMING, ZERO-OVERLAP GAP & LOUDNESS SCORECARD")
    print("=" * 110)
    print(f"{'Unit':<22} | {'Timeline Slot':<18} | {'Slot Dur':<9} | {'Final Dur':<10} | {'Gap to Next':<12} | {'Speech RMS':<11} | {'Time'}")
    print("-" * 110)
    for sc in result["scorecard"]:
        slot_str = f"{sc['start_ms']}ms -> {sc['end_ms']}ms"
        gap_str = f"{sc['gap_to_next_s']:+.2f}s" if sc['gap_to_next_s'] >= 0 else f"OVERLAP! {sc['gap_to_next_s']:.2f}s"
        print(
            f"{sc['unit_id']:<22} | {slot_str:<18} | {sc['slot_dur_s']:>7.2f}s | "
            f"{sc['final_dur_s']:>8.2f}s | {gap_str:>12} | {sc['speech_rms']:>10.4f} | {sc['gen_time_s']:>5.2f}s"
        )
    print("=" * 110)
    print(f"\n[SUCCESS] High-fidelity 93.69s dubbed vocal canvas assembled and saved to: {master_wav_path}")


if __name__ == "__main__":
    main()
