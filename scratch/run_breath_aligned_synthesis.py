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

app = modal.App("dubbing-breath-aligned-generation")

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
def generate_breath_aligned_modal(
    phrases: list[dict],
    reference_wav_bytes: bytes,
    master_duration_sec: float = 93.6925,
    tau: float = 0.35,
) -> dict:
    """
    Synthesizes and voice-converts breath-aligned dialogue phrases.
    Re-uses pre-cached audio for proven Units 1, 2, and 3.
    Preserves 1.00x native playback, levels active speech RMS to ~0.08,
    and composites onto a 48kHz master canvas matching Krishna's visual pauses.
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

        master_sr = 48000
        total_samples = int(round(master_duration_sec * master_sr))
        master_canvas = np.zeros(total_samples, dtype=np.float32)

        individual_phrases_wav = {}
        phrase_scorecard = []

        print(f"\n[Modal] Processing {len(phrases)} breath-aligned phrases...")
        TARGET_SPEECH_RMS = 0.08  # ~ -22 dBFS speech level

        for idx, p in enumerate(phrases, 1):
            phrase_id = p["phrase_id"]
            start_ms = p["start_ms"]
            end_ms = p["end_ms"]
            slot_dur_sec = (end_ms - start_ms) / 1000.0
            text = p.get("text", "")
            desc = p.get("style_description", "")
            cached_bytes = p.get("cached_wav_bytes")

            next_start_ms = phrases[idx]["start_ms"] if idx < len(phrases) else (master_duration_sec * 1000.0)
            max_allowed_sec = (next_start_ms - start_ms) / 1000.0 - 0.05  # 50ms clean silence cushion

            print(f"\n--- [{idx}/{len(phrases)}] {phrase_id} [{start_ms}ms -> {end_ms}ms] (Slot: {slot_dur_sec:.2f}s, Max: {max_allowed_sec:.2f}s) ---")

            t0 = time.time()

            if cached_bytes:
                print(f"  [Reusing Cached Audio] ({len(cached_bytes):,} bytes)")
                cached_path = os.path.join(tmpdir, f"{phrase_id}_cached.wav")
                with open(cached_path, "wb") as f:
                    f.write(cached_bytes)
                unit_data, c_sr = sf.read(cached_path)
                if c_sr != master_sr:
                    c48_path = os.path.join(tmpdir, f"{phrase_id}_c48.wav")
                    cmd = ["ffmpeg", "-y", "-i", cached_path, "-ar", str(master_sr), c48_path]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    unit_data, _ = sf.read(c48_path)
                if unit_data.ndim > 1:
                    unit_data = np.mean(unit_data, axis=1)
                individual_phrases_wav[phrase_id] = cached_bytes
            else:
                print(f"  Text: {text}")

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

                parler_tmp = os.path.join(tmpdir, f"{phrase_id}_parler.wav")
                sf.write(parler_tmp, parler_arr, parler_sr)

                # (B) Tone Color Conversion with OpenVoice v2
                source_se = converter.extract_se([parler_tmp])
                converted_tmp = os.path.join(tmpdir, f"{phrase_id}_converted.wav")
                converter.convert(
                    audio_src_path=parler_tmp,
                    src_se=source_se,
                    tgt_se=target_se,
                    output_path=converted_tmp,
                    tau=tau,
                )

                # (C) Clean resample to 48kHz
                resampled_tmp = os.path.join(tmpdir, f"{phrase_id}_48k.wav")
                cmd = ["ffmpeg", "-y", "-i", converted_tmp, "-ar", str(master_sr), resampled_tmp]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

                unit_data, _ = sf.read(resampled_tmp)
                if unit_data.ndim > 1:
                    unit_data = np.mean(unit_data, axis=1)

                # Trim trailing silence if exceeds max_allowed_sec
                current_dur_sec = len(unit_data) / float(master_sr)
                if current_dur_sec > max_allowed_sec:
                    max_samples = int(round(max_allowed_sec * master_sr))
                    unit_data = unit_data[:max_samples]
                    fade_len = int(round(0.04 * master_sr))
                    if len(unit_data) > fade_len:
                        fade_curve = np.linspace(1.0, 0.0, fade_len)
                        unit_data[-fade_len:] *= fade_curve
                    print(f"  [Safety Cushion] Trimmed tail to fit {max_allowed_sec:.2f}s slot.")

                # (D) Active-Speech Loudness Normalization
                active_frames = unit_data[np.abs(unit_data) > 0.01]
                if len(active_frames) > 0:
                    speech_rms = np.sqrt(np.mean(active_frames ** 2))
                    if speech_rms > 0.005:
                        gain = TARGET_SPEECH_RMS / speech_rms
                        clamped_gain = max(0.4, min(2.5, gain))
                        unit_data = unit_data * clamped_gain
                        unit_peak = np.max(np.abs(unit_data))
                        if unit_peak > 0.89:
                            unit_data = unit_data * (0.89 / unit_peak)
                        print(f"  [Leveling] Speech RMS: {speech_rms:.4f} -> Gain: {clamped_gain:.2f}x (Peak: {np.max(np.abs(unit_data)):.3f})")

                # Save phrase audio
                normalized_tmp = os.path.join(tmpdir, f"{phrase_id}_final.wav")
                sf.write(normalized_tmp, unit_data, master_sr, format="WAV", subtype="PCM_16")
                with open(normalized_tmp, "rb") as f:
                    individual_phrases_wav[phrase_id] = f.read()

            final_dur_sec = len(unit_data) / float(master_sr)
            elapsed = time.time() - t0

            # (E) Composite onto Master Canvas
            start_sample = int(round((start_ms / 1000.0) * master_sr))
            end_sample = start_sample + len(unit_data)

            if end_sample > total_samples:
                unit_data = unit_data[: total_samples - start_sample]
                end_sample = total_samples

            master_canvas[start_sample:end_sample] += unit_data

            gap_sec = (next_start_ms - (start_ms + final_dur_sec * 1000.0)) / 1000.0
            print(f"  Finished: {final_dur_sec:.2f}s (Gap to next: {gap_sec:+.2f}s) in {elapsed:.2f}s")

            phrase_scorecard.append({
                "phrase_id": phrase_id,
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
            "individual_phrases": individual_phrases_wav,
            "scorecard": phrase_scorecard,
        }


def get_breath_aligned_phrases() -> list[dict]:
    """
    Defines the breath-aligned dialogue phrases matching Krishna's actual
    visual delivery and contemplative pauses from WhisperX alignment.
    """
    # Load cached Unit 1, Unit 2, Unit 3 audio bytes
    cached_u1 = Path("data/stage_07_tts/unit_01_intro__tts__cloned.wav").read_bytes()
    cached_u2 = Path("data/stage_07_tts/unit_02_option_a__tts__cloned.wav").read_bytes()
    cached_u3 = Path("data/stage_07_tts/unit_03_definition__tts__cloned.wav").read_bytes()

    return [
        {
            "phrase_id": "unit_01_intro",
            "start_ms": 10479,
            "end_ms": 12342,
            "cached_wav_bytes": cached_u1,
        },
        {
            "phrase_id": "unit_02_option_a",
            "start_ms": 12903,
            "end_ms": 22981,
            "cached_wav_bytes": cached_u2,
        },
        {
            "phrase_id": "unit_03_definition",
            "start_ms": 23542,
            "end_ms": 31081,
            "cached_wav_bytes": cached_u3,
        },
        # Unit 4: The Aphorism (split across Krishna's 1.7s dramatic smile pause)
        {
            "phrase_id": "phrase_04a",
            "start_ms": 31783,
            "end_ms": 32600,
            "text": "অর্থাৎ,",
            "style_description": (
                "A deep, resonant male voice speaks with crisp philosophical clarity and reflective cadence, "
                "clear close-mic studio recording."
            ),
        },
        {
            "phrase_id": "phrase_04b",
            "start_ms": 34250,
            "end_ms": 36535,
            "text": "এই মুহূর্তের অনুভবই জীবন।",
            "style_description": (
                "A deep, resonant male voice delivers a thoughtful philosophical conclusion with deliberate emphasis, "
                "clear studio acoustics."
            ),
        },
        # Unit 5: The Human Dilemma (split across Krishna's 1.8s breath pause)
        {
            "phrase_id": "phrase_05a",
            "start_ms": 37517,
            "end_ms": 42130,
            "text": "কিন্তু আমরা এ কথা জেনেও এতটুকু...",
            "style_description": (
                "A deep, resonant male voice delivers an earnest, solemn philosophical observation "
                "with deliberate dramatic presence, clear studio acoustics."
            ),
        },
        {
            "phrase_id": "phrase_05b",
            "start_ms": 43920,
            "end_ms": 46273,
            "text": "সত্য অনুধাবন করতে পারি না।",
            "style_description": (
                "A deep, resonant male voice speaks with quiet, solemn regret and deliberate emphasis, "
                "clear studio acoustics."
            ),
        },
        # Unit 6: The Dilemma Clauses & Climax
        {
            "phrase_id": "phrase_06a",
            "start_ms": 47458,
            "end_ms": 51600,
            "text": "হয়তো আমরা অতীত সময়ের স্মরণকে আঁকড়ে বসে থাকি...",
            "style_description": (
                "A deep, resonant male voice delivers a reflective dramatic monologue with rising urgency "
                "and mournful gravitas, clear studio acoustics."
            ),
        },
        {
            "phrase_id": "phrase_06b",
            "start_ms": 52520,
            "end_ms": 57800,
            "text": "অথবা পুনরায় আগামী সময়ের জন্য আমরা পরিকল্পনা করতে থাকি...",
            "style_description": (
                "A deep, resonant male voice speaks with passionate intensity and deliberate cadence, "
                "clear studio acoustics."
            ),
        },
        {
            "phrase_id": "phrase_06c",
            "start_ms": 58500,
            "end_ms": 63481,
            "text": "আর জীবন... জীবন অতিবাহিত হয়ে যায়!",
            "style_description": (
                "A deep, resonant male voice delivers a sorrowful, dramatic climax with natural breath pauses "
                "and deep emotional resonance, clear studio acoustics."
            ),
        },
        # NOTE: 63,481ms -> 66,989ms (3.51s) is pure zero silence for the Orchestral BGM Swell
        # Unit 7: The Awakening Revelation
        {
            "phrase_id": "phrase_07a",
            "start_ms": 66989,
            "end_ms": 68600,
            "text": "একটি সত্য যদি আমরা হৃদয়ে ধারণ করে নিই...",
            "style_description": (
                "A deep, resonant male voice delivers an awakened revelation with calm authority "
                "and philosophical presence, clear studio acoustics."
            ),
        },
        {
            "phrase_id": "phrase_07b",
            "start_ms": 69830,
            "end_ms": 77654,
            "text": "যে না আমরা ভবিষ্যৎ দেখতে পারি, না তা নির্ধারণ করতে পারি!",
            "style_description": (
                "A deep, resonant male voice speaks with authoritative conviction and resonant power, "
                "clear studio acoustics."
            ),
        },
        # Unit 8: The Proclamation & Grand Climax
        {
            "phrase_id": "phrase_08a",
            "start_ms": 78216,
            "end_ms": 80200,
            "text": "আমরা তো কেবল ধৈর্য ও সাহসের সাথে ভবিষ্যৎকে আলিঙ্গন করতে পারি...",
            "style_description": (
                "A deep, resonant male voice speaks with noble philosophical encouragement, clear studio acoustics."
            ),
        },
        {
            "phrase_id": "phrase_08b",
            "start_ms": 80540,
            "end_ms": 93715,
            "text": (
                "স্বাগত জানাতে পারি ভবিষ্যতের—তবে কি জীবনের প্রতিটি মুহূর্ত জীবনে ভরপুর হয়ে উঠবে না? স্বয়ং বিচার করুন!"
            ),
            "style_description": (
                "A deep, resonant male voice delivers a soaring, authoritative proclamation, building in emotional power "
                "and ending with booming theatrical conviction, clear studio acoustics."
            ),
        },
    ]


def main():
    ref_path = Path("data/stage_07_tts/clip001_option_a_original_hindi.wav")
    out_dir = Path("data/stage_07_tts")

    if not ref_path.exists():
        print(f"Error: {ref_path} not found!")
        return

    print("Reading reference audio bytes...")
    ref_bytes = ref_path.read_bytes()

    phrases = get_breath_aligned_phrases()
    print(f"Prepared {len(phrases)} breath-aligned phrases (Units 1, 2, 3 cached).")

    print("\nSubmitting breath-aligned generation job to Modal L4 GPU...")
    with modal.enable_output():
        with app.run():
            result = generate_breath_aligned_modal.remote(
                phrases=phrases,
                reference_wav_bytes=ref_bytes,
                master_duration_sec=93.6925,
                tau=0.35,
            )

    # Save individual phrases
    for pid, wbytes in result["individual_phrases"].items():
        p_out = out_dir / f"{pid}__tts__cloned.wav"
        p_out.write_bytes(wbytes)
        print(f"Saved: {p_out} ({len(wbytes):,} bytes)")

    # Save master canvas
    master_path = out_dir / "clip001__dubbed_vocal_full.wav"
    master_path.write_bytes(result["master_wav_bytes"])
    print(f"\n[SUCCESS] Wrote new master dubbed vocal canvas: {master_path} ({len(result['master_wav_bytes']):,} bytes)")
    print(f"  Duration: {result['master_duration_sec']:.3f}s at {result['master_sample_rate']}Hz")

    # Print Scorecard
    print("\n" + "=" * 105)
    print("BREATH-ALIGNED TIMING, OVERLAP PREVENTION & PACING SCORECARD")
    print("=" * 105)
    print(f"{'Phrase ID':<22} | {'Timeline Slot':<20} | {'Slot Dur':>9} | {'Final Dur':>10} | {'Gap to Next':>12} | {'Speech RMS':>10}")
    print("-" * 105)
    for sc in result["scorecard"]:
        slot_str = f"{sc['start_ms']}ms -> {sc['end_ms']}ms"
        print(
            f"{sc['phrase_id']:<22} | "
            f"{slot_str:<20} | "
            f"{sc['slot_dur_s']:>8.2f}s | "
            f"{sc['final_dur_s']:>9.2f}s | "
            f"{sc['gap_to_next_s']:>+11.2f}s | "
            f"{sc['speech_rms']:>10.4f}"
        )
    print("=" * 105)


if __name__ == "__main__":
    main()
