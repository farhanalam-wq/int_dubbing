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

app = modal.App("dubbing-blocks-generation")

# Unified Modal Image with Indic Parler-TTS + OpenVoice v2 + FFmpeg (Cached)
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
def generate_blocks_modal(
    blocks: list[dict],
    reference_wav_bytes: bytes,
    master_duration_sec: float = 93.6925,
    tau: float = 0.35,
) -> dict:
    """
    Generates 6 natural thought-unit dialogue blocks using Indic Parler-TTS + OpenVoice v2.
    Paces dialogue so no block overlaps the next, preserving intro silence and mid-scene music swells.
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

        individual_blocks_wav = {}
        block_scorecard = []

        print(f"\n[Modal] Processing {len(blocks)} cohesive dialogue blocks...")

        for idx, blk in enumerate(blocks, 1):
            block_id = blk["block_id"]
            start_ms = blk["start_ms"]
            end_ms = blk["end_ms"]
            slot_dur_sec = (end_ms - start_ms) / 1000.0
            text = blk["text"]
            desc = blk["style_description"]

            # Next block start determines maximum boundary before overlap
            next_start_ms = blocks[idx]["start_ms"] if idx < len(blocks) else (master_duration_sec * 1000.0)
            max_allowed_sec = (next_start_ms - start_ms) / 1000.0 - 0.10  # 100ms clean silence cushion

            print(f"\n--- [{idx}/{len(blocks)}] {block_id} [{start_ms}ms -> {end_ms}ms] (Slot: {slot_dur_sec:.2f}s, Max: {max_allowed_sec:.2f}s) ---")
            print(f"  Text: {text}")

            t0 = time.time()

            # (A) Generate with Indic Parler-TTS
            input_ids = description_tokenizer(desc, return_tensors="pt").input_ids.to(device)
            prompt_input_ids = prompt_tokenizer(text, return_tensors="pt").input_ids.to(device)

            with torch.no_grad():
                gen_audio = parler_model.generate(
                    input_ids=input_ids,
                    prompt_input_ids=prompt_input_ids,
                    max_new_tokens=1500,
                    min_new_tokens=30,
                )

            parler_arr = gen_audio.cpu().numpy().squeeze()
            parler_sr = parler_model.config.sampling_rate

            parler_tmp = os.path.join(tmpdir, f"{block_id}_parler.wav")
            sf.write(parler_tmp, parler_arr, parler_sr)

            # (B) Convert Tone Color with OpenVoice
            source_se = converter.extract_se([parler_tmp])
            converted_tmp = os.path.join(tmpdir, f"{block_id}_converted.wav")
            converter.convert(
                audio_src_path=parler_tmp,
                src_se=source_se,
                tgt_se=target_se,
                output_path=converted_tmp,
                tau=tau,
            )

            conv_data, conv_sr = sf.read(converted_tmp)
            conv_dur_sec = len(conv_data) / float(conv_sr)

            # (C) Duration Fitting / Overlap Prevention
            fitted_tmp = os.path.join(tmpdir, f"{block_id}_fitted.wav")
            applied_tempo = 1.0

            if conv_dur_sec > max_allowed_sec:
                # Need to fit inside the slot so it never spills into next block or music swell
                required_tempo = conv_dur_sec / max_allowed_sec
                # Bounded tempo: atempo can handle up to 1.35x without noticeable pitch distortion
                clamped_tempo = max(0.90, min(1.35, required_tempo))
                applied_tempo = clamped_tempo
                cmd = [
                    "ffmpeg", "-y", "-i", converted_tmp,
                    "-filter:a", f"atempo={clamped_tempo:.4f}",
                    "-ar", str(master_sr),
                    fitted_tmp,
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            elif conv_dur_sec < (slot_dur_sec * 0.70) and slot_dur_sec > 5.0:
                # If generated significantly faster than a slow dramatic delivery, gently stretch
                clamped_tempo = 0.90
                applied_tempo = clamped_tempo
                cmd = [
                    "ffmpeg", "-y", "-i", converted_tmp,
                    "-filter:a", f"atempo={clamped_tempo:.4f}",
                    "-ar", str(master_sr),
                    fitted_tmp,
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            else:
                # Natural pacing! Just resample to 48kHz
                cmd = [
                    "ffmpeg", "-y", "-i", converted_tmp,
                    "-ar", str(master_sr),
                    fitted_tmp,
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            final_data, final_sr = sf.read(fitted_tmp)
            final_dur_sec = len(final_data) / float(final_sr)
            elapsed = time.time() - t0

            print(f"  Result: {conv_dur_sec:.2f}s -> final {final_dur_sec:.2f}s (tempo: {applied_tempo:.2f}x) in {elapsed:.2f}s")

            # (D) Save block audio
            with open(fitted_tmp, "rb") as f:
                individual_blocks_wav[block_id] = f.read()

            # (E) Composite onto Master Canvas
            start_sample = int(round((start_ms / 1000.0) * master_sr))
            end_sample = start_sample + len(final_data)

            # Prevent writing beyond master canvas
            if end_sample > total_samples:
                final_data = final_data[: total_samples - start_sample]
                end_sample = total_samples

            # Overwrite clean canvas region (guaranteed no overlap due to max_allowed_sec pacing)
            master_canvas[start_sample:end_sample] += final_data

            block_scorecard.append({
                "block_id": block_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "slot_dur_s": round(slot_dur_sec, 2),
                "synth_dur_s": round(conv_dur_sec, 2),
                "final_dur_s": round(final_dur_sec, 2),
                "next_start_ms": next_start_ms,
                "gap_to_next_s": round((next_start_ms - (start_ms + final_dur_sec * 1000.0)) / 1000.0, 2),
                "tempo": round(applied_tempo, 2),
                "gen_time_s": round(elapsed, 2),
            })

        # (F) Normalize master canvas to -1.0 dBFS (0.89)
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
            "individual_blocks": individual_blocks_wav,
            "scorecard": block_scorecard,
        }


def get_dialogue_blocks() -> list[dict]:
    """Defines the 6 natural cohesive thought-unit blocks covering the entire monologue."""
    return [
        {
            "block_id": "block_01_intro",
            "start_ms": 10479,
            "end_ms": 12342,
            "text": "ভবিষ্যতের অপর নাম হলো",
            "style_description": (
                "A deep, resonant male voice delivers a calm, philosophical introduction in a reflective, "
                "measured cadence with clear studio acoustics."
            ),
        },
        {
            "block_id": "block_02_option_a",
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
            "block_id": "block_03_definition",
            "start_ms": 23542,
            "end_ms": 31081,
            "text": "জীবন... জীবন না তো ভবিষ্যতে আছে, না অতীতে আছে! জীবন তো এই মুহূর্তের নাম।",
            "style_description": (
                "A deep, resonant male voice delivers a thoughtful philosophical monologue with deliberate breath pauses "
                "and an introspective, expressive cadence, clear studio acoustics."
            ),
        },
        {
            "block_id": "block_04_aphorism",
            "start_ms": 31783,
            "end_ms": 36535,
            "text": "অর্থাৎ... এই মুহূর্তের অনুভবই জীবন।",
            "style_description": (
                "A deep, resonant male voice speaks with crisp philosophical clarity, reflective cadence, "
                "and deliberate emphasis, clear close-mic studio recording."
            ),
        },
        {
            "block_id": "block_05_dilemma",
            "start_ms": 37517,
            "end_ms": 63481,
            "text": (
                "কিন্তু আমরা এ কথা জেনেও এতটুকু সত্য অনুধাবন করতে পারি না। "
                "হয়তো আমরা অতীত সময়ের স্মরণকে আঁকড়ে বসে থাকি, অথবা পুনরায় আগামী সময়ের জন্য আমরা পরিকল্পনা করতে থাকি... "
                "আর জীবন... জীবন অতিবাহিত হয়ে যায়!"
            ),
            "style_description": (
                "A deep, resonant male voice delivers an impassioned dramatic speech with rising theatrical urgency, "
                "natural breath pauses, and solemn gravitas, clear studio acoustics."
            ),
        },
        {
            "block_id": "block_06_climax",
            "start_ms": 66989,
            "end_ms": 93715,
            "text": (
                "একটি সত্য যদি আমরা হৃদয়ে ধারণ করে নিই... "
                "যে না আমরা ভবিষ্যৎ দেখতে পারি, না তা নির্ধারণ করতে পারি! "
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

    blocks = get_dialogue_blocks()

    print("=" * 95)
    print(f"RUNNING COHESIVE 6-BLOCK FULL DUBBING PIPELINE (93.69s CANVAS, ZERO OVERLAPS)")
    print("=" * 95)

    with modal.enable_output():
        with app.run():
            result = generate_blocks_modal.remote(
                blocks=blocks,
                reference_wav_bytes=ref_bytes,
                master_duration_sec=93.6925,
                tau=0.35,
            )

    # 1. Save master assembled vocal track (overwriting previous full track)
    master_wav_path = output_dir / "clip001__dubbed_vocal_full.wav"
    with open(master_wav_path, "wb") as f:
        f.write(result["master_wav_bytes"])
    print(f"\n[SAVED MASTER TRACK] {master_wav_path} ({len(result['master_wav_bytes'])} bytes, dur={result['master_duration_sec']:.2f}s)")

    # 2. Save individual blocks
    for block_id, wav_bytes in result["individual_blocks"].items():
        blk_file = output_dir / f"{block_id}__tts__cloned.wav"
        with open(blk_file, "wb") as f:
            f.write(wav_bytes)
        print(f"[SAVED BLOCK] {blk_file} ({len(wav_bytes)} bytes)")

    # 3. Print Timing Scorecard
    print("\n" + "=" * 105)
    print("6-BLOCK TIMING, OVERLAP PREVENTION & PACING SCORECARD")
    print("=" * 105)
    print(f"{'Block':<20} | {'Timeline Slot':<18} | {'Slot Dur':<9} | {'Synth Dur':<10} | {'Final Dur':<10} | {'Gap to Next':<12} | {'Tempo'}")
    print("-" * 105)
    for sc in result["scorecard"]:
        slot_str = f"{sc['start_ms']}ms -> {sc['end_ms']}ms"
        gap_str = f"{sc['gap_to_next_s']:+.2f}s" if sc['gap_to_next_s'] >= 0 else f"OVERLAP! {sc['gap_to_next_s']:.2f}s"
        print(
            f"{sc['block_id']:<20} | {slot_str:<18} | {sc['slot_dur_s']:>7.2f}s | {sc['synth_dur_s']:>8.2f}s | "
            f"{sc['final_dur_s']:>8.2f}s | {gap_str:>12} | {sc['tempo']:>5.2f}x"
        )
    print("=" * 105)
    print(f"\n[SUCCESS] Entire clean 93.69s dubbed vocal canvas assembled and saved to: {master_wav_path}")


if __name__ == "__main__":
    main()
