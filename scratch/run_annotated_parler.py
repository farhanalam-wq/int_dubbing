import io
import os
import sys
import time
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

app = modal.App("run-indic-parler-tts")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.2.0",
        "torchaudio>=2.2.0",
        "transformers",
        "soundfile",
        "scipy",
        "huggingface_hub",
        "git+https://github.com/huggingface/parler-tts.git",
    )
)

@app.function(
    gpu="L4",
    image=image,
    timeout=600,
    secrets=[modal.Secret.from_name("hf-secret")],
)
def generate_annotated_pairs(
    items: list[tuple[str, str, str]],  # (tag, text, description)
    repo_id: str = "ai4bharat/indic-parler-tts",
):
    import io
    import torch
    import soundfile as sf
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading {repo_id} on {device}...")

    model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id).to(device)
    prompt_tokenizer = AutoTokenizer.from_pretrained(repo_id)

    description_tokenizer_id = getattr(model.config, "text_encoder", None)
    if description_tokenizer_id and hasattr(description_tokenizer_id, "_name_or_path"):
        desc_name = description_tokenizer_id._name_or_path
    else:
        desc_name = "google/flan-t5-large"
    print(f"Loading description tokenizer: {desc_name}")
    desc_tokenizer = AutoTokenizer.from_pretrained(desc_name)

    sampling_rate = model.config.sampling_rate
    print(f"Model sampling rate: {sampling_rate} Hz")

    outputs = {}
    for tag, text, style_desc in items:
        print(f"\nGenerating [{tag}]:")
        print(f"  Style: '{style_desc}'")
        print(f"  Text: '{text}'")

        # 1. Tokenize style description (conditioned through Flan-T5 text encoder)
        desc_tokens = desc_tokenizer(style_desc, return_tensors="pt").to(device)

        # 2. Tokenize spoken Bengali text
        prompt_tokens = prompt_tokenizer(text, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            generation = model.generate(
                input_ids=desc_tokens.input_ids,
                attention_mask=desc_tokens.attention_mask,
                prompt_input_ids=prompt_tokens.input_ids,
                prompt_attention_mask=prompt_tokens.attention_mask,
            )
        gen_time = time.time() - t0

        audio_arr = generation.cpu().numpy().squeeze()
        buf = io.BytesIO()
        sf.write(buf, audio_arr, sampling_rate, format="WAV")
        outputs[tag] = (buf.getvalue(), sampling_rate, len(audio_arr) / sampling_rate, gen_time)
        print(f"Finished [{tag}]: {len(audio_arr)/sampling_rate:.2f}s audio in {gen_time:.2f}s")

    return outputs


def main():
    output_dir = Path("data/stage_07_tts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Option A Monologue (~10.08s in original scene)
    # Annotated Text with natural prosodic pauses (ellipsis & em-dash)
    annotated_text = (
        "সংঘর্ষ! হৃদয়ে আজ ইচ্ছা জাগে এবং তা যদি পূর্ণ না হতে পারে... "
        "তবে হৃদয় ভবিষ্যতের পরিকল্পনা প্রস্তুত করে—ভবিষ্যতে ইচ্ছা পূর্ণ হবে, "
        "এমন কল্পনা করতে থাকে... কিন্তু..."
    )

    # Baseline Raw Text (no expressive pause marks)
    baseline_text = (
        "সংঘর্ষ হৃদয়ে আজ ইচ্ছা জাগে এবং তা যদি পূর্ণ না হতে পারে "
        "তবে হৃদয় ভবিষ্যতের পরিকল্পনা প্রস্তুত করে ভবিষ্যতে ইচ্ছা পূর্ণ হবে "
        "এমন কল্পনা করতে থাকে কিন্তু"
    )

    # 1. Dynamically Annotated Style: Captures the opening burst + passion + reflective pause
    annotated_style = (
        "A deep, resonant male voice delivers an intensely dramatic theatrical monologue. "
        "He opens with an explosive, powerful proclamation on the first word, then transitions into a passionate, "
        "melodic cadence with deliberate breath pauses, ending in a quiet suspended tone with clear studio acoustics."
    )

    # 2. Baseline Flat Generic Style
    baseline_style = (
        "A male speaker speaks in a calm, neutral tone with normal speed, close-mic studio recording."
    )

    items = [
        ("option_a_annotated", annotated_text, annotated_style),
        ("option_a_baseline", baseline_text, baseline_style),
    ]

    print("=" * 80)
    print("RUNNING OPTION A (10.08s) INDIC PARLER-TTS PROSODY & TONALITY TEST")
    print("=" * 80)

    with modal.enable_output():
        with app.run():
            results = generate_annotated_pairs.remote(items)

    for tag, (wav_bytes, sr, dur, gen_time) in results.items():
        dst = output_dir / f"clip001_{tag}.wav"
        with open(dst, "wb") as f:
            f.write(wav_bytes)
        print(f"[SAVED] {dst} ({len(wav_bytes)} bytes, dur={dur:.2f}s, sr={sr}Hz, gen_time={gen_time:.2f}s)")

    print("\n[SUCCESS] Both Option A audio versions synthesized successfully!")


if __name__ == "__main__":
    main()
