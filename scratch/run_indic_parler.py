import sys
import os
from pathlib import Path

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ["PYTHONIOENCODING"] = "utf-8"

import modal

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
def generate_indic_parler(
    prompts: list[tuple[str, str]], # (tag, text)
    description: str,
):
    import io
    import torch
    import soundfile as sf
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading ai4bharat/indic-parler-tts on {device}...")

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "ai4bharat/indic-parler-tts",
        token=token,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts", token=token)
    desc_model_name = getattr(model.config.text_encoder, "_name_or_path", "google/flan-t5-base")
    print(f"Loading description tokenizer: {desc_model_name}")
    description_tokenizer = AutoTokenizer.from_pretrained(desc_model_name, token=token)

    sampling_rate = model.config.sampling_rate
    print(f"Model sampling rate: {sampling_rate} Hz")
    print(f"Style description: '{description}'")

    # In ParlerTTS: input_ids is the description, prompt_input_ids is the text to speak!
    desc_tokens = description_tokenizer(description, return_tensors="pt")
    input_ids = desc_tokens.input_ids.to(device)
    attention_mask = desc_tokens.attention_mask.to(device)

    outputs = {}
    for tag, text in prompts:
        print(f"Generating [{tag}]: '{text}'...")
        prompt_tokens = tokenizer(text, return_tensors="pt")
        prompt_input_ids = prompt_tokens.input_ids.to(device)
        prompt_attention_mask = prompt_tokens.attention_mask.to(device)

        with torch.inference_mode():
            generation = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
            )

        audio_arr = generation.cpu().numpy().squeeze()
        
        buf = io.BytesIO()
        sf.write(buf, audio_arr, sampling_rate, format="WAV")
        outputs[tag] = (buf.getvalue(), sampling_rate)
        print(f"Finished [{tag}]: {len(outputs[tag][0])} bytes")

    return outputs

def main():
    prompts = [
        ("simple", "আমি ভালো আছি"),
        ("seg001", "ভবিষ্যতের অপর নাম হলো"),
        ("seg002", "সংঘর্ষ! হৃদয়ে আজ ইচ্ছা জাগে এবং তা যদি পূর্ণ না হতে পারে,"),
    ]

    # Description matching our dramatic mythological male character
    description = (
        "A male speaker delivers a deep, dramatic, resonant, and theatrical dialogue "
        "with passionate emotional intensity, close-mic studio recording with clear acoustics."
    )

    print("Launching Indic Parler-TTS on Modal...")
    with modal.enable_output():
        with app.run():
            results = generate_indic_parler.remote(prompts, description)

    out_dir = Path("data/stage_07_tts")
    out_dir.mkdir(parents=True, exist_ok=True)

    filenames = {
        "simple": "clip001_simple__tts__indic_parler.wav",
        "seg001": "clip001_seg001__tts__indic_parler.wav",
        "seg002": "clip001_seg002__tts__indic_parler.wav",
    }

    for tag, (wav_bytes, sr) in results.items():
        dst = out_dir / filenames[tag]
        with open(dst, "wb") as f:
            f.write(wav_bytes)
        print(f"Saved: {dst} ({len(wav_bytes)} bytes, sr={sr})")

if __name__ == "__main__":
    main()
