import sys
import os
from pathlib import Path
sys.path.insert(0, ".")

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ["PYTHONIOENCODING"] = "utf-8"

import modal

app = modal.App("inspect-indicf5-and-run")

image = (
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
        "f5-tts",
        "vocos",
        "cached_path",
    )
)

@app.function(
    gpu="L4",
    image=image,
    timeout=600,
    secrets=[modal.Secret.from_name("hf-secret")],
)
def run_indicf5_tiers(
    ref_wav_bytes_seg001: bytes,
    ref_text_seg001: str,
    ref_wav_bytes_seg002: bytes,
    ref_text_seg002: str,
):
    import os
    import tempfile
    import inspect
    import torch
    import soundfile as sf
    import numpy as np
    from transformers import AutoModel

    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import contextlib
    try:
        import accelerate.big_modeling
        accelerate.big_modeling.init_empty_weights = contextlib.nullcontext
    except Exception as e:
        print("Accelerate patch note:", e)

    print("Loading ai4bharat/IndicF5...")
    model = AutoModel.from_pretrained(
        "ai4bharat/IndicF5",
        trust_remote_code=True,
        low_cpu_mem_usage=False,
        token=hf_token,
    ).to(device).eval()

    # Inspect tokenizer / G2P in IndicF5
    info = {
        "model_class": str(type(model)),
        "call_signature": str(inspect.signature(model.__call__)),
        "vocab_path": getattr(model, "vocab_path", None),
    }

    # Write prompt audios to temp files
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f1:
        f1.write(ref_wav_bytes_seg001)
        ref_path_1 = f1.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f2:
        f2.write(ref_wav_bytes_seg002)
        ref_path_2 = f2.name

    tiers = [
        ("simple", "আমি ভালো আছি", ref_path_1, ref_text_seg001),
        ("seg001", "ভবিষ্যতের অপর নাম হলো", ref_path_1, ref_text_seg001),
        ("seg002", "সংঘর্ষ! হৃদয়ে আজ ইচ্ছা জাগে এবং তা যদি পূর্ণ না হতে পারে,", ref_path_2, ref_text_seg002),
    ]

    audio_outputs = {}

    try:
        for tag, target_text, r_path, r_text in tiers:
            print(f"Synthesizing {tag}: '{target_text}'...")
            with torch.inference_mode():
                out_audio = model(
                    target_text,
                    ref_audio_path=r_path,
                    ref_text=r_text,
                )
            
            # Normalize to 16-bit PCM wav bytes
            if hasattr(out_audio, "cpu"):
                out_audio = out_audio.cpu().numpy()
            if isinstance(out_audio, np.ndarray):
                if out_audio.dtype == np.int16:
                    audio_float = out_audio.astype(np.float32) / 32768.0
                else:
                    audio_float = out_audio.astype(np.float32)
            else:
                audio_float = np.array(out_audio, dtype=np.float32)

            out_buf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(out_buf.name, audio_float, 24000)
            with open(out_buf.name, "rb") as f:
                audio_outputs[tag] = f.read()
            os.remove(out_buf.name)
            print(f"Done {tag}: {len(audio_outputs[tag])} bytes")

    finally:
        for p in [ref_path_1, ref_path_2]:
            if os.path.exists(p):
                os.remove(p)

    return info, audio_outputs

def main():
    import importlib
    mod = importlib.import_module("scripts.09_tts")
    slice_wav_bytes = mod.slice_wav_bytes

    vocal_path = "data/stage_02_separated/clip001__separated__vocal.wav"
    b1 = slice_wav_bytes(vocal_path, 10300, 12500, target_rate=24000)
    t1 = "भविष्य का दूसरा नाम है"

    b2 = slice_wav_bytes(vocal_path, 12850, 15350, target_rate=24000)
    t2 = "संघर्ष! हृदय में आज इच्छा होती है और यदि पूर्ण नहीं हो पाती"

    print("Launching IndicF5 synthesis on Modal...")
    with modal.enable_output():
        with app.run():
            info, audios = run_indicf5_tiers.remote(b1, t1, b2, t2)

    print("IndicF5 Model Info:", info)
    out_dir = Path("data/stage_07_tts")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save audio files following convention: clip001_{seg_id}__tts__{model_name}.wav
    naming = {
        "simple": "clip001_simple__tts__indicf5.wav",
        "seg001": "clip001_seg001__tts__indicf5.wav",
        "seg002": "clip001_seg002__tts__indicf5.wav",
    }
    for tag, fname in naming.items():
        if tag in audios:
            p = out_dir / fname
            with open(p, "wb") as f:
                f.write(audios[tag])
            print(f"Saved: {p}")

if __name__ == "__main__":
    main()
