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

app = modal.App("run-indicf5-bakeoff")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.2.0",
        "torchaudio>=2.2.0",
        "soundfile",
        "scipy",
        "huggingface_hub",
        "f5-tts",
        "vocos",
        "pydub",
    )
)

@app.function(
    gpu="L4",
    image=image,
    timeout=600,
    secrets=[modal.Secret.from_name("hf-secret")],
)
def synthesize_indicf5(
    ref_wav_bytes_seg001: bytes,
    ref_text_seg001: str,
    ref_wav_bytes_seg002: bytes,
    ref_text_seg002: str,
):
    import os
    import tempfile
    import torch
    import soundfile as sf
    import numpy as np
    from huggingface_hub import hf_hub_download
    from f5_tts.infer.utils_infer import (
        infer_process,
        load_model,
        load_vocoder,
        preprocess_ref_audio_text,
    )
    from f5_tts.model import DiT

    token = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"IndicF5 device: {device}")

    # 1. Vocoder
    print("Loading Vocos vocoder...")
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

    # 2. Download vocab & checkpoint
    print("Downloading IndicF5 checkpoints...")
    vocab_path = hf_hub_download("ai4bharat/IndicF5", filename="checkpoints/vocab.txt", token=token)
    
    from safetensors.torch import load_file
    safetensors_path = hf_hub_download("ai4bharat/IndicF5", filename="model.safetensors", token=token)
    sd = load_file(safetensors_path, device=device)

    # 3. Model
    print("Loading IndicF5 DiT backbone...")
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    from f5_tts.model import CFM, DiT
    from f5_tts.model.utils import get_tokenizer

    vocab_char_map, vocab_size = get_tokenizer(vocab_path, "custom")
    model = CFM(
        transformer=DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100),
        mel_spec_kwargs=dict(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=100,
            target_sample_rate=24000,
            mel_spec_type="vocos",
        ),
        vocab_char_map=vocab_char_map,
    )

    model_sd = {}
    vocoder_sd = {}
    for k, v in sd.items():
        if k.startswith("_orig_mod.transformer."):
            model_sd[k.replace("_orig_mod.", "")] = v
        elif k.startswith("transformer."):
            model_sd[k] = v
        elif k.startswith("vocoder._orig_mod."):
            vocoder_sd[k.replace("vocoder._orig_mod.", "")] = v
        elif k.startswith("vocoder."):
            vocoder_sd[k.replace("vocoder.", "")] = v
        else:
            clean_k = k.replace("_orig_mod.", "")
            model_sd[clean_k] = v

    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    print(f"Loaded DiT weights: mapped {len(model_sd)} keys (missing={len(missing)}, unexpected={len(unexpected)})")
    if vocoder_sd:
        v_missing, v_unexpected = vocoder.load_state_dict(vocoder_sd, strict=False)
        print(f"Loaded Vocos weights: mapped {len(vocoder_sd)} keys (missing={len(v_missing)}, unexpected={len(v_unexpected)})")

    model.to(device).eval()

    # 4. Reference audio files
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f1:
        f1.write(ref_wav_bytes_seg001)
        r1_path = f1.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f2:
        f2.write(ref_wav_bytes_seg002)
        r2_path = f2.name

    tiers = [
        ("simple", "আমি ভালো আছি", r1_path, ref_text_seg001),
        ("seg001", "ভবিষ্যতের অপর নাম হলো", r1_path, ref_text_seg001),
        ("seg002", "সংঘর্ষ! হৃদয়ে আজ ইচ্ছা জাগে এবং তা যদি পূর্ণ না হতে পারে,", r2_path, ref_text_seg002),
    ]

    results = {}
    try:
        for tag, target_text, ref_path, ref_text in tiers:
            print(f"Synthesizing Tier [{tag}]: '{target_text}'...")
            ref_audio, proc_ref_text = preprocess_ref_audio_text(ref_path, ref_text)
            
            with torch.inference_mode():
                audio, final_sr, _ = infer_process(
                    ref_audio,
                    proc_ref_text,
                    target_text,
                    model,
                    vocoder,
                    mel_spec_type="vocos",
                    speed=1.0,
                    device=device,
                )

            # Ensure numpy float32
            if hasattr(audio, "cpu"):
                audio = audio.cpu().numpy()
            audio = np.array(audio, dtype=np.float32)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_f:
                sf.write(out_f.name, audio, final_sr or 24000)
                out_path = out_f.name

            with open(out_path, "rb") as f:
                results[tag] = (f.read(), final_sr or 24000)
            os.remove(out_path)
            print(f"Completed Tier [{tag}]: {len(results[tag][0])} bytes @ {final_sr or 24000}Hz")

    finally:
        for p in [r1_path, r2_path]:
            if os.path.exists(p):
                os.remove(p)

    return results

def main():
    import importlib
    sys.path.insert(0, ".")
    mod = importlib.import_module("scripts.09_tts")
    slice_wav_bytes = mod.slice_wav_bytes

    vocal_path = "data/stage_02_separated/clip001__separated__vocal.wav"
    b1 = slice_wav_bytes(vocal_path, 10300, 12500, target_rate=24000)
    t1 = "भविष्य का दूसरा नाम है"

    b2 = slice_wav_bytes(vocal_path, 12850, 15350, target_rate=24000)
    t2 = "संघर्ष! हृदय में आज इच्छा होती है और यदि पूर्ण नहीं हो पाती"

    print(f"Reference slices extracted: seg001={len(b1)} bytes, seg002={len(b2)} bytes")

    with modal.enable_output():
        with app.run():
            results = synthesize_indicf5.remote(b1, t1, b2, t2)

    out_dir = Path("data/stage_07_tts")
    out_dir.mkdir(parents=True, exist_ok=True)

    filenames = {
        "simple": "clip001_simple__tts__indicf5.wav",
        "seg001": "clip001_seg001__tts__indicf5.wav",
        "seg002": "clip001_seg002__tts__indicf5.wav",
    }

    for tag, (wav_bytes, sr) in results.items():
        dst = out_dir / filenames[tag]
        with open(dst, "wb") as f:
            f.write(wav_bytes)
        print(f"Successfully saved {dst} ({len(wav_bytes)} bytes, sr={sr})")

if __name__ == "__main__":
    main()
