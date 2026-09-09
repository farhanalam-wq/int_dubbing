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

app = modal.App("run-svara-bakeoff")

bakeoff_image = (
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
        "safetensors",
        "loguru",
        "f5-tts",
        "snac",
        "vocos",
        "encodec",
    )
)

@app.function(
    gpu="L4",
    image=bakeoff_image,
    timeout=600,
    secrets=[modal.Secret.from_name("hf-secret")],
)
def synthesize_svara(
    ref_wav_bytes_seg001: bytes,
    ref_text_seg001: str,
    ref_wav_bytes_seg002: bytes,
    ref_text_seg002: str,
):
    import io
    import torch
    import soundfile as sf
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from snac import SNAC

    token = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Svara running on device: {device}")

    # 1. Tokenizer & Constants
    model_id = "kenpath/svara-tts-voiceclone-beta"
    print(f"Loading tokenizer {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

    BOS_TOKEN = 128000
    END_OF_TEXT = 128001
    END_OF_TURN = 128009
    AUDIO_TOKEN = 156939
    START_OF_SPEECH = 128257
    END_OF_SPEECH = 128258
    START_OF_HUMAN = 128259
    END_OF_HUMAN = 128260
    START_OF_AI = 128261
    END_OF_AI = 128262

    AUDIO_TOKENS_START = 128256 + 10  # 128266
    AUDIO_VOCAB_SIZE = 4096
    AUDIO_TOKEN_OFFSETS = [AUDIO_TOKENS_START + (i * AUDIO_VOCAB_SIZE) for i in range(7)]

    # 2. SNAC Codec
    print("Loading SNAC 24kHz codec...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)

    def encode_ref_audio(wav_bytes: bytes) -> list:
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=-1)
        audio_tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.inference_mode():
            codes = snac_model.encode(audio_tensor)
        
        all_codes = []
        num_coarse = codes[0].shape[1]
        for i in range(num_coarse):
            c0 = codes[0][0][i].item()
            c1 = codes[1][0][2 * i].item()
            c2 = codes[2][0][4 * i].item()
            c3 = codes[2][0][4 * i + 1].item()
            c4 = codes[1][0][2 * i + 1].item()
            c5 = codes[2][0][4 * i + 2].item()
            c6 = codes[2][0][4 * i + 3].item()
            
            all_codes.append(c0 + AUDIO_TOKEN_OFFSETS[0])
            all_codes.append(c1 + AUDIO_TOKEN_OFFSETS[1])
            all_codes.append(c2 + AUDIO_TOKEN_OFFSETS[2])
            all_codes.append(c3 + AUDIO_TOKEN_OFFSETS[3])
            all_codes.append(c4 + AUDIO_TOKEN_OFFSETS[4])
            all_codes.append(c5 + AUDIO_TOKEN_OFFSETS[5])
            all_codes.append(c6 + AUDIO_TOKEN_OFFSETS[6])
        return all_codes

    def build_prompt_ids(target_text: str, ref_audio_tokens: list, ref_transcript: str) -> torch.Tensor:
        blocks = [torch.tensor([[BOS_TOKEN]], dtype=torch.int64, device=device)]
        
        # Turn 1: Reference human transcript
        if ref_transcript:
            tr_ids = tokenizer(ref_transcript, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            turn_tr = torch.cat([
                torch.tensor([[START_OF_HUMAN, AUDIO_TOKEN]], device=device),
                tr_ids,
                torch.tensor([[END_OF_HUMAN, END_OF_TURN]], device=device)
            ], dim=1)
            blocks.append(turn_tr)
        
        # Turn 2: Reference AI audio
        aud_t = torch.tensor([ref_audio_tokens], dtype=torch.int64, device=device)
        turn_aud = torch.cat([
            torch.tensor([[START_OF_AI, START_OF_SPEECH]], device=device),
            aud_t,
            torch.tensor([[END_OF_SPEECH, END_OF_AI, END_OF_TURN]], device=device)
        ], dim=1)
        blocks.append(turn_aud)

        # Turn 3: Target human text
        tgt_ids = tokenizer(target_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        turn_tgt = torch.cat([
            torch.tensor([[START_OF_HUMAN, AUDIO_TOKEN]], device=device),
            tgt_ids,
            torch.tensor([[END_OF_HUMAN, END_OF_TURN]], device=device)
        ], dim=1)
        blocks.append(turn_tgt)

        # Start of AI speech generation
        blocks.append(torch.tensor([[START_OF_AI, START_OF_SPEECH]], device=device))
        
        return torch.cat(blocks, dim=1)

    # 3. Load Llama Model
    print(f"Loading {model_id} model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        token=token,
    ).eval()
    model.config.use_cache = True

    print("Encoding reference audio clips...")
    ref_tokens_1 = encode_ref_audio(ref_wav_bytes_seg001)
    ref_tokens_2 = encode_ref_audio(ref_wav_bytes_seg002)
    print(f"Encoded ref tokens: seg001={len(ref_tokens_1)}, seg002={len(ref_tokens_2)}")

    tiers = [
        ("simple", "আমি ভালো আছি", ref_tokens_1, ref_text_seg001),
        ("seg001", "ভবিষ্যতের অপর নাম হলো", ref_tokens_1, ref_text_seg001),
        ("seg002", "সংঘর্ষ! হৃদয়ে আজ ইচ্ছা জাগে এবং তা যদি পূর্ণ না হতে পারে,", ref_tokens_2, ref_text_seg002),
    ]

    results = {}
    for tag, target_text, ref_tokens, ref_text in tiers:
        print(f"Generating Tier [{tag}]: '{target_text}'...")
        prompt_ids = build_prompt_ids(target_text, ref_tokens, ref_text)
        prompt_len = prompt_ids.shape[1]

        with torch.inference_mode():
            outputs = model.generate(
                prompt_ids,
                max_new_tokens=600,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                use_cache=True,
                eos_token_id=[END_OF_SPEECH, END_OF_TEXT, END_OF_TURN],
                pad_token_id=tokenizer.pad_token_id or 128263,
            )

        gen_tokens = outputs[0][prompt_len:].tolist()
        print(f"Tier [{tag}] generated {len(gen_tokens)} tokens")

        # Trim at END_OF_SPEECH or special token
        audio_codes = []
        for t in gen_tokens:
            if t in [END_OF_SPEECH, END_OF_TEXT, END_OF_TURN, START_OF_AI, END_OF_AI]:
                break
            audio_codes.append(t)

        # Convert back to raw SNAC codes
        F = len(audio_codes) // 7
        if F == 0:
            print(f"Warning: No full audio frames for {tag}")
            continue

        valid_codes = audio_codes[: F * 7]
        raw_snac = []
        for idx, val in enumerate(valid_codes):
            offset = AUDIO_TOKEN_OFFSETS[idx % 7]
            code = val - offset
            raw_snac.append(max(0, min(4095, code)))

        t = torch.tensor(raw_snac, dtype=torch.int32, device=device).view(F, 7)
        codes_0 = t[:, 0].reshape(1, -1)
        codes_1 = t[:, [1, 4]].reshape(1, -1)
        codes_2 = t[:, [2, 3, 5, 6]].reshape(1, -1)

        with torch.inference_mode():
            audio = snac_model.decode([codes_0, codes_1, codes_2])
            audio = audio.detach().float().cpu().numpy().reshape(-1)

        # Convert to 16-bit WAV bytes
        buf = io.BytesIO()
        sf.write(buf, audio, 24000, format="WAV", subtype="PCM_16")
        results[tag] = buf.getvalue()
        print(f"Tier [{tag}] successfully synthesized: {len(results[tag])} bytes")

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

    print("Launching Svara-TTS synthesis on Modal...")
    with modal.enable_output():
        with app.run():
            results = synthesize_svara.remote(b1, t1, b2, t2)

    out_dir = Path("data/stage_07_tts")
    out_dir.mkdir(parents=True, exist_ok=True)

    filenames = {
        "simple": "clip001_simple__tts__svara.wav",
        "seg001": "clip001_seg001__tts__svara.wav",
        "seg002": "clip001_seg002__tts__svara.wav",
    }

    for tag, wav_bytes in results.items():
        dst = out_dir / filenames[tag]
        with open(dst, "wb") as f:
            f.write(wav_bytes)
        print(f"Saved: {dst} ({len(wav_bytes)} bytes)")

if __name__ == "__main__":
    main()
