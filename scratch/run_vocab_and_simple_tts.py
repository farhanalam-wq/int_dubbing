import os
import sys
import io
import json
import wave
import struct
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import modal

app = modal.App("dhvaani-vocab-and-simple-tts")

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
    timeout=300,
    secrets=[modal.Secret.from_name("hf-secret")],
)
def inspect_vocab_and_synthesize_simple(
    prompt_wav_bytes: bytes,
    prompt_text: str,
    simple_bn_text: str = "আমি ভালো আছি",
):
    import os
    import time
    import tempfile
    import torch
    import torchaudio
    import soundfile as sf
    from transformers import AutoModel

    # Monkey patch torchaudio load/save
    def _safe_torchaudio_load(filepath, *args, **kwargs):
        data, sr = sf.read(filepath)
        tensor = torch.from_numpy(data).float()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 2:
            if tensor.shape[1] > 1:
                tensor = tensor.mean(dim=-1, keepdim=True).t()
            else:
                tensor = tensor.t()
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

    model = AutoModel.from_pretrained(
        "ARTPARK-IISc/DhVaani-0.5",
        trust_remote_code=True,
        token=hf_token,
        low_cpu_mem_usage=False,
    ).to(device).eval()

    # Sanitize meta buffers
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

    rt = model._runtime()
    tokenizer = rt["tokenizer"]

    # 1. Inspect Tokenizer Vocab
    token2id = getattr(tokenizer, "token2id", {})
    id2token = getattr(tokenizer, "id2token", {})
    unk_id = getattr(tokenizer, "unk_id", None)
    pad_id = getattr(tokenizer, "pad_id", None)

    # Scan Bengali range U+0980 to U+09FF
    bengali_in_vocab = {}
    bengali_missing = {}
    for cp in range(0x0980, 0x0A00):
        ch = chr(cp)
        hex_cp = f"U+{cp:04X}"
        name = ""
        try:
            import unicodedata
            name = unicodedata.name(ch)
        except Exception:
            pass
        if ch in token2id:
            bengali_in_vocab[hex_cp] = {
                "char": ch,
                "token_id": token2id[ch],
                "name": name,
            }
        else:
            bengali_missing[hex_cp] = {
                "char": ch,
                "name": name,
            }

    # 2. Check test strings
    test_strings = {
        "seg001": "ভবিষ্যতের অপর নাম হলো",
        "seg002": "সংঘর্ষ! হৃদয়ে আজ ইচ্ছা জাগে এবং তা যদি পূর্ণ না হতে পারে,",
        "simple_phrase": simple_bn_text,
    }

    test_results = {}
    for key, text in test_strings.items():
        tokens_str = tokenizer.texts_to_tokens([text])[0]
        token_ids = tokenizer.tokens_to_token_ids([tokens_str])[0]
        char_breakdown = []
        missing_chars = []
        for ch in text:
            cp = ord(ch)
            hex_cp = f"U+{cp:04X}"
            in_vocab = ch in token2id
            tid = token2id.get(ch, unk_id)
            char_breakdown.append({
                "char": ch,
                "codepoint": hex_cp,
                "in_vocab": in_vocab,
                "token_id": tid,
            })
            if not in_vocab:
                missing_chars.append({"char": ch, "codepoint": hex_cp})
        test_results[key] = {
            "raw_text": text,
            "tokens_str": tokens_str,
            "token_ids": token_ids,
            "missing_chars": missing_chars,
            "char_breakdown": char_breakdown,
        }

    # 3. Synthesize simple phrase with fixed pipeline
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
        tmp_in.write(prompt_wav_bytes)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path.replace(".wav", "_simple_out.wav")

    try:
        t0 = time.time()
        audio = model.synthesize(
            text=simple_bn_text,
            prompt_wav=tmp_in_path,
            prompt_text=prompt_text,
            guidance_scale=2.5,
            num_step=32,
            speed=1.0,
            seed=666,
        )
        gen_time = time.time() - t0
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()

        sampling_rate = getattr(model, "sampling_rate", 24000)
        sf.write(tmp_out_path, audio, sampling_rate)
        with open(tmp_out_path, "rb") as f:
            simple_wav_bytes = f.read()

    finally:
        for p in [tmp_in_path, tmp_out_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    return {
        "tokenizer_class": str(type(tokenizer)),
        "vocab_size": len(token2id),
        "unk_id": unk_id,
        "pad_id": pad_id,
        "bengali_codepoints_present_count": len(bengali_in_vocab),
        "bengali_codepoints_missing_count": len(bengali_missing),
        "bengali_in_vocab": bengali_in_vocab,
        "bengali_missing": bengali_missing,
        "test_results": test_results,
        "simple_synth_time": gen_time,
        "simple_wav_bytes": simple_wav_bytes,
    }


def main():
    import importlib
    mod = importlib.import_module("scripts.09_tts")
    slice_wav_bytes = mod.slice_wav_bytes

    vocal_path = Path("data/stage_02_separated/clip001__separated__vocal.wav")
    # Clean mono 24kHz slice of seg001 (10300 - 12500ms, 2.2s)
    prompt_bytes = slice_wav_bytes(vocal_path, 10300, 12500, target_rate=24000)
    prompt_text = "भविष्य का दूसरा नाम है"

    print("Running inspect_vocab_and_synthesize_simple on Modal...")
    with modal.enable_output():
        with app.run():
            res = inspect_vocab_and_synthesize_simple.remote(
                prompt_wav_bytes=prompt_bytes,
                prompt_text=prompt_text,
                simple_bn_text="আমি ভালো আছি",
            )

    # Save audio output
    out_audio_path = Path("data/stage_07_tts/test_simple_bengali__tts__dhvaani.wav")
    out_audio_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_audio_path, "wb") as f:
        f.write(res["simple_wav_bytes"])
    print(f"Simple Bengali audio saved to: {out_audio_path}")

    # Remove audio bytes from json dump
    report = dict(res)
    del report["simple_wav_bytes"]

    out_json_path = Path("data/stage_07_tts/dhvaani_vocab_diagnostic.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Diagnostic report saved to: {out_json_path}")


if __name__ == "__main__":
    main()
