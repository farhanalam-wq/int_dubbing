import os
import sys

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import modal

app = modal.App("check-indicf5-keys")

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
        "f5-tts",
        "vocos",
    )
)

@app.function(
    gpu="L4",
    image=bakeoff_image,
    timeout=300,
    secrets=[modal.Secret.from_name("hf-secret")],
)
def check_keys():
    import os
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from f5_tts.model import CFM, DiT
    from f5_tts.model.utils import get_tokenizer

    token = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab_path = hf_hub_download("ai4bharat/IndicF5", filename="checkpoints/vocab.txt", token=token)
    safetensors_path = hf_hub_download("ai4bharat/IndicF5", filename="model.safetensors", token=token)
    sd = load_file(safetensors_path, device=device)

    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
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

    model_keys = list(model.state_dict().keys())
    sd_keys = list(sd.keys())

    return {
        "model_keys_sample": model_keys[:5],
        "sd_keys_sample": sd_keys[:5],
        "total_model_keys": len(model_keys),
        "total_sd_keys": len(sd_keys),
    }

def main():
    with app.run():
        res = check_keys.remote()
    import json
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
